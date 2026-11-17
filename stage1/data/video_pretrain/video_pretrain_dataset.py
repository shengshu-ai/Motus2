import fnmatch
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import duckdb
from megfile.s3_path import S3Path

from data.utils.smart_io import (
    is_remote,
    smart_exists,
    smart_isdir,
    smart_join,
    smart_listdir_dirs,
    smart_load_pt,
)

logger = logging.getLogger(__name__)

# Dedicated logger for dataset issues (missing latents, lang, cache, etc.) — wire to a file in train_video_pretrain.
DATASET_WARN_LOGGER_NAME = "motus2.video_pretrain.warnings"
warn_logger = logging.getLogger(DATASET_WARN_LOGGER_NAME)


def setup_dataset_warning_logging(log_path: Optional[str]) -> None:
    """Write VideoPretrainDataset warnings to a separate file (aligned with wan_new log split)."""
    if not log_path:
        return
    log_path = os.path.expanduser(log_path)
    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    warn_lg = logging.getLogger(DATASET_WARN_LOGGER_NAME)
    warn_lg.setLevel(logging.WARNING)
    for h in warn_lg.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "_motus_vp_warn", False):
            return
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh._motus_vp_warn = True
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    warn_lg.addHandler(fh)
    warn_lg.propagate = False


DEFAULT_UMT5_CANDIDATE_FILES = (
    "environment.pt",
    "environment_LRbehavior.pt",
    "environment_behavior.pt",
    "robot_LRbehavior.pt",
    "robot_behavior.pt",
    "vidu_json_LRbehavior.pt",
    "vidu_json_behavior.pt",
)

# These datasets only provide usable WAN latents under 0phase_5hz.
PHASE0_ONLY_DATASETS = {
    "EgoDex",
    "Egocentric-10K",
}


def _open_duckdb(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def load_duckdb_table_and_run_sql(sql: str, duckdb_path: str) -> List[dict]:
    con = _open_duckdb(duckdb_path)
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]
    finally:
        con.close()


class VideoPretrainDataset(torch.utils.data.Dataset):
    """Video-only ARIO-format dataset for WAN video pretraining.

    Episode discovery (pick one): DuckDB file or ``dataset_dir`` tree scan.

    Expected layout under each episode prefix:
        video_features/wan2.2/<phase_dir>/<latent_file_name>
        text_features/UMT5/*.pt
    """

    def __init__(
        self,
        dataset_dir=None,
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        max_latent_window: int = 10,
        cache_dir: Optional[str] = None,
        dataset_exclude: Optional[List[str]] = None,
        dataset_include: Optional[List[str]] = None,
        lang_uncond_prob: float = 0.0,
        empty_lang_embed_path: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        latent_root: str = "video_features/wan2.2",
        latent_file_name: str = "video_3v_VAE.pt",
        latent_file_name_by_dataset: Optional[Dict[str, str]] = None,
        lang_root: Optional[str] = "text_features/UMT5",
        lang_subpath: str = "text_features/UMT5/environment_robot_behavior.pt",
        latent_phases: Optional[List[int]] = None,
        max_episodes: Optional[int] = None,
        val: bool = False,
        duckdb_path: Optional[str] = None,
        duckdb_table: str = "ario_format_data",
        use_index_cache: bool = True,
        **kwargs,
    ):
        self.use_index_cache = bool(use_index_cache)
        self.duckdb_path = duckdb_path
        self.duckdb_table = duckdb_table
        self.dataset_dir = str(dataset_dir).rstrip("/") if dataset_dir else None
        self.task_mode = task_mode
        self.task_name = task_name
        self.max_latent_window = max_latent_window
        self.cache_dir = cache_dir
        self.dataset_exclude = list(dataset_exclude or [])
        self.dataset_include = list(dataset_include or []) if dataset_include else None
        self.lang_uncond_prob = 0.0 if val else float(lang_uncond_prob or 0.0)
        self._empty_lang_tensor: Optional[torch.Tensor] = None
        if self.lang_uncond_prob > 0:
            if not empty_lang_embed_path:
                raise ValueError(
                    f"lang_uncond_prob={self.lang_uncond_prob} needs dataset.empty_lang_embed_path "
                    "(create it with scripts/encode_caption.py --caption '')"
                )
            else:
                ep = str(Path(empty_lang_embed_path).expanduser())
                self._empty_lang_tensor = self._load_empty_lang_embedding(ep, cache_dir)
                logger.info(
                    "Language CFG: lang_uncond_prob=%s, empty embed %s shape=%s",
                    self.lang_uncond_prob,
                    ep,
                    tuple(self._empty_lang_tensor.shape),
                )
        self.index_cache_path = self._resolve_index_cache_path(index_cache_path)
        self.latent_root = latent_root.strip("/")
        self.latent_file_name = latent_file_name
        self.latent_file_name_by_dataset = {
            str(k): str(v) for k, v in (latent_file_name_by_dataset or {}).items()
        }
        self.lang_root = lang_root.strip("/") if lang_root else None
        self.lang_subpath = lang_subpath.strip("/") if lang_subpath else None
        self.latent_phases_set = set(latent_phases) if latent_phases else None
        self.max_episodes = max_episodes
        self.val = val

        self.episode_files: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.task_weights: Dict[str, float] = {}
        self.total_episodes = 0
        self._latent_candidates_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._language_candidates_cache: Dict[str, List[str]] = {}
        self._missing_latent_roots = set()
        self._missing_language_roots = set()

        if not self.duckdb_path and not self.dataset_dir:
            raise ValueError(
                "VideoPretrainDataset: provide duckdb_path "
                "and/or dataset_dir (S3/ARIO tree scan). At least one is required."
            )

        self._load_episodes_distributed()

        if self.duckdb_path:
            src = f"duckdb:{self.duckdb_path}"
        else:
            src = str(self.dataset_dir)
        logger.info(
            f"VideoPretrainDataset: source={src}, episodes={self.total_episodes}"
        )

    def __len__(self):
        return self.total_episodes * 10

    @staticmethod
    def _name_matches(name: str, pattern: str) -> bool:
        return fnmatch.fnmatchcase(name, pattern.replace("%", "*"))

    def _dataset_allowed(self, dataset_name: str) -> bool:
        # include 非空时：只训练匹配 include 的数据集，不再应用 dataset_exclude。
        if self.dataset_include:
            return any(self._name_matches(dataset_name, pat) for pat in self.dataset_include)
        if any(self._name_matches(dataset_name, pat) for pat in self.dataset_exclude):
            return False
        return True

    def _task_allowed(self, task_name: str) -> bool:
        if self.task_mode != "single" or not self.task_name:
            return True
        return self._name_matches(task_name, self.task_name)

    def _latent_file_name_for_dataset(self, dataset_name: Optional[str]) -> str:
        if not dataset_name or not self.latent_file_name_by_dataset:
            return self.latent_file_name
        for pattern, file_name in self.latent_file_name_by_dataset.items():
            if self._name_matches(str(dataset_name), pattern):
                return file_name
        return self.latent_file_name

    def _episode_where_sql(self) -> str:
        """WHERE clause for DuckDB (SQL LIKE with * → %)."""
        conditions: List[str] = []
        if self.task_mode == "single" and self.task_name:
            task_pat = self.task_name.replace("*", "%")
            task_sql = task_pat.replace("'", "''")
            if "%" in task_pat:
                conditions.append(f"task LIKE '{task_sql}'")
            else:
                conditions.append(f"task = '{task_sql}'")

        if self.dataset_include:
            include_terms = []
            for pat in self.dataset_include:
                pat_sql = pat.replace("*", "%").replace("'", "''")
                include_terms.append(f"dataset LIKE '{pat_sql}'")
            conditions.append("(" + " OR ".join(include_terms) + ")")
        elif self.dataset_exclude:
            exclude_terms = []
            for pat in self.dataset_exclude:
                pat_sql = pat.replace("*", "%").replace("'", "''")
                exclude_terms.append(f"dataset LIKE '{pat_sql}'")
            conditions.append("NOT (" + " OR ".join(exclude_terms) + ")")

        return (" WHERE " + " AND ".join(conditions)) if conditions else ""

    @staticmethod
    def _normalize_bos(path: str) -> str:
        if path.startswith("bos://"):
            return "s3://" + path[len("bos://") :]
        if path.startswith("bos:/"):
            return "s3://" + path[len("bos:/") :]
        return path

    def _rows_to_episode_dicts(self, rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        for row in rows:
            episode_bos_path = row.get("episode_bos_path")
            task = row.get("task")
            if episode_bos_path is None or task is None:
                continue
            if isinstance(episode_bos_path, float) or isinstance(task, float):
                # pandas NaN
                continue
            if not str(episode_bos_path).strip() or not str(task).strip():
                continue

            dataset_name = row.get("dataset") or ""
            if dataset_name and not self._dataset_allowed(str(dataset_name)):
                continue

            base = self._normalize_bos(str(episode_bos_path).rstrip("/"))
            episode_name = row.get("episode_id") or base.rsplit("/", 1)[-1]
            task_key = f"{dataset_name}/{task}" if dataset_name else str(task)

            episodes.append(
                {
                    "dataset_name": dataset_name,
                    "task_name": task,
                    "task_key": task_key,
                    "episode_name": str(episode_name),
                    "episode_path": base,
                    "latent_root": smart_join(base, self.latent_root),
                    "lang_root": smart_join(base, self.lang_root) if self.lang_root else None,
                    "lang_path": smart_join(base, self.lang_subpath) if self.lang_subpath else None,
                }
            )

        logger.info(f"{source} video-pretrain index: {len(episodes)} episodes")
        return episodes

    def _load_episodes_from_duckdb(self) -> List[Dict[str, Any]]:
        """Episode discovery from DuckDB (same catalog as wan_new pretrain; no qpos/endpose)."""
        where = self._episode_where_sql()
        sql = f"""
            SELECT
                episode_storage_key,
                episode_bos_path,
                dataset,
                task,
                episode_id
            FROM {self.duckdb_table}
            {where}
        """
        rows = load_duckdb_table_and_run_sql(sql, self.duckdb_path)
        return self._rows_to_episode_dicts(rows, f"DuckDB:{self.duckdb_table}")

    @staticmethod
    def _phase_index_from_component(name: str) -> Optional[int]:
        for pattern in (r"(\d+)\s*phase", r"phase[_-]?(\d+)", r"_p(\d+)$"):
            m = re.search(pattern, name, flags=re.IGNORECASE)
            if m is not None:
                return int(m.group(1))
        return None

    def _iter_subdirs(self, path: str):
        if is_remote(path):
            with S3Path(path).scandir() as entries:
                for entry in entries:
                    if entry.is_dir():
                        yield str(entry.path).rstrip("/")
            return
        for subdir in smart_listdir_dirs(path):
            yield str(subdir).rstrip("/")

    def _load_episodes_distributed(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if self.use_index_cache:
            if not is_dist:
                flat_episodes = self._load_or_build_index()
            elif rank == 0:
                flat_episodes = self._load_or_build_index()
            else:
                flat_episodes = self._wait_for_index_cache()
        else:
            # 全进程内存构建；分布式下各 rank 各自查库（避免写共享 JSON 缓存）。
            flat_episodes = self._build_episode_index()
            if is_dist:
                dist.barrier()

        self._organize_episodes(flat_episodes or [])

    def _resolve_index_cache_path(self, index_cache_path: Optional[str]) -> str:
        if not self.use_index_cache:
            return ""

        if index_cache_path:
            return str(Path(index_cache_path).expanduser())

        cache_key = {
            "dataset_dir": self.dataset_dir,
            "duckdb_path": self.duckdb_path,
            "duckdb_table": self.duckdb_table,
            "dataset_exclude": sorted(self.dataset_exclude),
            "dataset_include": sorted(self.dataset_include) if self.dataset_include else None,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "latent_root": self.latent_root,
            "latent_file_name": self.latent_file_name,
            "lang_root": self.lang_root,
            "lang_subpath": self.lang_subpath,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set else None,
            "max_episodes": self.max_episodes,
        }
        digest = hashlib.md5(
            json.dumps(cache_key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return str(Path.home() / ".cache" / "motus2" / f"ario_video_pretrain_index_{digest}.json")

    def _index_cache_meta(self) -> Dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "duckdb_path": self.duckdb_path,
            "duckdb_table": self.duckdb_table,
            "dataset_exclude": sorted(self.dataset_exclude),
            "dataset_include": sorted(self.dataset_include) if self.dataset_include else None,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "latent_root": self.latent_root,
            "latent_file_name": self.latent_file_name,
            "lang_root": self.lang_root,
            "lang_subpath": self.lang_subpath,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set else None,
            "max_episodes": self.max_episodes,
        }

    def _index_failure_path(self) -> Path:
        return Path(f"{self.index_cache_path}.failed")

    def _load_cached_index(self, silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        if not self.use_index_cache or not self.index_cache_path:
            return None
        cache_path = Path(self.index_cache_path)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            if not silent:
                warn_logger.warning(f"Failed to read cached video-pretrain index {cache_path}: {e}")
            return None

        if isinstance(payload, dict):
            cache_meta = payload.get("meta")
            if cache_meta is not None and cache_meta != self._index_cache_meta():
                if not silent:
                    warn_logger.warning(f"Ignore stale video-pretrain index cache: {cache_path}")
                return None
            episodes = payload.get("episodes", [])
        else:
            episodes = payload

        if not silent:
            logger.info(f"Loaded cached video-pretrain index: {cache_path} ({len(episodes)} episodes)")
        return episodes

    def _save_index_cache(self, episodes: List[Dict[str, Any]]):
        if not self.use_index_cache or not self.index_cache_path:
            return
        cache_path = Path(self.index_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        payload = {
            "meta": self._index_cache_meta(),
            "episodes": episodes,
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, cache_path)
        logger.info(f"Saved video-pretrain index: {cache_path}")

    def _write_index_failure(self, error: Exception):
        failure_path = self._index_failure_path()
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_path, "w") as f:
            json.dump({"error": repr(error), "time": time.time()}, f)

    def _clear_index_failure(self):
        failure_path = self._index_failure_path()
        if failure_path.exists():
            failure_path.unlink()

    def _read_index_failure(self) -> Optional[str]:
        failure_path = self._index_failure_path()
        if not failure_path.exists():
            return None
        try:
            with open(failure_path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return f"rank0 failed while building {self.index_cache_path}"
        return payload.get("error") or f"rank0 failed while building {self.index_cache_path}"

    def _wait_for_index_cache(
        self,
        timeout_s: float = 7200.0,
        poll_s: float = 5.0,
    ) -> List[Dict[str, Any]]:
        deadline = time.time() + timeout_s
        warned = False
        while time.time() < deadline:
            failure = self._read_index_failure()
            if failure:
                raise RuntimeError(
                    f"Video-pretrain index build failed on rank0 for {self.index_cache_path}: {failure}"
                )

            episodes = self._load_cached_index(silent=True)
            if episodes is not None:
                logger.info(
                    f"Loaded shared video-pretrain index: {self.index_cache_path} ({len(episodes)} episodes)"
                )
                return episodes

            if not warned:
                warn_logger.warning(
                    f"Waiting for rank0 to build video-pretrain index cache: {self.index_cache_path}"
                )
                warned = True
            time.sleep(poll_s)

        raise TimeoutError(
            f"Timed out after {timeout_s:.0f}s waiting for video-pretrain index cache: {self.index_cache_path}"
        )

    def _load_or_build_index(self) -> List[Dict[str, Any]]:
        episodes = self._load_cached_index()
        if episodes is not None:
            return episodes

        self._clear_index_failure()
        try:
            episodes = self._build_episode_index()
            self._save_index_cache(episodes)
            return episodes
        except Exception as e:
            self._write_index_failure(e)
            raise

    def _build_episode_index(self) -> List[Dict[str, Any]]:
        if self.duckdb_path is not None:
            episodes = self._load_episodes_from_duckdb()
            if self.max_episodes is not None:
                episodes = episodes[: self.max_episodes]
            logger.info(f"Built video-pretrain index with {len(episodes)} episodes (DuckDB)")
            return episodes

        assert self.dataset_dir is not None
        episodes: List[Dict[str, Any]] = []
        dataset_dirs = list(self._iter_subdirs(self.dataset_dir))
        logger.info(f"Scanning ARIO-format root {self.dataset_dir}: {len(dataset_dirs)} dataset dirs")

        for dataset_dir in dataset_dirs:
            dataset_dir = str(dataset_dir).rstrip("/")
            dataset_name = dataset_dir.rsplit("/", 1)[-1]
            if not self._dataset_allowed(dataset_name):
                continue

            task_dirs = list(self._iter_subdirs(dataset_dir))
            logger.info(f"  dataset={dataset_name}: {len(task_dirs)} task dirs")
            for task_dir in task_dirs:
                task_dir = str(task_dir).rstrip("/")
                task_name = task_dir.rsplit("/", 1)[-1]
                if not self._task_allowed(task_name):
                    continue

                task_key = f"{dataset_name}/{task_name}"
                for episode_dir in self._iter_subdirs(task_dir):
                    episode_dir = str(episode_dir).rstrip("/")
                    episode_name = episode_dir.rsplit("/", 1)[-1]
                    episodes.append(
                        {
                            "dataset_name": dataset_name,
                            "task_name": task_name,
                            "task_key": task_key,
                            "episode_name": episode_name,
                            "episode_path": episode_dir,
                            "latent_root": smart_join(episode_dir, self.latent_root),
                            "lang_root": smart_join(episode_dir, self.lang_root) if self.lang_root else None,
                            "lang_path": smart_join(episode_dir, self.lang_subpath) if self.lang_subpath else None,
                        }
                    )

        if self.max_episodes is not None:
            episodes = episodes[: self.max_episodes]
        logger.info(f"Built video-pretrain index with {len(episodes)} episodes")
        return episodes

    def _organize_episodes(self, episodes: List[Dict[str, Any]]):
        if self.task_mode == "single":
            self.episode_files = episodes
            self.total_episodes = len(self.episode_files)
            return

        self.task_to_episodes = {}
        for ep in episodes:
            self.task_to_episodes.setdefault(ep["task_key"], []).append(ep)
        num_tasks = len(self.task_to_episodes)
        self.task_weights = {t: 1.0 / num_tasks for t in self.task_to_episodes} if num_tasks else {}
        self.total_episodes = sum(len(eps) for eps in self.task_to_episodes.values())

    def _drop_episode(self, ep: Dict[str, Any]):
        if self.task_mode == "single":
            self.episode_files = [x for x in self.episode_files if x["episode_path"] != ep["episode_path"]]
            self.total_episodes = len(self.episode_files)
            return

        task_key = ep["task_key"]
        eps = [x for x in self.task_to_episodes.get(task_key, []) if x["episode_path"] != ep["episode_path"]]
        if eps:
            self.task_to_episodes[task_key] = eps
        else:
            self.task_to_episodes.pop(task_key, None)
        num_tasks = len(self.task_to_episodes)
        self.task_weights = {t: 1.0 / num_tasks for t in self.task_to_episodes} if num_tasks else {}
        self.total_episodes = sum(len(v) for v in self.task_to_episodes.values())

    def _resolve_latent_candidates(self, ep: Dict[str, Any]) -> List[Dict[str, Any]]:
        episode_path = ep["episode_path"]
        cached = self._latent_candidates_cache.get(episode_path)
        if cached is not None:
            return cached

        latent_root = ep["latent_root"]
        dataset_name = ep.get("dataset_name")
        latent_file_name = self._latent_file_name_for_dataset(dataset_name)
        candidates: List[Dict[str, Any]] = []

        forced_phase_indices: Optional[List[int]] = None
        if dataset_name in PHASE0_ONLY_DATASETS:
            forced_phase_indices = [0]
            if self.latent_phases_set is not None:
                forced_phase_indices = [p for p in forced_phase_indices if p in self.latent_phases_set]
            if not forced_phase_indices:
                self._latent_candidates_cache[episode_path] = []
                return []

        if forced_phase_indices is not None:
            for phase_index in forced_phase_indices:
                phase_dir = smart_join(latent_root, f"{phase_index}phase_5hz")
                candidates.append(
                    {
                        "latent_path": smart_join(phase_dir, latent_file_name),
                        "phase_start_30": int(phase_index),
                    }
                )
        elif self.latent_phases_set is not None:
            for phase_index in random.sample(
                sorted(self.latent_phases_set),
                k=len(self.latent_phases_set),
            ):
                phase_dir = smart_join(latent_root, f"{phase_index}phase_5hz")
                candidates.append(
                    {
                        "latent_path": smart_join(phase_dir, latent_file_name),
                        "phase_start_30": int(phase_index),
                    }
                )
        elif smart_isdir(latent_root):
            for phase_dir in self._iter_subdirs(latent_root):
                phase_name = phase_dir.rsplit("/", 1)[-1]
                phase_index = self._phase_index_from_component(phase_name)
                candidates.append(
                    {
                        "latent_path": smart_join(phase_dir, latent_file_name),
                        "phase_start_30": phase_index or 0,
                    }
                )

            direct_latent = smart_join(latent_root, latent_file_name)
            if not candidates and smart_exists(direct_latent):
                candidates.append({"latent_path": direct_latent, "phase_start_30": 0})
        elif latent_root not in self._missing_latent_roots:
            logger.info(f"latent root missing, skip for now: {latent_root}")
            self._missing_latent_roots.add(latent_root)

        self._latent_candidates_cache[episode_path] = candidates
        return candidates

    def _resolve_language_candidates(self, ep: Dict[str, Any]) -> List[str]:
        episode_path = ep["episode_path"]
        cached = self._language_candidates_cache.get(episode_path)
        if cached is not None:
            return cached

        candidates: List[str] = []
        lang_root = ep.get("lang_root")
        lang_path = ep.get("lang_path")

        if lang_root:
            for file_name in DEFAULT_UMT5_CANDIDATE_FILES:
                candidates.append(smart_join(lang_root, file_name))

        # Backward-compatible fallback for older single-file layout.
        if lang_path and lang_path not in candidates:
            candidates.append(lang_path)

        self._language_candidates_cache[episode_path] = candidates
        return candidates

    @staticmethod
    def _load_language_embedding(path: str, cache_dir: Optional[str]):
        embedding_data = smart_load_pt(path, cache_dir, map_location="cpu")
        if isinstance(embedding_data, list):
            embeddings = embedding_data[random.randint(0, len(embedding_data) - 1)]
        elif isinstance(embedding_data, torch.Tensor):
            embeddings = embedding_data[random.randint(0, embedding_data.shape[0] - 1)] if embedding_data.ndim == 3 else embedding_data
        else:
            raise ValueError(f"Unexpected language embedding type: {type(embedding_data)}")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings

    @staticmethod
    def _load_empty_lang_embedding(path: str, cache_dir: Optional[str]) -> torch.Tensor:
        """Load fixed 'empty caption' UMT5 embedding (.pt) for classifier-free–style language dropout."""
        embedding_data = smart_load_pt(path, cache_dir, map_location="cpu")
        if isinstance(embedding_data, list):
            embeddings = embedding_data[0]
        elif isinstance(embedding_data, torch.Tensor):
            embeddings = (
                embedding_data[0]
                if embedding_data.ndim == 3
                else embedding_data
            )
        else:
            raise ValueError(f"Unexpected empty language embedding type: {type(embedding_data)}")
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(f"Empty lang embed must be tensor, got {type(embeddings)}")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings.detach().clone()

    def __getitem__(self, idx):
        W = self.max_latent_window

        for _ in range(32):
            if self.task_mode == "single":
                if not self.episode_files:
                    continue
                ep = random.choice(self.episode_files)
            else:
                if not self.task_weights:
                    return None
                task = random.choices(
                    list(self.task_weights.keys()),
                    weights=list(self.task_weights.values()),
                    k=1,
                )[0]
                eps = self.task_to_episodes.get(task, [])
                if not eps:
                    continue
                ep = random.choice(eps)

            try:
                candidates = self._resolve_latent_candidates(ep)
                if not candidates:
                    self._drop_episode(ep)
                    continue

                chosen = None
                latent_data = None
                last_error = None
                for candidate in random.sample(candidates, k=len(candidates)):
                    try:
                        latent_data = smart_load_pt(
                            candidate["latent_path"],
                            self.cache_dir,
                            map_location="cpu",
                        )
                        chosen = candidate
                        break
                    except Exception as e:
                        last_error = e
                if chosen is None or latent_data is None:
                    # All latent candidates failed — drop this episode so we
                    # don't keep retrying on every sample attempt.
                    warn_logger.warning(
                        "Dropping episode %s: all latent candidates failed: %s",
                        ep.get("episode_name"), last_error,
                    )
                    self._drop_episode(ep)
                    continue
                if isinstance(latent_data, dict) and "phases" in latent_data:
                    phases = latent_data["phases"]
                    if self.latent_phases_set is not None:
                        phases = [p for i, p in enumerate(phases) if i in self.latent_phases_set]
                    if not phases:
                        continue
                    phase_data = random.choice(phases)
                    latents = phase_data["latents"]
                elif isinstance(latent_data, torch.Tensor):
                    latents = latent_data
                else:
                    latents = latent_data["latents"]

                T_lat = latents.shape[1]

                if T_lat > W:
                    start = random.randint(0, T_lat - W)
                    latents_crop = latents[:, start:start + W]
                    valid_len = W
                elif T_lat < W:
                    latents_crop = F.pad(latents, (0, 0, 0, 0, 0, W - T_lat))
                    valid_len = T_lat
                else:
                    latents_crop = latents
                    valid_len = W

                if valid_len <= 1:
                    continue

                sample = {
                    "latents": latents_crop,
                    "valid_len": valid_len,
                }
                language_candidates = self._resolve_language_candidates(ep)
                if language_candidates:
                    lang_embedding = None
                    last_lang_error = None
                    for chosen_lang in random.sample(language_candidates, k=len(language_candidates)):
                        try:
                            lang_embedding = self._load_language_embedding(chosen_lang, self.cache_dir)
                            break
                        except Exception as e:
                            last_lang_error = e
                    if lang_embedding is None:
                        # All language candidates failed — cache empty list
                        # so we don't retry 7 BOS calls next time, and return
                        # the sample without language embedding.
                        self._language_candidates_cache[ep["episode_path"]] = []
                        warn_logger.warning(
                            "No loadable language embedding for %s, proceeding without: %s",
                            ep.get("episode_name"), last_lang_error,
                        )
                    else:
                        sample["language_embedding"] = lang_embedding
                        if (
                            self._empty_lang_tensor is not None
                            and random.random() < self.lang_uncond_prob
                        ):
                            sample["language_embedding"] = self._empty_lang_tensor.clone()
                return sample

            except Exception as e:
                warn_logger.warning(
                    "Sample error (%s | dataset=%s task=%s latent_root=%s): %s",
                    ep.get("episode_name", "?"),
                    ep.get("dataset_name"),
                    ep.get("task_name"),
                    ep.get("latent_root"),
                    e,
                )
                continue

        return None


def _stack_language_embeddings_padded(lang_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Stack [L_i, D] UMT5 embeddings into [B, L_max, D] (pad shorter sequences at the end).

    training_step later pads/truncates each row to WAN ``text_len`` (e.g. 512).
    """
    if not lang_tensors:
        raise ValueError("empty language_embedding list")
    if any(t.dim() != 2 for t in lang_tensors):
        raise ValueError(
            "video_pretrain_collate_fn: expected language_embedding [L, D], got "
            f"{[tuple(t.shape) for t in lang_tensors]}"
        )
    d = int(lang_tensors[0].shape[1])
    if any(int(t.shape[1]) != d for t in lang_tensors):
        raise ValueError(
            "video_pretrain_collate_fn: language dim D must match across batch, got "
            f"{[tuple(t.shape) for t in lang_tensors]}"
        )
    L_max = max(int(t.shape[0]) for t in lang_tensors)
    rows = []
    for t in lang_tensors:
        L = int(t.shape[0])
        if L == L_max:
            rows.append(t)
        else:
            rows.append(F.pad(t, (0, 0, 0, L_max - L)))
    return torch.stack(rows, dim=0)


def video_pretrain_collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    if len(batch) == 1:
        sample = batch[0]
        sample["valid_len"] = int(sample["valid_len"])
        return sample

    latents = [s["latents"] for s in batch]
    latent_shapes = {tuple(x.shape) for x in latents}
    if len(latent_shapes) != 1:
        raise ValueError(
            "video_pretrain_collate_fn: all samples in a batch must share the same latent shape "
            f"(fixed resolution / same T,H,W). Got distinct shapes: {latent_shapes}. "
            "Use training.batch_size=1 or gradient_accumulation_steps if shapes vary."
        )
    out: Dict[str, Any] = {
        "latents": torch.stack(latents, dim=0),
        "valid_len": torch.tensor([int(s["valid_len"]) for s in batch], dtype=torch.long),
    }
    langs = [s.get("language_embedding") for s in batch]
    if any(l is not None for l in langs):
        if not all(l is not None for l in langs):
            raise ValueError(
                "video_pretrain_collate_fn: language_embedding must be present for all samples "
                "or missing for all when using batch_size>1."
            )
        out["language_embedding"] = _stack_language_embeddings_padded(langs)
    return out
