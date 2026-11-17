# Stage2 pretrain dataset for latent-space AR training.
#
# This is intentionally separate from data/realworld so stage2-specific logic
# (value targets, task-irrelevant mixing, cached episode indexing) stays
# isolated.

import fnmatch
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
import duckdb
from megfile.s3_path import S3Path

from data.utils.rotation_utils import upsample_abs_eef_16d, upsample_eef
from data.utils.smart_io import (
    is_remote,
    smart_glob_pt,
    smart_isdir,
    smart_join,
    smart_listdir_dirs,
    smart_load_json,
    smart_load_pt,
    stem,
)
from utils.relative_eef import eef_quat_wxyz_to_xyzw, eef_quat_xyzw_to_wxyz

logger = logging.getLogger(__name__)
INDEX_CACHE_VERSION = 3
PRETRAIN_DIR = Path(__file__).resolve().parent
ENDPOSE_GRIPPER_STATS_PATH = PRETRAIN_DIR / "endpose_gripper_all.json"
QPOS_GRIPPER_STATS_PATH = PRETRAIN_DIR / "qpos_gripper_all.json"

DEFAULT_UMT5_CANDIDATE_FILES = (
    "environment.pt",
    "environment_LRbehavior.pt",
    "environment_behavior.pt",
    "robot_LRbehavior.pt",
    "robot_behavior.pt",
    "vidu_json_LRbehavior.pt",
    "vidu_json_behavior.pt",
)


def _open_db(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def load_duckdb_table_and_run_sql(sql: str, duckdb_path: str) -> List[dict]:
    con = _open_db(duckdb_path)
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]
    finally:
        con.close()


class PretrainDataset(data.Dataset):
    """Stage2 video+action pretrain dataset."""

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        latent_root: str = "video_features/wan2.2",
        latent_file_name: str = "video_3v_VAE.pt",
        latent_subpath: Optional[str] = None,
        lang_root: Optional[str] = "text_features/UMT5",
        lang_subpath: Optional[str] = "text_features/UMT5/environment_robot_behavior.pt",
        lang_dir: str = "umt5_wan",
        robot: Optional[str] = None,
        duckdb_path: Optional[str] = None,
        duckdb_table: str = "real_world_data",
        dataset_dir=None,
        episode_index_source: str = "auto",
        catalog_path: Optional[str] = None,
        latent_dir: str = "vae_latents",
        latent_phases: Optional[List[int]] = None,
        latent_max_attempts: Optional[int] = None,
        index_cache_path: Optional[str] = None,
        metadata_cache_path: Optional[str] = None,
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        max_episodes: Optional[int] = None,
        max_latent_window: int = 10,
        video_action_freq_ratio: int = 6,
        action_mode: str = "joint",
        use_state: bool = True,
        val: bool = False,
        cache_dir: Optional[str] = None,
        action_upsample_ratio: int = 1,
        value_target: Optional[str] = None,
        progress_c: float = 0.01,
        treat_as_irrelevant: bool = False,
        empty_lang_path: Optional[str] = None,
        eef_quat_order: str = "xyzw",
        dataset_include: Optional[List[str]] = None,
        dataset_exclude: Optional[List[str]] = None,
        dataset_allowlist: Optional[List[str]] = None,
        dataset_denylist: Optional[List[str]] = None,
        **kwargs,
    ):
        self.jsonl_path = jsonl_path
        self.latent_root = latent_root.strip("/") if latent_root else None
        self.latent_file_name = latent_file_name
        self.latent_subpath = latent_subpath
        self.lang_root = lang_root.strip("/") if lang_root else None
        self.lang_subpath = lang_subpath.strip("/") if lang_subpath else None
        self.lang_dir = lang_dir
        self.robot_filter = robot
        self.duckdb_path = duckdb_path
        self.duckdb_table = duckdb_table
        self.episode_index_source = str(episode_index_source or "auto").lower()

        if dataset_dir is not None:
            def _to_dir(d):
                s = str(d)
                return s if is_remote(s) else Path(s)
            if isinstance(dataset_dir, (str, Path)):
                self.dataset_dirs = [_to_dir(dataset_dir)]
            else:
                self.dataset_dirs = [_to_dir(d) for d in dataset_dir]
        else:
            self.dataset_dirs = None

        self.catalog_path = catalog_path
        self.latent_dir = latent_dir
        self.latent_phases_set = set(latent_phases) if latent_phases else None
        self.latent_max_attempts = int(latent_max_attempts) if latent_max_attempts else None
        self.task_mode = task_mode
        self.task_name = task_name
        self.max_episodes = max_episodes
        self.max_latent_window = max_latent_window
        self.video_action_freq_ratio = video_action_freq_ratio
        self.action_mode = action_mode
        self.use_state = use_state
        self.val = val
        self.cache_dir = cache_dir
        self.action_upsample_ratio = action_upsample_ratio
        self.value_target = value_target
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.empty_lang_path = empty_lang_path
        self.eef_quat_order = eef_quat_order
        self.dataset_include = list(dataset_include or dataset_allowlist or [])
        self.dataset_exclude = list(dataset_exclude or dataset_denylist or [])
        self.index_cache_path = self._resolve_index_cache_path(index_cache_path)
        self.metadata_cache_path = (
            str(Path(metadata_cache_path).expanduser()) if metadata_cache_path else None
        )

        if self.eef_quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported eef_quat_order: {self.eef_quat_order}")
        if self.episode_index_source not in {"auto", "duckdb", "dataset_dir", "jsonl"}:
            raise ValueError(f"Unsupported episode_index_source: {self.episode_index_source}")
        if not jsonl_path and not dataset_dir and not duckdb_path:
            raise ValueError("Must provide one of: jsonl_path, dataset_dir, or duckdb_path")
        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name")

        self.actions_per_latent = 4 * video_action_freq_ratio * action_upsample_ratio
        self.episode_files: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.task_weights: Dict[str, float] = {}
        self.total_episodes = 0
        self._language_candidates_cache: Dict[str, List[str]] = {}
        self._episode_metadata_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._missing_language_roots = set()
        self._metadata_db_conn: Optional[sqlite3.Connection] = None
        self._metadata_db_pid: Optional[int] = None
        self._gripper_stats_cache: Dict[str, Dict[str, Any]] = {}
        self._load_episodes_distributed()

        logger.info(
            "PretrainDataset: task_mode=%s, episodes=%d, actions_per_latent=%d, "
            "upsample=%d, action_mode=%s, irrelevant=%s, value_target=%s, "
            "latent_max_attempts=%s, index_cache=%s, metadata_cache=%s",
            task_mode,
            self.total_episodes,
            self.actions_per_latent,
            action_upsample_ratio,
            action_mode,
            treat_as_irrelevant,
            value_target,
            self.latent_max_attempts,
            self.index_cache_path,
            self.metadata_cache_path,
        )

    # ------------------------------------------------------------------
    # Episode loading
    # ------------------------------------------------------------------

    @staticmethod
    def _name_matches(name: str, pattern: str) -> bool:
        return fnmatch.fnmatchcase(name, pattern.replace("%", "*"))

    def _task_allowed(self, task_name: str) -> bool:
        if self.task_mode != "single" or not self.task_name:
            return True
        return self._name_matches(task_name, self.task_name)

    def _episode_metadata_key(self, ep: Dict[str, Any]) -> Optional[str]:
        return ep.get("episode_path")

    def _metadata_connection(self) -> Optional[sqlite3.Connection]:
        if not self.metadata_cache_path:
            return None

        pid = os.getpid()
        if self._metadata_db_conn is not None and self._metadata_db_pid == pid:
            return self._metadata_db_conn

        if self._metadata_db_conn is not None:
            try:
                self._metadata_db_conn.close()
            except Exception:
                pass

        try:
            conn = sqlite3.connect(
                f"file:{self.metadata_cache_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.warning(f"Failed to open pretrain metadata cache {self.metadata_cache_path}: {e}")
            self.metadata_cache_path = None
            self._metadata_db_conn = None
            self._metadata_db_pid = None
            return None

        self._metadata_db_conn = conn
        self._metadata_db_pid = pid
        return conn

    def _lookup_episode_metadata(self, ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._episode_metadata_key(ep)
        if key is None or not self.metadata_cache_path:
            return None
        if key in self._episode_metadata_cache:
            return self._episode_metadata_cache[key]

        conn = self._metadata_connection()
        if conn is None:
            return None

        try:
            row = conn.execute(
                """
                SELECT phase_dirs_json, lang_files_json, latent_scanned, lang_scanned
                FROM episode_metadata
                WHERE episode_path = ?
                """,
                (key,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.warning(f"Failed to query pretrain metadata cache {self.metadata_cache_path}: {e}")
            self._episode_metadata_cache[key] = None
            return None

        if row is None:
            self._episode_metadata_cache[key] = None
            return None

        metadata = {
            "phase_dirs": json.loads(row["phase_dirs_json"] or "[]"),
            "lang_files": json.loads(row["lang_files_json"] or "[]"),
            "latent_scanned": bool(row["latent_scanned"]),
            "lang_scanned": bool(row["lang_scanned"]),
        }
        self._episode_metadata_cache[key] = metadata
        return metadata

    def _dataset_name_allowed(self, dataset_name: Optional[str]) -> bool:
        if dataset_name is None:
            return not self.dataset_include
        if self.dataset_include and not any(
            self._name_matches(dataset_name, pat) for pat in self.dataset_include
        ):
            return False
        if any(self._name_matches(dataset_name, pat) for pat in self.dataset_exclude):
            return False
        return True

    def _load_episodes_distributed(self):
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if not is_dist:
            flat_episodes = self._load_or_build_index()
        elif rank == 0:
            flat_episodes = self._load_or_build_index()
        elif not self.index_cache_path:
            flat_episodes = []
        else:
            flat_episodes = self._wait_for_index_cache()

        if is_dist:
            obj = [flat_episodes]
            dist.broadcast_object_list(obj, src=0)
            if rank != 0:
                flat_episodes = obj[0]

        self._organize_episodes(flat_episodes or [])
        self._log_episode_summary()

    def _resolve_index_cache_path(self, index_cache_path: Optional[str]) -> Optional[str]:
        if not index_cache_path:
            return None
        return str(Path(index_cache_path).expanduser())

    def _index_cache_meta(self) -> Dict[str, Any]:
        return {
            "index_cache_version": INDEX_CACHE_VERSION,
            "episode_index_source": self.episode_index_source,
            "jsonl_path": self.jsonl_path,
            "dataset_dirs": [str(d) for d in self.dataset_dirs] if self.dataset_dirs is not None else None,
            "duckdb_path": self.duckdb_path,
            "duckdb_table": self.duckdb_table,
            "catalog_path": self.catalog_path,
            "latent_dir": self.latent_dir,
            "latent_root": self.latent_root,
            "latent_file_name": self.latent_file_name,
            "latent_subpath": self.latent_subpath,
            "lang_root": self.lang_root,
            "lang_subpath": self.lang_subpath,
            "lang_dir": self.lang_dir,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "max_episodes": self.max_episodes,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set else None,
            "dataset_include": sorted(self.dataset_include),
            "dataset_exclude": sorted(self.dataset_exclude),
            "robot_filter": self.robot_filter,
        }

    def _discovery_source(self) -> str:
        if self.episode_index_source == "auto":
            if self.duckdb_path is not None:
                return "duckdb"
            if self.dataset_dirs is not None:
                return "dataset_dir"
            return "jsonl"
        return self.episode_index_source

    def _index_failure_path(self) -> Optional[Path]:
        if not self.index_cache_path:
            return None
        return Path(f"{self.index_cache_path}.failed")

    def _is_remote_dir_scan_mode(self) -> bool:
        return (
            self._discovery_source() == "dataset_dir"
            and
            self.dataset_dirs is not None
            and not (self.catalog_path and self.catalog_path != "manifest")
            and self.catalog_path != "manifest"
            and any(is_remote(str(d)) for d in self.dataset_dirs)
        )

    def _load_index_payload(self, silent: bool = False) -> Optional[Dict[str, Any]]:
        if not self.index_cache_path:
            return None
        cache_path = Path(self.index_cache_path)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            if not silent:
                logger.warning(f"Failed to read cached pretrain index {cache_path}: {e}")
            return None

        if not isinstance(payload, dict):
            payload = {
                "meta": self._index_cache_meta(),
                "episodes": payload,
                "scanned_datasets": [],
                "scanned_tasks": [],
                "complete": True,
            }

        cache_meta = payload.get("meta")
        if cache_meta is not None and cache_meta != self._index_cache_meta():
            if not silent:
                logger.warning(f"Ignore stale pretrain index cache: {cache_path}")
            return None
        return payload

    def _load_cached_index(self, silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        if not self.index_cache_path:
            return None
        cache_path = Path(self.index_cache_path)
        payload = self._load_index_payload(silent=silent)
        if payload is None:
            return None
        if not payload.get("complete", True):
            if not silent:
                logger.info(
                    f"Found partial pretrain index cache: {self.index_cache_path} "
                    f"({len(payload.get('episodes', []))} episodes, "
                    f"{len(payload.get('scanned_datasets', []))} datasets done, "
                    f"{len(payload.get('scanned_tasks', []))} tasks done)"
                )
            return None
        episodes = payload.get("episodes", [])

        if not silent:
            logger.info(f"Loaded cached pretrain index: {cache_path} ({len(episodes)} episodes)")
        return episodes

    def _save_index_cache(
        self,
        episodes: List[Dict[str, Any]],
        scanned_datasets: Optional[List[str]] = None,
        scanned_tasks: Optional[List[str]] = None,
        complete: bool = True,
    ):
        if not self.index_cache_path:
            return
        cache_path = Path(self.index_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        payload = {
            "meta": self._index_cache_meta(),
            "episodes": episodes,
            "scanned_datasets": list(scanned_datasets or []),
            "scanned_tasks": list(scanned_tasks or []),
            "complete": complete,
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, cache_path)
        logger.info(
            f"Saved pretrain index: {cache_path} "
            f"(episodes={len(episodes)}, datasets={len(payload['scanned_datasets'])}, "
            f"tasks={len(payload['scanned_tasks'])}, complete={complete})"
        )

    def _write_index_failure(self, error: Exception):
        if not self.index_cache_path:
            return
        failure_path = self._index_failure_path()
        if failure_path is None:
            return
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_path, "w") as f:
            json.dump({"error": repr(error), "time": time.time()}, f)

    def _clear_index_failure(self):
        failure_path = self._index_failure_path()
        if failure_path is None:
            return
        if failure_path.exists():
            failure_path.unlink()

    def _read_index_failure(self) -> Optional[str]:
        failure_path = self._index_failure_path()
        if failure_path is None:
            return None
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
                    f"Pretrain index build failed on rank0 for {self.index_cache_path}: {failure}"
                )

            episodes = self._load_cached_index(silent=True)
            if episodes is not None:
                logger.info(
                    f"Loaded shared pretrain index: {self.index_cache_path} ({len(episodes)} episodes)"
                )
                return episodes

            if not warned:
                logger.warning(
                    f"Waiting for rank0 to build pretrain index cache: {self.index_cache_path}"
                )
                warned = True
            time.sleep(poll_s)

        raise TimeoutError(
            f"Timed out after {timeout_s:.0f}s waiting for pretrain index cache: {self.index_cache_path}"
        )

    def _load_or_build_index(self) -> List[Dict[str, Any]]:
        if not self.index_cache_path:
            return self._build_episode_index()
        episodes = self._load_cached_index()
        if episodes is not None:
            return episodes

        self._clear_index_failure()
        try:
            resume_payload = self._load_index_payload(silent=True)
            episodes = self._build_episode_index(resume_payload=resume_payload)
            self._save_index_cache(episodes, complete=True)
            return episodes
        except Exception as e:
            self._write_index_failure(e)
            raise

    def _build_episode_index(self, resume_payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        discovery_source = self._discovery_source()

        if discovery_source == "duckdb":
            episodes = self._load_episodes_from_duckdb()
        elif discovery_source == "dataset_dir":
            if self.dataset_dirs is None:
                raise ValueError("episode_index_source=dataset_dir requires dataset_dir")
            if self.catalog_path and self.catalog_path != "manifest":
                episodes = self._load_episodes_from_iceberg()
            elif self.catalog_path == "manifest":
                episodes = self._load_episodes_from_manifest()
            elif self._is_remote_dir_scan_mode():
                episodes = self._load_episodes_from_remote_dir_resume(resume_payload)
            else:
                episodes = self._load_episodes_from_dir()
        else:
            if not self.jsonl_path:
                raise ValueError("episode_index_source=jsonl requires jsonl_path")
            episodes = self._load_episode_list_jsonl()

        if self.max_episodes is not None:
            episodes = episodes[: self.max_episodes]
        logger.info(
            "Built pretrain index with %d episodes (source=%s)",
            len(episodes),
            discovery_source,
        )
        return episodes

    def _load_episodes_from_remote_dir_resume(
        self,
        resume_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = list(resume_payload.get("episodes", [])) if resume_payload else []
        scanned_datasets = set(resume_payload.get("scanned_datasets", [])) if resume_payload else set()
        scanned_tasks = set(resume_payload.get("scanned_tasks", [])) if resume_payload else set()

        all_dataset_subdirs: List[str] = []
        for dataset_dir in self.dataset_dirs:
            dataset_dir_str = str(dataset_dir).rstrip("/")
            if not is_remote(dataset_dir_str):
                continue
            for dataset_subdir in smart_listdir_dirs(dataset_dir_str):
                dataset_subdir = str(dataset_subdir).rstrip("/")
                dataset_name = dataset_subdir.rsplit("/", 1)[-1]
                if self._dataset_name_allowed(dataset_name):
                    all_dataset_subdirs.append(dataset_subdir)

        total_datasets = len(all_dataset_subdirs)
        remaining = [d for d in all_dataset_subdirs if d not in scanned_datasets]
        logger.info(
            f"Remote pretrain index scan: total_datasets={total_datasets}, "
            f"completed={len(scanned_datasets)}, remaining={len(remaining)}, "
            f"resume={'yes' if resume_payload and scanned_datasets else 'no'}"
        )

        log_every_tasks = 100
        for idx, dataset_subdir in enumerate(remaining, start=1):
            dataset_name = dataset_subdir.rsplit("/", 1)[-1]
            before = len(episodes)
            dataset_scanned_tasks = sum(1 for t in scanned_tasks if t.startswith(dataset_subdir + "/"))
            logger.info(
                f"[pretrain-index] starting dataset {len(scanned_datasets) + 1}/{total_datasets}: "
                f"{dataset_name} (resume_tasks={dataset_scanned_tasks})"
            )
            task_count = 0
            new_task_count = 0
            for task_dir in self._iter_remote_subdirs(dataset_subdir):
                task_count += 1
                if task_dir in scanned_tasks:
                    if task_count % log_every_tasks == 0:
                        logger.info(
                            f"[pretrain-index] {dataset_name}: "
                            f"tasks_seen={task_count}, tasks_done={new_task_count}, "
                            f"episodes_added={len(episodes) - before}, total={len(episodes)}"
                        )
                    continue

                episodes.extend(self._scan_ario_task_folder(task_dir, dataset_name))
                scanned_tasks.add(task_dir)
                new_task_count += 1
                if task_count % log_every_tasks == 0:
                    logger.info(
                        f"[pretrain-index] {dataset_name}: "
                        f"tasks_seen={task_count}, tasks_done={new_task_count}, "
                        f"episodes_added={len(episodes) - before}, total={len(episodes)}"
                    )
                if self.max_episodes is not None and len(episodes) >= self.max_episodes:
                    episodes = episodes[: self.max_episodes]
                    scanned_datasets.add(dataset_subdir)
                    self._save_index_cache(
                        episodes,
                        scanned_datasets=sorted(scanned_datasets),
                        scanned_tasks=sorted(scanned_tasks),
                        complete=True,
                    )
                    logger.info(
                        f"Reached max_episodes={self.max_episodes} while scanning {dataset_name}; "
                        f"stopping early"
                    )
                    return episodes
                self._save_index_cache(
                    episodes,
                    scanned_datasets=sorted(scanned_datasets),
                    scanned_tasks=sorted(scanned_tasks),
                    complete=False,
                )

            added = len(episodes) - before
            scanned_datasets.add(dataset_subdir)
            scanned_tasks = {t for t in scanned_tasks if not t.startswith(dataset_subdir + "/")}
            self._save_index_cache(
                episodes,
                scanned_datasets=sorted(scanned_datasets),
                scanned_tasks=sorted(scanned_tasks),
                complete=False,
            )
            logger.info(
                f"[pretrain-index] finished {dataset_name}: tasks_seen={task_count}, "
                f"tasks_done={new_task_count}, +{added} episodes, "
                f"total={len(episodes)}, remaining_datasets={total_datasets - len(scanned_datasets)}"
            )

        return episodes

    def _iter_remote_subdirs(self, path: str):
        with S3Path(path).scandir() as entries:
            for entry in entries:
                if entry.is_dir():
                    yield str(entry.path).rstrip("/")

    def _load_episode_list_jsonl(self) -> List[Dict[str, Any]]:
        with open(self.jsonl_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        episodes: List[Dict[str, Any]] = []
        for entry in entries:
            if self.robot_filter and entry.get("robot") != self.robot_filter:
                continue

            dataset_name = entry.get("dataset") or entry.get("robot")
            if not self._dataset_name_allowed(dataset_name):
                continue

            task_name = entry.get("task")
            if not task_name or not self._task_allowed(task_name):
                continue

            qpos_path = self._resolve_entry_action_path(entry)
            if qpos_path is None:
                continue

            base = self._normalize_bos(entry["episode_bos_path"].rstrip("/"))
            episode_name = base.rsplit("/", 1)[-1]
            episodes.append({
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task_name,
                "dataset_name": dataset_name,
                "episode_path": base,
                "latent_root": smart_join(base, self.latent_root) if self.latent_root else None,
                "latent_path": smart_join(base, self.latent_subpath) if self.latent_subpath else None,
                "qpos_path": self._normalize_bos(qpos_path),
                "lang_root": f"{base}/{self.lang_root}" if self.lang_root else None,
                "lang_path": f"{base}/{self.lang_subpath}" if self.lang_subpath else None,
                "phase_start_30": 0,
            })
        return episodes

    def _organize_episodes(self, episodes: List[Dict[str, Any]]):
        # Sampling is always episode-uniform after filtering; keep task buckets
        # only for summaries/debugging.
        self.episode_files = list(episodes)
        self._rebuild_task_index()

    def _rebuild_task_index(self) -> None:
        self.task_to_episodes = {}
        for ep in self.episode_files:
            self.task_to_episodes.setdefault(self._task_bucket_key(ep), []).append(ep)
        self._refresh_task_weights()

    def _refresh_task_weights(self) -> None:
        self.total_episodes = len(self.episode_files)
        if self.total_episodes:
            self.task_weights = {
                t: len(v) / self.total_episodes
                for t, v in self.task_to_episodes.items()
            }
        else:
            self.task_weights = {}

    def _episode_identity(self, ep: Dict[str, Any]) -> str:
        return (
            ep.get("episode_storage_key")
            or ep.get("episode_path")
            or ep.get("latent_path")
            or ep.get("episode_name")
            or "<unknown>"
        )

    def _drop_episode(self, ep: Dict[str, Any], reason: str) -> None:
        episode_id = self._episode_identity(ep)
        before = len(self.episode_files)
        self.episode_files = [
            cur for cur in self.episode_files if self._episode_identity(cur) != episode_id
        ]
        removed = len(self.episode_files) != before

        if not removed:
            return

        self._rebuild_task_index()
        logger.warning(
            "Dropping episode %s | dataset=%s task=%s storage_key=%s: %s",
            ep.get("episode_name", "?"),
            ep.get("dataset_name"),
            ep.get("task_name"),
            ep.get("episode_storage_key"),
            reason,
        )

    @staticmethod
    def _task_bucket_key(ep: Dict[str, Any]) -> str:
        dataset_name = ep.get("dataset_name")
        task_name = ep.get("task_name") or "<unknown-task>"
        if dataset_name:
            return f"{dataset_name}/{task_name}"
        return task_name

    def _load_episodes_from_duckdb(self) -> List[Dict[str, Any]]:
        conditions: List[str] = []
        if self.robot_filter:
            robot_sql = self.robot_filter.replace("'", "''")
            conditions.append(
                f"(robot = '{robot_sql}' OR dataset = '{robot_sql}')"
            )

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

        if self.dataset_exclude:
            exclude_terms = []
            for pat in self.dataset_exclude:
                pat_sql = pat.replace("*", "%").replace("'", "''")
                exclude_terms.append(f"dataset LIKE '{pat_sql}'")
            conditions.append("NOT (" + " OR ".join(exclude_terms) + ")")

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT
                episode_storage_key,
                episode_bos_path,
                dataset,
                task,
                episode_id,
                robot
            FROM {self.duckdb_table}
            {where}
        """
        rows = load_duckdb_table_and_run_sql(sql, self.duckdb_path)

        episodes: List[Dict[str, Any]] = []
        for row in rows:
            episode_bos_path = row.get("episode_bos_path")
            task = row.get("task")
            if not episode_bos_path or not task:
                continue

            dataset_name = row.get("dataset")
            base = self._normalize_bos(str(episode_bos_path).rstrip("/"))
            episode_name = row.get("episode_id") or base.rsplit("/", 1)[-1]
            episodes.append({
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task,
                "dataset_name": dataset_name,
                "robot_name": row.get("robot"),
                "episode_storage_key": row.get("episode_storage_key"),
                "episode_path": base,
                "latent_root": smart_join(base, self.latent_root) if self.latent_root else None,
                "latent_path": smart_join(base, self.latent_subpath) if self.latent_subpath else None,
                "qpos_path": smart_join(base, "qpos.pt"),
                "endpose_path": smart_join(base, "endpose.pt"),
                "lang_root": smart_join(base, self.lang_root) if self.lang_root else None,
                "lang_path": smart_join(base, self.lang_subpath) if self.lang_subpath else None,
                "phase_start_30": 0,
            })

        logger.info(
            f"DuckDB index query returned {len(episodes)} episodes from table {self.duckdb_table}"
        )
        return episodes

    def _scan_task_folder(self, task_path, dataset_name: Optional[str]) -> List[Dict[str, Any]]:
        task_path_str = str(task_path)
        latent_dir = smart_join(task_path_str, self.latent_dir)
        qpos_dir = smart_join(task_path_str, self._default_task_action_dir())
        lang_dir = smart_join(task_path_str, self.lang_dir)
        if not all(smart_isdir(d) for d in (latent_dir, qpos_dir, lang_dir)):
            return []

        task_name = Path(task_path_str).name
        if not self._task_allowed(task_name):
            return []

        episodes = []
        for latent_file in smart_glob_pt(latent_dir):
            name = stem(latent_file)
            m = re.match(r"^(.*)_p(\d+)$", name)
            if m is not None:
                base_name = m.group(1)
                phase_start_30 = int(m.group(2))
            else:
                base_name = name
                phase_start_30 = 0

            qpos_file = smart_join(qpos_dir, f"{base_name}.pt")
            lang_candidates = [smart_join(lang_dir, f"{name}.pt")]
            fallback_lang = smart_join(lang_dir, f"{base_name}.pt")
            if fallback_lang != lang_candidates[0]:
                lang_candidates.append(fallback_lang)

            episodes.append({
                "episode_name": name,
                "source_episode_name": base_name,
                "task_name": task_name,
                "dataset_name": dataset_name,
                "latent_path": latent_file,
                "qpos_path": qpos_file,
                "lang_candidates": lang_candidates,
                "lang_path": lang_candidates[0],
                "phase_start_30": phase_start_30,
            })
        return episodes

    def _load_episodes_from_dir(self) -> List[Dict[str, Any]]:
        all_episodes: List[Dict[str, Any]] = []
        for dataset_dir in self.dataset_dirs:
            dataset_dir_str = str(dataset_dir).rstrip("/")
            if is_remote(dataset_dir_str):
                dataset_subdirs = [str(p).rstrip("/") for p in smart_listdir_dirs(dataset_dir_str)]
                for dataset_subdir in dataset_subdirs:
                    dataset_name = dataset_subdir.rsplit("/", 1)[-1]
                    if not self._dataset_name_allowed(dataset_name):
                        continue
                    for task_dir in smart_listdir_dirs(dataset_subdir):
                        all_episodes.extend(self._scan_ario_task_folder(str(task_dir).rstrip("/"), dataset_name))
                continue
            else:
                d = Path(dataset_dir)
                if not d.exists():
                    logger.warning(f"dataset_dir does not exist: {dataset_dir}")
                    continue
                dataset_name = d.name
                task_dirs = [str(p) for p in sorted(d.iterdir()) if p.is_dir()]
            if not self._dataset_name_allowed(dataset_name):
                continue

            for task_dir in task_dirs:
                local_style = all(
                    smart_isdir(smart_join(task_dir, name))
                    for name in (self.latent_dir, self._default_task_action_dir(), self.lang_dir)
                )
                if local_style:
                    all_episodes.extend(self._scan_task_folder(task_dir, dataset_name))
                else:
                    all_episodes.extend(self._scan_ario_task_folder(task_dir, dataset_name))
        return all_episodes

    def _load_episodes_from_manifest(self) -> List[Dict[str, Any]]:
        all_episodes: List[Dict[str, Any]] = []
        for dataset_dir in self.dataset_dirs:
            dataset_name = Path(str(dataset_dir).rstrip("/")).name
            if not self._dataset_name_allowed(dataset_name):
                continue

            manifest_url = smart_join(str(dataset_dir), "manifest.json")
            manifest = smart_load_json(manifest_url)
            for entry in manifest["episodes"]:
                task_name = entry["task_name"]
                if not self._task_allowed(task_name):
                    continue

                root = str(dataset_dir)
                qpos_rel = self._resolve_manifest_action_path(entry)
                if qpos_rel is None:
                    continue

                lang_root_rel = entry.get("lang_root")
                lang_path_rel = entry.get("lang_path")
                ep = {
                    "episode_name": entry["episode_name"],
                    "source_episode_name": entry["source_episode_name"],
                    "task_name": task_name,
                    "dataset_name": entry.get("dataset_name", dataset_name),
                    "phase_start_30": entry.get("phase_start_30", 0),
                    "episode_path": smart_join(root, entry.get("episode_path", entry["episode_name"])),
                    "latent_root": smart_join(root, entry["latent_root"]) if entry.get("latent_root") else None,
                    "latent_path": self._latent_path(smart_join(root, entry["latent_path"])) if entry.get("latent_path") else None,
                    "qpos_path": smart_join(root, qpos_rel),
                    "lang_root": smart_join(root, lang_root_rel) if lang_root_rel else None,
                    "lang_path": smart_join(root, lang_path_rel) if lang_path_rel else None,
                }
                all_episodes.append(ep)
        return all_episodes

    def _load_episodes_from_iceberg(self) -> List[Dict[str, Any]]:
        from pyiceberg.catalog import load_catalog
        from pyiceberg.expressions import EqualTo

        db_path = self.catalog_path
        warehouse = str(Path(db_path).parent)
        catalog = load_catalog(
            "iceberg",
            type="sql",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{Path(warehouse).resolve()}",
        )
        table = catalog.load_table("robotwin.episodes")

        row_filter = None
        if self.task_mode == "single":
            row_filter = EqualTo("task_name", self.task_name)
        scan = table.scan(row_filter=row_filter) if row_filter else table.scan()
        df = scan.to_pandas()

        multi_dir = len(self.dataset_dirs) > 1
        has_root_col = multi_dir and "dataset_root" in df.columns
        default_root = str(self.dataset_dirs[0])
        all_episodes: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            root = (row.get("dataset_root") or default_root) if has_root_col else default_root
            dataset_name = row.get("dataset_name") or Path(root.rstrip("/")).name
            if not self._dataset_name_allowed(dataset_name):
                continue

            task_name = row["task_name"]
            if not self._task_allowed(task_name):
                continue

            qpos_rel = self._resolve_manifest_action_path(row)
            if qpos_rel is None:
                continue

            lang_root_rel = row.get("lang_root")
            lang_path_rel = row.get("lang_path")
            ep = {
                "episode_name": row["episode_name"],
                "source_episode_name": row["source_episode_name"],
                "task_name": task_name,
                "dataset_name": dataset_name,
                "phase_start_30": row.get("phase_start_30", 0),
                "episode_path": smart_join(root, row.get("episode_path", row["episode_name"])),
                "latent_root": smart_join(root, row["latent_root"]) if row.get("latent_root") else None,
                "latent_path": self._latent_path(smart_join(root, row["latent_path"])) if row.get("latent_path") else None,
                "qpos_path": smart_join(root, qpos_rel),
                "lang_root": smart_join(root, lang_root_rel) if lang_root_rel else None,
                "lang_path": smart_join(root, lang_path_rel) if lang_path_rel else None,
            }
            all_episodes.append(ep)
        return all_episodes

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _latent_path(self, path: str) -> str:
        if path is None:
            return None
        if self.latent_dir == "vae_latents":
            return path
        return path.replace("/vae_latents/", f"/{self.latent_dir}/")

    def _default_episode_action_filename(self) -> str:
        if self.action_mode == "qpos20_rel6d":
            return "tjwj.pt"
        return "endpose.pt" if "eef" in self.action_mode else "qpos.pt"

    def _default_task_action_dir(self) -> str:
        if self.action_mode == "qpos20_rel6d":
            return "hand_qpos"
        return "endpose" if "eef" in self.action_mode else "qpos"

    def _resolve_entry_action_path(self, entry: Dict[str, Any]) -> Optional[str]:
        if self.action_mode == "qpos20_rel6d":
            return entry.get("tjwj_path") or entry.get("hand_path") or entry.get("hand_qpos_path") or entry.get("qpos_path")
        return entry.get("endpose_path") if "eef" in self.action_mode else entry.get("qpos_path")

    def _resolve_manifest_action_path(self, entry: Dict[str, Any]) -> Optional[str]:
        if self.action_mode == "qpos20_rel6d":
            return entry.get("tjwj_path") or entry.get("hand_path") or entry.get("hand_qpos_path") or entry.get("qpos_path")
        return entry.get("endpose_path") if "eef" in self.action_mode else entry.get("qpos_path")

    def _sample_action_path(self, ep: Dict[str, Any]) -> Optional[str]:
        if self.action_mode == "qpos20_rel6d":
            return ep.get("tjwj_path") or ep.get("hand_path") or ep.get("qpos_path")
        return ep.get("endpose_path") if "eef" in self.action_mode else ep.get("qpos_path")

    @staticmethod
    def _canonical_gripper_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
        if not dataset_name:
            return None
        base = str(dataset_name).rstrip("/").rsplit("/", 1)[-1]
        aliases = {
            "AgiBotDigitalWorld": "AgiBotDigitalWorld",
            "AgiBotWorld-Alpha": "AgiBotWorld-Alpha-Split",
            "AgiBotWorld-Alpha-Split": "AgiBotWorld-Alpha-Split",
            "AgiBotWorld-Beta": "AgiBotWorld-Beta-Split",
            "AgiBotWorld-Beta-Split": "AgiBotWorld-Beta-Split",
            "robomind1": "RoboMind1.0",
            "robomind2": "RoboMind2.0",
            "RoboMind1.0": "RoboMind1.0",
            "RoboMind2.0": "RoboMind2.0",
        }
        return aliases.get(base, base)

    @staticmethod
    def _gripper_stats_type_key(action_dim: int) -> Optional[str]:
        if action_dim == 14:
            return "14D"
        if action_dim == 16:
            return "16D"
        return None

    def _gripper_stats_path(self) -> Path:
        return ENDPOSE_GRIPPER_STATS_PATH if "eef" in self.action_mode else QPOS_GRIPPER_STATS_PATH

    def _load_gripper_stats(self) -> Dict[str, Any]:
        stats_path = str(self._gripper_stats_path())
        cached = self._gripper_stats_cache.get(stats_path)
        if cached is not None:
            return cached
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
        except Exception as e:
            logger.warning("Failed to load gripper stats %s: %s", stats_path, e)
            stats = {}
        self._gripper_stats_cache[stats_path] = stats
        return stats

    def _episode_gripper_stats(self, ep: Dict[str, Any], action_dim: int) -> Optional[Dict[str, Any]]:
        dataset_name = self._canonical_gripper_dataset_name(ep.get("dataset_name"))
        if not dataset_name:
            return None
        all_stats = self._load_gripper_stats()
        dataset_stats = all_stats.get(dataset_name)
        if not isinstance(dataset_stats, dict):
            return None

        robot_name = ep.get("robot_name")
        robot_stats = dataset_stats.get(robot_name) if robot_name else None
        if robot_stats is None and len(dataset_stats) == 1:
            robot_stats = next(iter(dataset_stats.values()))
        if not isinstance(robot_stats, dict):
            return None

        shape_key = self._gripper_stats_type_key(action_dim)
        if shape_key is None:
            return None
        shape_stats = robot_stats.get(shape_key)
        return shape_stats if isinstance(shape_stats, dict) else None

    def _normalize_gripper_columns(self, qpos: torch.Tensor, ep: Dict[str, Any]) -> torch.Tensor:
        stats = self._episode_gripper_stats(ep, qpos.shape[-1])
        if stats is None:
            return qpos

        gripper_range = stats.get("gripper_range")
        if not isinstance(gripper_range, dict):
            return qpos
        lo = gripper_range.get("p1")
        hi = gripper_range.get("p99")
        if lo is None or hi is None:
            return qpos
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return qpos

        qpos = qpos.clone()
        for side_name in ("gripper_left", "gripper_right"):
            side_stats = stats.get(side_name)
            if not isinstance(side_stats, dict):
                continue
            col_idx = side_stats.get("col_idx")
            if col_idx is None:
                continue
            col_idx = int(col_idx)
            if not (0 <= col_idx < qpos.shape[-1]):
                continue
            qpos[:, col_idx] = ((qpos[:, col_idx] - lo) / (hi - lo)).clamp_(0.0, 1.0)
        return qpos

    def _resolve_episode_action_path(self, episode_dir: str) -> Optional[str]:
        return smart_join(episode_dir, self._default_episode_action_filename())

    @staticmethod
    def _phase_index_from_component(name: str) -> Optional[int]:
        for pattern in (r"(\d+)\s*phase", r"phase[_-]?(\d+)", r"_p(\d+)$"):
            m = re.search(pattern, name, flags=re.IGNORECASE)
            if m is not None:
                return int(m.group(1))
        return None

    def _scan_ario_task_folder(self, task_path: str, dataset_name: Optional[str]) -> List[Dict[str, Any]]:
        task_name = task_path.rstrip("/").rsplit("/", 1)[-1]
        if not self._task_allowed(task_name):
            return []

        episodes = []
        for episode_dir in smart_listdir_dirs(task_path):
            episode_dir = str(episode_dir).rstrip("/")
            episode_name = episode_dir.rsplit("/", 1)[-1]
            latent_root = smart_join(episode_dir, self.latent_root) if self.latent_root else None
            latent_path = smart_join(episode_dir, self.latent_subpath) if self.latent_subpath else None
            episodes.append({
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task_name,
                "dataset_name": dataset_name,
                "episode_path": episode_dir,
                "latent_root": latent_root,
                "latent_path": latent_path,
                "qpos_path": smart_join(episode_dir, self._default_episode_action_filename()),
                "lang_root": smart_join(episode_dir, self.lang_root) if self.lang_root else None,
                "lang_path": smart_join(episode_dir, self.lang_subpath) if self.lang_subpath else None,
                "phase_start_30": 0,
            })
        return episodes

    def _resolve_latent_candidates(self, ep: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        latent_root = ep.get("latent_root")

        metadata = self._lookup_episode_metadata(ep)
        if metadata is not None and metadata.get("latent_scanned") and latent_root:
            phase_names = list(metadata.get("phase_dirs", []))
            for phase_name in phase_names:
                phase_index = self._phase_index_from_component(phase_name)
                if (
                    self.latent_phases_set is not None
                    and phase_index is not None
                    and phase_index not in self.latent_phases_set
                ):
                    continue
                phase_dir = smart_join(latent_root, phase_name)
                latent_path = smart_join(phase_dir, self.latent_file_name)
                candidates.append({
                    "latent_path": latent_path,
                    "phase_start_30": phase_index or 0,
                })
            random.shuffle(candidates)
        elif latent_root:
            if self.latent_phases_set is not None:
                for phase_index in random.sample(
                    sorted(self.latent_phases_set),
                    k=len(self.latent_phases_set),
                ):
                    phase_dir = smart_join(latent_root, f"{phase_index}phase_5hz")
                    latent_path = smart_join(phase_dir, self.latent_file_name)
                    candidates.append({
                        "latent_path": latent_path,
                        "phase_start_30": int(phase_index),
                    })
            elif smart_isdir(latent_root):
                for phase_dir in smart_listdir_dirs(latent_root):
                    phase_dir = str(phase_dir).rstrip("/")
                    phase_name = phase_dir.rsplit("/", 1)[-1]
                    phase_index = self._phase_index_from_component(phase_name)
                    latent_path = smart_join(phase_dir, self.latent_file_name)
                    candidates.append({
                        "latent_path": latent_path,
                        "phase_start_30": phase_index or 0,
                    })
        latent_path = ep.get("latent_path")
        if not candidates and latent_path:
            candidates.append({
                "latent_path": latent_path,
                "phase_start_30": int(ep.get("phase_start_30", 0)),
            })
        return candidates

    def _latent_to_action_range(self, latent_idx: int, frame_stride: int, phase_start_30: int = 0) -> Tuple[int, int]:
        r = self.action_upsample_ratio
        if latent_idx == 0:
            return phase_start_30 * r, phase_start_30 * r + r
        apl = 4 * frame_stride * r
        start = (phase_start_30 + 1) * r + (latent_idx - 1) * apl
        return start, start + apl

    @staticmethod
    def _upsample_qpos(qpos: torch.Tensor, ratio: int) -> torch.Tensor:
        if ratio == 1 or qpos.shape[0] <= 1:
            return qpos
        new_len = (qpos.shape[0] - 1) * ratio + 1
        return F.interpolate(
            qpos.T.unsqueeze(0),
            size=new_len,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T

    def _maybe_convert_eef_order(self, qpos: torch.Tensor) -> torch.Tensor:
        if qpos.shape[-1] == 16 and self.eef_quat_order == "wxyz":
            return eef_quat_wxyz_to_xyzw(qpos)
        return qpos

    def _maybe_output_wxyz(self, qpos: torch.Tensor) -> torch.Tensor:
        if self.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"} and qpos.shape[-1] == 16:
            return eef_quat_xyzw_to_wxyz(qpos)
        return qpos

    def _resolve_language_candidates(self, ep: Dict[str, Any]) -> List[str]:
        episode_key = self._episode_identity(ep)
        cached = self._language_candidates_cache.get(episode_key)
        if cached is not None:
            return cached

        candidates: List[str] = []
        lang_root = ep.get("lang_root")
        lang_path = ep.get("lang_path")
        explicit_candidates = ep.get("lang_candidates") or []
        metadata = self._lookup_episode_metadata(ep)

        if explicit_candidates:
            candidates.extend(explicit_candidates)
        elif lang_root:
            if metadata is not None and metadata.get("lang_scanned"):
                candidates.extend(
                    smart_join(lang_root, file_name)
                    for file_name in metadata.get("lang_files", [])
                )
            else:
                for file_name in DEFAULT_UMT5_CANDIDATE_FILES:
                    candidates.append(smart_join(lang_root, file_name))

        if not candidates and lang_path:
            candidates.append(lang_path)

        self._language_candidates_cache[episode_key] = candidates
        return candidates

    def _disable_language_for_episode(self, ep: Dict[str, Any]) -> None:
        self._language_candidates_cache[self._episode_identity(ep)] = []

    def _load_language_embedding(self, ep: Dict[str, Any]) -> Optional[torch.Tensor]:
        if self.treat_as_irrelevant and self.empty_lang_path:
            lang_path = self.empty_lang_path
        else:
            candidates = self._resolve_language_candidates(ep)
            if not candidates:
                return None
            lang_path = None
            last_error = None
            for candidate in random.sample(candidates, k=len(candidates)):
                try:
                    embedding_data = smart_load_pt(candidate, self.cache_dir, map_location="cpu")
                    lang_path = candidate
                    break
                except Exception as e:
                    last_error = e
            if lang_path is None:
                self._disable_language_for_episode(ep)
                logger.warning(
                    "No loadable language embedding for %s | dataset=%s task=%s storage_key=%s, proceeding without: %s",
                    ep.get("episode_name", "?"),
                    ep.get("dataset_name"),
                    ep.get("task_name"),
                    ep.get("episode_storage_key"),
                    last_error,
                )
                return None

        if self.treat_as_irrelevant and self.empty_lang_path:
            try:
                embedding_data = smart_load_pt(lang_path, self.cache_dir, map_location="cpu")
            except Exception as e:
                logger.warning(
                    "Failed to load irrelevant empty language embedding for %s, proceeding without: %s",
                    ep.get("episode_name", "?"),
                    e,
                )
                return None
        if isinstance(embedding_data, list):
            embeddings = embedding_data[random.randint(0, len(embedding_data) - 1)]
        elif isinstance(embedding_data, torch.Tensor):
            embeddings = embedding_data[random.randint(0, embedding_data.shape[0] - 1)] if embedding_data.ndim == 3 else embedding_data
        else:
            raise ValueError(f"Unexpected type: {type(embedding_data)}")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings

    def _load_actions_and_states(
        self,
        ep: Dict[str, Any],
        qpos_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        use_state: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        qpos = smart_load_pt(qpos_path, self.cache_dir, map_location="cpu")
        if self.action_mode == "qpos20_rel6d" and isinstance(qpos, dict):
            if isinstance(qpos.get("action"), torch.Tensor):
                qpos = qpos["action"]
            elif isinstance(qpos.get("action_xyzw"), torch.Tensor):
                qpos = qpos["action_xyzw"]
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos)
        qpos = qpos.float()
        if self.action_mode != "qpos20_rel6d":
            qpos = self._normalize_gripper_columns(qpos, ep)
            qpos = self._maybe_convert_eef_order(qpos)

        if self.action_mode == "qpos20_rel6d":
            if qpos.shape[-1] != 58:
                raise ValueError(f"qpos20_rel6d expects 58D qpos+wrist9 actions, got {tuple(qpos.shape)}")
            if self.action_upsample_ratio != 1:
                raise ValueError("qpos20_rel6d requires precomputed train-rate 58D actions in this dataset")
        elif self.action_mode in {"rel_eef", "abs_eef"}:
            qpos = upsample_eef(qpos, self.action_upsample_ratio)
        elif self.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"}:
            qpos = upsample_abs_eef_16d(qpos, self.action_upsample_ratio)
            qpos = self._maybe_output_wxyz(qpos)
        else:
            qpos = self._upsample_qpos(qpos, self.action_upsample_ratio)

        need_states = use_state or self.action_mode in {"rel_eef", "rel_eef_wxyz", "qpos20_rel6d"}
        n = len(qpos)
        apl = self.actions_per_latent
        action_chunks = []
        state_list = []
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            a_start = min(a_start, n - 1)
            a_end = min(a_end, n)
            chunk = qpos[a_start:a_end]
            if chunk.shape[0] < apl:
                chunk = torch.cat([chunk, chunk[-1:].expand(apl - chunk.shape[0], -1)], dim=0)
            else:
                chunk = chunk[:apl]
            action_chunks.append(chunk)
            if need_states:
                state_idx = max(min(a_start, n - 1) - 1, 0)
                state_list.append(qpos[state_idx])

        actions = torch.cat(action_chunks, dim=0)
        states = torch.stack(state_list, dim=0) if need_states else None
        return states, actions

    def _build_progress_values(self, start: int, window_size: int, total_latents: int) -> torch.Tensor:
        c = self.progress_c
        if c is None or c <= 0 or total_latents <= 1:
            return torch.zeros(window_size, dtype=torch.float32)
        abs_idx = torch.arange(window_size, dtype=torch.float32) + start
        s = (abs_idx / (total_latents - 1)).clamp(0, 1)
        values = (c ** (1 - s) - c) / (1 - c)
        values[abs_idx >= total_latents] = 0.0
        return values

    def _build_ttc_values(self, start: int, window_size: int, total_latents: int) -> torch.Tensor:
        if total_latents <= 1:
            return torch.zeros(window_size, dtype=torch.float32)
        abs_idx = torch.arange(window_size, dtype=torch.float32) + start
        remaining = (total_latents - 1) - abs_idx
        values = -remaining / float(total_latents - 1)
        values = values.clamp(min=-1.0, max=0.0)
        values[abs_idx >= total_latents] = -1.0
        return values

    def _build_vlac_values(
        self,
        start: int,
        window_size: int,
        total_latents: int,
        negate: bool = False,
    ) -> torch.Tensor:
        if total_latents <= 1:
            return torch.zeros(window_size, dtype=torch.float32)

        harmonic_T = (1.0 / torch.arange(1, total_latents + 1, dtype=torch.float64)).sum().item()
        per_frame_reward = 1.0 / (total_latents - torch.arange(total_latents, dtype=torch.float64))
        cum_value = per_frame_reward.cumsum(0) / harmonic_T

        abs_idx = torch.arange(window_size, dtype=torch.long) + start
        values = torch.zeros(window_size, dtype=torch.float32)
        valid = abs_idx < total_latents
        values[valid] = cum_value[abs_idx[valid]].float()
        values[~valid] = 1.0
        if negate:
            values = -values
        return values

    def _build_irrelevant_values(self, window_size: int) -> torch.Tensor:
        if self.value_target in {"progress", "vlac"}:
            return torch.zeros(window_size, dtype=torch.float32)
        return torch.full((window_size,), -1.0, dtype=torch.float32)

    def _build_value_targets(self, start: int, window_size: int, total_latents: int) -> Optional[torch.Tensor]:
        if self.value_target is None:
            return None
        if self.treat_as_irrelevant:
            if self.value_target == "vlac":
                return self._build_vlac_values(start, window_size, total_latents, negate=True)
            return self._build_irrelevant_values(window_size)
        if self.value_target == "progress":
            return self._build_progress_values(start, window_size, total_latents)
        if self.value_target == "ttc":
            return self._build_ttc_values(start, window_size, total_latents)
        if self.value_target == "vlac":
            return self._build_vlac_values(start, window_size, total_latents, negate=False)
        return None

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_episodes * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        window_size = self.max_latent_window

        for _ in range(8):
            if not self.episode_files:
                continue
            ep = random.choice(self.episode_files)

            try:
                candidates = self._resolve_latent_candidates(ep)
                if not candidates:
                    self._drop_episode(ep, "no latent candidates")
                    continue
                chosen = None
                latent_data = None
                last_error = None
                candidate_order = random.sample(candidates, k=len(candidates))
                if self.latent_max_attempts is not None:
                    candidate_order = candidate_order[: self.latent_max_attempts]
                attempted_candidates = len(candidate_order)
                for candidate in candidate_order:
                    try:
                        latent_data = smart_load_pt(
                            candidate["latent_path"], self.cache_dir, map_location="cpu"
                        )
                        chosen = candidate
                        break
                    except Exception as e:
                        last_error = e
                if chosen is None or latent_data is None:
                    self._drop_episode(
                        ep,
                        f"all latent candidates failed after {attempted_candidates} attempts: {last_error}",
                    )
                    continue
                if isinstance(latent_data, dict) and "phases" in latent_data:
                    phases = latent_data["phases"]
                    if self.latent_phases_set is not None:
                        phases = [p for i, p in enumerate(phases) if i in self.latent_phases_set]
                    if not phases:
                        self._drop_episode(ep, "latent file has no usable phases after filtering")
                        continue
                    phase_data = random.choice(phases)
                    latents = phase_data["latents"]
                    frame_stride = int(latent_data.get("frame_stride", phase_data.get("frame_stride", 6)))
                    phase_start_30 = int(phase_data.get("phase_start_30", 0))
                elif isinstance(latent_data, torch.Tensor):
                    latents = latent_data
                    frame_stride = self.video_action_freq_ratio
                    phase_start_30 = int(chosen.get("phase_start_30", ep.get("phase_start_30", 0)))
                else:
                    latents = latent_data["latents"]
                    frame_stride = int(latent_data.get("frame_stride", self.video_action_freq_ratio))
                    phase_start_30 = int(latent_data.get("phase_start_30", chosen.get("phase_start_30", ep.get("phase_start_30", 0))))

                total_latents = latents.shape[1]
                if total_latents > window_size:
                    start = random.randint(0, total_latents - window_size)
                    latents_crop = latents[:, start:start + window_size]
                    valid_len = window_size
                elif total_latents < window_size:
                    start = 0
                    latents_crop = F.pad(latents, (0, 0, 0, 0, 0, window_size - total_latents))
                    valid_len = total_latents
                else:
                    start = 0
                    latents_crop = latents
                    valid_len = window_size

                actual_latents = min(total_latents - start, window_size)
                action_path = self._sample_action_path(ep)
                if action_path is None:
                    self._drop_episode(ep, "missing action path")
                    continue
                try:
                    states, actions = self._load_actions_and_states(
                        ep, action_path, start, actual_latents, frame_stride, phase_start_30, self.use_state
                    )
                except Exception as e:
                    self._drop_episode(ep, f"action/state load failed from {action_path}: {e}")
                    continue

                target_action_len = window_size * self.actions_per_latent
                if actions.shape[0] < target_action_len:
                    pad_len = target_action_len - actions.shape[0]
                    actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
                if states is not None and states.shape[0] < window_size:
                    pad_len = window_size - states.shape[0]
                    states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)

                language_embedding = self._load_language_embedding(ep)
                sample: Dict[str, Any] = {
                    "latents": latents_crop,
                    "actions": actions,
                    "valid_len": valid_len,
                    "latent_start": int(start),
                    "is_task_irrelevant": int(self.treat_as_irrelevant),
                    "is_success": int(not self.treat_as_irrelevant),
                }
                if language_embedding is not None:
                    sample["language_embedding"] = language_embedding
                if states is not None:
                    sample["states"] = states
                values = self._build_value_targets(start, window_size, total_latents)
                if values is not None:
                    sample["values"] = values
                return sample
            except Exception as e:
                logger.warning(
                    "Sample error (%s | dataset=%s task=%s storage_key=%s action_path=%s): %s",
                    ep.get("episode_name", "?"),
                    ep.get("dataset_name"),
                    ep.get("task_name"),
                    ep.get("episode_storage_key"),
                    self._sample_action_path(ep),
                    e,
                )
                continue
        return None

    def _log_episode_summary(self):
        if self.task_to_episodes:
            logger.info(
                "PretrainDataset loaded %d episodes across %d tasks",
                self.total_episodes,
                len(self.task_to_episodes),
            )
        elif self.episode_files:
            logger.info("PretrainDataset loaded %d episodes", self.total_episodes)
        else:
            logger.warning("PretrainDataset loaded no episodes")

    @staticmethod
    def _normalize_bos(path: str) -> str:
        if path.startswith("bos://"):
            return "s3://" + path[len("bos://"):]
        if path.startswith("bos:/"):
            return "s3://" + path[len("bos:/"):]
        return path
