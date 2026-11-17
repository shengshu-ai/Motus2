# Real-world dataset base class for latent-space AR training.
#
# Standalone Dataset subclass — reads episode metadata from a JSONL file,
# DuckDB catalog, or by scanning a local/remote dataset_dir.  Produces
# sliding-window training samples from pre-computed VAE latents.
# Subclasses (AlohaAgilex2Dataset, ACOneDataset) only need to override
# ``_transform_sample()`` for per-robot normalization.

import json
import re
import random
import logging
from functools import reduce
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist

from data.utils.smart_io import (
    is_remote, smart_join, stem,
    smart_isdir, smart_glob_pt, smart_exists,
    smart_listdir,
    smart_load_pt,
)
from data.utils.rotation_utils import eef_rpy_to_quat, upsample_abs_eef_16d, upsample_eef
from utils.relative_eef import eef_quat_wxyz_to_xyzw, eef_quat_xyzw_to_wxyz

logger = logging.getLogger(__name__)

DEFAULT_UMT5_CANDIDATE_FILES = (
    "environment_robot_behavior.pt",
    "environment.pt",
    "environment_LRbehavior.pt",
    "environment_behavior.pt",
    "robot_LRbehavior.pt",
    "robot_behavior.pt",
    "vidu_json_LRbehavior.pt",
    "vidu_json_behavior.pt",
)


class RealWorldDataset(data.Dataset):
    """Base dataset for real-world robot data with pre-computed VAE latents.

    Three episode-loading modes (mutually exclusive, checked in order):

    1. **DuckDB mode** (``duckdb_path``): query a DuckDB catalog to discover
       episodes.  Supports SQL LIKE patterns for task filtering via
       ``task_name`` (e.g. ``"fold%"`` matches all fold tasks).  Multiple
       patterns are OR-ed: YAML list, or a comma-separated string.

    2. **JSONL mode** (``jsonl_path``): each line has
       ``episode_bos_path, task, qpos_path, (optional) robot``.
       Latent / language paths = episode_bos_path + subpath.

    3. **Directory mode** (``dataset_dir``): scans a local or remote directory
       with structure::

           dataset_dir/
             task_name/
               vae_latents/      (or vae_latents_24phase/)
                 episode1.pt
               qpos/
                 episode1.pt
               umt5_wan/
                 episode1.pt

       Supports local directories, BOS/S3 manifest.json, and PyIceberg catalogs.

    Training samples are produced by randomly cropping a window of
    ``max_latent_window`` latent frames (pad if episode is shorter).
    """

    def __init__(
        self,
        # --- JSONL mode ---
        jsonl_path: Optional[str] = None,
        latent_subpath: str = "video_features/wan2.2/0phase_5hz/video_3v_VAE.pt",
        lang_subpath: Optional[str] = "text_features/UMT5/environment_robot_behavior.pt",
        robot: Optional[str] = None,
        # --- DuckDB mode ---
        duckdb_path: Optional[str] = None,
        duckdb_table: str = "real_world_data",
        # --- Directory mode ---
        dataset_dir=None,
        catalog_path: Optional[str] = None,
        latent_dir: str = "vae_latents",
        latent_phases: Optional[List[int]] = None,
        # JSONL: if a line has no "latent_phases" and yaml latent_phases is null,
        # list episode_base/video_features/wan2.2/*phase* and train on all found phases.
        latent_phases_auto_discover: bool = False,
        latent_max_attempts: Optional[int] = None,
        # Task filtering
        # task_mode: "single" = filter to task_name, "multi" = keep all tasks.
        # Sampling is always episode-uniform after filtering.
        # task_name: one str, comma-separated str, or list of str; SQL LIKE via "%"
        # suffix (DuckDB / JSONL), or trailing "*" for prefix (JSONL/manifest).
        task_mode: str = "multi",
        task_name: Any = None,
        max_episodes: Optional[int] = None,
        # Sliding-window params
        max_latent_window: int = 10,
        video_action_freq_ratio: int = 6,
        action_mode: str = "joint",
        use_state: bool = True,
        eef_quat_order: str = "xyzw",
        # Standard params
        val: bool = False,
        # Remote data support
        cache_dir: Optional[str] = None,
        # Action upsampling (for higher-res latents with more spatial tokens).
        # 1 = no upsampling (30Hz), 4 = interpolate to 120Hz.
        action_upsample_ratio: int = 1,
        # Compat (ignored)
        **kwargs,
    ):
        # --- JSONL mode fields ---
        self.jsonl_path = jsonl_path
        self.latent_subpath = latent_subpath
        self.lang_subpath = lang_subpath
        self.robot_filter = robot

        # --- DuckDB mode fields ---
        self.duckdb_path = duckdb_path
        self.duckdb_table = duckdb_table

        # --- Directory mode fields ---
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
        self.latent_phases_auto_discover = bool(latent_phases_auto_discover)
        self.latent_max_attempts = int(latent_max_attempts) if latent_max_attempts else 1

        if not jsonl_path and not dataset_dir and not duckdb_path:
            raise ValueError("Must provide one of: jsonl_path, dataset_dir, or duckdb_path")

        # --- Common fields ---
        self.task_mode = task_mode
        self.task_name = task_name
        self.task_name_patterns = self._normalize_task_name_patterns(task_name)
        self.max_episodes = max_episodes
        self.max_latent_window = max_latent_window
        self.video_action_freq_ratio = video_action_freq_ratio
        self.action_mode = action_mode
        self.use_state = use_state
        self.eef_quat_order = eef_quat_order
        self.val = val
        self.cache_dir = cache_dir
        self.action_upsample_ratio = action_upsample_ratio

        # Each latent (after l_0) covers 4 video frames, each video frame
        # spans ``video_action_freq_ratio`` 30Hz action steps.
        # With action_upsample_ratio > 1, actions are interpolated to a higher
        # frequency so more action tokens match the increased spatial tokens.
        self.actions_per_latent = 4 * video_action_freq_ratio * action_upsample_ratio  # 24 or 96

        if task_mode == "single" and not self.task_name_patterns:
            raise ValueError("Single task mode requires non-empty task_name")
        if self.eef_quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported eef_quat_order: {self.eef_quat_order}")

        self.episode_files: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.task_weights: Dict[str, float] = {}
        self.total_episodes = 0
        self._language_candidates_cache: Dict[str, List[str]] = {}
        self._episode_failure_counts: Dict[str, int] = {}
        self._load_episodes_distributed()

        logger.info(
            f"RealWorldDataset: task_mode={task_mode}, "
            f"task_name_patterns={self.task_name_patterns!r}, "
            f"max_latent_window={max_latent_window}, "
            f"actions_per_latent={self.actions_per_latent}, "
            f"action_upsample_ratio={action_upsample_ratio}, "
            f"latent_max_attempts={self.latent_max_attempts}, "
            f"latent_phases_auto_discover={self.latent_phases_auto_discover}, "
            f"episodes={self.total_episodes}"
        )
        # Log sample paths for debugging
        sample_eps = []
        if self.episode_files:
            sample_eps = self.episode_files[:3]
        elif self.task_to_episodes:
            for t, eps in list(self.task_to_episodes.items())[:2]:
                sample_eps.extend(eps[:2])
        for ep in sample_eps:
            logger.info(f"  sample: ep={ep['episode_name']} "
                        f"latent={ep['latent_path']} "
                        f"action={self._sample_action_path(ep)}")

    # ------------------------------------------------------------------
    # Episode loading
    # ------------------------------------------------------------------

    def _load_episodes_distributed(self):
        """Load episodes on rank 0 only, then broadcast to all other ranks."""
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if rank == 0:
            self._load_episodes()

        if is_dist:
            obj = [(self.episode_files, self.task_to_episodes,
                    self.task_weights, self.total_episodes)]
            dist.broadcast_object_list(obj, src=0)
            if rank != 0:
                (self.episode_files, self.task_to_episodes,
                 self.task_weights, self.total_episodes) = obj[0]

    def _load_episodes(self):
        """Dispatch to DuckDB, JSONL, or directory-based loading."""
        if self.duckdb_path is not None:
            # DuckDB catalog mode
            self._load_episodes_from_duckdb()
        elif self.dataset_dirs is not None:
            # Directory mode
            if self.catalog_path and self.catalog_path != "manifest":
                self._load_episodes_from_iceberg()
            elif self.catalog_path == "manifest" or any(
                is_remote(str(d)) for d in self.dataset_dirs
            ):
                self._load_episodes_from_manifest()
            else:
                self._load_episodes_from_dir()
            self._log_episode_summary("Directory")
        else:
            # JSONL mode
            episodes = self._load_episode_list_jsonl()
            self._organize_episodes(episodes)
            self._log_episode_summary("JSONL")

    # ------------------------------------------------------------------
    # JSONL-based loading
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_episode_qpos_path(base: str, episode_name: str) -> str:
        if "/ARIO-format/" in base:
            return f"{base}/qpos.pt"
        return f"{base}/{episode_name}_qpos_tts.pt"

    @staticmethod
    def _infer_episode_endpose_path(base: str, episode_name: str) -> str:
        if "/ARIO-format/" in base:
            return f"{base}/endpose.pt"
        return f"{base}/{episode_name}_eef_tts.pt"

    @staticmethod
    def _strip_robot_task_prefix(task_name: Optional[str]) -> Optional[str]:
        if not task_name or "--" not in task_name:
            return task_name
        return task_name.split("--", 1)[1]

    @staticmethod
    def _normalize_task_name_patterns(task_name: Any) -> List[str]:
        """Build a non-empty list of filter patterns from YAML / CLI.

        - ``str`` with commas → split (OR semantics).
        - ``str`` without commas → one pattern.
        - sequence (list / tuple / ListConfig) → one pattern per element.
        """
        if task_name is None:
            return []
        if isinstance(task_name, str):
            s = task_name.strip()
            if not s:
                return []
            if "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            return [s]
        try:
            seq = list(task_name)
        except TypeError:
            s = str(task_name).strip()
            return [s] if s else []
        out: List[str] = []
        for p in seq:
            t = str(p).strip()
            if t:
                out.append(t)
        return out

    def _matches_any_task_pattern(self, task_name: Optional[str]) -> bool:
        """True if ``task_name`` matches any configured pattern (OR)."""
        if not task_name:
            return False
        stripped = self._strip_robot_task_prefix(task_name)
        for pat in self.task_name_patterns:
            if pat.endswith(("%", "*")):
                prefix = pat[:-1]
                if task_name.startswith(prefix) or (
                    stripped and stripped.startswith(prefix)
                ):
                    return True
            elif task_name == pat or stripped == pat:
                return True
        return False

    def _jsonl_entry_matches_robot(self, entry: Dict[str, Any]) -> bool:
        """Match JSONL row to ``robot`` filter.

        Catalog lines often omit ``robot``; then accept when ``task`` starts with
        ``"{robot}--"`` (e.g. ``ac_one--draw_on_...``).
        """
        if not self.robot_filter:
            return True
        er = entry.get("robot")
        if er is not None:
            return er == self.robot_filter
        task = entry.get("task") or ""
        return task.startswith(f"{self.robot_filter}--")

    def _task_matches(self, task_name: Optional[str]) -> bool:
        if not task_name:
            return False
        if self.task_mode != "single" or not self.task_name_patterns:
            return True
        return self._matches_any_task_pattern(task_name)

    def _discover_latent_phases_wan22(self, episode_base: str) -> List[int]:
        """Parse ``{episode_base}/video_features/wan2.2/<N>phase_*/`` directory names → ints."""
        root = smart_join(self._normalize_bos(episode_base.rstrip("/")), "video_features/wan2.2")
        if not smart_isdir(root):
            return []
        try:
            names = smart_listdir(root)
        except OSError as e:
            logger.debug("wan2.2 listdir failed %s: %s", root, e)
            return []
        out: List[int] = []
        for n in names:
            m = re.match(r"^(\d+)phase_", str(n), re.I)
            if m:
                out.append(int(m.group(1)))
        return sorted(set(out))

    def _resolve_latent_phases_for_jsonl_entry(
        self, entry: Dict[str, Any], base: str
    ) -> Optional[List[int]]:
        """Phases to expand into multiple training rows (or None = single latent_subpath row).

        Priority:
        1. JSONL field ``latent_phases`` (list of ints), if present and not null.
        2. Global yaml ``latent_phases`` (same as before).
        3. If ``latent_phases_auto_discover`` and global phases unset: scan wan2.2 on storage.
        """
        if "latent_phases" in entry and entry["latent_phases"] is not None:
            lp = entry["latent_phases"]
            if not lp:
                logger.warning("Skipping JSONL episode (latent_phases=[]): %s", base)
                return []
            return sorted({int(x) for x in lp})

        if self.latent_phases_set:
            return sorted(self.latent_phases_set)

        if self.latent_phases_auto_discover:
            disc = self._discover_latent_phases_wan22(base)
            if disc:
                return disc

        return None

    def _load_episode_list_jsonl(self) -> List[Dict[str, Any]]:
        """Read JSONL and build episode metadata list.

        Each JSONL line should have: episode_bos_path, task, qpos_path.
        Optionally: endpose_path, robot (for filtering; if omitted, ``task`` must
        start with ``"{robot}--"`` when yaml sets ``robot``).
        Optionally: latent_phases (list of int), e.g. ``[0,1,2,6,12,18]`` — which
        ``video_features/wan2.2/<n>phase_5hz/`` to use; overrides global yaml
        ``latent_phases`` for that line only.
        """
        with open(self.jsonl_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        episodes: List[Dict[str, Any]] = []
        for entry in entries:
            if not self._jsonl_entry_matches_robot(entry):
                continue
            task_name = entry.get("task")
            if not task_name:
                continue
            if not self._task_matches(task_name):
                continue

            base = self._normalize_bos(entry["episode_bos_path"].rstrip("/"))
            episode_name = base.rsplit("/", 1)[-1]

            qpos_path = entry.get("qpos_path")
            if qpos_path:
                qpos_path = self._normalize_bos(qpos_path)
            else:
                qpos_path = self._infer_episode_qpos_path(base, episode_name)
            endpose_path = entry.get("endpose_path")
            if endpose_path:
                endpose_path = self._normalize_bos(endpose_path)
            else:
                endpose_path = self._infer_episode_endpose_path(base, episode_name)

            # Always use text_features/UMT5 for language files
            lang_root = f"{base}/text_features/UMT5"
            lang_path = f"{base}/{self.lang_subpath}" if self.lang_subpath else None

            phase_list = self._resolve_latent_phases_for_jsonl_entry(entry, base)
            if phase_list == []:
                continue

            if phase_list:
                # Multi-phase: expand each episode into multiple entries
                # latent_subpath template has "0phase" which gets replaced per phase
                for phase in phase_list:
                    phase_subpath = self.latent_subpath.replace("0phase", f"{phase}phase")
                    episodes.append({
                        "episode_name": f"{episode_name}_p{phase}",
                        "source_episode_name": episode_name,
                        "task_name": task_name,
                        "latent_path": f"{base}/{phase_subpath}",
                        "qpos_path": qpos_path,
                        "endpose_path": endpose_path,
                        "lang_root": lang_root,
                        "lang_path": lang_path,
                        "phase_start_30": phase,
                    })
            else:
                episodes.append({
                    "episode_name": episode_name,
                    "source_episode_name": episode_name,
                    "task_name": task_name,
                    "latent_path": f"{base}/{self.latent_subpath}",
                    "qpos_path": qpos_path,
                    "endpose_path": endpose_path,
                    "lang_root": lang_root,
                    "lang_path": lang_path,
                    "phase_start_30": 0,
                })

        logger.info(f"JSONL: {len(episodes)} episodes from {self.jsonl_path}")
        return episodes

    def _organize_episodes(self, episodes: List[Dict[str, Any]]):
        """Finalize filtered episodes for episode-uniform sampling."""
        self.episode_files = (
            list(episodes[: self.max_episodes])
            if self.max_episodes
            else list(episodes)
        )
        self._rebuild_task_index()

    def _rebuild_task_index(self) -> None:
        self.task_to_episodes = {}
        for ep in self.episode_files:
            self.task_to_episodes.setdefault(self._task_bucket_key(ep), []).append(ep)
        self._refresh_task_weights()

    # ------------------------------------------------------------------
    # DuckDB-based loading
    # ------------------------------------------------------------------

    def _load_episodes_from_duckdb(self):
        """Load episodes from a DuckDB catalog.

        DuckDB tables (``real_world_data`` or ``ario_format_data``) contain:
            episode_bos_path, task, qpos_path, robot, dataset, ...

        Filtering (combined with AND):
        - ``robot``: matches ``robot`` OR ``dataset`` column (many rows have
          robot=NULL but dataset="aloha2", so we check both)
        - ``task_name`` + ``task_mode="single"``: supports SQL LIKE patterns
          (e.g. "fold%" matches fold_t-shirt..., fold_towel..., etc.)
        - ``task_mode="multi"``: loads all matching episodes, grouped by task
        """
        import duckdb

        logger.info(f"Loading episodes from DuckDB: {self.duckdb_path} "
                    f"table={self.duckdb_table}")
        # Copy to /tmp to avoid file-lock issues on shared filesystems (VEPFS/NFS)
        import shutil, tempfile
        tmp_db = os.path.join(tempfile.gettempdir(), f"duckdb_{os.getpid()}.duckdb")
        shutil.copy2(self.duckdb_path, tmp_db)
        con = duckdb.connect(tmp_db, read_only=True)

        # Build WHERE clause
        conditions = []
        params = []

        if self.robot_filter:
            # Match either robot or dataset column (many rows have robot=NULL)
            conditions.append("(robot = ? OR dataset = ?)")
            params.extend([self.robot_filter, self.robot_filter])

        if self.task_mode == "single" and self.task_name_patterns:
            stripped_expr = (
                "CASE WHEN strpos(task, '--') > 0 THEN split_part(task, '--', 2) "
                "ELSE task END"
            )
            or_parts: List[str] = []
            for pat in self.task_name_patterns:
                if "%" in pat:
                    or_parts.append(f"(task LIKE ? OR {stripped_expr} LIKE ?)")
                    params.extend([pat, pat])
                else:
                    or_parts.append(f"(task = ? OR {stripped_expr} = ?)")
                    params.extend([pat, pat])
            conditions.append("(" + " OR ".join(or_parts) + ")")

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = (
            f"SELECT episode_bos_path, task, qpos_path, endpose_path, episode_id "
            f"FROM {self.duckdb_table}{where}"
        )

        rows = con.execute(query, params).fetchall()
        con.close()
        os.remove(tmp_db)
        logger.info(f"DuckDB: {len(rows)} episodes matched")

        # Group rows by task to verify which episodes actually exist on BOS
        task_rows: Dict[str, List] = {}
        for row in rows:
            episode_bos_path, task, qpos_path, endpose_path, episode_id = row
            action_path = endpose_path if "eef" in self.action_mode else qpos_path
            if not episode_bos_path or not task or not action_path:
                continue
            task_rows.setdefault(task, []).append(row)

        # For each task, list actual episodes on BOS and filter
        existing_episodes = set()
        for task, task_row_list in task_rows.items():
            # Extract the task-level BOS directory from the first episode
            sample_path = self._normalize_bos(task_row_list[0][0].rstrip("/"))
            task_bos_dir = sample_path.rsplit("/", 1)[0]  # strip episode_id
            try:
                from megfile import smart_listdir as _megfile_listdir
                bos_episode_names = set(_megfile_listdir(task_bos_dir))
                logger.info(f"  BOS listing for {task}: {len(bos_episode_names)} episodes exist on storage")
            except Exception as e:
                logger.warning(f"  Cannot list BOS dir {task_bos_dir}: {e}, keeping all DuckDB episodes")
                bos_episode_names = None

            for row in task_row_list:
                episode_bos_path = self._normalize_bos(row[0].rstrip("/"))
                episode_name = row[4] or episode_bos_path.rsplit("/", 1)[-1]
                if bos_episode_names is not None and episode_name not in bos_episode_names:
                    continue  # skip episodes not on BOS
                existing_episodes.add((task, episode_name))

        episodes: List[Dict[str, Any]] = []
        for episode_bos_path, task, qpos_path, endpose_path, episode_id in rows:
            action_path = endpose_path if "eef" in self.action_mode else qpos_path
            if not episode_bos_path or not task or not action_path:
                continue
            base = self._normalize_bos(episode_bos_path.rstrip("/"))
            episode_name = episode_id or base.rsplit("/", 1)[-1]
            if (task, episode_name) not in existing_episodes:
                continue

            episodes.append({
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task,
                "latent_path": f"{base}/{self.latent_subpath}",
                "qpos_path": self._normalize_bos(qpos_path),
                "endpose_path": self._normalize_bos(endpose_path) if endpose_path else None,
                "lang_path": f"{base}/{self.lang_subpath}",
                "phase_start_30": 0,
            })

        logger.info(f"DuckDB: {len(episodes)} episodes after BOS existence filter "
                    f"(removed {len(rows) - len(episodes)} missing)")
        self._organize_episodes(episodes)

        self._log_episode_summary("DuckDB")

    # ------------------------------------------------------------------
    # Directory-based loading
    # ------------------------------------------------------------------

    def _latent_path(self, path: str) -> str:
        """Replace vae_latents subdir when latent_dir differs."""
        if self.latent_dir == "vae_latents":
            return path
        return path.replace("/vae_latents/", f"/{self.latent_dir}/")

    def _default_episode_action_filename(self) -> str:
        return "endpose.pt" if "eef" in self.action_mode else "qpos.pt"

    def _default_task_action_dir(self) -> str:
        return "endpose" if "eef" in self.action_mode else "qpos"

    def _sample_action_path(self, ep: Dict[str, Any]) -> Optional[str]:
        return ep.get("endpose_path") if "eef" in self.action_mode else ep.get("qpos_path")

    def _scan_task_folder(self, task_path) -> List[Dict[str, Any]]:
        """Scan a single task folder for episodes.

        Expected structure:
            task_path/{latent_dir}/*.pt
            task_path/{qpos|endpose}/*.pt
            task_path/umt5_wan/*.pt
        """
        task_path_str = str(task_path)
        latent_dir = smart_join(task_path_str, self.latent_dir)
        action_dir = smart_join(task_path_str, self._default_task_action_dir())
        umt5_dir = smart_join(task_path_str, "umt5_wan")

        if not all(smart_isdir(d) for d in [latent_dir, action_dir, umt5_dir]):
            return []

        task_name = Path(task_path_str).name
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

            action_file = smart_join(action_dir, f"{base_name}.pt")
            lang_file = smart_join(umt5_dir, f"{name}.pt")
            if not smart_exists(lang_file):
                lang_file = smart_join(umt5_dir, f"{base_name}.pt")
            if not (smart_exists(action_file) and smart_exists(lang_file)):
                continue

            episodes.append({
                "episode_name": name,
                "source_episode_name": base_name,
                "task_name": task_name,
                "latent_path": latent_file,
                "qpos_path": action_file if "eef" not in self.action_mode else None,
                "endpose_path": action_file if "eef" in self.action_mode else None,
                "lang_path": lang_file,
                "phase_start_30": phase_start_30,
            })
        return episodes

    def _load_episodes_from_dir(self):
        """Scan local dataset directories to build episode list.

        Unlike robotwin, real-world data has no clean/randomized split.
        Structure: dataset_dir/task_name/{latent_dir, qpos, umt5_wan}/
        """
        all_episodes: List[Dict[str, Any]] = []
        for dataset_dir in self.dataset_dirs:
            d = Path(dataset_dir) if not is_remote(str(dataset_dir)) else None
            if d is None:
                continue
            if not d.exists():
                logger.warning(f"dataset_dir does not exist: {dataset_dir}")
                continue

            if self.task_mode == "single":
                for pat in self.task_name_patterns:
                    if "%" in pat or pat.endswith("*"):
                        logger.warning(
                            "Directory mode: skipping wildcard task_name pattern %r "
                            "(no glob for task folders)",
                            pat,
                        )
                        continue
                    task_dir = d / pat
                    if task_dir.exists():
                        all_episodes.extend(self._scan_task_folder(task_dir))
            else:
                for task_dir in sorted(d.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    all_episodes.extend(self._scan_task_folder(task_dir))

        self._finalize_episodes(all_episodes)

    def _load_episodes_from_manifest(self):
        """Load episode list from manifest.json for each dataset_dir."""
        from data.utils.smart_io import smart_load_json
        all_episodes: List[Dict[str, Any]] = []

        for dataset_dir in self.dataset_dirs:
            manifest_url = smart_join(str(dataset_dir), "manifest.json")
            logger.info(f"Loading manifest: {manifest_url}")
            manifest = smart_load_json(manifest_url)
            logger.info(
                f"  Manifest v{manifest.get('version', '?')}: "
                f"{manifest['num_episodes']} episodes"
            )

            for entry in manifest["episodes"]:
                if self.task_mode == "single" and not self._task_matches(
                    entry["task_name"]
                ):
                    continue

                ep: Dict[str, Any] = {
                    "episode_name": entry["episode_name"],
                    "source_episode_name": entry["source_episode_name"],
                    "task_name": entry["task_name"],
                    "phase_start_30": entry.get("phase_start_30", 0),
                    "latent_path": self._latent_path(
                        smart_join(str(dataset_dir), entry["latent_path"])
                    ),
                    "qpos_path": (
                        smart_join(str(dataset_dir), entry["qpos_path"])
                        if entry.get("qpos_path") else None
                    ),
                    "endpose_path": (
                        smart_join(str(dataset_dir), entry["endpose_path"])
                        if entry.get("endpose_path") else None
                    ),
                    "lang_path": smart_join(str(dataset_dir), entry["lang_path"]),
                }

                all_episodes.append(ep)

        self._finalize_episodes(all_episodes)

    def _load_episodes_from_iceberg(self):
        """Load episode list from a PyIceberg catalog."""
        from pyiceberg.catalog import load_catalog
        from pyiceberg.expressions import EqualTo, Or

        db_path = self.catalog_path
        warehouse = str(Path(db_path).parent)
        logger.info(f"Loading PyIceberg catalog: {db_path}")

        catalog = load_catalog(
            "iceberg",
            type="sql",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{Path(warehouse).resolve()}",
        )
        table = catalog.load_table("robotwin.episodes")

        row_filter = None
        iceberg_exact_only: Optional[bool] = None
        if self.task_mode == "single" and self.task_name_patterns:
            all_exact = all(
                "%" not in p and not p.endswith("*") for p in self.task_name_patterns
            )
            iceberg_exact_only = all_exact
            if all_exact:
                if len(self.task_name_patterns) == 1:
                    row_filter = EqualTo(
                        "task_name", self.task_name_patterns[0]
                    )
                else:
                    row_filter = reduce(
                        lambda a, b: Or(a, b),
                        [
                            EqualTo("task_name", p)
                            for p in self.task_name_patterns
                        ],
                    )

        scan = table.scan(row_filter=row_filter) if row_filter else table.scan()
        df = scan.to_pandas()
        if (
            self.task_mode == "single"
            and self.task_name_patterns
            and iceberg_exact_only is False
        ):
            df = df[
                df["task_name"]
                .astype(str)
                .apply(self._matches_any_task_pattern)
            ]
        logger.info(f"Catalog: {len(df)} rows matched")

        multi_dir = len(self.dataset_dirs) > 1
        has_root_col = multi_dir and "dataset_root" in df.columns
        default_root = str(self.dataset_dirs[0])

        all_episodes: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            root = (row.get("dataset_root") or default_root) if has_root_col else default_root
            ep: Dict[str, Any] = {
                "episode_name": row["episode_name"],
                "source_episode_name": row["source_episode_name"],
                "task_name": row["task_name"],
                "phase_start_30": row["phase_start_30"],
                "latent_path": self._latent_path(smart_join(root, row["latent_path"])),
                "qpos_path": smart_join(root, row["qpos_path"]) if row.get("qpos_path") else None,
                "endpose_path": smart_join(root, row["endpose_path"]) if row.get("endpose_path") else None,
                "lang_path": smart_join(root, row["lang_path"]),
            }

            all_episodes.append(ep)

        self._finalize_episodes(all_episodes)

    def _finalize_episodes(self, all_episodes: List[Dict[str, Any]]):
        """Finalize episode_files / task_to_episodes after directory loading."""
        self._organize_episodes(all_episodes)

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
        return ep.get("latent_path") or ep.get("episode_name") or "<unknown>"

    @staticmethod
    def _task_bucket_key(ep: Dict[str, Any]) -> str:
        return ep.get("task_name") or "<unknown-task>"

    def _drop_episode(self, ep: Dict[str, Any], reason: str) -> None:
        episode_id = self._episode_identity(ep)
        before = len(self.episode_files)
        self.episode_files = [
            cur for cur in self.episode_files if self._episode_identity(cur) != episode_id
        ]
        removed = len(self.episode_files) != before

        if not removed:
            return

        self._language_candidates_cache.pop(episode_id, None)
        self._episode_failure_counts.pop(episode_id, None)
        self._rebuild_task_index()
        logger.warning(
            "Dropping episode %s | task=%s latent=%s: %s",
            ep.get("episode_name", "?"),
            ep.get("task_name"),
            ep.get("latent_path"),
            reason,
        )

    def _record_episode_failure(self, ep: Dict[str, Any], reason: str) -> None:
        episode_id = self._episode_identity(ep)
        failures = self._episode_failure_counts.get(episode_id, 0) + 1
        self._episode_failure_counts[episode_id] = failures
        threshold = max(1, self.latent_max_attempts)
        if failures >= threshold:
            self._drop_episode(ep, f"{reason} | consecutive_failures={failures}")

    def _clear_episode_failure(self, ep: Dict[str, Any]) -> None:
        self._episode_failure_counts.pop(self._episode_identity(ep), None)

    # ------------------------------------------------------------------
    # Action / state alignment helpers
    # ------------------------------------------------------------------

    def _latent_to_action_range(
        self, latent_idx: int, frame_stride: int, phase_start_30: int = 0
    ) -> Tuple[int, int]:
        """Map a latent index to an action index range (in upsampled space).

        When action_upsample_ratio == 1, indices are 30Hz.
        When action_upsample_ratio == 4, indices are 120Hz (after interpolation).

        - l_0 covers ``action_upsample_ratio`` steps at phase start.
        - l_k (k >= 1) covers 4 * frame_stride * action_upsample_ratio steps.
        """
        R = self.action_upsample_ratio
        if latent_idx == 0:
            return phase_start_30 * R, phase_start_30 * R + R
        apl = 4 * frame_stride * R
        start = (phase_start_30 + 1) * R + (latent_idx - 1) * apl
        return start, start + apl

    @staticmethod
    def _upsample_qpos(qpos: torch.Tensor, ratio: int) -> torch.Tensor:
        """Linearly interpolate qpos by ``ratio``.

        [T, D] → [(T-1)*ratio + 1, D].
        Between every two adjacent frames, ``ratio - 1`` new frames are inserted.
        """
        if ratio == 1 or qpos.shape[0] <= 1:
            return qpos
        T = qpos.shape[0]
        new_len = (T - 1) * ratio + 1
        # F.interpolate expects [N, C, L]
        upsampled = F.interpolate(
            qpos.T.unsqueeze(0),  # [1, D, T]
            size=new_len,
            mode="linear",
            align_corners=True,
        )
        return upsampled.squeeze(0).T  # [new_len, D]

    def _maybe_convert_eef_order(self, qpos: torch.Tensor) -> torch.Tensor:
        if qpos.shape[-1] == 16 and self.eef_quat_order == "wxyz":
            return eef_quat_wxyz_to_xyzw(qpos)
        return qpos

    def _maybe_output_wxyz(self, qpos: torch.Tensor) -> torch.Tensor:
        if self.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"} and qpos.shape[-1] == 16:
            return eef_quat_xyzw_to_wxyz(qpos)
        return qpos

    def _load_actions_and_states(
        self,
        action_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        use_state: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Load actions and states aligned to the cropped latent window."""
        qpos = smart_load_pt(action_path, self.cache_dir, map_location="cpu")
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos)
        qpos = qpos.float()

        if "eef" in self.action_mode and qpos.shape[-1] == 14:
            qpos = eef_rpy_to_quat(qpos)
            if not hasattr(self, "_logged_rpy_eef_conversion"):
                logger.info(
                    "Converted 14D RPY EEF to 16D quaternion EEF for action_mode=%s path=%s",
                    self.action_mode,
                    action_path,
                )
                self._logged_rpy_eef_conversion = True
        qpos = self._maybe_convert_eef_order(qpos)

        if not hasattr(self, '_qpos_shape_logged'):
            logger.info(f"[action] raw shape={list(qpos.shape)} path={action_path} "
                        f"frame_stride={frame_stride} num_latents={num_latents} latent_start={latent_start}")
            self._qpos_shape_logged = True

        # Upsample if needed (e.g. 30Hz → 120Hz)
        if self.action_mode == "rel_eef":
            # EEF: Slerp for rotations, linear for xyz/grip, output always 20D
            qpos = upsample_eef(qpos, self.action_upsample_ratio)
        elif self.action_mode == "abs_eef":
            qpos = upsample_eef(qpos, self.action_upsample_ratio)
        elif self.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"}:
            qpos = upsample_abs_eef_16d(qpos, self.action_upsample_ratio)
            qpos = self._maybe_output_wxyz(qpos)
        else:
            qpos = self._upsample_qpos(qpos, self.action_upsample_ratio)
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
                pad = chunk[-1:].expand(apl - chunk.shape[0], -1)
                chunk = torch.cat([chunk, pad], dim=0)
            else:
                chunk = chunk[:apl]
            action_chunks.append(chunk)

            if use_state:
                state_idx = max(min(a_start, n - 1) - 1, 0)
                state_list.append(qpos[state_idx])

        actions = torch.cat(action_chunks, dim=0)
        states = torch.stack(state_list, dim=0) if use_state else None
        return states, actions

    def _resolve_language_candidates(self, ep: Dict[str, Any]) -> List[str]:
        episode_key = self._episode_identity(ep)
        cached = self._language_candidates_cache.get(episode_key)
        if cached is not None:
            return cached

        candidates = []
        lang_root = ep.get("lang_root")
        lang_path = ep.get("lang_path")

        if lang_root:
            candidates.extend(
                smart_join(lang_root, file_name)
                for file_name in DEFAULT_UMT5_CANDIDATE_FILES
            )
        elif lang_path:
            candidates.append(lang_path)

        # Debug: print first candidate for first episode
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            logger.warning(f"DEBUG: ep keys={list(ep.keys())}, lang_root={lang_root}, lang_path={lang_path}")
            if candidates:
                logger.warning(f"DEBUG: first candidate={candidates[0]}")

        self._language_candidates_cache[episode_key] = candidates
        return candidates

    def _disable_language_for_episode(self, ep: Dict[str, Any]) -> None:
        self._language_candidates_cache[self._episode_identity(ep)] = []

    def _load_language_embedding(self, lang_path: str) -> torch.Tensor:
        embedding_data = smart_load_pt(lang_path, self.cache_dir, map_location="cpu")
        if isinstance(embedding_data, list):
            embeddings = embedding_data[random.randint(0, len(embedding_data) - 1)]
        elif isinstance(embedding_data, torch.Tensor):
            if embedding_data.ndim == 3:
                embeddings = embedding_data[random.randint(0, embedding_data.shape[0] - 1)]
            else:
                embeddings = embedding_data
        else:
            raise ValueError(f"Unexpected type: {type(embedding_data)}")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings

    # ------------------------------------------------------------------
    # Transform hook (override in subclasses for normalization etc.)
    # ------------------------------------------------------------------

    def _transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process a sample before returning from __getitem__.

        Default is identity. Subclasses override to add action normalization.
        """
        return sample

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_episodes * 100

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        W = self.max_latent_window

        for attempt in range(8):
            # Pick episode
            if not self.episode_files:
                continue
            ep = random.choice(self.episode_files)

            try:
                if not hasattr(self, '_logged_count'):
                    self._logged_count = 0
                latent_data = smart_load_pt(ep["latent_path"], self.cache_dir, map_location="cpu")

                # Support multi-phase files (robotwin) vs flat format (real-world)
                if "phases" in latent_data:
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
                else:
                    # Flat format: latents/frame_stride at top level
                    latents = latent_data["latents"]
                    frame_stride = int(latent_data.get("frame_stride", self.video_action_freq_ratio))
                    # Use phase_start_30 from episode entry (set by multi-phase expansion)
                    phase_start_30 = int(ep.get("phase_start_30", 0))

                T_lat = latents.shape[1]

                if self._logged_count < 5:
                    fmt = "phases" if "phases" in latent_data else "flat"
                    logger.debug(f"[getitem] ep={ep['episode_name']} task={ep['task_name']} "
                                 f"fmt={fmt} latents={list(latents.shape)} "
                                 f"frame_stride={frame_stride} phase_start_30={phase_start_30} "
                                 f"keys={list(latent_data.keys())} "
                                 f"qpos={ep['qpos_path']}")
                    self._logged_count += 1

                # Latent-space random crop or pad
                if T_lat > W:
                    start = random.randint(0, T_lat - W)
                    latents_crop = latents[:, start : start + W]
                    valid_len = W
                elif T_lat < W:
                    start = 0
                    pad_size = W - T_lat
                    latents_crop = F.pad(latents, (0, 0, 0, 0, 0, pad_size))
                    valid_len = T_lat
                else:
                    start = 0
                    latents_crop = latents
                    valid_len = W

                actual_latents = min(T_lat - start, W)

                # Load aligned actions and states
                use_state = self.use_state
                action_path = self._sample_action_path(ep)
                if not action_path:
                    raise ValueError(
                        f"Missing {'endpose_path' if 'eef' in self.action_mode else 'qpos_path'} for episode"
                    )
                states, actions = self._load_actions_and_states(
                    action_path, start, actual_latents, frame_stride, phase_start_30, use_state
                )

                # Pad actions/states to full window size
                apl = self.actions_per_latent
                target_action_len = W * apl

                if self._logged_count <= 5:
                    logger.debug(f"[getitem] actions_raw={list(actions.shape)} states={list(states.shape) if states is not None else None} "
                                 f"apl={apl} target={target_action_len} actual_latents={actual_latents} start={start}")

                if actions.shape[0] < target_action_len:
                    pad_len = target_action_len - actions.shape[0]
                    actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)

                if states is not None and states.shape[0] < W:
                    pad_len = W - states.shape[0]
                    states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)

                # Load language embedding from candidates
                candidates = self._resolve_language_candidates(ep)
                if candidates:
                    candidates = random.sample(candidates, k=len(candidates))
                language_embedding = None
                last_error = None
                for candidate in candidates:
                    try:
                        language_embedding = self._load_language_embedding(candidate)
                        break
                    except Exception as e:
                        last_error = (candidate, str(e))
                        continue

                if language_embedding is None:
                    if last_error:
                        logger.warning(f"Failed to load language for {ep.get('episode_name')}, last tried: {last_error[0]}, error: {last_error[1]}")
                    self._disable_language_for_episode(ep)
                    self._record_episode_failure(ep, "no valid language file")
                    continue

                out: Dict[str, Any] = {
                    "latents": latents_crop,
                    "actions": actions,
                    "language_embedding": language_embedding,
                    "valid_len": valid_len,
                    "latent_start": int(start),
                }
                if states is not None:
                    out["states"] = states

                self._clear_episode_failure(ep)
                return self._transform_sample(out)

            except Exception as e:
                self._record_episode_failure(ep, f"{type(e).__name__}: {e}")
                logger.warning(f"Sample error ({ep.get('episode_name', '?')}): {e}")
                continue

        return None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _log_episode_summary(self, source: str = ""):
        """Log per-task episode counts and spot-check one episode per task."""
        prefix = f"{source}: " if source else ""
        if self.task_to_episodes:
            # Multi-task mode
            logger.info(f"{prefix}{self.total_episodes} episodes across "
                        f"{len(self.task_to_episodes)} tasks:")
            for task in sorted(self.task_to_episodes.keys()):
                eps = self.task_to_episodes[task]
                # Spot-check: verify a random episode's latent path exists
                sample_ep = random.choice(eps)
                try:
                    ok = smart_exists(sample_ep["latent_path"])
                except Exception:
                    ok = False
                status = "" if ok else "  [WARNING: latent not found!]"
                logger.info(f"  {task}: {len(eps)} episodes{status}")
                if not ok:
                    logger.warning(f"    sample path: {sample_ep['latent_path']}")
        elif self.episode_files:
            # Single-task mode
            logger.info(f"{prefix}{self.total_episodes} episodes (single-task)")
        else:
            logger.warning(f"{prefix}No episodes loaded!")

    @staticmethod
    def _normalize_bos(path: str) -> str:
        """Normalize bos:// and bos:/ to s3:// for megfile compatibility."""
        if path.startswith("bos://"):
            return "s3://" + path[len("bos://"):]
        if path.startswith("bos:/"):
            return "s3://" + path[len("bos:/"):]
        return path
