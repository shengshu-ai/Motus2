# RobotWin Dataset for sliding-window AR training with pre-computed VAE latents.
# Loads pre-computed latents from vae_latents/ and crops in latent space.

import os
import random
import logging
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from utils.common import maybe_bootstrap_start_window_sample
from utils.relative_eef import eef_quat_wxyz_to_xyzw, abs_eef_16d_to_abs_eef_20d_torch
from data.utils.rotation_utils import eef_quat_to_rpy, eef_rpy_to_quat
from data.utils.smart_io import (
    is_remote, smart_join, stem,
    smart_isdir, smart_listdir_dirs, smart_glob_pt, smart_exists,
    smart_load_pt, smart_load_json,
)

try:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.expressions import And, EqualTo, In, NotNull
except ImportError:
    load_catalog = None  # pyiceberg is optional; only needed for iceberg catalog loading
import torch.distributed as dist

logger = logging.getLogger(__name__)


class RobotWinTaskDataset(data.Dataset):
    """RobotWin dataset with pre-computed VAE latents and latent-space random cropping.

    Pre-computed latents are stored per episode in:
        dataset_dir/{clean,randomized}/task_name/vae_latents/{name}.pt

    Preferred .pt format (multi-phase in one file):
        phases: list[dict], each with:
            latents:          [C, T_lat, H_lat, W_lat]
            num_latents:      int
            video_frame_ids:  list of original 30Hz frame indices
            phase_start_30:   int
            frame_stride:     int (= video_action_freq_ratio)

    Backward-compatible format (legacy single phase):
        latents / num_latents / video_frame_ids / frame_stride (+ optional phase_start_30)

    Training samples are produced by randomly cropping a window of
    ``max_latent_window`` latent frames (pad if episode is shorter).
    """

    def __init__(
        self,
        dataset_dir: str = "/path/to/robotwin2/",
        data_mode: str = "clean",
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        randomized_limit_per_task: Optional[int] = None,
        # Sliding-window params
        max_latent_window: int = 10,
        video_action_freq_ratio: int = 6,
        action_mode: str = "joint",
        use_state: bool = True,
        # Standard params
        max_episodes: Optional[int] = None,
        val: bool = False,
        # EEF modes only
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
        # Remote data (S3/BOS) support
        # If dataset_dir is a remote path (s3:// or bos://), files are downloaded
        # to cache_dir on first access using bcecmd (preferred) or megfile.
        cache_dir: Optional[str] = None,
        # Episode listing source (controls how _load_episodes() finds episodes):
        #   None       → auto: remote → manifest.json, local → directory scan
        #   "manifest" → force manifest.json (dataset_dir/manifest.json on BOS)
        #   "/path/to/robotwin_iceberg/pyiceberg_catalog.db" → PyIceberg catalog
        catalog_path: Optional[str] = None,
        # VAE latent directory name inside each task folder.
        # "vae_latents"       → single-phase files (default, legacy)
        # "vae_latents_24phase" → 24 phases per file
        latent_dir: str = "vae_latents",
        # Which phase indices to sample from multi-phase files.
        # null / None → sample uniformly from all phases.
        # [0, 6, 12, 18] → only sample from phases at these indices.
        latent_phases: Optional[List[int]] = None,
        # Optional per-task sampling weights for multi-task mode.
        # Example: {"hanging_mug": 4.0, "open_microwave": 3.0}
        task_sampling_weights: Optional[Dict[str, float]] = None,
        # Value prediction target construction.
        value_target: str = "progress",
        progress_c: Optional[float] = None,
        treat_as_irrelevant: bool = False,
        # Compat (ignored)
        **kwargs,
    ):
        # Support single path (str/Path) or list-like paths (e.g. OmegaConf ListConfig).
        # Use str for remote paths to avoid Path() collapsing s3:// → s3:/.
        def _to_dir(d):
            s = str(d)
            return s if is_remote(s) else Path(s)

        if isinstance(dataset_dir, (str, Path)):
            self.dataset_dirs = [_to_dir(dataset_dir)]
        else:
            self.dataset_dirs = [_to_dir(d) for d in dataset_dir]
        # Keep legacy attribute for compatibility with older code paths.
        self.dataset_dir = self.dataset_dirs[0]
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.task_name = task_name
        self.randomized_limit_per_task = randomized_limit_per_task
        self.max_latent_window = max_latent_window
        self.video_action_freq_ratio = video_action_freq_ratio
        self.action_mode = action_mode
        self.use_state = use_state
        if self.action_mode not in {"joint", "rel_joint", "rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self.max_episodes = max_episodes
        self.value_target = value_target
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.bootstrap_start_window = bool(kwargs.pop("bootstrap_start_window", False))
        self.bootstrap_chunk_size = int(kwargs.pop("bootstrap_chunk_size", 2))
        self.val = val
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.cache_dir = cache_dir
        self.catalog_path = catalog_path
        self.latent_dir = latent_dir
        self.latent_phases_set = set(latent_phases) if latent_phases else None
        if is_remote(self.dataset_dir) and not cache_dir:
            logger.warning(
                "dataset_dir is a remote path but cache_dir is not set. "
                "Files will be cached to /tmp/motus2_cache."
            )

        # Each latent (after l_0) covers 4 video frames, each video frame
        # spans ``video_action_freq_ratio`` 30Hz action steps.
        self.actions_per_latent = 4 * video_action_freq_ratio  # typically 24

        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name")

        self.episode_files: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.task_weights: Dict[str, float] = {}
        self.task_sampling_weights = (
            dict(task_sampling_weights) if task_sampling_weights is not None else {}
        )
        self.total_episodes = 0
        self._load_episodes_distributed()

        logger.info(
            f"RobotWin latent dataset: mode={data_mode}, task_mode={task_mode}, "
            f"max_latent_window={max_latent_window}, "
            f"actions_per_latent={self.actions_per_latent}, "
            f"episodes={self.total_episodes}"
        )

    # ------------------------------------------------------------------
    # Episode scanning
    # ------------------------------------------------------------------

    def _latent_path(self, path: str) -> str:
        """Replace the latent subdirectory name in a stored path.

        Paths from catalog/manifest contain "vae_latents/" as the subdir.
        When latent_dir differs (e.g. "vae_latents_24phase"), substitute it.
        """
        if self.latent_dir == "vae_latents":
            return path
        return path.replace("/vae_latents/", f"/{self.latent_dir}/")

    def _select_existing_dir(self, *candidates: str) -> Optional[str]:
        existing = []
        for path in candidates:
            if path and smart_isdir(path):
                existing.append(path)
        for path in existing:
            try:
                if smart_glob_pt(path):
                    return path
            except Exception:
                continue
        for path in existing:
            return path
        return None

    def _resolve_task_support_dirs(self, task_path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        latent_dir = self._select_existing_dir(
            smart_join(task_path, self.latent_dir),
        )
        qpos_dir = self._select_existing_dir(
            smart_join(task_path, "qpos"),
        )
        umt5_dir = self._select_existing_dir(
            smart_join(task_path, "umt5_wan"),
        )
        endpose_dir = self._select_existing_dir(
            smart_join(task_path, "endpose"),
        )
        return latent_dir, qpos_dir, umt5_dir, endpose_dir

    def _scan_task_folder(self, task_path: str) -> List[Dict[str, Any]]:
        latent_dir, qpos_dir, umt5_dir, endpose_dir = self._resolve_task_support_dirs(task_path)
        if not all(d is not None for d in [latent_dir, qpos_dir, umt5_dir]):
            return []

        task_name = Path(task_path).name
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
            lang_file = smart_join(umt5_dir, f"{name}.pt")
            if not smart_exists(lang_file):
                # language embedding is stored per original episode, not per phase
                lang_file = smart_join(umt5_dir, f"{base_name}.pt")
            if not (smart_exists(qpos_file) and smart_exists(lang_file)):
                continue

            ep: Dict[str, Any] = {
                "episode_name": name,
                "source_episode_name": base_name,
                "task_name": task_name,
                "latent_path": latent_file,
                "qpos_path": qpos_file,
                "lang_path": lang_file,
                "phase_start_30": phase_start_30,
            }
            if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
                if endpose_dir is None:
                    continue
                endpose_file = smart_join(endpose_dir, f"{base_name}.pt")
                if smart_exists(endpose_file):
                    ep["endpose_path"] = endpose_file
                else:
                    continue
            episodes.append(ep)
        return episodes

    def _load_episodes_distributed(self):
        """Load episodes on rank 0 only, then broadcast to all other ranks.

        Avoids all ranks concurrently hitting the SQLite catalog / BOS manifest.
        Uses broadcast_object_list which works with NCCL (no CPU tensor restriction).
        """

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if rank == 0:
            self._load_episodes()

        if is_dist:
            obj = [(
                self.episode_files,
                self.task_to_episodes,
                self.task_weights,
                self.total_episodes,
            )]
            dist.broadcast_object_list(obj, src=0)
            if rank != 0:
                (
                    self.episode_files,
                    self.task_to_episodes,
                    self.task_weights,
                    self.total_episodes,
                ) = obj[0]

    def _load_episodes(self):
        if self.catalog_path and self.catalog_path != "manifest":
            self._load_episodes_from_iceberg()
        elif self.catalog_path == "manifest" or any(is_remote(str(d)) for d in self.dataset_dirs):
            self._load_episodes_from_manifest()
        else:
            self._load_episodes_local()

    def _apply_randomized_limit_and_finalize(self, all_episodes: List[Dict[str, Any]]):
        """Apply randomized_limit_per_task and set episode_files / task_to_episodes."""
        if self.randomized_limit_per_task:
            for key, eps in list(self.task_to_episodes.items()):
                if key.endswith("|randomized"):
                    task = key[: -len("|randomized")]
                    eps_limited = sorted(eps, key=lambda e: e["episode_name"])[
                        : self.randomized_limit_per_task
                    ]
                    if self.task_mode == "single":
                        all_episodes.extend(eps_limited)
                    else:
                        self.task_to_episodes.setdefault(task, []).extend(eps_limited)
                    del self.task_to_episodes[key]

        if self.task_mode == "single":
            self.episode_files = (
                all_episodes[: self.max_episodes] if self.max_episodes else all_episodes
            )
            self.total_episodes = len(self.episode_files)
        else:
            num_tasks = len(self.task_to_episodes)
            if num_tasks == 0:
                self.task_weights = {}
            elif self.task_sampling_weights:
                unknown_tasks = sorted(
                    t for t in self.task_sampling_weights.keys() if t not in self.task_to_episodes
                )
                if unknown_tasks:
                    logger.warning(
                        "task_sampling_weights has %d unknown tasks (ignored): %s",
                        len(unknown_tasks),
                        ", ".join(unknown_tasks[:8]) + ("..." if len(unknown_tasks) > 8 else ""),
                    )
                raw_weights: Dict[str, float] = {}
                for task in self.task_to_episodes:
                    weight = float(self.task_sampling_weights.get(task, 1.0))
                    if weight <= 0:
                        logger.warning(
                            "task_sampling_weights[%s]=%s is non-positive; fallback to 1.0",
                            task,
                            weight,
                        )
                        weight = 1.0
                    raw_weights[task] = weight
                total_weight = sum(raw_weights.values())
                if total_weight <= 0:
                    self.task_weights = {t: 1.0 / num_tasks for t in self.task_to_episodes}
                else:
                    self.task_weights = {
                        task: weight / total_weight for task, weight in raw_weights.items()
                    }
                boosted = sorted(
                    ((t, w) for t, w in raw_weights.items() if abs(w - 1.0) > 1e-6),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if boosted:
                    logger.info(
                        "Using custom task sampling weights for %d tasks. Top boosts: %s",
                        len(boosted),
                        ", ".join([f"{t}x{w:g}" for t, w in boosted[:8]]),
                    )
            else:
                self.task_weights = {t: 1.0 / num_tasks for t in self.task_to_episodes}
            self.total_episodes = sum(len(e) for e in self.task_to_episodes.values())

    def _load_episodes_from_iceberg(self):
        """Load episode list from a PyIceberg catalog.

        The catalog is a local NFS .db file created by create_iceberg_catalog.py.
        Table: robotwin.episodes
        Paths are relative to dataset_dir (single dir) or to the per-row
        dataset_root column (multi-dir catalogs).
        """

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

        splits = ["clean", "randomized"] if self.data_mode == "both" else [self.data_mode]
        row_filter = In("split", splits) if len(splits) > 1 else EqualTo("split", splits[0])
        if self.task_mode == "single":
            row_filter = And(row_filter, EqualTo("task_name", self.task_name))
        if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
            row_filter = And(row_filter, NotNull("endpose_path"))

        df = table.scan(row_filter=row_filter).to_pandas()
        logger.info(f"Catalog: {len(df)} rows matched (splits={splits})")

        # dataset_root column is only used when multiple dataset_dirs are configured.
        # With a single dir, always use self.dataset_dirs[0] (the user-configured path,
        # e.g. s3://...) so the catalog's stored local path does not override it.
        multi_dir = len(self.dataset_dirs) > 1
        has_root_col = multi_dir and "dataset_root" in df.columns
        default_root = str(self.dataset_dirs[0])

        all_episodes: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            root = (row.get("dataset_root") or default_root) if has_root_col else default_root
            ep: Dict[str, Any] = {
                "episode_name":        row["episode_name"],
                "source_episode_name": row["source_episode_name"],
                "task_name":           row["task_name"],
                "phase_start_30":      row["phase_start_30"],
                "latent_path": self._latent_path(smart_join(root, row["latent_path"])),
                "qpos_path":   smart_join(root, row["qpos_path"]),
                "lang_path":   smart_join(root, row["lang_path"]),
            }
            if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"} and not row.get("endpose_path"):
                continue
            if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"} and row.get("endpose_path"):
                ep["endpose_path"] = smart_join(root, row["endpose_path"])

            split = row["split"]
            if split == "randomized" and self.randomized_limit_per_task:
                self.task_to_episodes.setdefault(row["task_name"] + "|randomized", []).append(ep)
                continue

            if self.task_mode == "single":
                all_episodes.append(ep)
            else:
                self.task_to_episodes.setdefault(row["task_name"], []).append(ep)

        self._apply_randomized_limit_and_finalize(all_episodes)

    def _load_episodes_from_manifest(self):
        """Load episode list from manifest.json for each dataset_dir."""
        splits = {"clean", "randomized"} if self.data_mode == "both" else {self.data_mode}
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
                if entry.get("split") not in splits:
                    continue
                if self.task_mode == "single" and entry["task_name"] != self.task_name:
                    continue
                if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"} and "endpose_path" not in entry:
                    continue

                ep: Dict[str, Any] = {
                    "episode_name":        entry["episode_name"],
                    "source_episode_name": entry["source_episode_name"],
                    "task_name":           entry["task_name"],
                    "phase_start_30":      entry.get("phase_start_30", 0),
                    "latent_path": self._latent_path(smart_join(str(dataset_dir), entry["latent_path"])),
                    "qpos_path":   smart_join(str(dataset_dir), entry["qpos_path"]),
                    "lang_path":   smart_join(str(dataset_dir), entry["lang_path"]),
                }
                if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
                    ep["endpose_path"] = smart_join(str(dataset_dir), entry["endpose_path"])

                split = entry.get("split", "clean")
                if split == "randomized" and self.randomized_limit_per_task:
                    self.task_to_episodes.setdefault(entry["task_name"] + "|randomized", []).append(ep)
                    continue

                if self.task_mode == "single":
                    all_episodes.append(ep)
                else:
                    self.task_to_episodes.setdefault(entry["task_name"], []).append(ep)

        self._apply_randomized_limit_and_finalize(all_episodes)

    def _load_episodes_local(self):
        """Scan local dataset directory to build episode list."""
        splits = ["clean", "randomized"] if self.data_mode == "both" else [self.data_mode]
        all_episodes: List[Dict[str, Any]] = []

        for dataset_dir in self.dataset_dirs:
            root_task_dirs = [
                task_dir for task_dir in sorted(dataset_dir.iterdir())
                if task_dir.is_dir() and smart_isdir(smart_join(str(task_dir), self.latent_dir))
            ] if isinstance(dataset_dir, Path) and dataset_dir.exists() else []
            if root_task_dirs:
                if self.task_mode == "single":
                    task_dir = dataset_dir / self.task_name
                    if task_dir.exists():
                        all_episodes.extend(self._scan_task_folder(str(task_dir)))
                else:
                    for task_dir in root_task_dirs:
                        eps = self._scan_task_folder(str(task_dir))
                        if eps:
                            self.task_to_episodes.setdefault(task_dir.name, []).extend(eps)
                continue

            for split in splits:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                if self.task_mode == "single":
                    task_dir = split_dir / self.task_name
                    if task_dir.exists():
                        eps = self._scan_task_folder(task_dir)
                        if split == "randomized" and self.randomized_limit_per_task:
                            eps = sorted(eps, key=lambda e: e["episode_name"])[
                                : self.randomized_limit_per_task
                            ]
                        all_episodes.extend(eps)
                else:
                    for task_dir in sorted(split_dir.iterdir()):
                        if not task_dir.is_dir():
                            continue
                        eps = self._scan_task_folder(task_dir)
                        if split == "randomized" and self.randomized_limit_per_task:
                            eps = sorted(eps, key=lambda e: e["episode_name"])[
                                : self.randomized_limit_per_task
                            ]
                        tname = task_dir.name
                        self.task_to_episodes.setdefault(tname, []).extend(eps)

        self._apply_randomized_limit_and_finalize(all_episodes)

    # ------------------------------------------------------------------
    # Action / state alignment helpers
    # ------------------------------------------------------------------

    def _latent_to_action_range(
        self, latent_idx: int, frame_stride: int, phase_start_30: int = 0
    ) -> Tuple[int, int]:
        """Map a latent index to a 30Hz action index range.

        Alignment follows streaming/raw semantics:
        - l_0 covers exactly 1 action step at phase start (observation anchor).
        - l_k (k >= 1) covers 4 video frames = 4 * frame_stride action steps.
        """
        if latent_idx == 0:
            return phase_start_30, phase_start_30 + 1
        apl = 4 * frame_stride
        start = phase_start_30 + 1 + (latent_idx - 1) * apl
        return start, start + apl

    def _load_actions_and_states(
        self,
        qpos_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        use_state: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Load actions and states aligned to the cropped latent window.

        Actions for each latent are flattened into a single sequence
        ``[num_latents * actions_per_latent, action_dim]``.
        States are the qpos at the start of each latent frame
        ``[num_latents, state_dim]``.
        """
        qpos = smart_load_pt(qpos_path, self.cache_dir, map_location="cpu")  # [T_30hz, action_dim]
        n = len(qpos)
        apl = self.actions_per_latent  # 24

        action_chunks = []
        state_list = []
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            a_start = min(a_start, n - 1)
            a_end = min(a_end, n)

            chunk = qpos[a_start:a_end].float()
            if chunk.shape[0] < apl:
                pad = chunk[-1:].expand(apl - chunk.shape[0], -1)
                chunk = torch.cat([chunk, pad], dim=0)
            else:
                chunk = chunk[:apl]
            action_chunks.append(chunk)

            if use_state or self.action_mode == "rel_joint":
                state_idx = max(min(a_start, n - 1) - 1, 0)
                state_list.append(qpos[state_idx].float())

        actions = torch.cat(action_chunks, dim=0)  # [num_latents * apl, dim]
        states = torch.stack(state_list, dim=0) if state_list else None
        return states, actions

    def _load_endpose_actions_and_states(
        self,
        endpose_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load absolute EEF actions and aligned states for the cropped window.

        Endpose files may store quaternions in legacy scalar-first order; convert to xyzw once.
        - rel_eef keeps 16D xyzw tensors
        - rel_eef_6d converts once to xyzw, then to 20D rot6d
        - abs_eef_rpy converts to 14D [xyz, rpy, grip] x 2
        """
        endpose = smart_load_pt(endpose_path, self.cache_dir, map_location="cpu")
        endpose = endpose if isinstance(endpose, torch.Tensor) else torch.as_tensor(endpose)
        endpose = endpose.float()
        if endpose.shape[-1] == 14:
            endpose = eef_rpy_to_quat(endpose)
        if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
            endpose = eef_quat_wxyz_to_xyzw(endpose)
        n = len(endpose)
        apl = self.actions_per_latent

        action_chunks = []
        state_list = []
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            a_start = min(a_start, n - 1)
            a_end = min(a_end, n)
            chunk = endpose[a_start:a_end]
            if chunk.shape[0] < apl:
                chunk = torch.cat([chunk, chunk[-1:].expand(apl - chunk.shape[0], -1)], dim=0)
            else:
                chunk = chunk[:apl]
            action_chunks.append(chunk)
            if self.use_state or self.action_mode in {"rel_eef", "rel_eef_6d"}:
                state_idx = max(min(a_start, n - 1) - 1, 0)
                state_list.append(endpose[state_idx].float())

        actions = torch.cat(action_chunks, dim=0)
        states = torch.stack(state_list, dim=0) if state_list else None
        if self.action_mode == "rel_eef_6d":
            actions = abs_eef_16d_to_abs_eef_20d_torch(actions)
            if states is not None:
                states = abs_eef_16d_to_abs_eef_20d_torch(states)
        elif self.action_mode == "abs_eef_rpy":
            actions = eef_quat_to_rpy(actions)
            if states is not None:
                states = eef_quat_to_rpy(states)
        return states, actions

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

    def _build_progress_values(self, start: int, W: int, T: int) -> torch.Tensor:
        c = self.progress_c
        if c is None or c <= 0 or T <= 1:
            return torch.zeros(W, dtype=torch.float32)
        abs_idx = torch.arange(W, dtype=torch.float32) + start
        s = (abs_idx / (T - 1)).clamp(0, 1)
        values = (c ** (1 - s) - c) / (1 - c)
        values[abs_idx >= T] = 0.0
        return values

    def _build_ttc_values(self, start: int, W: int, T: int) -> torch.Tensor:
        if T <= 1:
            return torch.zeros(W, dtype=torch.float32)
        abs_idx = torch.arange(W, dtype=torch.float32) + start
        remaining = (T - 1) - abs_idx
        values = -remaining / float(T - 1)
        values = values.clamp(min=-1.0, max=0.0)
        values[abs_idx >= T] = -1.0
        return values

    def _build_vlac_values(self, start: int, W: int, T: int, negate: bool = False) -> torch.Tensor:
        if T <= 1:
            return torch.zeros(W, dtype=torch.float32)

        harmonic_T = (1.0 / torch.arange(1, T + 1, dtype=torch.float64)).sum().item()
        per_frame_reward = 1.0 / (T - torch.arange(T, dtype=torch.float64))
        cum_value = per_frame_reward.cumsum(0) / harmonic_T

        abs_idx = torch.arange(W, dtype=torch.long) + start
        values = torch.zeros(W, dtype=torch.float32)
        valid = abs_idx < T
        values[valid] = cum_value[abs_idx[valid]].float()
        values[~valid] = 1.0
        if negate:
            values = -values
        return values

    def _build_irrelevant_values(self, W: int) -> torch.Tensor:
        if self.value_target in {"progress", "vlac"}:
            return torch.zeros(W, dtype=torch.float32)
        return torch.full((W,), -1.0, dtype=torch.float32)

    def _build_value_targets(self, start: int, W: int, T: int, task_name: str) -> Optional[torch.Tensor]:
        if self.value_target == "progress":
            if self.treat_as_irrelevant:
                return self._build_irrelevant_values(W)
            return self._build_progress_values(start, W, T)
        if self.value_target == "ttc":
            if self.treat_as_irrelevant:
                return self._build_irrelevant_values(W)
            return self._build_ttc_values(start, W, T)
        if self.value_target == "vlac":
            return self._build_vlac_values(start, W, T, negate=self.treat_as_irrelevant)
        return None

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_episodes * 10

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        W = self.max_latent_window

        for _ in range(8):
            # Pick episode
            if self.task_mode == "single":
                if not self.episode_files:
                    continue
                ep = random.choice(self.episode_files)
            else:
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
                latent_data = smart_load_pt(ep["latent_path"], self.cache_dir, map_location="cpu")
                # New format: one file contains all phases.
                if "phases" in latent_data:
                    phases = latent_data["phases"]
                    if self.latent_phases_set is not None:
                        phases = [p for i, p in enumerate(phases) if i in self.latent_phases_set]
                    if not phases:
                        continue
                    phase_data = random.choice(phases)
                    latents = phase_data["latents"]  # [C, T_lat, H, W]
                    frame_stride = int(latent_data.get("frame_stride", phase_data.get("frame_stride", 6)))
                    phase_start_30 = int(phase_data.get("phase_start_30", 0))
                else:
                    # Backward compatibility: single-phase file.
                    latents = latent_data["latents"]  # [C, T_lat, H, W]
                    frame_stride = int(latent_data["frame_stride"])
                    phase_start_30 = int(latent_data.get("phase_start_30", ep.get("phase_start_30", 0)))
                T_lat = latents.shape[1]

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
                if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"} and "endpose_path" in ep:
                    states, actions = self._load_endpose_actions_and_states(
                        ep["endpose_path"], start, actual_latents, frame_stride, phase_start_30
                    )
                else:
                    states, actions = self._load_actions_and_states(
                        ep["qpos_path"], start, actual_latents, frame_stride, phase_start_30, use_state
                    )

                # Pad actions/states to full window size
                apl = self.actions_per_latent
                target_action_len = W * apl
                if actions.shape[0] < target_action_len:
                    pad_len = target_action_len - actions.shape[0]
                    actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)

                if states is not None and states.shape[0] < W:
                    pad_len = W - states.shape[0]
                    states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)

                language_embedding = self._load_language_embedding(ep["lang_path"])

                values = self._build_value_targets(start, W, T_lat, ep["task_name"])
                latents_crop, actions, states, values, valid_len = maybe_bootstrap_start_window_sample(
                    latents_crop,
                    actions,
                    valid_len,
                    start,
                    actions_per_latent=apl,
                    states=states,
                    values=values,
                    chunk_size=self.bootstrap_chunk_size,
                    enabled=self.bootstrap_start_window,
                )

                out: Dict[str, Any] = {
                    "latents": latents_crop,  # [C, W, H_lat, W_lat]
                    "actions": actions,  # [W * actions_per_latent, action_dim]
                    "language_embedding": language_embedding,
                    "valid_len": valid_len,
                    # Absolute latent start index in this phase trajectory.
                    # Used for optional global-absolute RoPE training alignment.
                    "latent_start": int(start),
                    SAMPLE_CLASS_KEY: int(
                        SampleClass.TASK_IRRELEVANT
                        if self.treat_as_irrelevant
                        else SampleClass.MAIN
                    ),
                    "is_success": int(not self.treat_as_irrelevant),
                }
                if states is not None:
                    out["states"] = states  # [W, state_dim]
                if values is not None:
                    out["values"] = values  # [W]
                return out

            except Exception as e:
                logger.warning(f"Sample error ({ep.get('episode_name', '?')}): {e}")
                continue

        return None

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load('/path/to/robotwin.yaml')
    d = config.dataset
    c = config.common
    dataset = RobotWinTaskDataset(
        dataset_dir=d.dataset_dir,
        data_mode=d.get("data_mode", "both"),
        task_mode=d.get("task_mode", "multi"),
        task_name=d.get("task_name", None),
        max_latent_window=c.get("max_latent_window", 10),
        video_action_freq_ratio=c.get("video_action_freq_ratio", 6),
        action_mode=config.model.get("action_mode", "joint"),
        max_episodes=d.get("max_episodes", None),
        cache_dir=d.get("cache_dir", None),
        catalog_path=d.get("catalog_path", None),
        latent_dir=d.get("latent_dir", "vae_latents"),
        latent_phases=d.get("latent_phases", None),
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    if sample:
        for k, v in sample.items():
            print(f"  {k}: {v.shape if hasattr(v, 'shape') else v}")
    import IPython; IPython.embed()
