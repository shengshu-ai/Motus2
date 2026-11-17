# RobotWin Dataset for sliding-window AR training with pre-computed VAE latents.
# Loads pre-computed latents from vae_latents/ and crops in latent space.

import os
import json
import hashlib
import random
import logging
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
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


class _SkipSample(Exception):
    """Raised by sample assembly when a candidate is unusable (too short, etc.)."""


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

    Training samples return the complete latent trajectory. This worktree is
    dedicated to full-trajectory/varlen training, so there is no fixed-window
    random crop path in this dataset.

    Deterministic length index (``length_index=True``):
        By default ``__getitem__(idx)`` ignores ``idx`` and draws a random
        (task, episode, phase). That random access has no per-index length, so
        length bucketing cannot group similar-length trajectories. When
        ``length_index`` is enabled the dataset scans every latent file once
        (cached to disk, broadcast across ranks) and builds a flat, deterministic
        ``(episode, phase) -> T_lat`` index. ``__len__`` then equals the number of
        (episode, phase) pairs, ``__getitem__(idx)`` loads that specific pair, and
        ``get_sample_lengths()`` exposes per-index ``T_lat`` so
        ``LengthBucketBatchSampler`` becomes effective. Indexed mode enumerates
        each (episode, phase) once per epoch (uniform over pairs); this differs
        from the weighted random path when ``task_sampling_weights`` is set, so it
        is opt-in and normally auto-enabled only for the train split when
        ``training.length_bucketing.enabled`` is true.
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
        # Hybrid-memory prefix-prefill: extra leading warm-up latent frames that
        # condition (anchor/gist/recent) but are excluded from the loss. The
        # dataset returns fixed windows of (max_latent_window + prefix_frames)
        # frames so batches stay rectangular; training masks the loss on them.
        prefix_frames: int = 0,
        video_action_freq_ratio: int = 6,
        action_mode: str = "joint",
        use_state: bool = True,
        # Standard params
        max_episodes: Optional[int] = None,
        val: bool = False,
        # Deterministic, episode-level train/validation partition. A value of
        # 0 keeps the legacy behavior (no partition); otherwise every task keeps
        # the same validation episodes on every rank and across restarts.
        validation_fraction: float = 0.0,
        validation_seed: int = 0,
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
        #   "/path/to/robotwin_iceberg/pyiceberg_catalog.db" → PyIceberg catalog (internal style)
        catalog_path: Optional[str] = None,
        # VAE latent directory name inside each task folder.
        # "vae_latents"       → single-phase files (default, legacy)
        # "vae_latents_24phase" → 24 phases per file
        latent_dir: str = "vae_latents",
        # Precomputed UMT5 directory and candidate-selection policy. RMBench
        # full-episode training uses umt5_wan_global + single so a phase-local
        # annotation can never be sampled as the episode instruction.
        language_embedding_dir: str = "umt5_wan",
        language_embedding_policy: str = "random",
        # Which phase indices to sample from multi-phase files.
        # null / None → sample uniformly from all phases.
        # [0, 6, 12, 18] → only sample from phases at these indices.
        latent_phases: Optional[List[int]] = None,
        # "episode" enumerates each source episode once per epoch and samples
        # one phase within it, matching the successful vanilla recipe.
        # "random" preserves the legacy weighted random-access behavior.
        sequence_mode: str = "random",
        sequence_max_latents: Optional[int] = None,
        sequence_overflow_policy: str = "error",
        sequence_phase_policy: str = "random",
        # Optional per-task sampling weights for multi-task mode.
        # Example: {"hanging_mug": 4.0, "open_microwave": 3.0}
        task_sampling_weights: Optional[Dict[str, float]] = None,
        # Value prediction target construction.
        value_target: str = "progress",
        progress_c: Optional[float] = None,
        treat_as_irrelevant: bool = False,
        latent_budget: Optional[int] = None,
        over_budget_policy: str = "allow_single",
        # Deterministic per-index (episode, phase) -> T_lat length index. Required
        # for length bucketing; disabled by default to preserve random sampling.
        length_index: bool = False,
        # Optional explicit cache path for the length index JSON. When None a path
        # is derived from cache_dir / MOTUS2_CACHE_DIR / /tmp and a config signature.
        length_index_cache: Optional[str] = None,
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
        self.prefix_frames = max(0, int(prefix_frames))
        self.sequence_mode = str(sequence_mode).strip().lower()
        if self.sequence_mode not in {"random", "episode"}:
            raise ValueError(
                "sequence_mode must be 'random' or 'episode', "
                f"got {sequence_mode!r}"
            )
        self.sequence_max_latents = (
            int(sequence_max_latents) if sequence_max_latents is not None else None
        )
        if self.sequence_max_latents is not None and self.sequence_max_latents <= 0:
            raise ValueError("sequence_max_latents must be positive or null")
        self.sequence_overflow_policy = str(sequence_overflow_policy).strip().lower()
        if self.sequence_overflow_policy not in {"prefix", "skip", "error"}:
            raise ValueError(
                "sequence_overflow_policy must be one of {'prefix', 'skip', 'error'}"
            )
        self.sequence_phase_policy = str(sequence_phase_policy).strip().lower()
        if self.sequence_phase_policy not in {"first", "random"}:
            raise ValueError(
                "sequence_phase_policy must be 'first' or 'random', "
                f"got {sequence_phase_policy!r}"
            )
        self.video_action_freq_ratio = video_action_freq_ratio
        self.action_mode = action_mode
        self.use_state = use_state
        if self.action_mode not in {"joint", "rel_joint", "rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self.max_episodes = max_episodes
        self.value_target = value_target
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.latent_budget = int(latent_budget) if latent_budget is not None else None
        if self.latent_budget is not None and self.latent_budget <= 0:
            self.latent_budget = None
        self.over_budget_policy = str(over_budget_policy)
        if self.over_budget_policy not in {"allow_single", "random_window"}:
            raise ValueError(
                "RobotWinTaskDataset supports over_budget_policy='allow_single' "
                f"or 'random_window', got {self.over_budget_policy!r}"
            )
        self.bootstrap_start_window = bool(kwargs.pop("bootstrap_start_window", False))
        self.bootstrap_chunk_size = int(kwargs.pop("bootstrap_chunk_size", 2))
        if self.sequence_mode == "episode" and self.prefix_frames != 0:
            raise ValueError(
                "sequence_mode='episode' requires memory.prefix_frames=0"
            )
        if self.sequence_mode == "episode" and self.bootstrap_start_window:
            raise ValueError(
                "sequence_mode='episode' requires bootstrap_start_window=false"
            )
        if self.sequence_mode == "episode" and self.bootstrap_chunk_size < 1:
            raise ValueError(
                "episode sampling requires fixed_chunk_size >= 1, got "
                f"{self.bootstrap_chunk_size}"
            )
        if (
            self.sequence_mode == "episode"
            and self.sequence_max_latents is not None
            and self.sequence_max_latents < 1 + self.bootstrap_chunk_size
        ):
            raise ValueError(
                "sequence_max_latents must allow one observation plus one "
                f"fixed_chunk_size={self.bootstrap_chunk_size} prediction "
                f"block, got {self.sequence_max_latents}"
            )
        self.val = val
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be in [0, 1), "
                f"got {self.validation_fraction}"
            )
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.cache_dir = cache_dir
        self.catalog_path = catalog_path
        self.latent_dir = latent_dir
        self.language_embedding_dir = str(language_embedding_dir).strip()
        if not self.language_embedding_dir:
            raise ValueError("language_embedding_dir must be non-empty")
        self.language_embedding_policy = str(language_embedding_policy).strip().lower()
        if self.language_embedding_policy not in {"random", "single"}:
            raise ValueError(
                "language_embedding_policy must be 'random' or 'single', "
                f"got {language_embedding_policy!r}"
            )
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
        if self.sequence_mode == "episode" and self.task_sampling_weights:
            raise ValueError(
                "sequence_mode='episode' is episode-uniform and does not support "
                "task_sampling_weights"
            )
        self.total_episodes = 0
        self._load_episodes_distributed()
        if self.task_mode == "single":
            self.sequence_episodes = list(self.episode_files)
        else:
            self.sequence_episodes = [
                ep
                for task in sorted(self.task_to_episodes)
                for ep in sorted(
                    self.task_to_episodes[task],
                    key=lambda item: (
                        str(item.get("source_episode_name", "")),
                        str(item.get("latent_path", "")),
                    ),
                )
            ]

        # Deterministic length index (built lazily, only when requested).
        self.length_index_enabled = bool(length_index)
        if self.sequence_mode == "episode" and self.length_index_enabled:
            raise ValueError(
                "sequence_mode='episode' and length_index=true encode different "
                "sample units; disable length_index for vanilla-parity sampling"
            )
        self.length_index_cache = length_index_cache
        self._index_entries: List[Dict[str, Any]] = []
        self._index_active = False
        # Episode-unit lengths, built on first request instead of in __init__:
        # only length bucketing asks for them, so a run without bucketing never
        # pays for the scan.
        self._episode_lengths: Optional[List[int]] = None
        if self.length_index_enabled:
            self._build_length_index()

        logger.info(
            f"RobotWin latent dataset: mode={data_mode}, task_mode={task_mode}, "
            f"max_latent_window={max_latent_window}, prefix_frames={self.prefix_frames}, "
            f"sequence_mode={self.sequence_mode}, "
            f"sequence_max_latents={self.sequence_max_latents}, "
            f"sequence_overflow_policy={self.sequence_overflow_policy}, "
            f"sequence_phase_policy={self.sequence_phase_policy}, "
            f"language_embedding_dir={self.language_embedding_dir}, "
            f"language_embedding_policy={self.language_embedding_policy}, "
            f"actions_per_latent={self.actions_per_latent}, "
            f"latent_budget={self.latent_budget}, over_budget_policy={self.over_budget_policy}, "
            f"split={'val' if self.val else 'train'}, "
            f"validation_fraction={self.validation_fraction:g}, "
            f"episodes={self.total_episodes}, "
            f"length_index={'on(%d)' % len(self._index_entries) if self._index_active else 'off'}"
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
            smart_join(task_path, self.language_embedding_dir),
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
        is_local_scan = (
            not self.catalog_path
            and not any(is_remote(str(d)) for d in self.dataset_dirs)
        )
        if is_local_scan:
            # Local VEPFS/NFS directory scans are cheap and avoid fragile NCCL
            # object broadcasts during distributed startup.
            self._load_episodes()
            return

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

    def _episode_validation_score(self, ep: Dict[str, Any]) -> int:
        """Return a stable per-episode score used for train/val partitioning."""
        identity = "\0".join(
            [
                str(self.validation_seed),
                str(ep.get("task_name", "")),
                str(ep.get("source_episode_name", ep.get("episode_name", ""))),
                str(ep.get("latent_path", "")),
            ]
        )
        return int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )

    def _validation_window_start(
        self,
        ep: Dict[str, Any],
        *,
        phase_start_30: int,
        total_latents: int,
        window_latents: int,
    ) -> int:
        """Choose a stable over-budget crop for repeatable validation metrics."""
        max_start = int(total_latents) - int(window_latents)
        if max_start <= 0:
            return 0
        identity = "\0".join(
            [
                str(self.validation_seed),
                str(ep.get("task_name", "")),
                str(ep.get("latent_path", "")),
                str(phase_start_30),
                str(total_latents),
                str(window_latents),
            ]
        )
        score = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        return score % (max_start + 1)

    def _select_validation_partition(
        self, episodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select one side of a deterministic episode-level train/val split."""
        if self.validation_fraction <= 0.0 or not episodes:
            return episodes

        # Keep both sides non-empty whenever a group has at least two episodes.
        n_val = int(round(len(episodes) * self.validation_fraction))
        n_val = min(max(n_val, 1), max(len(episodes) - 1, 0))
        if n_val == 0:
            return [] if self.val else episodes

        ranked = sorted(
            episodes,
            key=lambda ep: (
                self._episode_validation_score(ep),
                str(ep.get("latent_path", "")),
            ),
        )
        validation_ids = {id(ep) for ep in ranked[:n_val]}
        if self.val:
            return [ep for ep in episodes if id(ep) in validation_ids]
        return [ep for ep in episodes if id(ep) not in validation_ids]

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
            limited_episodes = (
                all_episodes[: self.max_episodes] if self.max_episodes else all_episodes
            )
            self.episode_files = self._select_validation_partition(limited_episodes)
            self.total_episodes = len(self.episode_files)
        else:
            self.task_to_episodes = {
                task: selected
                for task, episodes in self.task_to_episodes.items()
                if (selected := self._select_validation_partition(episodes))
            }
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
        """Load episode list from a PyIceberg catalog (internal style).

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

    def _check_frame_stride(self, frame_stride: int, ep=None) -> None:
        """Reject latents whose baked time base differs from the configured one.

        ``actions_per_latent`` comes from ``common.video_action_freq_ratio`` while
        action ranges are derived from the ``frame_stride`` stored in each latent
        file. When the two disagree the sample is built on two different time
        bases and nothing downstream raises, so refuse here.
        """
        if int(frame_stride) == int(self.video_action_freq_ratio):
            return
        where = ""
        if isinstance(ep, dict):
            name = ep.get("episode_name") or ep.get("latent_path")
            if name:
                where = f" (episode {name})"
        raise ValueError(
            f"latent frame_stride={int(frame_stride)} does not match "
            f"common.video_action_freq_ratio={int(self.video_action_freq_ratio)}"
            f"{where}. dataset.latent_dir={self.latent_dir!r} holds latents on a "
            "different time base; move both keys together (ratio 6 -> "
            "'vae_latents', ratio r -> 'vae_latents_r<r>')."
        )

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

    def _slice_and_pad_action_chunk(
        self,
        sequence: torch.Tensor,
        action_start: int,
        action_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one fixed action block plus a mask for non-padding tokens."""
        n = int(sequence.shape[0])
        apl = self.actions_per_latent
        read_start = max(0, min(int(action_start), n))
        read_end = max(read_start, min(int(action_end), n))
        chunk = sequence[read_start:read_end][:apl]
        real_count = int(chunk.shape[0])

        valid_mask = torch.zeros(apl, dtype=torch.bool)
        valid_mask[:real_count] = True
        if real_count < apl:
            pad_value = chunk[-1:] if real_count > 0 else sequence[-1:]
            chunk = torch.cat(
                [chunk, pad_value.expand(apl - real_count, *pad_value.shape[1:])],
                dim=0,
            )
        return chunk, valid_mask

    def _load_actions_and_states(
        self,
        qpos_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        use_state: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Load actions and states aligned to the cropped latent window.

        Actions for each latent are flattened into a single sequence
        ``[num_latents * actions_per_latent, action_dim]``.
        States are the qpos immediately before each latent action range
        ``[num_latents, state_dim]``.
        """
        qpos = smart_load_pt(qpos_path, self.cache_dir, map_location="cpu")
        qpos = qpos if isinstance(qpos, torch.Tensor) else torch.as_tensor(qpos)
        if qpos.ndim < 2 or int(qpos.shape[0]) == 0:
            raise ValueError(f"Empty qpos trajectory: {qpos_path}")
        qpos = qpos.float()
        n = int(qpos.shape[0])

        action_chunks = []
        action_valid_masks = []
        state_list = []
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            chunk, chunk_valid_mask = self._slice_and_pad_action_chunk(
                qpos,
                a_start,
                a_end,
            )
            action_chunks.append(chunk)
            action_valid_masks.append(chunk_valid_mask)

            if use_state or self.action_mode == "rel_joint":
                state_idx = max(min(int(a_start) - 1, n - 1), 0)
                state_list.append(qpos[state_idx].float())

        actions = torch.cat(action_chunks, dim=0)
        action_valid_mask = torch.cat(action_valid_masks, dim=0)
        states = torch.stack(state_list, dim=0) if state_list else None
        return states, actions, action_valid_mask

    def _load_endpose_actions_and_states(
        self,
        endpose_path: str,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load absolute EEF actions and aligned states for the cropped window.

        Endpose files may store quaternions in legacy scalar-first order; convert to xyzw once.
        - rel_eef keeps 16D xyzw tensors
        - rel_eef_6d converts once to xyzw, then to 20D rot6d
        - abs_eef_rpy converts to 14D [xyz, rpy, grip] x 2
        """
        endpose = smart_load_pt(endpose_path, self.cache_dir, map_location="cpu")
        endpose = endpose if isinstance(endpose, torch.Tensor) else torch.as_tensor(endpose)
        if endpose.ndim < 2 or int(endpose.shape[0]) == 0:
            raise ValueError(f"Empty endpose trajectory: {endpose_path}")
        endpose = endpose.float()
        if endpose.shape[-1] == 14:
            endpose = eef_rpy_to_quat(endpose)
        if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"}:
            endpose = eef_quat_wxyz_to_xyzw(endpose)
        n = len(endpose)
        action_chunks = []
        action_valid_masks = []
        state_list = []
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            chunk, chunk_valid_mask = self._slice_and_pad_action_chunk(
                endpose,
                a_start,
                a_end,
            )
            action_chunks.append(chunk)
            action_valid_masks.append(chunk_valid_mask)
            if self.use_state or self.action_mode in {"rel_eef", "rel_eef_6d"}:
                state_idx = max(min(int(a_start) - 1, n - 1), 0)
                state_list.append(endpose[state_idx].float())

        actions = torch.cat(action_chunks, dim=0)
        action_valid_mask = torch.cat(action_valid_masks, dim=0)
        states = torch.stack(state_list, dim=0) if state_list else None
        if self.action_mode == "rel_eef_6d":
            actions = abs_eef_16d_to_abs_eef_20d_torch(actions)
            if states is not None:
                states = abs_eef_16d_to_abs_eef_20d_torch(states)
        elif self.action_mode == "abs_eef_rpy":
            actions = eef_quat_to_rpy(actions)
            if states is not None:
                states = eef_quat_to_rpy(states)
        return states, actions, action_valid_mask

    def _load_language_embedding(self, lang_path: str) -> torch.Tensor:
        embedding_data = smart_load_pt(lang_path, self.cache_dir, map_location="cpu")
        if isinstance(embedding_data, list):
            if not embedding_data:
                raise ValueError(f"Empty language embedding list: {lang_path}")
            if self.language_embedding_policy == "single" and len(embedding_data) != 1:
                raise ValueError(
                    "language_embedding_policy='single' requires exactly one "
                    f"embedding in {lang_path}, got {len(embedding_data)}"
                )
            index = (
                0
                if self.language_embedding_policy == "single"
                else random.randint(0, len(embedding_data) - 1)
            )
            embeddings = embedding_data[index]
        elif isinstance(embedding_data, torch.Tensor):
            if embedding_data.ndim == 3:
                candidate_count = int(embedding_data.shape[0])
                if candidate_count == 0:
                    raise ValueError(f"Empty language embedding tensor: {lang_path}")
                if self.language_embedding_policy == "single" and candidate_count != 1:
                    raise ValueError(
                        "language_embedding_policy='single' requires exactly one "
                        f"embedding in {lang_path}, got {candidate_count}"
                    )
                index = (
                    0
                    if self.language_embedding_policy == "single"
                    else random.randint(0, candidate_count - 1)
                )
                embeddings = embedding_data[index]
            else:
                embeddings = embedding_data
        else:
            raise ValueError(f"Unexpected type: {type(embedding_data)}")
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(
                f"Language embedding candidate in {lang_path} is not a tensor: "
                f"{type(embeddings)}"
            )
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
        if self._index_active:
            return len(self._index_entries)
        if self.sequence_mode == "episode":
            return len(self.sequence_episodes)
        return self.total_episodes * 10

    def get_sample_lengths(self) -> Optional[List[int]]:
        """Deterministic per-index T_lat, or None when this mode has no length source.

        Consumed by ``data.length_bucketing.resolve_sample_lengths`` so the bucket
        sampler can group similar-length trajectories.

        Two sample units can answer this. The ``(episode, phase)`` index answers it
        from ``_index_entries``. ``sequence_mode='episode'`` answers it per episode
        (see ``_build_episode_lengths``); reporting a length does not change what a
        sample is or how the index space is enumerated, so episode-uniform sampling
        is untouched. ``sequence_mode='random'`` has no per-index length by
        construction and still returns None.
        """
        if self._index_active:
            return [int(e["num_latents"]) for e in self._index_entries]
        if self.sequence_mode == "episode":
            return self._build_episode_lengths()
        return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if self._index_active:
            return self._getitem_indexed(idx)
        if self.sequence_mode == "episode":
            return self._getitem_episode(idx)
        return self._getitem_random()

    # --- random access (default) ---------------------------------------

    def _sample_random_episode(self) -> Optional[Dict[str, Any]]:
        if self.task_mode == "single":
            if not self.episode_files:
                return None
            return random.choice(self.episode_files)
        if not self.task_weights:
            return None
        task = random.choices(
            list(self.task_weights.keys()),
            weights=list(self.task_weights.values()),
            k=1,
        )[0]
        eps = self.task_to_episodes.get(task, [])
        if not eps:
            return None
        return random.choice(eps)

    def _select_phase_random(self, latent_data, ep):
        """Pick a phase the same way the legacy random path did (filter then choice)."""
        if "phases" in latent_data:
            phases = latent_data["phases"]
            if self.latent_phases_set is not None:
                phases = [p for i, p in enumerate(phases) if i in self.latent_phases_set]
            if not phases:
                return None
            if getattr(self, "val", False):
                phase_data = phases[self._episode_validation_score(ep) % len(phases)]
            else:
                phase_data = random.choice(phases)
            latents = phase_data["latents"]  # [C, T_lat, H, W]
            frame_stride = int(latent_data.get("frame_stride", phase_data.get("frame_stride", 6)))
            phase_start_30 = int(phase_data.get("phase_start_30", 0))
        else:
            latents = latent_data["latents"]  # [C, T_lat, H, W]
            frame_stride = int(latent_data["frame_stride"])
            phase_start_30 = int(latent_data.get("phase_start_30", ep.get("phase_start_30", 0)))
        return latents, frame_stride, phase_start_30

    def _getitem_random(self) -> Optional[Dict[str, Any]]:
        for _ in range(8):
            ep = self._sample_random_episode()
            if ep is None:
                continue
            try:
                latent_data = smart_load_pt(ep["latent_path"], self.cache_dir, map_location="cpu")
                sel = self._select_phase_random(latent_data, ep)
                if sel is None:
                    continue
                latents, frame_stride, phase_start_30 = sel
                return self._assemble_sample(ep, latents, frame_stride, phase_start_30)
            except _SkipSample:
                continue
            except Exception as e:
                logger.warning(f"Sample error ({ep.get('episode_name', '?')}): {e}")
                continue
        return None

    # --- episode-uniform access (successful vanilla parity) ------------

    def _select_phase_episode(self, latent_data, ep):
        if "phases" not in latent_data:
            return self._select_phase_random(latent_data, ep)
        phases = [
            phase
            for phase_idx, phase in enumerate(latent_data["phases"])
            if self.latent_phases_set is None or phase_idx in self.latent_phases_set
        ]
        if not phases:
            return None
        phase_data = (
            phases[0]
            if self.sequence_phase_policy == "first"
            else random.choice(phases)
        )
        latents = phase_data["latents"]
        frame_stride = int(
            latent_data.get("frame_stride", phase_data.get("frame_stride", 6))
        )
        phase_start_30 = int(phase_data.get("phase_start_30", 0))
        return latents, frame_stride, phase_start_30

    def _getitem_episode(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.sequence_episodes:
            raise RuntimeError("Episode sequence dataset is empty")
        episode_idx = int(idx)
        if episode_idx < 0:
            episode_idx += len(self.sequence_episodes)
        if episode_idx < 0 or episode_idx >= len(self.sequence_episodes):
            raise IndexError(
                f"Episode index {idx} outside [0, {len(self.sequence_episodes)})"
            )
        ep = self.sequence_episodes[episode_idx]
        try:
            latent_data = smart_load_pt(
                ep["latent_path"],
                self.cache_dir,
                map_location="cpu",
            )
            selected = self._select_phase_episode(latent_data, ep)
            if selected is None:
                raise ValueError(
                    "No latent phase matched latent_phases="
                    f"{sorted(self.latent_phases_set) if self.latent_phases_set else None}"
                )
            latents, frame_stride, phase_start_30 = selected
            return self._assemble_sample(
                ep,
                latents,
                frame_stride,
                phase_start_30,
            )
        except Exception as exc:
            raise RuntimeError(
                "Episode sample failed: "
                f"idx={idx}, latent_path={ep.get('latent_path', '<missing>')}: {exc}"
            ) from exc

    # --- deterministic indexed access -----------------------------------

    def _select_phase_indexed(self, latent_data, entry):
        """Load the exact (episode, phase) recorded in the length index."""
        ep = entry["ep"]
        phase_idx = int(entry["phase_idx"])
        if phase_idx >= 0:
            phases = latent_data.get("phases")
            if not phases or phase_idx >= len(phases):
                return None
            phase_data = phases[phase_idx]
            latents = phase_data["latents"]
            frame_stride = int(latent_data.get("frame_stride", phase_data.get("frame_stride", 6)))
            phase_start_30 = int(phase_data.get("phase_start_30", 0))
        else:
            latents = latent_data["latents"]
            frame_stride = int(latent_data["frame_stride"])
            phase_start_30 = int(latent_data.get("phase_start_30", ep.get("phase_start_30", 0)))
        return latents, frame_stride, phase_start_30

    def _getitem_indexed(self, idx: int) -> Optional[Dict[str, Any]]:
        n = len(self._index_entries)
        if n == 0:
            return None
        # Probe the requested index first, then nearby ones to stay robust to a
        # transiently unreadable file without collapsing the deterministic mapping.
        for off in range(8):
            entry = self._index_entries[(int(idx) + off) % n]
            ep = entry["ep"]
            try:
                latent_data = smart_load_pt(ep["latent_path"], self.cache_dir, map_location="cpu")
                sel = self._select_phase_indexed(latent_data, entry)
                if sel is None:
                    continue
                latents, frame_stride, phase_start_30 = sel
                return self._assemble_sample(ep, latents, frame_stride, phase_start_30)
            except _SkipSample:
                continue
            except Exception as e:
                logger.warning(f"Indexed sample error ({ep.get('episode_name', '?')}): {e}")
                continue
        return None

    # --- shared sample assembly -----------------------------------------

    def _assemble_sample(self, ep, latents, frame_stride, phase_start_30) -> Dict[str, Any]:
        """Build the training sample dict from a resolved (episode, phase)."""
        self._check_frame_stride(frame_stride, ep)
        T_lat = latents.shape[1]
        # Full-trajectory sample: start is always the phase's first latent.
        if T_lat < 2:
            # Need at least one observed latent and one predicted latent.
            raise _SkipSample(f"T_lat={T_lat} < 2")
        start = 0
        sequence_mode = getattr(self, "sequence_mode", "random")
        if sequence_mode == "episode":
            obs_latents = 1
            candidate_W = T_lat
            if (
                self.sequence_max_latents is not None
                and T_lat > self.sequence_max_latents
            ):
                if self.sequence_overflow_policy in {"error", "skip"}:
                    raise _SkipSample(
                        f"T_lat={T_lat} exceeds sequence_max_latents="
                        f"{self.sequence_max_latents} with policy="
                        f"{self.sequence_overflow_policy}"
                    )
                candidate_W = self.sequence_max_latents
            W = obs_latents + (
                (candidate_W - obs_latents) // self.bootstrap_chunk_size
            ) * self.bootstrap_chunk_size
            if W < obs_latents + self.bootstrap_chunk_size:
                raise _SkipSample(
                    f"T_lat={T_lat} has no complete fixed_chunk_size="
                    f"{self.bootstrap_chunk_size} prediction block"
                )
            latents_crop = latents[:, :W].contiguous()
        elif (
            self.over_budget_policy == "random_window"
            and self.latent_budget is not None
            and T_lat > self.latent_budget
        ):
            W = self.latent_budget
            if getattr(self, "val", False):
                start = self._validation_window_start(
                    ep,
                    phase_start_30=phase_start_30,
                    total_latents=T_lat,
                    window_latents=W,
                )
            else:
                start = random.randint(0, T_lat - W)
            latents_crop = latents[:, start:start + W].clone().contiguous()
        else:
            W = T_lat
            latents_crop = latents
        valid_len = W

        actual_latents = W

        # Load aligned actions and states
        use_state = self.use_state
        if self.action_mode in {"rel_eef", "rel_eef_6d", "abs_eef_rpy"} and "endpose_path" in ep:
            loaded_actions = self._load_endpose_actions_and_states(
                ep["endpose_path"], start, actual_latents, frame_stride, phase_start_30
            )
        else:
            loaded_actions = self._load_actions_and_states(
                ep["qpos_path"], start, actual_latents, frame_stride, phase_start_30, use_state
            )
        if len(loaded_actions) == 2:
            states, actions = loaded_actions
            source_action_valid_mask = torch.ones(
                actions.shape[0],
                dtype=torch.bool,
            )
        else:
            states, actions, source_action_valid_mask = loaded_actions

        # Pad actions/states to full window size
        apl = self.actions_per_latent
        target_action_len = W * apl
        if actions.shape[0] < target_action_len:
            pad_len = target_action_len - actions.shape[0]
            actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
            source_action_valid_mask = torch.cat(
                [
                    source_action_valid_mask,
                    torch.zeros(pad_len, dtype=torch.bool),
                ],
                dim=0,
            )
        if actions.shape[0] != target_action_len:
            raise ValueError(
                f"Action length mismatch: got {actions.shape[0]}, "
                f"expected {target_action_len}"
            )
        if source_action_valid_mask.shape[0] != target_action_len:
            raise ValueError(
                "Action-valid mask length mismatch: got "
                f"{source_action_valid_mask.shape[0]}, "
                f"expected {target_action_len}"
            )
        if sequence_mode == "episode":
            prediction_action_mask = source_action_valid_mask[apl:]
            if not bool(prediction_action_mask.any()):
                raise ValueError(
                    "Episode has no real action token after the observation latent"
                )

        if states is not None and states.shape[0] < W:
            pad_len = W - states.shape[0]
            states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)

        language_embedding = self._load_language_embedding(ep["lang_path"])

        values = self._build_value_targets(start, W, T_lat, ep["task_name"])
        (
            latents_crop,
            actions,
            states,
            values,
            valid_len,
            extra_action_sequences,
        ) = maybe_bootstrap_start_window_sample(
            latents_crop,
            actions,
            valid_len,
            start,
            actions_per_latent=apl,
            states=states,
            values=values,
            chunk_size=self.bootstrap_chunk_size,
            enabled=self.bootstrap_start_window,
            extra_action_sequences={
                "source_action_valid_mask": source_action_valid_mask,
            },
        )
        source_action_valid_mask = extra_action_sequences[
            "source_action_valid_mask"
        ].bool()
        video_valid_mask = torch.arange(W, dtype=torch.long) < int(valid_len)
        action_valid_mask = (
            video_valid_mask.repeat_interleave(apl)
            & source_action_valid_mask.bool()
        )

        # Legacy field kept for dataloader API compatibility (global mode uses L0 via
        # hybrid eviction on the first clean frame, not a separate prepended token).
        anchor_latents = latents[:, :1].clone().contiguous()  # [C, 1, H_lat, W_lat]

        out: Dict[str, Any] = {
            "latents": latents_crop,  # [C, W, H_lat, W_lat]
            "anchor_latents": anchor_latents,  # [C, 1, H_lat, W_lat] global L0
            "actions": actions,  # [W * actions_per_latent, action_dim]
            "language_embedding": language_embedding,
            "valid_len": valid_len,
            "video_valid_mask": video_valid_mask,
            "action_valid_mask": action_valid_mask,
            # Absolute latent start index in this phase trajectory.
            # Used for optional global-absolute RoPE training alignment.
            "latent_start": int(start),
            "is_task_irrelevant": int(self.treat_as_irrelevant),
            "is_success": int(not self.treat_as_irrelevant),
        }
        if sequence_mode == "episode":
            out["episode_num_latents"] = int(T_lat)
            out["sequence_num_latents"] = int(W)
            out["sequence_chunk_trimmed_latents"] = int(T_lat - W)
        if states is not None:
            out["states"] = states  # [W, state_dim]
        if values is not None:
            out["values"] = values  # [W]
        return out

    # ------------------------------------------------------------------
    # Deterministic length index (episode, phase) -> T_lat
    # ------------------------------------------------------------------

    def _iter_all_episodes(self):
        if self.task_mode == "single":
            yield from self.episode_files
        else:
            for eps in self.task_to_episodes.values():
                yield from eps

    @staticmethod
    def _phase_num_latents(phase: Dict[str, Any]) -> Optional[int]:
        if "num_latents" in phase:
            try:
                return int(phase["num_latents"])
            except (TypeError, ValueError):
                pass
        lat = phase.get("latents")
        if lat is not None and hasattr(lat, "shape") and len(lat.shape) >= 2:
            return int(lat.shape[1])
        return None

    @staticmethod
    def _legacy_num_latents(data: Dict[str, Any]) -> Optional[int]:
        if "num_latents" in data:
            try:
                return int(data["num_latents"])
            except (TypeError, ValueError):
                pass
        lat = data.get("latents")
        if lat is not None and hasattr(lat, "shape") and len(lat.shape) >= 2:
            return int(lat.shape[1])
        return None

    def _probe_latent_file(self, path: str):
        """Load a latent file as cheaply as possible to read its T_lat metadata.

        Local files are memory-mapped (``mmap=True``) so only tensor headers are
        touched, not bulk latent data. Remote files fall back to ``smart_load_pt``.
        """
        if not is_remote(str(path)):
            try:
                return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            except TypeError:
                # Older torch without mmap/weights_only kwargs.
                return torch.load(path, map_location="cpu")
            except Exception:
                return torch.load(path, map_location="cpu")
        return smart_load_pt(path, self.cache_dir, map_location="cpu")

    def _scan_length_index(self) -> List[Dict[str, Any]]:
        """Scan every episode once and emit (episode, phase, T_lat) index entries."""
        entries: List[Dict[str, Any]] = []
        n_files = 0
        for ep in self._iter_all_episodes():
            n_files += 1
            try:
                data = self._probe_latent_file(ep["latent_path"])
            except Exception as e:
                logger.warning("length index: cannot probe %s: %s", ep.get("latent_path", "?"), e)
                continue
            if isinstance(data, dict) and "phases" in data:
                phases = data["phases"]
                for orig_idx, phase in enumerate(phases):
                    if self.latent_phases_set is not None and orig_idx not in self.latent_phases_set:
                        continue
                    t = self._phase_num_latents(phase)
                    if t is None or t < 2:
                        continue
                    entries.append({"ep": ep, "phase_idx": int(orig_idx), "num_latents": int(t)})
            else:
                t = self._legacy_num_latents(data) if isinstance(data, dict) else None
                if t is not None and t >= 2:
                    entries.append({"ep": ep, "phase_idx": -1, "num_latents": int(t)})
            if n_files % 500 == 0:
                logger.info("length index scan: %d files probed, %d entries", n_files, len(entries))
        logger.info("length index scan complete: %d files, %d entries", n_files, len(entries))
        return entries

    def _corpus_identity_payload(self) -> Dict[str, Any]:
        """Everything that decides which episodes this split contains."""
        return {
            "dirs": sorted(str(d) for d in self.dataset_dirs),
            "data_mode": self.data_mode,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "latent_dir": self.latent_dir,
            "language_embedding_dir": self.language_embedding_dir,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set else None,
            "action_mode": self.action_mode,
            "randomized_limit": self.randomized_limit_per_task,
            "max_episodes": self.max_episodes,
            "validation_fraction": self.validation_fraction,
            "validation_seed": self.validation_seed,
            "split": "val" if self.val else "train",
            "total_episodes": self.total_episodes,
        }

    def _length_index_signature(self) -> str:
        raw = json.dumps(self._corpus_identity_payload(), sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _episode_length_signature(self) -> str:
        """Signature for the episode-unit cache.

        Carries the crop knobs on top of corpus identity, because the cached number
        is the packed length after cropping, not the raw ``T_lat``. Keeping this
        distinct from ``_length_index_signature`` is what stops an episode-unit
        cache and an ``(episode, phase)`` cache from being read for each other.
        """
        payload = self._corpus_identity_payload()
        payload.update({
            "unit": "episode",
            "sequence_max_latents": self.sequence_max_latents,
            "sequence_overflow_policy": self.sequence_overflow_policy,
            "chunk_size": self.bootstrap_chunk_size,
        })
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _length_index_cache_path(self) -> str:
        if self.length_index_cache:
            return str(self.length_index_cache)
        base = self.cache_dir or os.environ.get("MOTUS2_CACHE_DIR") or "/tmp/motus2_cache"
        return os.path.join(base, "length_index", f"robotwin_{self._length_index_signature()}.json")

    def _episode_length_cache_path(self) -> str:
        base = self.cache_dir or os.environ.get("MOTUS2_CACHE_DIR") or "/tmp/motus2_cache"
        return os.path.join(
            base, "length_index", f"robotwin_episode_{self._episode_length_signature()}.json"
        )

    def _load_length_index_cache(self, path: str) -> Optional[List[Dict[str, Any]]]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and obj.get("signature") == self._length_index_signature():
                    logger.info("length index: loaded cache %s", path)
                    return obj.get("entries")
        except Exception as e:
            logger.warning("length index: failed to read cache %s: %s", path, e)
        return None

    def _save_length_index_cache(self, path: str, entries: List[Dict[str, Any]]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"signature": self._length_index_signature(), "entries": entries}, f)
            os.replace(tmp, path)
            logger.info("length index: wrote cache %s (%d entries)", path, len(entries))
        except Exception as e:
            logger.warning("length index: failed to write cache %s: %s", path, e)

    def _build_length_index(self) -> None:
        """Build (or load) the deterministic length index, broadcasting across ranks."""
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        entries: Optional[List[Dict[str, Any]]] = None
        if rank == 0:
            path = self._length_index_cache_path()
            entries = self._load_length_index_cache(path)
            if entries is None:
                logger.info("length index: scanning latent files (one-time, cached)...")
                entries = self._scan_length_index()
                self._save_length_index_cache(path, entries)
        if is_dist:
            obj = [entries]
            dist.broadcast_object_list(obj, src=0)
            entries = obj[0]
        self._index_entries = entries or []
        self._index_active = len(self._index_entries) > 0
        if not self._index_active:
            logger.warning(
                "length_index requested but no entries were built; "
                "falling back to random sampling (bucketing will be inert)."
            )

    # --- episode-unit lengths (for length bucketing under sequence_mode=episode) ---

    def _episode_packed_latents(self, t_lat: int) -> Optional[int]:
        """Packed latent count for one episode-mode sample, or None if it is skipped.

        Mirrors the ``sequence_mode='episode'`` branch of ``_assemble_sample``. The
        bucket sampler has to pack against the length the forward pass actually sees,
        which is the chunk-aligned crop rather than the raw ``T_lat``.
        ``tests/test_episode_length_index.py`` asserts the two stay in agreement.
        """
        if t_lat < 2:
            return None
        candidate = int(t_lat)
        if self.sequence_max_latents is not None and candidate > self.sequence_max_latents:
            if self.sequence_overflow_policy in {"error", "skip"}:
                return None
            candidate = self.sequence_max_latents
        obs_latents = 1
        packed = obs_latents + (
            (candidate - obs_latents) // self.bootstrap_chunk_size
        ) * self.bootstrap_chunk_size
        if packed < obs_latents + self.bootstrap_chunk_size:
            return None
        return int(packed)

    def _scan_episode_lengths(self) -> List[int]:
        """One packed length per ``sequence_episodes`` index.

        Takes the longest eligible phase. The phases of one episode are the same
        trajectory at different 30Hz start offsets and differ by at most one latent,
        while ``_getitem_episode`` picks one at random, so the longest phase is the
        bound the packing budget must hold for that index.
        """
        lengths: List[int] = []
        unusable = 0
        total = len(self.sequence_episodes)
        for ep in self.sequence_episodes:
            try:
                data = self._probe_latent_file(ep["latent_path"])
            except Exception as e:
                logger.warning(
                    "episode lengths: cannot probe %s: %s", ep.get("latent_path", "?"), e
                )
                data = None
            t_lat: Optional[int] = None
            if isinstance(data, dict) and "phases" in data:
                usable = [
                    int(value)
                    for value in (
                        self._phase_num_latents(phase)
                        for orig_idx, phase in enumerate(data["phases"])
                        if self.latent_phases_set is None
                        or orig_idx in self.latent_phases_set
                    )
                    if value is not None
                ]
                t_lat = max(usable) if usable else None
            elif isinstance(data, dict):
                t_lat = self._legacy_num_latents(data)
            packed = self._episode_packed_latents(t_lat) if t_lat is not None else None
            if packed is None:
                # __getitem__ would raise on this index. Report the widest length so
                # the sampler isolates it rather than hiding it inside a full batch,
                # and let the loader surface the real error.
                unusable += 1
                packed = self.sequence_max_latents or self.max_latent_window
            lengths.append(int(packed))
            if len(lengths) % 500 == 0:
                logger.info("episode lengths: %d/%d probed", len(lengths), total)
        if unusable:
            logger.warning(
                "episode lengths: %d of %d episodes cannot produce a sample "
                "(unreadable, T_lat below one chunk, or above sequence_max_latents "
                "with policy=%s); reported at the maximum length",
                unusable,
                total,
                self.sequence_overflow_policy,
            )
        logger.info("episode lengths: %d entries, min=%d max=%d sum=%d",
                    len(lengths), min(lengths), max(lengths), sum(lengths))
        return lengths

    def _load_episode_length_cache(self, path: str) -> Optional[List[int]]:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                if (
                    isinstance(obj, dict)
                    and obj.get("signature") == self._episode_length_signature()
                    and isinstance(obj.get("lengths"), list)
                    and len(obj["lengths"]) == len(self.sequence_episodes)
                ):
                    logger.info("episode lengths: loaded cache %s", path)
                    return [int(v) for v in obj["lengths"]]
        except Exception as e:
            logger.warning("episode lengths: failed to read cache %s: %s", path, e)
        return None

    def _save_episode_length_cache(self, path: str, lengths: List[int]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {"signature": self._episode_length_signature(), "lengths": lengths}, f
                )
            os.replace(tmp, path)
            logger.info("episode lengths: wrote cache %s (%d entries)", path, len(lengths))
        except Exception as e:
            logger.warning("episode lengths: failed to write cache %s: %s", path, e)

    def _build_episode_lengths(self) -> Optional[List[int]]:
        """Scan (or load) per-episode packed lengths, broadcasting from rank 0.

        Called from ``get_sample_lengths()``, which only length bucketing calls, so
        a run without bucketing never touches the latent files for this.
        """
        if self._episode_lengths is not None:
            return self._episode_lengths
        if not self.sequence_episodes:
            return None
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        lengths: Optional[List[int]] = None
        if rank == 0:
            path = self._episode_length_cache_path()
            lengths = self._load_episode_length_cache(path)
            if lengths is None:
                logger.info(
                    "episode lengths: probing %d latent files (one-time, cached)...",
                    len(self.sequence_episodes),
                )
                lengths = self._scan_episode_lengths()
                self._save_episode_length_cache(path, lengths)
        if is_dist:
            obj = [lengths]
            dist.broadcast_object_list(obj, src=0)
            lengths = obj[0]
        self._episode_lengths = lengths
        return self._episode_lengths


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
