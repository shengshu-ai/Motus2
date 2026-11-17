"""Tianji/Wuji dexterous-hand latent dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import fnmatch
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from data.utils.smart_io import (
    is_remote,
    smart_exists,
    smart_glob_pt,
    smart_join,
    smart_listdir,
    smart_listdir_dirs,
    smart_load_pt,
)
from utils.common import maybe_bootstrap_start_window_sample


logger = logging.getLogger(__name__)

SIDES = ("left", "right")
ACTION_DIM = 58
TACTILE_DIM = 1536


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_remote_path(path: str) -> str:
    if path.startswith("bos://"):
        return "s3://" + path[len("bos://"):]
    if path.startswith("bos:/"):
        return "s3://" + path[len("bos:/"):]
    return path


def _strip_remote_scheme(path: str) -> str:
    if path.startswith("s3://"):
        return path[len("s3://"):].lstrip("/")
    if path.startswith("bos://"):
        return path[len("bos://"):].lstrip("/")
    if path.startswith("bos:/"):
        return path[len("bos:/"):].lstrip("/")
    return path.lstrip("/")


def _mapping_to_plain_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    if isinstance(obj, dict):
        return dict(obj)
    return dict(obj)


def _to_plain_container(obj: Any) -> Any:
    """Convert OmegaConf containers (ListConfig/DictConfig) into plain Python types."""
    if obj is None:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    return obj


def _rate_cfg_value(cfg: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    cfg = _to_plain_container(cfg)
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return None if value is None else int(value)


def _cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    cfg = _to_plain_container(cfg)
    if cfg is None:
        return bool(default)
    if isinstance(cfg, dict):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return bool(value)


# Folder/task names that already store action/tactile at 30Hz (no 60→30 downsample).
_NATIVE_30HZ_NAME_RE = re.compile(r"(?:^|[/_.-])30[_]?hz(?:[/_.-]|$)", re.IGNORECASE)


def _task_name_implies_native_30hz(task_name: Optional[str]) -> bool:
    """True when task path/name explicitly marks native 30Hz data (e.g. 30hz_0716_...)."""
    return bool(_NATIVE_30HZ_NAME_RE.search(str(task_name or "")))


def _resolve_rate_config(
    *,
    video_hz: Optional[int],
    action_cfg: Any,
    tactile_cfg: Any,
    raw_action_hz: Optional[int],
    train_action_hz: Optional[int],
    raw_tactile_hz: Optional[int],
    train_tactile_hz: Optional[int],
    action_upsample_ratio: int,
) -> Tuple[int, int, int, int, int, int, int, int]:
    video_hz = int(video_hz or 30)
    raw_action_hz = _rate_cfg_value(action_cfg, "raw_hz", raw_action_hz)
    train_action_hz = _rate_cfg_value(action_cfg, "train_hz", train_action_hz)
    raw_tactile_hz = _rate_cfg_value(tactile_cfg, "raw_hz", raw_tactile_hz)
    train_tactile_hz = _rate_cfg_value(tactile_cfg, "train_hz", train_tactile_hz)

    if train_action_hz is None:
        train_action_hz = video_hz * int(action_upsample_ratio)
    if raw_action_hz is None:
        raw_action_hz = train_action_hz
    if raw_action_hz % train_action_hz != 0:
        raise ValueError(f"raw_action_hz must be divisible by train_action_hz, got {raw_action_hz}/{train_action_hz}")
    if train_action_hz % video_hz != 0:
        raise ValueError(f"train_action_hz must be divisible by video_hz, got {train_action_hz}/{video_hz}")

    if train_tactile_hz is None:
        train_tactile_hz = train_action_hz
    if raw_tactile_hz is None:
        raw_tactile_hz = train_tactile_hz
    if raw_tactile_hz % train_tactile_hz != 0:
        raise ValueError(f"raw_tactile_hz must be divisible by train_tactile_hz, got {raw_tactile_hz}/{train_tactile_hz}")

    action_downsample = raw_action_hz // train_action_hz
    tactile_downsample = raw_tactile_hz // train_tactile_hz
    derived_upsample = train_action_hz // video_hz
    return video_hz, raw_action_hz, train_action_hz, action_downsample, raw_tactile_hz, train_tactile_hz, tactile_downsample, derived_upsample


def _column_alias_expr_for_sql(preferred: str, available_columns: List[str], aliases: List[str]) -> Optional[str]:
    for col in aliases:
        if col in available_columns:
            return col if col == preferred else f"{col} AS {preferred}"
    return None


def _clickhouse_query_json_each_row(sql: str, clickhouse: Dict[str, Any]) -> List[dict]:
    scheme = "https" if bool(clickhouse.get("secure", False)) else "http"
    host = str(clickhouse["host"])
    port = int(clickhouse.get("port", 8123))
    database = str(clickhouse.get("database", "default"))
    timeout = float(clickhouse.get("timeout_sec", os.environ.get("MOTUS2_CLICKHOUSE_TIMEOUT_SEC", 60)))
    url = f"{scheme}://{host}:{port}/?" + urllib.parse.urlencode({"database": database})
    password = clickhouse.get("password") or os.environ.get("CLICKHOUSE_PASSWORD", "")
    username = str(clickhouse.get("username", "default"))
    if "FORMAT" not in sql.upper():
        sql = f"{sql.rstrip()}\nFORMAT JSONEachRow"

    request = urllib.request.Request(url, data=sql.encode("utf-8"), method="POST")
    if username or password:
        import base64

        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured internal CH endpoint
        payload = response.read().decode("utf-8")
    rows = []
    for line in payload.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _as_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _quat_xyzw_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat.float(), dim=-1)
    x, y, z, w = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)


def _endpose_to_wrist9(endpose: torch.Tensor) -> torch.Tensor:
    xyz_m = endpose[:, :3].float()
    rot6d = _quat_xyzw_to_rot6d(endpose[:, 3:7])
    return torch.cat([xyz_m, rot6d], dim=-1)


class TianjiWujiDataset(data.Dataset):
    """Latent-space Tianji/Wuji dataset.

    The returned action/state layout is:
      [L_qpos20 | L_wrist_abs_xyz_m3_rot6d6 | R_qpos20 | R_wrist_abs_xyz_m3_rot6d6]

    Qpos dims are normalized with `wj_stat.json` (`qpos_normalization`:
    mean_std, min_max, or quantile). Wrist relative xyz is normalized in the
    model via `model.normalization.wrist_xyz` (mean_std | min_max | quantile).
    Wrist dims are absolute in meters/rot6d in the dataset and converted to
    chunk-relative form inside the model.
    """

    def __init__(
        self,
        dataset_dir: str = "/path/to/tianji_wuji_dataset",
        episode_index_source: str = "auto",
        episode_path_prefix: Optional[str] = None,
        clickhouse: Optional[Dict[str, Any]] = None,
        # Latent directory relative to each episode.
        # Single-phase (default/legacy): "video_features/wan2.2/0phase_5hz"
        # Multi-phase parent: "video_features/wan2.2" (+ latent_phases or null=scan)
        latent_root: str = "video_features/wan2.2/0phase_5hz",
        # Which {N}phase_5hz dirs to use:
        #   None + *phase_5hz latent_root → single-phase (backward compatible)
        #   None + parent latent_root     → scan all {N}phase_5hz per episode
        #   [0, 1, 6, ...]                → only those phase indices
        latent_phases: Optional[List[int]] = None,
        latent_resolution: str = "384_640",
        left_latent_name: Optional[str] = None,
        right_latent_name: Optional[str] = None,
        view_latent_names: Optional[List[str]] = None,
        view_merge: str = "stack",
        single_latent_name: Optional[str] = None,
        language_subpath: str = "text_features/UMT5/instruction.pt",
        tjwj_subpath: str = "tjwj.pt",
        action_stats_path: str = "data/tianji_wuji/wj_stat.json",
        qpos_normalization: str = "mean_std",
        tactile_stats_path: str = "data/tianji_wuji/wj_tactile.json",
        tactile_baseline_dir: Optional[str] = None,        # legacy: npy directory
        tactile_baseline_name: Optional[str] = None,        # NEW: name of entry in wj_tactile.json["baselines"]
        tactile_baseline_kind: str = "median",
        tactile_baseline_polarity: str = "positive",
        tactile_baseline_per_source: Optional[List[Dict[str, Any]]] = None,
        task_episode_skip_first: Optional[List[Dict[str, Any]]] = None,  # T22: per-task drop first N episodes
        max_latent_window: int = 10,
        video_hz: int = 30,
        video_action_freq_ratio: int = 6,
        action_upsample_ratio: int = 2,
        action: Optional[Dict[str, Any]] = None,
        tactile: Optional[Dict[str, Any]] = None,
        raw_action_hz: Optional[int] = None,
        train_action_hz: Optional[int] = None,
        raw_tactile_hz: Optional[int] = None,
        train_tactile_hz: Optional[int] = None,
        action_mode: str = "qpos20_rel6d",
        action_from_state: bool = True,
        use_state: bool = False,
        val: bool = False,
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        max_episodes: Optional[int] = None,
        include_fail: bool = False,
        include_old: bool = False,
        cache_dir: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        use_index_cache: bool = True,
        manifest_path: Optional[str] = None,
        manifest_sample_type: Optional[str] = None,
        value_target: Optional[str] = None,
        progress_c: float = 0.01,
        treat_as_irrelevant: bool = False,
        tactile_enabled: bool = True,
        bootstrap_start_window: bool = False,
        bootstrap_chunk_size: int = 2,
        bootstrap_start_window_prob: float = 0.5,
        view_dropout: Optional[Dict[str, Any]] = None,
        sequence_mode: str = "random",
        sequence_max_latents: Optional[int] = None,
        sequence_overflow_policy: str = "truncate",
        latent_budget: Optional[int] = None,
        over_budget_policy: str = "allow_single",
        length_index_cache: Optional[str] = None,
        mask_invalid_replay: bool = False,
        **kwargs,
    ):
        if action_mode not in {"qpos20_rel6d", "relqpos20_rel6d"}:
            raise ValueError(
                "Tianji/Wuji supports action_mode=qpos20_rel6d or "
                f"relqpos20_rel6d, got {action_mode}"
            )
        self.action_mode = str(action_mode)
        self.clickhouse = _mapping_to_plain_dict(clickhouse)
        self.episode_index_source = str(episode_index_source or "auto").lower()
        self.manifest_path = str(Path(manifest_path).expanduser()) if manifest_path else None
        self.manifest_sample_type = _to_plain_container(manifest_sample_type)
        if self.manifest_path and self.episode_index_source == "auto":
            self.episode_index_source = "manifest"
        if self.episode_index_source not in {"auto", "dataset_dir", "clickhouse", "manifest"}:
            raise ValueError(f"Unsupported Tianji/Wuji episode_index_source: {self.episode_index_source}")
        if self.episode_index_source == "manifest" and not self.manifest_path:
            raise ValueError("Tianji/Wuji episode_index_source=manifest requires manifest_path")
        if dataset_dir is None:
            self.dataset_dir = None
        else:
            self.dataset_dir = (
                _normalize_remote_path(str(dataset_dir).rstrip("/"))
                if is_remote(str(dataset_dir))
                else Path(dataset_dir).expanduser()
            )
        self.episode_path_prefix = (
            _normalize_remote_path(str(episode_path_prefix).rstrip("/"))
            if episode_path_prefix
            else None
        )
        self.latent_root = latent_root.strip("/")
        # OmegaConf may pass ListConfig; normalize to a plain set of ints or None.
        if latent_phases is None:
            self.latent_phases_set: Optional[set] = None
        else:
            phases = _to_plain_container(latent_phases)
            self.latent_phases_set = {int(p) for p in phases} if phases else set()
        self.latent_resolution = latent_resolution
        self.left_latent_name = left_latent_name or f"cam_left_{latent_resolution}_VAE.pt"
        self.right_latent_name = right_latent_name or f"cam_right_{latent_resolution}_VAE.pt"
        raw_view_latent_names = _to_plain_container(view_latent_names)
        if raw_view_latent_names is None:
            self.view_latent_names = None
        elif isinstance(raw_view_latent_names, str):
            self.view_latent_names = [raw_view_latent_names]
        else:
            self.view_latent_names = [str(name) for name in raw_view_latent_names if name]
        if self.view_latent_names is not None and not self.view_latent_names:
            raise ValueError("view_latent_names must contain at least one latent name")
        self.single_latent_name = single_latent_name
        if self.single_latent_name and self.view_latent_names:
            raise ValueError("single_latent_name and view_latent_names are mutually exclusive")
        self.use_single_latent = bool(self.single_latent_name)
        self.use_view_latent_names = bool(self.view_latent_names)
        # "stack" adds an explicit view axis, which this model does not consume.
        # "concat_width" tiles the views side by side inside one latent frame, so a
        # V-view episode stays a [C,T,H,V*W] tensor and every view keeps its own
        # width-axis RoPE positions.
        self.view_merge = str(view_merge or "stack").strip().lower()
        if self.view_merge not in {"stack", "concat_width"}:
            raise ValueError(
                f"view_merge must be 'stack' or 'concat_width', got {view_merge!r}"
            )
        self.language_subpath = language_subpath
        self.tjwj_subpath = str(tjwj_subpath).strip("/") if tjwj_subpath else None
        self._language_candidates_cache: Dict[str, List[str]] = {}
        self.max_latent_window = int(max_latent_window)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        (
            self.video_hz,
            self.raw_action_hz,
            self.train_action_hz,
            self.action_downsample_stride,
            self.raw_tactile_hz,
            self.train_tactile_hz,
            self.tactile_downsample_stride,
            self.action_upsample_ratio,
        ) = _resolve_rate_config(
            video_hz=video_hz,
            action_cfg=action,
            tactile_cfg=tactile,
            raw_action_hz=raw_action_hz,
            train_action_hz=train_action_hz,
            raw_tactile_hz=raw_tactile_hz,
            train_tactile_hz=train_tactile_hz,
            action_upsample_ratio=int(action_upsample_ratio),
        )
        # When True, per-episode downsample uses task_name: names matching 30hz/30_hz/30Hz
        # keep stride=1 (already 30Hz); others use the global raw_hz→train_hz stride.
        self.adaptive_raw_hz_from_task_name = _cfg_bool(
            action, "adaptive_raw_hz_from_task_name", False
        ) or _cfg_bool(tactile, "adaptive_raw_hz_from_task_name", False)
        self.actions_per_latent = 4 * self.video_action_freq_ratio * self.action_upsample_ratio
        self.use_state = bool(use_state)
        self.action_from_state = bool(action_from_state)
        self.mask_invalid_replay = bool(mask_invalid_replay)
        self._logged_arm_action_endpose = False
        self.val = bool(val)
        self.task_mode = str(task_mode)
        self.task_name = _to_plain_container(task_name)
        self.include_fail = bool(include_fail)
        self.include_old = bool(include_old)
        self.cache_dir = cache_dir
        self.index_cache_path = str(Path(index_cache_path).expanduser()) if index_cache_path else None
        self.use_index_cache = bool(use_index_cache)
        self.value_target = value_target
        if self.value_target is not None and self.value_target not in {"progress", "ttc", "vlac"}:
            raise ValueError(
                f"TianjiWujiDataset: unsupported value_target {self.value_target!r}; "
                "use 'progress', 'ttc', or 'vlac'"
            )
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.bootstrap_start_window = bool(bootstrap_start_window)
        self.bootstrap_chunk_size = int(bootstrap_chunk_size)
        self.bootstrap_start_window_prob = float(min(max(bootstrap_start_window_prob, 0.0), 1.0))
        self.sequence_mode = str(sequence_mode).strip().lower()
        if self.sequence_mode not in {"random", "episode"}:
            raise ValueError(
                f"sequence_mode must be 'random' or 'episode', got {sequence_mode!r}"
            )
        self.sequence_max_latents = (
            int(sequence_max_latents) if sequence_max_latents is not None else None
        )
        self.sequence_overflow_policy = str(sequence_overflow_policy).strip().lower()
        self.latent_budget = int(latent_budget) if latent_budget is not None else None
        self.over_budget_policy = str(over_budget_policy).strip().lower()
        self.length_index_cache = length_index_cache
        self._sample_lengths: Optional[List[int]] = None
        # A packed batch is assembled from whole trajectories whose lengths the
        # sampler reads ahead of time, so a sample has to be a pure function of
        # its index. The random path draws a fresh episode per call and cannot
        # answer that question; bootstrap replication would also change a
        # sample's length after the packer already committed to it.
        if self.sequence_mode == "episode" and self.bootstrap_start_window:
            raise ValueError(
                "sequence_mode='episode' cannot be combined with "
                "bootstrap_start_window: replicating the first latent changes "
                "the sample length the length index already reported"
            )
        # This repo's bootstrap helper has no probability knob; it applies
        # whenever enabled. Accepting a probability we cannot honour would
        # quietly train a different recipe than the config describes.
        if self.bootstrap_start_window and self.bootstrap_start_window_prob != 1.0:
            raise ValueError(
                "bootstrap_start_window_prob="
                f"{self.bootstrap_start_window_prob} is not supported here; "
                "bootstrap_start_window always applies when enabled"
            )
        self._init_view_dropout(view_dropout)
        if self.view_dropout_enabled and self.view_merge == "concat_width":
            raise ValueError(
                "view_dropout needs an explicit view axis; it cannot address views "
                "merged with view_merge='concat_width'"
            )
        self.tactile_enabled = bool(tactile_enabled)
        if self.task_mode == "single" and not self.task_name:
            raise ValueError("Single task mode requires task_name")

        self.action_stats = self._load_json(action_stats_path)
        self.qpos_normalization = str(qpos_normalization or "mean_std")
        self._init_qpos_normalization_stats()
        self.tactile_stats = self._load_json(tactile_stats_path) if self.tactile_enabled else None

        if self.tactile_enabled:
            norm = self.tactile_stats["normalization"]
            self.tactile_norm_method = str(norm.get("method") or "baseline_scale")
            if self.tactile_norm_method != "preprocessed_float_0_1":
                raise ValueError(
                    "Tianji/Wuji tactile.pt must be preprocessed float32 [0,1] now; "
                    f"got normalization method={self.tactile_norm_method}"
                )

            masks = self.tactile_stats.get("valid_channel_mask") or {}
            left_mask = torch.tensor(masks.get("left", [True] * 768), dtype=torch.bool)
            right_mask = torch.tensor(masks.get("right", [True] * 768), dtype=torch.bool)
            self.tactile_valid_mask = torch.cat([left_mask, right_mask], dim=0)
        else:
            self.tactile_norm_method = "disabled"
            self.tactile_valid_mask = torch.ones(TACTILE_DIM, dtype=torch.bool)

        # Optional per-taxel baseline subtraction (calibration from rest pose) — applied
        # after the existing min-max normalization. Both training and inference load from
        # the same npy directory via `data.tianji_wuji.tactile_baseline` for byte-parity.
        #
        # Two-level routing:
        #   1. tactile_baseline_per_source — list of {pattern, dir, kind} entries; each
        #      episode's `task_name` is fnmatched against `pattern` in list order, first
        #      match wins. Use this when training mixes data sources with different
        #      hardware baselines (e.g. human teleop + dexterous robot palms).
        #   2. tactile_baseline_dir — single default baseline applied to episodes that
        #      didn't match any per-source pattern (or used alone if no per-source list
        #      is given).
        # When BOTH are unset, no Stage-2 subtraction happens (only min-max + json mask).
        self.tactile_baseline = None              # default baseline tensor [1536] or None
        # Native-space full-scale/floor for the DEFAULT baseline path (per-source
        # entries carry their own in the _tactile_baseline_sources tuples).
        self.tactile_default_tactile_max = 1.0
        self.tactile_default_tactile_min = 0.0
        self.tactile_baseline_kind = str(tactile_baseline_kind)
        self.tactile_baseline_polarity = str(tactile_baseline_polarity)
        if self.tactile_baseline_polarity not in {"positive", "negative"}:
            raise ValueError(f"tactile_baseline_polarity must be positive/negative, got {tactile_baseline_polarity!r}")
        # Each tuple: (pattern, baseline[1536], valid_mask[1536], kind, polarity, tactile_max, tactile_min)
        self._tactile_baseline_sources: List[Tuple[str, torch.Tensor, torch.Tensor, str, str, float, float]] = []
        if self.tactile_enabled:
            from data.tianji_wuji.tactile_baseline import (
                load_baseline_calibration, load_baseline_from_json, load_tactile_scale_from_json,
            )

            def _resolve_one(*, name: Optional[str], dir_: Optional[str], kind: str,
                             polarity_default: str) -> Tuple[torch.Tensor, torch.Tensor, str, float, float]:
                """Load (baseline, mask, polarity, tactile_max, tactile_min).

                JSON name takes precedence over npy dir. tactile_max/tactile_min are the
                native-space full-scale/floor read from the JSON entry; npy dirs and
                legacy entries without them default to (1.0, 0.0).
                """
                if name and dir_:
                    raise ValueError(f"specify either name or dir, not both (got name={name!r}, dir={dir_!r})")
                if name:
                    b, m, p = load_baseline_from_json(tactile_stats_path, name, kind=kind)
                    try:
                        tmax, tmin = load_tactile_scale_from_json(tactile_stats_path, name)
                    except KeyError:
                        tmax, tmin = 1.0, 0.0
                    return b, m, p, tmax, tmin
                if dir_:
                    b, m = load_baseline_calibration(dir_, kind=kind)
                    return b, m, polarity_default, 1.0, 0.0
                raise ValueError("must specify either name (JSON) or dir (npy)")

            # default baseline (name OR dir; unconditional)
            if tactile_baseline_name or tactile_baseline_dir:
                base_def, mask_def, pol_def, tmax_def, tmin_def = _resolve_one(
                    name=tactile_baseline_name, dir_=tactile_baseline_dir,
                    kind=self.tactile_baseline_kind,
                    polarity_default=self.tactile_baseline_polarity,
                )
                self.tactile_baseline = base_def
                # When loaded via JSON name, polarity from JSON entry takes precedence over config default;
                # this matters when JSON encodes the "correct" polarity for that hardware.
                self.tactile_baseline_polarity = pol_def
                self.tactile_default_tactile_max = tmax_def
                self.tactile_default_tactile_min = tmin_def
                # Conservative: any taxel invalid in EITHER mask is invalid globally.
                self.tactile_valid_mask = self.tactile_valid_mask & mask_def

            # per-source baselines (override default at lookup time per episode)
            if tactile_baseline_per_source:
                for entry in tactile_baseline_per_source:
                    # Accept dict OR OmegaConf DictConfig; both expose .get().
                    if not hasattr(entry, "get"):
                        raise TypeError(
                            "tactile_baseline_per_source entries must be mappings with "
                            f"keys {{pattern, name|dir, [kind], [polarity]}}, got {type(entry).__name__}"
                        )
                    pat = entry.get("pattern")
                    if not pat:
                        raise ValueError(f"tactile_baseline_per_source entry missing 'pattern': {entry}")
                    kind = str(entry.get("kind") or self.tactile_baseline_kind)
                    polarity_entry = entry.get("polarity")  # may be None → use whatever _resolve_one returns
                    polarity_default = str(polarity_entry or self.tactile_baseline_polarity)
                    if polarity_default not in {"positive", "negative"}:
                        raise ValueError(
                            f"per-source baseline polarity must be positive/negative, got {polarity_default!r}"
                        )
                    base_s, mask_s, pol_resolved, tmax_s, tmin_s = _resolve_one(
                        name=entry.get("name"), dir_=entry.get("dir"),
                        kind=kind, polarity_default=polarity_default,
                    )
                    # yaml entry's explicit polarity overrides JSON's; otherwise use whatever _resolve_one returned
                    polarity = str(polarity_entry) if polarity_entry else pol_resolved
                    if polarity not in {"positive", "negative"}:
                        raise ValueError(
                            f"per-source resolved polarity invalid: {polarity!r}"
                        )
                    self._tactile_baseline_sources.append((str(pat), base_s, mask_s, kind, polarity, tmax_s, tmin_s))
                    src_label = f"name={entry.get('name')!r}" if entry.get("name") else f"dir={entry.get('dir')!r}"
                    logger.info(
                        "TianjiWujiDataset: registered per-source baseline pattern=%s kind=%s polarity=%s %s "
                        "(valid taxels=%d/%d)",
                        pat, kind, polarity, src_label, int(mask_s.sum()), mask_s.numel(),
                    )

        # === BUG FIX (T22): the rest of __init__ that follows MUST stay inside __init__.
        # Previously `def _resolve_baseline` got inserted ABOVE these lines during T16,
        # leaving them as unreachable code inside that method → self.episodes never set.
        self.episodes = self._load_episode_index()
        if max_episodes is not None:
            self.episodes = self.episodes[: int(max_episodes)]
        # T22: per-task episode skip (drop first N episodes of matching task_name pattern).
        # Useful when first few episodes of a task are calibration / bad takes.
        if task_episode_skip_first:
            kept: List[Dict[str, Any]] = []
            skipped_per_task: Dict[str, int] = {}
            quota: Dict[str, int] = {}   # remaining-to-skip per pattern
            for entry in task_episode_skip_first:
                # Accept both plain dict and OmegaConf DictConfig — both support `in` /
                # subscript access; do NOT use isinstance(entry, dict) (DictConfig is not
                # a subclass of dict).
                try:
                    has_keys = ("task" in entry) and ("skip_first" in entry)
                except TypeError:
                    has_keys = False
                if not has_keys:
                    raise ValueError(
                        f"task_episode_skip_first entries must be mappings with "
                        f"keys {{task, skip_first}}, got {entry!r}"
                    )
                quota[str(entry["task"])] = int(entry["skip_first"])
            for ep in self.episodes:
                tname = ep.get("task_name") if isinstance(ep, dict) else None
                skipped = False
                if tname is not None:
                    for pat in list(quota):
                        if self._name_matches(str(tname), pat) and quota[pat] > 0:
                            quota[pat] -= 1
                            skipped_per_task[pat] = skipped_per_task.get(pat, 0) + 1
                            skipped = True
                            break
                if not skipped:
                    kept.append(ep)
            for pat, n in skipped_per_task.items():
                logger.info("TianjiWujiDataset: skipped first %d episodes of task pattern %r", n, pat)
            self.episodes = kept
        if not self.episodes:
            raise RuntimeError(f"No Tianji/Wuji episodes found under {self.dataset_dir}")

        n_native_30 = sum(
            1
            for ep in self.episodes
            if _task_name_implies_native_30hz(ep.get("task_name") if isinstance(ep, dict) else None)
        )
        logger.info(
            "TianjiWujiDataset: episodes=%d, max_latent_window=%d, actions_per_latent=%d, "
            "action_hz=%d->%d/downsample=%d, qpos_norm=%s, tactile_hz=%d->%d/downsample=%d, "
            "adaptive_raw_hz_from_task_name=%s (native_30hz_eps=%d, default_raw_eps=%d), "
            "latent_resolution=%s, tactile=%s, action_from_state=%s, "
            "bootstrap_start=%s/%d@p=%.2f, value_target=%s, source=%s",
            len(self.episodes),
            self.max_latent_window,
            self.actions_per_latent,
            self.raw_action_hz,
            self.train_action_hz,
            self.action_downsample_stride,
            self.qpos_normalization,
            self.raw_tactile_hz,
            self.train_tactile_hz,
            self.tactile_downsample_stride,
            self.adaptive_raw_hz_from_task_name,
            n_native_30,
            len(self.episodes) - n_native_30,
            self.latent_resolution,
            self.tactile_enabled,
            self.action_from_state,
            self.bootstrap_start_window,
            self.bootstrap_chunk_size,
            self.bootstrap_start_window_prob,
            self.value_target,
            self._discovery_source(),
        )

    def _resolve_baseline(
        self, task_name: Optional[str]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, str, float, float]:
        """Return (baseline_or_None, valid_mask, polarity) for an episode's task_name.

        Per-source entries are matched in list order via fnmatch (same semantics as
        `_task_allowed`). When nothing matches:
          - strict=True AND no default baseline → raise ValueError (T21 防呆)
          - otherwise → falls back to default baseline + global mask + default polarity
        """
        if task_name is not None and self._tactile_baseline_sources:
            for pat, base_s, mask_s, _, polarity, tmax, tmin in self._tactile_baseline_sources:
                if self._name_matches(str(task_name), pat):
                    # Source-specific mask still AND'd with the dataset-wide json mask.
                    # (The json mask reflects channels physically dead across the whole
                    # corpus; per-source mask reflects calibration-specific deadness.)
                    return base_s, self.tactile_valid_mask & mask_s, polarity, tmax, tmin
        # No per-source match: a baseline is REQUIRED. There is intentionally NO strict
        # toggle — if there is also no default baseline, ALWAYS raise so a missing tactile
        # normalization can never pass silently.
        if self.tactile_baseline is None:
            patterns = [p for p, *_ in self._tactile_baseline_sources]
            raise ValueError(
                f"no per-source baseline pattern matched task_name={task_name!r}, "
                f"and no default tactile_baseline_name/dir is set. "
                f"Add a per-source pattern that covers this task_name, or set a default baseline. "
                f"(registered patterns: {patterns})"
            )
        return (self.tactile_baseline, self.tactile_valid_mask, self.tactile_baseline_polarity,
                self.tactile_default_tactile_max, self.tactile_default_tactile_min)

    @staticmethod
    def _load_json(path: str | Path) -> dict:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return json.loads(path.read_text(encoding="utf-8"))

    def _init_view_dropout(self, view_dropout: Optional[Dict[str, Any]]) -> None:
        cfg = _mapping_to_plain_dict(view_dropout) if view_dropout is not None else {}
        self.view_dropout_enabled = bool(cfg.get("enabled", False))
        self.view_dropout_prob = float(cfg.get("prob", 0.0))
        if not 0.0 <= self.view_dropout_prob <= 1.0:
            raise ValueError(f"view_dropout.prob must be in [0, 1], got {self.view_dropout_prob}")

        self.view_dropout_min_keep_views = int(cfg.get("min_keep_views", 1))
        if self.view_dropout_min_keep_views < 1:
            raise ValueError(
                f"view_dropout.min_keep_views must be >= 1, got {self.view_dropout_min_keep_views}"
            )

        self.view_dropout_fill = str(cfg.get("fill", "zero")).lower()
        if self.view_dropout_fill not in {"zero", "noise"}:
            raise ValueError(
                f"unsupported view_dropout.fill={self.view_dropout_fill!r}; "
                "supported fill modes are 'zero' and 'noise'"
            )
        self.view_dropout_noise_std = float(cfg.get("noise_std", 1.0))
        if self.view_dropout_noise_std < 0:
            raise ValueError(f"view_dropout.noise_std must be >= 0, got {self.view_dropout_noise_std}")

    def _apply_view_dropout(self, latents: torch.Tensor) -> torch.Tensor:
        if self.val or not self.view_dropout_enabled or self.view_dropout_prob <= 0.0:
            return latents
        if latents.ndim < 5:
            return latents

        num_views = int(latents.shape[0])
        if num_views <= 1:
            return latents

        max_drop = max(0, num_views - self.view_dropout_min_keep_views)
        if max_drop == 0:
            return latents

        drop_mask = torch.rand(num_views, device=latents.device) < self.view_dropout_prob
        drop_indices = torch.nonzero(drop_mask, as_tuple=False).flatten()
        if drop_indices.numel() == 0:
            return latents
        if drop_indices.numel() > max_drop:
            order = torch.randperm(drop_indices.numel(), device=latents.device)
            drop_indices = drop_indices[order[:max_drop]]

        latents = latents.clone()
        if self.view_dropout_fill == "zero":
            latents[drop_indices] = 0
        elif self.view_dropout_fill == "noise":
            latents[drop_indices] = torch.randn_like(latents[drop_indices]) * self.view_dropout_noise_std
        else:
            raise ValueError(
                f"unsupported view_dropout.fill={self.view_dropout_fill!r}; "
                "supported fill modes are 'zero' and 'noise'"
            )
        return latents

    def _as_path_str(self, path) -> str:
        if path is None:
            return ""
        return str(path).rstrip("/")

    def _join_path(self, base, *parts: str):
        base_s = self._as_path_str(base)
        if is_remote(base_s):
            return smart_join(base_s, *parts)
        return Path(base_s).joinpath(*parts)

    def _basename(self, path) -> str:
        return self._as_path_str(path).rsplit("/", 1)[-1]

    def _parent(self, path):
        path_s = self._as_path_str(path)
        if is_remote(path_s):
            return path_s.rsplit("/", 1)[0]
        return Path(path_s).parent

    def _exists(self, path) -> bool:
        return smart_exists(self._as_path_str(path)) if is_remote(self._as_path_str(path)) else Path(path).exists()

    def _language_present(self, ep_dir) -> bool:
        """Discovery existence check for the language source.

        `language_subpath` may be a single .pt file OR a directory of .pt files
        (directory mode -> ALL instruction .pt are used, one randomly sampled per
        __getitem__). For a directory, require >=1 .pt under it: object stores have
        no real directories, so a plain existence check on the prefix is unreliable.
        For the .pt-file form this is exactly the previous behavior.
        """
        lang = self._join_path(ep_dir, self.language_subpath)
        if str(self.language_subpath).endswith(".pt"):
            return self._exists(lang)
        return len(smart_glob_pt(self._as_path_str(lang))) > 0

    def _load_pt(self, path):
        return smart_load_pt(self._as_path_str(path), self.cache_dir, map_location="cpu")

    def _skip_path(self, path) -> bool:
        parts = [p.lower() for p in self._as_path_str(path).split("/")]
        if not self.include_fail and any("fail" in p for p in parts):
            return True
        if not self.include_old and any(p == "old" or p.startswith("old_") for p in parts):
            return True
        return False

    @staticmethod
    def _name_matches(name: str, pattern: str) -> bool:
        name = str(name).strip("/")
        pattern = str(pattern).strip("/").replace("%", "*")
        return fnmatch.fnmatchcase(name, pattern)

    # Per-sample embodiment tag carried into the batch (TactAlign Stage-0 plumbing).
    # 1 = human (human_hand glove), 0 = robot/dex (e.g. lingqiaoshou).
    # NOTE: this does NOT affect tactile normalization — that is already routed
    # per-episode by task_name in _resolve_baseline(). This tag exists purely so
    # cross-embodiment modules (e.g. a TactAlign rectified-flow vθ) can gate a
    # human-only latent transport at model.forward(); it is otherwise unused.
    SOURCE_ROBOT = 0
    SOURCE_HUMAN = 1

    @classmethod
    def _resolve_source(cls, task_name: Optional[str]) -> int:
        return cls.SOURCE_HUMAN if "human_hand" in str(task_name or "") else cls.SOURCE_ROBOT

    def _task_allowed(self, task_name: str) -> bool:
        if not self.task_name:
            return True
        patterns = self.task_name if isinstance(self.task_name, (list, tuple)) else [self.task_name]
        return any(self._name_matches(task_name, pat) for pat in patterns)

    def _discovery_source(self) -> str:
        if self.episode_index_source == "auto":
            return "clickhouse" if self.clickhouse is not None else "dataset_dir"
        return self.episode_index_source

    def _index_cache_meta(self) -> Dict[str, Any]:
        ch_meta = None
        if self.clickhouse:
            ch_meta = {
                k: self.clickhouse.get(k)
                for k in ("host", "port", "username", "database", "table", "secure", "use_final")
                if k in self.clickhouse
            }
        return {
            "schema": "tianji_wuji_episode_index_v1",
            "episode_index_source": self.episode_index_source,
            "episode_path_prefix": self.episode_path_prefix,
            "dataset_dir": str(self.dataset_dir),
            "clickhouse": ch_meta,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "include_fail": self.include_fail,
            "include_old": self.include_old,
            "manifest_path": self.manifest_path,
            "manifest_sample_type": self.manifest_sample_type,
            "latent_root": self.latent_root,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set is not None else None,
            "left_latent_name": self.left_latent_name,
            "right_latent_name": self.right_latent_name,
            "view_latent_names": self.view_latent_names,
            "view_merge": self.view_merge,
            "single_latent_name": self.single_latent_name,
            "language_subpath": self.language_subpath,
            "tjwj_subpath": self.tjwj_subpath,
            "tactile_enabled": self.tactile_enabled,
            "action_from_state": self.action_from_state,
        }

    def _load_cached_index(self) -> Optional[List[Dict[str, Any]]]:
        if not self.use_index_cache or not self.index_cache_path:
            return None
        path = Path(self.index_cache_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable Tianji/Wuji index cache %s: %s", path, exc)
            return None
        if payload.get("meta") != self._index_cache_meta():
            logger.warning("Ignoring stale Tianji/Wuji index cache: %s", path)
            return None
        episodes = payload.get("episodes", [])
        logger.info("Loaded Tianji/Wuji index cache: %s (%d episodes)", path, len(episodes))
        return episodes

    def _save_index_cache(self, episodes: List[Dict[str, Any]]) -> None:
        if not self.use_index_cache or not self.index_cache_path:
            return
        path = Path(self.index_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps({"meta": self._index_cache_meta(), "episodes": episodes}),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _load_episode_index(self) -> List[Dict[str, Any]]:
        source = self._discovery_source()
        cached = self._load_cached_index()
        if cached is not None:
            return cached
        if source == "clickhouse":
            episodes = self._load_episodes_from_clickhouse()
        elif source == "manifest":
            episodes = self._load_episodes_from_manifest()
        else:
            episodes = self._scan_episodes()
        episodes = self._expand_latent_phases(episodes)
        self._save_index_cache(episodes)
        return episodes

    @staticmethod
    def _phase_index_from_component(name: str) -> Optional[int]:
        m = re.match(r"^(\d+)phase_", str(name), flags=re.IGNORECASE)
        if m is not None:
            return int(m.group(1))
        return None

    def _latent_root_is_phase_dir(self) -> bool:
        """True when latent_root already points at a `{N}phase_*` directory."""
        return self._phase_index_from_component(self.latent_root.rsplit("/", 1)[-1]) is not None

    def _wan22_parent(self) -> str:
        """Parent of `{N}phase_5hz` dirs (e.g. `video_features/wan2.2`)."""
        if self._latent_root_is_phase_dir():
            parts = self.latent_root.rsplit("/", 1)
            return parts[0] if len(parts) > 1 else ""
        return self.latent_root

    def _phase_latent_root(self, phase_index: int) -> str:
        parent = self._wan22_parent()
        phase_dir = f"{int(phase_index)}phase_5hz"
        return f"{parent}/{phase_dir}" if parent else phase_dir

    def _default_phase_start_30(self) -> int:
        idx = self._phase_index_from_component(self.latent_root.rsplit("/", 1)[-1])
        return int(idx) if idx is not None else 0

    def _discover_phases_for_episode(self, ep_dir) -> List[int]:
        """List `{N}phase_*` dirs under the wan2.2 parent for one episode."""
        parent = self._wan22_parent()
        root = self._join_path(ep_dir, parent) if parent else ep_dir
        root_s = self._as_path_str(root)
        try:
            if is_remote(root_s):
                names = smart_listdir(root_s)
            else:
                path = Path(root_s)
                if not path.is_dir():
                    return []
                names = [p.name for p in path.iterdir()]
        except OSError as exc:
            logger.debug("phase listdir failed %s: %s", root_s, exc)
            return []
        out: List[int] = []
        for name in names:
            idx = self._phase_index_from_component(str(name))
            if idx is not None:
                out.append(idx)
        return sorted(set(out))

    def _latents_exist_under(self, ep_dir, latent_root: str) -> bool:
        if self.single_latent_name:
            return self._exists(self._join_path(ep_dir, latent_root, self.single_latent_name))
        if self.view_latent_names:
            return all(
                self._exists(self._join_path(ep_dir, latent_root, name))
                for name in self.view_latent_names
            )
        return (
            self._exists(self._join_path(ep_dir, latent_root, self.left_latent_name))
            and self._exists(self._join_path(ep_dir, latent_root, self.right_latent_name))
        )

    def _episode_has_any_configured_latents(self, ep_dir) -> bool:
        if self.latent_phases_set is not None:
            roots = [self._phase_latent_root(p) for p in sorted(self.latent_phases_set)]
        elif self._latent_root_is_phase_dir():
            roots = [self.latent_root]
        else:
            discovered = self._discover_phases_for_episode(ep_dir)
            roots = (
                [self._phase_latent_root(p) for p in discovered]
                if discovered
                else [self.latent_root]
            )
        return any(self._latents_exist_under(ep_dir, root) for root in roots)

    def _expand_latent_phases(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach phase-specific latent paths and phase_start_30 for action alignment.

        Backward compatible: when ``latent_phases`` is unset/None and ``latent_root``
        already ends with ``{N}phase_*``, keep a single row per episode using that root.
        """
        # Legacy single-phase: latent_root points at e.g. 0phase_5hz and no filter set.
        if self.latent_phases_set is None and self._latent_root_is_phase_dir():
            phase0 = self._default_phase_start_30()
            for ep in episodes:
                ep.setdefault("phase_start_30", phase0)
                ep.setdefault("latent_root", self.latent_root)
            return episodes

        expanded: List[Dict[str, Any]] = []
        skipped_missing = 0
        for ep in episodes:
            ep_dir = ep["episode_path"]
            task_name = str(ep["task_name"])
            base_name = str(
                ep.get("source_episode_name")
                or ep.get("episode_name")
                or self._basename(ep_dir)
            )
            if self.latent_phases_set is not None:
                phases = sorted(self.latent_phases_set)
            else:
                phases = self._discover_phases_for_episode(ep_dir)
                if not phases:
                    phases = [0]
            for phase in phases:
                rec = self._episode_record(
                    ep_dir,
                    task_name,
                    episode_name=f"{base_name}_p{phase}",
                    phase_start_30=int(phase),
                )
                for key in ("sample_type", "source_type"):
                    if key in ep:
                        rec[key] = ep[key]
                rec["source_episode_name"] = base_name
                if not self._latents_exist_under(ep_dir, rec["latent_root"]):
                    skipped_missing += 1
                    continue
                expanded.append(rec)

        if not expanded:
            raise RuntimeError(
                "Tianji/Wuji multi-phase expansion produced 0 episodes. "
                f"latent_root={self.latent_root!r}, latent_phases="
                f"{sorted(self.latent_phases_set) if self.latent_phases_set is not None else None}. "
                "Ensure `{N}phase_5hz` VAE features exist on disk."
            )
        logger.info(
            "Tianji/Wuji latent_phases expansion: in=%d out=%d skipped_missing_phase=%d "
            "latent_phases=%s latent_root=%s",
            len(episodes),
            len(expanded),
            skipped_missing,
            sorted(self.latent_phases_set) if self.latent_phases_set is not None else "scan_all",
            self.latent_root,
        )
        return expanded

    def _episode_path_from_clickhouse(self, raw_path: str) -> str:
        raw = str(raw_path).rstrip("/")
        if is_remote(raw):
            return _normalize_remote_path(raw)
        if self.episode_path_prefix and is_remote(self.episode_path_prefix):
            raw_core = _strip_remote_scheme(raw)
            prefix_core = _strip_remote_scheme(self.episode_path_prefix)
            if raw_core.startswith(prefix_core):
                return "s3://" + raw_core
        return raw

    def _clickhouse_where_sql(self, path_col: str, task_col: str) -> str:
        conditions = []
        prefix = self.episode_path_prefix or (
            self.dataset_dir if isinstance(self.dataset_dir, str) and is_remote(self.dataset_dir) else None
        )
        if prefix:
            core = _strip_remote_scheme(str(prefix).rstrip("/")).replace("'", "''")
            conditions.append(f"{path_col} LIKE '%{core}%'")
        if self.task_name:
            patterns = self.task_name if isinstance(self.task_name, (list, tuple)) else [self.task_name]
            task_terms = []
            for pat in patterns:
                sql_pat = str(pat).replace("*", "%").replace("'", "''")
                task_terms.append(f"{task_col} LIKE '{sql_pat}'")
            conditions.append("(" + " OR ".join(task_terms) + ")")
        return (" WHERE " + " AND ".join(conditions)) if conditions else ""

    def _load_episodes_from_clickhouse(self) -> List[Dict[str, Any]]:
        if not self.clickhouse:
            raise ValueError("Tianji/Wuji clickhouse config is missing")
        ch = self.clickhouse
        table = str(ch.get("table") or "ario_format_data")
        final_kw = " FINAL" if bool(ch.get("use_final", True)) else ""
        desc_rows = _clickhouse_query_json_each_row(f"DESCRIBE TABLE {table} FORMAT JSONEachRow", ch)
        available_cols = sorted(str(row.get("name")) for row in desc_rows if row.get("name"))

        path_expr = _column_alias_expr_for_sql(
            "episode_bos_path", available_cols, ["episode_bos_path", "episode_path", "bos_path"]
        )
        task_expr = _column_alias_expr_for_sql("task", available_cols, ["task", "task_name"])
        episode_id_expr = _column_alias_expr_for_sql(
            "episode_id", available_cols, ["episode_id", "episode_name"]
        )
        if not path_expr or not task_expr:
            raise ValueError(
                "ClickHouse table schema missing required Tianji/Wuji columns. "
                f"available columns: {available_cols}"
            )
        path_col = path_expr.split(" AS ", 1)[0]
        task_col = task_expr.split(" AS ", 1)[0]
        where = self._clickhouse_where_sql(path_col, task_col)
        sql = f"""
            SELECT
                {path_expr},
                {task_expr},
                {episode_id_expr or "'' AS episode_id"}
            FROM {table}{final_kw}
            {where}
            FORMAT JSONEachRow
        """
        rows = _clickhouse_query_json_each_row(sql, ch)
        episodes = []
        for row in rows:
            raw_path = row.get("episode_bos_path")
            task_name = row.get("task")
            if not raw_path or not task_name:
                continue
            ep_dir = self._episode_path_from_clickhouse(str(raw_path))
            if self._skip_path(ep_dir) or not self._task_allowed(str(task_name)):
                continue
            episodes.append(self._episode_record(ep_dir, str(task_name), row.get("episode_id")))
        logger.info("ClickHouse Tianji/Wuji index returned %d episodes from %s", len(episodes), table)
        return episodes

    def _manifest_sample_type_allowed(self, sample_type: str) -> bool:
        if not self.manifest_sample_type:
            return True
        allowed = (
            self.manifest_sample_type
            if isinstance(self.manifest_sample_type, (list, tuple, set))
            else [self.manifest_sample_type]
        )
        return str(sample_type).lower() in {str(item).lower() for item in allowed}

    def _load_episodes_from_manifest(self) -> List[Dict[str, Any]]:
        path = Path(str(self.manifest_path)).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
        if not isinstance(raw_episodes, list):
            raise ValueError(f"Tianji/Wuji manifest episodes must be a list: {path}")

        episodes: List[Dict[str, Any]] = []
        for row in raw_episodes:
            if not isinstance(row, dict):
                continue
            sample_type = str(row.get("sample_type", "positive")).lower()
            if not self._manifest_sample_type_allowed(sample_type):
                continue
            ep_dir = row.get("episode_path")
            if not ep_dir:
                continue
            ep_dir = _normalize_remote_path(str(ep_dir).rstrip("/"))
            task_name = str(row.get("task_name") or self._basename(self._parent(ep_dir)))
            if not self._task_allowed(task_name):
                continue
            record = self._episode_record(ep_dir, task_name, row.get("episode_name"))
            record["sample_type"] = sample_type
            record["source_type"] = str(row.get("source_type", sample_type)).lower()
            episodes.append(record)

        logger.info(
            "Loaded Tianji/Wuji manifest: %s sample_type=%s episodes=%d",
            path,
            self.manifest_sample_type or "all",
            len(episodes),
        )
        return episodes

    def _episode_record(
        self,
        ep_dir,
        task_name: str,
        episode_name: Optional[str] = None,
        phase_start_30: Optional[int] = None,
    ) -> Dict[str, Any]:
        if phase_start_30 is None:
            latent_root = self.latent_root
            phase_start_30 = self._default_phase_start_30()
        else:
            latent_root = self._phase_latent_root(int(phase_start_30))
            phase_start_30 = int(phase_start_30)
        return {
            "episode_path": ep_dir,
            "task_name": task_name,
            "episode_name": str(episode_name or self._basename(ep_dir)),
            "latent_root": latent_root,
            "phase_start_30": phase_start_30,
            "single_latent_path": (
                self._join_path(ep_dir, latent_root, self.single_latent_name)
                if self.single_latent_name
                else None
            ),
            "view_latent_paths": (
                [self._join_path(ep_dir, latent_root, name) for name in self.view_latent_names]
                if self.view_latent_names
                else None
            ),
            "left_latent_path": self._join_path(ep_dir, latent_root, self.left_latent_name),
            "right_latent_path": self._join_path(ep_dir, latent_root, self.right_latent_name),
            "hand_path": self._join_path(ep_dir, "hand_qpos.pt"),
            "endpose_path": self._join_path(ep_dir, "arm_endpose.pt"),
            "hand_action_path": self._join_path(ep_dir, "hand_action.pt"),
            "arm_action_path": self._join_path(ep_dir, "arm_action.pt"),
            "arm_action_endpose_path": self._join_path(ep_dir, "arm_action_endpose.pt"),
            "tjwj_path": self._join_path(ep_dir, self.tjwj_subpath) if self.tjwj_subpath else None,
            "tactile_path": self._join_path(ep_dir, "tactile_raw.pt"),
            "language_path": self._join_path(ep_dir, self.language_subpath),
        }

    def _scan_episodes(self) -> List[Dict[str, Any]]:
        if is_remote(self._as_path_str(self.dataset_dir)):
            return self._scan_remote_episodes()

        episodes: List[Dict[str, Any]] = []
        base = Path(self.dataset_dir)
        for hand_path in sorted(self.dataset_dir.rglob("hand_qpos.pt")):
            ep_dir = hand_path.parent
            if self._skip_path(ep_dir):
                continue
            try:
                task_name = ep_dir.parent.relative_to(base).as_posix()
            except ValueError:
                task_name = ep_dir.parent.name
            if not self._task_allowed(task_name):
                continue
            has_tjwj = bool(self.tjwj_subpath) and (ep_dir / self.tjwj_subpath).exists()
            required = []
            if not self._episode_has_any_configured_latents(ep_dir):
                required.append(f"{ep_dir}/{self.latent_root}/<configured phase latents>")
            if not has_tjwj:
                required.extend([hand_path, ep_dir / "arm_endpose.pt"])
            if not self.action_from_state:
                # True action: hand_action + arm_action_endpose (EE), not joint-deg arm_action.
                required.extend([ep_dir / "hand_action.pt", ep_dir / "arm_action_endpose.pt"])
            if self.tactile_enabled:
                required.append(ep_dir / "tactile_raw.pt")
            missing = []
            for p in required:
                if isinstance(p, str):
                    missing.append(p)
                elif not p.exists():
                    missing.append(str(p))
            if not self._language_present(ep_dir):
                missing.append(f"{ep_dir / self.language_subpath} (no language .pt)")
            if missing:
                logger.warning("Skipping %s due to missing files: %s", ep_dir, missing[:3])
                continue
            episodes.append(self._episode_record(ep_dir, task_name, ep_dir.name))
        return episodes

    def _scan_remote_episodes(self, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Recursively discover episodes under a remote root.

        Supports nesting deeper than root/<task>/<episode> (e.g.
        root/<group>/<task>/<episode>) so task_name patterns may be multi-segment
        relative paths like "lingqiaoshou/cut_asparagus..." or "human_hand/clean...".
        This mirrors the local rglob scanner's multi-segment task naming.
        """
        episodes: List[Dict[str, Any]] = []
        root = self._as_path_str(self.dataset_dir)
        root_prefix = root.rstrip("/") + "/"

        raw_patterns = (
            self.task_name if isinstance(self.task_name, (list, tuple))
            else ([self.task_name] if self.task_name else [])
        )
        pattern_segs = [
            [s for s in str(p).strip("/").replace("%", "*").split("/") if s]
            for p in raw_patterns
        ]

        def _rel(path) -> str:
            s = self._as_path_str(path)
            return s[len(root_prefix):].strip("/") if s.startswith(root_prefix) else self._basename(path)

        def _prefix_could_match(rel: str) -> bool:
            # True if `rel` is a path-prefix of some task pattern (so its subtree may match).
            if not pattern_segs:
                return True
            rel_segs = [s for s in rel.split("/") if s]
            for pat in pattern_segs:
                if len(rel_segs) <= len(pat) and all(
                    fnmatch.fnmatchcase(r, p) for r, p in zip(rel_segs, pat)
                ):
                    return True
            return False

        def _is_episode_dir(d) -> bool:
            return self._episode_has_any_configured_latents(d)

        def _record(ep_dir, task_name: str) -> None:
            has_tjwj = bool(self.tjwj_subpath) and self._exists(self._join_path(ep_dir, self.tjwj_subpath))
            required = []
            if not self._episode_has_any_configured_latents(ep_dir):
                required.append(f"{ep_dir}/{self.latent_root}/<configured phase latents>")
            if not has_tjwj:
                required.extend([
                    self._join_path(ep_dir, "hand_qpos.pt"),
                    self._join_path(ep_dir, "arm_endpose.pt"),
                ])
            if not self.action_from_state:
                required.extend([
                    self._join_path(ep_dir, "hand_action.pt"),
                    self._join_path(ep_dir, "arm_action_endpose.pt"),
                ])
            if self.tactile_enabled:
                required.append(self._join_path(ep_dir, "tactile_raw.pt"))
            missing = []
            for p in required:
                if isinstance(p, str) and "<configured phase latents>" in p:
                    missing.append(p)
                elif not self._exists(p):
                    missing.append(str(p))
            if not self._language_present(ep_dir):
                missing.append(f"{self._join_path(ep_dir, self.language_subpath)} (no language .pt)")
            if missing:
                logger.warning("Skipping %s due to missing files: %s", ep_dir, missing[:3])
                return
            episodes.append(self._episode_record(ep_dir, task_name, self._basename(ep_dir)))

        def _walk(cur_dir, depth: int) -> None:
            cur_rel = _rel(cur_dir)
            children = [c for c in smart_listdir_dirs(cur_dir) if not self._skip_path(c)]
            if not children:
                return
            # Episode dirs are conventionally named "episode*"; fall back to probing
            # the first child for the latent marker when the convention doesn't hold.
            ep_children = [c for c in children if self._basename(c).lower().startswith("episode")]
            if not ep_children and cur_rel and _is_episode_dir(children[0]):
                ep_children = children
            if ep_children:
                task_name = cur_rel or self._basename(cur_dir)
                if self._task_allowed(task_name):
                    for ep in ep_children:
                        _record(ep, task_name)
                return
            if depth >= max_depth:
                return
            for child in children:
                if _prefix_could_match(_rel(child)):
                    _walk(child, depth + 1)

        _walk(root, 1)
        return episodes

    def __len__(self) -> int:
        # The random path oversamples a virtual epoch. Episode mode addresses
        # each (episode, phase) exactly once, so index and length must line up
        # with what get_sample_lengths() reports to the packing sampler.
        if self.sequence_mode == "episode":
            return len(self.episodes)
        return max(1, len(self.episodes) * 10)

    def _init_qpos_normalization_stats(self) -> None:
        norm_mode = self.qpos_normalization
        if norm_mode not in {"mean_std", "min_max", "quantile"}:
            raise ValueError(
                f"Unsupported qpos_normalization={norm_mode!r}; "
                "expected mean_std, min_max, or quantile"
            )

        qpos_stats = self.action_stats.get("normalization", {}).get("qpos", {})
        summary = self.action_stats.get("summary", {})
        self.qpos_mean = None
        self.qpos_std = None
        self.qpos_min = None
        self.qpos_range = None

        if norm_mode == "mean_std":
            mean = qpos_stats.get("mean")
            std = qpos_stats.get("std")
            if mean is None or std is None:
                raise ValueError("wj_stat.json missing normalization.qpos mean/std for mean_std")
            self.qpos_mean = torch.tensor(mean, dtype=torch.float32)
            self.qpos_std = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
            if self.qpos_mean.shape[0] != 40 or self.qpos_std.shape[0] != 40:
                raise ValueError("wj_stat.json qpos mean/std must be 40D")
            return

        low_key, high_key = ("min", "max") if norm_mode == "min_max" else ("q01", "q99")
        low = qpos_stats.get(low_key)
        high = qpos_stats.get(high_key)
        if norm_mode == "min_max":
            if low is None:
                low = summary.get("qpos_min")
            if high is None:
                high = summary.get("qpos_max")
        if low is None or high is None:
            raise ValueError(
                f"wj_stat.json missing qpos {low_key}/{high_key} for {norm_mode}; "
                "re-run data/tianji_wuji/analyze_tianji_wuji_qpos.py"
            )
        self.qpos_min = torch.tensor(low, dtype=torch.float32)
        high_t = torch.tensor(high, dtype=torch.float32)
        self.qpos_range = (high_t - self.qpos_min).clamp(min=1e-6)
        if self.qpos_min.shape[0] != 40 or self.qpos_range.shape[0] != 40:
            raise ValueError(f"wj_stat.json qpos {low_key}/{high_key} must be 40D")

    def _normalize_qpos40(self, left_qpos: torch.Tensor, right_qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qpos40 = torch.cat([left_qpos, right_qpos], dim=-1).float()
        if self.qpos_normalization == "mean_std":
            qpos40 = (qpos40 - self.qpos_mean) / self.qpos_std
        elif self.qpos_normalization == "min_max":
            qpos40 = (qpos40 - self.qpos_min) / self.qpos_range
        else:
            qpos40 = torch.maximum(torch.minimum(qpos40, self.qpos_min + self.qpos_range), self.qpos_min)
            qpos40 = (qpos40 - self.qpos_min) / self.qpos_range
        return qpos40[:, :20], qpos40[:, 20:]

    def _action_downsample_stride_for_task(self, task_name: Optional[str] = None) -> int:
        if (
            self.adaptive_raw_hz_from_task_name
            and _task_name_implies_native_30hz(task_name)
        ):
            return 1
        return int(self.action_downsample_stride)

    def _tactile_downsample_stride_for_task(self, task_name: Optional[str] = None) -> int:
        if (
            self.adaptive_raw_hz_from_task_name
            and _task_name_implies_native_30hz(task_name)
        ):
            return 1
        return int(self.tactile_downsample_stride)

    def _downsample_action_sequence(
        self, sequence: torch.Tensor, task_name: Optional[str] = None
    ) -> torch.Tensor:
        stride = self._action_downsample_stride_for_task(task_name)
        if stride <= 1:
            return sequence
        return sequence[::stride].contiguous()

    def _load_replay_validity(
        self,
        ep: Dict[str, Any],
        *,
        expected_len: int,
    ) -> torch.Tensor:
        """Load replay-valid action rows without changing the timeline."""
        if not self.mask_invalid_replay:
            return torch.ones(expected_len, dtype=torch.bool)

        source_keys = (
            ("hand_path", "endpose_path")
            if self.action_from_state
            else ("hand_action_path", "arm_action_endpose_path")
        )
        task_name = ep.get("task_name")
        valid = torch.ones(expected_len, dtype=torch.bool)
        for key in source_keys:
            path = ep.get(key)
            if not path or not self._exists(path):
                continue
            payload = self._load_pt(path)
            if not isinstance(payload, dict):
                continue
            replay_valid = payload.get("replay_valid_flags")
            if replay_valid is None:
                continue
            source_valid = torch.as_tensor(replay_valid, dtype=torch.bool).flatten()
            source_valid = self._downsample_action_sequence(
                source_valid[:, None],
                task_name=task_name,
            ).squeeze(-1)
            if source_valid.shape[0] < expected_len:
                raise ValueError(
                    f"{key} replay_valid_flags has {source_valid.shape[0]} rows "
                    f"after downsampling, expected at least {expected_len}"
                )
            valid &= source_valid[:expected_len]
        return valid

    def _downsample_tactile_sequence(
        self, sequence: torch.Tensor, task_name: Optional[str] = None
    ) -> torch.Tensor:
        stride = self._tactile_downsample_stride_for_task(task_name)
        if stride <= 1:
            return sequence
        return sequence[::stride].contiguous()

    @staticmethod
    def _wrist7_to_wrist9(wrist7: torch.Tensor) -> torch.Tensor:
        xyz = wrist7[:, :3].float()
        rot6d = _quat_xyzw_to_rot6d(wrist7[:, 3:7].float())
        return torch.cat([xyz, rot6d], dim=-1)

    def _load_action58_from_tjwj(self, ep: Dict[str, Any]) -> torch.Tensor:
        tjwj_path = ep.get("tjwj_path")
        if not tjwj_path:
            raise FileNotFoundError("tjwj_path is missing")
        task_name = ep.get("task_name") if isinstance(ep, dict) else None
        raw = self._load_pt(tjwj_path)

        if isinstance(raw, dict):
            if "action" in raw and isinstance(raw["action"], torch.Tensor):
                raw = raw["action"]
            elif "action_xyzw" in raw and isinstance(raw["action_xyzw"], torch.Tensor):
                raw = raw["action_xyzw"]
            elif all(k in raw for k in ("left_qpos20", "right_qpos20", "left_wrist_xyz_xyzw", "right_wrist_xyz_xyzw")):
                left_q = _as_float_tensor(raw["left_qpos20"])
                right_q = _as_float_tensor(raw["right_qpos20"])
                left_w = _as_float_tensor(raw["left_wrist_xyz_xyzw"])
                right_w = _as_float_tensor(raw["right_wrist_xyz_xyzw"])
                n = min(left_q.shape[0], right_q.shape[0], left_w.shape[0], right_w.shape[0])
                left_q, right_q = self._normalize_qpos40(left_q[:n, :20], right_q[:n, :20])
                left_w9 = self._wrist7_to_wrist9(left_w[:n, :7])
                right_w9 = self._wrist7_to_wrist9(right_w[:n, :7])
                return self._downsample_action_sequence(
                    torch.cat([left_q, left_w9, right_q, right_w9], dim=-1),
                    task_name=task_name,
                )
            else:
                raise ValueError(f"Unsupported tjwj dict schema keys: {list(raw.keys())[:10]}")

        raw = _as_float_tensor(raw)
        if raw.ndim != 2:
            raise ValueError(f"tjwj tensor must be [T,D], got {tuple(raw.shape)}")
        if raw.shape[1] == 54:
            left_q = raw[:, :20]
            left_w7 = raw[:, 20:27]
            right_q = raw[:, 27:47]
            right_w7 = raw[:, 47:54]
            left_q, right_q = self._normalize_qpos40(left_q, right_q)
            left_w9 = self._wrist7_to_wrist9(left_w7)
            right_w9 = self._wrist7_to_wrist9(right_w7)
            return self._downsample_action_sequence(
                torch.cat([left_q, left_w9, right_q, right_w9], dim=-1),
                task_name=task_name,
            )
        if raw.shape[1] == 58:
            left_q = raw[:, :20]
            left_w9 = raw[:, 20:29]
            right_q = raw[:, 29:49]
            right_w9 = raw[:, 49:58]
            left_q, right_q = self._normalize_qpos40(left_q, right_q)
            return self._downsample_action_sequence(
                torch.cat([left_q, left_w9, right_q, right_w9], dim=-1),
                task_name=task_name,
            )
        raise ValueError(f"Unsupported tjwj tensor dim: {raw.shape[1]}")

    def _load_action58_abs(self, ep: Dict[str, Any]) -> torch.Tensor:
        hand_path = ep.get("hand_path")
        endpose_path = ep.get("endpose_path")
        task_name = ep.get("task_name") if isinstance(ep, dict) else None
        if hand_path and endpose_path and self._exists(hand_path) and self._exists(endpose_path):
            hand = self._load_pt(hand_path)
            endpose = self._load_pt(endpose_path)
            left_qpos = _as_float_tensor(hand["left"])
            right_qpos = _as_float_tensor(hand["right"])
            left_wrist_raw = _as_float_tensor(endpose["left"])
            right_wrist_raw = _as_float_tensor(endpose["right"])
            lengths = {
                "hand_qpos.left": left_qpos.shape[0],
                "hand_qpos.right": right_qpos.shape[0],
                "arm_endpose.left": left_wrist_raw.shape[0],
                "arm_endpose.right": right_wrist_raw.shape[0],
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(f"State sequence lengths must match: {lengths}")
            left_qpos, right_qpos = self._normalize_qpos40(left_qpos, right_qpos)
            left_wrist = _endpose_to_wrist9(left_wrist_raw)
            right_wrist = _endpose_to_wrist9(right_wrist_raw)
            return self._downsample_action_sequence(
                torch.cat([left_qpos, left_wrist, right_qpos, right_wrist], dim=-1),
                task_name=task_name,
            )

        tjwj_path = ep.get("tjwj_path")
        if tjwj_path and self._exists(tjwj_path):
            return self._load_action58_from_tjwj(ep)
        raise FileNotFoundError("qpos20_rel6d requires hand_qpos.pt+arm_endpose.pt or tjwj.pt")

    @staticmethod
    def _looks_like_endpose7(wrist7: torch.Tensor) -> bool:
        """True if 7D rows look like xyz(m)+quat, not joint targets in degrees."""
        if wrist7.ndim != 2 or wrist7.shape[-1] < 7:
            return False
        quat = wrist7[:, 3:7].float()
        norms = quat.norm(dim=-1)
        quat_ok = bool(((norms - 1.0).abs() <= 0.15).float().mean().item() >= 0.9)
        xyz_max = float(wrist7[:, :3].float().abs().max().item())
        return quat_ok and xyz_max < 5.0

    def _load_true_action58_abs(self, ep: Dict[str, Any]) -> torch.Tensor:
        """Load commanded action for qpos20_rel6d.

        Hand dims: hand_action.pt (same joint space as hand_qpos).
        Wrist dims: arm_action_endpose.pt (EE xyz+quat), not joint-deg arm_action
        and not state-side arm_endpose.
        """
        hand = self._load_pt(ep["hand_action_path"])
        arm_ee = self._load_pt(ep["arm_action_endpose_path"])
        left_qpos = _as_float_tensor(hand["left"])
        right_qpos = _as_float_tensor(hand["right"])
        left_wrist_raw = _as_float_tensor(arm_ee["left"])
        right_wrist_raw = _as_float_tensor(arm_ee["right"])
        lengths = {
            "hand_action.left": left_qpos.shape[0],
            "hand_action.right": right_qpos.shape[0],
            "arm_action_endpose.left": left_wrist_raw.shape[0],
            "arm_action_endpose.right": right_wrist_raw.shape[0],
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Action sequence lengths must match: {lengths}")
        if not (
            self._looks_like_endpose7(left_wrist_raw)
            and self._looks_like_endpose7(right_wrist_raw)
        ):
            raise ValueError(
                "arm_action_endpose.pt must be left/right (T,7) xyz+quat; "
                f"got left={tuple(left_wrist_raw.shape)} right={tuple(right_wrist_raw.shape)}"
            )
        if not self._logged_arm_action_endpose:
            logger.info(
                "action_from_state=false: loading wrist EE from arm_action_endpose.pt "
                "(hand from hand_action.pt); example lengths=%s",
                lengths,
            )
            self._logged_arm_action_endpose = True

        left_qpos, right_qpos = self._normalize_qpos40(left_qpos, right_qpos)
        left_wrist = _endpose_to_wrist9(left_wrist_raw)
        right_wrist = _endpose_to_wrist9(right_wrist_raw)
        task_name = ep.get("task_name") if isinstance(ep, dict) else None
        return self._downsample_action_sequence(
            torch.cat([left_qpos, left_wrist, right_qpos, right_wrist], dim=-1),
            task_name=task_name,
        )

    def _normalize_tactile(self, raw: torch.Tensor, side: str,
                           task_name: Optional[str] = None) -> torch.Tensor:
        raw = raw.float()
        baseline, valid_mask, polarity, tactile_max, tactile_min = self._resolve_baseline(task_name)
        if side == "left":
            mask = valid_mask[:768]
            base = baseline[:768] if baseline is not None else None
        else:
            mask = valid_mask[768:]
            base = baseline[768:] if baseline is not None else None
        if base is None:
            # 缺底噪归一化基线 → 直接报错（不静默 raw 直通）。触觉归一化必须有 per-taxel
            # baseline；未配置 tactile_baseline_per_source/tactile_baseline_name 时无法把
            # raw（dex raw/2035 float 或 human uint16）映到统一 [0,1] signal，静默直通会让
            # 错误尺度的触觉无声进入模型。
            raise ValueError(
                f"tactile normalization requires a per-taxel baseline, but none was "
                f"resolved for task_name={task_name!r} (base is None). Configure "
                f"tactile_baseline_per_source (or tactile_baseline_name) so this source's "
                f"tactile_raw.pt is normalized in its native space."
            )
        # Native-space per-taxel normalization: `raw` and `base` live in the SAME
        # source-native space (dex raw/2035 float [0,1] with tactile_max=1.0;
        # human uint16 ADC with tactile_max=2100). No deadzone, no reconstruction —
        # the polarity formula is scale-invariant. See tactile_baseline.normalize_polarity_native.
        from data.tianji_wuji.tactile_baseline import normalize_polarity_native
        normed = normalize_polarity_native(
            raw, base, polarity=polarity,
            tactile_max=tactile_max, tactile_min=tactile_min, valid_mask=mask,
        )
        return normed

    @staticmethod
    def _pick_tactile_lr_raw(tactile: Dict[str, Any]) -> Tuple[Any, Any]:
        """Read native-space left/right from tactile_raw.pt.

        Native space is per-source: dex stores raw_int/2035 float [0,1], human
        stores uint16 ADC [0,2100]. Per-source normalization (polarity + per-taxel
        baseline + tactile_max/tactile_min) happens in `_normalize_tactile` via
        `normalize_polarity_native`, which is scale-invariant across these spaces.
        """
        if "left" not in tactile or "right" not in tactile:
            raise KeyError(
                "tactile_raw.pt must contain left/right; "
                f"keys={list(tactile.keys())}"
            )
        return tactile["left"], tactile["right"]

    def _load_tactile1536(self, ep: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tactile = self._load_pt(ep["tactile_path"])
        left_raw, right_raw = self._pick_tactile_lr_raw(tactile)
        # task_name drives per-source baseline selection (human vs dex-hand etc.).
        task_name = ep.get("task_name") if isinstance(ep, dict) else None
        left = self._normalize_tactile(_as_float_tensor(left_raw), "left", task_name=task_name)
        right = self._normalize_tactile(_as_float_tensor(right_raw), "right", task_name=task_name)
        n = min(left.shape[0], right.shape[0])
        merged = torch.cat([left[:n], right[:n]], dim=-1)
        # Output mask is the EFFECTIVE mask for this episode (default ∩ source-specific),
        # not just the global one — keeps downstream loss masking source-aware.
        _, effective_mask, _, _, _ = self._resolve_baseline(task_name)
        mask = effective_mask.float().unsqueeze(0).expand(n, -1).clone()
        return (
            self._downsample_tactile_sequence(merged, task_name=task_name),
            self._downsample_tactile_sequence(mask, task_name=task_name),
        )

    def _latent_to_action_range(self, latent_idx: int, frame_stride: int, phase_start_30: int = 0) -> Tuple[int, int]:
        r = self.action_upsample_ratio
        if latent_idx == 0:
            return phase_start_30 * r, phase_start_30 * r + r
        apl = 4 * frame_stride * r
        start = (phase_start_30 + 1) * r + (latent_idx - 1) * apl
        return start, start + apl

    def _gather_per_latent(
        self,
        sequence: torch.Tensor,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        per_latent_len: int,
        return_valid_mask: bool = False,
    ) -> torch.Tensor:
        chunks = []
        masks = []
        n = sequence.shape[0]
        for i in range(num_latents):
            abs_lat = latent_start + i
            start, end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            start = min(start, n - 1)
            end = min(end, n)
            chunk = sequence[start:end]
            # Rows past the end of the recorded trajectory are repeats of the
            # last real one. They have to stay out of the loss, so the caller
            # can ask which rows were actually recorded.
            real_len = int(chunk.shape[0])
            if chunk.shape[0] < per_latent_len:
                chunk = torch.cat([chunk, chunk[-1:].expand(per_latent_len - chunk.shape[0], -1)], dim=0)
            else:
                chunk = chunk[:per_latent_len]
            chunks.append(chunk)
            if return_valid_mask:
                mask = torch.zeros(per_latent_len, dtype=torch.bool)
                mask[: min(real_len, per_latent_len)] = True
                masks.append(mask)
        gathered = torch.cat(chunks, dim=0)
        if return_valid_mask:
            return gathered, torch.cat(masks, dim=0)
        return gathered

    def _gather_states(
        self,
        sequence: torch.Tensor,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
    ) -> torch.Tensor:
        states = []
        n = sequence.shape[0]
        for i in range(num_latents):
            abs_lat = latent_start + i
            start, _ = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            state_idx = max(min(start, n - 1) - 1, 0)
            states.append(sequence[state_idx])
        return torch.stack(states, dim=0)

    def _load_latent(self, path) -> tuple[torch.Tensor, int, int]:
        data = self._load_pt(path)
        if isinstance(data, torch.Tensor):
            latents = data.float()
            return latents, latents.shape[1], 6
        latents = data["latents"].float()
        num_latents = int(data.get("num_latents", latents.shape[1]))
        frame_stride = int(data.get("frame_stride", 6))
        return latents, num_latents, frame_stride

    def _load_language_embedding(self, ep: Dict[str, Any]) -> torch.Tensor:
        candidates = self._resolve_language_candidates(ep)
        if not candidates:
            raise FileNotFoundError(f"No language .pt candidates under: {ep.get('language_path')}")

        last_exc: Optional[Exception] = None
        for cand in random.sample(candidates, k=len(candidates)):
            try:
                emb = self._load_pt(cand)
                if isinstance(emb, list):
                    emb = emb[random.randint(0, len(emb) - 1)]
                if isinstance(emb, torch.Tensor) and emb.dim() == 3:
                    emb = emb[random.randint(0, emb.shape[0] - 1)]
                if not isinstance(emb, torch.Tensor):
                    raise TypeError(f"Unexpected language embedding type: {type(emb)}")
                # Skip pooled 1D vectors (e.g. environment*.pt shape [4096]); collate expects [T, D].
                if emb.dim() != 2 or emb.shape[-1] < 2:
                    raise TypeError(
                        f"Skip non-sequence language embedding from {cand}: shape={tuple(emb.shape)}"
                    )
                return emb.float()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        raise RuntimeError(
            f"Failed to load language embedding from candidates ({len(candidates)}): {candidates[:3]}"
        ) from last_exc

    def _resolve_language_candidates(self, ep: Dict[str, Any]) -> List[str]:
        episode_path = str(ep.get("episode_path", ""))
        cached = self._language_candidates_cache.get(episode_path)
        if cached is not None:
            return cached

        language_path = self._as_path_str(ep.get("language_path"))
        candidates: List[str] = []
        if language_path:
            if language_path.endswith(".pt"):
                candidates = [language_path]
            else:
                # Directory mode: randomly pick one .pt under text_features/UMT5 (video-pretrain style).
                # Exclude pooled environment*.pt ([4096]) which break sequence collate ([T, 4096]).
                candidates = [
                    c for c in smart_glob_pt(language_path)
                    if not Path(c).name.startswith("environment")
                ]

        self._language_candidates_cache[episode_path] = candidates
        return candidates

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
    # Episode mode: whole-trajectory samples with a deterministic length
    # ------------------------------------------------------------------

    def _episode_window(self, total_latents: int) -> int:
        """Latents kept for one episode-mode sample.

        One latent is observed and the rest are predicted in fixed_chunk_size
        blocks, so a trailing partial block is dropped. The length index and
        __getitem__ must agree on this number, otherwise a packed batch can
        exceed the budget the sampler planned for.
        """
        obs = 1
        candidate = int(total_latents)
        if (
            self.sequence_max_latents is not None
            and candidate > self.sequence_max_latents
        ):
            if self.sequence_overflow_policy in {"error", "skip"}:
                raise ValueError(
                    f"episode has {total_latents} latents, over "
                    f"sequence_max_latents={self.sequence_max_latents} with "
                    f"sequence_overflow_policy={self.sequence_overflow_policy}"
                )
            candidate = self.sequence_max_latents
        chunk = max(1, int(self.bootstrap_chunk_size))
        return obs + ((candidate - obs) // chunk) * chunk

    def _episode_total_latents(self, ep: Dict[str, Any]) -> int:
        """Latent count of one (episode, phase) record, before chunk trimming."""
        if self.use_single_latent:
            latents, video_num, _ = self._load_latent(ep["single_latent_path"])
            return int(min(video_num, latents.shape[1]))
        paths = ep.get("view_latent_paths") or [
            ep["left_latent_path"],
            ep["right_latent_path"],
        ]
        loaded = [self._load_latent(p) for p in paths]
        return int(min([*(n for _, n, _ in loaded), *(t.shape[1] for t, _, _ in loaded)]))

    def _length_index_signature(self) -> str:
        payload = json.dumps(
            {
                "dataset_dir": str(self.dataset_dir),
                "episodes": [str(e.get("episode_name")) for e in self.episodes],
                "latent_root": str(self.latent_root),
                "single_latent_name": str(self.single_latent_name),
                "chunk": int(self.bootstrap_chunk_size),
                "max_latents": self.sequence_max_latents,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _build_episode_lengths(self) -> List[int]:
        """Scan every (episode, phase) once for its trimmed latent count.

        Reading a few thousand latent files costs about a minute, so the result
        is cached against a signature of the episode list. A stale cache is
        ignored rather than trusted, since a wrong length silently produces
        over-budget batches.
        """
        cache_path = getattr(self, "length_index_cache", None)
        signature = self._length_index_signature()
        if cache_path:
            try:
                with open(cache_path, "r") as handle:
                    cached = json.load(handle)
                if cached.get("signature") == signature:
                    lengths = [int(v) for v in cached["lengths"]]
                    if len(lengths) == len(self.episodes):
                        logger.info(
                            "TianjiWujiDataset: loaded length index for %d samples from %s",
                            len(lengths),
                            cache_path,
                        )
                        return lengths
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Length index cache unusable (%s): %s", cache_path, exc)

        logger.info(
            "TianjiWujiDataset: scanning %d (episode, phase) records for latent lengths",
            len(self.episodes),
        )
        lengths: List[int] = []
        for i, ep in enumerate(self.episodes):
            try:
                total = self._episode_total_latents(ep)
                lengths.append(self._episode_window(total) if total > 0 else 0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Length scan failed for %s: %s", ep.get("episode_path"), exc
                )
                lengths.append(0)
            if (i + 1) % 500 == 0:
                logger.info("  scanned %d/%d", i + 1, len(self.episodes))

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as handle:
                    json.dump({"signature": signature, "lengths": lengths}, handle)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write length index cache %s: %s", cache_path, exc)
        return lengths

    def get_sample_lengths(self) -> Optional[List[int]]:
        """Per-index latent counts, consumed by data.length_bucketing.

        Only episode mode has one: the random path draws a fresh episode per
        call, so an index carries no length.
        """
        if self.sequence_mode != "episode":
            return None
        if self._sample_lengths is None:
            self._sample_lengths = self._build_episode_lengths()
        return self._sample_lengths

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        episode_mode = self.sequence_mode == "episode"
        if episode_mode:
            if not self.episodes:
                raise RuntimeError("Tianji/Wuji episode-mode dataset is empty")
            episode_idx = int(idx)
            if episode_idx < 0:
                episode_idx += len(self.episodes)
            if episode_idx < 0 or episode_idx >= len(self.episodes):
                raise IndexError(
                    f"Episode index {idx} outside [0, {len(self.episodes)})"
                )
            fixed_episode = self.episodes[episode_idx]
        W = self.max_latent_window
        for _ in range(1 if episode_mode else 8):
            ep = fixed_episode if episode_mode else random.choice(self.episodes)
            try:
                if self.use_single_latent:
                    video_latents, video_num, frame_stride = self._load_latent(ep["single_latent_path"])
                    total_latents = min(video_num, video_latents.shape[1])
                else:
                    latent_paths = ep.get("view_latent_paths") or [
                        ep["left_latent_path"],
                        ep["right_latent_path"],
                    ]
                    loaded_views = [self._load_latent(path) for path in latent_paths]
                    view_latents = [item[0] for item in loaded_views]
                    view_nums = [item[1] for item in loaded_views]
                    view_strides = [item[2] for item in loaded_views]
                    total_latents = min([*view_nums, *[latents.shape[1] for latents in view_latents]])
                    frame_stride = min(view_strides)
                if total_latents <= 0:
                    if episode_mode:
                        raise ValueError(
                            f"episode has {total_latents} latents: "
                            f"{ep.get('episode_path')}"
                        )
                    continue

                if episode_mode:
                    # The whole trajectory is the sample: no crop, no padding.
                    # W is trimmed so the latents after the single observed one
                    # form complete fixed_chunk_size prediction blocks, which is
                    # also the length _episode_num_latents() reports.
                    W = self._episode_window(total_latents)
                    start = 0
                    valid_len = W
                    if self.use_single_latent:
                        latents = video_latents[:, :W]
                    else:
                        view_crops = [latents[:, :W] for latents in view_latents]
                elif total_latents > W:
                    start = random.randint(0, total_latents - W)
                    valid_len = W
                    if self.use_single_latent:
                        latents = video_latents[:, start:start + W]
                    else:
                        view_crops = [latents[:, start:start + W] for latents in view_latents]
                elif total_latents < W:
                    start = 0
                    valid_len = total_latents
                    pad = W - total_latents
                    if self.use_single_latent:
                        latents = F.pad(video_latents[:, :total_latents], (0, 0, 0, 0, 0, pad))
                    else:
                        view_crops = [
                            F.pad(latents[:, :total_latents], (0, 0, 0, 0, 0, pad))
                            for latents in view_latents
                        ]
                else:
                    start = 0
                    valid_len = W
                    if self.use_single_latent:
                        latents = video_latents[:, :W]
                    else:
                        view_crops = [latents[:, :W] for latents in view_latents]

                actual_latents = min(total_latents - start, W)
                if not self.use_single_latent:
                    if self.view_merge == "concat_width":
                        latents = torch.cat(view_crops, dim=-1)  # [C,W,H,V*W_lat]
                    else:
                        latents = torch.stack(view_crops, dim=0)  # [V,C,W,H,W]

                state58_abs = self._load_action58_abs(ep)
                action58_abs = (
                    state58_abs
                    if self.action_from_state
                    else self._load_true_action58_abs(ep)
                )
                if action58_abs.shape[0] != state58_abs.shape[0]:
                    raise ValueError(
                        "Action/state sequence lengths must match: "
                        f"{action58_abs.shape[0]} != {state58_abs.shape[0]}"
                    )
                replay_valid_full = self._load_replay_validity(
                    ep,
                    expected_len=action58_abs.shape[0],
                )
                phase_start_30 = int(ep.get("phase_start_30", 0))
                actions, source_action_valid_mask = self._gather_per_latent(
                    action58_abs,
                    start,
                    actual_latents,
                    frame_stride,
                    phase_start_30=phase_start_30,
                    per_latent_len=self.actions_per_latent,
                    return_valid_mask=True,
                )
                replay_action_valid_mask = self._gather_per_latent(
                    replay_valid_full[:, None].float(),
                    start,
                    actual_latents,
                    frame_stride,
                    phase_start_30=phase_start_30,
                    per_latent_len=self.actions_per_latent,
                ).squeeze(-1).bool()
                source_action_valid_mask &= replay_action_valid_mask
                states = self._gather_states(
                    state58_abs, start, actual_latents, frame_stride, phase_start_30=phase_start_30
                )

                target_action_len = W * self.actions_per_latent
                if actions.shape[0] < target_action_len:
                    pad_len = target_action_len - actions.shape[0]
                    actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
                    source_action_valid_mask = torch.cat(
                        [source_action_valid_mask, torch.zeros(pad_len, dtype=torch.bool)],
                        dim=0,
                    )
                if states.shape[0] < W:
                    states = torch.cat([states, states[-1:].expand(W - states.shape[0], -1)], dim=0)

                extra_action_sequences: Dict[str, torch.Tensor] = {}
                # Carried through the bootstrap helper so the mask keeps lining
                # up with the action rows if that path rewrites them.
                extra_action_sequences["source_action_valid_mask"] = source_action_valid_mask

                if self.tactile_enabled:
                    tactile, tactile_mask_full = self._load_tactile1536(ep)
                    tactile_seq = self._gather_per_latent(
                        tactile,
                        start,
                        actual_latents,
                        frame_stride,
                        phase_start_30=phase_start_30,
                        per_latent_len=self.actions_per_latent,
                    )
                    tactile_mask = self._gather_per_latent(
                        tactile_mask_full,
                        start,
                        actual_latents,
                        frame_stride,
                        phase_start_30=phase_start_30,
                        per_latent_len=self.actions_per_latent,
                    )
                    if tactile_seq.shape[0] < target_action_len:
                        pad = target_action_len - tactile_seq.shape[0]
                        tactile_seq = torch.cat([tactile_seq, tactile_seq[-1:].expand(pad, -1)], dim=0)
                        tactile_mask = torch.cat([tactile_mask, tactile_mask[-1:].expand(pad, -1)], dim=0)
                    extra_action_sequences["tactile"] = tactile_seq
                    extra_action_sequences["tactile_mask"] = tactile_mask

                values = self._build_value_targets(start, W, total_latents)
                latents, actions, states, values, valid_len, extra_action_sequences = maybe_bootstrap_start_window_sample(
                    latents,
                    actions,
                    valid_len,
                    start,
                    actions_per_latent=self.actions_per_latent,
                    states=states,
                    values=values,
                    chunk_size=self.bootstrap_chunk_size,
                    enabled=self.bootstrap_start_window,
                    extra_action_sequences=extra_action_sequences,
                )
                latents = self._apply_view_dropout(latents)

                source_action_valid_mask = extra_action_sequences.pop(
                    "source_action_valid_mask"
                ).bool()
                video_valid_mask = torch.arange(W, dtype=torch.long) < int(valid_len)
                action_valid_mask = (
                    video_valid_mask.repeat_interleave(self.actions_per_latent)
                    & source_action_valid_mask
                )

                sample: Dict[str, Any] = {
                    "latents": latents,
                    # Time dim is second-to-last three; works for [C,T,H,W] and
                    # the multi-view [V,C,T,H,W] layout alike.
                    "anchor_latents": latents[..., :1, :, :].clone().contiguous(),
                    "actions": actions,
                    "states": states,
                    "language_embedding": self._load_language_embedding(ep),
                    "valid_len": int(valid_len),
                    "video_valid_mask": video_valid_mask,
                    "action_valid_mask": action_valid_mask,
                    "latent_start": int(start),
                    "is_task_irrelevant": int(self.treat_as_irrelevant),
                    "is_success": int(not self.treat_as_irrelevant),
                    "source": self._resolve_source(ep.get("task_name")),
                }
                if episode_mode:
                    sample["episode_num_latents"] = int(total_latents)
                    sample["sequence_num_latents"] = int(W)
                    sample["sequence_chunk_trimmed_latents"] = int(total_latents - W)
                if extra_action_sequences:
                    sample.update(extra_action_sequences)
                if values is not None:
                    sample["values"] = values
                return sample
            except Exception as exc:  # noqa: BLE001
                # The random path can shrug a bad episode off and draw another.
                # Episode mode cannot: the sampler already reserved budget for
                # this index, so a silent None would corrupt the packed batch.
                if episode_mode:
                    raise
                logger.warning("Tianji/Wuji sample error (%s): %s", ep.get("episode_path"), exc)
        if episode_mode:
            raise RuntimeError(
                f"Episode-mode sample {idx} produced nothing "
                f"({fixed_episode.get('episode_path')})"
            )
        return None
