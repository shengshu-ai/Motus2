# Stage2 pretrain dataset for latent-space AR training.
#
# This is intentionally separate from data/realworld so stage2-specific logic
# (value targets, task-irrelevant mixing, cached episode indexing) stays
# isolated.

import fnmatch
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from .action_pad import pad_or_trim_chunk
from data.window_start import (
    ORDINAL_STRIDE,
    estimate_window_start_count,
    pack_index,
    select_window_start,
    unpack_index,
    window_start_count,
)
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
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
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from utils.common import maybe_bootstrap_start_window_sample
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


def _as_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _quat_xyzw_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat.float(), dim=-1)
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
    xyz_m = endpose[:, :3].float() / 1000.0
    rot6d = _quat_xyzw_to_rot6d(endpose[:, 3:7])
    return torch.cat([xyz_m, rot6d], dim=-1)


def _wrist7_to_wrist9(wrist7: torch.Tensor) -> torch.Tensor:
    xyz = wrist7[:, :3].float()
    # Keep tjwj wrist xyz in its original unit for pretrain.
    rot6d = _quat_xyzw_to_rot6d(wrist7[:, 3:7].float())
    return torch.cat([xyz, rot6d], dim=-1)


def _assert_finite_tensor(t: torch.Tensor, tensor_name: str, source_path: Optional[str] = None) -> None:
    bad = ~torch.isfinite(t)
    if not bad.any():
        return
    bad_count = int(bad.sum().item())
    total = int(t.numel())
    extra = f" from {source_path}" if source_path else ""
    raise ValueError(f"{tensor_name}{extra} contains non-finite values ({bad_count}/{total})")


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


def _resolve_rate_config(
    *,
    video_hz: Optional[int],
    action_cfg: Any,
    raw_action_hz: Optional[int],
    train_action_hz: Optional[int],
    action_upsample_ratio: int,
) -> Tuple[int, int, int, int, int]:
    video_hz = int(video_hz or 30)
    raw_action_hz = _rate_cfg_value(action_cfg, "raw_hz", raw_action_hz)
    train_action_hz = _rate_cfg_value(action_cfg, "train_hz", train_action_hz)

    if train_action_hz is None:
        train_action_hz = video_hz * int(action_upsample_ratio)
    if raw_action_hz is None:
        raw_action_hz = train_action_hz
    if min(video_hz, raw_action_hz, train_action_hz) <= 0:
        raise ValueError("pretrain video/action rates must be positive")
    # Pretrain exports already share the video's action timeline.  In
    # particular tjwj.pt is commonly a bare tensor without capture-rate
    # metadata: silently applying a config-derived stride here can halve a
    # 30 Hz sequence while latent indexing continues to assume 30 Hz.
    # Convert genuinely different rates offline, with timestamp evidence;
    # Tianji/Wuji/Sharpa capture loaders retain their separate rate handling.
    if raw_action_hz != train_action_hz:
        raise ValueError(
            "PretrainDataset does not resample action streams: "
            f"raw_action_hz={raw_action_hz} must equal train_action_hz={train_action_hz}. "
            "For the 30 Hz pretrain exports set action.raw_hz=30 and "
            "action.train_hz=30; convert other capture rates offline."
        )
    if train_action_hz % video_hz != 0:
        raise ValueError(f"train_action_hz must be divisible by video_hz, got {train_action_hz}/{video_hz}")

    action_downsample = raw_action_hz // train_action_hz
    derived_upsample = train_action_hz // video_hz
    return video_hz, raw_action_hz, train_action_hz, action_downsample, derived_upsample


def _normalize_prefixes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        values = [value]
    else:
        values = list(value)
    out: List[str] = []
    seen = set()
    for item in values:
        if item is None:
            continue
        prefix = _normalize_remote_path(str(item).rstrip("/"))
        if prefix and prefix not in seen:
            out.append(prefix)
            seen.add(prefix)
    return out


def _column_alias_expr_for_sql(preferred: str, available_columns: List[str], aliases: List[str]) -> Optional[str]:
    for col in aliases:
        if col in available_columns:
            return col if col == preferred else f"{col} AS {preferred}"
    return None


def _clickhouse_query_json_each_row(sql: str, clickhouse: Dict[str, Any]) -> List[dict]:
    """Run a ClickHouse HTTP query and parse JSONEachRow without extra deps."""
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
    data = sql.encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    if username or password:
        import base64

        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured internal CH endpoint
        payload = response.read().decode("utf-8")
    rows: List[dict] = []
    for line in payload.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class PretrainDataset(data.Dataset):
    """Stage2 video+action pretrain dataset."""

    @staticmethod
    def _normalize_jsonl_paths(jsonl_path: Optional[Any]) -> List[str]:
        if jsonl_path is None:
            return []
        if isinstance(jsonl_path, (str, Path)):
            values = [jsonl_path]
        else:
            try:
                values = list(jsonl_path)
            except TypeError as exc:
                raise TypeError("jsonl_path must be a path or a sequence of paths") from exc

        paths: List[str] = []
        seen = set()
        for value in values:
            if value is None or not str(value).strip():
                continue
            path = str(value)
            if path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def __init__(
        self,
        jsonl_path: Optional[str | List[str]] = None,
        latent_root: str = "video_features/wan2.2",
        latent_file_name: str = "video_3v_VAE.pt",
        latent_subpath: Optional[str] = None,
        lang_root: Optional[str] = "text_features/UMT5",
        lang_subpath: Optional[str] = "text_features/UMT5/environment_robot_behavior.pt",
        lang_dir: str = "umt5_wan",
        robot: Optional[str] = None,
        clickhouse: Optional[Dict[str, Any]] = None,
        use_index_cache: bool = True,
        dataset_dir=None,
        episode_path_prefix: Optional[str] = None,
        episode_path_prefixes: Optional[List[str]] = None,
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
        video_hz: int = 30,
        video_action_freq_ratio: int = 6,
        action_mode: str = "joint",
        use_state: bool = True,
        val: bool = False,
        samples_per_episode: int = 10,
        window_start_sampling: bool = False,
        cache_dir: Optional[str] = None,
        action_upsample_ratio: int = 1,
        action: Optional[Dict[str, Any]] = None,
        raw_action_hz: Optional[int] = None,
        train_action_hz: Optional[int] = None,
        value_target: Optional[str] = None,
        progress_c: float = 0.01,
        treat_as_irrelevant: bool = False,
        empty_lang_path: Optional[str] = None,
        eef_quat_order: str = "xyzw",
        dataset_include: Optional[List[str]] = None,
        dataset_exclude: Optional[List[str]] = None,
        dataset_allowlist: Optional[List[str]] = None,
        dataset_denylist: Optional[List[str]] = None,
        latent_file_name_by_dataset: Optional[Dict[str, str]] = None,
        stereo_left_latent_name: Optional[str] = None,
        stereo_right_latent_name: Optional[str] = None,
        stereo_latent_name_pairs: Optional[List[List[str]]] = None,
        stereo_latent_sources_path: Optional[str] = None,
        stereo_latent_sources: Optional[Any] = None,
        stereo_latent_name_pairs_by_dataset: Optional[Dict[str, Any]] = None,
        action_stats_path: Optional[str] = None,
        action_stats_by_dataset: Optional[Dict[str, str]] = None,
        wrist_xyz_per_dataset: bool = False,
        qpos_normalization: str = "mean_std",
        bootstrap_start_window: bool = False,
        bootstrap_chunk_size: int = 2,
        min_total_latent_frames: Optional[int] = None,
        source_weighted_sampling: bool = False,
        wrist_xyz_abs_max: Optional[float] = None,
        wrist_xyz_relative_max: Optional[float] = None,
        action_abs_max: Optional[float] = None,
        refine_spatial_bucket_keys: bool = False,
        freeze_val_indices: bool = False,
        exclude_explicit_fail: bool = False,
        mask_invalid_hand_tracking: bool = False,
        invalid_origin_wrist_marker: bool = False,
        **kwargs,
    ):
        self.jsonl_paths = self._normalize_jsonl_paths(jsonl_path)
        self.jsonl_path = (
            None
            if not self.jsonl_paths
            else self.jsonl_paths[0]
            if len(self.jsonl_paths) == 1
            else list(self.jsonl_paths)
        )
        self.latent_root = latent_root.strip("/") if latent_root else None
        self.latent_file_name = latent_file_name
        self.latent_file_name_by_dataset = {
            str(k): str(v) for k, v in (latent_file_name_by_dataset or {}).items()
        }
        self.stereo_left_latent_name = stereo_left_latent_name
        self.stereo_right_latent_name = stereo_right_latent_name
        self.stereo_latent_name_pairs = [
            (str(pair[0]), str(pair[1]))
            for pair in (stereo_latent_name_pairs or [])
            if len(pair) == 2
        ]
        self.stereo_latent_sources_path = stereo_latent_sources_path
        file_sources = self._load_source_definitions(stereo_latent_sources_path)
        stereo_sources = self._normalize_stereo_latent_sources(file_sources)
        stereo_sources.extend(self._normalize_stereo_latent_sources(stereo_latent_sources))
        self.stereo_latent_sources = stereo_sources
        # Sampling ratios apply to every source, including explicit mono readers.
        # Keep them separate from the complete stereo pairs used for expansion.
        self.source_sampling_sources = self._normalize_source_sampling_sources(file_sources)
        self.source_sampling_sources.extend(
            self._normalize_source_sampling_sources(stereo_latent_sources)
        )
        self.source_weighted_sampling = bool(source_weighted_sampling)
        self._episode_sample_weights = array("d")
        self.stereo_latent_name_pairs_by_dataset = {}
        for pattern, pairs in (stereo_latent_name_pairs_by_dataset or {}).items():
            normalized_pairs: List[Tuple[str, str]] = []
            if (
                isinstance(pairs, (list, tuple))
                and len(pairs) == 2
                and all(isinstance(x, str) for x in pairs)
            ):
                normalized_pairs.append((str(pairs[0]), str(pairs[1])))
            else:
                for pair in pairs or []:
                    if len(pair) == 2:
                        normalized_pairs.append((str(pair[0]), str(pair[1])))
            if normalized_pairs:
                self.stereo_latent_name_pairs_by_dataset[str(pattern)] = normalized_pairs
        self.latent_subpath = latent_subpath
        self.lang_root = lang_root.strip("/") if lang_root else None
        self.lang_subpath = lang_subpath.strip("/") if lang_subpath else None
        self.lang_dir = lang_dir
        self.robot_filter = robot
        self.clickhouse = _mapping_to_plain_dict(clickhouse)
        self.use_index_cache = bool(use_index_cache)
        self.episode_index_source = str(episode_index_source or "auto").lower()
        prefix_values = _normalize_prefixes(episode_path_prefixes)
        if not prefix_values:
            prefix_values = _normalize_prefixes(episode_path_prefix)
        if not prefix_values:
            prefix_values = _normalize_prefixes(
                source.get("source")
                for source in self.stereo_latent_sources
                if source.get("source")
            )
        self.episode_path_prefixes = prefix_values
        self.episode_path_prefix = prefix_values[0] if len(prefix_values) == 1 else None

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
        self.mask_invalid_hand_tracking = bool(mask_invalid_hand_tracking)
        if self.mask_invalid_hand_tracking and action_mode not in {"qpos20_rel6d", "relqpos20_rel6d"}:
            raise ValueError("mask_invalid_hand_tracking requires a qpos20/rot6d action mode")
        # A wrist exactly at the origin with an identity rotation is a lost-track
        # sentinel in some exports, not an observation (see _hand_tracking_valid_mask).
        self.invalid_origin_wrist_marker = bool(invalid_origin_wrist_marker)
        if self.invalid_origin_wrist_marker and not self.mask_invalid_hand_tracking:
            raise ValueError("invalid_origin_wrist_marker requires mask_invalid_hand_tracking")
        self.use_state = use_state
        self.val = val
        self.samples_per_episode = max(1, int(samples_per_episode))
        # When the batch sampler assigns a concrete latent window to every slot
        # (``window_start_sampling``), each dataset index carries an ordinal in
        # its high bits and the episode visit budget is expressed by the source
        # weighting instead of by ``samples_per_episode``.  See data/window_start.py.
        self.window_start_sampling = bool(window_start_sampling) and not val
        self.cache_dir = cache_dir
        (
            self.video_hz,
            self.raw_action_hz,
            self.train_action_hz,
            self.action_downsample_stride,
            self.action_upsample_ratio,
        ) = _resolve_rate_config(
            video_hz=video_hz,
            action_cfg=action,
            raw_action_hz=raw_action_hz,
            train_action_hz=train_action_hz,
            action_upsample_ratio=int(action_upsample_ratio),
        )
        # Seconds of video one latent covers.  Index rows that carry only
        # ``duration_sec`` (wuji2) are converted with it when estimating how many
        # window starts an episode offers.
        self.latent_seconds = 4.0 * float(video_action_freq_ratio) / float(max(1, self.video_hz))
        self.value_target = value_target
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.empty_lang_path = empty_lang_path
        self.eef_quat_order = eef_quat_order
        self.dataset_include = list(dataset_include or dataset_allowlist or [])
        self.dataset_exclude = list(dataset_exclude or dataset_denylist or [])
        self.bootstrap_start_window = bool(bootstrap_start_window)
        self.bootstrap_chunk_size = int(bootstrap_chunk_size)
        self.min_total_latent_frames = (
            int(min_total_latent_frames)
            if min_total_latent_frames is not None and int(min_total_latent_frames) > 0
            else None
        )
        self.wrist_xyz_abs_max = (
            float(wrist_xyz_abs_max)
            if wrist_xyz_abs_max is not None and float(wrist_xyz_abs_max) > 0
            else None
        )
        self.wrist_xyz_relative_max = None
        if wrist_xyz_relative_max is not None:
            threshold = float(wrist_xyz_relative_max)
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError('wrist_xyz_relative_max must be finite and positive, or null')
            self.wrist_xyz_relative_max = threshold
        self.action_abs_max = (
            float(action_abs_max)
            if action_abs_max is not None and float(action_abs_max) > 0
            else None
        )
        self.refine_spatial_bucket_keys = bool(refine_spatial_bucket_keys)
        self.freeze_val_indices = bool(freeze_val_indices) and bool(val)
        self.exclude_explicit_fail = bool(exclude_explicit_fail)
        self._explicit_fail_seen: Dict[str, int] = {}
        self._explicit_fail_excluded: Dict[str, int] = {}
        self.index_cache_path = self._resolve_index_cache_path(index_cache_path)
        self.metadata_cache_path = (
            str(Path(metadata_cache_path).expanduser()) if metadata_cache_path else None
        )

        self.action_stats = self._load_json(action_stats_path) if action_stats_path else None
        self.qpos_normalization = str(qpos_normalization or "mean_std")
        self.qpos_mean = None
        self.qpos_std = None
        self.qpos_min = None
        self.qpos_range = None
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            if self.action_stats is None:
                raise ValueError(f"{self.action_mode} pretrain requires action_stats_path")
            if self.qpos_normalization not in {"mean_std", "min_max", "quantile"}:
                raise ValueError(f"Unsupported qpos_normalization: {self.qpos_normalization}")
            qpos_stats = self.action_stats.get("qpos", self.action_stats)
            if self.qpos_normalization == "mean_std":
                self.qpos_mean = torch.tensor(qpos_stats["mean"], dtype=torch.float32)
                self.qpos_std = torch.tensor(qpos_stats["std"], dtype=torch.float32).clamp(min=1e-6)
                if self.qpos_mean.shape[0] != 40 or self.qpos_std.shape[0] != 40:
                    raise ValueError(f"{self.action_mode} qpos mean/std must be 40D")
            else:
                low_key, high_key = ("min", "max") if self.qpos_normalization == "min_max" else ("q01", "q99")
                qpos_low = torch.tensor(qpos_stats[low_key], dtype=torch.float32)
                qpos_high = torch.tensor(qpos_stats[high_key], dtype=torch.float32)
                self.qpos_min = qpos_low
                self.qpos_range = (qpos_high - qpos_low).clamp(min=1e-6)
                if self.qpos_min.shape[0] != 40 or self.qpos_range.shape[0] != 40:
                    raise ValueError(f"{self.action_mode} qpos {low_key}/{high_key} must be 40D")

        # Per-dataset qpos stats: {dataset_name: stats_path}. Sources whose hand
        # geometry differs (e.g. wuji1 vs wuji2) are normalized against their own
        # envelope instead of one global one. Falls back to the global stats above.
        self.qpos_min_by_dataset: Dict[str, torch.Tensor] = {}
        self.qpos_range_by_dataset: Dict[str, torch.Tensor] = {}
        self.qpos_mean_by_dataset: Dict[str, torch.Tensor] = {}
        self.qpos_std_by_dataset: Dict[str, torch.Tensor] = {}
        for _ds_name, _ds_path in (action_stats_by_dataset or {}).items():
            if not _ds_path:
                continue
            _st = self._load_json(str(_ds_path))
            _q = _st.get("qpos", _st)
            if self.qpos_normalization == "mean_std":
                self.qpos_mean_by_dataset[str(_ds_name)] = torch.tensor(_q["mean"], dtype=torch.float32)
                self.qpos_std_by_dataset[str(_ds_name)] = torch.tensor(_q["std"], dtype=torch.float32).clamp(min=1e-6)
            else:
                _lk, _hk = ("min", "max") if self.qpos_normalization == "min_max" else ("q01", "q99")
                _lo = torch.tensor(_q[_lk], dtype=torch.float32)
                _hi = torch.tensor(_q[_hk], dtype=torch.float32)
                self.qpos_min_by_dataset[str(_ds_name)] = _lo
                self.qpos_range_by_dataset[str(_ds_name)] = (_hi - _lo).clamp(min=1e-6)
        if self.qpos_min_by_dataset:
            logger.info(
                "PretrainDataset: per-dataset qpos stats for %s",
                sorted(self.qpos_min_by_dataset) + sorted(self.qpos_mean_by_dataset),
            )

        self.wrist_xyz_per_dataset = bool(wrist_xyz_per_dataset)

        if self.eef_quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported eef_quat_order: {self.eef_quat_order}")
        if self.episode_index_source not in {"auto", "clickhouse", "dataset_dir", "jsonl"}:
            raise ValueError(f"Unsupported episode_index_source: {self.episode_index_source}")
        if self.exclude_explicit_fail and self._discovery_source() != "jsonl":
            raise ValueError("exclude_explicit_fail currently requires episode_index_source=jsonl")
        if not self.jsonl_paths and not dataset_dir and not self.clickhouse:
            raise ValueError("Must provide one of: jsonl_path, dataset_dir, or clickhouse")
        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name")

        self.actions_per_latent = 4 * video_action_freq_ratio * self.action_upsample_ratio
        self.episode_files: List[Dict[str, Any]] = []
        self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.task_weights: Dict[str, float] = {}
        self.total_episodes = 0
        self._language_candidates_cache: Dict[str, List[str]] = {}
        self._episode_metadata_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._missing_language_roots = set()
        self._metadata_db_conn: Optional[sqlite3.Connection] = None
        self._metadata_db_pid: Optional[int] = None
        self._metadata_db_columns: Optional[set[str]] = None
        self._gripper_stats_cache: Dict[str, Dict[str, Any]] = {}
        self._dropped_episode_ids = set()
        self._wrist_xyz_filter_skips = 0
        self._bucket_to_episode_indices: Dict[str, List[int]] = {}
        self._frozen_val_indices: Optional[List[int]] = None
        self._freeze_build_mode = False
        self._load_episodes_distributed()
        self._assign_resolution_buckets()
        if self.refine_spatial_bucket_keys:
            self._refine_spatial_bucket_keys_from_latents()
        if self.freeze_val_indices:
            self._build_frozen_val_indices()

        logger.info(
            "PretrainDataset: task_mode=%s, episodes=%d, samples_per_episode=%d, "
            "actions_per_latent=%d, upsample=%d, action_hz=%d->%d/downsample=%d, "
            "action_mode=%s, irrelevant=%s, value_target=%s, "
            "latent_max_attempts=%s, index_cache=%s, metadata_cache=%s, clickhouse=%s",
            task_mode,
            self.total_episodes,
            self.samples_per_episode,
            self.actions_per_latent,
            self.action_upsample_ratio,
            self.raw_action_hz,
            self.train_action_hz,
            self.action_downsample_stride,
            action_mode,
            treat_as_irrelevant,
            value_target,
            self.latent_max_attempts,
            self.index_cache_path,
            self.metadata_cache_path,
            bool(self.clickhouse),
        )

    @staticmethod
    def _resolution_bucket_from_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        for pattern in (
            r"(?<!\d)(\d{3,4})[_xX-](\d{3,4})(?!\d)",
            r"(?<!\d)(\d{3,4})[pP][_-]?(\d{3,4})(?!\d)",
        ):
            matches = re.findall(pattern, text)
            if matches:
                h, w = matches[-1]
                return f"{int(h)}x{int(w)}"
        return None

    @staticmethod
    def _load_json(path: str | Path) -> dict:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_source_definitions(path: Optional[str]) -> Any:
        if not path:
            return []
        config_path = Path(path).expanduser()
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[2] / config_path
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_sources = payload.get("stereo_latent_sources") or payload.get("sources") or []
        else:
            raw_sources = payload
        return raw_sources

    @classmethod
    def _load_stereo_latent_sources(cls, path: Optional[str]) -> List[Dict[str, Any]]:
        return cls._normalize_stereo_latent_sources(cls._load_source_definitions(path))

    def _assign_resolution_buckets(self) -> None:
        """Attach a best-effort resolution bucket to every indexed episode.

        This is intentionally metadata/path based so multi-resolution batching
        does not need to open remote latent .pt files during dataset init.
        """
        for ep in self.episode_files:
            bucket = (
                ep.get("bucket_key")
                or ep.get("resolution_bucket")
                or ep.get("latent_resolution")
                or self._resolution_bucket_from_text(ep.get("latent_file_name"))
                or self._resolution_bucket_from_text(ep.get("latent_path"))
                or self._resolution_bucket_from_text(ep.get("latent_root"))
                or self._resolution_bucket_from_text(ep.get("episode_path"))
                or self._resolution_bucket_from_text(ep.get("dataset_name"))
                or "unknown"
            )
            ep["bucket_key"] = str(bucket)
        self._bucket_to_episode_indices = {}
        self._source_bucket_to_episode_indices = {}
        for episode_idx, ep in enumerate(self.episode_files):
            key = str(ep.get("bucket_key") or "unknown")
            self._bucket_to_episode_indices.setdefault(key, []).append(episode_idx)
            if self.source_weighted_sampling:
                source_key = (key, self._sampling_source_identity(ep))
                self._source_bucket_to_episode_indices.setdefault(source_key, []).append(episode_idx)

    def _refine_spatial_bucket_keys_from_latents(self) -> None:
        """Append latent H×W to bucket_key so SameResolutionBatchSampler never mixes spatial grids.

        Metadata buckets (from paths/filenames) can group episodes whose VAE latents
        have different patch grids; collate stacks ``latents`` and requires matching H/W.
        """
        ok = 0
        failed = 0
        for ep in self.episode_files:
            base = str(ep.get("bucket_key") or "unknown")
            try:
                candidates = self._resolve_latent_candidates(ep)
                if not candidates:
                    ep["bucket_key"] = f"{base}|latent_missing"
                    failed += 1
                    continue
                cand = sorted(candidates, key=self._latent_candidate_key)[0]
                # Use val-style phase pick so probe is deterministic (not random.choice).
                was_val = self.val
                self.val = True
                try:
                    latents, _, _ = self._load_latent_candidate(cand)
                finally:
                    self.val = was_val
                h, w = int(latents.shape[-2]), int(latents.shape[-1])
                ep["bucket_key"] = f"{base}|{h}x{w}"
                ok += 1
            except Exception as exc:
                logger.warning(
                    "refine_spatial_bucket_keys: episode=%s dataset=%s: %s",
                    ep.get("episode_name"),
                    ep.get("dataset_name"),
                    exc,
                )
                ep["bucket_key"] = f"{base}|latent_err"
                failed += 1

        self._bucket_to_episode_indices = {}
        for episode_idx, ep in enumerate(self.episode_files):
            key = str(ep.get("bucket_key") or "unknown")
            self._bucket_to_episode_indices.setdefault(key, []).append(episode_idx)

        logger.info(
            "refine_spatial_bucket_keys: ok=%d failed=%d distinct_buckets=%d",
            ok,
            failed,
            len(self._bucket_to_episode_indices),
        )

    def get_bucket_key(self, idx: int) -> str:
        if not self.episode_files:
            return "empty"
        if self.val and self._frozen_val_indices and not self._freeze_build_mode:
            idx = self._frozen_val_indices[idx % len(self._frozen_val_indices)]
        ep = self.episode_files[idx % self.total_episodes]
        return str(ep.get("bucket_key") or "unknown")

    def get_sample_weight(self, idx: int) -> float:
        """Return a source-normalized episode weight for replacement sampling."""
        if not self.source_weighted_sampling or not self._episode_sample_weights:
            return 1.0
        return max(0.0, float(self._episode_sample_weights[idx % self.total_episodes]))

    def _build_frozen_val_indices(self) -> None:
        if not self.val:
            return
        total = self.total_episodes * self.samples_per_episode
        if total <= 0:
            self._frozen_val_indices = []
            return
        logger.info("freeze_val_indices: building fixed validation index set from %d candidates", total)
        kept: List[int] = []
        self._freeze_build_mode = True
        prev_dropped = self._dropped_episode_ids
        self._dropped_episode_ids = set()
        try:
            for idx in range(total):
                sample = self.__getitem__(idx)
                if sample is not None:
                    kept.append(idx)
        finally:
            self._freeze_build_mode = False
            self._dropped_episode_ids = prev_dropped
        self._frozen_val_indices = kept
        logger.info(
            "freeze_val_indices: kept=%d dropped=%d",
            len(kept),
            max(0, total - len(kept)),
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
        self._metadata_db_columns = None
        return conn

    def _metadata_columns(self, conn: sqlite3.Connection) -> set[str]:
        if self._metadata_db_columns is None:
            try:
                rows = conn.execute("PRAGMA table_info(episode_metadata)").fetchall()
                self._metadata_db_columns = {str(row["name"]) for row in rows}
            except sqlite3.Error:
                self._metadata_db_columns = set()
        return self._metadata_db_columns

    def _lookup_episode_metadata(self, ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._episode_metadata_key(ep)
        if key is None or not self.metadata_cache_path:
            return None
        if key in self._episode_metadata_cache:
            return self._episode_metadata_cache[key]

        conn = self._metadata_connection()
        if conn is None:
            return None

        columns = self._metadata_columns(conn)
        select_cols = [
            "phase_dirs_json",
            "lang_files_json",
            "latent_scanned",
            "lang_scanned",
        ]
        optional_cols = [
            "phase_latent_frames_json",
            "total_latent_frames",
            "duration_sec",
        ]
        select_cols.extend(col for col in optional_cols if col in columns)
        try:
            row = conn.execute(
                f"""
                SELECT {", ".join(select_cols)}
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
        if "phase_latent_frames_json" in columns:
            metadata["phase_latent_frames"] = json.loads(row["phase_latent_frames_json"] or "[]")
        if "total_latent_frames" in columns:
            value = row["total_latent_frames"]
            metadata["total_latent_frames"] = int(value) if value is not None else None
        if "duration_sec" in columns:
            value = row["duration_sec"]
            metadata["duration_sec"] = float(value) if value is not None else None
        self._episode_metadata_cache[key] = metadata
        return metadata

    def _dataset_name_allowed(self, dataset_name: Optional[str]) -> bool:
        if dataset_name is None:
            return not self.dataset_include
        if self.dataset_include:
            return any(self._name_matches(dataset_name, pat) for pat in self.dataset_include)
        if any(self._name_matches(dataset_name, pat) for pat in self.dataset_exclude):
            return False
        return True

    def _latent_file_name_for_dataset(self, dataset_name: Optional[str]) -> str:
        if not dataset_name or not self.latent_file_name_by_dataset:
            return self.latent_file_name
        for pattern, file_name in self.latent_file_name_by_dataset.items():
            if self._name_matches(str(dataset_name), pattern):
                return file_name
        return self.latent_file_name

    @classmethod
    def _normalize_stereo_latent_sources(cls, raw_sources: Any) -> List[Dict[str, Any]]:
        return cls._normalize_source_specs(raw_sources, require_stereo=True)

    @classmethod
    def _normalize_source_sampling_sources(cls, raw_sources: Any) -> List[Dict[str, Any]]:
        return cls._normalize_source_specs(raw_sources, require_stereo=False)

    @staticmethod
    def _normalize_source_specs(raw_sources: Any, *, require_stereo: bool) -> List[Dict[str, Any]]:
        if raw_sources is None:
            return []
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(raw_sources):
                raw_sources = OmegaConf.to_container(raw_sources, resolve=True)
        except Exception:
            pass
        if isinstance(raw_sources, dict):
            items = []
            for name, spec in raw_sources.items():
                if isinstance(spec, dict):
                    cur = dict(spec)
                    cur.setdefault("name", str(name))
                    items.append(cur)
        else:
            items = list(raw_sources)

        sources: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            left = item.get("left_latent_name") or item.get("left") or item.get("left_name")
            right = item.get("right_latent_name") or item.get("right") or item.get("right_name")
            if not left or not right:
                pair = item.get("latent_pair") or item.get("pair")
                if pair and len(pair) == 2:
                    left, right = pair[0], pair[1]
            if require_stereo and (not left or not right):
                continue
            source = item.get("source")
            name = item.get("name") or source or (f"{left}|{right}" if left and right else "")
            if not name:
                continue

            match_patterns: List[str] = []
            raw_match = item.get("match_patterns") or item.get("matches") or item.get("match")
            if raw_match is None:
                raw_match = []
            if isinstance(raw_match, str):
                match_patterns.append(raw_match)
            elif isinstance(raw_match, dict):
                for key in ("dataset", "dataset_name", "episode_path", "path"):
                    value = raw_match.get(key)
                    if value is None:
                        continue
                    if isinstance(value, str):
                        match_patterns.append(value)
                    else:
                        match_patterns.extend(str(v) for v in value)
            else:
                match_patterns.extend(str(v) for v in raw_match)
            source = item.get("source")
            if source:
                source_text = str(source).rstrip("/")
                source_core = _strip_remote_scheme(source_text)
                match_patterns.extend([
                    source_text,
                    _normalize_remote_path(source_text),
                    f"*{source_core}*",
                ])
            if item.get("episode_path_prefix"):
                match_patterns.append(str(item["episode_path_prefix"]))

            sources.append({
                "source": str(source).rstrip("/") if source else "",
                "name": str(name),
                "match_patterns": match_patterns,
                "left_latent_name": str(left) if left else None,
                "right_latent_name": str(right) if right else None,
                "resolution_bucket": str(item.get("resolution_bucket") or item.get("bucket_key") or ""),
                "ratio": item.get("ratio", item.get("target_hours")),
                # Sources sharing a group pool their ratios and are split by
                # available window starts instead of a per-source ratio.
                "sampling_group": (
                    str(item["sampling_group"]) if item.get("sampling_group") else None
                ),
            })
        return sources

    def _episode_match_texts(self, dataset_name: Optional[str], ep: Optional[Dict[str, Any]] = None) -> List[str]:
        texts: List[str] = []
        if dataset_name:
            texts.append(str(dataset_name))
        if ep:
            for key in ("dataset_name", "episode_path", "episode_storage_key", "latent_root"):
                value = ep.get(key)
                if value is not None:
                    text = str(value)
                    if text not in texts:
                        texts.append(text)
        return texts

    def _stereo_source_specs_for_dataset(
        self,
        dataset_name: Optional[str],
        ep: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        texts = self._episode_match_texts(dataset_name, ep)
        specs: List[Dict[str, Any]] = []
        for source in self.stereo_latent_sources:
            patterns = source.get("match_patterns") or []
            if patterns and not any(self._name_matches(text, pat) for text in texts for pat in patterns):
                continue
            specs.append(source)
        if specs:
            return specs

        for pattern, pairs in self.stereo_latent_name_pairs_by_dataset.items():
            if any(self._name_matches(text, pattern) for text in texts):
                return [
                    {
                        "name": f"{left_name}|{right_name}",
                        "left_latent_name": left_name,
                        "right_latent_name": right_name,
                        "resolution_bucket": "",
                    }
                    for left_name, right_name in pairs
                ]

        if self.stereo_latent_name_pairs:
            return [
                {
                    "name": f"{left_name}|{right_name}",
                    "left_latent_name": left_name,
                    "right_latent_name": right_name,
                    "resolution_bucket": "",
                }
                for left_name, right_name in self.stereo_latent_name_pairs
            ]
        if self.stereo_left_latent_name and self.stereo_right_latent_name:
            return [{
                "name": f"{self.stereo_left_latent_name}|{self.stereo_right_latent_name}",
                "left_latent_name": self.stereo_left_latent_name,
                "right_latent_name": self.stereo_right_latent_name,
                "resolution_bucket": "",
            }]
        return []

    def _stereo_pairs_for_dataset(
        self,
        dataset_name: Optional[str],
        ep: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        return [
            (str(spec["left_latent_name"]), str(spec["right_latent_name"]))
            for spec in self._stereo_source_specs_for_dataset(dataset_name, ep)
        ]

    def _expand_stereo_pair_episodes(
        self,
        ep: Dict[str, Any],
        dataset_name: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Create one indexed sample per configured stereo latent resolution.

        Same-resolution batching only works if an indexed sample has a fixed
        latent shape. Without this expansion, one episode could randomly load
        different stereo pairs across __getitem__ calls while keeping the same
        bucket key.
        """
        specs = self._stereo_source_specs_for_dataset(dataset_name, ep)
        if not specs:
            return [ep]

        expanded: List[Dict[str, Any]] = []
        for spec in specs:
            left_name = str(spec["left_latent_name"])
            right_name = str(spec["right_latent_name"])
            left_bucket = self._resolution_bucket_from_text(left_name)
            right_bucket = self._resolution_bucket_from_text(right_name)
            explicit_bucket = str(spec.get("resolution_bucket") or "")
            bucket_key = explicit_bucket or (left_bucket if left_bucket == right_bucket else (left_bucket or right_bucket))
            cur = dict(ep)
            cur.update({
                "stereo_source": str(spec.get("name") or f"{left_name}|{right_name}"),
                "stereo_left_latent_name": left_name,
                "stereo_right_latent_name": right_name,
                "stereo_pair_key": f"{left_name}|{right_name}",
                "latent_file_name": f"{left_name}|{right_name}",
                "resolution_bucket": bucket_key or cur.get("resolution_bucket"),
                "bucket_key": bucket_key or cur.get("bucket_key") or "unknown",
            })
            expanded.append(cur)
        return expanded

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
        if self.exclude_explicit_fail and rank == 0:
            logger.info(
                "Pretrain explicit-fail filter: raw_seen_by_source=%s "
                "excluded_by_source=%s retained_episodes=%d; "
                "unlabelled episodes are retained, not certified successful",
                self._explicit_fail_seen,
                self._explicit_fail_excluded,
                self.total_episodes,
            )
        self._log_episode_summary()

    def _resolve_index_cache_path(self, index_cache_path: Optional[str]) -> Optional[str]:
        if not self.use_index_cache:
            return None
        if not index_cache_path:
            return None
        return str(Path(index_cache_path).expanduser())

    def _index_cache_meta(self) -> Dict[str, Any]:
        ch_meta = None
        if self.clickhouse:
            ch_meta = {
                k: self.clickhouse.get(k)
                for k in ("host", "port", "username", "database", "table", "secure", "use_final")
                if k in self.clickhouse
            }
        return {
            "index_cache_version": INDEX_CACHE_VERSION,
            "episode_index_source": self.episode_index_source,
            "jsonl_path": self.jsonl_path,
            "dataset_dirs": [str(d) for d in self.dataset_dirs] if self.dataset_dirs is not None else None,
            "episode_path_prefix": self.episode_path_prefix,
            "episode_path_prefixes": list(self.episode_path_prefixes),
            "clickhouse": ch_meta,
            "catalog_path": self.catalog_path,
            "latent_dir": self.latent_dir,
            "latent_root": self.latent_root,
            "latent_file_name": self.latent_file_name,
            "latent_file_name_by_dataset": dict(sorted(self.latent_file_name_by_dataset.items())),
            "stereo_left_latent_name": self.stereo_left_latent_name,
            "stereo_right_latent_name": self.stereo_right_latent_name,
            "stereo_latent_name_pairs": self.stereo_latent_name_pairs,
            "stereo_latent_sources_path": self.stereo_latent_sources_path,
            "stereo_latent_sources": self.stereo_latent_sources,
            **({"source_sampling_sources": self.source_sampling_sources}
               if getattr(self, "source_sampling_sources", self.stereo_latent_sources)
               != self.stereo_latent_sources else {}),
            "stereo_latent_name_pairs_by_dataset": self.stereo_latent_name_pairs_by_dataset,
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
            "action_mode": self.action_mode,
            **({"exclude_explicit_fail": True} if self.exclude_explicit_fail else {}),
        }

    def _discovery_source(self) -> str:
        if self.episode_index_source == "auto":
            if self.clickhouse is not None:
                return "clickhouse"
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
            if self.exclude_explicit_fail:
                # Legacy lists discarded the input success/fail fields.
                return None
            payload = {
                "meta": self._index_cache_meta(),
                "episodes": payload,
                "scanned_datasets": [],
                "scanned_tasks": [],
                "complete": True,
            }

        cache_meta = payload.get("meta")
        if (
            cache_meta is None and self.exclude_explicit_fail
        ) or (cache_meta is not None and cache_meta != self._index_cache_meta()):
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

        if discovery_source == "clickhouse":
            episodes = self._load_episodes_from_clickhouse()
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
            if not self.jsonl_paths:
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

    def _episode_where_sql(self, path_col: Optional[str] = None) -> str:
        conditions: List[str] = []
        if self.episode_path_prefixes and path_col:
            prefix_terms = []
            for prefix in self.episode_path_prefixes:
                core = _strip_remote_scheme(prefix).replace("'", "''")
                prefix_terms.append(f"{path_col} LIKE '%{core}%'")
            conditions.append("(" + " OR ".join(prefix_terms) + ")")

        if self.robot_filter:
            robot_sql = self.robot_filter.replace("'", "''")
            conditions.append(f"(robot = '{robot_sql}' OR dataset = '{robot_sql}')")

        if self.task_mode == "single" and self.task_name:
            task_pat = self.task_name.replace("*", "%")
            task_sql = task_pat.replace("'", "''")
            conditions.append(f"task LIKE '{task_sql}'" if "%" in task_pat else f"task = '{task_sql}'")

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

        return (" WHERE " + " AND ".join(conditions)) if conditions else ""

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
        episodes: List[Dict[str, Any]] = []
        for jsonl_path in self.jsonl_paths:
            with open(jsonl_path) as f:
                entry_iter = (json.loads(line) for line in f if line.strip())
                for entry in entry_iter:
                    added = self._episode_from_jsonl_entry(entry)
                    if not added:
                        continue
                    episodes.extend(added)
                    if self.max_episodes is not None and len(episodes) >= self.max_episodes:
                        return episodes[: self.max_episodes]
        return episodes

    @staticmethod
    def _explicit_failure_reason(entry: Dict[str, Any]) -> Optional[str]:
        """Only explicit episode labels or fail directory segments prove failure.

        MAIN/sample_class is a loss-routing label, not task-outcome evidence.
        Natural-language tasks containing words such as 'failsafe' are retained.
        """
        true_values = {"true", "1", "1.0", "yes", "fail", "failed", "failure"}
        false_values = {"false", "0", "0.0", "no", "fail", "failed", "failure"}
        for key in ("fail", "is_fail", "failed", "is_failure", "is_fail_class"):
            value = entry.get(key)
            if value is not None and str(value).strip().casefold() in true_values:
                return key
        for key in ("success", "is_success", "task_success", "episode_success"):
            value = entry.get(key)
            if value is not None and str(value).strip().casefold() in false_values:
                return key
        for key in ("status", "outcome", "result", "episode_status", "sample_type", "source_type"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip().casefold() in {"fail", "failed", "failure"}:
                return key
        for key in ("episode_bos_path", "episode_path"):
            path = str(entry.get(key) or "")
            if any(part.casefold() in {"fail", "failed", "failure"} for part in path.split("/")):
                return key
        return None

    def _episode_from_jsonl_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.exclude_explicit_fail:
            source = str(entry.get("dataset") or entry.get("robot") or "unknown")
            self._explicit_fail_seen[source] = self._explicit_fail_seen.get(source, 0) + 1
            if self._explicit_failure_reason(entry) is not None:
                self._explicit_fail_excluded[source] = self._explicit_fail_excluded.get(source, 0) + 1
                return []
        if self.robot_filter and entry.get("robot") != self.robot_filter:
            return []

        total_latent_frames = entry.get("total_latent_frames")
        if (
            self.min_total_latent_frames is not None
            and total_latent_frames is not None
            and int(total_latent_frames) < self.min_total_latent_frames
        ):
            return []

        dataset_name = entry.get("dataset") or entry.get("robot")
        if not self._dataset_name_allowed(dataset_name):
            return []

        task_name = entry.get("task")
        if not task_name or not self._task_allowed(task_name):
            return []

        qpos_path = self._resolve_entry_action_path(entry)
        if qpos_path is None:
            return []

        base = self._normalize_bos(entry["episode_bos_path"].rstrip("/"))
        episode_name = entry.get("episode_id") or base.rsplit("/", 1)[-1]
        ep = {
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
        }
        if entry.get("action_schema") == "handson-stage2/raw54-v1":
            # Opt-in source contract: never infer a measured-state stream from commands.
            ep.update({key: entry[key] for key in (
                "action_schema", "handson_action_frames", "phase_start_30",
                "latent_path", "lang_path")})
            ep["latent_root"] = None
            ep["lang_root"] = None
            ep["handson_language_path"] = entry["handson_language_path"]
            resolution = entry.get("resolution_bucket", "544x640")
            ep["bucket_key"] = f"handson|v1|{resolution}"
        if entry.get("total_latent_frames") is not None:
            ep["sc_total_latent_frames"] = int(entry["total_latent_frames"])
        if entry.get("duration_sec") is not None:
            ep["sc_duration_sec"] = float(entry["duration_sec"])
        if entry.get("valid_phase_dirs") is not None:
            ep["sc_valid_phase_dirs"] = list(entry["valid_phase_dirs"])
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            ep.update({
                "tjwj_path": self._normalize_bos(entry.get("tjwj_path") or smart_join(base, "tjwj.pt")),
                "hand_path": self._normalize_bos(entry.get("hand_path") or entry.get("hand_qpos_path") or smart_join(base, "hand_qpos.pt")),
                "qpos_path": self._normalize_bos(entry.get("hand_path") or entry.get("hand_qpos_path") or smart_join(base, "hand_qpos.pt")),
                "endpose_path": self._normalize_bos(entry.get("endpose_path") or smart_join(base, "arm_endpose.pt")),
            })
        if ep.get('action_schema') == 'handson-stage2/raw54-v1':
            return [ep]  # Global stereo presets must not duplicate or relabel this mono source.
        return self._expand_stereo_pair_episodes(ep, dataset_name)

    def _organize_episodes(self, episodes: List[Dict[str, Any]]):
        # Sampling is episode-uniform after filtering unless the opt-in source
        # weighting below is enabled; task buckets remain summaries/debugging.
        self.episode_files = list(episodes)
        self._rebuild_task_index()
        self._configure_source_sampling_weights()

    def _source_sampling_ratio_map(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for source in getattr(self, "source_sampling_sources", self.stereo_latent_sources):
            value = source.get("ratio")
            if value is None:
                continue
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                continue
            if ratio <= 0:
                continue
            name = str(source.get("name") or source.get("source") or "")
            if name:
                ratios[name] = ratio
        return ratios

    def _source_sampling_name(
        self,
        ep: Dict[str, Any],
        ratio_map: Dict[str, float],
    ) -> str:
        dataset_name = str(ep.get("dataset_name") or "")
        if dataset_name in ratio_map:
            return dataset_name

        texts = self._episode_match_texts(dataset_name, ep)
        for source in getattr(self, "source_sampling_sources", self.stereo_latent_sources):
            name = str(source.get("name") or source.get("source") or "")
            if name not in ratio_map:
                continue
            patterns = source.get("match_patterns") or []
            if any(self._name_matches(text, pattern) for text in texts for pattern in patterns):
                return name
        return dataset_name or "unknown"

    def _episode_start_budget(self, ep: Dict[str, Any]) -> int:
        """How many distinct latent windows this episode can offer.

        Sampling is normalized by this instead of by episode count so a source
        with short clips is not replayed many times over while a source with
        long clips is barely touched.  The value only has to be accurate enough
        to set relative budgets; the exact start count is recomputed from the
        loaded latent inside ``_select_latent_window``.
        """
        total_latents = ep.get("sc_total_latent_frames")
        if total_latents is None:
            duration = ep.get("sc_duration_sec")
            if duration is not None:
                try:
                    # Export duration is the normalized source frame count / video_hz.
                    # Mirror offline stride selection and repeat-last padding to
                    # 1+4k frames before the causal Wan VAE (temporal stride4).
                    video_hz = float(getattr(self, "video_hz", 30))
                    frame_stride = int(getattr(self, "video_action_freq_ratio", 6))
                    duration_seconds = float(duration)
                    if duration_seconds <= 0 or video_hz <= 0 or frame_stride <= 0:
                        raise ValueError("invalid duration/video_hz/frame_stride for latent budget")
                    source_frames = int(round(duration_seconds * video_hz))
                    if source_frames <= 0:
                        raise ValueError("duration has no source video frames")
                    sampled_frames = (source_frames + frame_stride - 1) // frame_stride
                    total_latents = 1 + (sampled_frames + 2) // 4
                except (TypeError, ValueError, OverflowError):
                    total_latents = None
        window = int(ep.get("window_size") or ep.get("sc_window_size") or self.max_latent_window)
        return estimate_window_start_count(total_latents, window)

    def _sampling_group_masses(
        self,
        ratio_map: Dict[str, float],
        source_start_mass: Dict[str, float],
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
        """Collapse sources that share a ``sampling_group`` into one pool.

        Grouping is what lets several small real-robot sources compete as a
        single mixture component: their configured ratios are summed and their
        start counts pooled, so the split inside the group follows available
        window starts instead of a hand-set per-source ratio.
        """
        group_of: Dict[str, str] = {}
        for source in self.source_sampling_sources:
            name = str(source.get("name") or source.get("source") or "")
            group = source.get("sampling_group")
            if name and group:
                group_of[name] = str(group)

        group_ratio: Dict[str, float] = {}
        group_start_mass: Dict[str, float] = {}
        for source, ratio in ratio_map.items():
            key = group_of.get(source, source)
            group_ratio[key] = group_ratio.get(key, 0.0) + float(ratio)
            group_start_mass[key] = group_start_mass.get(key, 0.0) + float(
                source_start_mass.get(source, 0.0)
            )
        return group_of, group_ratio, group_start_mass

    def _configure_source_sampling_weights(self) -> None:
        self._episode_sample_weights = array("d")
        if not self.source_weighted_sampling:
            return
        if not self.episode_files:
            raise ValueError("source_weighted_sampling requires a non-empty dataset")

        ratio_map = self._source_sampling_ratio_map()
        if not ratio_map:
            raise ValueError(
                "source_weighted_sampling requires positive ratio values in stereo_latent_sources"
            )

        # Constructed datasets always set this in __init__; getattr keeps
        # partial fixtures and subclasses on the legacy episode-count basis.
        window_start_sampling = bool(getattr(self, "window_start_sampling", False))
        source_counts: Dict[str, int] = {}
        source_start_mass: Dict[str, float] = {}
        episode_sources: List[str] = []
        episode_budgets: List[int] = []
        for ep in self.episode_files:
            source = self._source_sampling_name(ep, ratio_map)
            episode_sources.append(source)
            source_counts[source] = source_counts.get(source, 0) + 1
            if window_start_sampling:
                budget = self._episode_start_budget(ep)
                episode_budgets.append(budget)
                source_start_mass[source] = source_start_mass.get(source, 0.0) + budget

        missing = sorted(source for source in source_counts if source not in ratio_map)
        if missing:
            raise ValueError(
                "source_weighted_sampling has episodes without configured ratios: "
                + ", ".join(missing)
            )

        if window_start_sampling:
            # Start-budget basis: a source's share is split across its episodes
            # in proportion to how many distinct latent windows each offers, so
            # a short clip is not replayed many times over while a long clip is
            # barely touched.
            group_of, group_ratio, group_start_mass = self._sampling_group_masses(
                ratio_map, source_start_mass
            )
            self._sampling_group_of = group_of
            weights = array("d")
            for source, budget in zip(episode_sources, episode_budgets):
                key = group_of.get(source, source)
                mass = group_start_mass.get(key, 0.0)
                ratio = group_ratio.get(key, 0.0)
                if mass <= 0 or ratio <= 0:
                    weights.append(0.0)
                    continue
                # Mass conservation: sum over a source of weight == ratio.
                weights.append(ratio * float(budget) / mass)
            self._episode_sample_weights = weights
        else:
            # Legacy episode-count basis, kept verbatim for every config that
            # did not opt into window-start sampling.
            self._sampling_group_of = {}
            self._episode_sample_weights = array(
                "d",
                (
                    ratio_map[source] / source_counts[source]
                    for source in episode_sources
                ),
            )

        present_ratio_sum = sum(ratio_map[source] for source in source_counts)
        normalized_weights = {
            source: ratio_map[source] / present_ratio_sum
            for source in sorted(source_counts)
        }
        logger.info(
            "PretrainDataset source-weighted sampling: source_counts=%s source_weights=%s "
            "weight_basis=%s",
            dict(sorted(source_counts.items())),
            {source: round(weight, 9) for source, weight in normalized_weights.items()},
            "window_starts" if window_start_sampling else "episode_count",
        )
        if window_start_sampling:
            logger.info(
                "PretrainDataset window-start sampling: episodes=%d groups=%s "
                "per_source_start_mass=%s",
                len(self.episode_files),
                {key: round(group_ratio.get(key, 0.0), 6) for key in sorted(group_ratio)},
                {source: int(source_start_mass[source]) for source in sorted(source_start_mass)},
            )

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
        base = (
            ep.get("episode_storage_key")
            or ep.get("episode_path")
            or ep.get("latent_path")
            or ep.get("episode_name")
            or "<unknown>"
        )
        variant = ep.get("stereo_pair_key") or ep.get("latent_file_name") or ep.get("latent_path")
        return f"{base}::{variant}" if variant else str(base)

    def _drop_episode(self, ep: Dict[str, Any], reason: str) -> None:
        episode_id = self._episode_identity(ep)
        if episode_id in self._dropped_episode_ids:
            return

        self._dropped_episode_ids.add(episode_id)
        logger.warning(
            "Dropping episode %s | dataset=%s task=%s storage_key=%s: %s",
            ep.get("episode_name", "?"),
            ep.get("dataset_name"),
            ep.get("task_name"),
            ep.get("episode_storage_key"),
            reason,
        )

    def _episode_for_index(self, idx: int, attempt: int = 0) -> Optional[Dict[str, Any]]:
        if not self.episode_files or self.total_episodes <= 0:
            return None

        base_idx = idx % self.total_episodes
        base_ep = self.episode_files[base_idx]
        if self.val:
            if self._episode_identity(base_ep) not in self._dropped_episode_ids:
                return base_ep
            # Keep validation batches bucket-consistent: once an episode is dropped,
            # only substitute another episode from the same bucket_key.
            bucket_indices = self._fallback_episode_indices(base_ep)
            try:
                start_pos = bucket_indices.index(base_idx)
            except ValueError:
                start_pos = 0
            for offset in range(1, len(bucket_indices) + 1):
                candidate_idx = bucket_indices[(start_pos + offset) % len(bucket_indices)]
                candidate = self.episode_files[candidate_idx]
                if self._episode_identity(candidate) not in self._dropped_episode_ids:
                    return candidate
            return None if self.source_weighted_sampling else base_ep

        # Weighted samplers choose a concrete source-bearing episode index.
        # Preserve that choice instead of replacing it with a random episode
        # from the same resolution bucket.
        if (
            self.source_weighted_sampling
            and self._episode_identity(base_ep) not in self._dropped_episode_ids
        ):
            return base_ep

        # Keep samples_per_episode as a train epoch-length multiplier, matching
        # the older Tianji/Wuji datasets. The index only anchors the resolution
        # bucket so SameResolutionBatchSampler still builds collatable batches.
        candidate_indices = self._fallback_episode_indices(base_ep)
        if not candidate_indices:
            return None

        for _ in range(min(16, len(candidate_indices))):
            candidate_idx = random.choice(candidate_indices)
            candidate = self.episode_files[candidate_idx]
            if self._episode_identity(candidate) not in self._dropped_episode_ids:
                return candidate

        start = random.randrange(len(candidate_indices))
        for offset in range(len(candidate_indices)):
            candidate_idx = candidate_indices[(start + offset) % len(candidate_indices)]
            candidate = self.episode_files[candidate_idx]
            if self._episode_identity(candidate) not in self._dropped_episode_ids:
                return candidate
        return None if self.source_weighted_sampling else base_ep

    @staticmethod
    def _sampling_source_identity(ep: Dict[str, Any]) -> str:
        """The configured source survives dataset aliases and resolution variants."""
        return str(ep.get("stereo_source") or ep.get("dataset_name") or "unknown")

    def _fallback_episode_indices(self, ep: Dict[str, Any]) -> List[int]:
        bucket = str(ep.get("bucket_key") or "unknown")
        indices = self._bucket_to_episode_indices.get(bucket) or []
        if not self.source_weighted_sampling:
            return indices or list(range(self.total_episodes))
        key = (bucket, self._sampling_source_identity(ep))
        source_buckets = getattr(self, "_source_bucket_to_episode_indices", None)
        if source_buckets is None:
            source_buckets = self._source_bucket_to_episode_indices = {}
        if key not in source_buckets:
            source_buckets[key] = [
                i for i in indices
                if self._sampling_source_identity(self.episode_files[i]) == key[1]
            ]
        return source_buckets[key]

    @staticmethod
    def _task_bucket_key(ep: Dict[str, Any]) -> str:
        dataset_name = ep.get("dataset_name")
        task_name = ep.get("task_name") or "<unknown-task>"
        if dataset_name:
            return f"{dataset_name}/{task_name}"
        return task_name

    def _rows_to_episode_dicts(self, rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        for row in rows:
            episode_bos_path = row.get("episode_bos_path")
            task = row.get("task")
            if episode_bos_path is None or task is None:
                continue
            if isinstance(episode_bos_path, float) or isinstance(task, float):
                continue
            if not str(episode_bos_path).strip() or not str(task).strip():
                continue

            dataset_name = row.get("dataset")
            if not self._dataset_name_allowed(dataset_name):
                continue
            base = self._episode_path_from_clickhouse(str(episode_bos_path).rstrip("/"))
            episode_name = row.get("episode_id") or base.rsplit("/", 1)[-1]
            latent_file_name = self._latent_file_name_for_dataset(dataset_name)
            bucket_key = self._resolution_bucket_from_text(latent_file_name)
            ep = {
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task,
                "dataset_name": dataset_name,
                "robot_name": row.get("robot"),
                "episode_storage_key": row.get("episode_storage_key"),
                "episode_path": base,
                "latent_root": smart_join(base, self.latent_root) if self.latent_root else None,
                "latent_path": smart_join(base, self.latent_subpath) if self.latent_subpath else None,
                "latent_file_name": latent_file_name,
                "bucket_key": bucket_key,
                "lang_root": smart_join(base, self.lang_root) if self.lang_root else None,
                "lang_path": smart_join(base, self.lang_subpath) if self.lang_subpath else None,
                "phase_start_30": 0,
            }
            ep.update(self._episode_action_paths(base))
            episodes.extend(self._expand_stereo_pair_episodes(ep, dataset_name))

        logger.info("%s pretrain index query returned %d episodes", source, len(episodes))
        return episodes

    def _episode_path_from_clickhouse(self, raw_path: str) -> str:
        raw = str(raw_path).rstrip("/")
        if is_remote(raw):
            return _normalize_remote_path(raw)
        for prefix in self.episode_path_prefixes:
            if not is_remote(prefix):
                continue
            raw_core = _strip_remote_scheme(raw)
            prefix_core = _strip_remote_scheme(prefix)
            if raw_core.startswith(prefix_core):
                return "s3://" + raw_core
        return raw

    def _load_episodes_from_clickhouse(self) -> List[Dict[str, Any]]:
        if not self.clickhouse:
            raise ValueError("clickhouse config is missing")

        ch = self.clickhouse
        table = str(ch.get("table") or "ario_format_data")
        use_final = bool(ch.get("use_final", True))
        final_kw = " FINAL" if use_final else ""

        desc_rows = _clickhouse_query_json_each_row(
            f"DESCRIBE TABLE {table} FORMAT JSONEachRow",
            ch,
        )
        available_cols = sorted(str(row.get("name")) for row in desc_rows if row.get("name"))

        episode_bos_expr = _column_alias_expr_for_sql(
            "episode_bos_path",
            available_cols,
            ["episode_bos_path", "episode_path", "bos_path"],
        )
        task_expr = _column_alias_expr_for_sql("task", available_cols, ["task", "task_name"])
        dataset_expr = _column_alias_expr_for_sql("dataset", available_cols, ["dataset", "dataset_name"])
        episode_id_expr = _column_alias_expr_for_sql(
            "episode_id",
            available_cols,
            ["episode_id", "episode_name"],
        )
        storage_key_expr = _column_alias_expr_for_sql(
            "episode_storage_key",
            available_cols,
            ["episode_storage_key", "storage_key"],
        )
        robot_expr = _column_alias_expr_for_sql("robot", available_cols, ["robot", "robot_name"])

        if not episode_bos_expr or not task_expr:
            raise ValueError(
                "ClickHouse table schema missing required episode path/task columns. "
                f"available columns: {available_cols}"
            )
        select_exprs = [
            episode_bos_expr,
            dataset_expr or "'' AS dataset",
            task_expr,
            episode_id_expr or "'' AS episode_id",
            storage_key_expr or "'' AS episode_storage_key",
            robot_expr or "'' AS robot",
        ]

        path_col = episode_bos_expr.split(" AS ", 1)[0]
        where = self._episode_where_sql(path_col)
        sql = f"""
            SELECT
                {", ".join(select_exprs)}
            FROM {table}{final_kw}
            {where}
            FORMAT JSONEachRow
        """
        rows = _clickhouse_query_json_each_row(sql, ch)
        return self._rows_to_episode_dicts(rows, f"ClickHouse:{table}")

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

            ep = {
                "episode_name": name,
                "source_episode_name": base_name,
                "task_name": task_name,
                "dataset_name": dataset_name,
                "latent_path": latent_file,
                "latent_file_name": self._latent_file_name_for_dataset(dataset_name),
                "qpos_path": qpos_file,
                "lang_candidates": lang_candidates,
                "lang_path": lang_candidates[0],
                "phase_start_30": phase_start_30,
            }
            if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
                ep.update({
                    "hand_path": qpos_file,
                    "qpos_path": qpos_file,
                    "endpose_path": smart_join(task_path_str, "arm_endpose.pt"),
                })
            episodes.extend(self._expand_stereo_pair_episodes(ep, dataset_name))
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
                if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
                    ep.update({
                        "tjwj_path": smart_join(root, entry.get("tjwj_path", "tjwj.pt")),
                        "hand_path": smart_join(root, entry.get("hand_path", "hand_qpos.pt")),
                        "qpos_path": smart_join(root, entry.get("hand_path", "hand_qpos.pt")),
                        "endpose_path": smart_join(root, entry.get("endpose_path", "arm_endpose.pt")),
                    })
                all_episodes.extend(self._expand_stereo_pair_episodes(ep, ep.get("dataset_name")))
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
            if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
                ep.update({
                    "tjwj_path": smart_join(root, row.get("tjwj_path", "tjwj.pt")),
                    "hand_path": smart_join(root, row.get("hand_path", "hand_qpos.pt")),
                    "qpos_path": smart_join(root, row.get("hand_path", "hand_qpos.pt")),
                    "endpose_path": smart_join(root, row.get("endpose_path", "arm_endpose.pt")),
                })
            all_episodes.extend(self._expand_stereo_pair_episodes(ep, dataset_name))
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
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            return "tjwj.pt"
        return "endpose.pt" if "eef" in self.action_mode else "qpos.pt"

    def _default_task_action_dir(self) -> str:
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            return "hand_qpos"
        return "endpose" if "eef" in self.action_mode else "qpos"

    def _resolve_entry_action_path(self, entry: Dict[str, Any]) -> Optional[str]:
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            default_tjwj = None
            episode_bos_path = entry.get("episode_bos_path")
            if episode_bos_path:
                default_tjwj = smart_join(self._normalize_bos(str(episode_bos_path).rstrip("/")), "tjwj.pt")
            return (
                entry.get("tjwj_path")
                or default_tjwj
                or entry.get("hand_path")
                or entry.get("hand_qpos_path")
                or entry.get("qpos_path")
            )
        return entry.get("endpose_path") if "eef" in self.action_mode else entry.get("qpos_path")

    def _resolve_manifest_action_path(self, entry: Dict[str, Any]) -> Optional[str]:
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            return entry.get("tjwj_path") or entry.get("hand_path") or entry.get("hand_qpos_path") or entry.get("qpos_path")
        return entry.get("endpose_path") if "eef" in self.action_mode else entry.get("qpos_path")

    def _sample_action_path(self, ep: Dict[str, Any]) -> Optional[str]:
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            return ep.get("tjwj_path") or ep.get("hand_path") or ep.get("qpos_path")
        return ep.get("endpose_path") if "eef" in self.action_mode else ep.get("qpos_path")

    def _episode_action_paths(self, episode_dir: str) -> Dict[str, str]:
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            return {
                "tjwj_path": smart_join(episode_dir, "tjwj.pt"),
                "hand_path": smart_join(episode_dir, "hand_qpos.pt"),
                "qpos_path": smart_join(episode_dir, "hand_qpos.pt"),
                "endpose_path": smart_join(episode_dir, "arm_endpose.pt"),
            }
        return {
            "qpos_path": smart_join(episode_dir, self._default_episode_action_filename()),
            "endpose_path": smart_join(episode_dir, "endpose.pt"),
        }

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
            ep = {
                "episode_name": episode_name,
                "source_episode_name": episode_name,
                "task_name": task_name,
                "dataset_name": dataset_name,
                "episode_path": episode_dir,
                "latent_root": latent_root,
                "latent_path": latent_path,
                "latent_file_name": self._latent_file_name_for_dataset(dataset_name),
                "lang_root": smart_join(episode_dir, self.lang_root) if self.lang_root else None,
                "lang_path": smart_join(episode_dir, self.lang_subpath) if self.lang_subpath else None,
                "phase_start_30": 0,
                **self._episode_action_paths(episode_dir),
            }
            episodes.extend(self._expand_stereo_pair_episodes(ep, dataset_name))
        return episodes

    def _resolve_latent_candidates(self, ep: Dict[str, Any]) -> List[Dict[str, Any]]:
        if ep.get("action_schema") == "handson-stage2/raw54-v1":
            return [{"latent_path": ep["latent_path"], "phase_start_30": ep["phase_start_30"]}]
        candidates: List[Dict[str, Any]] = []
        latent_root = ep.get("latent_root")
        if ep.get("stereo_left_latent_name") and ep.get("stereo_right_latent_name"):
            stereo_pairs = [(ep["stereo_left_latent_name"], ep["stereo_right_latent_name"])]
        else:
            stereo_pairs = self._stereo_pairs_for_dataset(ep.get("dataset_name"), ep)

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
                if stereo_pairs:
                    for left_name, right_name in stereo_pairs:
                        candidates.append({
                            "left_latent_path": smart_join(phase_dir, left_name),
                            "right_latent_path": smart_join(phase_dir, right_name),
                            "latent_file_name": f"{left_name}|{right_name}",
                            "phase_start_30": phase_index or 0,
                        })
                else:
                    latent_path = smart_join(phase_dir, self.latent_file_name)
                    candidates.append({
                        "latent_path": latent_path,
                        "phase_start_30": phase_index or 0,
                    })
            if not self.val:
                random.shuffle(candidates)
        elif latent_root:
            if self.latent_phases_set is not None:
                for phase_index in random.sample(
                    sorted(self.latent_phases_set),
                    k=len(self.latent_phases_set),
                ):
                    phase_dir = smart_join(latent_root, f"{phase_index}phase_5hz")
                    if stereo_pairs:
                        for left_name, right_name in stereo_pairs:
                            candidates.append({
                                "left_latent_path": smart_join(phase_dir, left_name),
                                "right_latent_path": smart_join(phase_dir, right_name),
                                "latent_file_name": f"{left_name}|{right_name}",
                                "phase_start_30": int(phase_index),
                            })
                    else:
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
                    if stereo_pairs:
                        for left_name, right_name in stereo_pairs:
                            candidates.append({
                                "left_latent_path": smart_join(phase_dir, left_name),
                                "right_latent_path": smart_join(phase_dir, right_name),
                                "latent_file_name": f"{left_name}|{right_name}",
                                "phase_start_30": phase_index or 0,
                            })
                    else:
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
        if latent_idx < 0 or frame_stride <= 0 or phase_start_30 < 0:
            raise ValueError(
                f"invalid pretrain temporal coordinates: latent_idx={latent_idx}, "
                f"frame_stride={frame_stride}, phase_start_30={phase_start_30}"
            )
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

    def _stats_dataset_name(self, ep: Dict[str, Any]) -> Optional[str]:
        """Prefer an explicitly configured dataset, then its canonical source."""
        if not self.qpos_min_by_dataset and not self.qpos_mean_by_dataset:
            return None
        registered = self.qpos_min_by_dataset.keys() | self.qpos_mean_by_dataset.keys()
        for key in ("dataset", "dataset_name", "stereo_source"):
            value = ep.get(key)
            if value is not None and str(value) in registered:
                return str(value)
        for key in ("dataset", "dataset_name"):
            v = ep.get(key)
            if v:
                return str(v)
        path = str(ep.get("episode_bos_path") or "")
        return path.rstrip("/").split("/")[0] or None

    def _qpos_stats_for(self, dataset_name: Optional[str]):
        """Resolve normalisation stats for one source, falling back to the global pair.

        Returns (mode, low_or_mean, range_or_std).
        """
        if dataset_name:
            k = str(dataset_name)
            if k in self.qpos_mean_by_dataset:
                return ("mean_std", self.qpos_mean_by_dataset[k], self.qpos_std_by_dataset[k])
            if k in self.qpos_min_by_dataset:
                return (self.qpos_normalization, self.qpos_min_by_dataset[k], self.qpos_range_by_dataset[k])
        if self.qpos_normalization == "mean_std":
            return ("mean_std", self.qpos_mean, self.qpos_std)
        return (self.qpos_normalization, self.qpos_min, self.qpos_range)


    def _normalize_qpos40(self, left_qpos: torch.Tensor, right_qpos: torch.Tensor,
                          dataset_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        qpos40 = torch.cat([left_qpos, right_qpos], dim=-1).float()
        mode, low, span = self._qpos_stats_for(dataset_name)
        if mode == "mean_std":
            if low is None or span is None:
                raise RuntimeError(f"{self.action_mode} requires qpos mean/std stats")
            qpos40 = (qpos40 - low) / span
        elif mode == "min_max":
            if low is None or span is None:
                raise RuntimeError(f"{self.action_mode} requires qpos min/max stats")
            qpos40 = (qpos40 - low) / span
        elif mode == "quantile":
            if low is None or span is None:
                raise RuntimeError(f"{self.action_mode} requires qpos q01/q99 stats")
            qpos40 = torch.maximum(torch.minimum(qpos40, low + span), low)
            qpos40 = (qpos40 - low) / span
        else:
            raise RuntimeError(f"Unsupported qpos_normalization: {mode}")
        return qpos40[:, :20], qpos40[:, 20:]

    def _downsample_action_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        # Kept as a compatibility hook for the existing sequence loaders, but
        # resampling is deliberately forbidden on the pretrain path.
        if self.action_downsample_stride != 1:
            raise ValueError("PretrainDataset action resampling is disabled; expected stride=1")
        return sequence

    @staticmethod
    def _hand_tracking_valid_mask(
        payload: Any,
        count: int,
        left_quaternion: Optional[torch.Tensor] = None,
        right_quaternion: Optional[torch.Tensor] = None,
        left_xyz: Optional[torch.Tensor] = None,
        right_xyz: Optional[torch.Tensor] = None,
        origin_marker_is_invalid: bool = False,
    ) -> torch.Tensor:
        """Read visibility before converting missing raw quaternions to rot6d.

        Optional payload fields are ``hand_valid_mask`` / ``hand_visibility``
        ([T,2], left then right), or ``left_hand_valid`` / ``right_hand_valid``
        ([T]). Explicit visibility can only remove validity, never turn a zero
        or nonfinite quaternion into a valid tracking observation.

        With ``origin_marker_is_invalid``, a frame whose wrist sits exactly at
        the origin with an identity rotation is also invalid. Some exports
        (chaowei) write that pose for a lost track; its quaternion has unit
        norm, so the norm test alone accepts it as a real wrist.
        """
        explicit = [False, False]
        valid = torch.ones((count, 2), dtype=torch.bool)

        def flags(value: Any, shape: Tuple[int, ...]) -> torch.Tensor:
            value = torch.as_tensor(value)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"hand visibility must be finite with shape {shape}")
            if not ((value == 0) | (value == 1)).all():
                raise ValueError("hand visibility must contain only boolean/0/1 flags")
            return value.to(dtype=torch.bool, device="cpu")

        if isinstance(payload, dict):
            for key in ("hand_valid_mask", "hand_visibility"):
                if key in payload:
                    valid &= flags(payload[key], (count, 2))
                    explicit = [True, True]
            for hand, key in enumerate(("left_hand_valid", "right_hand_valid")):
                if key in payload:
                    valid[:, hand] &= flags(payload[key], (count,))
                    explicit[hand] = True
        if left_quaternion is None or right_quaternion is None:
            if not all(explicit):
                raise ValueError("converted rot6d actions require explicit hand visibility")
        else:
            for hand, quaternion in enumerate((left_quaternion, right_quaternion)):
                if quaternion.shape != (count, 4):
                    raise ValueError("raw hand quaternion must have shape [T,4]")
                valid[:, hand] &= (
                    torch.isfinite(quaternion).all(dim=-1)
                    & (quaternion.float().norm(dim=-1) > 1e-6)
                ).cpu()
            if origin_marker_is_invalid and left_xyz is not None and right_xyz is not None:
                for hand, (xyz, quaternion) in enumerate(((left_xyz, left_quaternion),
                                                          (right_xyz, right_quaternion))):
                    if xyz.shape != (count, 3):
                        raise ValueError("raw hand xyz must have shape [T,3]")
                    q = quaternion.float()
                    # Identity in either convention (xyzw or wxyz) and either sign.
                    identity = ((q[:, :3].abs().amax(dim=-1) < 1e-4) & ((q[:, 3].abs() - 1).abs() < 1e-4)) | (
                        (q[:, 1:4].abs().amax(dim=-1) < 1e-4) & ((q[:, 0].abs() - 1).abs() < 1e-4))
                    at_origin = xyz.float().abs().amax(dim=-1) < 1e-4
                    valid[:, hand] &= ~(identity & at_origin).cpu()
        return valid

    def _load_qpos20_wrist9_sequence(self, ep: Dict[str, Any], *, return_validity: bool = False):
        tjwj_path = ep.get("tjwj_path")
        if tjwj_path:
            try:
                raw = smart_load_pt(tjwj_path, self.cache_dir, map_location="cpu")
                payload = raw
                if isinstance(raw, dict):
                    if "action" in raw and isinstance(raw["action"], torch.Tensor):
                        raw = raw["action"]
                    elif "action_xyzw" in raw and isinstance(raw["action_xyzw"], torch.Tensor):
                        raw = raw["action_xyzw"]
                    elif all(
                        k in raw for k in ("left_qpos20", "right_qpos20", "left_wrist_xyz_xyzw", "right_wrist_xyz_xyzw")
                    ):
                        left_q = _as_float_tensor(raw["left_qpos20"])
                        right_q = _as_float_tensor(raw["right_qpos20"])
                        left_w = _as_float_tensor(raw["left_wrist_xyz_xyzw"])
                        right_w = _as_float_tensor(raw["right_wrist_xyz_xyzw"])
                        n = min(left_q.shape[0], right_q.shape[0], left_w.shape[0], right_w.shape[0])
                        validity = self._hand_tracking_valid_mask(
                            payload, n, left_w[:n, 3:7], right_w[:n, 3:7],
                            left_w[:n, :3], right_w[:n, :3], origin_marker_is_invalid=getattr(self, "invalid_origin_wrist_marker", False),
                        ) if return_validity else None
                        left_q, right_q = self._normalize_qpos40(left_q[:n, :20], right_q[:n, :20], self._stats_dataset_name(ep))
                        left_w9 = _wrist7_to_wrist9(left_w[:n, :7])
                        right_w9 = _wrist7_to_wrist9(right_w[:n, :7])
                        seq = self._downsample_action_sequence(torch.cat([left_q, left_w9, right_q, right_w9], dim=-1))
                        return (seq, validity) if return_validity else seq
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
                    validity = self._hand_tracking_valid_mask(
                        payload, len(raw), left_w7[:, 3:7], right_w7[:, 3:7],
                        left_w7[:, :3], right_w7[:, :3], origin_marker_is_invalid=getattr(self, "invalid_origin_wrist_marker", False),
                    ) if return_validity else None
                    left_q, right_q = self._normalize_qpos40(left_q, right_q, self._stats_dataset_name(ep))
                    left_w9 = _wrist7_to_wrist9(left_w7)
                    right_w9 = _wrist7_to_wrist9(right_w7)
                    seq = torch.cat([left_q, left_w9, right_q, right_w9], dim=-1)
                    _assert_finite_tensor(seq, "qpos20_rel6d action sequence", tjwj_path)
                    seq = self._downsample_action_sequence(seq)
                    return (seq, validity) if return_validity else seq
                if raw.shape[1] == 58:
                    left_q = raw[:, :20]
                    left_w9 = raw[:, 20:29]
                    right_q = raw[:, 29:49]
                    right_w9 = raw[:, 49:58]
                    validity = self._hand_tracking_valid_mask(payload, len(raw)) if return_validity else None
                    left_q, right_q = self._normalize_qpos40(left_q, right_q, self._stats_dataset_name(ep))
                    seq = torch.cat([left_q, left_w9, right_q, right_w9], dim=-1)
                    _assert_finite_tensor(seq, "qpos20_rel6d action sequence", tjwj_path)
                    seq = self._downsample_action_sequence(seq)
                    return (seq, validity) if return_validity else seq
                raise ValueError(f"Unsupported tjwj tensor dim: {raw.shape[1]}")
            except Exception as e:
                if return_validity and isinstance(e, ValueError):
                    raise
                logger.warning(
                    "Failed loading tjwj for %s from %s, fallback to hand/endpose: %s",
                    ep.get("episode_name", "?"),
                    tjwj_path,
                    e,
                )

        hand_path = ep.get("hand_path") or ep.get("qpos_path")
        endpose_path = ep.get("endpose_path")
        if not hand_path or not endpose_path:
            raise FileNotFoundError(f"{self.action_mode} requires tjwj.pt or hand_qpos.pt+arm_endpose.pt")
        hand = smart_load_pt(hand_path, self.cache_dir, map_location="cpu")
        endpose = smart_load_pt(endpose_path, self.cache_dir, map_location="cpu")
        left_qpos = _as_float_tensor(hand["left"])
        right_qpos = _as_float_tensor(hand["right"])
        left_endpose = _as_float_tensor(endpose["left"])
        right_endpose = _as_float_tensor(endpose["right"])
        n = min(left_qpos.shape[0], right_qpos.shape[0], left_endpose.shape[0], right_endpose.shape[0])
        validity = self._hand_tracking_valid_mask(
            endpose, n, left_endpose[:n, 3:7], right_endpose[:n, 3:7],
            left_endpose[:n, :3], right_endpose[:n, :3], origin_marker_is_invalid=getattr(self, "invalid_origin_wrist_marker", False),
        ) if return_validity else None
        left_wrist = _endpose_to_wrist9(left_endpose)
        right_wrist = _endpose_to_wrist9(right_endpose)
        left_qpos, right_qpos = self._normalize_qpos40(left_qpos[:n], right_qpos[:n], self._stats_dataset_name(ep))
        seq = torch.cat([left_qpos, left_wrist[:n], right_qpos, right_wrist[:n]], dim=-1)
        _assert_finite_tensor(seq, "qpos20_rel6d action sequence", hand_path)
        seq = self._downsample_action_sequence(seq)
        return (seq, validity) if return_validity else seq

    def _gather_per_latent(
        self,
        sequence: torch.Tensor,
        latent_start: int,
        num_latents: int,
        frame_stride: int,
        phase_start_30: int,
        per_latent_len: int,
    ) -> torch.Tensor:
        chunks = []
        n = sequence.shape[0]
        if n == 0:
            raise ValueError("cannot align an empty pretrain action sequence")
        expected_len = 4 * frame_stride * self.action_upsample_ratio
        if per_latent_len != expected_len:
            raise ValueError(
                "pretrain latent/action geometry mismatch: "
                f"actions_per_latent={per_latent_len}, expected {expected_len} "
                f"for frame_stride={frame_stride}"
            )
        for i in range(num_latents):
            abs_lat = latent_start + i
            a_start, a_end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            a_start = min(a_start, n - 1)
            a_end = min(a_end, n)
            chunk = sequence[a_start:a_end]
            chunk = pad_or_trim_chunk(chunk, per_latent_len, fill=sequence[-1])
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

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
            a_start, _ = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            state_idx = max(min(a_start, n - 1) - 1, 0)
            states.append(sequence[state_idx])
        return torch.stack(states, dim=0)

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

    def _load_language_embedding(self, ep: Dict[str, Any], sample_idx: int = 0) -> Optional[torch.Tensor]:
        if self.treat_as_irrelevant and self.empty_lang_path:
            lang_path = self.empty_lang_path
        else:
            candidates = self._resolve_language_candidates(ep)
            if not candidates:
                return None
            lang_path = None
            last_error = None
            if ep.get('action_schema') == 'handson-stage2/raw54-v1':
                if not hasattr(self, '_handson_language_cache'):
                    self._handson_language_cache = {}
                language_path = ep['handson_language_path']
                if language_path not in self._handson_language_cache:
                    self._handson_language_cache[language_path] = smart_load_json(language_path)
                language = self._handson_language_cache[language_path]
                candidate_order = [language['canonical'] if self.val or random.random() < 0.2
                                   else random.choice(language['train'])]
            else:
                candidate_order = sorted(candidates) if self.val else random.sample(candidates, k=len(candidates))
            for candidate in candidate_order:
                try:
                    embedding_data = smart_load_pt(candidate, self.cache_dir, map_location="cpu")
                    lang_path = candidate
                    break
                except Exception as e:
                    last_error = e
            if lang_path is None:
                self._disable_language_for_episode(ep)
                logger.warning(
                    "No loadable language embedding for %s | dataset=%s task=%s storage_key=%s, skipping sample: %s",
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
            idx = sample_idx % len(embedding_data) if self.val else random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[idx]
        elif isinstance(embedding_data, torch.Tensor):
            if embedding_data.ndim == 3:
                idx = sample_idx % embedding_data.shape[0] if self.val else random.randint(0, embedding_data.shape[0] - 1)
                embeddings = embedding_data[idx]
            else:
                embeddings = embedding_data
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
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor] | Tuple[
        Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        mask_tracking = bool(getattr(self, "mask_invalid_hand_tracking", False))
        if ep.get("action_schema") == "handson-stage2/raw54-v1":
            from .handson_adapter import decode_payload, gather, load_payload
            if (self.action_mode != "qpos20_rel6d" or self.actions_per_latent != 24
                    or frame_stride != 6 or self.action_upsample_ratio != 1):
                raise ValueError('HandsOn stage2 requires qpos20_rel6d, 30Hz and stride6')
            payload = load_payload(ep["tjwj_path"], lambda p: smart_load_pt(p, self.cache_dir, map_location="cpu"))
            hand_bounds = None
            if getattr(self, "wrist_xyz_per_dataset", False):
                mode, low, span = self._qpos_stats_for("HandsOn")
                if mode != "min_max" or "HandsOn" not in self.qpos_min_by_dataset:
                    raise ValueError("HandsOn stage2 requires configured HandsOn min_max hand stats")
                hand_bounds = (low, span)
            commands, measured = decode_payload(payload, hand_bounds=hand_bounds)
            if len(commands) != int(ep['handson_action_frames']):
                raise ValueError('HandsOn sidecar/index length mismatch')
            states, actions = gather(commands, measured, latent_start, num_latents, phase_start_30)
            if not mask_tracking:
                return states, actions
            raw_commands, raw_states = payload["commands"], payload["measured_states"]
            action_validity = self._hand_tracking_valid_mask(
                payload, len(raw_commands), raw_commands[:, 23:27], raw_commands[:, 50:54],
                raw_commands[:, 20:23], raw_commands[:, 47:50], origin_marker_is_invalid=getattr(self, "invalid_origin_wrist_marker", False),
            )
            state_validity = self._hand_tracking_valid_mask(
                payload, len(raw_states), raw_states[:, 23:27], raw_states[:, 50:54],
                raw_states[:, 20:23], raw_states[:, 47:50], origin_marker_is_invalid=getattr(self, "invalid_origin_wrist_marker", False),
            )
            action_mask = self._gather_per_latent(
                action_validity, latent_start, num_latents, frame_stride,
                phase_start_30, self.actions_per_latent,
            )
            state_mask = self._gather_states(
                state_validity, latent_start, num_latents, frame_stride, phase_start_30
            )
            if latent_start == 0:
                state_mask[0] = state_validity[phase_start_30]
            return states, actions, action_mask, state_mask
        if self.action_mode in {"qpos20_rel6d", "relqpos20_rel6d"}:
            if mask_tracking:
                sequence, validity = self._load_qpos20_wrist9_sequence(ep, return_validity=True)
            else:
                sequence = self._load_qpos20_wrist9_sequence(ep)
            actions = self._gather_per_latent(
                sequence,
                latent_start,
                num_latents,
                frame_stride,
                phase_start_30,
                self.actions_per_latent,
            )
            states = self._gather_states(
                sequence,
                latent_start,
                num_latents,
                frame_stride,
                phase_start_30,
            )
            if mask_tracking:
                action_mask = self._gather_per_latent(
                    validity, latent_start, num_latents, frame_stride,
                    phase_start_30, self.actions_per_latent,
                )
                state_mask = self._gather_states(
                    validity, latent_start, num_latents, frame_stride, phase_start_30
                )
                return states, actions, action_mask, state_mask
            return states, actions

        qpos = smart_load_pt(qpos_path, self.cache_dir, map_location="cpu")
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos)
        qpos = qpos.float()
        qpos = self._normalize_gripper_columns(qpos, ep)
        qpos = self._maybe_convert_eef_order(qpos)

        if self.action_mode in {"rel_eef", "abs_eef"}:
            qpos = upsample_eef(qpos, self.action_upsample_ratio)
        elif self.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"}:
            qpos = upsample_abs_eef_16d(qpos, self.action_upsample_ratio)
            qpos = self._maybe_output_wxyz(qpos)
        else:
            qpos = self._upsample_qpos(qpos, self.action_upsample_ratio)

        need_states = use_state or self.action_mode in {"rel_eef", "rel_eef_wxyz"}
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

    @staticmethod
    def _wrist_xyz_slices(action_dim: int) -> Optional[Tuple[slice, slice]]:
        if action_dim >= 58:
            return slice(20, 23), slice(49, 52)
        if action_dim >= 54:
            return slice(20, 23), slice(47, 50)
        if action_dim == 20:
            return slice(0, 3), slice(10, 13)
        if action_dim == 16:
            return slice(0, 3), slice(8, 11)
        if action_dim == 14:
            return slice(0, 3), slice(7, 10)
        return None

    def _actions_pass_wrist_xyz_abs_filter(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        valid_len: int,
    ) -> bool:
        if self.wrist_xyz_abs_max is None:
            return True

        slices = self._wrist_xyz_slices(int(actions.shape[-1]))
        if slices is None:
            return True

        action_tokens = max(0, int(valid_len)) * self.actions_per_latent
        tensors = [actions[:action_tokens]]
        if states is not None:
            tensors.append(states[:max(0, int(valid_len))])

        threshold = float(self.wrist_xyz_abs_max)
        left_slice, right_slice = slices
        for tensor in tensors:
            if tensor.numel() == 0:
                return False
            wrist_xyz = torch.cat([tensor[..., left_slice], tensor[..., right_slice]], dim=-1).float()
            if (not torch.isfinite(wrist_xyz).all().item()) or wrist_xyz.abs().max().item() > threshold:
                return False
        return True

    def _actions_pass_wrist_xyz_relative_filter(
        self, actions, states, valid_len, action_valid_mask=None, state_valid_mask=None,
    ):
        """Check Euclidean wrist displacement from each model chunk's anchor.

        Called after bootstrap so indices and anchor states match the model.
        The threshold uses decoded, unnormalized wrist-coordinate units. The
        Euclidean norm is unchanged by the anchor rotation used by the model.
        Non-finite coordinates always fail; invalid hands do not contribute to
        displacement checks because their relative targets are masked.
        """
        threshold = getattr(self, "wrist_xyz_relative_max", None)
        if threshold is None:
            return True
        slices = self._wrist_xyz_slices(int(actions.shape[-1]))
        if slices is None or states is None:
            raise ValueError('Relative wrist filtering requires wrist coordinates and anchor states')
        cs = self.bootstrap_chunk_size
        apl = self.actions_per_latent
        valid_len = max(0, min(int(valid_len), len(states), len(actions) // apl))
        if cs < 1 or valid_len == 0:
            return False
        for hand, wrist in enumerate(slices):
            xyz = actions[:valid_len * apl, wrist].float()
            anchors = states[:valid_len, wrist].float()
            if not torch.isfinite(xyz).all() or not torch.isfinite(anchors).all():
                return False
            for start in range(cs, valid_len, cs):
                if state_valid_mask is not None and not bool(state_valid_mask[start, hand]):
                    continue
                segment = xyz[start * apl:min(start + cs, valid_len) * apl]
                if action_valid_mask is not None:
                    segment = segment[action_valid_mask[start * apl:min(start + cs, valid_len) * apl, hand].bool()]
                if segment.numel() and (torch.linalg.vector_norm(segment - anchors[start], dim=-1) > threshold).any():
                    return False
        return True

    def _actions_pass_abs_filter(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        valid_len: int,
    ) -> bool:
        if self.action_abs_max is None:
            return True

        action_tokens = max(0, int(valid_len)) * self.actions_per_latent
        tensors = [actions[:action_tokens]]
        if states is not None:
            tensors.append(states[:max(0, int(valid_len))])

        threshold = float(self.action_abs_max)
        for tensor in tensors:
            if tensor.numel() == 0:
                return False
            tensor = tensor.float()
            if (not torch.isfinite(tensor).all().item()) or tensor.abs().max().item() > threshold:
                return False
        return True

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

    def _load_single_latent_candidate(self, candidate: Dict[str, Any]):
        latent_data = smart_load_pt(candidate["latent_path"], self.cache_dir, map_location="cpu")
        if isinstance(latent_data, dict) and "phases" in latent_data:
            phases = latent_data["phases"]
            if self.latent_phases_set is not None:
                phases = [p for i, p in enumerate(phases) if i in self.latent_phases_set]
            if not phases:
                raise ValueError("latent file has no usable phases after filtering")
            phase_data = phases[0] if self.val else random.choice(phases)
            latents = phase_data["latents"]
            frame_stride = int(latent_data.get("frame_stride", phase_data.get("frame_stride", 6)))
            phase_start_30 = int(phase_data.get("phase_start_30", 0))
        elif isinstance(latent_data, torch.Tensor):
            latents = latent_data
            frame_stride = self.video_action_freq_ratio
            phase_start_30 = int(candidate.get("phase_start_30", 0))
        else:
            latents = latent_data["latents"]
            frame_stride = int(latent_data.get("frame_stride", self.video_action_freq_ratio))
            phase_start_30 = int(latent_data.get("phase_start_30", candidate.get("phase_start_30", 0)))
        return latents.float(), frame_stride, phase_start_30

    def _load_latent_geometry_metadata(self, path, *, allow_payload_fallback=True):
        """Read real local torch-save geometry without loading tensor storage.

        PyTorch 2.10 still reads data/* storages for plain meta loads; skip_data
        keeps geometry/stride/dtype from zip metadata only. Never substitute index T.
        Remote streams and legacy non-zip files keep their original reader.
        """
        if is_remote(path):
            if not allow_payload_fallback:
                raise RuntimeError("metadata-only geometry cannot use remote payload fallback")
            return smart_load_pt(path, self.cache_dir, map_location="cpu")
        with open(path, "rb") as stream:
            is_torch_zip = stream.read(4) == b"PK\x03\x04"
        if not is_torch_zip:
            if not allow_payload_fallback:
                raise RuntimeError("metadata-only geometry requires a local torch-save zip")
            return smart_load_pt(path, self.cache_dir, map_location="cpu")
        try:
            with torch.serialization.skip_data():
                return torch.load(path, map_location="meta", weights_only=False)
        except TypeError:
            with torch.serialization.skip_data():
                return torch.load(path, map_location="meta")

    def _load_stereo_latent_candidate(self, candidate: Dict[str, Any], *, metadata_only=False,
                                      allow_payload_fallback=True):
        left_data = (self._load_latent_geometry_metadata(candidate["left_latent_path"],
                                                       allow_payload_fallback=allow_payload_fallback)
                     if metadata_only else smart_load_pt(candidate["left_latent_path"], self.cache_dir, map_location="cpu"))
        right_data = (self._load_latent_geometry_metadata(candidate["right_latent_path"],
                                                        allow_payload_fallback=allow_payload_fallback)
                      if metadata_only else smart_load_pt(candidate["right_latent_path"], self.cache_dir, map_location="cpu"))

        def _extract(data, side: str):
            if isinstance(data, torch.Tensor):
                return data.float(), self.video_action_freq_ratio, int(candidate.get("phase_start_30", 0))
            if not isinstance(data, dict) or "latents" not in data:
                raise ValueError(f"{side} stereo latent must be tensor or dict with latents")
            latents = data["latents"].float()
            frame_stride = int(data.get("frame_stride", self.video_action_freq_ratio))
            phase_start_30 = int(data.get("phase_start_30", candidate.get("phase_start_30", 0)))
            return latents, frame_stride, phase_start_30

        left, left_stride, left_phase = _extract(left_data, "left")
        right, right_stride, right_phase = _extract(right_data, "right")
        if (left_stride, left_phase) != (right_stride, right_phase):
            raise ValueError(
                "stereo latent timeline mismatch: "
                f"left=(frame_stride={left_stride}, phase_start_30={left_phase}), "
                f"right=(frame_stride={right_stride}, phase_start_30={right_phase})"
            )
        if left.shape[0] != right.shape[0] or left.shape[2:] != right.shape[2:]:
            raise ValueError(f"stereo latent shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")
        total_t = min(left.shape[1], right.shape[1])
        latents = torch.stack([left[:, :total_t], right[:, :total_t]], dim=0)
        return latents, left_stride, left_phase

    def _load_latent_candidate(self, candidate: Dict[str, Any]):
        if "left_latent_path" in candidate and "right_latent_path" in candidate:
            return self._load_stereo_latent_candidate(candidate)
        return self._load_single_latent_candidate(candidate)

    @staticmethod
    def _latent_total_frames(latents: torch.Tensor) -> int:
        return int(latents.shape[2] if latents.dim() == 5 else latents.shape[1])

    @staticmethod
    def _latent_crop_time(latents: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if latents.dim() == 5:
            return latents[:, :, start:end]
        return latents[:, start:end]

    @staticmethod
    def _latent_pad_time(latents: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return latents
        if latents.dim() == 5:
            return F.pad(latents, (0, 0, 0, 0, 0, pad))
        return F.pad(latents, (0, 0, 0, 0, 0, pad))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        if self.val and self._frozen_val_indices is not None and not self._freeze_build_mode:
            return len(self._frozen_val_indices)
        if self.window_start_sampling:
            # One slot per episode: the batch sampler expands the pool and packs
            # a window ordinal into each entry, so ``samples_per_episode`` must
            # not multiply the index space on top of that.
            return self.total_episodes
        return self.total_episodes * self.samples_per_episode

    def _validation_window_start(self, idx: int, max_start: int) -> int:
        if max_start <= 0:
            return 0
        sample_slot = idx // max(1, self.total_episodes)
        if self.samples_per_episode <= 1:
            return max_start // 2
        return int(round(max_start * (sample_slot % self.samples_per_episode) / (self.samples_per_episode - 1)))

    @staticmethod
    def _latent_candidate_key(candidate: Dict[str, Any]) -> tuple:
        return (
            str(candidate.get("latent_path", "")),
            str(candidate.get("left_latent_path", "")),
            str(candidate.get("right_latent_path", "")),
            str(candidate.get("phase_start_30", "")),
        )

    def _window_size_for_episode(self, ep: Dict[str, Any], idx: int) -> int:
        return int(ep.get("window_size") or ep.get("sc_window_size") or self.max_latent_window)

    def _effective_window_size_for_episode(
        self,
        ep: Dict[str, Any],
        idx: int,
        total_latents: int,
        requested_window_size: int,
    ) -> int:
        return int(requested_window_size)

    def _select_latent_window(
        self,
        idx: int,
        total_latents: int,
        window_size: int,
        visit_ordinal: Optional[int] = None,
        episode_key: Optional[str] = None,
    ) -> Tuple[int, int]:
        if total_latents > window_size:
            if self.window_start_sampling and visit_ordinal is not None:
                # Deterministic per-visit start: tile without overlap first, then
                # spend the unused starts before any of them repeats.
                return (
                    select_window_start(
                        visit_ordinal,
                        total_latents,
                        window_size,
                        episode_key or str(idx),
                    ),
                    window_size,
                )
            if self.val:
                start = self._validation_window_start(idx, total_latents - window_size)
            else:
                start = random.randint(0, total_latents - window_size)
            return start, window_size
        if total_latents < window_size:
            return 0, total_latents
        return 0, window_size

    def _strict_latent_candidates(self, ep: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enumerate candidates without spending the worker's augmentation RNG."""
        rng = random.getstate()
        try:
            return sorted(self._resolve_latent_candidates(ep), key=self._latent_candidate_key)
        finally:
            random.setstate(rng)

    def _strict_load_latent_variant(self, variant, *, select_phase=False, metadata_only=False,
                                    allow_payload_fallback=True):
        """Load one bound variant; never try another file or random phase."""
        bound = dict(variant)
        if "left_latent_path" in bound and "right_latent_path" in bound:
            latents, stride, phase = self._load_stereo_latent_candidate(
                bound, metadata_only=metadata_only, allow_payload_fallback=allow_payload_fallback)
            return latents, stride, phase, bound
        data = (self._load_latent_geometry_metadata(bound["latent_path"],
                                                   allow_payload_fallback=allow_payload_fallback) if metadata_only
                else smart_load_pt(bound["latent_path"], self.cache_dir, map_location="cpu"))
        if isinstance(data, dict) and "phases" in data:
            phases = data["phases"]
            allowed = [i for i in range(len(phases))
                       if self.latent_phases_set is None or i in self.latent_phases_set]
            if not allowed:
                raise ValueError("latent file has no usable phases after filtering")
            member = bound.get("strict_phase_index")
            if member is None:
                if not select_phase and len(allowed) != 1:
                    raise ValueError("multi-phase latent requires an explicit strict_phase_index")
                member = allowed[0]
            if isinstance(member, bool) or not isinstance(member, int) or member not in allowed:
                raise ValueError("strict_phase_index is not an allowed phase member")
            bound["strict_phase_index"] = member
            selected = phases[member]
            latents = selected["latents"]
            stride = int(data.get("frame_stride", selected.get("frame_stride", 6)))
            phase = int(selected.get("phase_start_30", 0))
        elif isinstance(data, torch.Tensor):
            latents = data
            stride = self.video_action_freq_ratio
            phase = int(bound.get("phase_start_30", 0))
        else:
            latents = data["latents"]
            stride = int(data.get("frame_stride", self.video_action_freq_ratio))
            phase = int(data.get("phase_start_30", bound.get("phase_start_30", 0)))
        return latents.float(), stride, phase, bound

    def _strict_latent_geometry(self, ep, latents, phase_start_30):
        if latents.ndim not in (4, 5) or any(int(n) <= 0 for n in latents.shape):
            raise ValueError("latent must have nonempty C,T,H,W or V,C,T,H,W geometry")
        total = self._latent_total_frames(latents)
        if ep.get("action_schema") == "handson-stage2/raw54-v1":
            from .handson_adapter import full_latents
            total = min(total, full_latents(ep["handson_action_frames"], phase_start_30))
            latents = self._latent_crop_time(latents, 0, total)
        return latents, int(total)

    def _window_spec_template(self, index: int) -> Dict[str, Any]:
        """Build catalog spec without JSON serialization (internal hot path)."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.episode_files):
            raise IndexError("strict catalog episode index is out of range")
        ep = self.episode_files[index]
        candidates = self._strict_latent_candidates(ep)
        if not candidates:
            raise ValueError("episode has no latent candidates")
        total = ep.get("sc_total_latent_frames", ep.get("total_latent_frames"))
        total = int(total) if total is not None else None
        variant = dict(candidates[0])
        phase = int(variant.get("phase_start_30", 0))
        phase_bound = self.latent_phases_set is not None and self.latent_phases_set == {phase}
        if phase_bound:
            variant["strict_phase_start_30"] = phase
            if "latent_path" in variant:
                variant["strict_phase_index"] = phase
        return {
            "episode_index": index,
            "episode_id": self._episode_identity(ep),
            "source_id": self._sampling_source_identity(ep),
            "bucket_id": str(ep.get("bucket_key") or "unknown"),
            "total_latents": total,
            "window_size": self._window_size_for_episode(ep, index),
            "latent_variant": variant,
            "geometry_known": False,
            "request_geometry_known": False,
        }

    def window_spec(self, index: int, resolve_geometry: bool = False) -> Dict[str, Any]:
        """JSON catalog entry; only selected episodes need latent-file discovery.

        Index lengths are advisory, even when present: they may describe a
        different export than the selected payload. Only an actual latent read
        makes request geometry known. Discovery binds exactly the first sorted
        variant and its deterministic phase member; I/O errors propagate to the
        coordinator.
        """
        spec = self._window_spec_template(index)
        if resolve_geometry:
            return self._resolve_window_spec_geometry(spec)
        return json.loads(json.dumps(spec, allow_nan=False))

    def _resolve_window_spec_geometry(self, spec: Dict[str, Any], *, allow_payload_fallback=True) -> Dict[str, Any]:
        """Resolve a main-thread-selected spec without discovery or RNG changes.

        The caller owns selection/order and any cache publication. This method
        copies its input, reads only its already bound candidate, and never
        mutates dataset caches or a coordinator ledger. Remote/non-zip fallback
        retains the original CPU reader and its normal I/O/error behavior only
        when allowed. Threaded prefill passes False; the main thread handles
        unsupported formats later through the normal True/default path.
        """
        spec = json.loads(json.dumps(spec, allow_nan=False))
        index = spec["episode_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.episode_files):
            raise IndexError("strict catalog episode index is out of range")
        ep = self.episode_files[index]
        identity = {"episode_id": self._episode_identity(ep),
                    "source_id": self._sampling_source_identity(ep),
                    "bucket_id": str(ep.get("bucket_key") or "unknown")}
        if any(spec.get(key) != value for key, value in identity.items()):
            raise ValueError("selected geometry spec changed episode/source/bucket identity")
        try:
            latents, stride, phase, variant = self._strict_load_latent_variant(
                spec["latent_variant"], select_phase=True, metadata_only=True,
                allow_payload_fallback=allow_payload_fallback,
            )
        except (RuntimeError, NotImplementedError) as exc:
            # Scalar tensor metadata needs its value, not just its shape.
            if not allow_payload_fallback or "meta" not in str(exc).lower():
                raise
            latents, stride, phase, variant = self._strict_load_latent_variant(
                spec["latent_variant"], select_phase=True,
            )
        latents, total = self._strict_latent_geometry(ep, latents, phase)
        variant.update(strict_expected_shape=list(latents.shape), strict_frame_stride=int(stride),
                       strict_phase_start_30=int(phase))
        spec.update(total_latents=total, latent_variant=variant, frame_stride=int(stride),
                    phase_start_30=int(phase), latent_shape=list(latents.shape),
                    window_size=self._effective_window_size_for_episode(
                        ep, index, total, spec["window_size"]), geometry_known=True,
                    request_geometry_known=True)
        return spec

    def _strict_window_actions(self, ep, start, window_size, total_latents, frame_stride, phase_start_30):
        """Actions, states and tracking masks for one strict window, padded to it.

        The action half of ``read_window``. Planning-time screening runs this
        same code, so a window judged there is judged exactly as the reader
        would judge it. Returns ``(error_code, detail, parts)``.
        """
        valid_len = actual_latents = min(total_latents - start, window_size)
        action_path = self._sample_action_path(ep)
        if action_path is None:
            return "MISSING_ACTION_PATH", "episode has no action path", None
        loaded = self._load_actions_and_states(
            ep, action_path, start, actual_latents, frame_stride, phase_start_30, self.use_state)
        mask_tracking = bool(getattr(self, "mask_invalid_hand_tracking", False))
        action_valid_mask = state_valid_mask = None
        if mask_tracking:
            states, actions, action_valid_mask, state_valid_mask = loaded
        else:
            states, actions = loaded
        target_action_len = window_size * self.actions_per_latent
        if actions.shape[0] < target_action_len:
            pad_len = target_action_len - actions.shape[0]
            actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
            if mask_tracking:
                action_valid_mask = torch.cat([action_valid_mask, action_valid_mask[-1:].expand(pad_len, -1)], dim=0)
        if states is not None and states.shape[0] < window_size:
            pad_len = window_size - states.shape[0]
            states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)
            if mask_tracking:
                state_valid_mask = torch.cat([state_valid_mask, state_valid_mask[-1:].expand(pad_len, -1)], dim=0)
        if not self._actions_pass_abs_filter(actions, states, valid_len):
            return "ACTION_ABS_FILTER", "requested window fails the action/state absolute filter", None
        if not self._actions_pass_wrist_xyz_abs_filter(actions, states, valid_len):
            return "WRIST_ABS_FILTER", "requested window fails the wrist absolute filter", None
        return None, "", dict(actions=actions, states=states, valid_len=valid_len, mask_tracking=mask_tracking,
                              action_valid_mask=action_valid_mask, state_valid_mask=state_valid_mask)

    def _strict_bootstrap_window(self, parts, latents_crop, start, window_size, total_latents):
        """Bootstrap one strict window and apply the chunk-relative wrist filter.

        Bootstrap reads only the latents' time length, so screening may pass a
        stand-in tensor of that length. Returns ``(passes, window)``.
        """
        actions, states, valid_len = parts["actions"], parts["states"], parts["valid_len"]
        extra = {}
        if parts["mask_tracking"]:
            extra["action_valid_mask"] = parts["action_valid_mask"]
            extra["_state_valid_mask_blocks"] = parts["state_valid_mask"].repeat_interleave(
                self.actions_per_latent, dim=0)
        values = self._build_value_targets(start, window_size, total_latents)
        latents_crop, actions, states, values, valid_len, extra = maybe_bootstrap_start_window_sample(
            latents_crop, actions, valid_len, start, actions_per_latent=self.actions_per_latent,
            states=states, values=values, chunk_size=self.bootstrap_chunk_size,
            enabled=self.bootstrap_start_window, extra_action_sequences=extra)
        window = dict(latents=latents_crop, actions=actions, states=states, values=values, valid_len=valid_len)
        blocks = extra.pop("_state_valid_mask_blocks", None)
        if blocks is not None:
            window["state_valid_mask"] = blocks.reshape(window_size, self.actions_per_latent, 2)[:, 0]
        window.update(extra)
        passes = self._actions_pass_wrist_xyz_relative_filter(
            actions, states, valid_len, window.get("action_valid_mask"), window.get("state_valid_mask"))
        return passes, window

    def strict_window_verdict(self, request: Dict[str, Any]) -> Optional[str]:
        """The reader's filter verdict for one request, without reading latents.

        Returns the error code ``read_window`` would return for a deterministic
        action-side rejection, or None when it would pass these checks (or when
        the request lacks resolved timing; the reader then decides alone).
        """
        variant = request.get("latent_variant") or {}
        stride, phase = variant.get("strict_frame_stride"), variant.get("strict_phase_start_30")
        index, start = request.get("episode_index"), request.get("window_start")
        window_size, total = request.get("window_size"), request.get("expected_total_latents")
        if stride is None or phase is None or not isinstance(index, int) or not 0 <= index < len(self.episode_files):
            return None
        if start > max(0, total - window_size):
            return None
        ep = self.episode_files[index]
        code, _, parts = self._strict_window_actions(ep, start, window_size, total, int(stride), int(phase))
        if code is not None:
            return code
        stand_in = torch.zeros(1, window_size, 1, 1)
        passes, _ = self._strict_bootstrap_window(parts, stand_in, start, window_size, total)
        return None if passes else "WRIST_RELATIVE_FILTER"

    def read_window(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Read exactly one coordinator request, with no local replacement/drop.

        Only ``sample`` carries local tensors. The remaining fields are small
        JSON metadata suitable for a rank-coordinator status exchange.
        """
        result = {"request_id": request.get("request_id")
                  if isinstance(request, dict) and isinstance(request.get("request_id"), str) else None,
                  "status": "INVALID_SAMPLE", "sample": None, "observed_geometry": {},
                  "error_code": None, "error_detail": ""}
        stage = "INVALID_REQUEST"

        def fail(code, detail, *, retryable=False):
            result.update(status="RETRYABLE_ERROR" if retryable else "INVALID_SAMPLE",
                          sample=None, error_code=code, error_detail=str(detail)[:2048])
            return result

        try:
            request = json.loads(json.dumps(request, allow_nan=False))
            if not isinstance(request.get("request_id"), str) or not request["request_id"]:
                return fail(stage, "request_id must be a nonempty string")
            result["request_id"] = request["request_id"]
            for field in ("episode_index", "window_start", "window_size", "expected_total_latents"):
                if isinstance(request.get(field), bool) or not isinstance(request.get(field), int):
                    return fail(stage, field + " must be an integer")
            index, start, window_size = request["episode_index"], request["window_start"], request["window_size"]
            if not 0 <= index < len(self.episode_files):
                return fail(stage, "episode_index is out of range")
            if window_size <= 0 or start < 0 or request["expected_total_latents"] <= 0:
                return fail(stage, "window geometry must be positive with a nonnegative start")
            ep = self.episode_files[index]
            actual_identity = {"episode_id": self._episode_identity(ep),
                               "source_id": self._sampling_source_identity(ep),
                               "bucket_id": str(ep.get("bucket_key") or "unknown")}
            result["observed_geometry"].update(actual_identity, window_size=window_size, window_start=start)
            if any(request.get(k) != v for k, v in actual_identity.items()):
                return fail("IDENTITY_MISMATCH", "episode/source/bucket does not match the requested catalog entry")
            chosen = request.get("latent_variant")
            if not isinstance(chosen, dict) or self._latent_candidate_key(chosen) not in {
                    self._latent_candidate_key(c) for c in self._strict_latent_candidates(ep)}:
                return fail("LATENT_VARIANT_MISMATCH", "requested paths/phase are not this episode's latent candidate")
            stage = "LATENT_READ_ERROR"
            latents, frame_stride, phase_start_30, bound = self._strict_load_latent_variant(chosen)
            latents, total_latents = self._strict_latent_geometry(ep, latents, phase_start_30)
            observed = result["observed_geometry"]
            observed.update(total_latents=total_latents, latent_shape=list(latents.shape),
                            frame_stride=int(frame_stride), phase_start_30=int(phase_start_30),
                            num_views=int(latents.shape[0]) if latents.ndim == 5 else 1,
                            latent_height=int(latents.shape[-2]), latent_width=int(latents.shape[-1]),
                            latent_variant=bound)
            if total_latents != request["expected_total_latents"]:
                return fail("TOTAL_LATENTS_MISMATCH", "loaded latent length differs from the exact plan")
            if (chosen.get("strict_expected_shape") is not None and list(latents.shape) != chosen["strict_expected_shape"]
                    or chosen.get("strict_frame_stride", frame_stride) != frame_stride
                    or chosen.get("strict_phase_start_30", phase_start_30) != phase_start_30):
                return fail("LATENT_GEOMETRY_MISMATCH", "loaded shape/timeline differs from the resolved catalog")
            if self.min_total_latent_frames is not None and total_latents < self.min_total_latent_frames:
                return fail("MIN_TOTAL_LATENTS", "loaded latent length is below the configured minimum")
            expected_window = self._effective_window_size_for_episode(
                ep, index, total_latents, self._window_size_for_episode(ep, index))
            if window_size != expected_window:
                return fail("WINDOW_SIZE_MISMATCH", "request changes the configured episode window size")
            if start > max(0, total_latents - window_size):
                return fail("WINDOW_OUT_OF_RANGE", "requested start cannot supply the configured window")
            valid_len = actual_latents = min(total_latents - start, window_size)
            latents_crop = self._latent_crop_time(latents, start, start + actual_latents)
            latents_crop = self._latent_pad_time(latents_crop, window_size - actual_latents)
            stage = "ACTION_READ_ERROR"
            code, detail, parts = self._strict_window_actions(
                ep, start, window_size, total_latents, frame_stride, phase_start_30)
            if code is not None:
                return fail(code, detail)
            stage = "LANGUAGE_READ_ERROR"
            # Augmentation may inspect/prune candidates on either a successful
            # fallback or a failed load. Neither outcome may change the next
            # strict request's candidate pool in only this worker.
            language_key = self._episode_identity(ep)
            language_candidates = list(self._resolve_language_candidates(ep))
            try:
                language_embedding = self._load_language_embedding(ep, sample_idx=index)
            finally:
                self._language_candidates_cache[language_key] = language_candidates
            if language_embedding is None:
                return fail("MISSING_LANGUAGE", "no loadable language embedding for the requested episode", retryable=True)
            stage = "SAMPLE_ASSEMBLY_ERROR"
            passes, window = self._strict_bootstrap_window(
                parts, latents_crop, start, window_size, total_latents)
            sample = {
                "latents": window["latents"], "actions": window["actions"], "valid_len": window["valid_len"],
                "latent_start": int(start),
                SAMPLE_CLASS_KEY: int(SampleClass.TASK_IRRELEVANT if self.treat_as_irrelevant else SampleClass.MAIN),
                "is_success": int(not self.treat_as_irrelevant),
                "language_embedding": language_embedding,
                "debug_episode": {"dataset": ep.get("dataset_name"), "task": ep.get("task_name"),
                    "episode": ep.get("episode_name"), "storage_key": ep.get("episode_storage_key"),
                    "total_latents": total_latents, "window_size": window_size, "start": start,
                    "valid_len": valid_len, "latent_path": chosen.get("latent_path"),
                    "left_latent_path": chosen.get("left_latent_path"), "right_latent_path": chosen.get("right_latent_path")},
                "strict_window_request": request,
            }
            if getattr(self, "wrist_xyz_per_dataset", False):
                sample["wrist_stats_source"] = self._stats_dataset_name(ep) or "_default"
            for key in ("states", "state_valid_mask", "action_valid_mask", "values"):
                if window.get(key) is not None:
                    sample[key] = window[key]
            if not passes:
                return fail("WRIST_RELATIVE_FILTER", "requested window fails the chunk-relative wrist filter")
            observed.update(valid_len=int(window["valid_len"]), action_shape=list(window["actions"].shape),
                            returned_latent_shape=list(window["latents"].shape))
            result.update(status="OK", sample=sample)
            return result
        except (OSError, TimeoutError, ConnectionError) as exc:
            return fail(stage, type(exc).__name__ + ": " + str(exc), retryable=True)
        except Exception as exc:
            return fail(stage, type(exc).__name__ + ": " + str(exc))

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if isinstance(idx, dict) and "request_id" in idx:
            return self.read_window(idx)
        if self.val and self._frozen_val_indices is not None and not self._freeze_build_mode:
            if not self._frozen_val_indices:
                return None
            idx = self._frozen_val_indices[idx % len(self._frozen_val_indices)]
        # The batch sampler packs (episode_index, visit_ordinal) into one int so
        # workers stay stateless.  ``visit_ordinal`` picks the latent window.
        visit_ordinal = 0
        if self.window_start_sampling:
            idx, visit_ordinal = unpack_index(idx)
        failure_reasons: List[str] = []

        def record_failure(ep: Optional[Dict[str, Any]], reason: str) -> None:
            if ep is None:
                failure_reasons.append(f"attempt=no_episode reason={reason}")
                return
            failure_reasons.append(
                "episode=%s dataset=%s task=%s storage_key=%s episode_path=%s qpos_path=%s lang_path=%s reason=%s" % (
                    ep.get("episode_name", "?"),
                    ep.get("dataset_name"),
                    ep.get("task_name"),
                    ep.get("episode_storage_key"),
                    ep.get("episode_path"),
                    ep.get("qpos_path"),
                    ep.get("lang_path"),
                    reason,
                )
            )

        def maybe_drop_episode(ep: Dict[str, Any], reason: str) -> None:
            if not self._freeze_build_mode:
                self._drop_episode(ep, reason)

        for attempt in range(8):
            ep = self._episode_for_index(idx, attempt)
            if ep is None:
                record_failure(None, "episode_for_index returned None")
                continue
            window_size = self._window_size_for_episode(ep, idx)

            try:
                candidates = self._resolve_latent_candidates(ep)
                if not candidates:
                    reason = "no latent candidates"
                    record_failure(ep, reason)
                    maybe_drop_episode(ep, reason)
                    continue
                chosen = None
                latents = None
                total_latents = 0
                last_error = None
                candidate_order = sorted(candidates, key=self._latent_candidate_key) if self.val else random.sample(candidates, k=len(candidates))
                if self.latent_max_attempts is not None:
                    candidate_order = candidate_order[: self.latent_max_attempts]
                attempted_candidates = len(candidate_order)
                for candidate in candidate_order:
                    try:
                        latents, frame_stride, phase_start_30 = self._load_latent_candidate(candidate)
                        candidate_total_latents = self._latent_total_frames(latents)
                        if ep.get("action_schema") == "handson-stage2/raw54-v1":
                            from .handson_adapter import full_latents
                            candidate_total_latents = min(candidate_total_latents,
                                full_latents(ep['handson_action_frames'], phase_start_30))
                            latents = self._latent_crop_time(latents, 0, candidate_total_latents)
                        if (
                            self.min_total_latent_frames is not None
                            and candidate_total_latents < self.min_total_latent_frames
                        ):
                            last_error = ValueError(
                                f"total_latents={candidate_total_latents} < "
                                f"min_total_latent_frames={self.min_total_latent_frames}"
                            )
                            continue
                        candidate_window_size = self._effective_window_size_for_episode(
                            ep,
                            idx,
                            candidate_total_latents,
                            window_size,
                        )
                        if candidate_window_size <= 0:
                            last_error = ValueError(
                                f"invalid dynamic window_size={candidate_window_size} "
                                f"for total_latents={candidate_total_latents}"
                            )
                            continue
                        total_latents = candidate_total_latents
                        window_size = candidate_window_size
                        chosen = candidate
                        break
                    except Exception as e:
                        last_error = e
                if chosen is None or latents is None:
                    reason = f"all latent candidates failed after {attempted_candidates} attempts: {last_error}"
                    record_failure(ep, reason)
                    maybe_drop_episode(ep, reason)
                    continue

                start, valid_len = self._select_latent_window(
                    idx,
                    total_latents,
                    window_size,
                    visit_ordinal=visit_ordinal,
                    episode_key=self._episode_identity(ep),
                )
                actual_latents = min(total_latents - start, window_size)
                if total_latents > window_size:
                    latents_crop = self._latent_crop_time(latents, start, start + window_size)
                elif total_latents < window_size:
                    latents_crop = self._latent_pad_time(latents, window_size - total_latents)
                else:
                    latents_crop = latents

                action_path = self._sample_action_path(ep)
                if action_path is None:
                    reason = "missing action path"
                    record_failure(ep, reason)
                    maybe_drop_episode(ep, reason)
                    continue
                try:
                    loaded_actions = self._load_actions_and_states(
                        ep, action_path, start, actual_latents, frame_stride, phase_start_30, self.use_state
                    )
                    mask_tracking = bool(getattr(self, "mask_invalid_hand_tracking", False))
                    if mask_tracking:
                        states, actions, action_valid_mask, state_valid_mask = loaded_actions
                    else:
                        states, actions = loaded_actions
                except Exception as e:
                    reason = f"action/state load failed from {action_path}: {e}"
                    record_failure(ep, reason)
                    maybe_drop_episode(ep, reason)
                    continue

                target_action_len = window_size * self.actions_per_latent
                if actions.shape[0] < target_action_len:
                    pad_len = target_action_len - actions.shape[0]
                    actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)
                    if mask_tracking:
                        action_valid_mask = torch.cat([
                            action_valid_mask, action_valid_mask[-1:].expand(pad_len, -1)
                        ], dim=0)
                if states is not None and states.shape[0] < window_size:
                    pad_len = window_size - states.shape[0]
                    states = torch.cat([states, states[-1:].expand(pad_len, -1)], dim=0)
                    if mask_tracking:
                        state_valid_mask = torch.cat([
                            state_valid_mask, state_valid_mask[-1:].expand(pad_len, -1)
                        ], dim=0)
                if not self._actions_pass_abs_filter(actions, states, valid_len):
                    reason = f"action/state abs filter failed start={int(start)} valid_len={int(valid_len)}"
                    record_failure(
                        ep,
                        reason,
                    )
                    maybe_drop_episode(ep, reason)
                    logger.debug(
                        "Skipping sample with action/state abs > %.6g (%s | dataset=%s task=%s storage_key=%s start=%s valid_len=%s)",
                        float(self.action_abs_max),
                        ep.get("episode_name", "?"),
                        ep.get("dataset_name"),
                        ep.get("task_name"),
                        ep.get("episode_storage_key"),
                        int(start),
                        int(valid_len),
                    )
                    continue
                if not self._actions_pass_wrist_xyz_abs_filter(actions, states, valid_len):
                    reason = f"wrist xyz abs filter failed start={int(start)} valid_len={int(valid_len)}"
                    record_failure(
                        ep,
                        reason,
                    )
                    maybe_drop_episode(ep, reason)
                    self._wrist_xyz_filter_skips += 1
                    if logger.isEnabledFor(logging.DEBUG) and (
                        self._wrist_xyz_filter_skips <= 20 or self._wrist_xyz_filter_skips % 100 == 0
                    ):
                        logger.debug(
                            "Skipping sample with wrist xyz abs > %.6g "
                            "(total_skipped=%d | %s | dataset=%s task=%s storage_key=%s start=%s valid_len=%s)",
                            float(self.wrist_xyz_abs_max),
                            self._wrist_xyz_filter_skips,
                            ep.get("episode_name", "?"),
                            ep.get("dataset_name"),
                            ep.get("task_name"),
                            ep.get("episode_storage_key"),
                            int(start),
                            int(valid_len),
                        )
                    continue

                language_candidates = self._resolve_language_candidates(ep)
                language_embedding = self._load_language_embedding(ep, sample_idx=idx)
                if language_embedding is None:
                    reason = (
                        "language embedding unavailable; candidates=%s lang_root=%s lang_path=%s"
                        % (
                            language_candidates,
                            ep.get("lang_root"),
                            ep.get("lang_path"),
                        )
                    )
                    record_failure(ep, reason)
                    maybe_drop_episode(ep, reason)
                    continue
                sample: Dict[str, Any] = {
                    "latents": latents_crop,
                    "actions": actions,
                    "valid_len": valid_len,
                    "latent_start": int(start),
                    SAMPLE_CLASS_KEY: int(
                        SampleClass.TASK_IRRELEVANT
                        if self.treat_as_irrelevant
                        else SampleClass.MAIN
                    ),
                    "is_success": int(not self.treat_as_irrelevant),
                    "debug_episode": {
                        "dataset": ep.get("dataset_name"),
                        "task": ep.get("task_name"),
                        "episode": ep.get("episode_name"),
                        "storage_key": ep.get("episode_storage_key"),
                        "total_latents": int(total_latents),
                        "window_size": int(window_size),
                        "start": int(start),
                        "valid_len": int(valid_len),
                        "latent_path": chosen.get("latent_path"),
                        "left_latent_path": chosen.get("left_latent_path"),
                        "right_latent_path": chosen.get("right_latent_path"),
                    },
                }
                if getattr(self, "wrist_xyz_per_dataset", False):
                    sample["wrist_stats_source"] = self._stats_dataset_name(ep) or "_default"
                sample["language_embedding"] = language_embedding
                if states is not None:
                    sample["states"] = states
                extra_action_sequences: Dict[str, torch.Tensor] = {}
                if mask_tracking:
                    extra_action_sequences["action_valid_mask"] = action_valid_mask
                    # Use the exact action-block bootstrap for state validity
                    # as well; each latent anchor is repeated across its block.
                    extra_action_sequences["_state_valid_mask_blocks"] = (
                        state_valid_mask.repeat_interleave(self.actions_per_latent, dim=0)
                    )
                values = self._build_value_targets(start, window_size, total_latents)
                latents_crop, actions, states, values, valid_len, extra_action_sequences = maybe_bootstrap_start_window_sample(
                    latents_crop,
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
                sample["latents"] = latents_crop
                sample["actions"] = actions
                sample["valid_len"] = valid_len
                if states is not None:
                    sample["states"] = states
                if extra_action_sequences:
                    state_mask_blocks = extra_action_sequences.pop("_state_valid_mask_blocks", None)
                    if state_mask_blocks is not None:
                        sample["state_valid_mask"] = state_mask_blocks.reshape(
                            window_size, self.actions_per_latent, 2
                        )[:, 0]
                    sample.update(extra_action_sequences)
                if values is not None:
                    sample["values"] = values
                if not self._actions_pass_wrist_xyz_relative_filter(
                    actions, states, valid_len,
                    sample.get("action_valid_mask"), sample.get("state_valid_mask"),
                ):
                    # This predicate describes one window, not the episode.
                    # Strict training coordinates replacements globally; legacy
                    # evaluation may omit this fixed window and count the samples
                    # actually returned. Do not select another window here.
                    logger.warning(
                        "Rejecting single window: wrist xyz relative filter failed "
                        "limit=%s (episode=%s dataset=%s task=%s start=%s valid_len=%s)",
                        self.wrist_xyz_relative_max,
                        ep.get("episode_name", "?"),
                        ep.get("dataset_name"),
                        ep.get("task_name"),
                        int(start),
                        int(valid_len),
                    )
                    return None
                return sample
            except Exception as e:
                record_failure(ep, f"sample exception: {e}")
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
        logger.debug(
            "PretrainDataset returning None for idx=%s after 8 attempts; recent_failures=[%s]",
            idx,
            " ; ".join(failure_reasons[-8:]) if failure_reasons else "no recorded failures",
        )
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
