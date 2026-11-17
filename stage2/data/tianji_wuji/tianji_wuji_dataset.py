"""Tianji/Wuji dexterous-hand latent dataset."""

from __future__ import annotations

import json
import hashlib
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
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from utils.common import maybe_bootstrap_start_window_sample


logger = logging.getLogger(__name__)

SIDES = ("left", "right")
ACTION_DIM = 58

SAMPLE_CLASS_MAIN = SampleClass.MAIN
SAMPLE_CLASS_TASK_IRRELEVANT = SampleClass.TASK_IRRELEVANT
SAMPLE_CLASS_SUBOPTIMAL = SampleClass.SUBOPTIMAL

# Cross-attention keys are a fixed-length block: `ar_collate_fn` zero-pads every
# caption to this many tokens and `Motus._t5_embed` refuses any other length.
# Anything longer is cut by the collate from the *tail*, which is where the
# instruction sits once environment context is prepended, so the dataset has to
# do its own budgeting instead. `data/dataset.py` and `models/wan_model.py`
# mirror this number.
TEXT_CONTEXT_LEN = 512


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
    if raw_action_hz % train_action_hz != 0:
        raise ValueError(f"raw_action_hz must be divisible by train_action_hz, got {raw_action_hz}/{train_action_hz}")
    if train_action_hz % video_hz != 0:
        raise ValueError(f"train_action_hz must be divisible by video_hz, got {train_action_hz}/{video_hz}")

    action_downsample = raw_action_hz // train_action_hz
    derived_upsample = train_action_hz // video_hz
    return video_hz, raw_action_hz, train_action_hz, action_downsample, derived_upsample


def _stream_hz(payload: Any) -> Optional[float]:
    """The sampling rate this particular episode's stream .pt declares.

    Every exported stream carries `hz` (and a float `timeline_hz`). Captures
    straddling a schema change ship mixed rates inside one task dir, so the
    per-episode field is the only trustworthy source; the config rate is a
    fallback for old exports that predate the field.
    """
    if not isinstance(payload, dict):
        return None
    for key in ("hz", "timeline_hz"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if rate > 0:
            return rate
    return None


def _episode_rate_stride(
    stored_hz: Optional[float],
    target_hz: int,
    *,
    fallback: int,
    stream: str,
    episode: Any,
) -> int:
    """Decimation stride taking one episode's `stream` from its own rate to `target_hz`.

    Raises rather than rounding when the episode cannot be normalized exactly: a
    wrong stride does not fail loudly downstream, it silently reads actions from
    the wrong timestamps and pads the tail with a repeated frame.
    """
    if stored_hz is None:
        return max(int(fallback), 1)
    ratio = float(stored_hz) / float(target_hz)
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(
            f"{episode}: {stream} stream is {stored_hz:g} Hz, which is not an integer "
            f"multiple of the {target_hz} Hz training rate (ratio {ratio:.4f}). This "
            f"episode cannot be resampled by striding; exclude it or re-export it."
        )
    return stride


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

    #: ``__init__`` sets this from ``dataset.suboptimal.sample_fraction``.
    #: Declared on the class as well so ``_configure_fail_sampling`` reads
    #: "there is no sub-optimal pool" rather than raising on an instance that
    #: was built without ``__init__``.
    suboptimal_sample_fraction: Optional[float] = None

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
        single_latent_name: Optional[str] = None,
        language_subpath: str = "text_features/UMT5/instruction.pt",
        instruction_bank_path: Optional[str] = None,
        environment_mix_prob: float = 0.0,
        environment_source: str = "task_pool",
        environment_variant: str = "environment_v2.pt",
        text_pool_source: str = "episode",
        tjwj_subpath: str = "tjwj.pt",
        action_stats_path: str = "data/tianji_wuji/wj_stat.json",
        qpos_normalization: str = "mean_std",
        task_episode_skip_first: Optional[List[Dict[str, Any]]] = None,  # T22: per-task drop first N episodes
        max_latent_window: int = 10,
        video_hz: int = 30,
        video_action_freq_ratio: int = 6,
        action_upsample_ratio: int = 2,
        action: Optional[Dict[str, Any]] = None,
        raw_action_hz: Optional[int] = None,
        train_action_hz: Optional[int] = None,
        action_mode: str = "qpos20_rel6d",
        action_from_state: bool = True,
        use_state: bool = False,
        val: bool = False,
        validation_fraction: float = 0.0,
        validation_seed: int = 42,
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        max_episodes: Optional[int] = None,
        include_fail: bool = False,
        fail_task_name: Optional[str] = None,
        fail_sample_fraction: Optional[float] = None,
        suboptimal: Optional[Dict[str, Any]] = None,
        include_old: bool = False,
        cache_dir: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        use_index_cache: bool = True,
        manifest_path: Optional[str] = None,
        manifest_sample_type: Optional[str] = None,
        value_target: Optional[str] = None,
        progress_c: float = 0.01,
        task_irrelevant_decaying_curve: bool = False,
        treat_as_irrelevant: bool = False,
        bootstrap_start_window: bool = False,
        bootstrap_chunk_size: int = 2,
        bootstrap_start_window_prob: float = 0.5,
        view_dropout: Optional[Dict[str, Any]] = None,
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
            self.action_upsample_ratio,
        ) = _resolve_rate_config(
            video_hz=video_hz,
            action_cfg=action,
            raw_action_hz=raw_action_hz,
            train_action_hz=train_action_hz,
            action_upsample_ratio=int(action_upsample_ratio),
        )
        self.actions_per_latent = 4 * self.video_action_freq_ratio * self.action_upsample_ratio
        self.use_state = bool(use_state)
        self.action_from_state = bool(action_from_state)
        self._logged_arm_action_endpose = False
        self.val = bool(val)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be in [0, 1), got "
                f"{self.validation_fraction}"
            )
        self.samples_per_episode = 10
        self.task_mode = str(task_mode)
        self.task_name = _to_plain_container(task_name)
        self.include_fail = bool(include_fail)
        self.fail_task_name = _to_plain_container(fail_task_name)
        self.fail_sample_fraction = (
            None if fail_sample_fraction is None else float(fail_sample_fraction)
        )
        if (
            self.fail_sample_fraction is not None
            and not 0.0 <= self.fail_sample_fraction <= 1.0
        ):
            raise ValueError(
                "fail_sample_fraction must be in [0, 1], got "
                f"{self.fail_sample_fraction}"
            )
        self._init_suboptimal(suboptimal)
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
        self.task_irrelevant_decaying_curve = bool(task_irrelevant_decaying_curve)
        self.treat_as_irrelevant = treat_as_irrelevant
        self.bootstrap_start_window = bool(bootstrap_start_window)
        self.bootstrap_chunk_size = int(bootstrap_chunk_size)
        self.bootstrap_start_window_prob = float(min(max(bootstrap_start_window_prob, 0.0), 1.0))
        self._init_view_dropout(view_dropout)
        self._init_instruction_bank(
            instruction_bank_path,
            environment_mix_prob,
            environment_source,
            environment_variant,
            text_pool_source,
        )
        if self.task_mode == "single" and not self.task_name:
            raise ValueError("Single task mode requires task_name")

        self.action_stats = self._load_json(action_stats_path)
        self.qpos_normalization = str(qpos_normalization or "mean_std")
        self._init_qpos_normalization_stats()
        self.episodes = self._mark_suboptimal_episodes(self._load_episode_index())
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
        self.episodes = self._select_validation_partition(self.episodes)
        if not self.episodes:
            partition = "validation" if self.val else "training"
            raise RuntimeError(
                f"No Tianji/Wuji {partition} episodes found under {self.dataset_dir}"
            )
        self._configure_fail_sampling()
        self._audit_instruction_bank()
        self._build_success_text_pool()

        logger.info(
            "TianjiWujiDataset: episodes=%d, max_latent_window=%d, actions_per_latent=%d, "
            "action_hz=%d(fallback)->%d/downsample=%d, qpos_norm=%s, "
            "latent_resolution=%s, action_from_state=%s, "
            "bootstrap_start=%s/%d@p=%.2f, value_target=%s, source=%s",
            len(self.episodes),
            self.max_latent_window,
            self.actions_per_latent,
            self.raw_action_hz,
            self.train_action_hz,
            self.action_downsample_stride,
            self.qpos_normalization,
            self.latent_resolution,
            self.action_from_state,
            self.bootstrap_start_window,
            self.bootstrap_chunk_size,
            self.bootstrap_start_window_prob,
            self.value_target,
            self._discovery_source(),
        )

    def _init_suboptimal(self, suboptimal: Optional[Dict[str, Any]]) -> None:
        """Read the opt-in ``suboptimal`` block, which only says *which* episodes.

        Sub-optimal means the episode reaches the goal but by a path not worth
        imitating (human intervention, a clumsy recovery, a different strategy).
        What that costs and buys — no action loss, success-shaped value target,
        WM/VM routing — lives with those mechanisms, not here.

        An absent or disabled block leaves every attribute inert, so no episode is
        ever marked and the two-class behavior is reached through the same code.
        """
        cfg = _mapping_to_plain_dict(suboptimal) if suboptimal is not None else {}
        self.suboptimal_enabled = bool(cfg.get("enabled", False))
        # Parsed even when disabled: `_skip_path` needs the name to hide the
        # directory from an index that was never meant to see it.
        self.suboptimal_dir_name = str(cfg.get("dir_name", "suboptimal")).strip("/").lower()
        self.suboptimal_from_directory = bool(cfg.get("from_directory", True))
        # `enabled` is the master switch, so flipping it off must not leave a
        # sample_fraction behind that then fails against an empty pool.
        self.suboptimal_sample_fraction = (
            None
            if not self.suboptimal_enabled or cfg.get("sample_fraction") is None
            else float(cfg.get("sample_fraction"))
        )
        if (
            self.suboptimal_sample_fraction is not None
            and not 0.0 <= self.suboptimal_sample_fraction <= 1.0
        ):
            raise ValueError(
                "suboptimal.sample_fraction must be in [0, 1], got "
                f"{self.suboptimal_sample_fraction}"
            )
        self.suboptimal_manifest_sample_types = {
            str(item).lower()
            for item in self._as_pattern_list(
                _to_plain_container(cfg.get("manifest_sample_types", ["suboptimal"]))
            )
        }
        self.suboptimal_task_patterns = self._as_pattern_list(
            _to_plain_container(cfg.get("task_name"))
        )
        list_path = cfg.get("episode_list_path")
        self.suboptimal_episode_list_path = (
            str(Path(str(list_path)).expanduser()) if list_path else None
        )
        self.suboptimal_episode_list_labels = {
            str(item).lower()
            for item in self._as_pattern_list(
                _to_plain_container(cfg.get("episode_list_labels", ["suboptimal"]))
            )
        }
        label_key = cfg.get("episode_list_label_key")
        self.suboptimal_episode_list_label_key = str(label_key) if label_key else None

    def _load_suboptimal_episode_paths(self) -> set:
        """Episode paths named by ``suboptimal.episode_list_path``.

        Two shapes are accepted because triage tooling produces both: a text file
        of episode paths, and JSON carrying a per-episode label (a list of rows, a
        ``{"episodes": [...]}`` envelope, or a path -> label mapping). A row whose
        label is not recognized is kept, so an already-filtered list needs no label
        configuration. A missing file raises rather than silently marking nothing:
        the list is the whole reason the run was configured this way.
        """
        if not self.suboptimal_episode_list_path:
            return set()
        path = Path(self.suboptimal_episode_list_path)
        if not path.exists():
            raise FileNotFoundError(
                f"suboptimal.episode_list_path does not exist: {path}"
            )
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            rows = json.loads(text)
            if isinstance(rows, dict):
                rows = rows.get("episodes", rows)
            if isinstance(rows, dict):
                rows = [{"episode_path": key, "label": value} for key, value in rows.items()]
        else:
            rows = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

        label_keys = (
            (self.suboptimal_episode_list_label_key,)
            if self.suboptimal_episode_list_label_key
            else ("label", "sample_type", "triage_label", "category")
        )
        selected: set = set()
        for row in rows:
            if isinstance(row, str):
                episode_path, label = row, None
            elif isinstance(row, dict):
                episode_path = row.get("episode_path") or row.get("path")
                label = next((row[key] for key in label_keys if key in row), None)
            else:
                continue
            if not episode_path:
                continue
            if (
                label is not None
                and self.suboptimal_episode_list_labels
                and str(label).lower() not in self.suboptimal_episode_list_labels
            ):
                continue
            selected.add(_normalize_remote_path(str(episode_path).rstrip("/")))
        return selected

    def _mark_suboptimal_episodes(
        self, episodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tag the sub-optimal subset on top of an already-built episode index.

        This is the label-based route, for episodes that have *not* been moved
        into ``<task>/<dir_name>`` on disk. Directory members are already tagged
        by ``_classify_task`` during index construction; they are counted here
        only so the "nothing matched" warning stays truthful.

        An explicit label beats the directory the episode sits in: ``/fail`` on
        disk is a catch-all bucket, and a record already classified fail-class is
        re-labeled here rather than the other way round.
        """
        if not self.suboptimal_enabled:
            return episodes

        listed = self._load_suboptimal_episode_paths()
        marked = sum(1 for ep in episodes if bool(ep.get("is_suboptimal_class")))
        from_directory = marked
        reclassified_fail = 0
        for ep in episodes:
            if bool(ep.get("is_suboptimal_class")):
                continue
            sample_type = str(ep.get("sample_type", "")).lower()
            episode_path = _normalize_remote_path(self._as_path_str(ep.get("episode_path")))
            if not (
                (sample_type and sample_type in self.suboptimal_manifest_sample_types)
                or episode_path in listed
                or (
                    self.suboptimal_task_patterns
                    and self._patterns_match(
                        str(ep.get("task_name", "")), self.suboptimal_task_patterns
                    )
                )
            ):
                continue
            ep["is_suboptimal_class"] = True
            marked += 1
            reclassified_fail += int(bool(ep.get("is_fail_class")))

        if marked == 0:
            logger.warning(
                "TianjiWujiDataset: suboptimal is enabled but no record matched "
                "(dir=%s active=%s, episode_list=%s entries=%d, "
                "manifest_sample_types=%s, task_name=%s)",
                self.suboptimal_dir_name,
                self._suboptimal_dir_active(),
                self.suboptimal_episode_list_path,
                len(listed),
                sorted(self.suboptimal_manifest_sample_types),
                self.suboptimal_task_patterns,
            )
        else:
            logger.info(
                "TianjiWujiDataset: marked %d/%d records sub-optimal "
                "(%d from the %s/ directory, %d re-labeled from fail-class)",
                marked,
                len(episodes),
                from_directory,
                self.suboptimal_dir_name,
                reclassified_fail,
            )
        return episodes

    def _configure_fail_sampling(self) -> None:
        """Partition the index into the disjoint pools the sampler draws from.

        Sub-optimal is carved out of both of the other two, so an episode that is
        fail-class on disk stops being drawn as fail once it is re-labeled.
        """
        self._suboptimal_episodes = [
            ep for ep in self.episodes if bool(ep.get("is_suboptimal_class"))
        ]
        self._main_episodes = [
            ep
            for ep in self.episodes
            if not bool(ep.get("is_fail_class")) and not bool(ep.get("is_suboptimal_class"))
        ]
        self._fail_episodes = [
            ep
            for ep in self.episodes
            if bool(ep.get("is_fail_class")) and not bool(ep.get("is_suboptimal_class"))
        ]
        self._non_suboptimal_episodes = self._main_episodes + self._fail_episodes
        self._fail_sampling_probability = None
        self._suboptimal_sampling_probability = None
        if self.val:
            return

        suboptimal_probability = self.suboptimal_sample_fraction
        if suboptimal_probability is not None:
            if suboptimal_probability > 0.0 and not self._suboptimal_episodes:
                raise ValueError(
                    "suboptimal.sample_fraction requires sub-optimal episodes, but no "
                    "record matched suboptimal.{episode_list_path, "
                    "manifest_sample_types, task_name}"
                )
            if (self.fail_sample_fraction or 0.0) + suboptimal_probability > 1.0:
                raise ValueError(
                    "fail_sample_fraction + suboptimal.sample_fraction must be <= 1, "
                    f"got {self.fail_sample_fraction} + {suboptimal_probability}"
                )
            self._suboptimal_sampling_probability = suboptimal_probability
            logger.info(
                "TianjiWujiDataset: suboptimal sample fraction=%.4f, "
                "suboptimal_records=%d",
                suboptimal_probability,
                len(self._suboptimal_episodes),
            )

        if self.fail_sample_fraction is None:
            return

        fail_probability = self.fail_sample_fraction
        if fail_probability > 0.0 and not self._fail_episodes:
            reclassified = len(self._suboptimal_episodes)
            raise ValueError(
                "fail_sample_fraction requires fail episodes, but the fail pool is empty"
                + (f"; {reclassified} record(s) were re-labeled sub-optimal" if reclassified else "")
            )
        if fail_probability < 1.0 and not self._main_episodes:
            raise ValueError(
                "fail_sample_fraction requires success episodes, but the success pool is empty"
            )
        self._fail_sampling_probability = fail_probability
        logger.info(
            "TianjiWujiDataset: fail sample fraction=%.4f, "
            "success_records=%d, fail_records=%d",
            fail_probability,
            len(self._main_episodes),
            len(self._fail_episodes),
        )

    def _sample_training_episode(self) -> Dict[str, Any]:
        """Draw one episode from the two- or three-way pool split.

        The three pools are disjoint and share a single uniform draw, so the
        sub-optimal fraction is carved out of the whole step budget rather than
        out of the fail fraction: the two categories carry different value
        targets, and one has to stay adjustable without moving the other.
        """
        fail_probability = self._fail_sampling_probability
        suboptimal_probability = self._suboptimal_sampling_probability
        if fail_probability is None and suboptimal_probability is None:
            return random.choice(self.episodes)
        draw = random.random()
        if suboptimal_probability is not None:
            if draw < suboptimal_probability:
                return random.choice(self._suboptimal_episodes)
            if fail_probability is None:
                return random.choice(self._non_suboptimal_episodes)
        pool = (
            self._fail_episodes
            if draw < (suboptimal_probability or 0.0) + fail_probability
            else self._main_episodes
        )
        return random.choice(pool)

    @staticmethod
    def _episode_partition_key(ep: Dict[str, Any]) -> tuple[str, str, str]:
        """Identify one trajectory independently of its expanded latent phases."""
        return (
            str(ep.get("task_name", "")),
            str(ep.get("source_episode_name", ep.get("episode_name", ""))),
            str(ep.get("episode_path", "")).rstrip("/"),
        )

    def _episode_validation_score(self, ep: Dict[str, Any]) -> int:
        """Return a stable score for one Tianji/Wuji trajectory."""
        identity = "\0".join(
            [
                str(self.validation_seed),
                *self._episode_partition_key(ep),
            ]
        )
        return int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )

    def _select_validation_partition(
        self, episodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select one deterministic side of the shared episode split."""
        if self.validation_fraction <= 0.0 or not episodes:
            return episodes

        by_task: Dict[str, Dict[tuple[str, str, str], Dict[str, Any]]] = {}
        for ep in episodes:
            key = self._episode_partition_key(ep)
            by_task.setdefault(key[0], {}).setdefault(key, ep)

        validation_keys: set[tuple[str, str, str]] = set()
        for trajectories in by_task.values():
            count = len(trajectories)
            n_val = int(round(count * self.validation_fraction))
            n_val = min(max(n_val, 1), max(count - 1, 0))
            ranked = sorted(
                trajectories,
                key=lambda key: (
                    self._episode_validation_score(trajectories[key]),
                    key,
                ),
            )
            validation_keys.update(ranked[:n_val])

        return [
            ep
            for ep in episodes
            if (self._episode_partition_key(ep) in validation_keys) == self.val
        ]

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

    def _init_instruction_bank(
        self,
        instruction_bank_path: Optional[str],
        environment_mix_prob: float,
        environment_source: str,
        environment_variant: str,
        text_pool_source: str = "episode",
    ) -> None:
        """Read the opt-in paraphrase bank that replaces per-episode captions.

        The text on disk was written by a captioner that saw the outcome: within
        one task, success episodes carry a full instruction while fail and
        sub-optimal ones carry the directory slug, and the ``behavior`` variants
        narrate the failure itself. Both the wording and the token count differ
        by pool, so a value head can read the caption instead of the video. The
        bank replaces that with N paraphrases per *task*, drawn uniformly, which
        makes the conditioning text identically distributed in all three pools.

        Loaded here, before the episode index is built, so that a missing or
        malformed bank stops the job in the first second rather than at step
        30000 with the conditioning silently degraded back to the leaking
        captions.
        """
        self.environment_mix_prob = float(environment_mix_prob)
        if not 0.0 <= self.environment_mix_prob <= 1.0:
            raise ValueError(
                "environment_mix_prob must be in [0, 1], got "
                f"{self.environment_mix_prob}"
            )
        self.environment_source = str(environment_source or "task_pool").lower()
        if self.environment_source not in {"task_pool", "episode"}:
            raise ValueError(
                "environment_source must be 'task_pool' or 'episode', got "
                f"{self.environment_source!r}"
            )
        self.environment_variant = str(environment_variant or "").strip("/")
        self.text_pool_source = str(text_pool_source or "episode").lower()
        if self.text_pool_source not in {"episode", "task_success"}:
            raise ValueError(
                "text_pool_source must be 'episode' or 'task_success', got "
                f"{self.text_pool_source!r}"
            )
        #: task scope -> sorted success-pool ``language_path`` values to borrow from.
        self._success_text_pool: Dict[str, List[str]] = {}
        self.instruction_bank_path = (
            str(instruction_bank_path) if instruction_bank_path else None
        )
        self.instruction_bank: Optional[Dict[str, Dict[str, Any]]] = None
        self.instruction_bank_task_map: Dict[str, str] = {}
        self.instruction_bank_dim: Optional[int] = None
        self._instruction_bank_key_cache: Dict[str, Optional[str]] = {}
        self._environment_task_pool: Dict[str, List[str]] = {}
        self._environment_fallback_reasons: set = set()
        # Two ways to stop the caption carrying the outcome, and they are
        # alternatives, not layers. The bank does not read per-episode text at
        # all, so there is nothing left for the borrow to redirect; accepting
        # both would leave a config that looks like it borrows and silently does
        # not. Refused rather than warned, because the whole point of either
        # switch is that you can state which one a run used.
        if self.instruction_bank_path is not None and self.text_pool_source != "episode":
            raise ValueError(
                "instruction_bank_path and text_pool_source="
                f"{self.text_pool_source!r} are alternatives: the bank replaces "
                "every per-episode caption, so borrowing one from the success pool "
                "would have no effect. Set text_pool_source=episode to use the bank."
            )
        if self.instruction_bank_path is None:
            if self.environment_mix_prob > 0.0:
                logger.warning(
                    "TianjiWujiDataset: environment_mix_prob=%.3f has no effect while "
                    "instruction_bank_path is unset; per-episode captions are used "
                    "unchanged and still carry the outcome",
                    self.environment_mix_prob,
                )
            return
        if self.environment_mix_prob > 0.0 and not self.environment_variant:
            raise ValueError(
                "environment_mix_prob > 0 requires a non-empty environment_variant"
            )
        # Directory mode globs every .pt that is not named environment*, so a
        # variant outside that prefix would also join the legacy candidate pool
        # and be drawn as if it were an instruction.
        if (
            self.environment_variant
            and not self.environment_variant.startswith("environment")
            and not str(self.language_subpath).endswith(".pt")
        ):
            logger.warning(
                "TianjiWujiDataset: environment_variant=%r is not named environment*, "
                "so it also matches the legacy language candidate glob under %s",
                self.environment_variant,
                self.language_subpath,
            )

        path = self.instruction_bank_path
        if not self._exists(path):
            raise FileNotFoundError(
                f"dataset.instruction_bank_path does not exist: {path}. Set it to null "
                "to fall back to per-episode captions (which leak the outcome)."
            )
        try:
            payload = self._load_pt(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"dataset.instruction_bank_path is unreadable: {path}: {exc}"
            ) from exc
        self.instruction_bank, self.instruction_bank_task_map, self.instruction_bank_dim = (
            self._parse_instruction_bank(payload, path)
        )
        logger.info(
            "TianjiWujiDataset: instruction bank %s -> %d canonical key(s) %s, "
            "dim=%d, environment_mix_prob=%.3f, environment_source=%s, "
            "environment_variant=%s",
            path,
            len(self.instruction_bank),
            {
                key: f"{len(entry['embeddings'])}x[{min(e.shape[0] for e in entry['embeddings'])}"
                f"-{max(e.shape[0] for e in entry['embeddings'])} tok]"
                for key, entry in self.instruction_bank.items()
            },
            self.instruction_bank_dim,
            self.environment_mix_prob,
            self.environment_source,
            self.environment_variant or "-",
        )

    @staticmethod
    def _parse_instruction_bank(
        payload: Any, path: str
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], int]:
        """Validate the bank artifact into ``(entries, task_to_canonical, dim)``.

        Every shape assumption is checked here because the alternative is a
        per-item ``except`` that falls back to the leaking captions: a bank that
        is half-usable must not produce a half-conditioned run.
        """

        def bad(detail: str) -> ValueError:
            return ValueError(f"Malformed instruction bank {path}: {detail}")

        payload = _mapping_to_plain_dict(payload) if payload is not None else None
        if not isinstance(payload, dict):
            raise bad(f"expected a dict at the top level, got {type(payload).__name__}")
        version = payload.get("version")
        try:
            version = int(version)
        except (TypeError, ValueError):
            raise bad(f"version is missing or not an int (got {version!r})") from None
        if version != 1:
            raise bad(f"version={version} is not supported by this loader (expects 1)")

        raw_entries = _mapping_to_plain_dict(payload.get("canonical_keys"))
        if not isinstance(raw_entries, dict) or not raw_entries:
            raise bad("canonical_keys must be a non-empty mapping")

        entries: Dict[str, Dict[str, Any]] = {}
        dims: set = set()
        for key, entry in raw_entries.items():
            entry = _mapping_to_plain_dict(entry)
            if not isinstance(entry, dict):
                raise bad(f"canonical_keys[{key!r}] must be a mapping")
            texts = _to_plain_container(entry.get("texts"))
            embeddings = _to_plain_container(entry.get("embeddings"))
            if not isinstance(texts, list) or not isinstance(embeddings, list):
                raise bad(f"canonical_keys[{key!r}] needs list 'texts' and 'embeddings'")
            if not embeddings:
                raise bad(f"canonical_keys[{key!r}] has no embeddings")
            if len(texts) != len(embeddings):
                raise bad(
                    f"canonical_keys[{key!r}] has {len(texts)} texts but "
                    f"{len(embeddings)} embeddings; the draw indexes both"
                )
            for index, emb in enumerate(embeddings):
                if not isinstance(emb, torch.Tensor) or emb.dim() != 2 or emb.shape[0] < 1:
                    raise bad(
                        f"canonical_keys[{key!r}].embeddings[{index}] must be a 2D "
                        f"[L, D] tensor, got "
                        f"{tuple(emb.shape) if isinstance(emb, torch.Tensor) else type(emb).__name__}"
                    )
                if int(emb.shape[0]) > TEXT_CONTEXT_LEN:
                    raise bad(
                        f"canonical_keys[{key!r}].embeddings[{index}] is "
                        f"{int(emb.shape[0])} tokens, over the {TEXT_CONTEXT_LEN}-token "
                        "cross-attention block; the collate would cut the instruction"
                    )
                dims.add(int(emb.shape[-1]))
            entries[str(key)] = {"texts": [str(t) for t in texts], "embeddings": list(embeddings)}
        if len(dims) != 1:
            raise bad(f"embeddings disagree on the feature dim: {sorted(dims)}")

        raw_map = _mapping_to_plain_dict(payload.get("task_to_canonical"))
        if not isinstance(raw_map, dict) or not raw_map:
            raise bad("task_to_canonical must be a non-empty mapping")
        task_map = {str(task): str(key) for task, key in raw_map.items()}
        dangling = sorted({key for key in task_map.values() if key not in entries})
        if dangling:
            raise bad(
                f"task_to_canonical points at {dangling} which are absent from "
                f"canonical_keys {sorted(entries)}"
            )
        return entries, task_map, dims.pop()

    def _bank_canonical_key(self, task_name: str) -> Optional[str]:
        """Which paraphrase list a task directory draws from, or None.

        ``task_name`` is the episode's parent directory relative to the dataset
        root, so the same trajectory pool shows up as ``<task>``,
        ``<task>/fail`` and ``<task>/<suboptimal dir>``. Those trailing
        components are dropped before the lookup, which is what makes all three
        pools resolve to one list; a lookup that instead depended on the pool
        would hand the pools different text and rebuild the leak. Remaining
        components are peeled off one at a time so that a dataset nested as
        ``<group>/<task>/<episode>`` resolves on either the full relative path or
        the task directory alone.
        """
        if task_name in self._instruction_bank_key_cache:
            return self._instruction_bank_key_cache[task_name]

        parts = [p for p in self._task_scope(task_name).split("/") if p]
        resolved: Optional[str] = None
        for end in range(len(parts), 0, -1):
            prefix = parts[:end]
            for candidate in ("/".join(prefix), prefix[-1]):
                resolved = self.instruction_bank_task_map.get(candidate)
                if resolved is not None:
                    break
            if resolved is not None:
                break
        self._instruction_bank_key_cache[task_name] = resolved
        return resolved

    def _task_scope(self, task_name: str) -> str:
        """The task directory a record belongs to, with the pool suffix removed.

        ``<task>``, ``<task>/fail`` and ``<task>/<suboptimal dir>`` all collapse
        to ``<task>``. Both text draws are scoped by this: it is the unit within
        which the three pools must be indistinguishable, and it is also the unit
        that shares one physical scene setup, since a task directory is one
        capture day.
        """
        parts = [p for p in str(task_name).strip("/").split("/") if p]
        while parts and self._is_pool_component(parts[-1]):
            parts.pop()
        return "/".join(parts)

    def _is_pool_component(self, name: str) -> bool:
        """Whether one path component names an outcome pool rather than a task.

        Mirrors the two predicates that route episodes into pools during
        indexing: ``_path_has_fail``'s substring test and
        ``_path_has_suboptimal_dir``'s exact match.
        """
        name = str(name).lower()
        return "fail" in name or name == self.suboptimal_dir_name

    @staticmethod
    def _pool_label(ep: Dict[str, Any]) -> str:
        """Which on-disk pool a record came from, for coverage accounting.

        Deliberately not ``_episode_sample_class``: that folds in the
        corpus-wide ``treat_as_irrelevant``, which would collapse the three
        candidate pools being compared.
        """
        if bool(ep.get("is_suboptimal_class")):
            return "suboptimal"
        if bool(ep.get("is_fail_class")):
            return "fail"
        return "success"

    def _build_success_text_pool(self) -> None:
        """Collect the success captions that fail and sub-optimal records borrow.

        ``text_pool_source=task_success`` keeps per-episode captions but stops
        them being *this* episode's caption for anything off the policy loss:
        a fail or sub-optimal record reads the text of a uniformly drawn success
        episode from the same task directory instead. The marginal distribution
        of the conditioning text is then the success distribution in all three
        pools, which closes the wording and token-count leaks together, and it
        closes them for every channel at once rather than one file at a time --
        the redirect swaps the directory the candidate glob runs in, so
        ``instruction``, ``robot_behavior`` and ``vidu_json_behavior`` are all
        borrowed from the same success episode and stay mutually consistent.

        This is weaker than the bank in one specific way, and it is worth being
        explicit about it: a success record's caption still describes its own
        video, while a borrowed one does not, so text/video *agreement* remains
        correlated with the outcome even though the text distribution no longer
        is. The bank removes that too by making every pool's caption unrelated to
        its video. Which is why both exist and get their own run.

        Scoped by task directory, like the environment draw, because that is the
        unit within which the pools have to be indistinguishable and the unit
        that shares one physical scene.
        """
        if self.text_pool_source != "task_success" or not self.episodes:
            return

        needed: Dict[str, int] = {}
        for ep in self.episodes:
            scope = self._task_scope(str(ep.get("task_name", "")))
            if self._pool_label(ep) == "success":
                path = self._as_path_str(ep.get("language_path"))
                if path:
                    self._success_text_pool.setdefault(scope, []).append(path)
            else:
                needed[scope] = needed.get(scope, 0) + 1
        # Sorted so a given seed draws the same caption on every rank and every
        # restart, and so ``val`` can freeze on index 0; the filesystem walk
        # order is not stable enough for either.
        for scope in self._success_text_pool:
            self._success_text_pool[scope] = sorted(set(self._success_text_pool[scope]))

        # A task with off-policy records and no success pool cannot borrow, and
        # the fallback would be to read its own caption -- exactly the leak this
        # switch exists to close, reinstated for a subset of the corpus that the
        # log would not name. 0821 is the real case: 10 fail episodes, no success
        # pool at all. Refuse, and make the caller drop the task or list a day
        # that has successes.
        starved = sorted(s for s in needed if not self._success_text_pool.get(s))
        if starved:
            raise ValueError(
                "text_pool_source=task_success requires a success pool in every task "
                "directory that contributes fail or sub-optimal records, but "
                f"{len(starved)} task(s) have none: {starved[:8]}. "
                "Remove those tasks from fail_task_name/suboptimal, or add their "
                "success episodes."
            )
        logger.info(
            "TianjiWujiDataset: text_pool_source=task_success, success captions per "
            "task %s; off-policy records borrowing %s",
            {s: len(v) for s, v in sorted(self._success_text_pool.items())},
            {s: n for s, n in sorted(needed.items())},
        )

    def _draw_success_text_path(self, ep: Dict[str, Any]) -> Optional[str]:
        """Which success episode's caption directory this record reads, or None.

        None means "read your own", which happens only for success records
        themselves. Off-policy records are guaranteed a non-empty pool by
        ``_build_success_text_pool``.
        """
        if self.text_pool_source != "task_success":
            return None
        if self._pool_label(ep) == "success":
            return None
        pool = self._success_text_pool.get(self._task_scope(str(ep.get("task_name", ""))))
        if not pool:
            return None
        return pool[0] if self.val else pool[random.randint(0, len(pool) - 1)]

    def _audit_instruction_bank(self) -> None:
        """Prove the bank covers the index, and measure what still varies by pool.

        Two different failure modes, handled differently. A task with no
        canonical key raises: there is no non-leaking text to hand those
        episodes, so the run must not start. A missing ``environment_variant``
        only costs the environment prefix, and what that costs depends on
        ``environment_source``: under ``episode`` *which* episodes lose it is a
        pool-correlated signal, under ``task_pool`` an empty list takes out one
        whole task directory instead. Both are counted and warned about by name
        rather than silently degraded.
        """
        if self.instruction_bank is None:
            return

        unresolved: Dict[str, int] = {}
        per_pool: Dict[str, int] = {}
        for ep in self.episodes:
            task_name = str(ep.get("task_name", ""))
            pool = self._pool_label(ep)
            per_pool[pool] = per_pool.get(pool, 0) + 1
            if self._bank_canonical_key(task_name) is None:
                unresolved[task_name] = unresolved.get(task_name, 0) + 1
        if unresolved:
            raise ValueError(
                f"instruction bank {self.instruction_bank_path} has no canonical key "
                f"for {len(unresolved)} indexed task(s): {sorted(unresolved)[:8]}. "
                f"It maps {sorted(self.instruction_bank_task_map)[:8]} "
                f"({len(self.instruction_bank_task_map)} total). Every indexed task "
                "needs an entry, otherwise those episodes would keep the leaking "
                "per-episode captions."
            )
        logger.info(
            "TianjiWujiDataset: instruction bank covers all %d records "
            "(pools: %s), paraphrase draw is uniform and pool-independent",
            len(self.episodes),
            {pool: per_pool[pool] for pool in sorted(per_pool)},
        )

        if self.environment_mix_prob <= 0.0:
            logger.info(
                "TianjiWujiDataset: environment mixing off (environment_mix_prob=0), "
                "text is instruction-only"
            )
            return

        pool_by_path: Dict[str, str] = {}
        scope_by_path: Dict[str, str] = {}
        for ep in self.episodes:
            env_path = self._as_path_str(self._environment_path(ep))
            if env_path:
                pool_by_path.setdefault(env_path, self._pool_label(ep))
                scope_by_path.setdefault(env_path, self._task_scope(str(ep.get("task_name", ""))))
        present: Dict[str, int] = {}
        total: Dict[str, int] = {}
        per_task_present: Dict[str, int] = {}
        per_task_total: Dict[str, int] = {}
        self._environment_task_pool = {}
        for env_path, pool in pool_by_path.items():
            scope = scope_by_path[env_path]
            total[pool] = total.get(pool, 0) + 1
            per_task_total[scope] = per_task_total.get(scope, 0) + 1
            if self._exists(env_path):
                present[pool] = present.get(pool, 0) + 1
                per_task_present[scope] = per_task_present.get(scope, 0) + 1
                self._environment_task_pool.setdefault(scope, []).append(env_path)
        # Sorted so that a given seed draws the same caption on every rank and
        # every restart; the filesystem walk order is not stable enough for that.
        for scope in self._environment_task_pool:
            self._environment_task_pool[scope].sort()

        coverage = {
            pool: present.get(pool, 0) / total[pool] for pool in total if total[pool]
        }
        counts = {
            pool: f"{present.get(pool, 0)}/{total[pool]}" for pool in sorted(total)
        }
        logger.info(
            "TianjiWujiDataset: %s coverage per pool %s (source=%s, mixed in with "
            "p=%.3f)",
            self.environment_variant,
            counts,
            self.environment_source,
            self.environment_mix_prob,
        )
        if self.environment_source == "task_pool":
            logger.info(
                "TianjiWujiDataset: %s task pools (episodes contributing a caption "
                "per task directory): %s",
                self.environment_variant,
                {
                    scope: f"{per_task_present.get(scope, 0)}/{per_task_total[scope]}"
                    for scope in sorted(per_task_total)
                },
            )
            # Under task_pool an empty list is the only way the environment can go
            # missing, and it takes out a whole task directory at once rather than
            # scattered episodes.
            starved = sorted(
                scope for scope in per_task_total if not per_task_present.get(scope)
            )
            if starved:
                logger.warning(
                    "TianjiWujiDataset: %d configured task directory(ies) contribute "
                    "no %s and are conditioned on the instruction alone: %s",
                    len(starved),
                    self.environment_variant,
                    starved[:8],
                )
        spread = (max(coverage.values()) - min(coverage.values())) if coverage else 0.0
        if spread > 0.02:
            if self.environment_source == "episode":
                logger.warning(
                    "TianjiWujiDataset: LEAK RISK -- %s coverage differs by pool "
                    "(spread %.3f): %s. With environment_source=episode the "
                    "instruction-only fallback is then itself a pool-correlated "
                    "signal. Write the missing files, switch to "
                    "environment_source=task_pool, or set environment_mix_prob=0.",
                    self.environment_variant,
                    spread,
                    counts,
                )
            else:
                logger.warning(
                    "TianjiWujiDataset: %s coverage differs by pool (spread %.3f): "
                    "%s. Not a leak under environment_source=task_pool, where the "
                    "draw ignores which episode owns the file, but the shared set is "
                    "unevenly sourced",
                    self.environment_variant,
                    spread,
                    counts,
                )
        elif not present:
            logger.warning(
                "TianjiWujiDataset: no %s found for any of the %d episodes, so "
                "environment_mix_prob=%.3f can never fire and the text is "
                "instruction-only",
                self.environment_variant,
                sum(total.values()),
                self.environment_mix_prob,
            )
        self._log_environment_token_lengths()

    def _log_environment_token_lengths(
        self, probe: int = 8, max_task_dirs: int = 6
    ) -> None:
        """Replay the real draw and report the environment token length per pool.

        Length is visible to the model even when the wording is not: the collate
        zero-pads to a fixed block and cross-attention runs without a key mask,
        so the number of non-zero keys is itself readable. Outcome-neutral
        wording is therefore not sufficient, and these numbers are what says
        whether the length is neutral too.

        Measured *within* a task directory, and not across the corpus, because
        that is where the claim lives. Capture days differ in how long their
        captions are, and the pools are not mixed the same way on every day, so a
        corpus-wide per-pool mean differs by several tokens no matter how the
        draw works -- it is reading the day, which the video already shows. What
        would be a leak is a difference between two pools of the *same* day, and
        under ``task_pool`` that is zero by construction because both draw from
        one list. Comparing per day is therefore both the sensitive test and the
        one that does not cry wolf.

        The lengths come from ``_draw_environment_path``, so this measures the
        channel as the model receives it rather than what sits on disk. A file
        that exists but is rejected matters here for the same reason a missing
        one does, so both are named.
        """
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for ep in self.episodes:
            key = (self._task_scope(str(ep.get("task_name", ""))), self._pool_label(ep))
            buckets.setdefault(key, []).append(ep)
        # Largest task directories first, capped so that a corpus with many days
        # does not turn construction into a read of the whole caption set.
        scope_size: Dict[str, int] = {}
        for (scope, _), eps in buckets.items():
            scope_size[scope] = scope_size.get(scope, 0) + len(eps)
        scopes = sorted(scope_size, key=lambda s: (-scope_size[s], s))[:max_task_dirs]

        # Isolated RNG stream: sampling must not disturb the global RNG that the
        # per-item draws run on, and its own numbers should not move run to run.
        rng = random.Random(0)
        lengths: Dict[Any, List[int]] = {}
        rejected_total = 0
        for scope in scopes:
            for pool in sorted({p for s, p in buckets if s == scope}):
                eps = buckets[(scope, pool)]
                for _ in range(probe):
                    env_path = self._draw_environment_path(
                        eps[rng.randrange(len(eps))], rng=rng
                    )
                    if not env_path:
                        continue
                    emb = self._load_environment_embedding(env_path, quiet=True)
                    if emb is None:
                        rejected_total += 1
                    else:
                        lengths.setdefault((scope, pool), []).append(int(emb.shape[0]))

        summary: Dict[str, str] = {}
        for pool in sorted({p for _, p in lengths}):
            flat = [n for (_, p), v in lengths.items() if p == pool for n in v]
            summary[pool] = (
                f"n={len(flat)} tok={min(flat)}-{max(flat)} "
                f"(mean {sum(flat) / len(flat):.1f})"
            )
        instruction_lengths = [
            int(emb.shape[0])
            for entry in self.instruction_bank.values()
            for emb in entry["embeddings"]
        ]
        logger.info(
            "TianjiWujiDataset: text token budget %d; instruction %d-%d tok, "
            "drawn %s per pool over %d task dir(s): %s",
            TEXT_CONTEXT_LEN,
            min(instruction_lengths),
            max(instruction_lengths),
            self.environment_variant,
            len(scopes),
            summary or "none readable",
        )
        if rejected_total:
            logger.warning(
                "TianjiWujiDataset: %d sampled %s file(s) are unusable and fall back "
                "to instruction-only. %s Re-export them or set "
                "environment_mix_prob=0.",
                rejected_total,
                self.environment_variant,
                (
                    "LEAK RISK -- with environment_source=episode which episodes "
                    "lose the prefix follows the pool."
                    if self.environment_source == "episode"
                    else "Not pool-correlated under environment_source=task_pool, "
                    "but it lowers the effective mix rate."
                ),
            )
        self._warn_on_within_task_length_spread(lengths, scopes)

    def _warn_on_within_task_length_spread(
        self, lengths: Dict[Any, List[int]], scopes: List[str]
    ) -> None:
        """Warn when two pools of one task directory get different-length text.

        The threshold is the sampling noise of the two cell means rather than a
        fixed number of tokens: captions inside one day already vary from 21 to
        135, so a few samples per cell scatter far enough that any constant
        would either fire on every run or never fire at all.
        """
        worst: Optional[tuple] = None
        for scope in scopes:
            cells = {
                pool: vals
                for (s, pool), vals in lengths.items()
                if s == scope and len(vals) >= 3
            }
            if len(cells) < 2:
                continue
            stats = {}
            for pool, vals in cells.items():
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                stats[pool] = (mean, (variance / len(vals)) ** 0.5, len(vals))
            hi = max(stats, key=lambda p: stats[p][0])
            lo = min(stats, key=lambda p: stats[p][0])
            spread = stats[hi][0] - stats[lo][0]
            noise = 4.0 * ((stats[hi][1] ** 2 + stats[lo][1] ** 2) ** 0.5)
            if spread > max(noise, 1.0) and (worst is None or spread > worst[1]):
                worst = (scope, spread, noise, hi, lo, stats)
        if worst is None:
            return
        scope, spread, noise, hi, lo, stats = worst
        logger.warning(
            "TianjiWujiDataset: LEAK RISK -- inside task dir %s the drawn %s is %.1f "
            "tokens longer for %s than for %s (%s), beyond the %.1f expected from "
            "sampling. Token count is readable through the zero-padded, unmasked "
            "cross-attention block even when the wording is neutral. "
            "environment_source=task_pool removes this by construction.",
            scope,
            self.environment_variant,
            spread,
            hi,
            lo,
            {p: f"n={n} mean={m:.1f}+-{se:.1f}" for p, (m, se, n) in sorted(stats.items())},
            noise,
        )

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
        if not self.include_fail and self._path_has_fail(path):
            return True
        if not self._suboptimal_dir_active() and self._path_has_suboptimal_dir(path):
            return True
        parts = [p.lower() for p in self._as_path_str(path).split("/")]
        if not self.include_old and any(p == "old" or p.startswith("old_") for p in parts):
            return True
        return False

    @staticmethod
    def _name_matches(name: str, pattern: str) -> bool:
        name = str(name).strip("/")
        pattern = str(pattern).strip("/").replace("%", "*")
        return fnmatch.fnmatchcase(name, pattern)

    # Per-sample embodiment tag carried into the batch.
    # 1 = human (human_hand glove), 0 = robot/dex (e.g. lingqiaoshou).
    SOURCE_ROBOT = 0
    SOURCE_HUMAN = 1

    @classmethod
    def _resolve_source(cls, task_name: Optional[str]) -> int:
        return cls.SOURCE_HUMAN if "human_hand" in str(task_name or "") else cls.SOURCE_ROBOT

    def _task_allowed(self, task_name: str) -> bool:
        """Success/main index membership. Does not include fail-class dirs."""
        patterns = self._main_task_patterns()
        if not patterns:
            return True
        return self._patterns_match(task_name, patterns)

    def _path_has_fail(self, path) -> bool:
        """True if any path component contains ``fail`` (used only by ``_skip_path``)."""
        parts = [p.lower() for p in self._as_path_str(path).split("/")]
        return any("fail" in p for p in parts)

    def _path_has_suboptimal_dir(self, path) -> bool:
        """True if a path component *is* the sub-optimal directory.

        Exact component match, unlike ``_path_has_fail``'s substring test: this
        name is a common English word and a task called
        ``..._suboptimal_grasp_study`` must not be swept out of the main pool.
        """
        parts = [p.lower() for p in self._as_path_str(path).split("/")]
        return any(p == self.suboptimal_dir_name for p in parts)

    def _suboptimal_dir_active(self) -> bool:
        """Whether ``<task>/<dir_name>`` participates in indexing at all.

        Deliberately not a separate ``include_suboptimal`` gate parallel to
        ``include_fail``: two switches can disagree, and the dangerous
        disagreement (episodes indexed but not labeled) would silently drop
        sub-optimal data into the success pool and train the policy on it. Tying
        visibility to the same ``suboptimal.enabled`` that drives labeling makes
        "indexed" and "labeled sub-optimal" the same condition by construction.
        """
        return bool(self.suboptimal_enabled and self.suboptimal_from_directory)

    @staticmethod
    def _as_pattern_list(value) -> List[str]:
        if not value:
            return []
        items = value if isinstance(value, (list, tuple)) else [value]
        return [str(p) for p in items if str(p).strip()]

    def _main_task_patterns(self) -> List[str]:
        return self._as_pattern_list(self.task_name)

    def _fail_task_patterns(self) -> List[str]:
        """Fail-class patterns. Empty when ``include_fail`` is false (ignored).

        If ``fail_task_name`` is set, use that list. Otherwise fall back to
        ``<task>/fail`` under each main ``task_name``, or ``*/fail`` when
        ``task_name`` is also empty.
        """
        if not self.include_fail:
            return []
        explicit = self._as_pattern_list(self.fail_task_name)
        if explicit:
            return explicit
        main = self._main_task_patterns()
        if not main:
            return ["*/fail"]
        extra = []
        for pat in main:
            stripped = pat.strip("/")
            last = stripped.rsplit("/", 1)[-1].lower()
            if last == "fail":
                continue
            extra.append(f"{stripped}/fail")
        return extra

    def _suboptimal_task_patterns(self) -> List[str]:
        """Sub-optimal-class patterns. Empty unless the directory route is active.

        Unlike ``fail_task_name``, an explicit ``suboptimal.task_name`` *adds* to
        the ``<task>/<dir_name>`` defaults instead of replacing them. Replacing
        would let a config that names some other pattern leave a real
        ``suboptimal/`` directory unclassified but still walked, which under a
        wildcard ``task_name`` lands those episodes in the success pool.
        """
        if not self._suboptimal_dir_active():
            return []
        main = self._main_task_patterns()
        if not main:
            derived = [f"*/{self.suboptimal_dir_name}"]
        else:
            derived = []
            for pat in main:
                stripped = pat.strip("/")
                if stripped.rsplit("/", 1)[-1].lower() == self.suboptimal_dir_name:
                    continue
                derived.append(f"{stripped}/{self.suboptimal_dir_name}")
        extra = [p for p in self.suboptimal_task_patterns if p not in derived]
        return derived + extra

    def _suboptimal_task_allowed(self, task_name: str) -> bool:
        patterns = self._suboptimal_task_patterns()
        return bool(patterns) and self._patterns_match(task_name, patterns)

    def _index_task_patterns(self) -> List[str]:
        return (
            self._main_task_patterns()
            + self._fail_task_patterns()
            + self._suboptimal_task_patterns()
        )

    def _patterns_match(self, task_name: str, patterns: List[str]) -> bool:
        return any(self._name_matches(task_name, pat) for pat in patterns)

    def _fail_task_allowed(self, task_name: str) -> bool:
        patterns = self._fail_task_patterns()
        return bool(patterns) and self._patterns_match(task_name, patterns)

    def _classify_task(self, task_name: str) -> Optional[str]:
        """Return ``'suboptimal'``, ``'fail'``, ``'main'``, or None.

        Sub-optimal is checked first for the same reason ``_episode_sample_class``
        prefers it: it is the only label a human assigned per episode, so it
        overrides both the ``/fail`` bucket and the success pool. Returns None
        when nothing matches, which is how the caller drops the episode.
        """
        if self._suboptimal_task_allowed(task_name):
            return "suboptimal"
        if self._fail_task_allowed(task_name):
            return "fail"
        if self._task_allowed(task_name):
            return "main"
        return None

    def _episode_is_task_irrelevant(self, ep: Dict[str, Any]) -> bool:
        """Fail-class samples take the TASK_IRRELEVANT routing class (never policy).

        ``training_step`` treats ``sample_class=TASK_IRRELEVANT`` as:
          * value ON  -> WM XOR VM, never policy (``wm_ratio`` is unused here)
          * value OFF -> forced WM (``use_wm |= off_policy_active``), no action loss
        ``wm_ratio`` remains the WM fraction of regular/success data only
        (``main_mask``). The first ``max(warmup_steps, uniform_loss_weight_warmup_steps)``
        steps mark these samples ``off_policy_pending`` (fully masked). That is
        acceptable: fail-class contributes nothing during warmup, then WM-only
        / WM XOR VM. The dedicated ``task_irrelevant`` sub-dataset still sets
        ``treat_as_irrelevant=True`` for every sample and is unchanged.

        Sub-optimal samples share all of the above and differ only in the sign of
        their value target, so ``_episode_sample_class`` splits them out of this
        answer instead of widening it.
        """
        if self.treat_as_irrelevant:
            return True
        return bool(ep.get("is_fail_class"))

    def _episode_sample_class(self, ep: Dict[str, Any]) -> int:
        """Which of the three routing classes this episode belongs to.

        The sub-optimal label wins over every other signal, including the
        ``/fail`` directory and the corpus-wide ``treat_as_irrelevant``: it is the
        only one of the three derived from looking at what the episode actually
        did, and overriding a coarse bucket is the reason it exists. An episode
        marked sub-optimal that lives in the success pool is therefore also pulled
        out of policy training.
        """
        if bool(ep.get("is_suboptimal_class")):
            return SAMPLE_CLASS_SUBOPTIMAL
        if self._episode_is_task_irrelevant(ep):
            return SAMPLE_CLASS_TASK_IRRELEVANT
        return SAMPLE_CLASS_MAIN

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
        meta = {
            "schema": "tianji_wuji_episode_index_v1",
            "episode_index_source": self.episode_index_source,
            "episode_path_prefix": self.episode_path_prefix,
            "dataset_dir": str(self.dataset_dir),
            "clickhouse": ch_meta,
            "task_mode": self.task_mode,
            "task_name": self.task_name,
            "fail_task_name": self.fail_task_name,
            "include_fail": self.include_fail,
            "include_old": self.include_old,
            "manifest_path": self.manifest_path,
            "manifest_sample_type": self.manifest_sample_type,
            "latent_root": self.latent_root,
            "latent_phases": sorted(self.latent_phases_set) if self.latent_phases_set is not None else None,
            "left_latent_name": self.left_latent_name,
            "right_latent_name": self.right_latent_name,
            "view_latent_names": self.view_latent_names,
            "single_latent_name": self.single_latent_name,
            "language_subpath": self.language_subpath,
            "tjwj_subpath": self.tjwj_subpath,
            "action_from_state": self.action_from_state,
        }
        # Include directory-routing parameters in the index-cache identity.
        # A configuration change triggers a rescan of `<task>/<dir_name>`.
        if self._suboptimal_dir_active():
            meta["suboptimal_index"] = {
                "dir_name": self.suboptimal_dir_name,
                "task_name": self.suboptimal_task_patterns,
            }
        return meta

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
                for key in ("sample_type", "source_type", "is_fail_class", "is_suboptimal_class"):
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
            patterns = self._index_task_patterns()
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
            if self._skip_path(ep_dir):
                continue
            cls = self._classify_task(str(task_name))
            if cls is None:
                continue
            rec = self._episode_record(ep_dir, str(task_name), row.get("episode_id"))
            rec["is_fail_class"] = cls == "fail"
            rec["is_suboptimal_class"] = cls == "suboptimal"
            episodes.append(rec)
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
            cls = self._classify_task(task_name)
            if cls is None:
                continue
            record = self._episode_record(ep_dir, task_name, row.get("episode_name"))
            record["is_fail_class"] = cls == "fail"
            record["is_suboptimal_class"] = cls == "suboptimal"
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
            cls = self._classify_task(task_name)
            if cls is None:
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
            rec = self._episode_record(ep_dir, task_name, ep_dir.name)
            rec["is_fail_class"] = cls == "fail"
            rec["is_suboptimal_class"] = cls == "suboptimal"
            episodes.append(rec)
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

        # Unrestricted walk when task_name is empty (scan-all). Fail-class
        # membership is decided later by `_classify_task`.
        raw_patterns = self._index_task_patterns() if self._main_task_patterns() else []
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
            cls = self._classify_task(task_name)
            if cls is None:
                return
            rec = self._episode_record(ep_dir, task_name, self._basename(ep_dir))
            rec["is_fail_class"] = cls == "fail"
            rec["is_suboptimal_class"] = cls == "suboptimal"
            episodes.append(rec)

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
                if self._classify_task(task_name) is not None:
                    for ep in ep_children:
                        _record(ep, task_name)
                # Keep walking sibling dirs such as <task>/fail. Returning here
                # would make include_fail a no-op whenever the parent already
                # has episode* children.
                if depth >= max_depth:
                    return
                ep_set = {self._as_path_str(c) for c in ep_children}
                for child in children:
                    if self._as_path_str(child) in ep_set:
                        continue
                    if _prefix_could_match(_rel(child)):
                        _walk(child, depth + 1)
                return
            if depth >= max_depth:
                return
            for child in children:
                if _prefix_could_match(_rel(child)):
                    _walk(child, depth + 1)

        _walk(root, 1)
        return episodes

    def __len__(self) -> int:
        return max(1, len(self.episodes) * self.samples_per_episode)

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

    def _action_stride(self, payload: Any, ep: Dict[str, Any]) -> int:
        """Stride that brings THIS episode's action timeline to `train_action_hz`.

        Normalizing per episode rather than per run is what lets 30 Hz and 60 Hz
        captures train together: afterwards every episode is at the same rate, so
        `_latent_to_action_range` keeps indexing in plain train-rate action steps.
        """
        return _episode_rate_stride(
            _stream_hz(payload),
            self.train_action_hz,
            fallback=self.action_downsample_stride,
            stream="action",
            episode=ep.get("episode_path") if isinstance(ep, dict) else ep,
        )

    @staticmethod
    def _downsample_sequence(sequence: torch.Tensor, stride: int) -> torch.Tensor:
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
        raw = self._load_pt(tjwj_path)
        stride = self._action_stride(raw, ep)

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
                return self._downsample_sequence(
                    torch.cat([left_q, left_w9, right_q, right_w9], dim=-1), stride
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
            return self._downsample_sequence(
                torch.cat([left_q, left_w9, right_q, right_w9], dim=-1), stride
            )
        if raw.shape[1] == 58:
            left_q = raw[:, :20]
            left_w9 = raw[:, 20:29]
            right_q = raw[:, 29:49]
            right_w9 = raw[:, 49:58]
            left_q, right_q = self._normalize_qpos40(left_q, right_q)
            return self._downsample_sequence(
                torch.cat([left_q, left_w9, right_q, right_w9], dim=-1), stride
            )
        raise ValueError(f"Unsupported tjwj tensor dim: {raw.shape[1]}")

    def _load_action58_abs(self, ep: Dict[str, Any]) -> torch.Tensor:
        hand_path = ep.get("hand_path")
        endpose_path = ep.get("endpose_path")
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
            return self._downsample_sequence(
                torch.cat([left_qpos, left_wrist, right_qpos, right_wrist], dim=-1),
                self._action_stride(hand, ep),
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
        return self._downsample_sequence(
            torch.cat([left_qpos, left_wrist, right_qpos, right_wrist], dim=-1),
            self._action_stride(hand, ep),
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
    ) -> torch.Tensor:
        chunks = []
        n = sequence.shape[0]
        for i in range(num_latents):
            abs_lat = latent_start + i
            start, end = self._latent_to_action_range(abs_lat, frame_stride, phase_start_30)
            start = min(start, n - 1)
            end = min(end, n)
            chunk = sequence[start:end]
            if chunk.shape[0] < per_latent_len:
                chunk = torch.cat([chunk, chunk[-1:].expand(per_latent_len - chunk.shape[0], -1)], dim=0)
            else:
                chunk = chunk[:per_latent_len]
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
        if self.instruction_bank is not None:
            return self._bank_language_embedding(ep)
        borrowed = self._draw_success_text_path(ep)
        candidates = self._resolve_language_candidates(ep, borrowed)
        if not candidates:
            source = borrowed or ep.get("language_path")
            raise FileNotFoundError(f"No language .pt candidates under: {source}")

        last_exc: Optional[Exception] = None
        candidate_order = sorted(candidates) if self.val else random.sample(candidates, k=len(candidates))
        for cand in candidate_order:
            try:
                emb = self._load_pt(cand)
                if isinstance(emb, list):
                    emb = emb[0] if self.val else emb[random.randint(0, len(emb) - 1)]
                if isinstance(emb, torch.Tensor) and emb.dim() == 3:
                    emb = emb[0] if self.val else emb[random.randint(0, emb.shape[0] - 1)]
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

    def _resolve_language_candidates(
        self, ep: Dict[str, Any], borrowed_language_path: Optional[str] = None
    ) -> List[str]:
        # Keyed on the language path rather than the episode path, because under
        # ``text_pool_source=task_success`` one record resolves a different
        # directory on every draw while many records resolve the same one. The
        # cache is a glob cache, so the path is the correct key either way.
        language_path = self._as_path_str(
            borrowed_language_path
            if borrowed_language_path is not None
            else ep.get("language_path")
        )
        cached = self._language_candidates_cache.get(language_path)
        if cached is not None:
            return cached

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

        self._language_candidates_cache[language_path] = candidates
        return candidates

    def _bank_language_embedding(self, ep: Dict[str, Any]) -> torch.Tensor:
        """One uniformly drawn paraphrase, optionally prefixed with scene context.

        The draw reads the task directory and nothing else -- not the pool, not
        ``is_fail_class``, not ``_episode_sample_class`` -- so its distribution
        is identical for success, fail and sub-optimal records. That identity is
        the entire point of the bank.

        Drawing from the module-level ``random`` is what the rest of this file
        does (variant choice, window start, pool choice), which is also what
        makes it worker-safe: ``DataLoader`` re-seeds ``random`` per worker from
        the base seed, so workers differ from each other and a run stays
        reproducible. ``val`` freezes the draw on paraphrase 0 and skips the
        environment mix, mirroring how ``_load_language_embedding`` freezes its
        own choices, so validation loss does not move with the caption.
        """
        task_name = str(ep.get("task_name", ""))
        key = self._bank_canonical_key(task_name)
        if key is None:
            raise KeyError(
                f"instruction bank has no canonical key for task {task_name!r} "
                f"({self.instruction_bank_path})"
            )
        embeddings = self.instruction_bank[key]["embeddings"]
        index = 0 if self.val else random.randint(0, len(embeddings) - 1)
        instruction = embeddings[index].to(torch.float32, copy=True)
        if self.val or self.environment_mix_prob <= 0.0:
            return instruction
        if random.random() >= self.environment_mix_prob:
            return instruction

        environment = self._load_environment_embedding(self._draw_environment_path(ep))
        if environment is None:
            return instruction
        budget = TEXT_CONTEXT_LEN - int(instruction.shape[0])
        if budget <= 0:
            return instruction
        # Environment first, instruction last, so the pair reads as context then
        # command -- and so the truncation the collate would apply at the tail is
        # taken here from the environment side instead, leaving the instruction
        # whole.
        if int(environment.shape[0]) > budget:
            environment = environment[:budget]
        return torch.cat([environment, instruction], dim=0)

    def _draw_environment_path(self, ep: Dict[str, Any], rng=None):
        """Which scene description to prefix this item with, or None.

        Under ``task_pool`` the file is drawn uniformly from every episode in the
        same task directory that has one, across all three pools. The point is
        that the length is then pool-independent *by construction* rather than by
        measurement: the real ``environment`` captions run ~35 words for success
        against ~49 for fail, and since the collate zero-pads to a fixed block
        with no key mask, prefixing each episode with its own caption would hand
        the model the outcome through token count alone -- the leak this bank
        exists to close, reopened one field over. Scoping the draw to the task
        directory keeps the caption describing a plausible scene, because a task
        directory is one capture day and the pools within it share the physical
        setup.

        Only paths are kept, never tensors: 848 captions at ~63 x 4096 floats
        would be hundreds of MB in each of the 16 workers per node, so the file
        is read on the item that uses it.

        ``episode`` restores reading the record's own file. It is kept for
        comparison and is known to leak; ``task_pool`` is the default.
        """
        if self.environment_source == "episode":
            return self._environment_path(ep)
        candidates = self._environment_task_pool.get(
            self._task_scope(str(ep.get("task_name", "")))
        )
        if not candidates:
            return None
        draw = rng.randint if rng is not None else random.randint
        return candidates[draw(0, len(candidates) - 1)]

    def _environment_path(self, ep: Dict[str, Any]):
        """Path to this episode's scene-description embedding, or None.

        Derived from the record's ``language_path`` without changing the
        serialized episode record.
        """
        if not self.environment_variant:
            return None
        language_path = self._as_path_str(ep.get("language_path"))
        if not language_path:
            return None
        base = (
            self._parent(language_path)
            if language_path.endswith(".pt")
            else language_path
        )
        return self._join_path(base, self.environment_variant)

    def _load_environment_embedding(self, path, quiet: bool = False) -> Optional[torch.Tensor]:
        """The drawn scene text as ``[L, D]``, or None to use instruction only.

        Every rejection here falls back to instruction-only. Under ``episode``
        that is a pool-correlated signal exactly to the extent that the
        rejections are; under ``task_pool`` the drawn path does not depend on the
        record, so it only lowers the mix rate. Either way
        ``_audit_instruction_bank`` counts coverage up front and the first
        rejection of each kind is logged. The stale ``environment.pt`` this
        replaces is the reason a pooled ``[D]`` vector is rejected rather than
        read as one token: that file is ``[D]`` for the fail and sub-optimal
        pools and ``[L, D]`` for success, so silently widening it would hand the
        pools text of systematically different length -- the same leak in a new
        place. ``_load_language_embedding`` refuses those vectors for the same
        reason.
        """
        if path is None:
            return None
        try:
            raw = self._load_pt(path)
        except Exception as exc:  # noqa: BLE001
            return self._environment_fallback(
                f"unreadable ({exc.__class__.__name__})", path, quiet
            )
        emb = raw[0] if isinstance(raw, list) and raw else raw
        if isinstance(emb, torch.Tensor) and emb.dim() == 3 and emb.shape[0] >= 1:
            emb = emb[0]
        if not isinstance(emb, torch.Tensor) or emb.dim() != 2 or emb.shape[0] < 1:
            shape = tuple(emb.shape) if isinstance(emb, torch.Tensor) else type(emb).__name__
            return self._environment_fallback(f"not a [L, D] tensor ({shape})", path, quiet)
        if int(emb.shape[-1]) != int(self.instruction_bank_dim):
            return self._environment_fallback(
                f"dim {int(emb.shape[-1])} != instruction dim {self.instruction_bank_dim}",
                path,
                quiet,
            )
        return emb.to(torch.float32)

    def _environment_fallback(self, reason: str, path, quiet: bool) -> None:
        if not quiet and reason not in self._environment_fallback_reasons:
            self._environment_fallback_reasons.add(reason)
            logger.warning(
                "TianjiWujiDataset: %s %s -- conditioning on the instruction alone "
                "for these episodes (first occurrence of this reason only)",
                reason,
                path,
            )
        return None

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
        decaying_slope: bool = False,
    ) -> torch.Tensor:
        """Cumulative harmonic return in [0, 1], optionally negated.

        Two reward orderings, both normalised by the same ``harmonic_T`` so they
        reach exactly 1.0 at the last frame and stay comparable:

        * default -- reward ``1/(T-k)`` grows with k, so the curve is convex and
          its slope increases: nearly flat early, accelerating to the endpoint.
          Negated for the fail class this reads "looked plausible, collapsed late".
        * ``decaying_slope`` -- reward ``1/(k+1)`` shrinks with k, giving
          ``H_{k+1}/H_T``: concave, steep at frame 0 then flattening. Negated for
          task-irrelevant data this reads "never was on task, and we knew at once".

        Neither curve is pinned to 0 on frame 0. Both open at one frame's worth
        of their own reward -- ``1/(T*H_T)`` for the growing order, ``1/H_T`` for
        the decaying one -- which is what makes the decaying curve open steep
        instead of easing in. That opening step is the point: an off-task episode
        is recognisable from its first frame, and the target says so.
        """
        if total_latents <= 1:
            return torch.zeros(window_size, dtype=torch.float32)

        harmonic_T = (1.0 / torch.arange(1, total_latents + 1, dtype=torch.float64)).sum().item()
        if decaying_slope:
            per_frame_reward = 1.0 / (1.0 + torch.arange(total_latents, dtype=torch.float64))
        else:
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
        """Non-vlac irrelevant targets: the saturated floor of each target's own range.

        Unaffected by ``task_irrelevant_decaying_curve``, which only shapes the
        vlac return. ``progress`` is a fraction in [0, 1] with no negative half,
        so its floor is 0 and a decaying curve would leave the range entirely;
        ``ttc`` already encodes "no completion in sight" as the constant -1.
        Neither is a cumulative return, so there is no slope to shape.
        """
        if self.value_target in {"progress", "vlac"}:
            return torch.zeros(window_size, dtype=torch.float32)
        return torch.full((window_size,), -1.0, dtype=torch.float32)

    def _irrelevant_curve_is_decaying(self) -> bool:
        """Whether the vlac irrelevant target uses the decaying-slope curve.

        Requires ``self.treat_as_irrelevant``: only the dedicated task-irrelevant
        sub-dataset gets the new shape. Fail-class records also route through the
        ``treat_as_irrelevant`` branch of ``_build_value_targets``, but they are a
        genuine attempt at the task and keep the increasing-slope curve.
        """
        return bool(self.task_irrelevant_decaying_curve and self.treat_as_irrelevant)

    def _build_value_targets(
        self,
        start: int,
        window_size: int,
        total_latents: int,
        treat_as_irrelevant: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        if self.value_target is None:
            return None
        if treat_as_irrelevant is None:
            treat_as_irrelevant = self.treat_as_irrelevant
        if treat_as_irrelevant:
            if self.value_target == "vlac":
                return self._build_vlac_values(
                    start,
                    window_size,
                    total_latents,
                    negate=True,
                    decaying_slope=self._irrelevant_curve_is_decaying(),
                )
            return self._build_irrelevant_values(window_size)
        if self.value_target == "progress":
            return self._build_progress_values(start, window_size, total_latents)
        if self.value_target == "ttc":
            return self._build_ttc_values(start, window_size, total_latents)
        if self.value_target == "vlac":
            return self._build_vlac_values(start, window_size, total_latents, negate=False)
        return None

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        W = self.max_latent_window
        for attempt in range(8):
            if self.val:
                episode_idx = (idx // self.samples_per_episode + attempt) % len(self.episodes)
                ep = self.episodes[episode_idx]
            else:
                ep = self._sample_training_episode()
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
                    continue

                if total_latents > W:
                    if self.val:
                        slot = idx % self.samples_per_episode
                        start = round(
                            slot
                            * (total_latents - W)
                            / max(self.samples_per_episode - 1, 1)
                        )
                    else:
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
                phase_start_30 = int(ep.get("phase_start_30", 0))
                actions = self._gather_per_latent(
                    action58_abs,
                    start,
                    actual_latents,
                    frame_stride,
                    phase_start_30=phase_start_30,
                    per_latent_len=self.actions_per_latent,
                )
                states = self._gather_states(
                    state58_abs, start, actual_latents, frame_stride, phase_start_30=phase_start_30
                )

                target_action_len = W * self.actions_per_latent
                if actions.shape[0] < target_action_len:
                    actions = torch.cat([actions, actions[-1:].expand(target_action_len - actions.shape[0], -1)], dim=0)
                if states.shape[0] < W:
                    states = torch.cat([states, states[-1:].expand(W - states.shape[0], -1)], dim=0)

                extra_action_sequences: Dict[str, torch.Tensor] = {}

                sample_class = self._episode_sample_class(ep)
                is_irrelevant = sample_class != SAMPLE_CLASS_MAIN
                values = self._build_value_targets(
                    start,
                    W,
                    total_latents,
                    treat_as_irrelevant=sample_class == SAMPLE_CLASS_TASK_IRRELEVANT,
                )
                latents, actions, states, values, valid_len, extra_action_sequences = maybe_bootstrap_start_window_sample(
                    latents,
                    actions,
                    valid_len,
                    start,
                    actions_per_latent=self.actions_per_latent,
                    states=states,
                    values=values,
                    chunk_size=self.bootstrap_chunk_size,
                    enabled=self.bootstrap_start_window and not self.val,
                    prob=self.bootstrap_start_window_prob,
                    extra_action_sequences=extra_action_sequences,
                )
                latents = self._apply_view_dropout(latents)

                sample: Dict[str, Any] = {
                    "latents": latents,
                    "actions": actions,
                    "states": states,
                    "language_embedding": self._load_language_embedding(ep),
                    "valid_len": int(valid_len),
                    "latent_start": int(start),
                    SAMPLE_CLASS_KEY: int(sample_class),
                    "is_success": int(not is_irrelevant),
                    "source": self._resolve_source(ep.get("task_name")),
                }
                if extra_action_sequences:
                    sample.update(extra_action_sequences)
                if values is not None:
                    sample["values"] = values
                return sample
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tianji/Wuji sample error (%s): %s", ep.get("episode_path"), exc)
        return None
