"""Tianji/Sharpa dexterous-hand latent dataset.

Parallel sibling of :mod:`data.tianji_wuji.tianji_wuji_dataset` for the Sharpa
embodiment. The only embodiment differences are dimensional and are all derived
from the named constants below:

===================  ==========================  ==========================
quantity             Wuji (lingqiaoshou)         Sharpa
===================  ==========================  ==========================
hand qpos per side   20                          ``HAND_DIM`` = 22
both-hand qpos       40                          ``QPOS_DIM`` = 44
action/state vector  58                          ``ACTION_DIM`` = 62
raw stream rate      60 Hz action                60 Hz action (per-episode ``hz``)
===================  ==========================  ==========================

Source-tensor notes (see ``episode_manifest.json`` / ``postprocess_report.json``
in each episode). The manifest declares the *packed* on-the-wire stream widths,
while the exported ``.pt`` files hold the already-extracted training fields:

* ``sharpa_hand_*_feedback`` is a 88-wide pack; ``hand_qpos.pt`` holds channels
  ``[44:66]`` = ``joint_position`` -> 22 per side.
* ``luna_arm_*_action`` is a 14-wide pack; ``arm_action.pt`` holds ``[7:14]`` =
  ``joint_position`` -> 7, and ``arm_action_endpose.pt`` holds ``[0:7]`` =
  ``eef_pose`` (xyz metres + quat xyzw).
* ``luna_arm_*_feedback`` is a 28-wide pack; ``arm_endpose.pt`` holds ``[0:7]``
  = ``eef_pose``. ``arm_qpos.pt`` is the *unextracted* 28-wide pack
  (``status="needs_adapter"``), laid out as
  ``[eef_pose(7) | eef_twist(7) | joint_position(7) | joint_velocity(7)]``.
  This loader never consumes ``arm_qpos.pt``: the wrist block comes from
  ``arm_endpose.pt`` / ``arm_action_endpose.pt``, exactly like Wuji.
"""

from __future__ import annotations

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
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from utils.common import maybe_bootstrap_start_window_sample


logger = logging.getLogger(__name__)

SIDES = ("left", "right")

# --- Sharpa embodiment dimensions (single source of truth) -------------------
# Per-hand dexterous DoF. Wuji is 20, Sharpa is 22.
HAND_DIM = 22
# Wrist block appended after each hand: xyz(3, metres) + rot6d(6).
WRIST_DIM = 9
# Both hands stacked for qpos normalization: [L_qpos | R_qpos].
QPOS_DIM = 2 * HAND_DIM
# Full action/state vector: [L_qpos | L_wrist9 | R_qpos | R_wrist9].
ACTION_DIM = 2 * (HAND_DIM + WRIST_DIM)
# Offsets into the ACTION_DIM vector, derived so nothing is a magic number.
LEFT_QPOS_SLICE = slice(0, HAND_DIM)
LEFT_WRIST_SLICE = slice(HAND_DIM, HAND_DIM + WRIST_DIM)
RIGHT_QPOS_SLICE = slice(HAND_DIM + WRIST_DIM, 2 * HAND_DIM + WRIST_DIM)
RIGHT_WRIST_SLICE = slice(2 * HAND_DIM + WRIST_DIM, ACTION_DIM)

SUPPORTED_ACTION_MODES = (f"qpos{HAND_DIM}_rel6d", f"relqpos{HAND_DIM}_rel6d")


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


class TianjiSharpaDataset(data.Dataset):
    """Latent-space Tianji/Sharpa dataset (22-DoF hands).

    The returned action/state layout is:
      [L_qpos22 | L_wrist_abs_xyz_m3_rot6d6 | R_qpos22 | R_wrist_abs_xyz_m3_rot6d6]

    Qpos dims are normalized with the sharpa stat JSON (`qpos_normalization`:
    mean_std, min_max, or quantile). Wrist relative xyz is normalized in the
    model via `model.normalization.wrist_xyz` (mean_std | min_max | quantile).
    Wrist dims are absolute in meters/rot6d in the dataset and converted to
    chunk-relative form inside the model.
    """

    # Embodiment dimensions exposed as class attributes so callers (stat
    # scripts, tests, model config) can introspect instead of hard-coding.
    HAND_DIM = HAND_DIM
    WRIST_DIM = WRIST_DIM
    QPOS_DIM = QPOS_DIM
    ACTION_DIM = ACTION_DIM

    def __init__(
        self,
        dataset_dir: str = "/path/to/tianji_sharpa_dataset",
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
        # Sharpa exports pre-extracted per-signal .pt files; there is no packed
        # tjwj.pt. Kept as a knob only so the config surface matches Wuji.
        tjwj_subpath: Optional[str] = None,
        action_stats_path: str = "data/tianji_sharpa/stat/sharpa_stat.json",
        qpos_normalization: str = "mean_std",
        task_episode_skip_first: Optional[List[Dict[str, Any]]] = None,  # per-task drop first N episodes
        max_latent_window: int = 10,
        video_hz: int = 30,
        video_action_freq_ratio: int = 6,
        action_upsample_ratio: int = 2,
        action: Optional[Dict[str, Any]] = None,
        raw_action_hz: Optional[int] = 60,
        train_action_hz: Optional[int] = 30,
        action_mode: str = f"qpos{HAND_DIM}_rel6d",
        action_from_state: bool = True,
        use_state: bool = False,
        val: bool = False,
        task_mode: str = "multi",
        task_name: Optional[str] = None,
        max_episodes: Optional[int] = None,
        include_fail: bool = False,
        fail_task_name: Optional[str] = None,
        fail_sample_fraction: Optional[float] = None,
        text_pool_source: str = "episode",
        include_old: bool = False,
        cache_dir: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        use_index_cache: bool = True,
        manifest_path: Optional[str] = None,
        manifest_sample_type: Optional[str] = None,
        value_target: Optional[str] = None,
        progress_c: float = 0.01,
        treat_as_irrelevant: bool = False,
        bootstrap_start_window: bool = False,
        bootstrap_chunk_size: int = 2,
        bootstrap_start_window_prob: float = 0.5,
        view_dropout: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if action_mode not in set(SUPPORTED_ACTION_MODES):
            raise ValueError(
                "Tianji/Sharpa supports action_mode="
                f"{' or '.join(SUPPORTED_ACTION_MODES)}, got {action_mode}"
            )
        self.action_mode = str(action_mode)
        self.clickhouse = _mapping_to_plain_dict(clickhouse)
        self.episode_index_source = str(episode_index_source or "auto").lower()
        self.manifest_path = str(Path(manifest_path).expanduser()) if manifest_path else None
        self.manifest_sample_type = _to_plain_container(manifest_sample_type)
        if self.manifest_path and self.episode_index_source == "auto":
            self.episode_index_source = "manifest"
        if self.episode_index_source not in {"auto", "dataset_dir", "clickhouse", "manifest"}:
            raise ValueError(f"Unsupported Tianji/Sharpa episode_index_source: {self.episode_index_source}")
        if self.episode_index_source == "manifest" and not self.manifest_path:
            raise ValueError("Tianji/Sharpa episode_index_source=manifest requires manifest_path")
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
        self.text_pool_source = str(text_pool_source or "episode").lower()
        if self.text_pool_source not in {"episode", "task_success"}:
            raise ValueError(
                "text_pool_source must be 'episode' or 'task_success', got "
                f"{text_pool_source!r}"
            )
        self._success_text_pool: Dict[str, List[str]] = {}
        self.include_old = bool(include_old)
        self.cache_dir = cache_dir
        self.index_cache_path = str(Path(index_cache_path).expanduser()) if index_cache_path else None
        self.use_index_cache = bool(use_index_cache)
        self.value_target = value_target
        if self.value_target is not None and self.value_target not in {"progress", "ttc", "vlac"}:
            raise ValueError(
                f"TianjiSharpaDataset: unsupported value_target {self.value_target!r}; "
                "use 'progress', 'ttc', or 'vlac'"
            )
        self.progress_c = progress_c
        self.treat_as_irrelevant = treat_as_irrelevant
        self.bootstrap_start_window = bool(bootstrap_start_window)
        self.bootstrap_chunk_size = int(bootstrap_chunk_size)
        self.bootstrap_start_window_prob = float(min(max(bootstrap_start_window_prob, 0.0), 1.0))
        self._init_view_dropout(view_dropout)
        if self.task_mode == "single" and not self.task_name:
            raise ValueError("Single task mode requires task_name")

        self.action_stats = self._load_json(action_stats_path)
        self.qpos_normalization = str(qpos_normalization or "mean_std")
        self._init_qpos_normalization_stats()
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
                logger.info("TianjiSharpaDataset: skipped first %d episodes of task pattern %r", n, pat)
            self.episodes = kept
        if not self.episodes:
            raise RuntimeError(f"No Tianji/Sharpa episodes found under {self.dataset_dir}")
        self._configure_fail_sampling()
        self._build_success_text_pool()

        logger.info(
            "TianjiSharpaDataset: episodes=%d, max_latent_window=%d, actions_per_latent=%d, "
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

    def _task_text_scope(self, task_name: str) -> str:
        """The task directory a record belongs to, with the pool suffix removed.

        ``<task>`` and ``<task>/fail`` collapse to ``<task>``: it is the unit
        within which the two pools have to be indistinguishable, and also the
        unit that shares one physical scene, since a Sharpa task directory is
        one capture day at one table height.
        """
        parts = [p for p in str(task_name).strip("/").split("/") if p]
        while parts and parts[-1].lower() in {"fail", "suboptimal", "success"}:
            parts.pop()
        return "/".join(parts)

    def _build_success_text_pool(self) -> None:
        """Collect the success captions that fail records borrow.

        Same leak and same remedy as ``TianjiWujiDataset``: the exported
        captions describe the outcome, so a value or world model can read the
        text instead of the video. Under ``text_pool_source=task_success`` a
        fail record reads the caption directory of a uniformly drawn success
        episode from the same task directory, which makes the conditioning
        text's marginal distribution the success distribution in both pools and
        closes every channel at once, because the redirect swaps the directory
        the candidate glob runs in rather than one file at a time.
        """
        if self.text_pool_source != "task_success" or not self.episodes:
            return

        needed: Dict[str, int] = {}
        for ep in self.episodes:
            scope = self._task_text_scope(str(ep.get("task_name", "")))
            if bool(ep.get("is_fail_class")):
                needed[scope] = needed.get(scope, 0) + 1
            else:
                path = self._as_path_str(ep.get("language_path"))
                if path:
                    self._success_text_pool.setdefault(scope, []).append(path)
        # Sorted so a given seed draws the same caption on every rank and every
        # restart, and so ``val`` can freeze on index 0; the filesystem walk
        # order is not stable enough for either.
        for scope in self._success_text_pool:
            self._success_text_pool[scope] = sorted(set(self._success_text_pool[scope]))

        # A task with fail records and no success pool cannot borrow, and the
        # fallback would be reading its own caption -- exactly the leak this
        # switch closes, reinstated for a subset the log would not name.
        # 0828_80cm_cut_the_cucumber is the real case: 9 fail, no success pool.
        starved = sorted(s for s in needed if not self._success_text_pool.get(s))
        if starved:
            raise ValueError(
                "text_pool_source=task_success requires a success pool in every task "
                "directory that contributes fail records, but "
                f"{len(starved)} task(s) have none: {starved[:8]}. "
                "Remove those tasks from fail_task_name, or add their success episodes."
            )
        logger.info(
            "TianjiSharpaDataset: text_pool_source=task_success, success captions per "
            "task %s; fail records borrowing %s",
            {s: len(v) for s, v in sorted(self._success_text_pool.items())},
            {s: n for s, n in sorted(needed.items())},
        )

    def _draw_success_text_path(self, ep: Dict[str, Any]) -> Optional[str]:
        """Which success episode's caption directory this record reads, or None.

        None means "read your own", which happens only for success records.
        Fail records are guaranteed a non-empty pool by
        ``_build_success_text_pool``.
        """
        if self.text_pool_source != "task_success":
            return None
        if not bool(ep.get("is_fail_class")):
            return None
        pool = self._success_text_pool.get(
            self._task_text_scope(str(ep.get("task_name", "")))
        )
        if not pool:
            return None
        return pool[0] if self.val else pool[random.randint(0, len(pool) - 1)]

    def _configure_fail_sampling(self) -> None:
        self._main_episodes = [
            ep for ep in self.episodes if not bool(ep.get("is_fail_class"))
        ]
        self._fail_episodes = [
            ep for ep in self.episodes if bool(ep.get("is_fail_class"))
        ]
        self._fail_sampling_probability = None
        if self.val or self.fail_sample_fraction is None:
            return

        fail_probability = self.fail_sample_fraction
        if fail_probability > 0.0 and not self._fail_episodes:
            raise ValueError(
                "fail_sample_fraction requires fail episodes, but the fail pool is empty"
            )
        if fail_probability < 1.0 and not self._main_episodes:
            raise ValueError(
                "fail_sample_fraction requires success episodes, but the success pool is empty"
            )
        self._fail_sampling_probability = fail_probability
        logger.info(
            "TianjiSharpaDataset: fail sample fraction=%.4f, "
            "success_records=%d, fail_records=%d",
            fail_probability,
            len(self._main_episodes),
            len(self._fail_episodes),
        )

    def _sample_training_episode(self) -> Dict[str, Any]:
        fail_probability = self._fail_sampling_probability
        if fail_probability is None:
            return random.choice(self.episodes)
        pool = (
            self._fail_episodes
            if random.random() < fail_probability
            else self._main_episodes
        )
        return random.choice(pool)

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
        if not self.include_fail and self._path_has_fail(path):
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

    # Per-sample embodiment tag carried into the batch (TactAlign Stage-0 plumbing).
    # 1 = human (human_hand glove), 0 = robot/dex. Sharpa episodes are all robot,
    # so this is constant here; kept for batch-schema parity with the Wuji loader.
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

    def _index_task_patterns(self) -> List[str]:
        return self._main_task_patterns() + self._fail_task_patterns()

    def _patterns_match(self, task_name: str, patterns: List[str]) -> bool:
        return any(self._name_matches(task_name, pat) for pat in patterns)

    def _fail_task_allowed(self, task_name: str) -> bool:
        patterns = self._fail_task_patterns()
        return bool(patterns) and self._patterns_match(task_name, patterns)

    def _classify_task(self, task_name: str) -> Optional[str]:
        """Return ``'fail'``, ``'main'``, or None. Fail wins if both match."""
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
        """
        if self.treat_as_irrelevant:
            return True
        return bool(ep.get("is_fail_class"))

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
            "schema": "tianji_sharpa_episode_index_v1",
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

    def _load_cached_index(self) -> Optional[List[Dict[str, Any]]]:
        if not self.use_index_cache or not self.index_cache_path:
            return None
        path = Path(self.index_cache_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable Tianji/Sharpa index cache %s: %s", path, exc)
            return None
        if payload.get("meta") != self._index_cache_meta():
            logger.warning("Ignoring stale Tianji/Sharpa index cache: %s", path)
            return None
        episodes = payload.get("episodes", [])
        logger.info("Loaded Tianji/Sharpa index cache: %s (%d episodes)", path, len(episodes))
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
                for key in ("sample_type", "source_type", "is_fail_class"):
                    if key in ep:
                        rec[key] = ep[key]
                rec["source_episode_name"] = base_name
                if not self._latents_exist_under(ep_dir, rec["latent_root"]):
                    skipped_missing += 1
                    continue
                expanded.append(rec)

        if not expanded:
            raise RuntimeError(
                "Tianji/Sharpa multi-phase expansion produced 0 episodes. "
                f"latent_root={self.latent_root!r}, latent_phases="
                f"{sorted(self.latent_phases_set) if self.latent_phases_set is not None else None}. "
                "Ensure `{N}phase_5hz` VAE features exist on disk."
            )
        logger.info(
            "Tianji/Sharpa latent_phases expansion: in=%d out=%d skipped_missing_phase=%d "
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
            raise ValueError("Tianji/Sharpa clickhouse config is missing")
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
                "ClickHouse table schema missing required Tianji/Sharpa columns. "
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
            episodes.append(rec)
        logger.info("ClickHouse Tianji/Sharpa index returned %d episodes from %s", len(episodes), table)
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
            raise ValueError(f"Tianji/Sharpa manifest episodes must be a list: {path}")

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
            record["sample_type"] = sample_type
            record["source_type"] = str(row.get("source_type", sample_type)).lower()
            episodes.append(record)

        logger.info(
            "Loaded Tianji/Sharpa manifest: %s sample_type=%s episodes=%d",
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
                raise ValueError("sharpa stat JSON missing normalization.qpos mean/std for mean_std")
            self.qpos_mean = torch.tensor(mean, dtype=torch.float32)
            self.qpos_std = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
            if self.qpos_mean.shape[0] != QPOS_DIM or self.qpos_std.shape[0] != QPOS_DIM:
                raise ValueError(
                    f"sharpa stat JSON qpos mean/std must be {QPOS_DIM}D "
                    f"({HAND_DIM} per hand), got {self.qpos_mean.shape[0]}"
                )
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
                f"sharpa stat JSON missing qpos {low_key}/{high_key} for {norm_mode}; "
                "re-run data/tianji_sharpa/analyze_tianji_sharpa_qpos.py"
            )
        self.qpos_min = torch.tensor(low, dtype=torch.float32)
        high_t = torch.tensor(high, dtype=torch.float32)
        self.qpos_range = (high_t - self.qpos_min).clamp(min=1e-6)
        if self.qpos_min.shape[0] != QPOS_DIM or self.qpos_range.shape[0] != QPOS_DIM:
            raise ValueError(
                f"sharpa stat JSON qpos {low_key}/{high_key} must be {QPOS_DIM}D "
                f"({HAND_DIM} per hand), got {self.qpos_min.shape[0]}"
            )

    def _normalize_qpos(
        self, left_qpos: torch.Tensor, right_qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize both hands jointly, then split back into HAND_DIM halves."""
        for side, tensor in (("left", left_qpos), ("right", right_qpos)):
            if tensor.shape[-1] != HAND_DIM:
                raise ValueError(
                    f"Sharpa hand qpos {side} must be {HAND_DIM}-DoF, "
                    f"got {tuple(tensor.shape)}"
                )
        qpos = torch.cat([left_qpos, right_qpos], dim=-1).float()
        if self.qpos_normalization == "mean_std":
            qpos = (qpos - self.qpos_mean) / self.qpos_std
        elif self.qpos_normalization == "min_max":
            qpos = (qpos - self.qpos_min) / self.qpos_range
        else:
            qpos = torch.maximum(torch.minimum(qpos, self.qpos_min + self.qpos_range), self.qpos_min)
            qpos = (qpos - self.qpos_min) / self.qpos_range
        return qpos[:, :HAND_DIM], qpos[:, HAND_DIM:]

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
            episode=ep.get("episode_path"),
        )

    @staticmethod
    def _downsample_sequence(sequence: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1:
            return sequence
        return sequence[::stride].contiguous()

    def _load_state_abs(self, ep: Dict[str, Any]) -> torch.Tensor:
        """Observed state as [L_qpos22 | L_wrist9 | R_qpos22 | R_wrist9]."""
        hand_path = ep.get("hand_path")
        endpose_path = ep.get("endpose_path")
        if not (hand_path and endpose_path and self._exists(hand_path) and self._exists(endpose_path)):
            raise FileNotFoundError(
                f"{self.action_mode} requires hand_qpos.pt + arm_endpose.pt in the episode dir "
                f"(hand={hand_path}, endpose={endpose_path})"
            )
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
        left_qpos, right_qpos = self._normalize_qpos(left_qpos, right_qpos)
        left_wrist = _endpose_to_wrist9(left_wrist_raw)
        right_wrist = _endpose_to_wrist9(right_wrist_raw)
        return self._downsample_sequence(
            torch.cat([left_qpos, left_wrist, right_qpos, right_wrist], dim=-1),
            self._action_stride(hand, ep),
        )

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

    def _load_true_action_abs(self, ep: Dict[str, Any]) -> torch.Tensor:
        """Load the commanded action vector.

        Hand dims: hand_action.pt (22-DoF, same joint space as hand_qpos).
        Wrist dims: arm_action_endpose.pt (EE xyz+quat, the eef_pose field of the
        14-wide luna arm action pack), not the 7-DoF joint-space arm_action.pt
        and not the state-side arm_endpose.pt.
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

        left_qpos, right_qpos = self._normalize_qpos(left_qpos, right_qpos)
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
        borrowed = self._draw_success_text_path(ep)
        candidates = self._resolve_language_candidates(ep, borrowed)
        if not candidates:
            source = borrowed or ep.get("language_path")
            raise FileNotFoundError(f"No language .pt candidates under: {source}")

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
                return self._build_vlac_values(start, window_size, total_latents, negate=True)
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
        for _ in range(8):
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

                state_abs = self._load_state_abs(ep)
                action_abs = (
                    state_abs
                    if self.action_from_state
                    else self._load_true_action_abs(ep)
                )
                if action_abs.shape[0] != state_abs.shape[0]:
                    raise ValueError(
                        "Action/state sequence lengths must match: "
                        f"{action_abs.shape[0]} != {state_abs.shape[0]}"
                    )
                if action_abs.shape[-1] != ACTION_DIM:
                    raise ValueError(
                        f"Sharpa action vector must be {ACTION_DIM}D "
                        f"({HAND_DIM}+{WRIST_DIM} per side), got {action_abs.shape[-1]}"
                    )
                phase_start_30 = int(ep.get("phase_start_30", 0))
                actions = self._gather_per_latent(
                    action_abs,
                    start,
                    actual_latents,
                    frame_stride,
                    phase_start_30=phase_start_30,
                    per_latent_len=self.actions_per_latent,
                )
                states = self._gather_states(
                    state_abs, start, actual_latents, frame_stride, phase_start_30=phase_start_30
                )

                target_action_len = W * self.actions_per_latent
                if actions.shape[0] < target_action_len:
                    actions = torch.cat([actions, actions[-1:].expand(target_action_len - actions.shape[0], -1)], dim=0)
                if states.shape[0] < W:
                    states = torch.cat([states, states[-1:].expand(W - states.shape[0], -1)], dim=0)

                extra_action_sequences: Dict[str, torch.Tensor] = {}

                is_irrelevant = self._episode_is_task_irrelevant(ep)
                values = self._build_value_targets(
                    start, W, total_latents, treat_as_irrelevant=is_irrelevant
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
                    enabled=self.bootstrap_start_window,
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
                    SAMPLE_CLASS_KEY: int(
                        SampleClass.TASK_IRRELEVANT if is_irrelevant else SampleClass.MAIN
                    ),
                    "is_success": int(not is_irrelevant),
                    "source": self._resolve_source(ep.get("task_name")),
                }
                if extra_action_sequences:
                    sample.update(extra_action_sequences)
                if values is not None:
                    sample["values"] = values
                return sample
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tianji/Sharpa sample error (%s): %s", ep.get("episode_path"), exc)
        return None
