"""Mixing corpora that were recorded by different robots into one run.

Two robots running the same 20-DoF hand still have their own joint calibration,
so their qpos envelopes do not line up: normalizing wuji2 against
lingqiaoshou's `min_max` puts 25 of 40 dimensions outside [0, 1], and the
loaders do not clip. Hand qpos is therefore normalized inside each dataset,
against that corpus's own `action_stats_path`.

Wrist xyz is normalized after conversion to chunk-relative coordinates. Mixed
pretrain can use the same source-specific stats files for hand and wrist.
Other configurations retain their configured shared wrist envelope.

These helpers are kept out of `train/train.py` so they can be exercised
without importing the model, which touches CUDA at class definition time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def co_training_shares(
    lengths: Sequence[int], explicit: Sequence[Optional[float]]
) -> list[float]:
    """Share of the policy pool each corpus gets.

    Default is size-proportional, which is what a plain shuffle over the
    concatenation would give. A corpus naming `sampling_ratio` takes exactly
    that share; the ones that do not split the remainder in proportion to their
    size, so pinning one corpus does not silently reweight the others against
    each other.
    """
    fixed = sum(float(r) for r in explicit if r is not None)
    if fixed > 1.0 + 1e-9:
        raise ValueError(
            f"co-dataset sampling_ratio values sum to {fixed:.3f}, which leaves "
            "nothing for the main dataset"
        )
    free_len = sum(length for length, r in zip(lengths, explicit) if r is None)
    remaining = max(0.0, 1.0 - fixed)
    shares = []
    for length, r in zip(lengths, explicit):
        if r is not None:
            shares.append(float(r))
        elif free_len > 0:
            shares.append(remaining * length / free_len)
        else:
            shares.append(0.0)
    return shares


def assert_stats_share_action_space(stats_paths: Sequence[str], *, allow_untyped: bool = False) -> None:
    """Refuse corpora whose action vectors are not the same layout.

    Sharpa's 22-DoF hands make a 62-D action; wuji's 20-DoF hands make 58. One
    action expert cannot serve both, and nothing downstream would notice.
    """
    seen = {}
    for path in stats_paths:
        stats = json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))
        signature = (stats.get("action_dim"), stats.get("action_mode"))
        if allow_untyped and signature == (None, None):
            continue
        seen[path] = signature
    if len(set(seen.values())) > 1:
        detail = "; ".join(
            f"{p}: action_dim={d} action_mode={m}" for p, (d, m) in seen.items()
        )
        raise ValueError(
            f"co-training corpora disagree on the action space ({detail}). "
            "One action expert cannot serve both layouts."
        )


def qpos20_wrist_xyz_bounds_from_stats(
    stats_paths: Sequence[str],
    wrist_xyz_normalization: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Wrist xyz envelope covering every corpus, in the mode's own keys."""
    if not stats_paths:
        return tuple([-0.05] * 6), tuple([0.05] * 6)
    mode = str(wrist_xyz_normalization or "min_max").lower()
    if mode in {"quantile", "q01_q99"}:
        low_key, high_key = "xyz_q01", "xyz_q99"
    elif mode == "min_max":
        low_key, high_key = "xyz_min", "xyz_max"
    else:
        return tuple([-0.05] * 6), tuple([0.05] * 6)

    lows: list[list[float]] = []
    highs: list[list[float]] = []
    for stats_path in stats_paths:
        stats = json.loads(resolve_repo_path(stats_path).read_text(encoding="utf-8"))
        wrist_norm = stats.get("normalization", {}).get("wrist_relative", {})
        xyz_min = wrist_norm.get(low_key)
        xyz_max = wrist_norm.get(high_key)
        if xyz_min is None or xyz_max is None:
            raise ValueError(
                f"{stats_path} missing normalization.wrist_relative {low_key}/{high_key} "
                f"for wrist_xyz_normalization={mode!r}; "
                "re-run the embodiment stat script (data/tianji_{wuji,sharpa}/analyze_tianji_*_qpos.py)"
            )
        if len(xyz_min) != 6 or len(xyz_max) != 6:
            raise ValueError(f"Expected 6D wrist relative xyz bounds in {stats_path}")
        lows.append([float(x) for x in xyz_min])
        highs.append([float(x) for x in xyz_max])

    return (
        tuple(min(col) for col in zip(*lows)),
        tuple(max(col) for col in zip(*highs)),
    )


def qpos20_wrist_xyz_mean_std_from_stats(
    stats_paths: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if not stats_paths:
        return tuple([0.0] * 6), tuple([1.0] * 6), 1e-4
    if len(stats_paths) > 1:
        # Pooled mean/std cannot be recovered from per-corpus mean/std without
        # the sample counts, and averaging them is a different statistic.
        raise ValueError(
            "wrist_xyz_normalization=mean_std cannot span co-training corpora "
            f"({', '.join(stats_paths)}); use min_max or quantile, whose envelopes union."
        )
    stats_path = stats_paths[0]
    stats = json.loads(resolve_repo_path(stats_path).read_text(encoding="utf-8"))
    wrist_norm = stats.get("normalization", {}).get("wrist_relative", {})
    xyz_mean = wrist_norm.get("xyz_mean")
    xyz_std = wrist_norm.get("xyz_std")
    if xyz_mean is None or xyz_std is None:
        raise ValueError(
            f"{stats_path} missing normalization.wrist_relative xyz_mean/xyz_std; "
            "re-run the embodiment stat script (data/tianji_{wuji,sharpa}/analyze_tianji_*_qpos.py)"
        )
    if len(xyz_mean) != 6 or len(xyz_std) != 6:
        raise ValueError(f"Expected 6D wrist relative xyz mean/std in {stats_path}")
    std_min = float(wrist_norm.get("std_min", 1e-4))
    return (
        tuple(float(x) for x in xyz_mean),
        tuple(float(x) for x in xyz_std),
        std_min,
    )


def _wrist_relative_keys(wrist_xyz_normalization: str) -> tuple[str, str]:
    mode = str(wrist_xyz_normalization or 'min_max').lower()
    if mode in {'quantile', 'q01_q99'}:
        return 'xyz_q01', 'xyz_q99'
    if mode == 'min_max':
        return 'xyz_min', 'xyz_max'
    if mode == 'mean_std':
        return 'xyz_mean', 'xyz_std'
    return 'xyz_min', 'xyz_max'


def qpos20_wrist_xyz_stats_from_path(
    stats_path: str,
    wrist_xyz_normalization: str,
) -> dict[str, tuple[float, ...]]:
    stats = json.loads(resolve_repo_path(stats_path).read_text(encoding='utf-8'))
    wrist_norm = stats.get('normalization', {}).get('wrist_relative', {})
    low_key, high_key = _wrist_relative_keys(wrist_xyz_normalization)
    xyz_min = wrist_norm.get(low_key)
    xyz_max = wrist_norm.get(high_key)
    if xyz_min is None or xyz_max is None:
        raise ValueError(
            f'{stats_path} missing normalization.wrist_relative {low_key}/{high_key} '
            f'for wrist_xyz_normalization={wrist_xyz_normalization!r}'
        )
    if len(xyz_min) != 6 or len(xyz_max) != 6:
        raise ValueError(f'Expected 6D wrist relative stats in {stats_path}')
    low = tuple(float(x) for x in xyz_min)
    high = tuple(float(x) for x in xyz_max)
    if not all(math.isfinite(x) for x in (*low, *high)):
        raise ValueError(f'{stats_path} has nonfinite wrist stats')
    if wrist_xyz_normalization == 'mean_std':
        if any(x <= 0 for x in high):
            raise ValueError(f'{stats_path} has nonpositive wrist std')
        return {'min': low, 'max': high, 'mean': low, 'std': high}
    if any(lo >= hi for lo, hi in zip(low, high)):
        raise ValueError(f'{stats_path} has an invalid wrist interval')
    return {'min': low, 'max': high}


def wrist_stats_paths_by_source(config: Any) -> dict[str, str]:
    ds = cfg_get(config, 'dataset', config)
    default_path = cfg_get(ds, 'action_stats_path')
    if not default_path:
        raise ValueError('Per-source wrist stats require dataset.action_stats_path')
    mapping = {'_default': str(default_path)}
    for name, path in (cfg_get(ds, 'action_stats_by_dataset') or {}).items():
        if name == '_default' or not path:
            raise ValueError(f'Invalid per-source stats entry: {name!r}')
        mapping[str(name)] = str(path)
    return mapping


def mixed_pretrain_wrist_per_dataset_enabled(config: Any) -> bool:
    ds = cfg_get(config, 'dataset', config)
    return bool(cfg_get(ds, 'wrist_xyz_per_dataset', False))


def build_wrist_bounds_by_source(
    config: Any,
    wrist_xyz_normalization: str,
) -> dict[str, dict[str, tuple[float, ...]]]:
    mapping = wrist_stats_paths_by_source(config)
    return {
        source: qpos20_wrist_xyz_stats_from_path(path, wrist_xyz_normalization)
        for source, path in mapping.items()
    }
