"""Tactile baseline calibration loader — shared by training dataset and
inference normalizer for byte-identical normalization across both pipelines.

Baseline files (produced by an offline analysis run; see
`tactile_wear_strain_noise_final_<timestamp>/` directory):

    baseline_mean_left.npy   / baseline_mean_right.npy    — per-taxel time mean   [24, 32] float32
    baseline_median_left.npy / baseline_median_right.npy  — per-taxel time median [24, 32] float32
    baseline_std_left.npy    / baseline_std_right.npy     — per-taxel noise scale [24, 32] float32
    valid_mask_left.npy      / valid_mask_right.npy       — per-taxel valid mask  [24, 32] bool

Baselines are stored in the **post-min-max-normalization** space (i.e. they
were collected via raw_adc → `(raw - 300) / (2100 - 300)` clip [0,1], then
median/mean reduced across time). Subtraction therefore happens AFTER the
existing min-max stage:

    x = clip((raw_adc - base_adc) / (max_adc - base_adc), 0, 1)
    x = clip(x - baseline, 0, 1)         # ← this module
    x[~valid_mask] = 0

`median` is the recommended `kind` for static (rest-pose) calibration data:
robust to occasional spikes from random taxel firings during recording.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

TACTILE_DIM = 1536
SIDE_DIM = 768


def load_baseline_calibration(
    baseline_dir: str | Path,
    kind: str = "median",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (baseline, valid_mask) both shape [1536] (left then right).

    kind ∈ {"median", "mean"} selects which reduction to load. Median is the
    recommended default (see module docstring).
    """
    if kind not in {"median", "mean"}:
        raise ValueError(f"kind must be 'median' or 'mean', got {kind!r}")
    d = Path(baseline_dir).expanduser()
    if not d.is_dir():
        raise FileNotFoundError(f"baseline_dir not found or not a directory: {d}")

    def _load(name: str) -> np.ndarray:
        p = d / f"{name}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"missing baseline file: {p}")
        return np.load(p)

    base_l = _load(f"baseline_{kind}_left").astype(np.float32).reshape(-1)
    base_r = _load(f"baseline_{kind}_right").astype(np.float32).reshape(-1)
    mask_l = _load("valid_mask_left").astype(bool).reshape(-1)
    mask_r = _load("valid_mask_right").astype(bool).reshape(-1)
    if base_l.size != SIDE_DIM or base_r.size != SIDE_DIM:
        raise ValueError(
            f"baseline must flatten to {SIDE_DIM} per side, got {base_l.size}/{base_r.size}"
        )
    if mask_l.size != SIDE_DIM or mask_r.size != SIDE_DIM:
        raise ValueError(
            f"valid_mask must flatten to {SIDE_DIM} per side, got {mask_l.size}/{mask_r.size}"
        )

    baseline = torch.from_numpy(np.concatenate([base_l, base_r], axis=0)).to(device=device, dtype=dtype)
    valid_mask = torch.from_numpy(np.concatenate([mask_l, mask_r], axis=0)).to(device=device, dtype=torch.bool)
    return baseline, valid_mask


def load_baseline_from_json(
    json_path: str | Path,
    name: str,
    kind: str = "median",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load a named baseline entry from `wj_tactile.json["baselines"][name]`.

    Returns (baseline[1536], valid_mask[1536], polarity) — analogous to
    load_baseline_calibration() but from JSON instead of npy directory.

    JSON schema expected under `baselines.<name>`:
        {
          "polarity": "positive" | "negative",
          "median_left":  [...768 floats...],
          "median_right": [...],
          "mean_left":    [...],   # optional
          "mean_right":   [...],   # optional
          "valid_mask_left":  [...768 bools...],   # optional; if absent, copy from top-level valid_channel_mask
          "valid_mask_right": [...],               # same
          ...other metadata...
        }
    """
    import json
    if kind not in {"median", "mean"}:
        raise ValueError(f"kind must be 'median' or 'mean', got {kind!r}")
    p = Path(json_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"json_path not found: {p}")
    data = json.loads(p.read_text())
    baselines_section = data.get("baselines")
    if not isinstance(baselines_section, dict) or name not in baselines_section:
        available = list(baselines_section.keys()) if isinstance(baselines_section, dict) else []
        raise KeyError(f"baseline name {name!r} not in {p} (available: {available})")
    entry = baselines_section[name]

    key_l, key_r = f"{kind}_left", f"{kind}_right"
    if key_l not in entry or key_r not in entry:
        raise KeyError(f"baseline {name!r} missing {key_l}/{key_r} (has: {sorted(entry.keys())})")
    base_l = np.asarray(entry[key_l], dtype=np.float32).reshape(-1)
    base_r = np.asarray(entry[key_r], dtype=np.float32).reshape(-1)
    if base_l.size != SIDE_DIM or base_r.size != SIDE_DIM:
        raise ValueError(f"baseline {name!r} {kind} must be {SIDE_DIM}-D per side, got {base_l.size}/{base_r.size}")

    # valid_mask: entry's own takes precedence; fall back to top-level json valid_channel_mask
    if "valid_mask_left" in entry and "valid_mask_right" in entry:
        mask_l = np.asarray(entry["valid_mask_left"], dtype=bool).reshape(-1)
        mask_r = np.asarray(entry["valid_mask_right"], dtype=bool).reshape(-1)
    elif "valid_channel_mask" in data:
        mask_l = np.asarray(data["valid_channel_mask"]["left"], dtype=bool).reshape(-1)
        mask_r = np.asarray(data["valid_channel_mask"]["right"], dtype=bool).reshape(-1)
    else:
        mask_l = np.ones(SIDE_DIM, dtype=bool)
        mask_r = np.ones(SIDE_DIM, dtype=bool)

    polarity = str(entry.get("polarity", "positive"))
    if polarity not in {"positive", "negative"}:
        raise ValueError(f"baseline {name!r} has invalid polarity {polarity!r}")

    baseline = torch.from_numpy(np.concatenate([base_l, base_r])).to(device=device, dtype=dtype)
    valid_mask = torch.from_numpy(np.concatenate([mask_l, mask_r])).to(device=device, dtype=torch.bool)
    return baseline, valid_mask, polarity


def load_tactile_scale_from_json(json_path: str | Path, name: str) -> tuple[float, float]:
    """Return (tactile_max, tactile_min) for a named baseline entry.

    These define the native-space full-scale / floor used by
    `normalize_polarity_native`:
        - dex_default: (1.0, 0.0)   — raw/2035 float space, full-scale = 1.0
        - human_v1_raw: (2100.0, 0.0) — uint16 ADC space, full-scale = 2100

    Raises KeyError if the entry lacks `tactile_max` (native-space normalization
    requires it — legacy [0,1]-space baselines without it should not be used here).
    """
    import json
    p = Path(json_path).expanduser()
    data = json.loads(p.read_text())
    entry = data.get("baselines", {}).get(name)
    if entry is None:
        raise KeyError(f"baseline name {name!r} not in {p}")
    if "tactile_max" not in entry:
        raise KeyError(
            f"baseline {name!r} missing 'tactile_max' — native-space normalization "
            f"requires tactile_max/tactile_min on the baseline entry"
        )
    return float(entry["tactile_max"]), float(entry.get("tactile_min", 0.0))


def append_baseline_to_json(
    json_path: str | Path,
    name: str,
    polarity: str,
    median_left: np.ndarray,  median_right: np.ndarray,
    mean_left: np.ndarray,    mean_right: np.ndarray,
    std_left: np.ndarray,     std_right: np.ndarray,
    valid_mask_left: Optional[np.ndarray] = None,
    valid_mask_right: Optional[np.ndarray] = None,
    extra_meta: Optional[dict] = None,
) -> None:
    """Append a named baseline entry to wj_tactile.json["baselines"][name].

    Overwrites if `name` already exists. Creates top-level `baselines: {}` section
    if missing. All arrays are stored as lists of native floats / bools for JSON.
    """
    import json
    if polarity not in {"positive", "negative"}:
        raise ValueError(f"polarity must be 'positive'|'negative', got {polarity!r}")
    p = Path(json_path).expanduser()
    data = json.loads(p.read_text()) if p.is_file() else {}
    baselines = data.setdefault("baselines", {})

    def _to_list(a: np.ndarray) -> list:
        return np.asarray(a, dtype=np.float32).reshape(-1).tolist()

    entry: dict = {
        "polarity": polarity,
        "median_left":  _to_list(median_left),  "median_right": _to_list(median_right),
        "mean_left":    _to_list(mean_left),    "mean_right":   _to_list(mean_right),
        "std_left":     _to_list(std_left),     "std_right":    _to_list(std_right),
    }
    if valid_mask_left is not None and valid_mask_right is not None:
        entry["valid_mask_left"]  = np.asarray(valid_mask_left,  dtype=bool).reshape(-1).tolist()
        entry["valid_mask_right"] = np.asarray(valid_mask_right, dtype=bool).reshape(-1).tolist()
    if extra_meta:
        entry.update(extra_meta)
    baselines[name] = entry
    p.write_text(json.dumps(data, indent=2))


def apply_baseline(
    x_normed: torch.Tensor,
    baseline: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    polarity: str = "positive",
    adc_base: float = 300.0,
    adc_max: float = 2100.0,
    adc_min: float = 0.0,
) -> torch.Tensor:
    """Per-taxel dynamic-range baseline subtraction.

    Inputs are in the normalized `[0,1]` Stage-1 space (i.e. after
    `clip((raw_adc - adc_base) / (adc_max - adc_base), 0, 1)`). Both x_normed and
    baseline are reconstructed to raw ADC, then the per-taxel formula is applied:

        positive polarity (contact → ADC up, e.g. dex hand piezoresistive):
            signal = (raw - base_raw) / (adc_max - base_raw)

        negative polarity (contact → ADC down, e.g. human glove capacitive):
            signal = (base_raw - raw) / (base_raw - adc_min)

    Each taxel gets its OWN denominator (= its physical dynamic range from baseline
    to the saturation/floor limit), so two taxels with different baselines but the
    same ADC drop will report different signal magnitudes proportional to how much
    of their available range was consumed. This is physically more meaningful than
    the previous fixed-denominator (adc_max - adc_base = 1800) formula.

    Args:
        x_normed: [..., D] tensor in [0,1] (Stage-1 normalized).
        baseline: [..., D] per-taxel baseline in [0,1] (loaded from npy).
        valid_mask: optional [D] bool tensor; invalid taxels → 0.
        polarity: "positive" or "negative".
        adc_base / adc_max: scalar hardware constants used to reconstruct raw from
            normalized (defaults from wj_tactile.json: 300, 2100).
        adc_min: scalar physical floor for the negative-polarity denominator
            (default 0 = ADC zero).

    Returns:
        signal in [0,1], invalid taxels zeroed.
    """
    if x_normed.shape[-1] != baseline.shape[-1]:
        raise ValueError(
            f"shape mismatch: x last-dim={x_normed.shape[-1]}, baseline last-dim={baseline.shape[-1]}"
        )
    base_norm = baseline.to(device=x_normed.device, dtype=x_normed.dtype)
    scale = float(adc_max) - float(adc_base)
    # Reconstruct raw ADC values (exact for any non-clipped input; for clipped
    # inputs at 0 we recover adc_base, at 1 we recover adc_max — both negligible
    # for valid taxels in practice).
    x_raw = x_normed * scale + float(adc_base)
    base_raw = base_norm * scale + float(adc_base)

    if polarity == "positive":
        denom = (float(adc_max) - base_raw).clamp(min=1.0)
        out = ((x_raw - base_raw) / denom).clamp_(0.0, 1.0)
    elif polarity == "negative":
        denom = (base_raw - float(adc_min)).clamp(min=1.0)
        out = ((base_raw - x_raw) / denom).clamp_(0.0, 1.0)
    else:
        raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")

    if valid_mask is not None:
        m = valid_mask.to(device=x_normed.device)
        out[..., ~m] = 0.0
    return out


def normalize_polarity_native(
    x: torch.Tensor,
    base: torch.Tensor,
    polarity: str = "positive",
    tactile_max: float = 1.0,
    tactile_min: float = 0.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-taxel polarity normalization, operating in the source's NATIVE space.

    Unlike `apply_baseline` (which reconstructs raw ADC from a (raw-300)/1800 [0,1]
    space via fixed adc_base/adc_max), this works DIRECTLY on whatever native space
    x/base already live in:
        - dex stores raw_int/2035 floats  → native [0,1], tactile_max=1.0
        - human stores uint16 ADC         → native [0,2100], tactile_min=0
    The polarity formula is scale-invariant, so as long as x, base, tactile_max,
    tactile_min share ONE space the output signal is identical regardless of scale.

        positive (contact → value up):    clip((x - base) / (tactile_max - base), 0, 1)
        negative (contact → value down):  clip((base - x) / (base - tactile_min), 0, 1)

    Each taxel uses its OWN denominator (its physical dynamic range from baseline to
    the ceiling/floor), matching apply_baseline's per-taxel semantics — but with no
    deadzone and no raw reconstruction.

    Args:
        x: [..., D] tactile values in the source's native space.
        base: [..., D] per-taxel baseline in the SAME native space.
        polarity: "positive" or "negative".
        tactile_max: scalar full-scale in native space (used by positive denominator).
        tactile_min: scalar floor in native space (used by negative denominator; default 0).
        valid_mask: optional [D] bool tensor; invalid taxels → 0.

    Returns:
        signal in [0, 1], invalid taxels zeroed.
    """
    base = base.to(device=x.device, dtype=x.dtype)
    if polarity == "positive":
        denom = (float(tactile_max) - base).clamp(min=1e-6)
        out = ((x - base) / denom).clamp_(0.0, 1.0)
    elif polarity == "negative":
        denom = (base - float(tactile_min)).clamp(min=1e-6)
        out = ((base - x) / denom).clamp_(0.0, 1.0)
    else:
        raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")
    if valid_mask is not None:
        m = valid_mask.to(device=x.device)
        out[..., ~m] = 0.0
    return out
