"""Orthogonal action representation + nested normalization config.

Dimensions (independent):
  1. Hand qpos representation: abs (qpos20_rel6d) vs relative (relqpos20_rel6d)
  2. Wrist eef representation: relative xyz+rot6d (encoded in action_mode)
  3. Hand qpos normalization: normalization.hand
  4. Wrist relative-xyz normalization: normalization.wrist_xyz
     (rot6d is not normalized)

Preferred YAML surface (under ``model``)::

    action_mode: qpos20_rel6d
    normalization:
      hand: min_max        # min_max | mean_std | quantile (q01q99)
      wrist_xyz: min_max   # min_max | mean_std | quantile

Optional shorthand profiles::

    action_mode: qpos20_rel6d_mean_std
    # == layout qpos20_rel6d + hand/wrist_xyz both mean_std
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


_NORM_ALIASES = {
    "min_max": "min_max",
    "minmax": "min_max",
    "mean_std": "mean_std",
    "mean-std": "mean_std",
    "quantile": "quantile",
    "q01q99": "quantile",
    "q01_q99": "quantile",
}

# profile -> (base_layout, hand_norm, wrist_xyz_norm)
_ACTION_MODE_PROFILES = {
    "qpos20_rel6d_mean_std": ("qpos20_rel6d", "mean_std", "mean_std"),
    "relqpos20_rel6d_mean_std": ("relqpos20_rel6d", "mean_std", "mean_std"),
    "qpos20_rel6d_q01q99": ("qpos20_rel6d", "quantile", "quantile"),
    "relqpos20_rel6d_q01q99": ("relqpos20_rel6d", "quantile", "quantile"),
    "qpos20_rel6d_quantile": ("qpos20_rel6d", "quantile", "quantile"),
    "relqpos20_rel6d_quantile": ("relqpos20_rel6d", "quantile", "quantile"),
}

_BASE_QPOS20_MODES = {"qpos20_rel6d", "relqpos20_rel6d"}


@dataclass(frozen=True)
class ResolvedActionMode:
    """Resolved orthogonal knobs for model + dataset construction."""

    profile: str
    action_mode: str
    hand_normalization: Optional[str]
    wrist_xyz_normalization: str

    @property
    def qpos_normalization(self) -> Optional[str]:
        """Alias used by dataset loaders (hand qpos)."""
        return self.hand_normalization


def _normalize_norm_name(value: Optional[str], *, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    key = str(value).strip().lower()
    if not key:
        return default
    if key not in _NORM_ALIASES:
        raise ValueError(
            f"Unsupported normalization={value!r}; expected one of "
            f"{sorted(set(_NORM_ALIASES.values()))} (aliases: q01q99→quantile)"
        )
    return _NORM_ALIASES[key]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    # OmegaConf DictConfig supports .get / keys
    try:
        return {k: value.get(k) for k in list(value.keys())}
    except Exception:
        return None


def _pick_norm(*values: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Return the first non-None normalized name; error if conflicting values."""
    chosen: Optional[str] = None
    raw_chosen: Optional[str] = None
    for raw in values:
        if raw is None:
            continue
        normed = _normalize_norm_name(raw)
        if chosen is None:
            chosen = normed
            raw_chosen = raw
            continue
        if normed != chosen:
            raise ValueError(
                f"Conflicting normalization values: {raw_chosen!r} vs {raw!r}"
            )
    return chosen if chosen is not None else default


def resolve_normalization_block(
    model_cfg: Any = None,
    dataset_cfg: Any = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (hand, wrist_xyz) from model.normalization.

    Returns ``(hand_or_None, wrist_xyz_or_None)`` where None means "not set"
    (caller may apply defaults / profile fills).
    """
    norm_block = _as_mapping(_cfg_get(model_cfg, "normalization", None)) or {}
    for owner, keys in ((model_cfg, ("qpos_normalization", "wrist_xyz_normalization", "wrist_relative_normalization")),
                        (dataset_cfg, ("qpos_normalization",))):
        for key in keys:
            if _cfg_get(owner, key, None) is not None:
                raise ValueError(f"Use model.normalization.hand / wrist_xyz instead of {key}")
    return _pick_norm(norm_block.get("hand")), _pick_norm(norm_block.get("wrist_xyz"))


def resolve_action_mode(
    action_mode: str,
    *,
    hand_normalization: Optional[str] = None,
    wrist_xyz_normalization: Optional[str] = None,
) -> ResolvedActionMode:
    """Resolve layout + optional shorthand profile into orthogonal knobs."""
    profile = str(action_mode or "joint")
    hand_from_cfg = _pick_norm(hand_normalization)
    wrist_from_cfg = _pick_norm(wrist_xyz_normalization)

    if profile in _ACTION_MODE_PROFILES:
        base, hand, wrist = _ACTION_MODE_PROFILES[profile]
        if hand_from_cfg is not None and hand_from_cfg != hand:
            raise ValueError(
                f"action_mode={profile!r} implies normalization.hand={hand!r}, "
                f"got {hand_from_cfg!r}"
            )
        if wrist_from_cfg is not None and wrist_from_cfg != wrist:
            raise ValueError(
                f"action_mode={profile!r} implies normalization.wrist_xyz={wrist!r}, "
                f"got {wrist_from_cfg!r}"
            )
        return ResolvedActionMode(
            profile=profile,
            action_mode=base,
            hand_normalization=hand,
            wrist_xyz_normalization=wrist,
        )

    return ResolvedActionMode(
        profile=profile,
        action_mode=profile,
        hand_normalization=hand_from_cfg or ("min_max" if "qpos" in profile else None),
        wrist_xyz_normalization=wrist_from_cfg or "min_max",
    )


def resolve_action_mode_from_config(
    model_cfg: Any,
    dataset_cfg: Any = None,
) -> ResolvedActionMode:
    """Read nested ``model.normalization`` from config."""
    hand, wrist = resolve_normalization_block(model_cfg, dataset_cfg)
    return resolve_action_mode(
        _cfg_get(model_cfg, "action_mode", "joint"),
        hand_normalization=hand,
        wrist_xyz_normalization=wrist,
    )


def is_qpos20_rel6d_family(action_mode: str) -> bool:
    mode = str(action_mode or "")
    if mode in _BASE_QPOS20_MODES or mode in _ACTION_MODE_PROFILES:
        return True
    return resolve_action_mode(mode).action_mode in _BASE_QPOS20_MODES


def known_action_mode_profiles() -> tuple[str, ...]:
    return tuple(sorted(_ACTION_MODE_PROFILES))
