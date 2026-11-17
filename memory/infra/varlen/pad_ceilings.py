"""Monotone global pad ceilings for factored varlen attention budgets.

Tracks max observed lengths (including planned-KV duplication) across steps.
Call ``grow_pad_ceilings`` outside compiled regions; ``reset_pad_ceilings``
between epochs or when re-profiling.
"""

from __future__ import annotations

_MAX: dict[str, int] = {
    "clean_q": 0,
    "gist_q": 0,
    "action_q": 0,
    "noisy_q": 0,
    "planned_kv": 0,
}


def grow_pad_ceilings(lengths: dict[str, int]) -> dict[str, int]:
    """Raise ceilings for known keys; ignore unknown keys. Returns a copy."""
    for k, v in lengths.items():
        if k in _MAX:
            _MAX[k] = max(_MAX[k], int(v))
    return dict(_MAX)


def reset_pad_ceilings() -> None:
    for k in _MAX:
        _MAX[k] = 0


def get_pad_ceilings() -> dict[str, int]:
    return dict(_MAX)
