"""Datatypes for cross-rank batch load balancing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleCost:
    """Lightweight cost record for one sample on one rank."""

    src_rank: int
    sample_id: int
    cost: int
    shapes: tuple[tuple[int, ...], ...]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LPTAssignment:
    """One sample assigned to a target rank by LPT."""

    src_rank: int
    sample_id: int
    shapes: tuple[tuple[int, ...], ...]
    cost: int
    tgt_rank: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RebalancePlan:
    """Per-rank send/recv plan produced by LPT bin-packing."""

    kept_indices: list[int]
    send_plan: dict[int, list[int]]
    recv_plan: dict[int, list[tuple[int, tuple[tuple[int, ...], ...], dict[str, Any]]]]
