"""Longest Processing Time (LPT) bin-packing for cross-rank load balance."""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

from .types import LPTAssignment, RebalancePlan, SampleCost


def build_lpt_plan(
    local_costs: list[SampleCost],
    *,
    dp_rank: int,
    dp_size: int,
    gathered_costs: list[list[SampleCost]] | None = None,
) -> RebalancePlan:
    """Build a deterministic LPT rebalance plan.

    When ``gathered_costs`` is provided (one list per DP rank), no distributed
    collect is needed — useful for unit tests. Otherwise callers should gather
    local costs with ``all_gather_object`` before invoking this function.
    """
    if dp_size <= 1:
        return RebalancePlan(
            kept_indices=list(range(len(local_costs))),
            send_plan={},
            recv_plan={},
        )

    if gathered_costs is None:
        gathered_costs = [local_costs]

    all_tasks: list[LPTAssignment] = []
    for src_rank, rank_costs in enumerate(gathered_costs):
        for item in rank_costs:
            all_tasks.append(
                LPTAssignment(
                    src_rank=src_rank,
                    sample_id=item.sample_id,
                    shapes=item.shapes,
                    cost=item.cost,
                    tgt_rank=-1,
                    meta=dict(item.meta),
                )
            )

    if len(all_tasks) <= 1:
        kept = [t.sample_id for t in all_tasks if t.src_rank == dp_rank]
        return RebalancePlan(kept_indices=kept, send_plan={}, recv_plan={})

    all_tasks.sort(key=lambda t: t.cost, reverse=True)

    rank_loads: list[tuple[int, int]] = [(0, r) for r in range(dp_size)]
    heapq.heapify(rank_loads)
    for task in all_tasks:
        min_load, target_rank = heapq.heappop(rank_loads)
        task.tgt_rank = target_rank
        heapq.heappush(rank_loads, (min_load + task.cost, target_rank))

    kept_indices: list[int] = []
    send_plan: dict[int, list[int]] = defaultdict(list)
    recv_plan: dict[int, list[tuple[int, tuple[tuple[int, ...], ...], dict[str, Any]]]] = defaultdict(list)

    for task in all_tasks:
        src_rank, dst_rank = task.src_rank, task.tgt_rank
        if src_rank == dp_rank and dst_rank == dp_rank:
            kept_indices.append(task.sample_id)
        elif src_rank == dp_rank and dst_rank != dp_rank:
            send_plan[dst_rank].append(task.sample_id)
        elif src_rank != dp_rank and dst_rank == dp_rank:
            recv_plan[src_rank].append((task.sample_id, task.shapes, task.meta))

    return RebalancePlan(
        kept_indices=sorted(kept_indices),
        send_plan=dict(send_plan),
        recv_plan=dict(recv_plan),
    )
