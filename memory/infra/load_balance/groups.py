"""Data-parallel process group helpers for load balancing.

Motus training today uses pure data parallelism (``dp_size == world_size``).
When context / tensor parallelism is enabled later, ranks that share the same
model shard must **not** exchange samples — only the data-parallel group does.

Rank layout convention (Megatron-style, outer -> inner):
  ``global_rank = dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank``

Override sizes via env ``CP_SIZE`` / ``TP_SIZE`` (default 1) or pass explicitly
to :func:`resolve_data_parallel_context`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

_DP_CONTEXT: DataParallelContext | None = None


@dataclass(frozen=True)
class DataParallelContext:
    """Resolved DP group and rank mapping for P2P / collectives."""

    group: Any
    rank: int
    size: int
    cp_size: int
    tp_size: int

    def global_rank(self, dp_rank: int) -> int:
        """Map a DP-local rank to the global process index."""
        if self.size <= 1:
            return dp_rank
        return int(dist.get_global_rank(self.group, dp_rank))

    def is_initialized(self) -> bool:
        return dist.is_initialized() and self.size > 0


def _megatron_dp_group(cp_size: int, tp_size: int) -> tuple[Any, int, int]:
    """Build or select the DP subgroup for the current global rank."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")

    world_size = dist.get_world_size()
    parallel_width = cp_size * tp_size
    if world_size % parallel_width != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by cp_size*tp_size={parallel_width}"
        )

    dp_size = world_size // parallel_width
    global_rank = dist.get_rank()

    if parallel_width == 1:
        return dist.group.WORLD, global_rank, dp_size

    tp_rank = global_rank % tp_size
    remainder = global_rank // tp_size
    cp_rank = remainder % cp_size
    dp_rank = remainder // cp_size

    dp_group = None
    for cp in range(cp_size):
        for tp in range(tp_size):
            ranks = [
                dp * parallel_width + cp * tp_size + tp
                for dp in range(dp_size)
            ]
            group = dist.new_group(ranks)
            if cp == cp_rank and tp == tp_rank:
                dp_group = group
                local_dp_rank = dp_rank

    if dp_group is None:
        raise RuntimeError("failed to resolve data-parallel process group")

    return dp_group, local_dp_rank, dp_size


def resolve_data_parallel_context(
    *,
    accelerator: Any | None = None,
    cp_size: int | None = None,
    tp_size: int | None = None,
) -> DataParallelContext:
    """Resolve DP group, rank, and size for the current process."""
    cp = int(cp_size if cp_size is not None else os.environ.get("CP_SIZE", "1"))
    tp = int(tp_size if tp_size is not None else os.environ.get("TP_SIZE", "1"))
    if cp < 1 or tp < 1:
        raise ValueError(f"cp_size and tp_size must be >= 1, got cp={cp} tp={tp}")

    if not dist.is_initialized():
        # Single-process / pre-init fallback for unit tests.
        rank = int(getattr(accelerator, "process_index", 0) if accelerator is not None else 0)
        size = int(getattr(accelerator, "num_processes", 1) if accelerator is not None else 1)
        if cp > 1 or tp > 1:
            size = max(1, size // (cp * tp))
            rank = rank // (cp * tp)
        return DataParallelContext(
            group=None,
            rank=rank,
            size=size,
            cp_size=cp,
            tp_size=tp,
        )

    group, rank, size = _megatron_dp_group(cp, tp)
    return DataParallelContext(group=group, rank=rank, size=size, cp_size=cp, tp_size=tp)


def get_data_parallel_context() -> DataParallelContext:
    """Return cached DP context (call :func:`init_data_parallel_group` first)."""
    if _DP_CONTEXT is None:
        return resolve_data_parallel_context()
    return _DP_CONTEXT


def init_data_parallel_group(
    *,
    accelerator: Any | None = None,
    cp_size: int | None = None,
    tp_size: int | None = None,
) -> DataParallelContext:
    """Initialize and cache the DP context for this training job."""
    global _DP_CONTEXT
    _DP_CONTEXT = resolve_data_parallel_context(
        accelerator=accelerator,
        cp_size=cp_size,
        tp_size=tp_size,
    )
    return _DP_CONTEXT


def get_data_parallel_group() -> Any:
    return get_data_parallel_context().group


def get_data_parallel_rank() -> int:
    return get_data_parallel_context().rank


def get_data_parallel_world_size() -> int:
    return get_data_parallel_context().size


def reset_data_parallel_group() -> None:
    """Clear cached context (for tests)."""
    global _DP_CONTEXT
    _DP_CONTEXT = None
