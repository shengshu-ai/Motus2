"""Per-slot window-start assignment shared by the sampler and the dataset.

The batch sampler emits ``base_index * ORDINAL_STRIDE + visit_ordinal`` so a
dataloader worker can resolve which latent window to train on without any
shared mutable state.  That keeps the assignment identical across the 16
worker processes, immune to worker restarts, and reproducible after a resume.

Consuming episodes one *row* at a time biases the mixture: a source with short
episodes is visited far more often per episode than a long-episode source, so
the same handful of clips is replayed while long clips are barely touched.  The
ordinal fixes this by giving every episode a visit budget proportional to how
many distinct window starts it can offer, and by spending those visits on
windows that have not been seen yet:

1. starts that tile the episode without any overlap (``0, W, 2W, ...``),
2. then the remaining starts in a keyed permutation, so no start repeats while
   unused ones are still available,
3. only after every start has been used does the cycle advance and reuse them.

Because one epoch allocates fewer slots per episode than the episode has
starts, step 3 is not reached within a normal epoch; it exists so the mapping
stays total for shorter sequences or larger budgets.
"""

import hashlib
from functools import lru_cache
from typing import List

# Packing stride for ``base_index * ORDINAL_STRIDE + visit_ordinal``.  Must
# exceed the largest number of visits any single episode receives inside one
# epoch, which is bounded by its window-start count (<= max_total_latent_frames).
ORDINAL_STRIDE = 1 << 24

_MASK64 = (1 << 64) - 1
_PERM_CACHE_SIZE = 8192


def pack_index(base_index: int, ordinal: int) -> int:
    """Combine an episode index and its visit ordinal into one dataset key."""
    return int(base_index) * ORDINAL_STRIDE + int(ordinal)


def unpack_index(packed: int):
    """Inverse of :func:`pack_index`; returns ``(base_index, ordinal)``."""
    return divmod(int(packed), ORDINAL_STRIDE)


def window_start_count(total_latents, window_size) -> int:
    """Number of distinct window starts an episode of this length offers."""
    total = int(total_latents)
    window = int(window_size)
    if total <= 0 or window <= 0:
        return 1
    return max(1, total - window + 1)


def estimate_window_start_count(total_latents, window_size) -> int:
    """Same as :func:`window_start_count`, but tolerates a missing length.

    Index rows for some sources omit ``total_latent_frames``; those fall back to
    one start so they are never dropped from the sampling plan.
    """
    try:
        total = int(total_latents)
    except (TypeError, ValueError):
        return 1
    if total <= 0:
        return 1
    return window_start_count(total, window_size)


def _seeded_state(key: str) -> int:
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _next_state(state: int) -> int:
    return (state * 6364136223846793005 + 1442695040888963407) & _MASK64


def _permute(count: int, key: str) -> List[int]:
    order = list(range(count))
    state = _seeded_state(key)
    for i in range(count - 1, 0, -1):
        state = _next_state(state)
        j = (state >> 33) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


@lru_cache(maxsize=_PERM_CACHE_SIZE)
def _tiling_starts(n_starts: int, window_size: int) -> tuple:
    """Starts that tile the episode with minimal overlap (tier 1).

    Multiples of the window size walk the episode without any overlap, but they
    rarely land exactly on the end: a 20-latent episode with 8-latent windows
    tiles [0,8) and [8,16) and leaves [16,20) unseen.  Appending the last legal
    start covers that tail, at the cost of overlapping the previous window by
    less than one window — strictly better than never showing the tail at all.
    """
    window = max(1, int(window_size))
    max_start = max(0, int(n_starts) - 1)
    starts = list(range(0, int(n_starts), window))
    if max_start not in starts:
        starts.append(max_start)
    return tuple(starts)


@lru_cache(maxsize=_PERM_CACHE_SIZE)
def _tier2_order(n_starts: int, window_size: int, episode_key: str, cycle: int) -> tuple:
    """Every start outside tier 1, in a keyed order."""
    tiling = set(_tiling_starts(n_starts, window_size))
    return tuple(
        s for s in _permute(n_starts, f"{episode_key}|t2|{cycle}") if s not in tiling
    )


@lru_cache(maxsize=_PERM_CACHE_SIZE)
def _full_order(n_starts: int, episode_key: str, cycle: int) -> tuple:
    return tuple(_permute(n_starts, f"{episode_key}|all|{cycle}"))


def select_window_start(
    visit_ordinal: int,
    total_latents: int,
    window_size: int,
    episode_key: str,
) -> int:
    """Map the n-th visit of an episode to a window start.

    Deterministic in ``(visit_ordinal, total_latents, window_size,
    episode_key)`` alone, so every worker computes the same answer and a
    resumed run reproduces the exact windows it would have seen.
    """
    n_starts = window_start_count(total_latents, window_size)
    if n_starts <= 1:
        return 0

    ordinal = max(0, int(visit_ordinal))
    cycle, pos = divmod(ordinal, n_starts)
    if cycle == 0:
        # Tier 1: cover the episode first, overlapping as little as possible.
        tiling = _tiling_starts(n_starts, window_size)
        if pos < len(tiling):
            return int(tiling[pos])
        # Tier 2: every start tier 1 skipped, each used once before any repeat.
        tier2 = _tier2_order(n_starts, window_size, str(episode_key), 0)
        index = pos - len(tiling)
        if index < len(tier2):
            return int(tier2[index])
        return int(tier2[-1])
    # Later cycles only run if a budget exceeds the episode's start count.
    order = _full_order(n_starts, str(episode_key), cycle)
    return int(order[pos])
