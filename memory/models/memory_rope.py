"""Model-free layout helpers for hybrid-memory chunk bookkeeping.

MemoryWAM / varlen uses absolute episode temporal
positions from ``models/motus.py``; this module only keeps small shared utilities.
"""

from typing import List, Optional, Tuple

import torch


def pad_window_to_chunk(window_length: int, obs_latents: int, chunk_size: int) -> Tuple[int, int]:
    """Round the predict suffix ``N = W - obs`` up to whole chunks: ``(W_padded, N_padded)``.

    Deployment always predicts ``chunk_size`` latents at once, so a trajectory whose
    last real latent would start a lonely chunk is trained as a full chunk whose
    second latent is loss-masked padding. Dropping that latent instead loses the
    final actions of every such trajectory, which is where the return to the start
    pose happens.
    """
    cs = max(1, int(chunk_size))
    obs = max(1, int(obs_latents))
    N = max(0, int(window_length) - obs)
    N = -(-N // cs) * cs
    return N + obs, N


def repeat_last(x: Optional[torch.Tensor], dim: int, count: int) -> Optional[torch.Tensor]:
    """Extend ``x`` along ``dim`` by ``count`` copies of its last slice."""
    if x is None or count <= 0:
        return x
    tail = x.narrow(dim, x.shape[dim] - 1, 1)
    sizes = [-1] * x.dim()
    sizes[dim] = count
    return torch.cat([x, tail.expand(*sizes)], dim=dim)


def extend_valid_mask(mask: torch.Tensor, count: int, dim: int = -1) -> torch.Tensor:
    """Append ``count`` invalid positions to a boolean mask along ``dim``."""
    if count <= 0:
        return mask
    pad_shape = list(mask.shape)
    pad_shape[dim] = count
    return torch.cat([mask.bool(), torch.zeros(pad_shape, dtype=torch.bool, device=mask.device)], dim=dim)


def align_window_to_chunk(window_length: int, obs_latents: int, chunk_size: int) -> Tuple[int, int]:
    """Drop a short predict tail so ``N = W - obs`` is a multiple of ``chunk_size``.

    Predicts whole chunks only (e.g. cs=2): never condition on a missing adjacent
    full to predict a lonely last latent. Returns ``(W_aligned, N_aligned)``.
    """
    cs = max(1, int(chunk_size))
    obs = max(1, int(obs_latents))
    W = int(window_length)
    N = max(0, W - obs)
    N = (N // cs) * cs
    return N + obs, N


def chunk_latent_counts(num_frames: int, chunk_size: int) -> List[int]:
    """Partition ``num_frames`` predicted latents into chunks of ``chunk_size``."""
    cs = max(1, int(chunk_size))
    counts = []
    off = 0
    while off < num_frames:
        counts.append(min(cs, num_frames - off))
        off += cs
    return counts


def memory_chunk_fid(phys: int, obs_latents: int, chunk_size: int) -> int:
    """Causal chunk id for a latent at physical index ``phys``.

    ``obs_latents`` latents (phys ``0..obs-1``) are pure observation (chunk id 0);
    every following chunk of ``chunk_size`` latents gets the next id.
    """
    cs = max(1, int(chunk_size))
    obs = max(1, int(obs_latents))
    if phys < obs:
        return 0
    return 1 + (phys - obs) // cs
