"""Shared per-latent action-chunk padding, used by both pretrain action paths.

The generic Wuji path and the HandsOn sidecar path index their action streams
differently (frame_stride-driven vs a fixed 24 commands per latent), but the
last chunk of an episode behaves the same way in both: it can run short, and is
then padded by repeating the final real command. Keeping that in one place means
the two paths cannot drift apart.
"""
from typing import Optional

import torch


def pad_or_trim_chunk(
    chunk: torch.Tensor,
    per_latent_len: int,
    fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Force `chunk` to exactly `per_latent_len` rows.

    Short chunks are padded by repeating their last row; an empty chunk uses
    `fill` (or raises). Long chunks are trimmed. The padding row is synthetic,
    but every other row is real data and all rows are trained on.
    """
    if per_latent_len < 1:
        raise ValueError(f"per_latent_len must be positive, got {per_latent_len}")
    n = chunk.shape[0]
    if n == per_latent_len:
        return chunk
    if n > per_latent_len:
        return chunk[:per_latent_len]
    if n == 0:
        if fill is None:
            raise ValueError("cannot pad an empty action chunk without a fill row")
        seed = fill.unsqueeze(0) if fill.dim() == 1 else fill[:1]
    else:
        seed = chunk[-1:]
    return torch.cat([chunk, seed.expand(per_latent_len - n, -1)], dim=0)
