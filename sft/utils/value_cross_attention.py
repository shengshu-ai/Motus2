"""Shared value-token cross-attention helpers."""

from __future__ import annotations

from typing import Optional

import torch


def apply_value_cross_attention(
    block,
    value_tokens: torch.Tensor,
    t5_ctx: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the residual used by non-MoT training value tokens."""
    if t5_ctx is None:
        return value_tokens
    return value_tokens + block.cross_attn(
        block.norm3(value_tokens), t5_ctx, context_lens=None
    )
