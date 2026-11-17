"""Open varlen batch preprocessor for Motus memory training.

This is the project-owned analogue of the closed stack's closed varlen preprocessors: it
turns per-sample lengths into flat stream metadata, block sizes, block indices,
valid-frame metadata, caption lengths, and RoPE offsets. Tensor concatenation is
still performed by `packing.py` because Motus tokenization happens inside the
model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .block_mask import build_packed_memory_window_block_mask
from .metadata import as_int_list
from .types import VarlenBatchMeta


def build_memory_varlen_batch(
    num_latent_frames_list: Sequence[int],
    *,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    action_tokens_per_frame: int,
    chunk_size: int,
    anchor_frames: int,
    recent_frames: int,
    use_state: bool = True,
    state_tokens_per_frame: int = 1,
    value_tokens_per_frame: int = 0,
    attention_mode: str = "fdm",
    device: str | torch.device = "cuda",
    obs_latents: int | None = None,
    separate_anchor: bool = False,
    caption_lengths: Sequence[int] | torch.Tensor | None = None,
    caption_padded_lengths: Sequence[int] | torch.Tensor | None = None,
    valid_lens: Sequence[int] | torch.Tensor | None = None,
    latent_starts: Sequence[int] | torch.Tensor | None = None,
    latent_hw: tuple[int, int] | None = None,
    cp_size: int = 1,
    retained: Sequence[Sequence[int] | None] | None = None,
) -> VarlenBatchMeta:
    """Build full open varlen metadata for a packed Motus batch."""

    B = len(num_latent_frames_list)
    caps = as_int_list(caption_lengths, default_len=B, default=0)
    cap_padded = as_int_list(caption_padded_lengths, default_len=B, default=0)
    # None -> defer to the mask builder, which defaults valid_len to the full
    # window W (not 0). A 0 default here would zero out every loss mask.
    valid = None if valid_lens is None else as_int_list(valid_lens, default_len=B, default=0)
    starts = as_int_list(latent_starts, default_len=B, default=0)
    return build_packed_memory_window_block_mask(
        num_latent_frames_list,
        video_tokens_per_frame=video_tokens_per_frame,
        gist_tokens_per_frame=gist_tokens_per_frame,
        action_tokens_per_frame=action_tokens_per_frame,
        chunk_size=chunk_size,
        anchor_frames=anchor_frames,
        recent_frames=recent_frames,
        use_state=use_state,
        state_tokens_per_frame=state_tokens_per_frame,
        value_tokens_per_frame=value_tokens_per_frame,
        attention_mode=attention_mode,
        device=device,
        obs_latents=obs_latents,
        separate_anchor=separate_anchor,
        caption_lengths=caps,
        caption_padded_lengths=cap_padded,
        valid_lens=valid,
        latent_starts=starts,
        latent_hw=latent_hw,
        cp_size=cp_size,
        retained=retained,
    )
