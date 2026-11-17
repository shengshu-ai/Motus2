"""Metadata helpers for open varlen packing."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .types import VarlenRopeMeta, VarlenStreamMeta


def as_int_list(values: Sequence[int] | torch.Tensor | None, *, default_len: int | None = None, default: int = 0) -> list[int]:
    if values is None:
        if default_len is None:
            return []
        return [int(default)] * int(default_len)
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().view(-1).tolist()]
    return [int(v) for v in values]


def cumulative_lengths(lengths: Sequence[int], *, device=None, dtype=torch.int32) -> torch.Tensor:
    vals = torch.tensor([0, *[int(v) for v in lengths]], device=device, dtype=dtype)
    return torch.cumsum(vals, dim=0)


def make_stream_meta(
    name: str,
    lengths: Sequence[int],
    *,
    padded_lengths: Sequence[int] | None = None,
    cp_size: int = 1,
    device=None,
) -> VarlenStreamMeta:
    lengths_l = [int(v) for v in lengths]
    if not lengths_l:
        lengths_l = [0]
    lengths_t = torch.tensor(lengths_l, device=device, dtype=torch.int32)
    cu = cumulative_lengths(lengths_l, device=device)
    max_len = int(max(lengths_l)) if lengths_l else 0
    total = int(cu[-1].item())

    padded_t = cu_padded = split_t = cu_split = None
    max_padded = total_padded = max_split = None
    if padded_lengths is not None:
        padded_l = [int(v) for v in padded_lengths]
        padded_t = torch.tensor(padded_l, device=device, dtype=torch.int32)
        cu_padded = cumulative_lengths(padded_l, device=device)
        max_padded = int(max(padded_l)) if padded_l else 0
        total_padded = int(cu_padded[-1].item())
        cp = max(1, int(cp_size))
        split_l = [v // cp for v in padded_l]
        split_t = torch.tensor(split_l, device=device, dtype=torch.int32)
        cu_split = cumulative_lengths(split_l, device=device)
        max_split = int(max(split_l)) if split_l else 0

    return VarlenStreamMeta(
        name=name,
        lengths=lengths_t,
        cu=cu,
        max=max_len,
        total=total,
        padded_lengths=padded_t,
        cu_padded=cu_padded,
        max_padded=max_padded,
        total_padded=total_padded,
        split_lengths_by_cp=split_t,
        cu_split_by_cp=cu_split,
        max_split_by_cp=max_split,
    )


def make_rope_meta(
    lengths: Sequence[int],
    *,
    f_offsets: Sequence[int] | torch.Tensor | None = None,
    h_offsets: Sequence[int] | torch.Tensor | None = None,
    w_offsets: Sequence[int] | torch.Tensor | None = None,
    frames: Sequence[int] | None = None,
    heights: Sequence[int] | None = None,
    widths: Sequence[int] | None = None,
    device=None,
) -> VarlenRopeMeta:
    lengths_l = [int(v) for v in lengths] or [0]

    def _tensor_or_none(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.int32)
        return torch.tensor([int(v) for v in x], device=device, dtype=torch.int32)

    frames_l = [int(v) for v in frames] if frames is not None else None
    heights_l = [int(v) for v in heights] if heights is not None else None
    widths_l = [int(v) for v in widths] if widths is not None else None
    return VarlenRopeMeta(
        cu_seqlens=cumulative_lengths(lengths_l, device=device),
        max_seqlen=int(max(lengths_l)) if lengths_l else 0,
        f_offsets=_tensor_or_none(f_offsets),
        h_offsets=_tensor_or_none(h_offsets),
        w_offsets=_tensor_or_none(w_offsets),
        cu_frames=cumulative_lengths(frames_l, device=device) if frames_l is not None else None,
        max_frames=max(frames_l) if frames_l else None,
        cu_heights=cumulative_lengths(heights_l, device=device) if heights_l is not None else None,
        max_height=max(heights_l) if heights_l else None,
        cu_widths=cumulative_lengths(widths_l, device=device) if widths_l is not None else None,
        max_width=max(widths_l) if widths_l else None,
    )
