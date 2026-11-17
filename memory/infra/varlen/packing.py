"""PyTorch-native packing helpers for variable-length streams.

Packing and splitting use ordinary differentiable PyTorch operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import torch

from .types import PackedStreams


def _normalize_lengths(lengths: Iterable[int]) -> list[int]:
    out = [int(v) for v in lengths]
    if any(v < 0 for v in out):
        raise ValueError(f"lengths must be non-negative, got {out}")
    return out


def pad_to_length(
    tensor: torch.Tensor,
    target_length: int,
    *,
    dim: int = 0,
    value: float = 0.0,
) -> torch.Tensor:
    """Pad `tensor` along `dim` to `target_length` without changing existing data."""

    target_length = int(target_length)
    dim = dim % tensor.ndim
    current = int(tensor.shape[dim])
    if current > target_length:
        raise ValueError(f"cannot pad length {current} down to {target_length}")
    if current == target_length:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current
    pad = tensor.new_full(pad_shape, value)
    return torch.cat([tensor, pad], dim=dim)


def pack_streams(
    segments: Sequence[torch.Tensor],
    *,
    pad_to: int | None = None,
    output_batch_dim: bool = True,
    pad_value: float = 0.0,
) -> PackedStreams:
    """Concatenate variable-length tensors into one packed row.

    Each input tensor is expected to have sequence length in dimension 0, e.g.
    `[L_i, D]`. The returned tensor is `[1, sum(L_i), D]` by default so it can be
    fed directly to the existing single-row packed attention path.
    """

    if not segments:
        raise ValueError("pack_streams requires at least one segment")
    lengths = _normalize_lengths(seg.shape[0] for seg in segments)
    trailing = segments[0].shape[1:]
    for idx, seg in enumerate(segments):
        if seg.shape[1:] != trailing:
            raise ValueError(
                f"segment {idx} trailing shape {tuple(seg.shape[1:])} does not match {tuple(trailing)}"
            )
    total = sum(lengths)
    padded = int(pad_to) if pad_to is not None else total
    if padded < total:
        raise ValueError(f"pad_to {padded} is smaller than packed length {total}")
    packed = torch.cat(list(segments), dim=0)
    packed = pad_to_length(packed, padded, dim=0, value=pad_value)
    if output_batch_dim:
        packed = packed.unsqueeze(0)
    return PackedStreams(tensor=packed, lengths=lengths, padded_length=padded, original_dim=segments[0].ndim)


def unpack_streams(
    packed: torch.Tensor,
    lengths: Sequence[int],
    *,
    has_batch_dim: bool = True,
) -> list[torch.Tensor]:
    """Split a packed tensor back into its variable-length pieces."""

    lengths = _normalize_lengths(lengths)
    row = packed.squeeze(0) if has_batch_dim else packed
    total = sum(lengths)
    if row.shape[0] < total:
        raise ValueError(f"packed length {row.shape[0]} is smaller than requested split total {total}")
    pieces = []
    offset = 0
    for length in lengths:
        pieces.append(row[offset:offset + length])
        offset += length
    return pieces


def cumulative_lengths(lengths: Sequence[int], *, device=None, dtype=torch.int32) -> torch.Tensor:
    """Return `[0, cumsum(lengths)]`, matching common varlen metadata."""

    lengths = _normalize_lengths(lengths)
    vals = torch.tensor([0, *lengths], device=device, dtype=dtype)
    return torch.cumsum(vals, dim=0)


# ---------------------------------------------------------------------------
# Metadata-driven multi-stream packing
#
# The closed stack interleaves *named streams* (v_clean / v_noisy / a_noisy /
# caption / gist) and unites them with a custom `.so` kernel. The open analogue
# below uses the VarlenBatchMeta stream lengths as the single source of truth:
# every per-segment stream tensor is validated against `seg.stream_lengths`
# before it is concatenated, so the metadata genuinely *drives* packing instead
# of being descriptive only.
# ---------------------------------------------------------------------------

DEFAULT_STREAM_ORDER = ("anchor", "clean_video", "gist", "chunk_ar")


def _expected_stream_len(seg, name: str) -> int:
    if name == "chunk_ar":
        return (
            int(seg.stream_lengths.get("action", 0))
            + int(seg.stream_lengths.get("noisy_video", 0))
            + int(seg.stream_lengths.get("value", 0))
        )
    return int(seg.stream_lengths.get(name, 0))


def _as_seq_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def pack_named_streams(
    segment_streams: Sequence[dict],
    meta,
    *,
    stream_order: Sequence[str] = DEFAULT_STREAM_ORDER,
    pad_to: int | None = None,
    pad_value: float = 0.0,
    output_batch_dim: bool = True,
) -> PackedStreams:
    """Pack per-segment named stream tensors into one row, validated by metadata.

    `segment_streams[i]` is a mapping ``{stream_name: tensor}`` for the i-th
    segment of `meta`. Each tensor may be `[L, D]` or `[1, L, D]`. Streams are
    concatenated per segment in `stream_order`, segments are concatenated in
    order, and every length is checked against `meta.segments[i].stream_lengths`.
    """

    segments = list(meta.segments)
    if len(segment_streams) != len(segments):
        raise ValueError(f"got {len(segment_streams)} stream dicts for {len(segments)} segments")
    rows = []
    for seg, streams in zip(segments, segment_streams):
        parts = []
        for name in stream_order:
            expected = _expected_stream_len(seg, name)
            if expected == 0:
                continue
            tensor = streams.get(name)
            if tensor is None:
                raise ValueError(f"segment {seg.index} missing stream '{name}' (expected len {expected})")
            tensor = _as_seq_first(tensor)
            if int(tensor.shape[0]) != expected:
                raise ValueError(
                    f"segment {seg.index} stream '{name}' length {int(tensor.shape[0])} != metadata {expected}"
                )
            parts.append(tensor)
        if not parts:
            raise ValueError(f"segment {seg.index} produced no stream tokens")
        row = torch.cat(parts, dim=0)
        if int(row.shape[0]) != int(seg.token_length):
            raise ValueError(
                f"segment {seg.index} packed length {int(row.shape[0])} != metadata token_length {int(seg.token_length)}"
            )
        rows.append(row)
    target = int(pad_to) if pad_to is not None else int(getattr(meta, "padded_total", sum(int(s.token_length) for s in segments)))
    return pack_streams(rows, pad_to=target, output_batch_dim=output_batch_dim, pad_value=pad_value)


def unpack_named_streams(
    packed: torch.Tensor,
    meta,
    *,
    stream_order: Sequence[str] = DEFAULT_STREAM_ORDER,
    has_batch_dim: bool = True,
) -> list[dict]:
    """Inverse of `pack_named_streams`: slice the packed row back per segment.

    Returns a list aligned with `meta.segments`; each item maps stream name to a
    `[L, D]` (or `[1, L, D]` when `has_batch_dim`) slice. Slicing offsets come
    entirely from `meta`, so this is the metadata-driven `vars_split` analogue.
    """

    row = packed.squeeze(0) if has_batch_dim else packed
    out = []
    for seg in meta.segments:
        cursor = int(seg.offset)
        seg_out = {}
        for name in stream_order:
            length = _expected_stream_len(seg, name)
            if length == 0:
                continue
            piece = row[cursor:cursor + length]
            seg_out[name] = piece.unsqueeze(0) if has_batch_dim else piece
            cursor += length
        out.append(seg_out)
    return out
