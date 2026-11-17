"""Typed metadata for packed variable-length training.

Includes cumulative lengths, maximum and total lengths, block indices,
block sizes, and rotary-position offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class VarlenStreamMeta:
    """Per-stream cumulative length metadata."""

    name: str
    lengths: torch.Tensor
    cu: torch.Tensor
    max: int
    total: int
    padded_lengths: torch.Tensor | None = None
    cu_padded: torch.Tensor | None = None
    max_padded: int | None = None
    total_padded: int | None = None
    split_lengths_by_cp: torch.Tensor | None = None
    cu_split_by_cp: torch.Tensor | None = None
    max_split_by_cp: int | None = None


@dataclass(frozen=True)
class VarlenRopeMeta:
    """RoPE metadata for a packed stream."""

    cu_seqlens: torch.Tensor
    max_seqlen: int
    f_offsets: torch.Tensor | None = None
    h_offsets: torch.Tensor | None = None
    w_offsets: torch.Tensor | None = None
    cu_frames: torch.Tensor | None = None
    max_frames: int | None = None
    cu_heights: torch.Tensor | None = None
    max_height: int | None = None
    cu_widths: torch.Tensor | None = None
    max_width: int | None = None


@dataclass(frozen=True)
class VarlenSegment:
    """One sample/trajectory inside a packed attention row."""

    index: int
    token_length: int
    offset: int
    stream_lengths: Mapping[str, int] = field(default_factory=dict)
    block_sizes: Sequence[int] = field(default_factory=tuple)
    block_offset: int = 0
    n_pred_frames: int = 0
    valid_pred_frames: int = 0
    latent_start: int = 0
    # RoPE start frame for the clean-history and noisy-prediction video streams.
    # These are the source of truth consumed by the packed RoPE assembler instead
    # of recomputing start frames inside the model.
    clean_rope_start: int = 0
    noisy_rope_start: int = 0
    # Per-chunk latent counts for the noisy/action streams of this segment.
    block_latent_counts: Sequence[int] = field(default_factory=tuple)

    @property
    def end(self) -> int:
        return self.offset + self.token_length

    def stream_span(self, name: str) -> tuple[int, int]:
        """Absolute [start, end) token span of a named stream inside the packed row.

        Stream order matches the canonical Motus packed layout
        ``anchor | clean_video | gist | (action+noisy per chunk)`` so the noisy
        and action streams resolve to the combined ``chunk_ar`` region.
        """

        order = ("anchor", "clean_video", "gist", "chunk_ar")
        spans = {}
        cursor = self.offset
        chunk_ar_len = int(self.stream_lengths.get("action", 0)) + int(self.stream_lengths.get("noisy_video", 0)) + int(self.stream_lengths.get("value", 0))
        sizes = {
            "anchor": int(self.stream_lengths.get("anchor", 0)),
            "clean_video": int(self.stream_lengths.get("clean_video", 0)),
            "gist": int(self.stream_lengths.get("gist", 0)),
            "chunk_ar": chunk_ar_len,
        }
        for key in order:
            spans[key] = (cursor, cursor + sizes[key])
            cursor += sizes[key]
        if name in spans:
            return spans[name]
        raise KeyError(f"unknown stream span '{name}'")


@dataclass(frozen=True)
class VarlenBatchMeta:
    """Complete metadata for one packed-row varlen batch."""

    segments: Sequence[VarlenSegment]
    total_tokens: int
    padded_total: int
    block_sizes: Sequence[int] = field(default_factory=tuple)
    logical_block_mask: torch.Tensor | None = None
    flex_block_mask: object | None = None
    q_block_sizes: Sequence[int] = field(default_factory=tuple)
    kv_block_sizes: Sequence[int] = field(default_factory=tuple)
    q_block_indices: torch.Tensor | None = None
    kv_block_indices: torch.Tensor | None = None
    streams: Mapping[str, VarlenStreamMeta] = field(default_factory=dict)
    rope: Mapping[str, VarlenRopeMeta] = field(default_factory=dict)
    caption_lengths: torch.Tensor | None = None
    valid_lens: torch.Tensor | None = None
    latent_starts: torch.Tensor | None = None
    # Scalar layout parameters (video/gist/action/state tokens-per-frame, chunk
    # size, obs latents, ...). These let downstream consumers (RoPE assembler,
    # stream packer) reconstruct stream geometry purely from metadata.
    params: Mapping[str, int] = field(default_factory=dict)
    # Per-segment factored attention plans / sample-major layouts (Task 7).
    # Built at preprocess time from the same N=W-obs geometry as the block mask.
    # Empty when factored path is unsupported (e.g. separate_anchor / value tokens).
    factored_plans: Sequence[Any] = field(default_factory=tuple)
    factored_layouts: Sequence[Any] = field(default_factory=tuple)

    @property
    def block_mask(self):
        """Compatibility alias used by older packed-path call sites."""

        return self.flex_block_mask

    @property
    def segment_token_lengths(self) -> list[int]:
        return [int(seg.token_length) for seg in self.segments]

    @property
    def valid_pred_frames(self) -> list[int]:
        return [int(seg.valid_pred_frames) for seg in self.segments]

    @property
    def pad_tokens(self) -> int:
        return int(self.padded_total) - int(self.total_tokens)

    def offsets(self) -> list[int]:
        return [int(seg.offset) for seg in self.segments]


# Backward-compatible name used by the first refactor pass.
VarlenLayout = VarlenBatchMeta


@dataclass(frozen=True)
class PackedStreams:
    """Result of concatenating variable-length stream tensors."""

    tensor: torch.Tensor
    lengths: Sequence[int]
    padded_length: int
    original_dim: int

    @property
    def total_length(self) -> int:
        return int(sum(int(v) for v in self.lengths))

    @property
    def pad_length(self) -> int:
        return int(self.padded_length) - self.total_length
