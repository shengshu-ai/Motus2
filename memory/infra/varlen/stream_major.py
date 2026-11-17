"""Stream-major packing (cross-sample per-stream interleave).

The default packer (:func:`infra.varlen.packing.pack_named_streams`) is
*sample-major*: each segment's streams are contiguous
(``anchor | clean_video | gist | chunk_ar`` for sample 0, then sample 1, ...).
Some attention backends instead want a *stream-major* layout where the same
stream from every sample is contiguous (all anchors, then all clean_video, ...),
with each stream carrying its own ``cu_seqlens``. That is the layout the closed
``vars_unite`` produces for fa4-style varlen kernels.

This module derives the reordering purely from :class:`VarlenBatchMeta` (so the
metadata stays load-bearing) and provides:

* a permutation ``perm`` (+ ``inverse_perm``) mapping sample-major <-> stream-major
  token positions over the whole padded row;
* per-stream ``cu_seqlens`` / ``max_seqlen`` / per-sample lengths (closed-stack-style);
* helpers to reorder a packed row, RoPE freqs, and a dense ``[L, L]`` mask, plus
  the inverse to restore sample-major order before decode.

Attention is permutation-equivariant: for any permutation ``P``,

    ``Attn(Q[P], K[P], V[P], M[P][:, P])[P^{-1}] == Attn(Q, K, V, M)``,

so *reorder -> attend (stream-major) -> invert* is numerically identical to the
sample-major path. ``test_stream_major_*`` asserts this against SDPA.

This is a backend-enabling capability: the Motus model stays sample-major today
(SDPA / FlexAttention do not require interleave). A backend that needs stream
interleave wires ``build_stream_major_plan`` + ``to_stream_major`` around its
attention call and ``to_sample_major`` before decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .packing import DEFAULT_STREAM_ORDER, _expected_stream_len


@dataclass(frozen=True)
class StreamMajorPlan:
    """Reordering plan + per-stream varlen metadata for a packed batch."""

    perm: torch.Tensor          # [padded_total] stream-major pos -> sample-major index
    inverse_perm: torch.Tensor  # [padded_total] sample-major idx -> stream-major pos
    stream_order: tuple[str, ...]
    stream_lengths: dict[str, list[int]]        # per stream: per-segment token length
    stream_cu_seqlens: dict[str, torch.Tensor]  # per stream: [num_seg + 1] cumulative
    stream_max_seqlen: dict[str, int]
    total_tokens: int
    padded_total: int

    @property
    def num_real_tokens(self) -> int:
        return int(self.total_tokens)

    def stream_total(self, name: str) -> int:
        return int(sum(self.stream_lengths.get(name, [])))


def build_stream_major_plan(
    meta,
    *,
    stream_order=DEFAULT_STREAM_ORDER,
    device=None,
) -> StreamMajorPlan:
    """Build a stream-major reordering plan from a :class:`VarlenBatchMeta`.

    Streams that are zero-length across every segment are dropped from the plan
    (e.g. ``anchor`` when no separate anchor is used). Any real token not covered
    by the canonical stream groups (future value tokens, etc.) is preserved in
    sample-major order after the named streams, and padding stays at the tail, so
    ``perm`` is always a valid permutation of ``[0, padded_total)``.
    """

    segments = list(meta.segments)
    total = int(meta.total_tokens)
    padded = int(meta.padded_total)

    present = [
        name for name in stream_order
        if any(_expected_stream_len(seg, name) > 0 for seg in segments)
    ]

    index_parts: list[torch.Tensor] = []
    stream_lengths: dict[str, list[int]] = {}
    for name in present:
        lens: list[int] = []
        for seg in segments:
            ln = _expected_stream_len(seg, name)
            lens.append(int(ln))
            if ln > 0:
                start, end = seg.stream_span(name)
                index_parts.append(torch.arange(int(start), int(end), dtype=torch.long))
        stream_lengths[name] = lens

    covered = torch.cat(index_parts) if index_parts else torch.empty(0, dtype=torch.long)
    if int(covered.numel()) < total:
        keep = torch.ones(total, dtype=torch.bool)
        if covered.numel():
            keep[covered] = False
        uncovered = torch.arange(total, dtype=torch.long)[keep]
    else:
        uncovered = torch.empty(0, dtype=torch.long)
    pad = torch.arange(total, padded, dtype=torch.long)

    perm = torch.cat([covered, uncovered, pad])
    if int(perm.numel()) != padded:
        raise ValueError(
            f"stream-major perm has {int(perm.numel())} entries, expected padded_total={padded}"
        )
    # Validate it really is a permutation before trusting the inverse.
    if not torch.equal(perm.sort().values, torch.arange(padded, dtype=torch.long)):
        raise ValueError("stream-major perm is not a permutation of [0, padded_total)")
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(padded, dtype=torch.long)

    cu: dict[str, torch.Tensor] = {}
    mx: dict[str, int] = {}
    for name, lens in stream_lengths.items():
        cu[name] = torch.cumsum(torch.tensor([0, *lens], dtype=torch.long), dim=0)
        mx[name] = int(max(lens)) if lens else 0

    if device is not None:
        perm = perm.to(device)
        inverse = inverse.to(device)
        cu = {k: v.to(device) for k, v in cu.items()}

    return StreamMajorPlan(
        perm=perm,
        inverse_perm=inverse,
        stream_order=tuple(present),
        stream_lengths=stream_lengths,
        stream_cu_seqlens=cu,
        stream_max_seqlen=mx,
        total_tokens=total,
        padded_total=padded,
    )


def to_stream_major(x: torch.Tensor, plan: StreamMajorPlan, *, dim: int = 1) -> torch.Tensor:
    """Reorder ``x`` along the token dimension into stream-major order."""

    return x.index_select(dim % x.ndim, plan.perm.to(x.device))


def to_sample_major(x: torch.Tensor, plan: StreamMajorPlan, *, dim: int = 1) -> torch.Tensor:
    """Inverse of :func:`to_stream_major` (restore sample-major order for decode)."""

    return x.index_select(dim % x.ndim, plan.inverse_perm.to(x.device))


def remap_dense_mask(mask: torch.Tensor, plan: StreamMajorPlan) -> torch.Tensor:
    """Reorder a dense ``[..., L, L]`` attention mask into stream-major order."""

    perm = plan.perm.to(mask.device)
    return mask.index_select(-2, perm).index_select(-1, perm)
