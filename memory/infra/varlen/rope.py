"""Metadata-driven RoPE assembly for packed memory training.

The closed stack passes ``f_seqlen_offsets`` / ``cuFs`` metadata straight into a
fused RoPE kernel. WAN instead consumes a materialized ``full_freqs`` tensor, so
the open analogue here *materializes that tensor from the VarlenBatchMeta*: the
per-segment clean/noisy RoPE start frames, gist/anchor/state identity spans, and
per-chunk action lengths all come from metadata rather than being recomputed
inline in the model. The model only supplies the raw frequency primitives
(``video_freq_fn`` and the action frequency table), keeping this layer model
agnostic while making the offsets genuinely load-bearing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .packing import pack_streams
from .types import PackedStreams, VarlenBatchMeta


def gist_freqs_from_clean(
    clean_freqs: torch.Tensor,
    *,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    num_frames: int,
) -> torch.Tensor:
    """Per-frame 3D RoPE for gist tokens (same temporal position as clean frame)."""
    M = int(gist_tokens_per_frame)
    vpf = int(video_tokens_per_frame)
    n_frames = int(num_frames)
    if M <= 0 or n_frames <= 0:
        return clean_freqs.new_zeros(0, *clean_freqs.shape[1:])
    parts = []
    for f in range(n_frames):
        frame_freq = clean_freqs[f * vpf : f * vpf + 1]
        parts.append(frame_freq.expand(M, -1, -1))
    return torch.cat(parts, dim=0)


def assemble_packed_freqs(
    meta: VarlenBatchMeta,
    *,
    video_freq_fn,
    action_freqs: torch.Tensor,
    action_freq_fn=None,
    identity_template: torch.Tensor,
    pad_to: int | None = None,
    pad_value: float = 1.0,
    output_batch_dim: bool = False,
    gist_use_frame_rope: bool = False,
) -> PackedStreams:
    """Build the packed ``full_freqs`` row from metadata.

    Args:
        meta: batch metadata produced by ``build_packed_memory_window_block_mask``.
        video_freq_fn: ``f(n_frames, start_frame) -> [n_frames*vpf, 1, C]`` 3D RoPE
            frequencies (e.g. ``Motus._create_video_freqs``).
        action_freqs: legacy chunk-local action RoPE table.
        action_freq_fn: optional
            ``f(chunk_start_latent, action_len, actions_per_frame=apf)`` used
            for absolute video-clock action RoPE.
        identity_template: any tensor whose last dim is ``C`` and whose dtype/device
            should be used for identity (no-rotation) spans (e.g. ``freqs_state``).
        pad_to: pad the packed row to this length (defaults to ``meta.padded_total``).

    Returns:
        ``PackedStreams`` whose ``.tensor`` is ``[total_or_padded, 1, C]`` (or with a
        leading batch dim when ``output_batch_dim``), matching the order produced by
        the model's legacy ``_build_memory_full_freqs`` + anchor prepend.
    """

    params = meta.params or {}
    vpf = int(params.get("video_tokens_per_frame", 0))
    apf = int(params.get("action_tokens_per_frame", 0))
    spf = int(params.get("state_tokens_per_frame", 0))
    nvpf = int(params.get("value_tokens_per_frame", 0))
    gbs = int(params.get("gist_tokens_per_frame", 0))
    C = int(identity_template.shape[-1])
    device = identity_template.device
    dtype = identity_template.dtype

    def ident(length: int) -> torch.Tensor:
        return torch.ones((int(length), 1, C), device=device, dtype=dtype)

    seg_rows = []
    for seg in meta.segments:
        N = int(seg.n_pred_frames)
        parts = []
        anchor_len = int(seg.stream_lengths.get("anchor", 0))
        if anchor_len > 0:
            parts.append(ident(anchor_len))
        clean_freqs = video_freq_fn(N, int(seg.clean_rope_start))
        parts.append(clean_freqs)
        gist_len = int(seg.stream_lengths.get("gist", 0))
        if gist_len > 0:
            if gist_use_frame_rope:
                parts.append(
                    gist_freqs_from_clean(
                        clean_freqs,
                        video_tokens_per_frame=vpf,
                        gist_tokens_per_frame=gbs,
                        num_frames=N,
                    )
                )
            else:
                parts.append(ident(gist_len))
        noisy_freqs = video_freq_fn(N, int(seg.noisy_rope_start))
        v_off = 0
        latent_offset = 0
        for count in seg.block_latent_counts:
            count = int(count)
            if spf > 0:
                parts.append(ident(spf))
            a_len = count * apf
            if a_len > 0:
                if action_freq_fn is None:
                    parts.append(action_freqs[:a_len].reshape(a_len, 1, -1))
                else:
                    parts.append(
                        action_freq_fn(
                            int(seg.noisy_rope_start) + latent_offset,
                            a_len,
                            actions_per_frame=apf,
                        )
                    )
            v_len = count * vpf
            parts.append(noisy_freqs[v_off:v_off + v_len])
            v_off += v_len
            latent_offset += count
            if nvpf > 0:
                parts.append(ident(nvpf))
        row = torch.cat(parts, dim=0)
        if int(row.shape[0]) != int(seg.token_length):
            raise ValueError(
                f"segment {seg.index} freq length {int(row.shape[0])} != token_length {int(seg.token_length)}"
            )
        seg_rows.append(row)

    target = int(pad_to) if pad_to is not None else int(meta.padded_total)
    return pack_streams(seg_rows, pad_to=target, output_batch_dim=output_batch_dim, pad_value=pad_value)
