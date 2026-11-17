"""Logical and FlexAttention masks for packed memory-window training.

The expensive parts of building a packed mask are (1) the per-sample logical
block matrix built with a Python double loop, (2) ``torch.block_diag`` over all
samples, and (3) compiling the FlexAttention ``BlockMask``. None of those depend
on the per-batch caption / valid-length / latent-start metadata, so they are
computed once per *structural signature* and cached. Only the cheap per-batch
metadata (stream cumulative lengths, RoPE offsets, valid-frame counts) is rebuilt
on every call.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import create_block_mask

from .attention import block_indices_from_sizes
from .factored_attention import build_sample_major_layout_meta
from .factored_plans import build_segment_factored_plans
from .metadata import make_rope_meta, make_stream_meta
from .pad_ceilings import grow_pad_ceilings
from .types import VarlenBatchMeta, VarlenSegment

compiled_create_block_mask = torch.compile(create_block_mask)

TOKEN_CLEAN_VIDEO = 1
TOKEN_GIST = 2
TOKEN_ACTION = 3
TOKEN_NOISY_VIDEO = 4
TOKEN_VALUE = 5
TOKEN_ANCHOR = 6
TOKEN_PAD = 0


def normalize_memory_mode(attention_mode):
    mode = str(attention_mode or "joint").lower()
    if mode == "action_only":
        mode = "fdm"
    if mode not in {"joint", "video_idm", "fdm", "decoupled"}:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    return mode


def _round_up(value: int, multiple: int) -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    return int(math.ceil(value / multiple) * multiple)


def _make_local_mask(blocks: list[dict], *, anchor_frames: int, recent_frames: int, separate_anchor: bool, mode: str, device) -> torch.Tensor:
    """Per-block visibility matching :func:`build_memory_visibility` (§1)."""
    n = len(blocks)
    local = torch.zeros((n, n), device=device, dtype=torch.bool)
    update_history_frames = max(0, int(recent_frames) - 1)
    for qi, qb in enumerate(blocks):
        for ki, kb in enumerate(blocks):
            if qi == ki:
                local[qi, ki] = True
                continue
            q_type = qb["type"]
            k_type = kb["type"]
            fq, fk = qb["frame"], kb["frame"]
            pq, pk = qb["phys"], kb["phys"]

            is_clean_q = q_type in {TOKEN_CLEAN_VIDEO, TOKEN_ANCHOR}
            is_gist_q = q_type == TOKEN_GIST
            is_pred_q = q_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO, TOKEN_VALUE}
            is_anchor_k = k_type == TOKEN_ANCHOR
            is_clean_k = k_type == TOKEN_CLEAN_VIDEO
            is_gist_k = k_type == TOKEN_GIST

            hist_ok = False
            if is_clean_q:
                if is_clean_k:
                    if pk == pq:
                        hist_ok = True
                    elif pk < pq:
                        in_recent = pk >= (pq - update_history_frames)
                        if separate_anchor:
                            hist_ok = in_recent
                        else:
                            hist_ok = in_recent or (pk < anchor_frames)
                elif is_gist_k:
                    hist_ok = pk <= pq
                elif is_anchor_k:
                    hist_ok = True
            elif is_gist_q:
                if is_clean_k:
                    if pk == pq:
                        hist_ok = True
                    elif pk < pq:
                        in_recent = pk >= (pq - update_history_frames)
                        if separate_anchor:
                            hist_ok = in_recent
                        else:
                            hist_ok = in_recent or (pk < anchor_frames)
                elif is_gist_k:
                    hist_ok = pk <= pq
                elif is_anchor_k:
                    hist_ok = True
            elif is_pred_q:
                if is_clean_k:
                    # Same retained window the inference KV cache keeps, so that
                    # anchor_frames / recent_frames mean the same thing here as they
                    # do at deploy time (see build_memory_visibility §1.1).
                    if pk < pq:
                        in_recent = pk >= (pq - recent_frames)
                        if separate_anchor:
                            hist_ok = in_recent
                        else:
                            hist_ok = in_recent or (pk < anchor_frames)
                elif is_gist_k:
                    hist_ok = pk < pq
                elif is_anchor_k:
                    hist_ok = True

            same = fq == fk
            if mode == "video_idm":
                allowed_act = k_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO}
                allowed_vid = k_type == TOKEN_NOISY_VIDEO
            elif mode == "fdm":
                allowed_act = k_type == TOKEN_ACTION
                allowed_vid = k_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO}
            elif mode == "decoupled":
                allowed_act = k_type == TOKEN_ACTION
                allowed_vid = k_type == TOKEN_NOISY_VIDEO
            else:
                allowed_act = k_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO}
                allowed_vid = k_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO}
            allowed_val = k_type in {TOKEN_ACTION, TOKEN_NOISY_VIDEO, TOKEN_VALUE}
            intra_ok = same and (
                (q_type == TOKEN_ACTION and allowed_act)
                or (q_type == TOKEN_NOISY_VIDEO and allowed_vid)
                or (q_type == TOKEN_VALUE and allowed_val)
            )
            local[qi, ki] = bool(hist_ok or intra_ok)
    return local


# ---------------------------------------------------------------------------
# Structural skeleton + cache
# ---------------------------------------------------------------------------

@dataclass
class _SegSkeleton:
    n_pred_frames: int
    token_length: int
    offset: int
    block_offset: int
    n_blocks: int
    block_sizes: tuple
    block_latent_counts: tuple
    anchor_len: int
    clean_len: int
    gist_len: int
    noisy_len: int
    action_len: int
    value_len: int
    video_family_len: int
    block_types: tuple = ()
    block_phys: tuple = ()


@dataclass
class _BlockSkeleton:
    segments: list
    total_tokens: int
    padded_total: int
    q_block_sizes: tuple
    q_block_indices: torch.Tensor
    logical_block_mask: torch.Tensor
    flex_block_mask: object
    anchor_lengths: list
    clean_lengths: list
    noisy_lengths: list
    action_lengths: list
    gist_lengths: list
    video_family_lengths: list


_SKELETON_CACHE: "OrderedDict[tuple, _BlockSkeleton]" = OrderedDict()
_SKELETON_CACHE_MAX = 64


def clear_block_mask_cache() -> None:
    """Drop all cached skeletons (used by tests / device switches)."""

    _SKELETON_CACHE.clear()


def block_mask_cache_info() -> dict:
    return {"size": len(_SKELETON_CACHE), "max": _SKELETON_CACHE_MAX}


def _skeleton_signature(W_list, *, vbs, gbs, abs_, sbs, nvpf, anchor_frames, recent_frames,
                        n_anchor_tok, mode, obs, cs, pad_multiple, separate_anchor, device) -> tuple:
    dev = torch.device(device)
    return (
        tuple(int(w) for w in W_list),
        int(vbs), int(gbs), int(abs_), int(sbs), int(nvpf),
        int(anchor_frames), int(recent_frames), int(n_anchor_tok),
        str(mode), int(obs), int(cs), int(pad_multiple), bool(separate_anchor),
        dev.type, dev.index,
    )


def _compute_block_skeleton(W_list, *, vbs, gbs, abs_, sbs, nvpf, anchor_frames, recent_frames,
                            n_anchor_tok, mode, obs, cs, pad_multiple, separate_anchor, device) -> _BlockSkeleton:
    def _fid(phys):
        return 0 if phys < obs else 1 + (phys - obs) // cs

    local_masks = []
    all_block_sizes: list[int] = []
    seg_skeletons: list[_SegSkeleton] = []
    token_offset = 0
    block_offset = 0
    anchor_lengths, clean_lengths, noisy_lengths = [], [], []
    action_lengths, gist_lengths, video_family_lengths = [], [], []

    for W_raw in W_list:
        # Whole predict chunks only: drop short tail so N % cs == 0.
        N = ((int(W_raw) - obs) // cs) * cs
        W = N + obs
        if N < 1:
            raise ValueError(
                f"packed segment window({W_raw}) with obs={obs}, cs={cs} "
                f"has no full predict chunk (aligned N={N})"
            )

        blocks: list[dict] = []

        def add_block(token_type, frame, phys, size):
            size = int(size)
            if size <= 0:
                return
            blocks.append({"type": int(token_type), "frame": int(frame), "phys": int(phys), "size": size})

        for _a in range(n_anchor_tok):
            add_block(TOKEN_ANCHOR, -1, -1, vbs)
        for f in range(N):
            add_block(TOKEN_CLEAN_VIDEO, _fid(f), f, vbs)
        for f in range(N):
            add_block(TOKEN_GIST, _fid(f), f, gbs)

        block_latent_counts = []
        noisy_offset = 0
        action_len_total = 0
        noisy_len_total = 0
        value_len_total = 0
        while noisy_offset < N:
            count = cs  # N is cs-aligned; no partial last chunk
            block_latent_counts.append(count)
            chunk_phys = obs + noisy_offset
            chunk_fid = _fid(chunk_phys)
            action_len = sbs + count * abs_
            noisy_len = count * vbs
            action_len_total += action_len
            noisy_len_total += noisy_len
            add_block(TOKEN_ACTION, chunk_fid, chunk_phys, action_len)
            add_block(TOKEN_NOISY_VIDEO, chunk_fid, chunk_phys, noisy_len)
            add_block(TOKEN_VALUE, chunk_fid, chunk_phys, nvpf)
            value_len_total += nvpf
            noisy_offset += count

        local_masks.append(_make_local_mask(blocks, anchor_frames=anchor_frames, recent_frames=recent_frames,
                                             separate_anchor=separate_anchor, mode=mode, device=device))
        block_sizes = [b["size"] for b in blocks]
        all_block_sizes.extend(block_sizes)
        token_length = sum(block_sizes)
        anchor_len = n_anchor_tok * vbs
        clean_len = N * vbs
        gist_len = N * gbs
        video_family_len = anchor_len + clean_len + noisy_len_total
        anchor_lengths.append(anchor_len)
        clean_lengths.append(clean_len)
        noisy_lengths.append(noisy_len_total)
        action_lengths.append(action_len_total)
        gist_lengths.append(gist_len)
        video_family_lengths.append(video_family_len)
        seg_skeletons.append(_SegSkeleton(
            n_pred_frames=N,
            token_length=token_length,
            offset=token_offset,
            block_offset=block_offset,
            n_blocks=len(blocks),
            block_sizes=tuple(block_sizes),
            block_latent_counts=tuple(block_latent_counts),
            anchor_len=anchor_len,
            clean_len=clean_len,
            gist_len=gist_len,
            noisy_len=noisy_len_total,
            action_len=action_len_total,
            value_len=value_len_total,
            video_family_len=video_family_len,
            block_types=tuple(b["type"] for b in blocks),
            block_phys=tuple(b["phys"] for b in blocks),
        ))
        token_offset += token_length
        block_offset += len(blocks)

    total = token_offset
    padded_total = _round_up(total, pad_multiple)
    pad = padded_total - total
    logical = torch.block_diag(*local_masks) if local_masks else torch.zeros((0, 0), device=device, dtype=torch.bool)
    q_block_sizes = list(all_block_sizes)
    if pad > 0:
        pad_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
        logical = torch.block_diag(logical, pad_mask) if logical.numel() else pad_mask
        q_block_sizes.append(pad)

    q_block_indices = block_indices_from_sizes(q_block_sizes, device=device)

    def mask_mod(b, h, q_idx, kv_idx):
        return logical[q_block_indices[q_idx], q_block_indices[kv_idx]]

    flex_mask = compiled_create_block_mask(mask_mod, 1, 1, padded_total, padded_total, device=device, _compile=True)

    return _BlockSkeleton(
        segments=seg_skeletons,
        total_tokens=total,
        padded_total=padded_total,
        q_block_sizes=tuple(q_block_sizes),
        q_block_indices=q_block_indices,
        logical_block_mask=logical,
        flex_block_mask=flex_mask,
        anchor_lengths=anchor_lengths,
        clean_lengths=clean_lengths,
        noisy_lengths=noisy_lengths,
        action_lengths=action_lengths,
        gist_lengths=gist_lengths,
        video_family_lengths=video_family_lengths,
    )


def _get_block_skeleton(W_list, **kwargs) -> _BlockSkeleton:
    key = _skeleton_signature(W_list, **kwargs)
    cached = _SKELETON_CACHE.get(key)
    if cached is not None:
        _SKELETON_CACHE.move_to_end(key)
        return cached
    skeleton = _compute_block_skeleton(W_list, **kwargs)
    _SKELETON_CACHE[key] = skeleton
    _SKELETON_CACHE.move_to_end(key)
    while len(_SKELETON_CACHE) > _SKELETON_CACHE_MAX:
        _SKELETON_CACHE.popitem(last=False)
    return skeleton


def _retention_logical_mask(skeleton: _BlockSkeleton, retained, device) -> torch.Tensor:
    """Widen the cached visibility with per-segment retained clean frames.

    ``retained[seg]`` lists predict-window clean frames (physical index) that stay
    visible beyond the recent window: a clean/gist query sees a retained clean
    frame with ``pk < pq``; a predict query (action / noisy video / value) sees it
    with ``pk < pq``. This is the training-side twin of a KV cache that keeps
    selected frames after they leave the recent window.
    """
    logical = skeleton.logical_block_mask.clone()
    for ss, frames in zip(skeleton.segments, retained):
        if frames is None or len(frames) == 0:
            continue
        types = torch.tensor(ss.block_types, device=device, dtype=torch.long)
        phys = torch.tensor(ss.block_phys, device=device, dtype=torch.long)
        keep = torch.zeros(int(ss.n_pred_frames), device=device, dtype=torch.bool)
        idx = torch.tensor(sorted(int(f) for f in frames), device=device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < keep.numel())]
        keep[idx] = True
        is_clean = types == TOKEN_CLEAN_VIDEO
        key_ok = is_clean & keep[phys.clamp(min=0, max=max(keep.numel() - 1, 0))]
        query_ok = (types == TOKEN_CLEAN_VIDEO) | (types == TOKEN_GIST) | (
            (types == TOKEN_ACTION) | (types == TOKEN_NOISY_VIDEO) | (types == TOKEN_VALUE)
        )
        add = query_ok.view(-1, 1) & key_ok.view(1, -1) & (phys.view(1, -1) < phys.view(-1, 1))
        lo, hi = ss.block_offset, ss.block_offset + ss.n_blocks
        logical[lo:hi, lo:hi] |= add
    return logical


def build_packed_memory_window_block_mask(
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
    pad_multiple: int = 128,
    caption_lengths: Sequence[int] | None = None,
    caption_padded_lengths: Sequence[int] | None = None,
    valid_lens: Sequence[int] | None = None,
    latent_starts: Sequence[int] | None = None,
    latent_hw: tuple[int, int] | None = None,
    cp_size: int = 1,
    retained: Sequence[Sequence[int] | None] | None = None,
) -> VarlenBatchMeta:
    """Build block-diagonal packed logical block-diag mask and full metadata.

    The structural mask (logical block-diag matrix + compiled FlexAttention mask)
    is cached by signature; only the per-batch metadata below is rebuilt.
    """

    W_list = [int(v) for v in num_latent_frames_list]
    B = len(W_list)
    cs = max(1, int(chunk_size))
    obs = cs if obs_latents is None else max(1, int(obs_latents))
    vbs = int(video_tokens_per_frame)
    gbs = int(gist_tokens_per_frame)
    abs_ = int(action_tokens_per_frame)
    sbs = int(state_tokens_per_frame) if use_state else 0
    nvpf = int(value_tokens_per_frame)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    n_anchor_tok = anchor_frames if separate_anchor else 0
    mode = normalize_memory_mode(attention_mode)
    captions = [0] * B if caption_lengths is None else [int(v) for v in caption_lengths]
    captions_padded = captions if caption_padded_lengths is None else [int(v) for v in caption_padded_lengths]
    valid = list(W_list) if valid_lens is None else [int(v) for v in valid_lens]
    starts = [0] * B if latent_starts is None else [int(v) for v in latent_starts]
    lat_h, lat_w = latent_hw or (1, 1)

    skeleton = _get_block_skeleton(
        W_list, vbs=vbs, gbs=gbs, abs_=abs_, sbs=sbs, nvpf=nvpf,
        anchor_frames=anchor_frames, recent_frames=recent_frames, n_anchor_tok=n_anchor_tok,
        mode=mode, obs=obs, cs=cs, pad_multiple=pad_multiple, separate_anchor=separate_anchor,
        device=device,
    )

    segments: list[VarlenSegment] = []
    f_offsets: list[int] = []
    noisy_f_offsets: list[int] = []
    gist_f_offsets: list[int] = []
    gist_h_offsets: list[int] = []
    gist_w_offsets: list[int] = []

    for seg, ss in enumerate(skeleton.segments):
        # Skeleton already dropped a short predict tail; use aligned W=N+obs.
        N = ss.n_pred_frames
        W = N + obs
        valid_w = min(int(valid[seg]), W)
        valid_N = max(0, valid_w - obs)
        valid_N = (valid_N // cs) * cs

        if separate_anchor:
            clean_rope_start, noisy_rope_start = 0, obs
        else:
            clean_rope_start, noisy_rope_start = starts[seg], starts[seg] + obs

        for f in range(N):
            f_offsets.append(clean_rope_start + f)
            noisy_f_offsets.append(noisy_rope_start + f)
        for f in range(N):
            for _ in range(gbs):
                gist_f_offsets.append(starts[seg] + f)
                gist_h_offsets.append(lat_h)
                gist_w_offsets.append(lat_w)

        segments.append(VarlenSegment(
            index=seg,
            token_length=ss.token_length,
            offset=ss.offset,
            stream_lengths={
                "anchor": ss.anchor_len,
                "clean_video": ss.clean_len,
                "noisy_video": ss.noisy_len,
                "action": ss.action_len,
                "value": ss.value_len,
                "gist": ss.gist_len,
                "caption": captions[seg],
                "window": W,
                "predict_latents": N,
            },
            block_sizes=ss.block_sizes,
            block_offset=ss.block_offset,
            n_pred_frames=N,
            valid_pred_frames=valid_N,
            latent_start=starts[seg],
            clean_rope_start=clean_rope_start,
            noisy_rope_start=noisy_rope_start,
            block_latent_counts=ss.block_latent_counts,
        ))

    streams = {
        "anchor": make_stream_meta("anchor", skeleton.anchor_lengths, device=device, cp_size=cp_size),
        "clean_video": make_stream_meta("clean_video", skeleton.clean_lengths, device=device, cp_size=cp_size),
        "noisy_video": make_stream_meta("noisy_video", skeleton.noisy_lengths, device=device, cp_size=cp_size),
        "video": make_stream_meta("video", skeleton.video_family_lengths, device=device, cp_size=cp_size),
        "action": make_stream_meta("action", skeleton.action_lengths, device=device, cp_size=cp_size),
        "caption": make_stream_meta("caption", captions, padded_lengths=captions_padded, device=device, cp_size=cp_size),
        "gist": make_stream_meta("gist", skeleton.gist_lengths, device=device, cp_size=cp_size),
    }
    rope = {
        "video": make_rope_meta(skeleton.clean_lengths, f_offsets=f_offsets, device=device),
        "noisy_video": make_rope_meta(skeleton.noisy_lengths, f_offsets=noisy_f_offsets, device=device),
        "gist": make_rope_meta(skeleton.gist_lengths, f_offsets=gist_f_offsets, h_offsets=gist_h_offsets, w_offsets=gist_w_offsets, device=device),
    }
    params = {
        "video_tokens_per_frame": vbs,
        "gist_tokens_per_frame": gbs,
        "action_tokens_per_frame": abs_,
        "state_tokens_per_frame": sbs,
        "value_tokens_per_frame": nvpf,
        "chunk_size": cs,
        "obs_latents": obs,
        "anchor_frames": anchor_frames,
        "recent_frames": recent_frames,
        "n_anchor_tokens": n_anchor_tok,
        "separate_anchor": int(bool(separate_anchor)),
        "latent_height": lat_h,
        "latent_width": lat_w,
    }

    # Factored plans/layouts: one per segment. Geometry uses N = W - obs
    # (predict/clean frame count), matching build_memory_visibility(num_frames=N)
    # and the skeleton's n_pred_frames. Skip when layout diverges (separate
    # anchor stream or value tokens) — packed path falls back to Flex.
    factored_plans = []
    factored_layouts = []
    logical_block_mask = skeleton.logical_block_mask
    flex_block_mask = skeleton.flex_block_mask
    has_retention = retained is not None and any(r is not None and len(r) > 0 for r in retained)
    if has_retention:
        if len(retained) != B:
            raise ValueError(f"retained has {len(retained)} entries for {B} segments")
        logical_block_mask = _retention_logical_mask(skeleton, retained, device)
        q_block_indices = skeleton.q_block_indices

        def retention_mask_mod(b, h, q_idx, kv_idx):
            return logical_block_mask[q_block_indices[q_idx], q_block_indices[kv_idx]]

        flex_block_mask = compiled_create_block_mask(
            retention_mask_mod, 1, 1, skeleton.padded_total, skeleton.padded_total,
            device=device, _compile=True,
        )

    factored_ok = (n_anchor_tok == 0) and (nvpf == 0) and not has_retention
    if factored_ok:
        for seg, ss in enumerate(skeleton.segments):
            N = int(ss.n_pred_frames)
            plans = build_segment_factored_plans(
                num_clean_frames=N,
                obs_latents=obs,
                chunk_size=cs,
                video_tokens_per_frame=vbs,
                gist_tokens_per_frame=gbs,
                action_tokens_per_frame=abs_,
                state_tokens_per_frame=sbs,
                recent_frames=recent_frames,
                anchor_frames=anchor_frames,
            )
            layout = build_sample_major_layout_meta(
                num_clean_frames=N,
                obs_latents=obs,
                chunk_size=cs,
                video_tokens_per_frame=vbs,
                gist_tokens_per_frame=gbs,
                action_tokens_per_frame=abs_,
                state_tokens_per_frame=sbs,
                value_tokens_per_frame=0,
            )
            if layout.total_len != int(ss.token_length):
                raise ValueError(
                    f"factored layout total_len({layout.total_len}) != "
                    f"segment token_length({ss.token_length}) for seg {seg} "
                    f"(N={N}, W={W_list[seg]}, obs={obs})"
                )
            grow_pad_ceilings(plans.planned_kv_token_lengths)
            factored_plans.append(plans)
            factored_layouts.append(layout)

    return VarlenBatchMeta(
        segments=segments,
        total_tokens=skeleton.total_tokens,
        padded_total=skeleton.padded_total,
        block_sizes=skeleton.q_block_sizes,
        logical_block_mask=logical_block_mask,
        flex_block_mask=flex_block_mask,
        q_block_sizes=skeleton.q_block_sizes,
        kv_block_sizes=skeleton.q_block_sizes,
        q_block_indices=skeleton.q_block_indices,
        kv_block_indices=skeleton.q_block_indices,
        streams=streams,
        rope=rope,
        caption_lengths=torch.tensor(captions, device=device, dtype=torch.int32),
        valid_lens=torch.tensor(valid, device=device, dtype=torch.int32),
        latent_starts=torch.tensor(starts, device=device, dtype=torch.int32),
        params=params,
        factored_plans=tuple(factored_plans),
        factored_layouts=tuple(factored_layouts),
    )
