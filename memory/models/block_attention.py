# Flex-attention masks for sliding-window AR teacher-forcing training.
# Combines lingbot-va style frame_id/noise_id encoding with DreamZero style
# joint video-action prediction (no clean action).

import math
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from einops import rearrange
from infra.varlen.block_mask import build_packed_memory_window_block_mask

# Static specializations cover grad/no-grad, VM sub-batches and packed lengths.
# Support both the older and newer Dynamo configuration names.
for _cache_key in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _cache_key):
        setattr(torch._dynamo.config, _cache_key,
                max(getattr(torch._dynamo.config, _cache_key), 64))

flex_attention_compiled = torch.compile(
    flex_attention, dynamic=False, fullgraph=True,
)
compiled_create_block_mask = torch.compile(create_block_mask)


# ---------------------------------------------------------------------------
# Hybrid-memory (anchor + gist + recent) visibility — single source of truth
# ---------------------------------------------------------------------------
# These token-type ids are shared by the dense visibility helper
# (`build_memory_visibility`), the compiled training mask
# (`create_memory_window_mask`), and streaming inference, so all three apply
# the exact same anchor/recent/gist + FDM rules.
TOKEN_CLEAN_VIDEO = 1   # full per-frame clean (history) video tokens
TOKEN_GIST = 2          # compact per-frame learnable memory tokens
TOKEN_ACTION = 3        # action register (includes state token when use_state)
TOKEN_NOISY_VIDEO = 4   # noisy (to-predict) video tokens
TOKEN_VALUE = 5         # chunk-level value query tokens
TOKEN_ANCHOR = 6        # separate global-L0 anchor (identity RoPE, always visible)


def normalize_memory_mode(attention_mode):
    """action_only is the public name of the fdm visibility rule.

    decoupled drops the same-chunk action -> noisy_video edge that fdm carries, so the video
    branch has to reach the future through memory instead of reading the actions that produce
    it. That matches MemoryWAM Eq. 2, whose video update takes no action argument.
    """
    mode = str(attention_mode or "joint").lower()
    if mode == "action_only":
        mode = "fdm"
    if mode not in {"joint", "video_idm", "fdm", "decoupled"}:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    return mode


def _retained_full_mask(
    pk: torch.Tensor,
    pq: torch.Tensor,
    *,
    anchor_frames: int,
    recent_frames: int,
    anchor_start: int = 0,
    separate_anchor: bool = False,
    recent_includes_query: bool = False,
) -> torch.Tensor:
    """Return retained full-frame history for update or prediction queries."""
    recent_history_frames = max(
        0,
        int(recent_frames) - int(bool(recent_includes_query)),
    )
    retained = (pk < pq) & (pk >= (pq - recent_history_frames))
    if not separate_anchor and int(anchor_frames) > 0:
        retained |= (
            (pk < pq)
            & (pk >= int(anchor_start))
            & (pk < int(anchor_start) + int(anchor_frames))
        )
    return retained


def build_memory_visibility(
    num_frames: int,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    anchor_frames: int,
    recent_frames: int,
    action_tokens_per_frame: int,
    state_tokens_per_frame: int = 0,
    value_tokens_per_frame: int = 0,
    chunk_size: int = 1,
    attention_mode: str = "fdm",
    device: str = "cpu",
    obs_latents: int = None,
    separate_anchor: bool = False,
):
    """Dense boolean visibility for the action-first hybrid-memory layout.

    This is the single source of truth: the compiled training mask and the
    streaming inference KV both reproduce the rules encoded here. Layout::

        [ clean_video[f] (per frame) | gist[f] (per frame)
          | per-chunk( state?, action, noisy_video, value? ) ]

    Frame indexing follows the sliding-window AR convention: clean frames use
    physical positions ``0..N-1``; noisy chunks use physical positions
    ``obs..W-1``. ``N = num_frames``. Chunk id (``fid``) drives intra-chunk FDM
    only; history visibility is per query class in *latent* units (``phys``).

    Visibility rules (§1; ``pq``/``pk`` = latent phys; for action/noisy/value,
    ``pq`` is the chunk start ``q0``):

      * Clean queries (§1.3): full clean at ``pk == pq`` or retained recent/anchor
        history (``pk < pq`` and in recent/anchor window); gists with ``pk <= pq``.
      * Gist queries (§1.2, latent-unit): own clean ``pk == pq`` (required);
        retained full-resolution frames; gist chain ``pk <= pq``; optional separate
        anchor tokens. Must not use chunk-strict ``fk < fq`` (that blocks
        ``gist_i → clean_i``).
      * Action / noisy / value (§1.1, chunk-anchored): retained full clean, i.e.
        the ``recent_frames`` window before ``q0`` plus the leading
        ``anchor_frames``; past gists ``pk < q0``; optional separate anchors.
        This is the same set the inference KV cache retains, which
        ``Motus._build_infer_chunk_mask`` exposes in full to the chunk.
        ``anchor_frames`` may be ``0`` (no permanent L0 full pin).
      * Intra-chunk (same ``fid``) noisy-side rules implement FDM/action_only:
        action sees only same-chunk action, noisy_video sees action+video,
        value sees action+video+value (read-only for everyone else).
    """
    num_frames = int(num_frames)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    chunk_size = max(1, int(chunk_size))
    obs = chunk_size if obs_latents is None else max(1, int(obs_latents))
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if anchor_frames < 0:
        raise ValueError("anchor_frames must be >= 0")
    if recent_frames < 1:
        raise ValueError("recent_frames must be >= 1")
    mode = normalize_memory_mode(attention_mode)
    # When a separate global-L0 anchor token is used, the in-window frames no
    # longer act as anchors (only the prepended anchor token is the landmark).
    n_anchor_tok = anchor_frames if separate_anchor else 0

    vbs = int(video_tokens_per_frame)
    gbs = int(gist_tokens_per_frame)
    abs_ = int(action_tokens_per_frame)
    sbs = int(state_tokens_per_frame)
    nvpf = int(value_tokens_per_frame)

    N = num_frames
    # Noisy side is grouped into chunks of chunk_size frames.
    block_latent_counts = []
    offset = 0
    while offset < N:
        block_latent_counts.append(min(chunk_size, N - offset))
        offset += chunk_size

    types: list = []
    frames: list = []   # chunk id: drives intra-chunk FDM only
    physes: list = []   # physical latent index: drives history visibility

    def _fid(phys):
        # obs observation latents -> chunk id 0; each later cs-chunk -> next id.
        return 0 if phys < obs else 1 + (phys - obs) // chunk_size

    def _emit(token_type, frame_id, phys_id, count):
        for _ in range(int(count)):
            types.append(token_type)
            frames.append(int(frame_id))
            physes.append(int(phys_id))

    # Separate global-L0 anchor tokens (identity RoPE, always visible). phys = -1
    # so they never satisfy the in_recent rule; they are kept via is_anchor.
    for _a in range(n_anchor_tok):
        _emit(TOKEN_ANCHOR, -1, -1, vbs)
    # Clean video (history), per frame, physical positions 0..N-1.
    for f in range(N):
        _emit(TOKEN_CLEAN_VIDEO, _fid(f), f, vbs)
    # Gist memory, per frame, sharing the clean frame/phys ids.
    for f in range(N):
        _emit(TOKEN_GIST, _fid(f), f, gbs)
    # Noisy side, action-first per chunk: [state?, action, noisy_video, value?].
    # All tokens in a chunk share the chunk id (FDM) and the chunk's first
    # physical latent index q0 (adjacent-full history is keyed by q0-1).
    noisy_offset = 0
    for count in block_latent_counts:
        chunk_phys = obs + noisy_offset                 # first noisy latent's physical index
        chunk_fid = _fid(chunk_phys)
        _emit(TOKEN_ACTION, chunk_fid, chunk_phys, sbs)             # state register
        _emit(TOKEN_ACTION, chunk_fid, chunk_phys, count * abs_)    # action register
        _emit(TOKEN_NOISY_VIDEO, chunk_fid, chunk_phys, count * vbs)
        _emit(TOKEN_VALUE, chunk_fid, chunk_phys, nvpf)
        noisy_offset += count

    token_types = torch.tensor(types, device=device, dtype=torch.long)
    frame_ids = torch.tensor(frames, device=device, dtype=torch.long)
    phys_ids = torch.tensor(physes, device=device, dtype=torch.long)
    total = token_types.numel()

    tq = token_types.view(-1, 1)
    tk = token_types.view(1, -1)
    fq = frame_ids.view(-1, 1)
    fk = frame_ids.view(1, -1)
    pq = phys_ids.view(-1, 1)
    pk = phys_ids.view(1, -1)

    is_anchor_k = (tk == TOKEN_ANCHOR)
    is_gist_q = tq == TOKEN_GIST
    is_clean_q = (tq == TOKEN_CLEAN_VIDEO) | (tq == TOKEN_ANCHOR)
    is_pred_q = (tq == TOKEN_ACTION) | (tq == TOKEN_NOISY_VIDEO) | (tq == TOKEN_VALUE)

    # A clean/gist query counts its own frame inside recent_frames. Prediction
    # queries sit after the latest clean frame, so all recent_frames are history.
    update_retained_full = _retained_full_mask(
        pk,
        pq,
        anchor_frames=anchor_frames,
        recent_frames=recent_frames,
        separate_anchor=separate_anchor,
        recent_includes_query=True,
    )
    pred_retained_full = _retained_full_mask(
        pk,
        pq,
        anchor_frames=anchor_frames,
        recent_frames=recent_frames,
        separate_anchor=separate_anchor,
    )

    clean_full = is_clean_q & (tk == TOKEN_CLEAN_VIDEO) & (
        (pk == pq) | update_retained_full
    )
    clean_gist = is_clean_q & (tk == TOKEN_GIST) & (pk <= pq)
    clean_anchor = is_clean_q & is_anchor_k
    gist_full = is_gist_q & (tk == TOKEN_CLEAN_VIDEO) & (
        (pk == pq) | update_retained_full
    )
    gist_chain = is_gist_q & (tk == TOKEN_GIST) & (pk <= pq)
    gist_anchor = is_gist_q & is_anchor_k
    pred_full = is_pred_q & (tk == TOKEN_CLEAN_VIDEO) & pred_retained_full
    pred_gist = is_pred_q & (tk == TOKEN_GIST) & (pk < pq)
    pred_anchor = is_pred_q & is_anchor_k

    hist_ok = (
        clean_full | clean_gist | clean_anchor
        | gist_full | gist_chain | gist_anchor
        | pred_full | pred_gist | pred_anchor
    )

    same = fk == fq
    act_q = tq == TOKEN_ACTION
    vid_q = tq == TOKEN_NOISY_VIDEO
    val_q = tq == TOKEN_VALUE
    k_act = tk == TOKEN_ACTION
    k_vid = tk == TOKEN_NOISY_VIDEO
    k_val = tk == TOKEN_VALUE
    if mode == "video_idm":
        allowed_act = k_act | k_vid
        allowed_vid = k_vid
    elif mode == "fdm":
        allowed_act = k_act
        allowed_vid = k_act | k_vid
    elif mode == "decoupled":
        allowed_act = k_act
        allowed_vid = k_vid
    else:  # joint
        allowed_act = k_act | k_vid
        allowed_vid = k_act | k_vid
    allowed_val = k_act | k_vid | k_val
    intra = same & ((act_q & allowed_act) | (vid_q & allowed_vid) | (val_q & allowed_val))

    mask = hist_ok | intra
    mask = mask | torch.eye(total, device=device, dtype=torch.bool)
    return {
        "mask": mask,
        "token_types": token_types,
        "frame_ids": frame_ids,
        "phys_ids": phys_ids,
        "block_latent_counts": block_latent_counts,
        "n_anchor_tokens": int(n_anchor_tok),
        "TOKEN_CLEAN_VIDEO": TOKEN_CLEAN_VIDEO,
        "TOKEN_GIST": TOKEN_GIST,
        "TOKEN_ACTION": TOKEN_ACTION,
        "TOKEN_NOISY_VIDEO": TOKEN_NOISY_VIDEO,
        "TOKEN_VALUE": TOKEN_VALUE,
        "TOKEN_ANCHOR": TOKEN_ANCHOR,
    }


def build_fill_cache_mask(
    cache_frame_ids: torch.Tensor,
    cache_is_gist: torch.Tensor,
    new_frame_start: int,
    K: int,
    *,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    anchor_frames: int,
    recent_frames: int,
    obs_latents: int,
    chunk_size: int,
    cache_first_frame: int,
    separate_anchor: bool = False,
    device: torch.device,
    retained_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bool visibility [Lq, Lcache+Lq] for streaming KV fill (clean+gist queries).

    Reproduces the clean/gist slice of :func:`build_memory_visibility` over
    ``[retained cache | new clean+gist batch]`` so inference fill matches training.
    Gist→clean uses latent-unit own-clean (``pk == pq``), not chunk-strict
    ``fk < fq``.
    """
    del obs_latents, chunk_size
    vbs = int(video_tokens_per_frame)
    gbs = int(gist_tokens_per_frame)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    if anchor_frames < 0:
        raise ValueError("anchor_frames must be >= 0")
    if recent_frames < 1:
        raise ValueError("recent_frames must be >= 1")
    K = int(K)
    if cache_frame_ids.numel() != cache_is_gist.numel():
        raise ValueError("cache frame/type metadata lengths differ")

    new_frames = torch.arange(
        int(new_frame_start),
        int(new_frame_start) + K,
        device=device,
        dtype=torch.long,
    )
    clean_physical = torch.repeat_interleave(new_frames, vbs)
    gist_physical = torch.repeat_interleave(new_frames, gbs)
    query_physical = torch.cat([clean_physical, gist_physical])
    query_is_gist = torch.cat([
        torch.zeros(clean_physical.numel(), device=device, dtype=torch.bool),
        torch.ones(gist_physical.numel(), device=device, dtype=torch.bool),
    ])

    cache_frame_ids = cache_frame_ids.to(device=device, dtype=torch.long)
    cache_is_gist = cache_is_gist.to(device=device, dtype=torch.bool)
    key_physical = torch.cat([cache_frame_ids, query_physical])
    key_is_gist = torch.cat([cache_is_gist, query_is_gist])

    query_physical_2d = query_physical.view(-1, 1)
    key_physical_2d = key_physical.view(1, -1)
    clean_query = (~query_is_gist).view(-1, 1)
    gist_query = query_is_gist.view(-1, 1)
    clean_key = (~key_is_gist).view(1, -1)
    gist_key = key_is_gist.view(1, -1)

    update_retained_full = _retained_full_mask(
        key_physical_2d,
        query_physical_2d,
        anchor_frames=anchor_frames,
        recent_frames=recent_frames,
        anchor_start=int(cache_first_frame),
        separate_anchor=separate_anchor,
        recent_includes_query=True,
    )
    if retained_frames is not None and retained_frames.numel() > 0:
        key_retained = torch.isin(key_physical, retained_frames.to(device=device, dtype=torch.long)).view(1, -1)
        update_retained_full = update_retained_full | (
            key_retained & (key_physical_2d < query_physical_2d)
        )
    clean_full = clean_query & clean_key & (
        (key_physical_2d == query_physical_2d) | update_retained_full
    )
    clean_history = clean_full | (
        clean_query & gist_key & (key_physical_2d <= query_physical_2d)
    )

    gist_history = gist_query & (
        (clean_key & (
            (key_physical_2d == query_physical_2d) | update_retained_full
        ))
        | (gist_key & (key_physical_2d <= query_physical_2d))
    )

    mask = clean_history | gist_history
    cache_len = cache_frame_ids.numel()
    diag = torch.arange(query_physical.numel(), device=device)
    mask[diag, cache_len + diag] = True
    return mask


# ---------------------------------------------------------------------------
# Sliding-window mask for training
# ---------------------------------------------------------------------------

def create_sliding_window_mask(
    num_latent_frames: int,
    video_tokens_per_frame: int,
    action_tokens_per_frame: int,
    chunk_size: int,
    window_size: int,
    use_state: bool = True,
    state_tokens_per_frame: int = 1,
    value_tokens_per_frame: int = 0,
    attention_mode: str = "joint",
    device: str = "cuda",
):
    """Create flex-attention mask for sliding-window teacher-forcing training.

    Token layout (robot-policy style):
      [clean_vid (frames 0..W-cs-1) | noisy_vid (frames cs..W-1) | noisy_act (frames cs..W-1)]

    The first chunk_size frames (block 0) are the observation — clean only, never predicted.
    Noisy side starts from the second block onwards.
    N = W - chunk_size frames per side.

    Args:
        num_latent_frames: W, total latent frames in the training window.
            Clean side: N frames at physical positions 0..W-cs-1.
            Noisy side: N frames at physical positions cs..W-1.
        video_tokens_per_frame: tokens per latent frame (H*W / patch_area).
        action_tokens_per_frame: action tokens per latent frame.
        chunk_size: how many latent frames per "block" in the mask (random 1~K_max).
        window_size: sliding window size in frame_id units (random W_min~W_max).
        use_state: whether to include 1 state token per frame in the action segment.
        state_tokens_per_frame: number of state tokens per frame (usually 1).
        device: target device.

    Returns:
        (block_mask, padded_total): FlexAttention BlockMask and padded sequence length.
    """
    W = num_latent_frames
    N = W - chunk_size  # frames per side; clean 0..N-1, noisy cs..W-1
    vbs = video_tokens_per_frame
    abs_ = action_tokens_per_frame
    sbs = state_tokens_per_frame if use_state else 0
    attention_mode = str(attention_mode or "joint").lower()
    if attention_mode not in {"joint", "video_idm", "action_only", "fdm"}:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    if attention_mode == "action_only":
        attention_mode = "fdm"

    block_latent_counts = []
    offset = 0
    while offset < N:
        count = min(chunk_size, N - offset)
        block_latent_counts.append(count)
        offset += count

    clean_len = N * vbs
    noisy_vid_len = N * vbs
    noisy_act_len = len(block_latent_counts) * value_tokens_per_frame + N * abs_ + len(block_latent_counts) * sbs
    total = clean_len + noisy_vid_len + noisy_act_len
    padded_total = math.ceil(total / 128) * 128

    # --- Build per-token metadata tensors ---
    # noise_id: 1 = clean, 0 = noisy
    # frame_id: PHYSICAL frame index // chunk_size
    #   Clean uses physical positions 0..W-2.
    #   Noisy uses physical positions 1..W-1.
    #   With block_causal_mask_exclude_self (fid_k < fid_q):
    #     noisy frame 1 sees clean frame 0  (fid 0 < fid 1)
    #     noisy frame W-1 sees clean 0..W-2 (all fid < fid W-1)
    noise_ids = torch.zeros(padded_total, device=device, dtype=torch.long)
    frame_ids = torch.full((padded_total,), -1, device=device, dtype=torch.long)
    phys_ids = torch.full((padded_total,), -1, device=device, dtype=torch.long)
    is_value = torch.zeros(padded_total, device=device, dtype=torch.long)
    token_types = torch.zeros(padded_total, device=device, dtype=torch.long)
    TOKEN_CLEAN_VIDEO = 1
    TOKEN_NOISY_VIDEO = 2
    TOKEN_ACTION = 3
    TOKEN_VALUE = 4

    pos = 0

    # Clean video: physical positions 0..N-1, noise_id=1
    for f in range(N):
        phys = f  # 0, 1, ..., W-cs-1
        fid = phys // chunk_size
        noise_ids[pos:pos + vbs] = 1
        frame_ids[pos:pos + vbs] = fid
        phys_ids[pos:pos + vbs] = phys
        token_types[pos:pos + vbs] = TOKEN_CLEAN_VIDEO
        pos += vbs

    # Noisy video: physical positions cs..W-1, noise_id=0
    for f in range(N):
        phys = f + chunk_size  # cs, cs+1, ..., W-1
        fid = phys // chunk_size
        noise_ids[pos:pos + vbs] = 0
        frame_ids[pos:pos + vbs] = fid
        phys_ids[pos:pos + vbs] = phys
        token_types[pos:pos + vbs] = TOKEN_NOISY_VIDEO
        pos += vbs

    # Noisy action/value registers: grouped per noisy chunk.
    noisy_offset = 0
    for count in block_latent_counts:
        chunk_phys_start = chunk_size + noisy_offset
        chunk_fid = chunk_phys_start // chunk_size

        if value_tokens_per_frame > 0:
            noise_ids[pos:pos + value_tokens_per_frame] = 0
            frame_ids[pos:pos + value_tokens_per_frame] = chunk_fid
            phys_ids[pos:pos + value_tokens_per_frame] = chunk_phys_start
            is_value[pos:pos + value_tokens_per_frame] = 1
            token_types[pos:pos + value_tokens_per_frame] = TOKEN_VALUE
            pos += value_tokens_per_frame

        if sbs > 0:
            noise_ids[pos:pos + sbs] = 0
            frame_ids[pos:pos + sbs] = chunk_fid
            phys_ids[pos:pos + sbs] = chunk_phys_start
            token_types[pos:pos + sbs] = TOKEN_ACTION
            pos += sbs

        for local_idx in range(count):
            phys = chunk_phys_start + local_idx
            noise_ids[pos:pos + abs_] = 0
            frame_ids[pos:pos + abs_] = chunk_fid
            phys_ids[pos:pos + abs_] = phys
            token_types[pos:pos + abs_] = TOKEN_ACTION
            pos += abs_

        noisy_offset += count

    # Padding stays at defaults (noise_id=0, frame_id=-1)

    # --- Pre-compute boolean metadata to avoid ANY Python int constants in
    # compiled mask closures (cutlass/inductor constant-folds scalar tensors
    # back to Python ints, which ssa_to_indexable rejects). ---
    is_clean = (noise_ids == 1)   # bool[padded_total]
    is_noisy = (noise_ids == 0)   # bool[padded_total]
    is_valid = (frame_ids >= 0)   # bool[padded_total]
    is_val = (is_value == 1)      # bool[padded_total]
    is_not_val = (is_value == 0)  # bool[padded_total]
    is_noisy_video = token_types == TOKEN_NOISY_VIDEO
    is_action = token_types == TOKEN_ACTION
    # --- Mask functions ---
    def seq_mask(b, h, q_idx, kv_idx):
        return is_valid[q_idx] & is_valid[kv_idx]

    def block_causal_mask(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] <= frame_ids[q_idx]

    def block_causal_mask_exclude_self(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] < frame_ids[q_idx]

    def block_self_mask(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] == frame_ids[q_idx]

    def clean2clean_mask(b, h, q_idx, kv_idx):
        return is_clean[q_idx] & is_clean[kv_idx]

    def noise2clean_mask(b, h, q_idx, kv_idx):
        return is_noisy[q_idx] & is_clean[kv_idx]

    def noise2noise_mask(b, h, q_idx, kv_idx):
        return is_noisy[q_idx] & is_noisy[kv_idx]

    def block_window_mask(b, h, q_idx, kv_idx, ws: int):
        return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= ws

    def and_masks(*fns):
        def combined(b, h, q, k):
            result = fns[0](b, h, q, k)
            for fn in fns[1:]:
                result = result & fn(b, h, q, k)
            return result
        return combined

    def or_masks(*fns):
        def combined(b, h, q, k):
            result = fns[0](b, h, q, k)
            for fn in fns[1:]:
                result = result | fn(b, h, q, k)
            return result
        return combined

    # clean → clean: causal within window
    rule_cc = and_masks(clean2clean_mask, block_causal_mask)
    # noisy → clean: strictly causal within window (cannot see own clean frame)
    rule_nc = and_masks(noise2clean_mask, block_causal_mask_exclude_self)
    # noisy/action intra-block visibility. In action_only/FDM, action queries
    # cannot see same-chunk noisy video, while video queries may still use action
    # conditions. This avoids action shortcutting through teacher-forced future video.
    def action_to_action(b, h, q_idx, kv_idx):
        return is_action[q_idx] & is_action[kv_idx] & block_self_mask(b, h, q_idx, kv_idx)

    def action_to_action_video(b, h, q_idx, kv_idx):
        allowed_k = is_action[kv_idx] | is_noisy_video[kv_idx]
        return is_action[q_idx] & allowed_k & block_self_mask(b, h, q_idx, kv_idx)

    def video_to_video(b, h, q_idx, kv_idx):
        return is_noisy_video[q_idx] & is_noisy_video[kv_idx] & block_self_mask(b, h, q_idx, kv_idx)

    def video_to_action_video(b, h, q_idx, kv_idx):
        allowed_k = is_action[kv_idx] | is_noisy_video[kv_idx]
        return is_noisy_video[q_idx] & allowed_k & block_self_mask(b, h, q_idx, kv_idx)

    if attention_mode == "video_idm":
        rule_nn = or_masks(action_to_action_video, video_to_video)
    elif attention_mode == "fdm":
        rule_nn = or_masks(action_to_action, video_to_action_video)
    else:
        rule_nn = or_masks(action_to_action_video, video_to_action_video)

    combined = or_masks(rule_cc, rule_nc, rule_nn)
    combined = and_masks(combined, seq_mask)
    combined = and_masks(combined, partial(block_window_mask, ws=window_size))

    # Scheme B: value token read-only mask
    if value_tokens_per_frame > 0:
        def value_exclusion_mask(b, h, q_idx, kv_idx):
            """Non-value queries cannot attend to value keys."""
            return is_val[q_idx] | is_not_val[kv_idx]

        def value_phys_mask(b, h, q_idx, kv_idx):
            """Chunk-level value queries: noisy keys restricted to same chunk."""
            val_q = is_val[q_idx]
            noisy_k = is_noisy[kv_idx]
            same_chunk = frame_ids[q_idx] == frame_ids[kv_idx]
            return (~val_q) | (~noisy_k) | same_chunk

        combined = and_masks(combined, value_exclusion_mask)
        combined = and_masks(combined, value_phys_mask)

    # Add self-attention fallback
    def final_mask(b, h, q_idx, kv_idx):
        return (q_idx == kv_idx) | combined(b, h, q_idx, kv_idx)

    block_mask = compiled_create_block_mask(
        final_mask, 1, 1,
        padded_total, padded_total,
        device=device, _compile=True,
    )
    return block_mask, padded_total


# ---------------------------------------------------------------------------
# Hybrid-memory window mask for training (compiled BlockMask)
# ---------------------------------------------------------------------------

def create_memory_window_mask(
    num_latent_frames: int,
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
    device: str = "cuda",
    obs_latents: int = None,
    separate_anchor: bool = False,
):
    """Compiled FlexAttention BlockMask for the action-first hybrid-memory layout.

    Reproduces :func:`build_memory_visibility` exactly so that the dense CPU
    reference, this training mask, and streaming inference all agree. Layout::

        [ anchor? | clean_video[f] | gist[f] | per-chunk(state?, action, noisy_video, value?) ]

    ``num_latent_frames`` is the full window W; clean/gist cover frames 0..N-1,
    noisy chunks cover physical frames ``obs..W-1``, with ``N = W - obs`` and
    ``obs`` the number of pure-observation latents (legacy ``obs == chunk_size``).
    When ``separate_anchor`` the global-L0 anchor is prepended as identity-RoPE
    tokens that are always visible, and in-window frames no longer act as anchors.
    """
    W = int(num_latent_frames)
    cs = max(1, int(chunk_size))
    obs = cs if obs_latents is None else max(1, int(obs_latents))
    N = W - obs
    vbs = int(video_tokens_per_frame)
    gbs = int(gist_tokens_per_frame)
    abs_ = int(action_tokens_per_frame)
    sbs = int(state_tokens_per_frame) if use_state else 0
    nvpf = int(value_tokens_per_frame)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    n_anchor_tok = anchor_frames if separate_anchor else 0
    mode = normalize_memory_mode(attention_mode)

    def _fid(phys):
        return 0 if phys < obs else 1 + (phys - obs) // cs

    block_latent_counts = []
    offset = 0
    while offset < N:
        block_latent_counts.append(min(cs, N - offset))
        offset += cs

    anchor_len = n_anchor_tok * vbs
    clean_len = N * vbs
    gist_len = N * gbs
    noisy_len = sum(sbs + count * abs_ + count * vbs + nvpf for count in block_latent_counts)
    total = anchor_len + clean_len + gist_len + noisy_len
    padded_total = math.ceil(total / 128) * 128

    token_types = torch.zeros(padded_total, device=device, dtype=torch.long)
    frame_ids = torch.full((padded_total,), -1, device=device, dtype=torch.long)
    # physical latent index (latent units): drives anchor/recent. Causal/intra
    # use frame_ids (chunk id). cs=1 => phys == frame id (legacy behaviour).
    phys_ids = torch.full((padded_total,), -1, device=device, dtype=torch.long)

    pos = 0
    for _a in range(n_anchor_tok):
        token_types[pos:pos + vbs] = TOKEN_ANCHOR
        frame_ids[pos:pos + vbs] = -1
        phys_ids[pos:pos + vbs] = -1
        pos += vbs
    for f in range(N):
        fid = _fid(f)
        token_types[pos:pos + vbs] = TOKEN_CLEAN_VIDEO
        frame_ids[pos:pos + vbs] = fid
        phys_ids[pos:pos + vbs] = f
        pos += vbs
    for f in range(N):
        fid = _fid(f)
        token_types[pos:pos + gbs] = TOKEN_GIST
        frame_ids[pos:pos + gbs] = fid
        phys_ids[pos:pos + gbs] = f
        pos += gbs
    noisy_offset = 0
    for count in block_latent_counts:
        chunk_phys = obs + noisy_offset         # first noisy latent's physical index
        chunk_fid = _fid(chunk_phys)
        action_len = sbs + count * abs_
        if action_len > 0:
            token_types[pos:pos + action_len] = TOKEN_ACTION
            frame_ids[pos:pos + action_len] = chunk_fid
            phys_ids[pos:pos + action_len] = chunk_phys
            pos += action_len
        video_len = count * vbs
        if video_len > 0:
            token_types[pos:pos + video_len] = TOKEN_NOISY_VIDEO
            frame_ids[pos:pos + video_len] = chunk_fid
            phys_ids[pos:pos + video_len] = chunk_phys
            pos += video_len
        if nvpf > 0:
            token_types[pos:pos + nvpf] = TOKEN_VALUE
            frame_ids[pos:pos + nvpf] = chunk_fid
            phys_ids[pos:pos + nvpf] = chunk_phys
            pos += nvpf
        noisy_offset += count

    # Pre-computed boolean metadata (no Python int constants inside the
    # compiled mask closures — inductor rejects scalar-tensor constant folding).
    # Anchor tokens carry frame_id=-1 (like padding) but are real, so validity is
    # keyed on token_type (>0) rather than frame_id.
    is_valid = token_types > 0
    is_anchor = token_types == TOKEN_ANCHOR
    is_clean = token_types == TOKEN_CLEAN_VIDEO
    is_gist = token_types == TOKEN_GIST
    is_action = token_types == TOKEN_ACTION
    is_noisy = token_types == TOKEN_NOISY_VIDEO
    is_value = token_types == TOKEN_VALUE
    is_clean_q = is_clean | is_anchor

    def seq_mask(b, h, q_idx, kv_idx):
        return is_valid[q_idx] & is_valid[kv_idx]

    def hist_mask(b, h, q_idx, kv_idx):
        pq = phys_ids[q_idx]
        pk = phys_ids[kv_idx]
        gist_query = is_gist[q_idx]
        clean_query = is_clean_q[q_idx]
        pred_query = is_action[q_idx] | is_noisy[q_idx] | is_value[q_idx]

        # Clean queries: own + recent/anchor previous fulls + gists pk<=pq
        clean_self_or_recent = (pk == pq) | ((pk < pq) & (pk >= (pq - recent_frames)))
        if separate_anchor:
            clean_full = is_clean[kv_idx] & clean_self_or_recent
        else:
            in_anchor = pk < anchor_frames
            clean_full = is_clean[kv_idx] & (
                (pk == pq) | ((pk < pq) & ((pk >= (pq - recent_frames)) | in_anchor))
            )
        clean_ok = clean_query & (
            clean_full | (is_gist[kv_idx] & (pk <= pq)) | is_anchor[kv_idx]
        )

        # Gist queries: latent-unit own-clean + retained hist + gist chain
        retained_hist = (pk < pq) & (pk >= (pq - recent_frames))
        if not separate_anchor:
            retained_hist = retained_hist | ((pk < pq) & (pk < anchor_frames))
        gist_ok = gist_query & (
            (is_clean[kv_idx] & ((pk == pq) | retained_hist))
            | (is_gist[kv_idx] & (pk <= pq))
            | is_anchor[kv_idx]
        )

        # Action/noisy/value: retained full (recent window + anchor); past gists pk < q0
        pred_ok = pred_query & (
            (is_clean[kv_idx] & retained_hist)
            | (is_gist[kv_idx] & (pk < pq))
            | is_anchor[kv_idx]
        )
        return clean_ok | gist_ok | pred_ok

    def intra_mask(b, h, q_idx, kv_idx):
        same = frame_ids[q_idx] == frame_ids[kv_idx]
        k_act = is_action[kv_idx]
        k_vid = is_noisy[kv_idx]
        k_val = is_value[kv_idx]
        if mode == "video_idm":
            allowed_act = k_act | k_vid
            allowed_vid = k_vid
        elif mode == "fdm":
            allowed_act = k_act
            allowed_vid = k_act | k_vid
        elif mode == "decoupled":
            allowed_act = k_act
            allowed_vid = k_vid
        else:
            allowed_act = k_act | k_vid
            allowed_vid = k_act | k_vid
        allowed_val = k_act | k_vid | k_val
        return same & (
            (is_action[q_idx] & allowed_act)
            | (is_noisy[q_idx] & allowed_vid)
            | (is_value[q_idx] & allowed_val)
        )

    def final_mask(b, h, q_idx, kv_idx):
        return (q_idx == kv_idx) | (
            seq_mask(b, h, q_idx, kv_idx)
            & (hist_mask(b, h, q_idx, kv_idx) | intra_mask(b, h, q_idx, kv_idx))
        )

    block_mask = compiled_create_block_mask(
        final_mask, 1, 1,
        padded_total, padded_total,
        device=device, _compile=True,
    )
    return block_mask, padded_total


def create_packed_memory_window_mask(
    num_latent_frames_list,
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
    device: str = "cuda",
    obs_latents: int = None,
    separate_anchor: bool = False,
):
    """Packed BlockMask for multiple full trajectories in one attention row.

    Compatibility wrapper around ``infra.varlen.block_mask``. The infra layer now
    owns the block-diagonal packed layout metadata and sample-isolating mask
    construction, while this public helper returns the documented output shape.
    """
    layout = build_packed_memory_window_block_mask(
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
    )
    return layout.block_mask, layout.padded_total, layout.segment_token_lengths


# ---------------------------------------------------------------------------
# Flex attention dispatch
# ---------------------------------------------------------------------------

def BHLD_flex_with_mask(q, k, v, attention_mask=None):
    """Attend to contiguous [B, H, L, D] tensors without a layout round trip."""
    kernel_options = {}
    if (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and torch.cuda.get_device_capability(q.device) in {(10, 0), (10, 3)}
    ):
        # Blackwell's newer four-warp heuristic regresses this backward kernel.
        # Eight warps retain the same tiles and arithmetic (including mask),
        # and match the earlier compiler's choice for these training heads.
        kernel_options["bwd_num_warps"] = 8
    return flex_attention_compiled(
        q, k, v, block_mask=attention_mask, kernel_options=kernel_options,
    )


def BLHD_flex_with_mask(
    q,
    k,
    v,
    attention_mask=None,
):
    """Apply compiled flex_attention. Input/output shape: [B, L, H, D]."""
    q, k, v = (rearrange(x, "b l h d -> b h l d").contiguous() for x in (q, k, v))
    o = BHLD_flex_with_mask(q, k, v, attention_mask)
    return rearrange(o, "b h l d -> b l h d")


# ---------------------------------------------------------------------------
# Legacy masks kept for inference prefill
# ---------------------------------------------------------------------------

def get_flex_causal_block_mask_for_prefill(
    num_denoise_block, block_size=16 * 16, device="cuda"
):
    """Causal prefill mask for inference cache filling."""
    seq_len = num_denoise_block * block_size
    def block_mask_mod(b, h, q_idx, kv_idx):
        bqi = q_idx // block_size
        bki = kv_idx // block_size
        return bqi >= bki
    return create_block_mask(
        block_mask_mod, None, None,
        Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True
    )
