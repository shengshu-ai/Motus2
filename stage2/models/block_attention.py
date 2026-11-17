# Flex-attention masks for sliding-window AR teacher-forcing training.
# joint video-action prediction (no clean action).

import math
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from einops import rearrange

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
flex_attention_compiled_static = torch.compile(
    flex_attention,
    dynamic=False,
    fullgraph=True,
)
compiled_create_block_mask = torch.compile(create_block_mask)


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

    Supported visibility modes:
      - joint: legacy current behavior. Video/action noisy tokens in the same
        chunk can see each other; value is read-only.
      - video_idm: action can see same-chunk video, but video cannot see action.
      - action_only/fdm: video can see same-chunk action, but action cannot see video.

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
        use_state: (deprecated as gate; kept for backward-compat) — clean-prefix-token
            count is now driven entirely by `state_tokens_per_frame`.
        state_tokens_per_frame: number of clean prefix tokens per chunk
            (qpos state). 0 means no clean prefix.
        device: target device.

    Returns:
        (block_mask, padded_total): FlexAttention BlockMask and padded sequence length.
    """
    W = num_latent_frames
    N = W - chunk_size  # frames per side; clean 0..N-1, noisy cs..W-1
    vbs = video_tokens_per_frame
    abs_ = action_tokens_per_frame
    # state_tokens_per_frame is the single source of truth for clean prefix-token count per chunk

    sbs = state_tokens_per_frame
    attention_mode = str(attention_mode or "joint").lower()
    if attention_mode not in {"joint", "video_idm", "action_only", "fdm"}:
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    block_latent_counts = []
    offset = 0
    while offset < N:
        count = min(chunk_size, N - offset)
        block_latent_counts.append(count)
        offset += count

    return _create_action_first_chunk_ar_mask(
        N=N,
        vbs=vbs,
        abs_=abs_,
        sbs=sbs,
        block_latent_counts=block_latent_counts,
        chunk_size=chunk_size,
        window_size=window_size,
        value_tokens_per_frame=value_tokens_per_frame,
        attention_mode=attention_mode,
        device=device,
    )


def _create_action_first_chunk_ar_mask(
    N: int,
    vbs: int,
    abs_: int,
    sbs: int,
    block_latent_counts,
    chunk_size: int,
    window_size: int,
    value_tokens_per_frame: int,
    attention_mode: str,
    device: str,
):
    """Create chunk-AR mask for [clean | per-chunk(action | noisy_video | value)]."""
    attention_mode = str(attention_mode or "joint").lower()
    if attention_mode == "action_only":
        attention_mode = "fdm"
    noisy_len = sum(sbs + count * abs_ + count * vbs + value_tokens_per_frame for count in block_latent_counts)
    total = N * vbs + noisy_len
    padded_total = math.ceil(total / 128) * 128

    token_types = torch.zeros(padded_total, device=device, dtype=torch.long)
    frame_ids = torch.full((padded_total,), -1, device=device, dtype=torch.long)

    TOKEN_CLEAN_VIDEO = 1
    TOKEN_ACTION = 2
    TOKEN_NOISY_VIDEO = 3
    TOKEN_VALUE = 4

    pos = 0
    for f in range(N):
        fid = f // chunk_size
        token_types[pos:pos + vbs] = TOKEN_CLEAN_VIDEO
        frame_ids[pos:pos + vbs] = fid
        pos += vbs

    noisy_offset = 0
    for count in block_latent_counts:
        chunk_phys_start = chunk_size + noisy_offset
        chunk_fid = chunk_phys_start // chunk_size

        action_len = sbs + count * abs_
        if action_len > 0:
            token_types[pos:pos + action_len] = TOKEN_ACTION
            frame_ids[pos:pos + action_len] = chunk_fid
            pos += action_len

        video_len = count * vbs
        token_types[pos:pos + video_len] = TOKEN_NOISY_VIDEO
        frame_ids[pos:pos + video_len] = chunk_fid
        pos += video_len

        if value_tokens_per_frame > 0:
            token_types[pos:pos + value_tokens_per_frame] = TOKEN_VALUE
            frame_ids[pos:pos + value_tokens_per_frame] = chunk_fid
            pos += value_tokens_per_frame

        noisy_offset += count

    is_valid = frame_ids >= 0
    is_clean_video = token_types == TOKEN_CLEAN_VIDEO
    is_action = token_types == TOKEN_ACTION
    is_noisy_video = token_types == TOKEN_NOISY_VIDEO
    is_value = token_types == TOKEN_VALUE
    is_pred = is_action | is_noisy_video | is_value

    def seq_mask(b, h, q_idx, kv_idx):
        return is_valid[q_idx] & is_valid[kv_idx]

    def same_chunk(b, h, q_idx, kv_idx):
        return frame_ids[q_idx] == frame_ids[kv_idx]

    def past_chunk_only(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] < frame_ids[q_idx]

    def chunk_window_mask(b, h, q_idx, kv_idx, ws: int):
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

    def clean_to_clean(b, h, q_idx, kv_idx):
        return is_clean_video[q_idx] & is_clean_video[kv_idx] & (frame_ids[kv_idx] <= frame_ids[q_idx])

    def pred_to_clean(b, h, q_idx, kv_idx):
        return is_pred[q_idx] & is_clean_video[kv_idx] & past_chunk_only(b, h, q_idx, kv_idx)

    def action_to_action(b, h, q_idx, kv_idx):
        return is_action[q_idx] & is_action[kv_idx] & same_chunk(b, h, q_idx, kv_idx)

    def action_to_action_video(b, h, q_idx, kv_idx):
        allowed_k = is_action[kv_idx] | is_noisy_video[kv_idx]
        return is_action[q_idx] & allowed_k & same_chunk(b, h, q_idx, kv_idx)

    def video_to_video(b, h, q_idx, kv_idx):
        return is_noisy_video[q_idx] & is_noisy_video[kv_idx] & same_chunk(b, h, q_idx, kv_idx)

    def video_to_action_video(b, h, q_idx, kv_idx):
        allowed_k = is_action[kv_idx] | is_noisy_video[kv_idx]
        return is_noisy_video[q_idx] & allowed_k & same_chunk(b, h, q_idx, kv_idx)

    def value_to_chunk_all(b, h, q_idx, kv_idx):
        allowed_k = is_action[kv_idx] | is_noisy_video[kv_idx] | is_value[kv_idx]
        return is_value[q_idx] & allowed_k & same_chunk(b, h, q_idx, kv_idx)

    if attention_mode == "video_idm":
        combined = or_masks(clean_to_clean, pred_to_clean, action_to_action_video, video_to_video, value_to_chunk_all)
    elif attention_mode == "fdm":
        combined = or_masks(clean_to_clean, pred_to_clean, action_to_action, video_to_action_video, value_to_chunk_all)
    else:
        combined = or_masks(clean_to_clean, pred_to_clean, action_to_action_video, video_to_action_video, value_to_chunk_all)
    combined = and_masks(combined, seq_mask)
    combined = and_masks(combined, partial(chunk_window_mask, ws=window_size))

    def final_mask(b, h, q_idx, kv_idx):
        return (q_idx == kv_idx) | combined(b, h, q_idx, kv_idx)

    block_mask = compiled_create_block_mask(
        final_mask, 1, 1,
        padded_total, padded_total,
        device=device, _compile=True,
    )
    return block_mask, padded_total


# ---------------------------------------------------------------------------
# Flex attention dispatch
# ---------------------------------------------------------------------------

def _training_backward_kernel_options(q, blackwell_backward: bool):
    """Select the measured Blackwell backward schedule only when requested.

    Keep the default call unchanged, including on unsupported hardware/dtypes.
    This does not alter the compiler's dynamic/static policy or math precision.
    """
    if (
        blackwell_backward
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and torch.cuda.get_device_capability(q.device) in {(10, 0), (10, 3)}
    ):
        return {"bwd_num_warps": 8}
    return {}


def BLHD_flex_with_mask(
    q,
    k,
    v,
    attention_mask=None,
    *,
    static_shapes: bool = False,
    blackwell_backward: bool = False,
):
    """Apply compiled flex_attention. Input/output shape: [B, L, H, D]."""
    q, k, v = (rearrange(x, "b l h d -> b h l d").contiguous() for x in (q, k, v))
    compiled = (
        flex_attention_compiled_static
        if static_shapes
        else flex_attention_compiled
    )
    kernel_options = _training_backward_kernel_options(q, blackwell_backward)
    if kernel_options:
        o = compiled(q, k, v, block_mask=attention_mask, kernel_options=kernel_options)
    else:
        o = compiled(q, k, v, block_mask=attention_mask)
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
