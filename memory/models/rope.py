# RoPE utilities combining:
#   - 3D video RoPE (rope_params, rope_apply_one)
#   - 1D action RoPE (rope_action_apply, causal_rope_action_apply)

import torch
import torch.amp


def _identity_freqs(length: int, ref: torch.Tensor) -> torch.Tensor:
    """Identity complex frequencies: applying them leaves tokens unchanged."""
    return torch.ones((length, 1, ref.shape[-1]), device=ref.device, dtype=ref.dtype)


# ---------------------------------------------------------------------------
# Frequency table generation
# ---------------------------------------------------------------------------

@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """Generate complex-valued RoPE frequency table."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def video_basis_temporal_dim(dim: int) -> int:
    """Channel count of the temporal block in the [t | h | w] 3D video split."""
    c = dim // 2
    return c - 2 * (c // 3)


@torch.amp.autocast('cuda', enabled=False)
def rope_video_basis_freqs(positions, dim, theta=10000, device=None, dtype=None):
    """Place non-video tokens on the video 3D-RoPE clock at spatial origin."""
    assert dim % 2 == 0
    c = dim // 2
    t_dim = video_basis_temporal_dim(dim)
    spectrum_dim = 2 * t_dim
    pos = torch.as_tensor(positions, device=device, dtype=torch.float64).reshape(-1)
    inv_freq = 1.0 / torch.pow(
        torch.as_tensor(float(theta), dtype=torch.float64, device=pos.device),
        torch.arange(0, spectrum_dim, 2, device=pos.device, dtype=torch.float64).div(spectrum_dim),
    )
    angles = torch.outer(pos, inv_freq[:t_dim])
    freqs = torch.ones((pos.shape[0], c), device=pos.device, dtype=torch.complex128)
    freqs[:, :t_dim] = torch.polar(torch.ones_like(angles), angles)
    freqs = freqs.unsqueeze(1)
    return freqs if dtype is None else freqs.to(dtype=dtype)


# ---------------------------------------------------------------------------
# 3D Video RoPE
# ---------------------------------------------------------------------------

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_one(x, grid_size, freqs_in, block_idx=None, block_size=None):
    """Apply 3D RoPE to a batch of video tokens.

    Args:
        x: [B, L, H, D] query or key tensor.
        grid_size: [f, h, w] spatial-temporal grid dimensions.
        freqs_in: precomputed frequency table from rope_params.
        block_idx: if set, select freqs for this specific temporal block.
        block_size: tokens per temporal block (required when block_idx is set).

    Returns:
        RoPE-applied tensor [B, L, H, D].
    """
    B, L, H, D = x.shape
    n, c = H, D // 2
    freqs = freqs_in.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_size.tolist()
    if block_idx is not None:
        f = max(f, block_idx + 1)
    seq_len = f * h * w

    x_re = x.permute(1, 0, 2, 3)  # [L, B, H, D]
    x_c = torch.view_as_complex(x_re.to(torch.float64).reshape(L, n * B, -1, 2))

    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, 1, -1)

    if block_idx is not None:
        freqs_i = freqs_i[block_idx * block_size:(block_idx + 1) * block_size]

    out = torch.view_as_real(x_c * freqs_i).flatten(2).reshape(L, B, H, D)
    return out.permute(1, 0, 2, 3).float()


# ---------------------------------------------------------------------------
# 1D Action RoPE — training
# ---------------------------------------------------------------------------

@torch.amp.autocast('cuda', enabled=False)
def rope_action_apply(x, freqs, freqs_action, freqs_state,
                      action_register_length=None,
                      num_action_per_block=None,
                      num_state_per_block=None,
                      num_value_per_block=None):
    """Apply RoPE to [video | action] or [video | state | action] sequence during training.

    This function unifies RoPE application for video (3D) and action/state (1D) tokens.
    Current policy semantics:
      - state tokens do NOT use RoPE
      - action tokens use chunk-local 1D RoPE

    Args:
        x: [B, seq_len, n_heads, head_dim] input tensor (already in [B, L, H, D] format)
        freqs: [video_seq_len, 1, head_dim//2] complex video frequencies (3D RoPE)
        freqs_action: [max_action_tokens, 1, head_dim//2] complex action frequencies (1D RoPE)
        freqs_state: [max_state_tokens, 1, head_dim//2] complex state frequencies (1D RoPE)
        action_register_length: Total length of action+state tokens (None if no action tokens)
        num_action_per_block: Action tokens per chunk (K * action_per_latent_frame)
        num_state_per_block: State tokens per chunk (1 if use_state, else 0)

    Returns:
        RoPE-applied tensor [B, seq_len, n_heads, head_dim]

    Token structure (StateActionEncoder, use_state=True):
        [video | state | action1..actionN] — state is 1 token at start, then M*action_chunk_size action tokens
        So: num_state_tokens=1, num_action_tokens=action_register_length-1
    Token structure (ActionEncoder, use_state=False):
        [video | action1..actionN] — no state, num_action_tokens=action_register_length
    """
    B, seq_len, n, _ = x.shape

    # Convert to complex: [B, seq_len, n_heads, head_dim] -> [B, seq_len, n_heads, head_dim//2] (complex)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None and num_state_per_block is not None
        # Token order: [value | state | action]
        num_value_tokens = num_value_per_block if (num_value_per_block and num_value_per_block > 0) else 0
        num_state_tokens = num_state_per_block if num_state_per_block > 0 else 0
        num_action_tokens = action_register_length - num_state_tokens - num_value_tokens

        parts = []
        if num_value_tokens > 0:
            parts.append(_identity_freqs(num_value_tokens, freqs_state))
        if num_state_tokens > 0:
            parts.append(_identity_freqs(num_state_tokens, freqs_state))
        parts.append(freqs_action[:num_action_tokens].view(num_action_tokens, 1, freqs_action.shape[-1]))
        # Order matches: [video | value | state | action]
        freqs = torch.cat([freqs] + parts, dim=0)

    # Apply RoPE: x_complex * freqs (broadcast)
    # freqs can be [seq_len, 1, C] (shared across batch) or [B, seq_len, 1, C] (per-sample)
    if freqs.dim() == 3:
        # Pad shared freqs if x was padded for block_mask alignment (flex_attention 128 multiple)
        if freqs.shape[0] < seq_len:
            pad_len = seq_len - freqs.shape[0]
            freqs_pad = freqs[-1:].expand(pad_len, *freqs.shape[1:]).clone()
            freqs = torch.cat([freqs, freqs_pad], dim=0)
        freqs = freqs.unsqueeze(0)  # [1, seq_len, 1, C]
    elif freqs.dim() != 4:
        raise ValueError(f"Unsupported freqs shape: {tuple(freqs.shape)}")

    # Normalize possible accidental [seq_len, B, 1, C] layout to [B, seq_len, 1, C].
    if freqs.dim() == 4 and freqs.shape[0] != B:
        if freqs.shape[1] == B:
            freqs = freqs.permute(1, 0, 2, 3).contiguous()

    # Pad batched freqs along sequence dim if x was padded.
    if freqs.dim() == 4 and freqs.shape[1] < seq_len:
        pad_len = seq_len - freqs.shape[1]
        freqs_pad = freqs[:, -1:, :, :].expand(freqs.shape[0], pad_len, freqs.shape[2], freqs.shape[3]).clone()
        freqs = torch.cat([freqs, freqs_pad], dim=1)

    if freqs.shape[0] not in (1, B) or freqs.shape[1] != seq_len:
        raise ValueError(
            f"RoPE freqs/x shape mismatch: x={tuple(x.shape)}, freqs={tuple(freqs.shape)}"
        )
    out = torch.view_as_real(x_complex * freqs).flatten(3)  # [B, seq_len, n_heads, head_dim]
    return out


# ---------------------------------------------------------------------------
# 1D Action RoPE — inference
# ---------------------------------------------------------------------------

@torch.amp.autocast('cuda', enabled=False)
def causal_rope_action_apply(x, freqs, freqs_action, freqs_state,
                             action_register_length=None,
                             num_action_per_block=None,
                             num_state_per_block=None,
                             action_offset=0,
                             state_offset=0,
                             num_value_per_block=None,
                             action_freqs_override=None):
    """Apply RoPE to [video | action | state] for a single inference chunk.

    This is the inference version of rope_action_apply. Current policy semantics:
      - state tokens do NOT use RoPE
      - action tokens use chunk-local 1D RoPE and reset every chunk
      - value tokens do NOT use RoPE
      - video tokens keep global temporal RoPE

    Args:
        x: [B, seq_len, n_heads, head_dim] input tensor for current chunk
        freqs: [video_seq_len, 1, head_dim//2] complex video frequencies for current chunk
        freqs_action: [max_action_tokens, 1, head_dim//2] complex action frequencies
        freqs_state: [max_state_tokens, 1, head_dim//2] complex state frequencies
        action_register_length: Should equal (num_action_per_block + num_state_per_block) for single chunk
        num_action_per_block: Action tokens per chunk
        num_state_per_block: State tokens per chunk
        action_offset: Cumulative action token offset (unused for chunk-local RoPE, kept for API compat)
        state_offset: Cumulative state token offset (unused for chunk-local RoPE, kept for API compat)
        num_value_per_block: Number of value tokens per chunk (identity freqs)

    Returns:
        RoPE-applied tensor [B, seq_len, n_heads, head_dim]

    How it works:
        1. Convert x to complex
        2. If action_register_length is set:
           - action tokens always use freqs_action[:num_action_per_block]
           - state tokens use identity frequencies
           - each chunk is encoded in its own local coordinates
        3. Concatenate: freqs = [video_freqs | action_freqs_slice | state_freqs_slice]
        4. Apply RoPE: x_complex * freqs

    Example:
        Chunk k: action tokens use positions [0 : num_action_per_block] again.
    """
    B, seq_len, n, _ = x.shape

    # Convert to complex: [B, seq_len, n_heads, head_dim] -> [B, seq_len, n_heads, head_dim//2] (complex)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None and num_state_per_block is not None
        # Token order: [value | state | action]
        num_value_tokens = num_value_per_block if (num_value_per_block and num_value_per_block > 0) else 0
        num_state_tokens = num_state_per_block if num_state_per_block > 0 else 0
        num_action_tokens = action_register_length - num_state_tokens - num_value_tokens

        # For inference, action_register_length should equal single chunk size
        assert action_register_length == (num_value_tokens + num_state_tokens + num_action_tokens), \
            f"action_register_length ({action_register_length}) mismatch"

        # Select frequency slice for current chunk
        parts = []
        if num_value_tokens > 0:
            parts.append(_identity_freqs(num_value_tokens, freqs_state).squeeze(1))
        if num_state_tokens > 0:
            parts.append(_identity_freqs(num_state_tokens, freqs_state).squeeze(1))
        if action_freqs_override is not None:
            if action_freqs_override.shape[0] != num_action_tokens:
                raise ValueError(
                    f"action_freqs_override has {action_freqs_override.shape[0]} rows "
                    f"but the chunk holds {num_action_tokens} action tokens"
                )
            parts.append(
                action_freqs_override.reshape(num_action_tokens, -1).to(freqs_state.dtype)
            )
        else:
            parts.append(freqs_action[:num_action_tokens])

        # Order: [value | state | action]
        freqs_1d = torch.cat(parts, dim=0).view(
            action_register_length, 1, freqs_action.shape[-1]
        )
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # Apply RoPE: x_complex * freqs (broadcast)
    freqs = freqs.unsqueeze(0)  # [1, seq_len, 1, head_dim//2]
    out = torch.view_as_real(x_complex * freqs).flatten(3)  # [B, seq_len, n_heads, head_dim]
    return out
