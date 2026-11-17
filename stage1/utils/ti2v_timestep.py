"""Per-token diffusion timesteps for WAN TI2V-style conditioning.

Condition latent frames use discrete timestep 0 (sigma=0 in WAN convention), including when
latents are augmented with extra Gaussian noise (time embedding stays 0; see wan_new Motus).
"""

from __future__ import annotations

import torch


def build_ti2v_per_token_timestep(
    timestep: torch.Tensor,
    n_cond_latent_frames,
    seq_len: int,
    patch_size: tuple,
    latent_shape: tuple,
) -> torch.Tensor:
    """Build [B, seq_len] timesteps: cond temporal patches -> 0, else -> timestep.

    Args:
        timestep: [B] current diffusion timestep (same units as WAN training, e.g. sigma * 1000).
        n_cond_latent_frames: Number of leading latent time steps treated as conditioning.
            Can be a scalar ``int`` or a ``[B]`` tensor for per-sample conditioning.
        seq_len: Padded sequence length passed to WanModel (F_pat * H_pat * W_pat).
        patch_size: (t_patch, h_patch, w_patch) from WanModel.
        latent_shape: (C, T, H, W) for one sample (T = latent frames along time).

    Returns:
        Tensor [B, seq_len] suitable for WanModel.forward(t=...).
    """
    B = timestep.shape[0]
    device = timestep.device
    dtype = timestep.dtype
    _, T, H, W = latent_shape
    pt, ph, pw = patch_size

    F_pat = T // pt
    H_pat = H // ph
    W_pat = W // pw
    tokens_per_temporal_patch = H_pat * W_pat
    expected = F_pat * tokens_per_temporal_patch
    if expected != seq_len:
        raise ValueError(
            f"seq_len mismatch: got seq_len={seq_len}, from latent {tuple(latent_shape)} "
            f"and patch_size={patch_size} expected {expected}"
        )

    if isinstance(n_cond_latent_frames, int):
        n_cond = torch.full((B,), int(n_cond_latent_frames), device=device, dtype=torch.long)
    elif isinstance(n_cond_latent_frames, torch.Tensor):
        n_cond = n_cond_latent_frames.to(device=device, dtype=torch.long).reshape(B)
    else:
        raise TypeError(
            "n_cond_latent_frames must be an int or torch.Tensor, "
            f"got {type(n_cond_latent_frames)!r}"
        )

    out = timestep.view(B, 1).expand(B, seq_len).clone()
    for f in range(F_pat):
        t_start = f * pt
        t_end = f * pt + pt - 1
        t0 = f * tokens_per_temporal_patch
        t1 = (f + 1) * tokens_per_temporal_patch
        full_cond_mask = t_end < n_cond
        if full_cond_mask.any():
            out[full_cond_mask, t0:t1] = 0
        if (t_start >= n_cond).all():
            break
        # straddling patch: leave as `timestep` (rare if patch_size[0] > 1 and small n_cond)
    return out.to(device=device, dtype=dtype)
