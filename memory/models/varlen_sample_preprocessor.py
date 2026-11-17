"""Motus packed-varlen per-sample preparation.

This keeps the model's packed training loop focused on batch-level assembly and
transformer execution. Token/noise/action stream preparation still calls the
Motus instance because patch embedding, action encoding, timestep embeddings and
RoPE helpers are model-owned modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .memory_rope import extend_valid_mask, pad_window_to_chunk, repeat_last


@dataclass
class PreparedVarlenSample:
    stream: Dict[str, torch.Tensor]
    timestep_stream: Dict[str, torch.Tensor]
    freq_part: Optional[torch.Tensor]
    segment_info: Dict[str, Any]
    pred_targets: Tuple[torch.Tensor, torch.Tensor]
    window_length: int


class MotusVarlenSamplePreprocessor:
    """Prepare one full-trajectory sample for packed varlen training."""

    def __init__(self, model):
        self.model = model

    def prepare(
        self,
        *,
        latents_i: torch.Tensor,
        actions_i: torch.Tensor,
        action_valid_mask_i: Optional[torch.Tensor],
        video_valid_mask_i: Optional[torch.Tensor],
        states_i: Optional[torch.Tensor],
        values_i: Optional[torch.Tensor],
        window_length: int,
        valid_len: int,
        latent_start: int,
        sigma_v: torch.Tensor,
        sigma_a: torch.Tensor,
        noisy_vid_t: torch.Tensor,
        noisy_act_t: torch.Tensor,
        rope_self_check: bool,
    ) -> PreparedVarlenSample:
        m = self.model
        device = m.device
        obs = m._obs_latents()
        chunk_size = m._fixed_chunk_size()
        apl = int(m.config.actions_per_latent)

        lat_i = latents_i.unsqueeze(0).to(device=device, dtype=m.dtype)
        act_i = actions_i.unsqueeze(0).to(device=device, dtype=torch.float32)
        action_valid_i = (
            None
            if action_valid_mask_i is None
            else action_valid_mask_i.to(device=device, dtype=torch.bool)
        )
        video_valid_i = (
            None
            if video_valid_mask_i is None
            else video_valid_mask_i.to(device=device, dtype=torch.bool)
        )
        state_i = None if states_i is None else states_i.unsqueeze(0).to(device=device, dtype=torch.float32)

        W_real = int(window_length)
        W_i, N_i = pad_window_to_chunk(W_real, obs, chunk_size)
        if N_i < chunk_size:
            raise ValueError(
                f"packed varlen sample has W={window_length}, obs={obs}, cs={chunk_size} "
                f"→ N={N_i} (need >= cs)"
            )
        pad = W_i - W_real
        if pad:
            if action_valid_i is None:
                action_valid_i = torch.ones(W_real * apl, dtype=torch.bool, device=device)
            if video_valid_i is None:
                video_valid_i = torch.ones(W_real, dtype=torch.bool, device=device)
            lat_i = repeat_last(lat_i, 2, pad)
            act_i = repeat_last(act_i, 1, pad * apl)
            action_valid_i = extend_valid_mask(action_valid_i, pad * apl)
            video_valid_i = extend_valid_mask(video_valid_i, pad)
            state_i = repeat_last(state_i, 1, pad)
            values_i = repeat_last(values_i, 0, pad)
        valid_len = min(int(valid_len), W_real)
        valid_N_i = max(0, valid_len - obs)

        clean_latents = lat_i[:, :, :N_i]
        clean_latents = m._apply_self_forcing_clean_latents(clean_latents)
        noisy_latents = lat_i[:, :, obs:W_i]
        action_block_info = m._prepare_noisy_action_blocks(
            act_i, state_i, chunk_size, dtype=m.dtype, obs_latents=obs
        )
        noisy_actions = action_block_info["flat_actions"]
        noisy_states = action_block_info["states"]
        block_latent_counts = action_block_info["block_latent_counts"]
        block_action_token_lengths = action_block_info["block_action_token_lengths"]
        target_actions_raw = noisy_actions.clone()
        noisy_action_valid_mask = (
            None
            if action_valid_i is None
            else action_valid_i[obs * apl : W_i * apl]
        )
        noisy_video_valid_mask = (
            None
            if video_valid_i is None
            else video_valid_i[obs:W_i]
        )

        noise_v = torch.randn_like(noisy_latents)
        sv = sigma_v.view(1, 1, 1, 1, 1)
        noisy_video = (1 - sv) * noisy_latents + sv * noise_v
        target_video = noise_v - noisy_latents

        noise_a = torch.randn_like(noisy_actions)
        sa = sigma_a.view(1, 1, 1)
        noisy_action = (1 - sa) * noisy_actions + sa * noise_a
        target_action = noise_a - target_actions_raw

        clean_video_tokens = m._video_patch_embed(clean_latents)
        noisy_video_tokens = m._video_patch_embed(noisy_video)
        noisy_action_chunks = m._split_flat_action_chunks(noisy_action, block_action_token_lengths)
        action_tokens = m._encode_action_chunk_tokens(noisy_action_chunks, noisy_states)
        action_tokens = m.action_expert.projection.project_in(action_tokens)

        value_query = None
        if m.config.value_enabled and m.config.value_tokens_per_frame > 0:
            num_chunks = len(block_latent_counts)
            value_query = m.value_tokens.expand(1, num_chunks)

        gist_tokens = m.gist_tokens.expand(1, N_i)
        chunk_ar_tokens = m._interleave_memory_chunk_ar_tokens(
            action_tokens, noisy_video_tokens, block_latent_counts, value_tokens=value_query,
        )
        include_value = value_query is not None
        stream = {
            "clean_video": clean_video_tokens,
            "gist": gist_tokens,
            "chunk_ar": chunk_ar_tokens,
        }
        anchor_len = 0

        clean_t_vals = torch.zeros(1, clean_video_tokens.shape[1], dtype=m.dtype, device=device)
        noisy_v_t_vals = noisy_vid_t.view(1, 1).expand(-1, clean_video_tokens.shape[1])
        # The output heads consume embeddings only; AdaLN for the transformer
        # is computed once from the assembled packed timestep stream.
        v_emb_full_i = m._video_time_embedding(torch.cat([clean_t_vals, noisy_v_t_vals], dim=1))
        sa_t_vals = m._build_chunked_action_timesteps(
            noisy_act_t.view(1), block_latent_counts, include_value=False
        )
        a_emb_i = m._action_time_embedding(sa_t_vals)
        gist_t_vals = torch.zeros(1, gist_tokens.shape[1], dtype=m.dtype, device=device)
        chunk_ar_t_vals = m._build_memory_chunk_ar_timesteps(
            noisy_act_t.view(1), noisy_vid_t.view(1), block_latent_counts, include_value=include_value,
        )
        timestep_stream = {
            "clean_video": clean_t_vals,
            "gist": gist_t_vals,
            "chunk_ar": chunk_ar_t_vals,
        }
        freq_part = None
        if rope_self_check:
            clean_video_freqs = m._create_video_freqs(N_i, start_frame=int(latent_start))
            noisy_video_freqs = m._create_video_freqs(
                N_i, start_frame=int(latent_start) + obs
            )
            freq_part = m._build_memory_full_freqs(
                clean_video_freqs, noisy_video_freqs, block_latent_counts,
                noisy_start=int(latent_start) + obs,
            )
        segment_info = {
            "anchor_len": anchor_len,
            "clean_len": clean_video_tokens.shape[1],
            "gist_len": gist_tokens.shape[1],
            "chunk_ar_len": chunk_ar_tokens.shape[1],
            "block_latent_counts": block_latent_counts,
            "block_action_token_lengths": block_action_token_lengths,
            "N": N_i,
            "valid_N": valid_N_i,
            "action_valid_mask": noisy_action_valid_mask,
            "video_valid_mask": noisy_video_valid_mask,
            "latent_start": int(latent_start),
            "v_emb_full": v_emb_full_i,
            "a_emb": a_emb_i,
            "num_chunks": len(block_latent_counts),
            "values": None if values_i is None else values_i,
        }

        return PreparedVarlenSample(
            stream=stream,
            timestep_stream=timestep_stream,
            freq_part=freq_part,
            segment_info=segment_info,
            pred_targets=(target_video, target_action),
            window_length=W_i,
        )
