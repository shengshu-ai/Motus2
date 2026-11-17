#!/usr/bin/env python3
"""Evaluation for sliding-window AR training.

During training eval (e.g. every 500 steps):
  - No KV cache, no streaming inference
  - Load pre-computed latents from dataset, use clean half as conditioning
  - Predict noisy video + action with num_inference_steps denoising
  - Separate evaluation for Policy and WM modes
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.set_loglevel("WARNING")
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Optional
from collections import defaultdict
import logging

from data.dataset import materialize_varlen_eval_batch
from models.block_attention import create_sliding_window_mask

from utils.evaluation_metrics import append_metrics, finalize_metrics

logger = logging.getLogger(__name__)


def _compute_action_metrics(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    valid_len: Optional[torch.Tensor],
    actions_per_latent: int,
) -> Dict[str, float]:
    """Compute action MSE and MAE."""
    B = pred_actions.shape[0]
    mse_per_sample, mae_per_sample = [], []
    for i in range(B):
        vl = int(valid_len[i].item()) if valid_len is not None else pred_actions.shape[1] // actions_per_latent
        n = min(vl * actions_per_latent, pred_actions.shape[1], gt_actions.shape[1])
        pred_i = pred_actions[i, :n].float().flatten()
        gt_i = gt_actions[i, :n].float().flatten()
        if pred_i.numel() == 0:
            continue
        mse_per_sample.append(F.mse_loss(pred_i, gt_i, reduction="mean").item())
        mae_per_sample.append(F.l1_loss(pred_i, gt_i, reduction="mean").item())
    if not mse_per_sample:
        return {}
    return {
        "action_mse_loss": float(np.mean(mse_per_sample)),
        "action_mae_loss": float(np.mean(mae_per_sample)),
        "_count/action_mse_loss": len(mse_per_sample),
        "_count/action_mae_loss": len(mae_per_sample),
    }


def _compute_value_metrics(
    pred_values: torch.Tensor,
    gt_values: torch.Tensor,
    valid_len: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Compute value prediction MSE and MAE."""
    B, N = pred_values.shape
    mse_list, mae_list = [], []
    for i in range(B):
        vl = int(valid_len[i].item()) if valid_len is not None else N
        vl = min(vl, N)
        if vl == 0:
            continue
        p = pred_values[i, :vl].float()
        g = gt_values[i, :vl].float()
        mse_list.append(F.mse_loss(p, g, reduction="mean").item())
        mae_list.append(F.l1_loss(p, g, reduction="mean").item())
    if not mse_list:
        return {}
    return {
        "value_mse": float(np.mean(mse_list)),
        "value_mae": float(np.mean(mae_list)),
        "_count/value_mse": len(mse_list),
        "_count/value_mae": len(mae_list),
    }


@torch.no_grad()
def inference_sample(model, batch, config, mode="policy", full_denoise: bool = False):
    """Run inference on a batch for evaluation.

    Matches the new training layout:
      clean_vid: frames 0..cs-1 (observation conditioning)
      noisy_vid: frames cs..W-1 (prediction targets, denoised iteratively)
      noisy_act: frames cs..W-1

    Returns:
        video_pred: [B, C, N, H, W] predicted video latents for frames cs..W-1
        action_pred: [B, N * actions_per_latent, action_dim] predicted actions
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = model.dtype

    latents = batch["latents"].to(device, dtype=dtype)  # [B, C, W, H_lat, W_lat]
    B, C_lat, W, H_lat, W_lat = latents.shape
    apl = model.config.actions_per_latent

    actions = batch["actions"].to(device, dtype=torch.float32)
    states = batch.get("states")
    if states is not None:
        states = states.to(device, dtype=torch.float32)
    lang = batch.get("language_embedding")
    if lang is not None:
        lang = lang.to(device, dtype=dtype)

    num_steps = config.model.inference.num_inference_timesteps
    chunk_size = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))

    cs = chunk_size
    obs = model._obs_latents()
    N = W - obs
    clean_latents = latents[:, :, :N]                # phys 0..W-obs-1
    anchor_latents = batch.get("anchor_latents")
    if anchor_latents is not None:
        anchor_latents = anchor_latents.to(device, dtype=dtype)
    action_block_info = model._prepare_noisy_action_blocks(actions, states, cs, dtype=dtype, obs_latents=obs)
    noisy_actions_gt = action_block_info["flat_actions"]
    noisy_states = action_block_info["states"]
    block_latent_counts = action_block_info["block_latent_counts"]
    block_action_token_lengths = action_block_info["block_action_token_lengths"]
    noisy_video = torch.randn(B, C_lat, N, H_lat, W_lat, device=device, dtype=dtype)
    if mode == "wm":
        noisy_action = noisy_actions_gt.clone()
    else:
        noisy_action = torch.randn_like(noisy_actions_gt, dtype=dtype)

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    # solver_order=1 avoids torch.linalg.solve which can OOM during training eval
    sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False, solver_order=1)
    sched_action = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False, solver_order=1)
    sched_video.set_timesteps(num_steps, device=device, shift=5.0)
    sched_action.set_timesteps(num_steps, device=device, shift=5.0)

    if mode == "policy" and model.config.decouple_inference_noise and not full_denoise:
        sigma_max = sched_video.sigmas[0].item()
        final_noise = model.config.video_inference_final_noise
        sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
        sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)

    # Match training: action-first hybrid-memory mask for the WAN path, legacy
    # video-first sliding-window mask for the (unused) MoT path.
    if not model.config.use_mot:
        ar_mask, padded_len = model._create_training_mask(W, chunk_size)
    else:
        ar_mask, padded_len = model._create_legacy_mot_mask(W, chunk_size)

    napf = model.config.noisy_action_tokens_per_frame
    # napf for action decoder (state+action only, no value tokens)
    napf_sa = model.config.state_tokens_per_frame + model.config.action_tokens_per_frame
    t5_ctx = model._t5_embed(lang) if lang is not None else None

    latent_start = batch.get("latent_start")
    if latent_start is not None:
        latent_start = latent_start.to(device=device, dtype=torch.long)
        clean_start = latent_start
        noisy_start = latent_start + obs
        clean_video_freqs = model._create_video_freqs_batch(N, clean_start)
        noisy_video_freqs = model._create_video_freqs_batch(N, noisy_start)
        noisy_freqs = model._build_noisy_freqs(
            noisy_video_freqs, N,
            block_latent_counts=block_latent_counts,
            include_value=False,
        )
    else:
        clean_video_freqs = model._create_video_freqs(N, start_frame=0)
        noisy_video_freqs = model._create_video_freqs(N, start_frame=obs)
        noisy_freqs = model._build_noisy_freqs(
            noisy_video_freqs, N, block_latent_counts=block_latent_counts, include_value=False
        )

    with torch.autocast(device_type="cuda", dtype=dtype):
        for step_idx in range(len(sched_video.timesteps)):
            t_vid = sched_video.timesteps[step_idx]
            t_act = sched_action.timesteps[step_idx]

            clean_video_tokens = model._video_patch_embed(clean_latents)
            noisy_video_tokens = model._video_patch_embed(noisy_video)
            half_seq_len = clean_video_tokens.shape[1]

            noisy_action_chunks = model._split_flat_action_chunks(noisy_action.to(dtype), block_action_token_lengths)
            action_tokens = model._encode_action_chunk_tokens(noisy_action_chunks, noisy_states.to(dtype) if noisy_states is not None else None)
            if not model.config.use_mot:
                action_tokens = model.action_expert.projection.project_in(action_tokens)
            action_seq_len = action_tokens.shape[1]

            clean_t = torch.zeros(B, half_seq_len, device=device)
            noisy_v_t = torch.full((B, half_seq_len), float(t_vid.item()), device=device)
            action_t_val = 0.0 if mode == "wm" else float(t_act.item())

            # a_emb for action decoder (state+action only)
            noisy_a_t_sa = model._build_chunked_action_timesteps(
                torch.full((B,), action_t_val, device=device, dtype=dtype),
                block_latent_counts,
                include_value=False,
            )
            a_emb, a_e0 = model._action_time_embed(noisy_a_t_sa)

            noisy_a_t = noisy_a_t_sa

            v_emb_full, v_e0 = model._video_time_embed(torch.cat([clean_t, noisy_v_t], dim=1))

            if not model.config.use_mot:
                # Action-first hybrid-memory layout (mirror training_step):
                # [ clean_video[f] | gist[f] | per-chunk(state?, action, noisy_video) ]
                gist_tokens = model.gist_tokens.expand(B, N)
                chunk_ar_tokens = model._interleave_memory_chunk_ar_tokens(
                    action_tokens, noisy_video_tokens, block_latent_counts,
                )
                anchor_len = 0
                x = torch.cat([clean_video_tokens, gist_tokens, chunk_ar_tokens], dim=1)
                clean_region_len = anchor_len + clean_video_tokens.shape[1] + gist_tokens.shape[1]
                chunk_ar_len = chunk_ar_tokens.shape[1]
                total_seq_len = x.shape[1]

                gist_t = torch.zeros(B, gist_tokens.shape[1], device=device)
                noisy_v_t_b = torch.full((B,), float(t_vid.item()), device=device, dtype=dtype)
                noisy_a_t_b = torch.full((B,), action_t_val, device=device, dtype=dtype)
                chunk_ar_t = model._build_memory_chunk_ar_timesteps(
                    noisy_a_t_b, noisy_v_t_b, block_latent_counts,
                )
                anchor_t = torch.zeros(B, anchor_len, device=device)
                combined_t = torch.cat([anchor_t, clean_t, gist_t, chunk_ar_t], dim=1)
                if total_seq_len < padded_len:
                    pad_n = padded_len - total_seq_len
                    x = torch.cat([x, x.new_zeros(B, pad_n, x.shape[2])], dim=1)
                    combined_t = torch.cat([combined_t, combined_t.new_zeros(B, pad_n)], dim=1)
                _, combined_e0 = model._video_time_embed(combined_t)

                full_freqs = model._build_memory_full_freqs(
                    clean_video_freqs, noisy_video_freqs, block_latent_counts,
                )
                for layer_idx in range(model.config.num_layers):
                    x = model._forward_layer_memory(
                        x, combined_e0, layer_idx, t5_ctx, ar_mask, full_freqs,
                    )

                clean_video_hidden = x[:, anchor_len:anchor_len + clean_video_tokens.shape[1]]
                chunk_ar_out = x[:, clean_region_len:clean_region_len + chunk_ar_len]
                sa_hidden, noisy_video_hidden = model._split_memory_chunk_ar_hidden(
                    chunk_ar_out, block_latent_counts,
                )
                video_out = torch.cat([clean_video_hidden, noisy_video_hidden], dim=1)
                action_out = model.action_expert.projection.project_out(sa_hidden)
            else:
                video_tokens = torch.cat([clean_video_tokens, noisy_video_tokens], dim=1)
                mot_action_tokens = action_tokens
                for layer_idx in range(model.config.num_layers):
                    video_tokens, mot_action_tokens, _ = model._forward_layer_mot(
                        video_tokens, mot_action_tokens, v_e0, a_e0,
                        layer_idx, t5_ctx, ar_mask,
                        is_training=True, video_freqs=clean_video_freqs, noisy_freqs=noisy_freqs,
                        padded_len=padded_len,
                    )
                video_out = video_tokens
                action_out = mot_action_tokens.to(dtype)

            v_pred = model.video_model.wan_model.head(video_out, v_emb_full)
            grid_size = torch.tensor([[2 * N, model.lat_H, model.lat_W]], dtype=torch.long, device=device).expand(B, -1)
            v_pred_unpatched = model.video_model.wan_model.unpatchify(v_pred, grid_size)
            if isinstance(v_pred_unpatched, list):
                v_pred_unpatched = torch.stack(v_pred_unpatched)

            a_pred = model.action_expert.decoder(action_out, a_emb)
            a_pred = model._strip_chunk_state_predictions(a_pred, block_action_token_lengths)

            noisy_half_pred = v_pred_unpatched[:, :, N:]
            noisy_video = sched_video.step(
                noisy_half_pred.transpose(1, 2), t_vid,
                noisy_video.transpose(1, 2), return_dict=False
            )[0].transpose(1, 2)

            if mode != "wm":
                noisy_action = sched_action.step(a_pred, t_act, noisy_action, return_dict=False)[0]

    return noisy_video, noisy_action


@torch.no_grad()
def inference_value(model, batch, config):
    """Run value-only inference: t_vid=0, t_act=0, single forward pass.

    Returns:
        value_pred: [B, M] predicted values per noisy chunk (None if value disabled)
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = model.dtype
    nvpf = model.config.value_tokens_per_frame
    if not model.config.value_enabled or nvpf == 0:
        return None

    latents = batch["latents"].to(device, dtype=dtype)
    B, C_lat, W, H_lat, W_lat = latents.shape
    apl = model.config.actions_per_latent

    actions = batch["actions"].to(device, dtype=torch.float32)
    states = batch.get("states")
    if states is not None:
        states = states.to(device, dtype=torch.float32)
    lang = batch.get("language_embedding")
    if lang is not None:
        lang = lang.to(device, dtype=dtype)

    chunk_size = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))
    ws_range = config.common.get("chunk_window_range", [1, 4])
    window_size = int(config.common.get("chunk_window_size", ws_range[-1]))
    cs = chunk_size
    obs = model._obs_latents()
    N = W - obs

    napf_sa = model.config.state_tokens_per_frame + model.config.action_tokens_per_frame

    clean_latents = latents[:, :, :N]
    # Value mode: "noisy" video/action are actually GT (t=0)
    noisy_video_gt = latents[:, :, obs:]
    action_block_info = model._prepare_noisy_action_blocks(
        actions, states, cs, dtype=dtype, obs_latents=obs
    )
    noisy_actions_gt = action_block_info["flat_actions"]
    noisy_states = action_block_info["states"]
    block_latent_counts = action_block_info["block_latent_counts"]
    block_action_token_lengths = action_block_info["block_action_token_lengths"]
    num_noisy_chunks = len(block_latent_counts)

    ar_mask, padded_len = create_sliding_window_mask(
        num_latent_frames=W,
        video_tokens_per_frame=model.frame_seqlen,
        action_tokens_per_frame=model.config.action_tokens_per_frame,
        chunk_size=chunk_size,
        window_size=window_size,
        use_state=model.config.use_state,
        state_tokens_per_frame=model.config.state_tokens_per_frame,
        value_tokens_per_frame=nvpf,
        device=device,
    )

    t5_ctx = model._t5_embed(lang) if lang is not None else None

    latent_start = batch.get("latent_start")
    if latent_start is not None:
        latent_start = latent_start.to(device=device, dtype=torch.long)
        clean_start = latent_start
        noisy_start = latent_start + obs
        clean_video_freqs = model._create_video_freqs_batch(N, clean_start)
        noisy_video_freqs = model._create_video_freqs_batch(N, noisy_start)
        noisy_freqs = model._build_noisy_freqs(
            noisy_video_freqs, N,
            block_latent_counts=block_latent_counts,
        )
    else:
        clean_video_freqs = model._create_video_freqs(N, start_frame=0)
        noisy_video_freqs = model._create_video_freqs(N, start_frame=obs)
        noisy_freqs = model._build_noisy_freqs(
            noisy_video_freqs, N,
            block_latent_counts=block_latent_counts,
        )

    with torch.autocast(device_type="cuda", dtype=dtype):
        clean_video_tokens = model._video_patch_embed(clean_latents)
        noisy_video_tokens = model._video_patch_embed(noisy_video_gt)
        half_seq_len = clean_video_tokens.shape[1]

        noisy_action_chunks = model._split_flat_action_chunks(noisy_actions_gt.to(dtype), block_action_token_lengths)
        action_tokens = model._encode_action_chunk_tokens(
            noisy_action_chunks, noisy_states.to(dtype) if noisy_states is not None else None
        )

        if not model.config.use_mot:
            action_tokens = model.action_expert.projection.project_in(action_tokens)
            value_query = model.value_tokens.expand(B, num_noisy_chunks)
            action_tokens = model._interleave_value_tokens(action_tokens, value_query, block_latent_counts)
            action_seq_len = action_tokens.shape[1]

            # t=0 for everything
            clean_t = torch.zeros(B, half_seq_len, device=device)
            noisy_v_t = torch.zeros(B, half_seq_len, device=device)
            noisy_a_t = torch.zeros(B, action_seq_len, device=device)

            x = torch.cat([clean_video_tokens, noisy_video_tokens, action_tokens], dim=1)
            total_seq_len = x.shape[1]
            combined_t = torch.cat([clean_t, noisy_v_t, noisy_a_t], dim=1)
            if total_seq_len < padded_len:
                pad_n = padded_len - total_seq_len
                x = torch.cat([x, x.new_zeros(B, pad_n, x.shape[2])], dim=1)
                combined_t = torch.cat([combined_t, combined_t.new_zeros(B, pad_n)], dim=1)
            _, combined_e0 = model._video_time_embed(combined_t)

            for layer_idx in range(model.config.num_layers):
                x = model._forward_layer_simple(
                    x, combined_e0, layer_idx, t5_ctx, ar_mask,
                    is_training=True, action_register_length=action_seq_len,
                    half_seq_len=half_seq_len, video_freqs=clean_video_freqs, noisy_freqs=noisy_freqs,
                )

            x = x[:, :total_seq_len]
            full_noisy_out = x[:, 2 * half_seq_len:2 * half_seq_len + action_seq_len]
            val_hidden, _ = model._split_value_and_action_hidden(full_noisy_out, block_latent_counts)
        else:
            # MoT mode
            action_seq_len = action_tokens.shape[1]
            video_tokens = torch.cat([clean_video_tokens, noisy_video_tokens], dim=1)
            mot_action_tokens = action_tokens

            clean_t = torch.zeros(B, half_seq_len, device=device)
            noisy_v_t = torch.zeros(B, half_seq_len, device=device)
            _, v_e0 = model._video_time_embed(torch.cat([clean_t, noisy_v_t], dim=1))
            noisy_a_t = torch.zeros(B, action_seq_len, device=device)
            _, a_e0 = model._action_time_embed(noisy_a_t)

            mot_value_tokens = model.value_tokens.expand(B, num_noisy_chunks)
            val_t_vals = torch.zeros(B, num_noisy_chunks * nvpf, dtype=dtype, device=device)
            _, val_e0 = model._video_time_embed(val_t_vals)

            for layer_idx in range(model.config.num_layers):
                video_tokens, mot_action_tokens, mot_value_tokens = model._forward_layer_mot(
                    video_tokens, mot_action_tokens, v_e0, a_e0,
                    layer_idx, t5_ctx, ar_mask,
                    is_training=True, video_freqs=clean_video_freqs, noisy_freqs=noisy_freqs,
                    padded_len=padded_len,
                    value_tokens=mot_value_tokens, value_e0=val_e0,
                    block_latent_counts=block_latent_counts,
                )
            val_hidden = mot_value_tokens.reshape(B, num_noisy_chunks, nvpf, -1)

        # Value prediction
        from models.value_head import expected_value
        val_logits = model.value_head(val_hidden.to(model.dtype).reshape(B * num_noisy_chunks * nvpf, -1))
        val_logits = val_logits.reshape(B, num_noisy_chunks, nvpf, -1).mean(dim=2)
        value_pred = expected_value(
            val_logits, model.config.value_num_bins,
            model.config.value_range[0], model.config.value_range[1]
        )

    return value_pred


def _compute_video_metrics(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    valid_len: Optional[torch.Tensor],
    model=None,
) -> Dict[str, float]:
    """Compute video MSE in both latent space and pixel space.

    Pixel-space MSE requires ``model`` (for VAE decode). If model is None or
    decode fails, only latent MSE is returned.
    """
    B, C, W, H, Wd = pred_latents.shape
    metrics: Dict[str, float] = {}

    # --- Latent-space MSE ---
    lat_mse_list = []
    for i in range(B):
        vl = int(valid_len[i].item()) if valid_len is not None else W
        pred_i = pred_latents[i, :, :vl].flatten()
        gt_i = gt_latents[i, :, :vl].flatten()
        if pred_i.numel() == 0:
            continue
        lat_mse_list.append(F.mse_loss(pred_i.float(), gt_i.float(), reduction="mean").item())
    if lat_mse_list:
        metrics["video_latent_mse"] = float(np.mean(lat_mse_list))
        metrics["_count/video_latent_mse"] = len(lat_mse_list)

    # --- Pixel-space MSE (via VAE decode) ---
    if model is not None:
        try:
            dtype = model.dtype
            with torch.no_grad():
                pred_pixels = model.video_model.vae.decode(
                    [pred_latents[i].to(dtype) for i in range(B)]
                )
                gt_pixels = model.video_model.vae.decode(
                    [gt_latents[i].to(dtype) for i in range(B)]
                )
            pix_mse_list = []
            for i in range(B):
                vl = int(valid_len[i].item()) if valid_len is not None else W
                if vl <= 0:
                    continue
                pix_vl = 1 + (vl - 1) * 4  # latent→pixel: 1 + (W-1)*4
                p = ((pred_pixels[i][:, :pix_vl].float().clamp(-1, 1) + 1) / 2)
                g = ((gt_pixels[i][:, :pix_vl].float().clamp(-1, 1) + 1) / 2)
                pix_mse_list.append(F.mse_loss(p, g, reduction="mean").item())
            if pix_mse_list:
                metrics["video_pixel_mse"] = float(np.mean(pix_mse_list))
                metrics["_count/video_pixel_mse"] = len(pix_mse_list)
        except Exception as e:
            logger.warning(f"Pixel-space MSE failed (skipping): {e}")

    return metrics


def create_video_grid(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    model,
    valid_len: Optional[torch.Tensor] = None,
    num_samples: int = 2,
    max_frames: int = 6,
    context_latent: Optional[torch.Tensor] = None,
) -> Optional["Image.Image"]:
    """Create GT vs Pred comparison grid.

    Both GT and Pred are VAE-decoded from latents. context_latent is prepended
    for temporal context, then stripped from the output pixel frames.

    Args:
        context_latent: [B, C, 1, H, W] frame-0 latent for decoder temporal context.
    """
    device = next(model.parameters()).device
    B = min(pred_latents.shape[0], num_samples)
    W = pred_latents.shape[2]

    try:
        frames_to_show = min(W, max_frames)
        with torch.no_grad():
            # --- Pred: prepend context latent for temporal context ---
            if context_latent is not None:
                pred_with_ctx = [
                    torch.cat([context_latent[i, :, :, :, :], pred_latents[i, :, :frames_to_show]], dim=1).to(model.dtype)
                    for i in range(B)
                ]
                pred_decoded = model.video_model.vae.decode(pred_with_ctx)
                # Strip context pixel frames (frame 0 → 1 pixel frame)
                pred_pixels = [p[:, 1:] for p in pred_decoded]
            else:
                pred_pixels = model.video_model.vae.decode(
                    [pred_latents[i, :, :frames_to_show].to(model.dtype) for i in range(B)]
                )

            # --- GT: decode with context ---
            if context_latent is not None:
                gt_with_ctx = [
                    torch.cat([context_latent[i, :, :, :, :], gt_latents[i, :, :frames_to_show]], dim=1).to(model.dtype)
                    for i in range(B)
                ]
                gt_decoded = model.video_model.vae.decode(gt_with_ctx)
                gt_pixel_list = [g[:, 1:] for g in gt_decoded]
            else:
                gt_pixel_list = model.video_model.vae.decode(
                    [gt_latents[i, :, :frames_to_show].to(model.dtype) for i in range(B)]
                )
    except Exception as e:
        logger.warning(f"VAE decode failed during visualization: {e}")
        return None

    nrow = B * 2
    n_pix = pred_pixels[0].shape[1]
    ncol = min(n_pix, 8)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)

    for i in range(B):
        # Pred row
        pred_np = ((pred_pixels[i].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for j in range(ncol):
            pj = min(j, pred_np.shape[0] - 1)
            axes[i * 2, j].imshow(np.clip(pred_np[pj], 0, 1))
            axes[i * 2, j].set_title(f"S{i} Pred f{j}")
            axes[i * 2, j].axis("off")

        # GT row
        gt_np = ((gt_pixel_list[i].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for j in range(ncol):
            gj = min(j, gt_np.shape[0] - 1)
            axes[i * 2 + 1, j].imshow(np.clip(gt_np[gj], 0, 1))
            axes[i * 2 + 1, j].set_title(f"S{i} GT f{j}")
            axes[i * 2 + 1, j].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img_array)


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = None) -> Dict[str, float]:
    """Evaluate: policy_decouple, policy_full_denoise, wm, value. All with metrics + visualization."""
    wm_ratio = config.model.get("wm_ratio", 0.2)
    vm_ratio = config.model.get("value", {}).get("vm_ratio", 0.0)

    # Build dynamic eval schedule: list of eval modes in order
    eval_schedule = ["policy_decouple", "policy_full"]
    if wm_ratio > 0:
        eval_schedule.append("wm")
    if vm_ratio > 0:
        eval_schedule.append("value")

    # The val dataloader has batch_size=1, so one batch == one episode. With a
    # single episode per mode the metrics are noise-dominated; average over
    # `val_batches_per_mode` episodes instead (~30s per episode).
    per_mode = int(getattr(config, "system", {}).get("val_batches_per_mode", 1) or 1)
    if num_eval_batches is None:
        num_eval_batches = len(eval_schedule) * max(1, per_mode)
    logger.info(
        f"Running AR evaluation for {num_eval_batches} batches "
        f"({eval_schedule} x {max(1, per_mode)})..."
    )
    model = getattr(model, "module", model)
    previous_training = model.training
    model.eval()
    metrics = defaultdict(list)
    apl = model.config.actions_per_latent

    # chunk_size used by inference_sample -- must match for GT alignment
    cs = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue
        batch = materialize_varlen_eval_batch(batch)

        device = next(model.parameters()).device
        dtype = torch.bfloat16
        all_latents = batch["latents"].to(device, dtype=dtype)
        obs = model._obs_latents()
        context_latent = all_latents[:, :, :obs]
        gt_latents = all_latents[:, :, obs:]
        valid_len = batch.get("valid_len")
        noisy_valid_len = None
        if valid_len is not None:
            valid_len = valid_len.to(device)
            noisy_valid_len = (valid_len - obs).clamp(min=0)

        mode = eval_schedule[step % len(eval_schedule)] if eval_schedule else None
        if mode is None:
            continue

        if mode == "policy_decouple":
            pred_video, pred_actions = inference_sample(
                model, batch, config, mode="policy", full_denoise=False
            )
            vid_metrics = _compute_video_metrics(pred_video, gt_latents, noisy_valid_len, model=model)
            append_metrics(metrics, "policy_decouple", vid_metrics)
            if "actions" in batch and pred_actions is not None:
                gt_action_info = model._prepare_noisy_action_blocks(
                    batch["actions"].to(pred_actions.device, dtype=torch.float32),
                    batch.get("states").to(pred_actions.device, dtype=torch.float32)
                    if batch.get("states") is not None else None,
                    cs,
                    dtype=pred_actions.dtype,
                    obs_latents=obs,
                )
                gt_actions = gt_action_info["flat_actions"]
                gt_actions = gt_actions[:, :pred_actions.shape[1]]
                act_metrics = _compute_action_metrics(pred_actions, gt_actions, noisy_valid_len, apl)
                append_metrics(metrics, "policy_decouple", act_metrics)
            grid = create_video_grid(
                pred_video, gt_latents, model, noisy_valid_len,
                context_latent=context_latent,
            )
            if grid is not None:
                metrics["_visualization_policy_decouple"] = [grid]

        elif mode == "policy_full":
            pred_video, pred_actions = inference_sample(
                model, batch, config, mode="policy", full_denoise=True
            )
            vid_metrics = _compute_video_metrics(pred_video, gt_latents, noisy_valid_len, model=model)
            append_metrics(metrics, "policy_full", vid_metrics)
            if "actions" in batch and pred_actions is not None:
                gt_action_info = model._prepare_noisy_action_blocks(
                    batch["actions"].to(pred_actions.device, dtype=torch.float32),
                    batch.get("states").to(pred_actions.device, dtype=torch.float32)
                    if batch.get("states") is not None else None,
                    cs,
                    dtype=pred_actions.dtype,
                    obs_latents=obs,
                )
                gt_actions = gt_action_info["flat_actions"]
                gt_actions = gt_actions[:, :pred_actions.shape[1]]
                act_metrics = _compute_action_metrics(pred_actions, gt_actions, noisy_valid_len, apl)
                append_metrics(metrics, "policy_full", act_metrics)
            grid = create_video_grid(
                pred_video, gt_latents, model, noisy_valid_len,
                context_latent=context_latent,
            )
            if grid is not None:
                metrics["_visualization_policy_full"] = [grid]

        elif mode == "wm":
            pred_video_wm, _ = inference_sample(model, batch, config, mode="wm")
            vid_metrics = _compute_video_metrics(pred_video_wm, gt_latents, noisy_valid_len, model=model)
            append_metrics(metrics, "wm", vid_metrics)
            grid = create_video_grid(
                pred_video_wm, gt_latents, model, noisy_valid_len,
                context_latent=context_latent,
            )
            if grid is not None:
                metrics["_visualization_wm"] = [grid]

        elif mode == "value":
            if "values" in batch:
                pred_values_vm = inference_value(model, batch, config)
                if pred_values_vm is not None:
                    gt_noisy_values = batch["values"][:, cs:].to(pred_values_vm.device, dtype=pred_values_vm.dtype)
                    block_latent_counts = model._chunk_latent_counts(gt_noisy_values.shape[1], cs)
                    gt_values = model._select_chunk_end_values(gt_noisy_values, block_latent_counts)
                    chunk_valid_len = None
                    if noisy_valid_len is not None:
                        chunk_end_idx = torch.tensor(
                            model._chunk_end_indices(block_latent_counts),
                            device=pred_values_vm.device,
                            dtype=torch.long,
                        ).unsqueeze(0)
                        chunk_valid_len = (chunk_end_idx < noisy_valid_len.unsqueeze(1)).sum(dim=1)
                    val_metrics = _compute_value_metrics(pred_values_vm, gt_values, chunk_valid_len)
                    append_metrics(metrics, "value", val_metrics)

    final = finalize_metrics(metrics, next(model.parameters()).device)
    model.train(previous_training)
    return final


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """Log evaluation metrics to tensorboard/wandb."""
    if not accelerator.is_main_process:
        return
    log_dict = {f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    if log_dict:
        accelerator.log(log_dict, step=global_step)
    if writer:
        for k, v in log_dict.items():
            writer.add_scalar(k, v, global_step)
        for tag, key in [
            ("eval/video_grid_policy_decouple", "visualization_policy_decouple"),
            ("eval/video_grid_policy_full", "visualization_policy_full"),
            ("eval/video_grid_wm", "visualization_wm"),
        ]:
            if key in metrics and metrics[key] is not None:
                img = metrics[key]
                arr = np.array(img)
                if arr.ndim == 3:
                    writer.add_image(tag, arr.transpose(2, 0, 1), global_step)
    try:
        report_to = getattr(accelerator, "report_to", []) or []
        if isinstance(report_to, str):
            report_to = [report_to]
        if "wandb" in report_to:
            import wandb
            wandb_log = {}
            for k, tag in [
                ("visualization_policy_decouple", "eval/video_grid_policy_decouple"),
                ("visualization_policy_full", "eval/video_grid_policy_full"),
                ("visualization_wm", "eval/video_grid_wm"),
            ]:
                if k in metrics and metrics[k] is not None:
                    wandb_log[tag] = wandb.Image(metrics[k])
            if wandb_log:
                accelerator.log(wandb_log, step=global_step)
    except Exception:
        pass
    logger.info(f"Eval @ step {global_step}: {log_dict}")
