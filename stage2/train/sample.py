#!/usr/bin/env python3
"""Evaluation for sliding-window AR training.

During training eval (e.g. every 500 steps):
  - Load pre-computed latents from dataset, use clean history as conditioning
  - Non-loop modes use the parallel sliding-window sampler
  - loop mode uses true KV-cache streaming inference for policy_full/scaling
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

from models.block_attention import create_sliding_window_mask

logger = logging.getLogger(__name__)


def _build_evaluation_schedule(
    modes,
    *,
    batches_per_mode: int = 1,
    num_eval_batches: Optional[int] = None,
):
    if not modes:
        return []
    total = (
        max(0, int(num_eval_batches))
        if num_eval_batches is not None
        else len(modes) * max(1, int(batches_per_mode))
    )
    return [modes[step % len(modes)] for step in range(total)]


def _compute_action_metrics(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    valid_len: Optional[torch.Tensor],
    actions_per_latent: int,
    sample_mask: Optional[torch.Tensor] = None,
    action_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute action MSE and MAE."""
    B = pred_actions.shape[0]
    mse_per_sample, mae_per_sample = [], []
    for i in range(B):
        if sample_mask is not None and not bool(sample_mask[i].item()):
            continue
        vl = int(valid_len[i].item()) if valid_len is not None else pred_actions.shape[1] // actions_per_latent
        n = min(vl * actions_per_latent, pred_actions.shape[1], gt_actions.shape[1])
        pred_i = pred_actions[i, :n].float()
        gt_i = gt_actions[i, :n].float()
        if action_valid_mask is not None:
            keep = _action_coordinate_valid_mask(action_valid_mask[i, :n], pred_i)
            pred_i, gt_i = pred_i[keep], gt_i[keep]
        if pred_i.numel() == 0:
            continue
        mse_per_sample.append(F.mse_loss(pred_i, gt_i, reduction="mean").item())
        mae_per_sample.append(F.l1_loss(pred_i, gt_i, reduction="mean").item())
    if not mse_per_sample:
        return {}
    return {
        "action_mse_loss": float(np.mean(mse_per_sample)),
        "action_mae_loss": float(np.mean(mae_per_sample)),
        "_sample_count": float(len(mse_per_sample)),
    }


def _action_coordinate_valid_mask(hand_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Expand a [T,2] target/anchor validity mask over the two hand halves."""
    if hand_mask.shape != (actions.shape[0], 2) or actions.shape[-1] % 2:
        raise ValueError("action_valid_mask must be [T,2] for even-width bimanual actions")
    return hand_mask.to(device=actions.device, dtype=torch.bool).repeat_interleave(
        actions.shape[-1] // 2, dim=-1
    )


def _accumulate_metric_batch(metrics, counts, prefix: str, batch_metrics) -> None:
    """Accumulate sample-weighted sums; counts stay attached to each metric."""
    for key, value in batch_metrics.items():
        if key.startswith("_"):
            continue
        count = float(batch_metrics.get(f"_count/{key}", batch_metrics.get("_sample_count", 1.0)))
        if count <= 0:
            continue
        tag = f"{prefix}{key}"
        metrics[tag].append(float(value) * count)
        counts[tag] += count


def _compute_action_component_metrics(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    valid_len: Optional[torch.Tensor],
    actions_per_latent: int,
    sample_mask: Optional[torch.Tensor] = None,
    action_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute EgoScale-style mean MSE over valid action chunk tokens."""
    B = pred_actions.shape[0]
    buckets = defaultdict(list)
    sample_count = 0
    for i in range(B):
        if sample_mask is not None and not bool(sample_mask[i].item()):
            continue
        vl = int(valid_len[i].item()) if valid_len is not None else pred_actions.shape[1] // actions_per_latent
        n = min(vl * actions_per_latent, pred_actions.shape[1], gt_actions.shape[1])
        if n <= 0:
            continue
        sample_count += 1
        pred_i = pred_actions[i, :n].float()
        gt_i = gt_actions[i, :n].float()
        keep = None if action_valid_mask is None else _action_coordinate_valid_mask(action_valid_mask[i, :n], pred_i)
        if keep is not None and not bool(keep.any()):
            sample_count -= 1
            continue
        all_pred = pred_i if keep is None else pred_i[keep]
        all_gt = gt_i if keep is None else gt_i[keep]
        buckets["action_mse"].append(F.mse_loss(all_pred, all_gt, reduction="mean").item())
        buckets["action_mae"].append(F.l1_loss(all_pred, all_gt, reduction="mean").item())
        if pred_i.shape[-1] >= 58:
            hand_pred = torch.cat([pred_i[..., 0:20], pred_i[..., 29:49]], dim=-1)
            hand_gt = torch.cat([gt_i[..., 0:20], gt_i[..., 29:49]], dim=-1)
            wrist_pos_pred = torch.cat([pred_i[..., 20:23], pred_i[..., 49:52]], dim=-1)
            wrist_pos_gt = torch.cat([gt_i[..., 20:23], gt_i[..., 49:52]], dim=-1)
            wrist_rot_pred = torch.cat([pred_i[..., 23:29], pred_i[..., 52:58]], dim=-1)
            wrist_rot_gt = torch.cat([gt_i[..., 23:29], gt_i[..., 52:58]], dim=-1)
            if keep is not None:
                hand_keep = torch.cat([keep[..., 0:20], keep[..., 29:49]], dim=-1)
                pos_keep = torch.cat([keep[..., 20:23], keep[..., 49:52]], dim=-1)
                rot_keep = torch.cat([keep[..., 23:29], keep[..., 52:58]], dim=-1)
                hand_pred, hand_gt = hand_pred[hand_keep], hand_gt[hand_keep]
                wrist_pos_pred, wrist_pos_gt = wrist_pos_pred[pos_keep], wrist_pos_gt[pos_keep]
                wrist_rot_pred, wrist_rot_gt = wrist_rot_pred[rot_keep], wrist_rot_gt[rot_keep]
            buckets["hand_qpos_mse"].append(F.mse_loss(hand_pred, hand_gt, reduction="mean").item())
            buckets["wrist_pos_mse"].append(F.mse_loss(wrist_pos_pred, wrist_pos_gt, reduction="mean").item())
            buckets["wrist_rot_mse"].append(F.mse_loss(wrist_rot_pred, wrist_rot_gt, reduction="mean").item())
            wrist_pred = torch.cat([wrist_pos_pred, wrist_rot_pred], dim=-1)
            wrist_gt = torch.cat([wrist_pos_gt, wrist_rot_gt], dim=-1)
            buckets["wrist_mse"].append(F.mse_loss(wrist_pred, wrist_gt, reduction="mean").item())
    out = {key: float(np.mean(vals)) for key, vals in buckets.items() if vals}
    out["_sample_count"] = float(sample_count)
    return out


def _aligned_gt_actions(model, batch, pred_actions: torch.Tensor, chunk_size: int,
                        return_valid_mask: bool = False):
    """Build GT actions in the same chunk-relative layout as inference_sample."""
    action_block_info = model._prepare_noisy_action_blocks(
        batch["actions"].to(pred_actions.device, dtype=torch.float32),
        batch.get("states").to(pred_actions.device, dtype=torch.float32)
        if batch.get("states") is not None else None,
        chunk_size,
        dtype=pred_actions.dtype,
        **_action_prepare_kwargs(batch, pred_actions.device),
    )
    gt = action_block_info["flat_actions"][:, :pred_actions.shape[1]]
    valid_mask = action_block_info.get("flat_action_valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask[:, :pred_actions.shape[1]]
    return (gt, valid_mask) if return_valid_mask else gt


def _action_prepare_kwargs(batch, device):
    """Pass optional tracking masks and the source of each sample's wrist stats."""
    kwargs = {}
    action_valid = batch.get("action_valid_mask")
    state_valid = batch.get("state_valid_mask")
    if action_valid is not None or state_valid is not None:
        if action_valid is None or state_valid is None:
            raise ValueError("action_valid_mask and state_valid_mask must be supplied together")
        kwargs["action_valid_mask"] = action_valid.to(device=device, dtype=torch.bool)
        kwargs["state_valid_mask"] = state_valid.to(device=device, dtype=torch.bool)
    if batch.get("wrist_stats_source") is not None:
        kwargs["wrist_stats_sources"] = batch["wrist_stats_source"]
    return kwargs


def _action_eval_sample_mask(
    actions: torch.Tensor,
    valid_len: Optional[torch.Tensor],
    actions_per_latent: int,
    wrist_xyz_abs_max: Optional[float],
    latent_offset: int = 0,
    action_valid_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return per-sample mask for eval action windows with sane raw wrist xyz."""
    if wrist_xyz_abs_max is None or float(wrist_xyz_abs_max) <= 0:
        return None
    if actions.shape[-1] >= 58:
        left_xyz_slice = slice(20, 23)
        right_xyz_slice = slice(49, 52)
    elif actions.shape[-1] >= 54:
        left_xyz_slice = slice(20, 23)
        right_xyz_slice = slice(47, 50)
    else:
        return None

    threshold = float(wrist_xyz_abs_max)
    actions_cpu = actions.detach().cpu().float()
    valid_cpu = valid_len.detach().cpu() if valid_len is not None else None
    B = actions_cpu.shape[0]
    mask = torch.ones(B, dtype=torch.bool)
    offset_latents = max(0, int(latent_offset))
    start = offset_latents * actions_per_latent
    for i in range(B):
        total_latents = int(valid_cpu[i].item()) if valid_cpu is not None else actions_cpu.shape[1] // actions_per_latent
        usable_latents = max(total_latents - offset_latents, 0)
        n = min(usable_latents * actions_per_latent, max(0, actions_cpu.shape[1] - start))
        if n <= 0:
            mask[i] = False
            continue
        cur = actions_cpu[i, start:start + n]
        wrist_xyz = torch.cat([cur[..., left_xyz_slice], cur[..., right_xyz_slice]], dim=-1)
        if action_valid_mask is not None:
            keep = action_valid_mask[i, start:start + n].detach().cpu().bool().repeat_interleave(3, dim=-1)
            wrist_xyz = wrist_xyz[keep]
        mask[i] = wrist_xyz.numel() == 0 or bool(
            torch.isfinite(wrist_xyz).all().item() and wrist_xyz.abs().max().item() <= threshold
        )
    return mask


def _latent_time_len(latents: torch.Tensor) -> int:
    return int(latents.shape[3] if latents.dim() == 6 else latents.shape[2])


def _latent_slice_time(latents: torch.Tensor, start: int, end: Optional[int] = None) -> torch.Tensor:
    if latents.dim() == 6:
        return latents[:, :, :, start:end]
    return latents[:, :, start:end]


def _latent_slice_sample_time(latents: torch.Tensor, sample_idx: int, start: int, end: Optional[int] = None) -> torch.Tensor:
    if latents.dim() == 6:
        return latents[sample_idx, :, :, start:end]
    return latents[sample_idx, :, start:end]


def _latent_sample_time_flat(latents: torch.Tensor, sample_idx: int, valid_len: int) -> torch.Tensor:
    return _latent_slice_sample_time(latents, sample_idx, 0, valid_len).flatten()


def _video_grid_from_latents(model, latents: torch.Tensor):
    if hasattr(model, "_video_grid_from_latent"):
        return model._video_grid_from_latent(latents)
    if latents.dim() == 6:
        views = int(latents.shape[1])
        h_lat = int(latents.shape[4])
        w_lat = int(latents.shape[5])
    else:
        views = 1
        h_lat = int(latents.shape[3])
        w_lat = int(latents.shape[4])
    lat_h = h_lat // 2
    lat_w = w_lat // 2
    per_view = lat_h * lat_w
    return views, lat_h, lat_w, per_view, views * per_view


def _step_video_scheduler(scheduler, model, model_output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_step_video_scheduler"):
        return model._step_video_scheduler(scheduler, model_output, timestep, sample)
    if sample.dim() == 6:
        out = scheduler.step(
            model_output.permute(0, 1, 3, 2, 4, 5),
            timestep,
            sample.permute(0, 1, 3, 2, 4, 5),
            return_dict=False,
        )[0]
        return out.permute(0, 1, 3, 2, 4, 5)
    return scheduler.step(
        model_output.transpose(1, 2), timestep, sample.transpose(1, 2), return_dict=False
    )[0].transpose(1, 2)


def _unpack_sample_output(output):
    pred_video, pred_actions = output[:2]
    return pred_video, pred_actions


def _is_loop_mode(model) -> bool:
    return str(getattr(model.config, "mode", "joint") or "joint").lower() == "loop"


def _latent_cat_time(chunks):
    if not chunks:
        return None
    time_dim = 3 if chunks[0].dim() == 6 else 2
    return torch.cat(chunks, dim=time_dim)


@torch.no_grad()
def inference_loop_streaming_sample(model, batch, config):
    """Run true streaming loop inference using the first chunk as KV context."""
    device = next(model.parameters()).device
    dtype = model.dtype

    latents = batch["latents"].to(device, dtype=dtype)
    B = latents.shape[0]
    W = _latent_time_len(latents)
    chunk_size = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))
    N = max(W - chunk_size, 0)
    apl = model.config.actions_per_latent

    lang = batch.get("language_embedding")
    if lang is not None:
        lang = lang.to(device, dtype=dtype)
    states = batch.get("states")
    if states is not None:
        states = states.to(device, dtype=torch.float32)

    start_frame_abs = 0
    latent_start = batch.get("latent_start")
    if getattr(model.config, "video_rope_mode", "window") == "global" and latent_start is not None:
        latent_start = latent_start.to(device=device, dtype=torch.long).flatten()
        if latent_start.numel() > 0:
            if not torch.equal(latent_start, latent_start[:1].expand_as(latent_start)):
                logger.warning("Loop streaming eval with global RoPE received per-sample latent_start; using the first value.")
            start_frame_abs = int(latent_start[0].item())

    sources = batch.get("wrist_stats_source")
    wrist_kwargs = {"wrist_stats_sources": sources} if sources is not None else {}
    saved_state = model.export_inference_state() if hasattr(model, "export_inference_state") else None
    pred_video_chunks = []
    pred_action_chunks = []
    try:
        model.init_kv_cache(batch_size=B, start_frame_abs=start_frame_abs)
        context_latent = _latent_slice_time(latents, 0, chunk_size)
        model.fill_kv_cache_from_latent(context_latent, language_embeddings=lang)

        offset = 0
        while offset < N:
            take = min(chunk_size, N - offset)
            state_chunk = None
            if states is not None and model.config.use_state:
                state_chunk = states[:, chunk_size + offset]

            step_out = model.inference_step_streaming(
                state=state_chunk,
                language_embeddings=lang,
                num_inference_steps=config.model.inference.num_inference_timesteps,
                num_action_inference_steps=config.model.inference.num_action_inference_timesteps,
                predict_video=True,
                **wrist_kwargs,
            )
            z, pred_actions = step_out[:2]
            pred_video_chunks.append(_latent_slice_time(z, 0, take))
            # Streaming APIs return user-space qpos20_rel6d; metrics compare in training space.
            pred_action_chunks.append(model._scale_qpos20_rel_wrist_xyz(pred_actions, **wrist_kwargs)[:, :take * apl])

            offset += take
            if offset < N:
                model.fill_kv_cache_from_latent(z, language_embeddings=lang)
    finally:
        if hasattr(model, "load_inference_state"):
            model.load_inference_state(saved_state)

    pred_video = _latent_cat_time(pred_video_chunks)
    if pred_video is None:
        pred_video = _latent_slice_time(latents, chunk_size, chunk_size)
    pred_actions = torch.cat(pred_action_chunks, dim=1) if pred_action_chunks else latents.new_zeros(B, 0, model.config.action_dim)
    return pred_video, pred_actions


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
        "_sample_count": float(len(mse_list)),
    }


@torch.no_grad()
def inference_sample(model, batch, config, mode="policy", full_denoise: bool = False):
    """Run inference on a batch for evaluation.

    Matches the new training layout:
      clean_vid: frames 0..cs-1 (observation conditioning)
      noisy_vid: frames cs..W-1 (prediction targets, denoised iteratively)
      noisy_act: frames cs..W-1

    Returns:
        video_pred: [B, C, N, H, W] or [B, V, C, N, H, W] predicted video latents for frames cs..W-1
        action_pred: [B, N * actions_per_latent, action_dim] predicted actions
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = model.dtype

    latents = batch["latents"].to(device, dtype=dtype)  # [B,C,W,H,W] or [B,V,C,W,H,W]
    B = latents.shape[0]
    W = _latent_time_len(latents)
    num_views, lat_H, lat_W, _, frame_seqlen = _video_grid_from_latents(model, latents)
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
    ws_range = config.common.get("chunk_window_range", [1, 4])
    window_size = int(config.common.get("chunk_window_size", ws_range[-1]))

    cs = chunk_size
    N = W - cs
    clean_latents = _latent_slice_time(latents, 0, N)  # phys 0..W-cs-1
    action_block_info = model._prepare_noisy_action_blocks(
        actions, states, cs, dtype=dtype, **_action_prepare_kwargs(batch, device)
    )
    noisy_actions_gt = action_block_info["flat_actions"]
    noisy_states = action_block_info["states"]
    block_latent_counts = action_block_info["block_latent_counts"]
    block_action_token_lengths = action_block_info["block_action_token_lengths"]
    noisy_video = torch.randn_like(_latent_slice_time(latents, cs, None), dtype=dtype)
    if mode == "wm":
        noisy_action = noisy_actions_gt.clone()
    else:
        noisy_action = torch.randn_like(noisy_actions_gt, dtype=dtype)

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    # solver_order=1 avoids torch.linalg.solve which can OOM during training eval
    sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False, solver_order=1)
    sched_action = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False, solver_order=1)
    sched_video.set_timesteps(num_steps, device=device, shift=float(model.config.video_timestep_shift))
    sched_action.set_timesteps(num_steps, device=device, shift=float(model.config.action_timestep_shift))

    if mode == "policy" and model.config.decouple_inference_noise and not full_denoise:
        sigma_max = sched_video.sigmas[0].item()
        final_noise = model.config.video_inference_final_noise
        sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
        sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)

    attention_mode = model.config.mode
    # Loop training uniformly samples IDM or FDM mask per iteration; mirror that
    # at inference by building both masks and alternating them across denoise
    # steps in a single batched forward pass.
    loop_alternating_masks = attention_mode == "loop"
    if loop_alternating_masks:
        ar_mask_idm, padded_len = create_sliding_window_mask(
            num_latent_frames=W,
            video_tokens_per_frame=frame_seqlen,
            action_tokens_per_frame=model.config.action_register_tokens_per_frame,
            chunk_size=chunk_size,
            window_size=window_size,
            use_state=model.config.use_state,
            state_tokens_per_frame=model.config.state_tokens_per_frame,
            value_tokens_per_frame=0,
            attention_mode="video_idm",
            device=device,
        )
        ar_mask_fdm, _ = create_sliding_window_mask(
            num_latent_frames=W,
            video_tokens_per_frame=frame_seqlen,
            action_tokens_per_frame=model.config.action_register_tokens_per_frame,
            chunk_size=chunk_size,
            window_size=window_size,
            use_state=model.config.use_state,
            state_tokens_per_frame=model.config.state_tokens_per_frame,
            value_tokens_per_frame=0,
            attention_mode="fdm",
            device=device,
        )
        ar_mask = ar_mask_idm  # overwritten per-step in the denoise loop below
    else:
        ar_mask_idm = None
        ar_mask_fdm = None
        ar_mask, padded_len = create_sliding_window_mask(
            num_latent_frames=W,
            video_tokens_per_frame=frame_seqlen,
            action_tokens_per_frame=model.config.action_register_tokens_per_frame,
            chunk_size=chunk_size,
            window_size=window_size,
            use_state=model.config.use_state,
            state_tokens_per_frame=model.config.state_tokens_per_frame,
            value_tokens_per_frame=0,
            attention_mode=attention_mode,
            device=device,
        )

    napf = model.config.noisy_action_tokens_per_frame
    # napf for action decoder (state+action only, no value tokens)
    napf_sa = model.config.state_tokens_per_frame + model.config.action_tokens_per_frame
    t5_ctx = model._t5_embed(lang) if lang is not None else None
    use_action_only_layout = True

    latent_start = batch.get("latent_start")
    if model.config.video_rope_mode == "global" and latent_start is not None:
        latent_start = latent_start.to(device=device, dtype=torch.long)
        clean_start = latent_start
        noisy_start = latent_start + cs
        clean_video_freqs = model._create_video_freqs_batch(
            N, clean_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        noisy_video_freqs = model._create_video_freqs_batch(
            N, noisy_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        if use_action_only_layout:
            noisy_freqs = None
            full_freqs = model._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=False,
                video_tokens_per_frame=frame_seqlen,
            )
        else:
            full_freqs = None
            noisy_freqs = model._build_noisy_freqs(
                noisy_video_freqs, N,
                block_latent_counts=block_latent_counts,
                include_value=False,
            )
    else:
        clean_video_freqs = model._create_video_freqs(
            N, start_frame=0, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        noisy_video_freqs = model._create_video_freqs(
            N, start_frame=cs, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        if use_action_only_layout:
            noisy_freqs = None
            full_freqs = model._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=False,
                video_tokens_per_frame=frame_seqlen,
            )
        else:
            full_freqs = None
            noisy_freqs = model._build_noisy_freqs(
                noisy_video_freqs, N, block_latent_counts=block_latent_counts, include_value=False
            )

    with torch.autocast(device_type="cuda", dtype=dtype):
        # Non-MoT sample layout: [clean | per-chunk(state/action | noisy_video | value)].
        for step_idx in range(len(sched_video.timesteps)):
            t_vid = sched_video.timesteps[step_idx]
            t_act = sched_action.timesteps[step_idx]

            if loop_alternating_masks:
                # Step 0 uses FDM (action denoise blind to current video, video
                # absorbs action structure); subsequent steps alternate so each
                # modality periodically refines using the partner's latest est.
                ar_mask = ar_mask_fdm if (step_idx % 2 == 0) else ar_mask_idm

            clean_video_tokens = model._video_patch_embed(clean_latents)
            noisy_video_tokens = model._video_patch_embed(noisy_video)
            half_seq_len = clean_video_tokens.shape[1]

            noisy_action_chunks = model._split_flat_action_chunks(noisy_action.to(dtype), block_action_token_lengths)
            action_tokens = model._encode_action_chunk_tokens(
                noisy_action_chunks,
                noisy_states.to(dtype) if noisy_states is not None else None,
            )
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

            if use_action_only_layout:
                noisy_a_t = model._build_chunk_ar_timesteps(
                    torch.full((B,), action_t_val, device=device, dtype=dtype),
                    torch.full((B,), float(t_vid.item()), device=device, dtype=torch.float32),
                    block_latent_counts,
                    include_value=False,
                    video_tokens_per_frame=frame_seqlen,
                )
            else:
                noisy_a_t = noisy_a_t_sa

            v_emb_full, v_e0 = model._video_time_embed(torch.cat([clean_t, noisy_v_t], dim=1))

            if not model.config.use_mot:
                if use_action_only_layout:
                    # Current noisy suffix is packed per chunk as [state/action | noisy_video].
                    chunk_ar_tokens = model._interleave_chunk_ar_tokens(
                        action_tokens,
                        noisy_video_tokens,
                        None,
                        block_latent_counts,
                        video_tokens_per_frame=frame_seqlen,
                    )
                    x = torch.cat([clean_video_tokens, chunk_ar_tokens], dim=1)
                else:
                    chunk_ar_tokens = None
                    x = torch.cat([clean_video_tokens, noisy_video_tokens, action_tokens], dim=1)
                total_seq_len = x.shape[1]
                combined_t = torch.cat(
                    [clean_t, noisy_a_t] if use_action_only_layout else [clean_t, noisy_v_t, noisy_a_t],
                    dim=1,
                )
                if total_seq_len < padded_len:
                    pad_n = padded_len - total_seq_len
                    x = torch.cat([x, x.new_zeros(B, pad_n, x.shape[2])], dim=1)
                    combined_t = torch.cat([combined_t, combined_t.new_zeros(B, pad_n)], dim=1)
                _, combined_e0 = model._video_time_embed(combined_t)

                for layer_idx in range(model.config.num_layers):
                    x = model._forward_layer_simple(
                        x, combined_e0, layer_idx, t5_ctx, ar_mask,
                        is_training=True, action_register_length=action_seq_len,
                        half_seq_len=half_seq_len, video_freqs=clean_video_freqs,
                        noisy_freqs=noisy_freqs, full_freqs=full_freqs,
                    )

                x = x[:, :total_seq_len]
                if use_action_only_layout:
                    # Split the per-chunk [state/action | noisy_video] current chunk layout.
                    clean_video_out = x[:, :half_seq_len]
                    full_noisy_out = x[:, half_seq_len:half_seq_len + chunk_ar_tokens.shape[1]]
                    action_hidden, noisy_video_hidden, _ = model._split_chunk_ar_hidden(
                        full_noisy_out,
                        block_latent_counts,
                        include_value=False,
                        video_tokens_per_frame=frame_seqlen,
                    )
                    video_out = torch.cat([clean_video_out, noisy_video_hidden], dim=1)
                    action_out = model.action_expert.projection.project_out(action_hidden)
                else:
                    video_out = x[:, :2 * half_seq_len]
                    full_noisy_out = x[:, 2 * half_seq_len:2 * half_seq_len + action_seq_len]
                    action_out = model.action_expert.projection.project_out(full_noisy_out)
            else:
                video_tokens = torch.cat([clean_video_tokens, noisy_video_tokens], dim=1)
                mot_action_tokens = action_tokens
                for layer_idx in range(model.config.num_layers):
                    video_tokens, mot_action_tokens, _ = model._forward_layer_mot(
                        video_tokens, mot_action_tokens, v_e0, a_e0,
                        layer_idx, t5_ctx, ar_mask,
                        is_training=True, video_freqs=clean_video_freqs, noisy_freqs=noisy_freqs,
                        padded_len=padded_len,
                        full_freqs=full_freqs,
                    )
                video_out = video_tokens
                action_out = mot_action_tokens.to(dtype)

            v_pred = model.video_model.wan_model.head(video_out, v_emb_full)
            v_pred_unpatched = model._unpatchify_video_tokens(
                v_pred, 2 * N, B, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )

            a_pred = model.action_expert.decoder(action_out, a_emb)
            a_pred = model._strip_chunk_state_predictions(a_pred, block_action_token_lengths)

            noisy_half_pred = _latent_slice_time(v_pred_unpatched, N, None)
            noisy_video = _step_video_scheduler(sched_video, model, noisy_half_pred, t_vid, noisy_video)

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
    B = latents.shape[0]
    W = _latent_time_len(latents)
    num_views, lat_H, lat_W, _, frame_seqlen = _video_grid_from_latents(model, latents)
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
    N = W - cs

    clean_latents = _latent_slice_time(latents, 0, N)
    # Value mode: "noisy" video/action are actually GT (t=0)
    noisy_video_gt = _latent_slice_time(latents, cs, None)
    action_block_info = model._prepare_noisy_action_blocks(
        actions, states, cs, dtype=dtype, **_action_prepare_kwargs(batch, device)
    )
    noisy_actions_gt = action_block_info["flat_actions"]
    noisy_states = action_block_info["states"]
    block_latent_counts = action_block_info["block_latent_counts"]
    block_action_token_lengths = action_block_info["block_action_token_lengths"]
    num_noisy_chunks = len(block_latent_counts)

    attention_mode = model.config.mode
    if attention_mode == "loop":
        attention_mode = "video_idm"
    ar_mask, padded_len = create_sliding_window_mask(
        num_latent_frames=W,
        video_tokens_per_frame=frame_seqlen,
        action_tokens_per_frame=model.config.action_register_tokens_per_frame,
        chunk_size=chunk_size,
        window_size=window_size,
        use_state=model.config.use_state,
        state_tokens_per_frame=model.config.state_tokens_per_frame,
        value_tokens_per_frame=nvpf,
        attention_mode=attention_mode,
        device=device,
    )

    t5_ctx = model._t5_embed(lang) if lang is not None else None
    use_action_only_layout = True

    latent_start = batch.get("latent_start")
    if model.config.video_rope_mode == "global" and latent_start is not None:
        latent_start = latent_start.to(device=device, dtype=torch.long)
        clean_start = latent_start
        noisy_start = latent_start + cs
        clean_video_freqs = model._create_video_freqs_batch(
            N, clean_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        noisy_video_freqs = model._create_video_freqs_batch(
            N, noisy_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        if use_action_only_layout:
            noisy_freqs = None
            full_freqs = model._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=True,
                video_tokens_per_frame=frame_seqlen,
            )
        else:
            full_freqs = None
            noisy_freqs = model._build_noisy_freqs(
                noisy_video_freqs, N,
                block_latent_counts=block_latent_counts,
            )
    else:
        clean_video_freqs = model._create_video_freqs(
            N, start_frame=0, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        noisy_video_freqs = model._create_video_freqs(
            N, start_frame=cs, lat_H=lat_H, lat_W=lat_W, num_views=num_views
        )
        if use_action_only_layout:
            noisy_freqs = None
            full_freqs = model._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=True,
                video_tokens_per_frame=frame_seqlen,
            )
        else:
            full_freqs = None
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
            if use_action_only_layout:
                base_action_tokens = action_tokens
                chunk_ar_tokens = model._interleave_chunk_ar_tokens(
                    base_action_tokens,
                    noisy_video_tokens,
                    value_query,
                    block_latent_counts,
                    video_tokens_per_frame=frame_seqlen,
                )
                action_seq_len = base_action_tokens.shape[1]
            else:
                action_tokens = model._interleave_value_tokens(action_tokens, value_query, block_latent_counts)
                chunk_ar_tokens = None
                action_seq_len = action_tokens.shape[1]

            # t=0 for everything
            clean_t = torch.zeros(B, half_seq_len, device=device)
            noisy_v_t = torch.zeros(B, half_seq_len, device=device)
            noisy_a_t = (
                model._build_chunk_ar_timesteps(
                    torch.zeros(B, device=device, dtype=dtype),
                    torch.zeros(B, device=device, dtype=dtype),
                    block_latent_counts,
                    include_value=True,
                    video_tokens_per_frame=frame_seqlen,
                )
                if use_action_only_layout else torch.zeros(B, action_tokens.shape[1], device=device)
            )

            x = (
                torch.cat([clean_video_tokens, chunk_ar_tokens], dim=1)
                if use_action_only_layout else torch.cat([clean_video_tokens, noisy_video_tokens, action_tokens], dim=1)
            )
            total_seq_len = x.shape[1]
            combined_t = torch.cat(
                [clean_t, noisy_a_t] if use_action_only_layout else [clean_t, noisy_v_t, noisy_a_t],
                dim=1,
            )
            if total_seq_len < padded_len:
                pad_n = padded_len - total_seq_len
                x = torch.cat([x, x.new_zeros(B, pad_n, x.shape[2])], dim=1)
                combined_t = torch.cat([combined_t, combined_t.new_zeros(B, pad_n)], dim=1)
            _, combined_e0 = model._video_time_embed(combined_t)

            for layer_idx in range(model.config.num_layers):
                x = model._forward_layer_simple(
                    x, combined_e0, layer_idx, t5_ctx, ar_mask,
                    is_training=True, action_register_length=action_seq_len,
                    half_seq_len=half_seq_len, video_freqs=clean_video_freqs,
                    noisy_freqs=noisy_freqs, full_freqs=full_freqs,
                )

            x = x[:, :total_seq_len]
            if use_action_only_layout:
                full_noisy_out = x[:, half_seq_len:half_seq_len + chunk_ar_tokens.shape[1]]
                _, _, val_hidden = model._split_chunk_ar_hidden(
                    full_noisy_out,
                    block_latent_counts,
                    include_value=True,
                    video_tokens_per_frame=frame_seqlen,
                )
            else:
                full_noisy_out = x[:, 2 * half_seq_len:2 * half_seq_len + action_tokens.shape[1]]
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
                    full_freqs=full_freqs,
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
    B = pred_latents.shape[0]
    W = _latent_time_len(pred_latents)
    metrics: Dict[str, float] = {}

    # --- Latent-space MSE ---
    lat_mse_list = []
    for i in range(B):
        vl = int(valid_len[i].item()) if valid_len is not None else W
        vl = min(vl, W, _latent_time_len(gt_latents))
        pred_i = _latent_sample_time_flat(pred_latents, i, vl)
        gt_i = _latent_sample_time_flat(gt_latents, i, vl)
        if pred_i.numel() == 0:
            continue
        lat_mse_list.append(F.mse_loss(pred_i.float(), gt_i.float(), reduction="mean").item())
    if lat_mse_list:
        metrics["video_latent_mse"] = float(np.mean(lat_mse_list))
        metrics["_count/video_latent_mse"] = float(len(lat_mse_list))

    # --- Pixel-space MSE (via VAE decode) ---
    if model is not None:
        try:
            dtype = model.dtype
            if pred_latents.dim() == 6:
                pred_decode = [
                    pred_latents[i, v].to(dtype)
                    for i in range(B)
                    for v in range(pred_latents.shape[1])
                ]
                gt_decode = [
                    gt_latents[i, v].to(dtype)
                    for i in range(B)
                    for v in range(gt_latents.shape[1])
                ]
                sample_view = [
                    (i, v)
                    for i in range(B)
                    for v in range(pred_latents.shape[1])
                ]
            else:
                pred_decode = [pred_latents[i].to(dtype) for i in range(B)]
                gt_decode = [gt_latents[i].to(dtype) for i in range(B)]
                sample_view = [(i, 0) for i in range(B)]
            with torch.no_grad():
                pred_pixels = model.video_model.vae.decode(pred_decode)
                gt_pixels = model.video_model.vae.decode(gt_decode)
            pix_mse_list = []
            for dec_idx, (i, _view_idx) in enumerate(sample_view):
                vl = int(valid_len[i].item()) if valid_len is not None else W
                vl = min(vl, W)
                if vl <= 0:
                    continue
                pix_vl = 1 + (vl - 1) * 4  # latent→pixel: 1 + (W-1)*4
                p = ((pred_pixels[dec_idx][:, :pix_vl].float().clamp(-1, 1) + 1) / 2)
                g = ((gt_pixels[dec_idx][:, :pix_vl].float().clamp(-1, 1) + 1) / 2)
                pix_mse_list.append(F.mse_loss(p, g, reduction="mean").item())
            if pix_mse_list:
                metrics["video_pixel_mse"] = float(np.mean(pix_mse_list))
                metrics["_count/video_pixel_mse"] = float(len(pix_mse_list))
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
        context_latent: [B, C, Tctx, H, W] or [B, V, C, Tctx, H, W]
            latents used only to provide VAE temporal context.
    """
    B = min(pred_latents.shape[0], num_samples)
    W = _latent_time_len(pred_latents)

    def _sample_frames_to_show(sample_idx: int) -> int:
        if valid_len is None:
            return min(W, max_frames)
        sample_valid = int(valid_len[sample_idx].item())
        return min(max(sample_valid, 0), W, max_frames)

    context_pixel_len = 0
    if context_latent is not None:
        context_latent_len = _latent_time_len(context_latent)
        # Wan VAE maps T latent frames to 1 + 4 * (T - 1) pixel frames.
        context_pixel_len = 1 + max(context_latent_len - 1, 0) * 4

    def _strip_context(decoded):
        if context_pixel_len <= 0:
            return decoded
        return [p[:, min(context_pixel_len, p.shape[1]):] for p in decoded]

    try:
        if pred_latents.dim() == 6:
            V = pred_latents.shape[1]
            view_labels = ["L", "R"] + [f"V{i}" for i in range(2, V)]
            pred_inputs, gt_inputs, row_meta = [], [], []
            for i in range(B):
                frames_to_show = _sample_frames_to_show(i)
                if frames_to_show <= 0:
                    continue
                for v in range(V):
                    pred_item = pred_latents[i, v, :, :frames_to_show]
                    gt_item = gt_latents[i, v, :, :frames_to_show]
                    if context_latent is not None:
                        pred_item = torch.cat([context_latent[i, v], pred_item], dim=1)
                        gt_item = torch.cat([context_latent[i, v], gt_item], dim=1)
                    pred_inputs.append(pred_item.to(model.dtype))
                    gt_inputs.append(gt_item.to(model.dtype))
                    row_meta.append((i, view_labels[v] if v < len(view_labels) else f"V{v}"))
            if not pred_inputs:
                return None
            with torch.no_grad():
                pred_decoded = model.video_model.vae.decode(pred_inputs)
                gt_decoded = model.video_model.vae.decode(gt_inputs)
            if context_latent is not None:
                pred_pixels = _strip_context(pred_decoded)
                gt_pixel_list = _strip_context(gt_decoded)
            else:
                pred_pixels = pred_decoded
                gt_pixel_list = gt_decoded
        else:
            row_meta, pred_inputs, gt_inputs = [], [], []
            for i in range(B):
                frames_to_show = _sample_frames_to_show(i)
                if frames_to_show <= 0:
                    continue
                row_meta.append((i, ""))
                if context_latent is not None:
                    pred_inputs.append(
                        torch.cat([context_latent[i, :, :, :, :], pred_latents[i, :, :frames_to_show]], dim=1).to(model.dtype)
                    )
                    gt_inputs.append(
                        torch.cat([context_latent[i, :, :, :, :], gt_latents[i, :, :frames_to_show]], dim=1).to(model.dtype)
                    )
                else:
                    pred_inputs.append(pred_latents[i, :, :frames_to_show].to(model.dtype))
                    gt_inputs.append(gt_latents[i, :, :frames_to_show].to(model.dtype))
            if not pred_inputs:
                return None
            with torch.no_grad():
                pred_decoded = model.video_model.vae.decode(pred_inputs)
                gt_decoded = model.video_model.vae.decode(gt_inputs)
            if context_latent is not None:
                pred_pixels = _strip_context(pred_decoded)
                gt_pixel_list = _strip_context(gt_decoded)
            else:
                pred_pixels = pred_decoded
                gt_pixel_list = gt_decoded
    except Exception as e:
        logger.warning(f"VAE decode failed during visualization: {e}")
        return None

    n_items = len(pred_pixels)
    if n_items == 0:
        return None
    nrow = n_items * 2
    n_pix = max([p.shape[1] for p in pred_pixels] + [g.shape[1] for g in gt_pixel_list])
    if n_pix <= 0:
        return None
    ncol = min(n_pix, 8)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_items):
        sample_idx, view_label = row_meta[i]
        prefix = f"S{sample_idx} {view_label}".strip()
        # Pred row
        pred_np = ((pred_pixels[i].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for j in range(ncol):
            if j < pred_np.shape[0]:
                axes[i * 2, j].imshow(np.clip(pred_np[j], 0, 1))
                axes[i * 2, j].set_title(f"{prefix} Pred f{j}")
            axes[i * 2, j].axis("off")

        # GT row
        gt_np = ((gt_pixel_list[i].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
        for j in range(ncol):
            if j < gt_np.shape[0]:
                axes[i * 2 + 1, j].imshow(np.clip(gt_np[j], 0, 1))
                axes[i * 2 + 1, j].set_title(f"{prefix} GT f{j}")
            axes[i * 2 + 1, j].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img_array)


# Inference-only acceleration toggles on model.config that we force-disable
# inside evaluate_model. Any flag here either (a) installs persistent
# monkey-patched module methods that survive eval, or (b) writes to instance
# attributes that the patched forwards keep reading later. Leaving any of them
# active during rank-0-only training-time eval risks desyncing rank 0 vs peers
# in the next training step (different forward branches -> different gradient
# pattern -> DeepSpeed bucket allreduce mismatch -> NCCL hang).
_EVAL_DISABLE_INFERENCE_ACCEL_FLAGS = (
    "t5_kv_caching",
    "cache_inference_time_tensors",
    "cache_inference_rope_tensors",
    "prewarm_inference_caches",
    "qkv_fuse",
    "bf16_norm",
)


def _snapshot_and_disable_inference_accel(model) -> Dict[str, object]:
    """Read every inference-only accel flag off model.config, save the original
    value, and force it to False for the duration of evaluate_model. Returns
    the snapshot dict that _reset_post_eval_module_state restores in finally.
    """
    snap: Dict[str, object] = {}
    cfg = getattr(model, "config", None)
    if cfg is None:
        return snap
    _SENTINEL = object()
    for name in _EVAL_DISABLE_INFERENCE_ACCEL_FLAGS:
        prev = getattr(cfg, name, _SENTINEL)
        if prev is _SENTINEL:
            continue
        snap[name] = prev
        try:
            setattr(cfg, name, False)
        except Exception:
            logger.exception("could not force-disable %s for eval", name)
            snap.pop(name, None)
    return snap


def _reset_post_eval_module_state(
    model,
    prev_training: bool,
    accel_snapshot: Optional[Dict[str, object]] = None,
) -> None:
    """Undo all rank-0-only side effects of evaluate_model so the next training
    forward dispatches identically on every rank.

    Why this matters: evaluate_model runs ONLY on rank 0 (see train.py). The
    forward pass it triggers can populate persistent module state on the inner
    MotusVLA module:
      - WanCrossAttention._motus2_t5_*_cache + ``_motus2_t5_cache_valid``
        (T5 K/V cache, installed and flipped to True inside ``_t5_embed``
        whenever ``not self.training``). Once valid, the patched cross-attn
        forward SKIPS ``self.k(ctx)`` / ``self.v(ctx)`` projections and reads
        directly from the cached tensors -- great for inference, but if
        rank 0 keeps the cache valid after eval while peer ranks never enter
        that branch, rank 0 stops producing gradients on those k/v weights
        while peers do. DeepSpeed's gradient allreduce then mismatches across
        ranks and NCCL deadlocks on the bucket reduction PG ~1-10 training
        steps after eval (observed: step 1008 in loop mode, step ~16 in
        joint mode with val_interval=10).
      - ``self.kv_caches`` / ``kv_cache_valid_len`` / ``current_start_frame``
        (populated by ``inference_loop_streaming_sample`` in loop eval).
      - ``self.training`` flag (set False by ``model.eval()`` inside both
        ``inference_sample`` and ``evaluate_model`` itself).
      - Inference-accel toggles on model.config (restored from accel_snapshot
        if provided; the corresponding flags were force-disabled for the
        duration of evaluate_model).

    All cleanup is guarded so a partial setup (missing wan submodule, etc.)
    cannot mask the original error.
    """
    try:
        from models.t5_kv_cache import invalidate_t5_kv_cache
        wan = getattr(getattr(model, "video_model", None), "wan_model", None)
        if wan is not None:
            invalidate_t5_kv_cache(wan)
    except Exception:
        logger.exception("invalidate_t5_kv_cache failed in evaluate_model cleanup")
    try:
        model.load_inference_state(None)
    except Exception:
        logger.exception("load_inference_state(None) failed in evaluate_model cleanup")
    # Clear any per-instance dict cache populated by cache_inference_time_tensors
    # (write gate is "not self.training", but the dict object persists on rank 0
    # and burns ~MBs of memory; dropping it is also hygiene).
    inf_time_cache = getattr(model, "_inference_time_cache", None)
    if isinstance(inf_time_cache, dict):
        try:
            inf_time_cache.clear()
        except Exception:
            logger.exception("clearing _inference_time_cache failed in evaluate_model cleanup")
    if accel_snapshot:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for name, prev in accel_snapshot.items():
                try:
                    setattr(cfg, name, prev)
                except Exception:
                    logger.exception("could not restore %s after eval", name)
    model.train(prev_training)


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = None) -> Dict[str, float]:
    """Evaluate policy_full (plus wm/value when enabled) with metrics and visualization."""
    if isinstance(dataloader, dict):
        from data.evaluation_groups import prefix_evaluation_metrics
        combined = {'_sum_keys': []}
        for name, group_loader in dataloader.items():
            logger.info('Evaluating fixed pool %s: %s', name,
                        config.get('eval_sample', {}).get('groups', {}).get(name, {}))
            group_metrics = prefix_evaluation_metrics(
                name, evaluate_model(model, group_loader, accelerator, config, num_eval_batches))
            combined['_sum_keys'].extend(group_metrics.pop('_sum_keys', []))
            combined.update(group_metrics)
        return combined
    model = getattr(model, "module", model)
    wm_ratio = config.model.get("wm_ratio", 0.2)
    vm_ratio = config.model.get("value", {}).get("vm_ratio", 0.0)

    # Build dynamic eval schedule: list of eval modes in order.
    # policy_decouple is not a separate algorithm unless decouple_inference_noise
    # is enabled: with that flag off it runs the identical denoise path as
    # policy_full, so scoring both just halves the coverage of the real metric.
    loop_mode = _is_loop_mode(model)
    modes = ["policy_full"]
    if wm_ratio > 0 and not loop_mode:
        modes.append("wm")
    if vm_ratio > 0:
        modes.append("value")

    batches_per_mode = int(config.get("system", {}).get("validation_batches_per_mode", 1))
    if num_eval_batches is None and batches_per_mode <= 0:
        # batches_per_mode <= 0 means "sweep the whole evaluation pool": keep the
        # mode list as a cycle and let the dataloader end the loop.
        eval_schedule, num_eval_batches = modes, None
        logger.info(
            f"Running AR evaluation over the whole pool, cycling modes {modes} "
            f"({len(dataloader)} batches on this rank)..."
        )
    else:
        eval_schedule = _build_evaluation_schedule(
            modes,
            batches_per_mode=max(1, batches_per_mode),
            num_eval_batches=num_eval_batches,
        )
        num_eval_batches = len(eval_schedule)
        logger.info(f"Running AR evaluation for {num_eval_batches} batches...")
    prev_training = model.training
    # Evaluation draws torch RNG noise per batch, and rank-strided loaders hand
    # each rank a different number of batches, so without a snapshot the ranks
    # would leave eval with different RNG state and diverge on the next training
    # step's noise draw.
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    accel_snapshot = _snapshot_and_disable_inference_accel(model)
    if accel_snapshot:
        logger.info(
            "evaluate_model: force-disabled inference accel flags for eval: %s",
            sorted(accel_snapshot.keys()),
        )
    model.eval()
    try:
        metrics = defaultdict(list)
        metric_counts = defaultdict(float)
        sum_keys = set()
        apl = model.config.actions_per_latent
        common_cfg = config.get("common", {}) or {}
        eval_scaling_cfg = config.get("eval_scaling", {}) or {}
        wrist_xyz_abs_max = eval_scaling_cfg.get("wrist_xyz_abs_max", common_cfg.get("wrist_xyz_abs_max", None))

        # chunk_size used by inference_sample -- must match for GT alignment
        cs = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))

        logged_batches = 0
        for step, batch in enumerate(dataloader):
            if num_eval_batches is not None and step >= num_eval_batches:
                break
            if batch is None:
                continue
            logged_batches += 1

            device = next(model.parameters()).device
            dtype = torch.bfloat16
            all_latents = batch["latents"].to(device, dtype=dtype)
            context_latent = _latent_slice_time(all_latents, 0, cs)  # first cs frames for decoder temporal context
            gt_latents = _latent_slice_time(all_latents, cs, None)  # frames cs..W-1 (prediction targets)
            valid_len = batch.get("valid_len")
            noisy_valid_len = None
            if valid_len is not None:
                valid_len = valid_len.to(device)
                noisy_valid_len = (valid_len - cs).clamp(min=0)

            mode = eval_schedule[step % len(eval_schedule)]

            if mode == "policy_full":
                # For loop-trained models we use inference_sample with alternating
                # IDM/FDM masks across denoise steps (configured inside
                # inference_sample). This replaces the chunk-by-chunk KV-cache
                # streaming path (inference_loop_streaming_sample), which made
                # rank 0 run ~10x more forward calls than the other ranks during
                # evalulation and triggered post-step-1000 NCCL collective hangs.
                pred_video, pred_actions = _unpack_sample_output(
                    inference_sample(model, batch, config, mode="policy", full_denoise=True)
                )
                vid_metrics = _compute_video_metrics(pred_video, gt_latents, noisy_valid_len, model=model)
                _accumulate_metric_batch(metrics, metric_counts, "policy_full_", vid_metrics)
                if "actions" in batch and pred_actions is not None:
                    gt_actions, action_valid_mask = _aligned_gt_actions(
                        model, batch, pred_actions, cs, return_valid_mask=True
                    )
                    action_sample_mask = _action_eval_sample_mask(
                        batch["actions"], valid_len, apl, wrist_xyz_abs_max, latent_offset=cs,
                        action_valid_mask=batch.get("action_valid_mask"),
                    )
                    if action_sample_mask is not None:
                        action_sample_mask = action_sample_mask.to(pred_actions.device)
                        metrics["policy_full_skipped_windows_extreme_wrist_xyz"].append(
                            float(pred_actions.shape[0] - int(action_sample_mask.sum().item()))
                        )
                        sum_keys.add("policy_full_skipped_windows_extreme_wrist_xyz")
                    act_metrics = _compute_action_metrics(
                        pred_actions, gt_actions, noisy_valid_len, apl, sample_mask=action_sample_mask,
                        action_valid_mask=action_valid_mask,
                    )
                    _accumulate_metric_batch(metrics, metric_counts, "policy_full_", act_metrics)
                if logged_batches == 1:
                    # One reference picture per pool; a whole-pool sweep would
                    # otherwise pay a VAE decode per batch for an image that the
                    # next batch overwrites.
                    grid = create_video_grid(
                        pred_video, gt_latents, model, noisy_valid_len,
                        context_latent=context_latent,
                    )
                    if grid is not None:
                        metrics["_visualization_policy_full"] = [grid]

            elif mode == "wm":
                pred_video_wm, _ = _unpack_sample_output(inference_sample(model, batch, config, mode="wm"))
                vid_metrics = _compute_video_metrics(pred_video_wm, gt_latents, noisy_valid_len, model=model)
                _accumulate_metric_batch(metrics, metric_counts, "wm_", vid_metrics)
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
                        _accumulate_metric_batch(metrics, metric_counts, "value_", val_metrics)

        final = {}
        for key, vals in metrics.items():
            if key.startswith("_visualization_"):
                tag = key.replace("_visualization_", "visualization_")
                final[tag] = vals[0] if vals else None
            elif vals and isinstance(vals[0], (int, float)):
                if key in sum_keys:
                    final[key] = float(sum(vals))
                else:
                    count = metric_counts[key]
                    if count > 0:
                        final[key] = float(sum(vals) / count)
                        final[f"_count/{key}"] = count
        sum_keys.add("eval_batches")
        final["_sum_keys"] = sorted(sum_keys)
        final["eval_batches"] = float(logged_batches)
    finally:
        _reset_post_eval_module_state(model, prev_training, accel_snapshot)
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    return final


@torch.no_grad()
def evaluate_scaling_model(model, dataloader, accelerator, config, num_eval_batches: int = None) -> Dict[str, float]:
    """Evaluate scaling-law action MSE following EgoScale's sample-then-average protocol."""
    scaling_cfg = config.get("eval_scaling", {}) or {}
    flow_samples = max(1, int(scaling_cfg.get("flow_samples", 16)))
    max_batches = scaling_cfg.get("max_batches", None)
    if num_eval_batches is None and max_batches not in (None, "", 0):
        num_eval_batches = int(max_batches)
    # Non-loop inference_sample builds both schedules from this model setting.
    # eval_scaling is a logging field, not a sampler argument.
    timesteps_per_trajectory = int(config.model.inference.num_inference_timesteps)
    legacy_timesteps = scaling_cfg.get("timesteps_per_trajectory", None)
    if legacy_timesteps is not None and int(legacy_timesteps) != timesteps_per_trajectory:
        logger.warning(
            "eval_scaling.timesteps_per_trajectory=%s is ignored; sampler uses "
            "model.inference.num_inference_timesteps=%d",
            legacy_timesteps,
            timesteps_per_trajectory,
        )
    common_cfg = config.get("common", {}) or {}
    wrist_xyz_abs_max = scaling_cfg.get("wrist_xyz_abs_max", common_cfg.get("wrist_xyz_abs_max", None))

    logger.info(
        "Running scaling law eval: flow_samples=%d, timesteps_per_trajectory=%d, max_batches=%s",
        flow_samples,
        timesteps_per_trajectory,
        "all" if num_eval_batches is None else num_eval_batches,
    )
    model = getattr(model, "module", model)
    model.eval()
    metrics = defaultdict(list)
    apl = model.config.actions_per_latent
    cs = int(getattr(model.config, "fixed_chunk_size", config.model.get("fixed_chunk_size", 2)))

    batches_seen = 0
    samples_seen = 0
    skipped_samples = 0
    for step, batch in enumerate(dataloader):
        if num_eval_batches is not None and step >= num_eval_batches:
            break
        if batch is None:
            continue

        valid_len = batch.get("valid_len")
        noisy_valid_len = None
        device = next(model.parameters()).device
        if valid_len is not None:
            valid_len = valid_len.to(device)
            noisy_valid_len = (valid_len - cs).clamp(min=0)

        pred_sum = None
        pred_len = None
        for _ in range(flow_samples):
            pred_video, pred_actions = _unpack_sample_output(
                inference_loop_streaming_sample(model, batch, config)
                if _is_loop_mode(model)
                else inference_sample(model, batch, config, mode="policy", full_denoise=True)
            )
            pred_actions_f32 = pred_actions.float()
            pred_len = pred_actions_f32.shape[1] if pred_len is None else min(pred_len, pred_actions_f32.shape[1])
            pred_actions_f32 = pred_actions_f32[:, :pred_len]
            pred_sum = pred_actions_f32 if pred_sum is None else pred_sum[:, :pred_len] + pred_actions_f32
            del pred_video, pred_actions

        if pred_sum is None:
            continue
        pred_avg = pred_sum / float(flow_samples)
        # Keep scaling eval in the training-normalized action space. For
        # qpos20_rel6d, hand qpos remains dataset-normalized and wrist xyz
        # remains q01/q99-normalized, matching _aligned_gt_actions below.
        gt_actions, action_valid_mask = _aligned_gt_actions(
            model, batch, pred_avg, cs, return_valid_mask=True
        )
        action_sample_mask = _action_eval_sample_mask(
            batch["actions"], valid_len, apl, wrist_xyz_abs_max, latent_offset=cs,
            action_valid_mask=batch.get("action_valid_mask"),
        )
        if action_sample_mask is not None:
            action_sample_mask = action_sample_mask.to(pred_avg.device)
        component_metrics = _compute_action_component_metrics(
            pred_avg, gt_actions, noisy_valid_len, apl, sample_mask=action_sample_mask,
            action_valid_mask=action_valid_mask,
        )
        kept_count = int(component_metrics.pop("_sample_count", 0))
        skipped_samples += int(pred_avg.shape[0]) - kept_count
        if kept_count <= 0:
            batches_seen += 1
            continue
        for key, value in component_metrics.items():
            metrics[f"eval_scaling/{key}"].append(value)
        batches_seen += 1
        samples_seen += kept_count

    final = {
        key: float(np.mean(vals))
        for key, vals in metrics.items()
        if vals and isinstance(vals[0], (int, float))
    }
    if final or batches_seen > 0:
        final["eval_scaling/flow_samples"] = float(flow_samples)
        final["eval_scaling/timesteps_per_trajectory"] = float(timesteps_per_trajectory)
        final["eval_scaling/eval_batches"] = float(batches_seen)
        final["eval_scaling/eval_windows"] = float(samples_seen)
        final["eval_scaling/skipped_windows_extreme_wrist_xyz"] = float(skipped_samples)
        if wrist_xyz_abs_max is not None and float(wrist_xyz_abs_max) > 0:
            final["eval_scaling/wrist_xyz_abs_max"] = float(wrist_xyz_abs_max)

    model.train()
    return final


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """Log evaluation metrics to tensorboard/wandb."""
    if not accelerator.is_main_process:
        return
    log_dict = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        tag = k if str(k).startswith("eval_scaling/") else f"eval/{k}"
        log_dict[tag] = v
    if log_dict:
        accelerator.log(log_dict, step=global_step)
    if writer:
        for k, v in log_dict.items():
            writer.add_scalar(k, v, global_step)
        for key, img in metrics.items():
            if str(key).startswith('visualization_') and img is not None:
                tag = f"eval/video_grid_{key[14:]}"
                arr = np.array(img)
                if arr.ndim == 3:
                    writer.add_image(tag, arr.transpose(2, 0, 1), global_step)
        writer.flush()
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb_log = dict(log_dict)
            for k, img in metrics.items():
                if str(k).startswith('visualization_') and img is not None:
                    wandb_log[f"eval/video_grid_{k[14:]}"] = wandb.Image(img)
            if wandb_log:
                wandb.log(wandb_log, step=global_step)
    except Exception:
        pass
    logger.info(f"Eval @ step {global_step}: {log_dict}")
