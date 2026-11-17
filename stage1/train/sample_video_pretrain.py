#!/usr/bin/env python3

import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import matplotlib
matplotlib.set_loglevel("WARNING")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from omegaconf import OmegaConf
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from utils.ti2v_timestep import build_ti2v_per_token_timestep

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _dtype_from_config(config) -> torch.dtype:
    """Training compute dtype from YAML (do not use parameter.dtype: DeepSpeed may keep FP32 master weights)."""
    p = OmegaConf.select(config, "model.wan.precision", default="bfloat16")
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(p), torch.bfloat16
    )


def _prepare_context_list(model, language_embeddings, device, dtype):
    text_len = model.video_model.wan_model.text_len
    if language_embeddings is not None:
        emb = language_embeddings.to(device=device, dtype=dtype)
        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb[0]
        if emb.shape[0] < text_len:
            emb = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            emb = emb[:text_len]
        return [emb]

    text_dim = model.video_model.wan_model.text_dim
    return [torch.zeros(text_len, text_dim, device=device, dtype=dtype)]


def _compute_seq_len(model, latents: torch.Tensor) -> int:
    _, _, frames, height, width = latents.shape
    patch_size = model.video_model.wan_model.patch_size
    return (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])


@torch.no_grad()
def inference_video_pretrain_sample(model, batch, config):
    model.eval()
    device = next(model.parameters()).device
    dtype = _dtype_from_config(config)

    latents = batch["latents"].to(device=device, dtype=dtype)
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    valid_len = int(batch.get("valid_len", latents.shape[2]))
    valid_len = min(valid_len, latents.shape[2])
    if valid_len <= 1:
        raise ValueError("Evaluation sample has no denoising target frames (valid_len <= 1).")

    eval_cfg = config.get("evaluation", {})
    requested_cond = int(eval_cfg.get("num_condition_latents", 1))
    n_cond = max(1, min(requested_cond, valid_len - 1))

    clean_latents = latents[:, :, :valid_len]
    sample_latents = torch.randn_like(clean_latents)
    sample_latents[:, :, :n_cond] = clean_latents[:, :, :n_cond]

    context_list = _prepare_context_list(model, batch.get("language_embedding"), device, dtype)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    num_steps = int(eval_cfg.get("num_inference_steps", 30))
    scheduler.set_timesteps(num_steps, device=device, shift=float(config.common.get("shift", 5.0)))
    seq_len = _compute_seq_len(model, sample_latents)

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    patch_size = model.video_model.wan_model.patch_size
    latent_shape = tuple(sample_latents[0].shape)
    for t in scheduler.timesteps:
        timestep = torch.tensor([float(t.item())], device=device, dtype=dtype)
        t_per_token = build_ti2v_per_token_timestep(
            timestep, n_cond, seq_len, patch_size, latent_shape
        )
        x_in = sample_latents[0].to(dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            pred_list = model.video_model.wan_model(
                x=[x_in],
                t=t_per_token,
                context=context_list,
                seq_len=seq_len,
                y=None,
            )
        pred = pred_list[0].unsqueeze(0)
        sample_latents = scheduler.step(
            pred.transpose(1, 2),
            t,
            sample_latents.transpose(1, 2),
            return_dict=False,
        )[0].transpose(1, 2)
        sample_latents = sample_latents.to(dtype=dtype)
        sample_latents[:, :, :n_cond] = clean_latents[:, :, :n_cond]

    pred_latents = sample_latents[:, :, n_cond:valid_len]
    gt_latents = clean_latents[:, :, n_cond:valid_len]
    context_latents = clean_latents[:, :, :n_cond]
    pred_valid_len = pred_latents.shape[2]
    return pred_latents, gt_latents, context_latents, pred_valid_len, n_cond


def _compute_video_metrics(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    model,
    pred_valid_len: Optional[int] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if pred_latents.numel() == 0 or gt_latents.numel() == 0:
        return metrics

    batch_size = pred_latents.shape[0]
    latent_mse = []
    for idx in range(batch_size):
        valid = pred_valid_len if pred_valid_len is not None else pred_latents.shape[2]
        pred_i = pred_latents[idx, :, :valid].float().flatten()
        gt_i = gt_latents[idx, :, :valid].float().flatten()
        if pred_i.numel() == 0:
            continue
        latent_mse.append(F.mse_loss(pred_i, gt_i, reduction="mean").item())
    if latent_mse:
        metrics["video_latent_mse"] = float(np.mean(latent_mse))

    try:
        with torch.no_grad():
            pred_pixels = model.video_model.vae.decode([pred_latents[i].to(model.dtype) for i in range(batch_size)])
            gt_pixels = model.video_model.vae.decode([gt_latents[i].to(model.dtype) for i in range(batch_size)])
        pixel_mse = []
        for idx in range(batch_size):
            valid = pred_valid_len if pred_valid_len is not None else pred_latents.shape[2]
            pix_valid = 1 + (valid - 1) * 4
            pred_i = ((pred_pixels[idx][:, :pix_valid].float().clamp(-1, 1) + 1) / 2).flatten()
            gt_i = ((gt_pixels[idx][:, :pix_valid].float().clamp(-1, 1) + 1) / 2).flatten()
            if pred_i.numel() == 0:
                continue
            pixel_mse.append(F.mse_loss(pred_i, gt_i, reduction="mean").item())
        if pixel_mse:
            metrics["video_pixel_mse"] = float(np.mean(pixel_mse))
    except Exception as exc:
        logger.warning(f"Video-pretrain pixel MSE decode failed (skipping): {exc}")

    return metrics


def _latent_pixels_to_tb_video(pixel_chw: torch.Tensor) -> torch.Tensor:
    """(C, T, H, W) in approx [-1, 1] -> (1, T, C, H, W) in [0, 1], float32 CPU."""
    x = pixel_chw.float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    return x.permute(1, 0, 2, 3).unsqueeze(0)


def _subsample_time(vid: torch.Tensor, max_frames: int) -> torch.Tensor:
    """vid: (1, T, C, H, W). If max_frames > 0 and T > max_frames, evenly sample indices."""
    if max_frames <= 0 or vid.shape[1] <= max_frames:
        return vid
    t = vid.shape[1]
    idx = torch.linspace(0, t - 1, steps=max_frames).long()
    return vid[:, idx]


@torch.no_grad()
def build_tensorboard_eval_videos(
    model,
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    pred_valid_len: Optional[int],
    eval_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Decode pred/gt latent segments to full-res RGB videos for TensorBoard ``add_video``.

    Returns tensors ``(1, T, 3, H, W)`` in [0, 1], CPU float32. Keys: ``pred``, ``gt``, ``compare`` (pred|gt
    concatenated along width per frame). Frame-aligned: same latent temporal length before decode.
    """
    out: Dict[str, torch.Tensor] = {}
    m = _unwrap_model(model)
    md = next(m.parameters()).device
    mdt = next(m.parameters()).dtype

    b = 0
    pv = pred_valid_len if pred_valid_len is not None else pred_latents.shape[2]
    pv = max(0, min(int(pv), pred_latents.shape[2], gt_latents.shape[2]))
    if pv == 0:
        return out

    pl = pred_latents[b, :, :pv].to(device=md, dtype=mdt)
    gl = gt_latents[b, :, :pv].to(device=md, dtype=mdt)

    pred_pix = m.video_model.vae.decode([pl])[0]
    gt_pix = m.video_model.vae.decode([gl])[0]

    vp = _latent_pixels_to_tb_video(pred_pix)
    vg = _latent_pixels_to_tb_video(gt_pix)

    tmin = min(vp.shape[1], vg.shape[1])
    if vp.shape[1] != vg.shape[1]:
        logger.warning(
            "TB video: pred/gt decode length mismatch %s vs %s, trimming to %s",
            vp.shape[1],
            vg.shape[1],
            tmin,
        )
        vp = vp[:, :tmin]
        vg = vg[:, :tmin]

    max_f = int(eval_cfg.get("tensorboard_max_video_frames", 0) or 0)
    vp = _subsample_time(vp, max_f)
    vg = _subsample_time(vg, max_f)
    if vp.shape[1] != vg.shape[1]:
        tmin = min(vp.shape[1], vg.shape[1])
        vp = vp[:, :tmin]
        vg = vg[:, :tmin]

    out["pred"] = vp
    out["gt"] = vg
    out["compare"] = torch.cat([vp, vg], dim=-1)
    return out


def create_video_pretrain_grid(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    context_latents: torch.Tensor,
    model,
    max_frames: int = 6,
):
    batch_size = min(pred_latents.shape[0], 1)
    n_cond = context_latents.shape[2]
    context_pix = 1 + max(n_cond - 1, 0) * 4

    try:
        with torch.no_grad():
            context_pixels = model.video_model.vae.decode([context_latents[i].to(model.dtype) for i in range(batch_size)])
            pred_with_ctx = [
                torch.cat([context_latents[i], pred_latents[i]], dim=1).to(model.dtype)
                for i in range(batch_size)
            ]
            gt_with_ctx = [
                torch.cat([context_latents[i], gt_latents[i]], dim=1).to(model.dtype)
                for i in range(batch_size)
            ]
            pred_pixels_ctx = model.video_model.vae.decode(pred_with_ctx)
            gt_pixels_ctx = model.video_model.vae.decode(gt_with_ctx)
            pred_pixels = [p[:, context_pix:] for p in pred_pixels_ctx]
            gt_pixels = [g[:, context_pix:] for g in gt_pixels_ctx]
    except Exception as exc:
        logger.warning(f"Video-pretrain visualization decode failed (skipping): {exc}")
        return None

    max_context_cols = min(max(context_pix, 1), max_frames)
    max_pred_cols = min(max(pred_pixels[0].shape[1], 1), max_frames)
    n_cols = max(max_context_cols, max_pred_cols)

    context_np = ((context_pixels[0].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    pred_np = ((pred_pixels[0].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
    gt_np = ((gt_pixels[0].cpu().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()

    # Match subplot cell aspect to decoded frame (T,H,W,C) so imshow uses native H:W without letterbox/stretch.
    def _frame_hw(arr: np.ndarray):
        if arr.ndim != 4 or arr.shape[0] == 0:
            return None
        return int(arr.shape[1]), int(arr.shape[2])

    hw = _frame_hw(pred_np) or _frame_hw(gt_np) or _frame_hw(context_np)
    if hw is None:
        return None
    frame_h, frame_w = hw
    if frame_w <= 0:
        return None
    n_rows_vis = 3
    col_w_in = min(2.5, max(0.35, 140.0 / max(n_cols, 1)))
    row_h_in = col_w_in * (frame_h / frame_w)
    fig, axes = plt.subplots(n_rows_vis, n_cols, figsize=(col_w_in * n_cols, row_h_in * n_rows_vis))
    if n_cols == 1:
        axes = np.asarray(axes).reshape(n_rows_vis, 1)

    rows = [
        ("Cond", context_np),
        ("Pred", pred_np),
        ("GT", gt_np),
    ]
    for row_idx, (title, frames) in enumerate(rows):
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].axis("off")
            if frames.shape[0] == 0:
                continue
            frame_idx = min(col_idx, frames.shape[0] - 1)
            axes[row_idx, col_idx].imshow(
                np.clip(frames[frame_idx], 0, 1),
                aspect="equal",
                interpolation="nearest",
            )
            axes[row_idx, col_idx].set_title(f"{title} f{col_idx}", fontsize=7)

    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img_array)


@torch.no_grad()
def evaluate_video_pretrain(model, dataloader, accelerator, config, num_eval_batches: Optional[int] = None) -> Dict[str, float]:
    eval_cfg = config.get("evaluation", {})
    if num_eval_batches is None:
        num_eval_batches = int(eval_cfg.get("num_batches", 1))

    model = accelerator.unwrap_model(model)
    device = next(model.parameters()).device
    dtype = _dtype_from_config(config)
    metrics_acc = defaultdict(list)
    visualization = None
    tb_videos: Optional[Dict[str, torch.Tensor]] = None

    model.eval()
    for step, batch in enumerate(dataloader):
        if batch is None:
            continue
        if step >= num_eval_batches:
            break

        latents = batch["latents"].to(device=device, dtype=dtype)
        lang = batch.get("language_embedding")
        if lang is not None:
            lang = lang.to(device=device, dtype=dtype)
        valid_len = batch.get("valid_len")

        loss_dict = model.training_step(
            latents=latents,
            language_embeddings=lang,
            valid_len=valid_len,
        )
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                metrics_acc[f"val_{key}"].append(value.item())
            elif isinstance(value, (int, float)):
                metrics_acc[f"val_{key}"].append(float(value))

        pred_latents, gt_latents, context_latents, pred_valid_len, n_cond = inference_video_pretrain_sample(model, batch, config)
        sample_metrics = _compute_video_metrics(pred_latents, gt_latents, model, pred_valid_len=pred_valid_len)
        for key, value in sample_metrics.items():
            metrics_acc[key].append(float(value))
        metrics_acc["eval_condition_latents"].append(float(n_cond))

        if visualization is None and bool(eval_cfg.get("log_matplotlib_grid", False)):
            visualization = create_video_pretrain_grid(
                pred_latents=pred_latents,
                gt_latents=gt_latents,
                context_latents=context_latents,
                model=model,
                max_frames=int(eval_cfg.get("max_visualization_frames", 6)),
            )

        if tb_videos is None and bool(eval_cfg.get("tensorboard_log_video", True)):
            try:
                tb_videos = build_tensorboard_eval_videos(
                    model,
                    pred_latents,
                    gt_latents,
                    pred_valid_len,
                    eval_cfg,
                )
            except Exception as exc:
                logger.warning("TensorBoard add_video pack failed (skipping): %s", exc)

    metrics = {
        key: float(np.mean(values))
        for key, values in metrics_acc.items()
        if values
    }
    if visualization is not None:
        metrics["_visualization_video_pretrain"] = [visualization]
    if tb_videos:
        metrics["_tb_eval_videos"] = tb_videos
        metrics["_tb_eval_video_fps"] = int(eval_cfg.get("tensorboard_video_fps", 8))
    return metrics


def log_video_pretrain_metrics(metrics: Dict, writer, global_step: int):
    if writer is None:
        return

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"eval/{key}", value, global_step)

    image_list = metrics.get("_visualization_video_pretrain")
    if image_list:
        image = image_list[0]
        if image is not None:
            arr = np.asarray(image)
            writer.add_image("eval/video_pretrain_grid", arr.transpose(2, 0, 1), global_step)

    tb_videos = metrics.get("_tb_eval_videos")
    if tb_videos:
        fps = max(1, int(metrics.get("_tb_eval_video_fps", 8)))
        for name, vid in tb_videos.items():
            if vid is None or not isinstance(vid, torch.Tensor) or vid.numel() == 0:
                continue
            try:
                writer.add_video(f"eval/video_{name}", vid, global_step, fps=fps)
            except Exception as exc:
                logger.warning("TensorBoard add_video(%s) failed: %s", name, exc)
