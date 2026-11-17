"""Stage 1: Bidirectional WAN 2.2 TI2V 5B Video Pretrain.

Standard flow matching denoising on robot videos using Motus2's own WanModel.
Uses pre-computed VAE latents (48ch) + UMT5 embeddings from a directory tree (local or S3) or a DuckDB index.
No action expert, no sliding-window AR — pure bidirectional video denoising.

WAN 2.2 TI2V 5B specifics:
  - VAE latent: 48 channels, vae_stride=(4,16,16)
  - model_type='ti2v', in_dim=48, out_dim=48
  - No CLIP, no y channel concat — image conditioning is done by replacing
    the first frame(s) of noisy latents with clean latents
  - y=None in WanModel.forward()

Conditioning strategy (per sample):
  - 50% single-frame condition (1 latent frame), 50% multi-frame (2 latent frames)
  - Condition frames: 50% clean, 50% add Gaussian noise directly in latent space
    (extra noise is regularization only; per-token timestep for those tokens stays 0, same as wan_new Motus.)
  - Language CFG: common.lang_uncond_prob drops real UMT5 for dataset.empty_lang_embed (.pt) on train only (eval unchanged).
  - Per-token timesteps: condition patches use t=0, denoising patches use sampled t (TI2V / WanModel [B, seq_len])
  - common.timestep_sample_method: logit_normal (raw = 1 - sigmoid(N(0,1)), same as Motus WM video σ) or uniform (torch.rand)
  - Loss computed only on non-condition frames
  - training.batch_size>1: collate stacks samples; all latents in a batch must share [C,T,H,W] (fixed shape). Mixed shapes → batch_size=1 + gradient_accumulation_steps.

Usage:
    torchrun --nnodes=1 --nproc_per_node=8 train/train_video_pretrain.py \\
        --deepspeed configs/zero2.json \\
        --config configs/video_pretrain_example.yaml
"""

import os
import sys
import argparse
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, GradientAccumulationPlugin
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

_MOTUS2_ROOT = Path(__file__).resolve().parents[1]

from data.utils.smart_io import is_remote
from models.wan_model import WanVideoModel
from data.video_pretrain import (
    VideoPretrainDataset,
    setup_dataset_warning_logging,
    video_pretrain_collate_fn,
)
from sample_video_pretrain import evaluate_video_pretrain, log_video_pretrain_metrics

WAN_ROOT = str((Path(__file__).parent.parent / "third_party").resolve())
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)
from wan.utils.fm import FlowMatchScheduler
from utils.ti2v_timestep import build_ti2v_per_token_timestep
from utils.gradient_clipping import configure_gradient_clipping, clip_gradients
from utils.video_pretrain_checkpoint import save_trainer_state, read_trainer_state, restore_scheduler

logger = logging.getLogger(__name__)


def _resolve_checkpoint_file(path: str) -> Path:
    """Resolve checkpoint dir/file to a concrete model state file."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    if p.is_file():
        return p

    candidates = [
        p / "pytorch_model" / "mp_rank_00_model_states.pt",  # DeepSpeed ZeRO save_state
        p / "pytorch_model.bin",  # accelerate save_state fallback
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Unsupported checkpoint layout under {p}. "
        "Expected one of: pytorch_model/mp_rank_00_model_states.pt, pytorch_model.bin"
    )


def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        for key in ("module", "state_dict", "model", "model_state_dict"):
            value = raw_obj.get(key)
            if isinstance(value, dict) and value:
                return value
    return raw_obj


def _choose_state_dict_variant_for_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Pick raw/stripped-key variant that best matches model keys."""
    model_keys = set(model.state_dict().keys())
    variants = [("raw", state_dict)]
    for prefix in ("module.", "model."):
        stripped = {k[len(prefix):]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}
        if stripped:
            variants.append((f"strip:{prefix}", stripped))

    best_name, best_sd = variants[0]
    best_overlap = -1
    for name, sd in variants:
        overlap = len(model_keys.intersection(sd.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name, best_sd = name, sd
    return best_name, best_sd, best_overlap


def _load_finetune_weights(model: nn.Module, checkpoint_path: str, strict: bool = False) -> None:
    """Load model-only weights for finetune (no optimizer/scheduler/global_step)."""
    ckpt_file = _resolve_checkpoint_file(checkpoint_path)
    raw_obj = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(raw_obj)
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Failed to parse state_dict from checkpoint: {ckpt_file}")

    variant_name, chosen_state_dict, overlap = _choose_state_dict_variant_for_model(model, state_dict)
    logger.info("Finetune load variant=%s matched_keys=%d", variant_name, overlap)

    missing, unexpected = model.load_state_dict(chosen_state_dict, strict=strict)
    if missing:
        logger.warning("Finetune load missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Finetune load unexpected keys: %d", len(unexpected))
    logger.info("Loaded finetune weights from %s", ckpt_file)


def _apply_distributed_env_from_config(config) -> None:
    """Before Accelerator: align with wan_new multi-node scripts (NCCL heartbeat / async errors).

    Config keys under ``system``:
    - ``distributed_timeout_sec``: seconds for TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC and TORCH_DISTRIBUTED_TIMEOUT
    - ``nccl_async_error_handling``: enable TORCH_NCCL_ASYNC_ERROR_HANDLING / NCCL_ASYNC_ERROR_HANDLING
    """
    sys_cfg = config.get("system")
    if sys_cfg is None:
        return
    if sys_cfg.get("nccl_async_error_handling", True):
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dto = sys_cfg.get("distributed_timeout_sec")
    if dto is not None:
        s = str(int(dto))
        # Override launcher defaults so yaml is authoritative when set
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = s
        os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = s


# ---------------------------------------------------------------------------
# Video Pretrain Model
# ---------------------------------------------------------------------------

class VideoPretrainModel(nn.Module):
    """Bidirectional flow matching video pretrain using WAN 2.2 TI2V 5B.

    TI2V 5B conditioning: replace first N latent frames in noisy input with
    clean (or lightly noised) latents. No y concat, no CLIP.
    """

    def __init__(
        self,
        wan_checkpoint_path: str,
        vae_path: str = "",
        wan_config_path: str = "",
        precision: str = "bfloat16",
        gradient_checkpointing: bool = False,
        gradient_checkpointing_offload: bool = False,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
        cond_frame_prob_multi: float = 0.5,
        cond_noise_prob: float = 0.5,
        cond_noise_std: float = 1.0,
        timestep_sample_method: str = "logit_normal",
    ):
        super().__init__()
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]

        self.video_model = WanVideoModel.from_pretrained(
            wan_checkpoint_path,
            vae_path=vae_path or None,
            config_path=wan_config_path or None,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_offload=gradient_checkpointing_offload,
        )

        self.scheduler = FlowMatchScheduler(
            num_inference_steps=num_train_timesteps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=1.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(num_train_timesteps, training=True)

        self.cond_frame_prob_multi = cond_frame_prob_multi
        self.cond_noise_prob = cond_noise_prob
        self.cond_noise_std = cond_noise_std
        _tsm = (timestep_sample_method or "logit_normal").lower()
        if _tsm not in ("logit_normal", "uniform"):
            raise ValueError(
                f"timestep_sample_method must be 'logit_normal' or 'uniform', got {timestep_sample_method!r}"
            )
        self.timestep_sample_method = _tsm

    def forward(self, latents, language_embeddings=None, valid_len=None):
        # Enter DDP/DeepSpeed through forward so distributed hooks run normally.
        return self.training_step(latents, language_embeddings, valid_len)

    def training_step(
        self,
        latents: torch.Tensor,           # [C=48, W, H_lat, W_lat]
        language_embeddings=None,
        valid_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        elif latents.dim() != 5:
            raise ValueError(
                f"Expected latents [C,T,H,W] or batched [B,C,T,H,W], got {tuple(latents.shape)}"
            )
        B, C, W, H, W_lat = latents.shape
        device = latents.device
        _shift = self.scheduler.shift
        valid_len_tensor = None
        if valid_len is not None:
            if isinstance(valid_len, torch.Tensor):
                valid_len_tensor = valid_len.to(device=device, dtype=torch.long).reshape(B)
            else:
                valid_len_tensor = torch.full((B,), int(valid_len), device=device, dtype=torch.long)

        # === 1. Choose conditioning per sample: 1 or 2 latent frames ===
        use_multi = torch.rand(B, device=device) < self.cond_frame_prob_multi
        n_cond_per_sample = torch.where(
            use_multi,
            torch.full((B,), 2, device=device, dtype=torch.long),
            torch.ones(B, device=device, dtype=torch.long),
        )
        if valid_len_tensor is not None:
            n_cond_per_sample = torch.where(
                valid_len_tensor <= 2,
                torch.ones_like(n_cond_per_sample),
                n_cond_per_sample,
            )
        if valid_len_tensor is not None and (valid_len_tensor <= 1).any():
            raise ValueError("Video pretrain sample has no denoising target frames (valid_len <= 1).")

        # === 2. Sample sigma for denoising ===
        # raw in (0,1) before FlowMatch shift; logit_normal matches Motus WM video branch:
        #   u ~ N(0,1), raw = 1 - sigmoid(u)
        if self.timestep_sample_method == "logit_normal":
            u = torch.randn(B, device=device)
            raw = (1.0 - torch.sigmoid(u)).clamp(0, 1)
        else:
            raw = torch.rand(B, device=device, dtype=self.dtype)
        sigma = (_shift * raw / (1 + (_shift - 1) * raw)).to(self.dtype)
        timestep = sigma * self.scheduler.num_train_timesteps

        # === 3. Add noise to all frames ===
        clean_latents = latents.to(self.dtype)
        noise = torch.randn_like(clean_latents)
        sv = sigma.view(B, 1, 1, 1, 1)
        noisy_latents = (1 - sv) * clean_latents + sv * noise
        target = noise - clean_latents

        # === 4. Condition frames: replace with clean or Gaussian-noised latents per sample ===
        cond_noise_mask = torch.rand(B, device=device) < self.cond_noise_prob
        max_n_cond = int(n_cond_per_sample.max().item())
        cond_source = clean_latents[:, :, :max_n_cond]
        if max_n_cond > 0:
            cond_noise = torch.randn_like(cond_source) * self.cond_noise_std
            cond_source = torch.where(
                cond_noise_mask.view(B, 1, 1, 1, 1),
                cond_source + cond_noise,
                cond_source,
            )
            noisy_latents[:, :, 0:1] = cond_source[:, :, 0:1]
        if max_n_cond > 1:
            use_second_cond = (n_cond_per_sample >= 2).view(B, 1, 1, 1, 1)
            noisy_latents[:, :, 1:2] = torch.where(
                use_second_cond,
                cond_source[:, :, 1:2],
                noisy_latents[:, :, 1:2],
            )

        # === 5. Only real frames enter bidirectional attention. Wan derives
        # seq_lens/grid_sizes from each unpadded sample, so FlashAttention excludes
        # padded keys and unpatchify returns only each sample's real frames.
        lengths = ([W] * B if valid_len_tensor is None else valid_len_tensor.tolist())
        if any(length < 2 or length > W for length in lengths):
            raise ValueError(f"valid_len must be in [2, {W}], got {lengths}")
        forward_frames = max(lengths)
        x_list = [noisy_latents[i, :, :length] for i, length in enumerate(lengths)]

        # T5 context
        text_len = self.video_model.wan_model.text_len
        if language_embeddings is not None:
            if isinstance(language_embeddings, torch.Tensor):
                emb = language_embeddings
                if emb.dim() == 3:
                    if emb.shape[0] == 1:
                        emb = emb[0]
                        if emb.shape[0] < text_len:
                            emb = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                        else:
                            emb = emb[:text_len]
                        context_list = [emb]
                    elif emb.shape[0] == B:
                        context_list = []
                        for i in range(B):
                            e = emb[i]
                            if e.shape[0] < text_len:
                                e = torch.cat([e, e.new_zeros(text_len - e.shape[0], e.shape[1])])
                            else:
                                e = e[:text_len]
                            context_list.append(e)
                    else:
                        raise ValueError(
                            f"language_embeddings [B,L,D] must have B={B}, got shape {tuple(emb.shape)}"
                        )
                elif emb.dim() == 2:
                    if emb.shape[0] < text_len:
                        emb = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                    else:
                        emb = emb[:text_len]
                    context_list = [emb for _ in range(B)]
                else:
                    raise ValueError(f"Unexpected language_embeddings tensor shape {tuple(emb.shape)}")
                if len(context_list) == 1 and B > 1:
                    context_list = context_list * B
            else:
                context_list = language_embeddings
        else:
            text_dim = self.video_model.wan_model.text_dim
            context_list = [torch.zeros(text_len, text_dim, device=device, dtype=self.dtype) for _ in range(B)]

        patch_size = self.video_model.wan_model.patch_size
        if any(length % patch_size[0] for length in lengths):
            raise ValueError("valid_len must be divisible by the temporal patch size")
        F_pat = forward_frames // patch_size[0]
        H_pat = H // patch_size[1]
        W_pat = W_lat // patch_size[2]
        seq_len = F_pat * H_pat * W_pat

        latent_shape = (C, forward_frames, H, W_lat)
        t_per_token = build_ti2v_per_token_timestep(
            timestep, n_cond_per_sample, seq_len, patch_size, latent_shape
        )

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            pred_list = self.video_model.wan_model(
                x=x_list,
                t=t_per_token,
                context=context_list,
                seq_len=seq_len,
                y=None,  # TI2V 5B: no y concat
            )

        # Re-pad outputs only for the masked loss, after all attention is done.
        pred = torch.stack([
            F.pad(p, (0, 0, 0, 0, 0, W - length))
            for p, length in zip(pred_list, lengths)
        ])  # [B, C, T, H, W]

        # === 6. Loss: only on non-condition, non-padding frames ===
        t_arange = torch.arange(W, device=device).unsqueeze(0)
        invalid = t_arange < n_cond_per_sample.unsqueeze(1)
        if valid_len_tensor is not None:
            invalid = invalid | (t_arange >= valid_len_tensor.unsqueeze(1))
        loss_mask = (~invalid).to(self.dtype).unsqueeze(1).unsqueeze(3).unsqueeze(4)

        mse = F.mse_loss(pred.float(), target.float(), reduction="none") * loss_mask
        denom = loss_mask.sum(dim=(1, 2, 3, 4)).clamp(min=1) * C * H * W_lat
        loss_per_sample = mse.sum(dim=(1, 2, 3, 4)) / denom

        bsmntw = self.scheduler.training_weight(timestep.cpu()).to(device=device, dtype=self.dtype)
        video_loss = (loss_per_sample * bsmntw).mean()

        return {
            "total_loss": video_loss,
            "video_loss": video_loss,
            "sigma_mean": sigma.mean(),
            "n_cond_frames_mean": n_cond_per_sample.float().mean(),
            "cond_two_frame_ratio": (n_cond_per_sample == 2).float().mean(),
            "cond_noise_ratio": cond_noise_mask.float().mean(),
            "seq_len": float(seq_len),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VideoPretrainTrainer:
    def __init__(self, model, train_dataloader, val_dataloader=None, optimizer=None, scheduler=None,
                 device="cuda", rank=0, checkpoint_dir="./checkpoints",
                 log_interval=1, save_interval=5000, val_interval=500, accelerator=None,
                 report_to="tensorboard", tb_writer=None, config=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.accelerator = accelerator
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.config = config
        self.global_step = 0

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self):
        ckpt_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}"
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(ckpt_dir))
        if self.rank == 0:
            save_trainer_state(ckpt_dir, self.global_step, self.scheduler)
        self.accelerator.wait_for_everyone()
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        state = read_trainer_state(checkpoint_path, self.scheduler)
        self.accelerator.load_state(str(Path(checkpoint_path).expanduser()))
        self.global_step = state["global_step"]
        restore_scheduler(self.scheduler, state)
        logger.info(f"Resumed from step {self.global_step}")

    def train_step(self, batch):
        self.model.train()
        did_step = False

        with self.accelerator.accumulate(self.model):
            latents = batch["latents"].to(self.device, dtype=self.dtype)
            lang = batch.get("language_embedding")
            if lang is not None:
                lang = lang.to(self.device, dtype=self.dtype)
            valid_len = batch.get("valid_len")

            loss_dict = self.model(
                latents=latents,
                language_embeddings=lang,
                valid_len=valid_len,
            )

            self.accelerator.backward(loss_dict["total_loss"])

            if self.accelerator.sync_gradients:
                did_step = True
                grad_clip = self.config.training.get("grad_clip_norm", 1.0) if self.config else 1.0
                clip_gradients(self.accelerator, self.model.parameters(), max_norm=grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

        if did_step and self.scheduler:
            self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, did_step

    def train(self, max_steps: int, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting video pretrain for {max_steps} steps")
        start_time = time.time()
        data_iter = iter(self.train_dataloader)
        epoch = 0

        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            if batch is None:
                continue

            if not hasattr(self, "_step_start"):
                self._step_start = time.time()
            metrics, did_step = self.train_step(batch)
            if not did_step:
                continue
            step_time = time.time() - self._step_start
            self._step_start = time.time()
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                if dist.is_available() and dist.is_initialized():
                    for k in list(metrics.keys()):
                        if isinstance(metrics[k], (int, float)):
                            t = torch.tensor(metrics[k], dtype=torch.float32, device=self.device)
                            dist.all_reduce(t, op=dist.ReduceOp.AVG)
                            metrics[k] = t.item()
                if self.rank == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Step {self.global_step}/{max_steps}, "
                        f"Loss: {metrics['total_loss']:.4f}, "
                        f"sigma: {metrics['sigma_mean']:.3f}, "
                        f"cond_mean: {metrics['n_cond_frames_mean']:.2f}f, "
                        f"cond2_ratio: {metrics['cond_two_frame_ratio']:.2f}, "
                        f"cond_noise: {metrics['cond_noise_ratio']:.2f}, "
                        f"seq: {metrics['seq_len']:.0f}, "
                        f"LR: {lr:.2e}, T: {step_time:.2f}s"
                    )
                    if self.tb_writer:
                        for k, v in metrics.items():
                            self.tb_writer.add_scalar(f"train/{k}", v, self.global_step)

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

            if self.val_interval > 0 and self.global_step % self.val_interval == 0 and self.val_dataloader is not None:
                if self.rank == 0:
                    val_metrics = evaluate_video_pretrain(
                        self.model,
                        self.val_dataloader,
                        self.accelerator,
                        self.config,
                    )
                    scalar_metrics = {
                        k: v for k, v in val_metrics.items()
                        if not k.startswith("_") and isinstance(v, (int, float))
                    }
                    if scalar_metrics:
                        metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in scalar_metrics.items())
                        logger.info(f"Eval step {self.global_step}: {metric_str}")
                    log_video_pretrain_metrics(val_metrics, self.tb_writer, self.global_step)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        if self.rank == 0:
            logger.info(f"Training complete: {self.global_step} steps in {time.time() - start_time:.1f}s")
        # save_state is collective under DeepSpeed; every rank must enter it.
        if self.global_step % self.save_interval != 0:
            self.save_checkpoint()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Video Pretrain")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="video_pretrain")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from Accelerate checkpoint dir (overrides config.resume.checkpoint_path if set).",
    )
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Load model-only weights for finetune (overrides config.finetune.checkpoint_path if set).",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override config.system.checkpoint_dir (shell launcher can set this to match mkdir -p).",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    resume_from = args.resume_from
    if resume_from is None:
        resume_from = OmegaConf.select(config, "resume.checkpoint_path", default=None)
    if resume_from in ("", "null", "None"):
        resume_from = None
    finetune_from = args.finetune_from
    if finetune_from is None:
        finetune_from = OmegaConf.select(config, "finetune.checkpoint_path", default=None)
    if finetune_from in ("", "null", "None"):
        finetune_from = None
    if resume_from and finetune_from:
        logger.warning("Both resume_from and finetune_from are set; finetune_from will be ignored.")
        finetune_from = None

    if args.checkpoint_dir is not None:
        if config.get("system") is None:
            config.system = OmegaConf.create({})
        config.system.checkpoint_dir = args.checkpoint_dir

    _apply_distributed_env_from_config(config)

    ds_plugin = None
    if args.deepspeed:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed)
    ckpt_dir = config.system.get("checkpoint_dir", "./checkpoints")
    tb_rel = OmegaConf.select(config, "logging.tensorboard_log_dir", default="tensorboard_logs")
    tb_root = os.path.join(ckpt_dir, tb_rel)
    project_config = ProjectConfiguration(
        project_dir=ckpt_dir,
        logging_dir=tb_root,
    )
    accelerator_kwargs = dict(
        deepspeed_plugin=ds_plugin,
        project_config=project_config,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.training.get("gradient_accumulation_steps", 1),
            sync_with_dataloader=False,
        ),
    )
    configure_gradient_clipping(accelerator_kwargs, config.training)
    accelerator = Accelerator(**accelerator_kwargs)

    rank = accelerator.process_index
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[R{rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    warn_subdir = OmegaConf.select(config, "logging.dataset_warning_log_dir", default="dataset_warnings")
    if warn_subdir not in (None, "", "null"):
        warn_path = os.path.join(ckpt_dir, str(warn_subdir), f"dataset_warnings_rank{rank}.log")
        setup_dataset_warning_logging(warn_path)
        if rank == 0:
            logger.info(f"Dataset warnings log: {warn_path}")

    # Dataset
    d = config.dataset
    c = config.common

    _elp = d.get("empty_lang_embed_path")
    if _elp in (None, "", "null"):
        empty_lang_embed_path = None
    else:
        _elp_s = str(_elp)
        if is_remote(_elp_s):
            empty_lang_embed_path = _elp_s
        else:
            _p = Path(_elp_s).expanduser()
            empty_lang_embed_path = (
                str(_p.resolve())
                if _p.is_absolute()
                else str((_MOTUS2_ROOT / _p).resolve())
            )

    dataset = VideoPretrainDataset(
        dataset_dir=d.get("dataset_dir"),
        duckdb_path=d.get("duckdb_path"),
        duckdb_table=d.get("duckdb_table", "ario_format_data"),
        use_index_cache=d.get("use_index_cache", True),
        task_mode=d.get("task_mode", "multi"),
        task_name=d.get("task_name", None),
        max_latent_window=c.get("max_latent_window", 10),
        max_episodes=d.get("max_episodes", None),
        cache_dir=d.get("cache_dir", None),
        dataset_exclude=d.get("dataset_exclude", None),
        dataset_include=d.get("dataset_include", None),
        index_cache_path=d.get("index_cache_path", None),
        latent_root=d.get("latent_root", "video_features/wan2.2"),
        latent_file_name=d.get("latent_file_name", "video_3v_VAE.pt"),
        latent_file_name_by_dataset=d.get("latent_file_name_by_dataset", None),
        lang_root=d.get("lang_root", "text_features/UMT5"),
        lang_subpath=d.get("lang_subpath", "text_features/UMT5/environment_robot_behavior.pt"),
        latent_phases=d.get("latent_phases", None),
        lang_uncond_prob=c.get("lang_uncond_prob", 0.0),
        empty_lang_embed_path=empty_lang_embed_path,
    )
    val_dataset = None
    if config.get("evaluation", {}).get("enabled", True):
        val_dataset = VideoPretrainDataset(
            dataset_dir=d.get("dataset_dir"),
            duckdb_path=d.get("duckdb_path"),
            duckdb_table=d.get("duckdb_table", "ario_format_data"),
            use_index_cache=d.get("use_index_cache", True),
            task_mode=d.get("task_mode", "multi"),
            task_name=d.get("task_name", None),
            max_latent_window=c.get("max_latent_window", 10),
            max_episodes=d.get("val_max_episodes", d.get("max_episodes", None)),
            cache_dir=d.get("cache_dir", None),
            dataset_exclude=d.get("dataset_exclude", None),
            dataset_include=d.get("dataset_include", None),
            index_cache_path=d.get("index_cache_path", None),
            latent_root=d.get("latent_root", "video_features/wan2.2"),
            latent_file_name=d.get("latent_file_name", "video_3v_VAE.pt"),
            latent_file_name_by_dataset=d.get("latent_file_name_by_dataset", None),
            lang_root=d.get("lang_root", "text_features/UMT5"),
            lang_subpath=d.get("lang_subpath", "text_features/UMT5/environment_robot_behavior.pt"),
            latent_phases=d.get("latent_phases", None),
            lang_uncond_prob=c.get("lang_uncond_prob", 0.0),
            empty_lang_embed_path=empty_lang_embed_path,
            val=True,
        )

    per_device_batch_size = int(config.training.get("batch_size", 1))
    if per_device_batch_size < 1:
        raise ValueError("training.batch_size must be >= 1")

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=config.system.get("num_workers", 4),
        collate_fn=video_pretrain_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.system.get("num_workers", 4),
            collate_fn=video_pretrain_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    # Model
    m = config.model
    timestep_sample_method = c.get("timestep_sample_method")
    if timestep_sample_method is None:
        leg = c.get("timestep_raw_sampling")
        if leg is not None:
            ls = str(leg).lower()
            timestep_sample_method = "logit_normal" if ls == "wm" else ls
    if not timestep_sample_method:
        timestep_sample_method = "logit_normal"

    model = VideoPretrainModel(
        wan_checkpoint_path=m.wan.checkpoint_path,
        vae_path=m.wan.get("vae_path", ""),
        wan_config_path=m.wan.get("config_path", ""),
        precision=m.wan.get("precision", "bfloat16"),
        gradient_checkpointing=m.wan.get("gradient_checkpointing", False),
        gradient_checkpointing_offload=m.wan.get("gradient_checkpointing_offload", False),
        shift=c.get("shift", 5.0),
        cond_frame_prob_multi=c.get("cond_frame_prob_multi", 0.5),
        cond_noise_prob=c.get("cond_noise_prob", 0.5),
        cond_noise_std=c.get("cond_noise_std", 1.0),
        timestep_sample_method=timestep_sample_method,
    )
    if finetune_from:
        logger.info(f"Loading finetune weights from {finetune_from} (model only)...")
        _load_finetune_weights(model, finetune_from, strict=False)

    # Freeze VAE (not used, latents are pre-computed), train DiT
    # video_model.vae is VAEWrapper (not nn.Module); underlying weights live on .vae (diffusers) or .vae.model (pth).
    vw = model.video_model.vae
    if vw.vae_type == "diffusers":
        for p in vw.vae.parameters():
            p.requires_grad = False
    else:
        for p in vw.vae.model.parameters():
            p.requires_grad = False
    for p in model.video_model.wan_model.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.get("learning_rate", 1e-5),
        weight_decay=config.training.get("weight_decay", 0.01),
    )
    from utils.scheduler import create_scheduler
    lr_scheduler = create_scheduler(optimizer, config) if hasattr(config, "training") else None

    to_prepare = [model, optimizer, dataloader]
    if val_dataloader is not None:
        to_prepare.append(val_dataloader)
    prepared = accelerator.prepare(*to_prepare)
    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = prepared
    else:
        model, optimizer, dataloader = prepared
    if lr_scheduler is not None:
        # Explicit trainer_state.pt saves the counter even for plain Python schedulers.
        lr_scheduler.optimizer = optimizer
        if hasattr(lr_scheduler, "inner"):
            lr_scheduler.inner.optimizer = optimizer

    tb_writer = None
    if rank == 0 and "tensorboard" in args.report_to:
        os.makedirs(tb_root, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_root))

    trainer = VideoPretrainTrainer(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device=accelerator.device,
        rank=rank,
        checkpoint_dir=config.system.get("checkpoint_dir", "./checkpoints"),
        log_interval=config.system.get("log_interval", 1),
        save_interval=config.system.get("save_interval", 5000),
        val_interval=config.get("evaluation", {}).get("interval", 500),
        accelerator=accelerator,
        report_to=args.report_to,
        tb_writer=tb_writer,
        config=config,
    )

    trainer.train(
        max_steps=config.training.get("max_steps", 100000),
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
