# Motus AR Architecture
# Chunk-wise parallel training + streaming KV cache inference.
# Video (WAN) + Action, no VLM/UndExpert.

import os
import sys
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from torch.distributions import Beta

WAN_ROOT = str((Path(__file__).parent.parent / "third_party").resolve())
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm import FlowMatchScheduler

from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .varlen_sample_preprocessor import MotusVarlenSamplePreprocessor
from .rope import (
    rope_params,
    rope_video_basis_freqs,
    rope_action_apply,
    causal_rope_action_apply,
)
from .block_attention import (
    create_sliding_window_mask,
    create_memory_window_mask,
    BLHD_flex_with_mask,
    build_fill_cache_mask,
)
from .memory_noise import conditioning_noise_sigma
from .memory_tokens import GistTokens
from infra.varlen.attention import varlen_block_attention
from infra.varlen.factored_attention import factored_memory_attention
from infra.varlen.preprocessor import build_memory_varlen_batch
from infra.varlen.packing import pack_streams, pack_named_streams, unpack_named_streams
from infra.varlen.rope import assemble_packed_freqs
from infra.varlen.types import VarlenBatchMeta
from utils.relative_eef import (
    abs_eef_20d_to_relative_eef_20d_torch,
    abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch,
    abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch,
    abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
)

logger = logging.getLogger(__name__)


def _angle_mse_loss_wrap(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token wrapped-angle MSE using atan2(sin d, cos d)."""
    delta = pred.float() - target.float()
    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
    return wrapped.pow(2).mean(dim=-1).to(pred.dtype)


def _abs_joint_to_rel_joint(state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
    """Convert absolute joint targets to chunk-anchor-relative joints."""
    return action_chunk - state.unsqueeze(-2)


def _quat_mse_loss_sign_invariant(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token quaternion loss invariant to q <-> -q."""
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target.float(), dim=-1)
    mse_pos = (pred_n - target_n).pow(2).mean(dim=-1)
    mse_neg = (pred_n + target_n).pow(2).mean(dim=-1)
    return torch.minimum(mse_pos, mse_neg).to(pred.dtype)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MotusConfig:
    device: str = "cuda"
    # WAN model
    wan_checkpoint_path: str = ""
    vae_path: str = ""  # Optional: auto-inferred from checkpoint_path if not specified
    wan_config_path: str = ""  # Optional: auto-inferred from checkpoint_path if not specified
    video_precision: str = "bfloat16"
    gradient_checkpointing: bool = False

    # Video dimensions
    video_height: int = 384
    video_width: int = 320

    # Sliding-window AR params
    video_action_freq_ratio: int = 6   # action 30Hz, video 5Hz
    action_upsample_ratio: int = 1     # 1 = no upsampling, 4 = interpolate to 120Hz
    max_latent_window: int = 10        # W: training window in latent frames
    fixed_chunk_size: int = 2          # fixed latent chunk size for training and inference

    # Action expert
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-6

    # AR modes
    use_state: bool = True
    state_mask_ratio: float = 0.0
    idm_mode: bool = False
    action_mode: str = "joint"        # "joint", "rel_joint", "rel_eef"(16D xyzw), "rel_eef_6d"(20D rot6d), or "abs_eef_rpy"(14D)
    mode: str = "joint"               # "joint", "video_idm", "action_only", or "loop"

    # Hybrid memory (anchor + gist + recent), always on. anchor_latents=0 means
    # no permanent L0 full pin (open path); >0 keeps the first N latents as anchors.
    # recent_latents replaces the old chunk_window_size; the KV cache capacity is
    # derived from anchor + recent (+ gist), no separate attn_window needed.
    memory_enabled: bool = True
    memory_anchor_latents: int = 0
    memory_gist_tokens_per_frame: int = 8
    memory_recent_latents: int = 1
    memory_prefix_frames: int = 0     # >0: extra leading warm-up latents (loss-masked) for long-horizon prefix-prefill
    # Training-only random shift of every global-RoPE position in a sample, in latents.
    # Tokens with identity RoPE (the per-chunk state token) read the absolute phase of the
    # rotated keys they attend to, i.e. the time since the episode began; a random start per
    # sample makes that phase uninformative. Deployment keeps start 0, inside the trained range.
    memory_train_rope_offset_max: int = 128
    # Random retention: each training sample keeps a random set of clean frames
    # visible beyond the recent window (all of them with retention_full_prob, none
    # with retention_empty_prob, otherwise 1..retention_max_frames drawn uniformly).
    # retention_max_frames=0 disables it.
    memory_retention_max_frames: int = 0
    memory_retention_full_prob: float = 0.0
    memory_retention_empty_prob: float = 0.0
    # Deployment-time selector deciding which frames leaving the recent window stay
    # in the KV cache: "none" (recent window only), "all", "fifo", "first", "random",
    # "uniform", or "gate". Budget counts retained frames beyond the recent window.
    memory_selector: str = "none"
    memory_selector_budget: int = 0
    memory_selector_seed: int = 0
    memory_selector_gate_path: str = ""
    memory_selector_gate_sha256: str = ""

    # Dual-t sampling
    policy_timestep_method: str = "beta"  # "beta" or "logit_normal"
    beta_alpha: float = 7.0
    beta_beta: float = 1.0
    policy_sigmoid_scale: float = 1.0
    policy_min_t: float = 0.0
    policy_max_t: float = 1.0
    action_timestep_method: str = "logit_normal"  # "beta" or "logit_normal"
    action_sigmoid_scale: float = 1.0
    action_min_t: float = 0.0
    action_max_t: float = 1.0
    video_timestep_shift: float = 5.0
    action_timestep_shift: float = 1.0
    wm_ratio: float = 0.2

    # Action-register positional basis. video_shared matches the successful
    # vanilla memory path by placing actions on the absolute latent video clock.
    action_rope_mode: str = "chunk_local"

    # Training
    batch_size: int = 8
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    warmup_steps: int = 0
    uniform_loss_weight_warmup_steps: int = 0  # Warm up EEF pos/rot/grip weights to 1/1/1 for first N steps
    rel_eef_position_loss_weight: float = 1.0
    rel_eef_rotation_loss_weight: float = 1.0
    rel_eef_gripper_loss_weight: float = 1.0
    hand_qpos_loss_weight: float = 1.0
    # Wrist relative-eef xyz normalization (orthogonal to action_mode layout).
    wrist_xyz_normalization: str = "min_max"  # mean_std | min_max | quantile
    qpos20_wrist_xyz_min: tuple = (-0.05, -0.05, -0.05, -0.05, -0.05, -0.05)
    qpos20_wrist_xyz_max: tuple = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    qpos20_wrist_xyz_mean: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qpos20_wrist_xyz_std: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    wrist_xyz_std_min: float = 1e-4

    # Self-forcing (MemoryWAM §4.1 / Huang et al. Self Forcing): per-frame noise on clean
    # conditioning latents. mixed = (1-alpha)*clean + alpha*noise, alpha ~ Uniform(0, sigma_max).
    # prob=1.0 + sigma_max=1.0 matches MemoryWAM; sigma_max<=0 disables.
    noisy_history_prob: float = 0.5       # apply probability per clean frame (1.0 = always)
    noisy_history_sigma_max: float = 0.1  # max mix ratio alpha, Uniform(0, sigma_max)
    # Leading conditioning latents left verbatim. Their full tokens leave the cache once they
    # fall out of the recent window, so their gist is the only permanent record of the opening
    # scene; noising them corrupts that record before it is written. 0 is MemoryWAM's uniform
    # schedule. This is an augmentation boundary, deliberately independent of the attention
    # budget in memory.anchor_latents.
    noisy_history_clean_prefix: int = 0

    # Inference
    num_inference_timesteps: int = 10
    video_steps: Optional[int] = None  # joint denoise steps before action-only IDM phase
    # Decoupled inference: video stays noisy, action fully denoises
    decouple_inference_noise: bool = True
    video_inference_final_noise: float = 0.8

    # Value prediction
    value_enabled: bool = False
    value_tokens_per_frame: int = 0       # K value query tokens per chunk (0 = disabled)
    value_num_bins: int = 201
    value_range: tuple = (0.0, 1.0)
    value_target: str = "progress"        # "progress", "ttc", or "vlac"
    value_loss_weight: float = 1.0
    vm_ratio: float = 0.0                 # Value mode ratio (0 = disabled)
    progress_c: float = 0.01              # Adaptive gamma parameter for progress targets

    # rel_eef gripper normalization range (must match dataset)
    gripper_min: float = 0.0
    gripper_max: float = 1.0

    # Misc
    load_pretrained_backbones: Optional[bool] = None
    # Packed varlen self-attn backend: auto | sdpa_planned | flash_attn_varlen | flex.
    # Env MOTUS2_VARLEN_ATTN=flex forces the legacy Flex block-mask path.
    varlen_attention_backend: str = "auto"

    def __post_init__(self):
        if self.value_target not in {"progress", "ttc", "vlac"}:
            raise ValueError(f"Unsupported value_target: {self.value_target}")
        raw_mode = str(self.mode).lower()
        if raw_mode not in {"joint", "video_idm", "action_only", "loop"}:
            raise ValueError(f"Unsupported mode: {raw_mode}")
        self.mode = raw_mode
        self.wrist_xyz_normalization = str(self.wrist_xyz_normalization or "min_max")
        self.memory_anchor_latents = int(self.memory_anchor_latents)
        self.memory_gist_tokens_per_frame = int(self.memory_gist_tokens_per_frame)
        self.memory_recent_latents = int(self.memory_recent_latents)
        self.memory_prefix_frames = int(self.memory_prefix_frames)
        self.memory_train_rope_offset_max = int(self.memory_train_rope_offset_max)
        if self.memory_train_rope_offset_max < 0:
            raise ValueError("memory_train_rope_offset_max must be >= 0")
        self.memory_retention_max_frames = int(self.memory_retention_max_frames)
        self.memory_retention_full_prob = float(self.memory_retention_full_prob)
        self.memory_retention_empty_prob = float(self.memory_retention_empty_prob)
        if self.memory_retention_max_frames < 0:
            raise ValueError("memory_retention_max_frames must be >= 0")
        if not (0.0 <= self.memory_retention_full_prob and 0.0 <= self.memory_retention_empty_prob
                and self.memory_retention_full_prob + self.memory_retention_empty_prob <= 1.0):
            raise ValueError("memory_retention_full_prob + memory_retention_empty_prob must lie in [0, 1]")
        self.memory_selector = str(self.memory_selector or "none")
        if self.memory_selector not in {"none", "all", "fifo", "first", "random", "uniform", "gate"}:
            raise ValueError(f"Unsupported memory_selector: {self.memory_selector}")
        self.memory_selector_budget = int(self.memory_selector_budget)
        self.memory_selector_seed = int(self.memory_selector_seed)
        if self.memory_selector not in {"none", "all"} and self.memory_selector_budget < 1:
            raise ValueError("memory_selector_budget must be >= 1 for a budgeted selector")
        if self.memory_selector == "gate" and not self.memory_selector_gate_path:
            raise ValueError("memory_selector=gate requires memory_selector_gate_path")
        if self.memory_anchor_latents < 0:
            raise ValueError("memory_anchor_latents must be >= 0")
        # gist can be fully disabled (0): GistTokens(M=0) is an empty parameter -> no-op.
        if self.memory_gist_tokens_per_frame < 0:
            raise ValueError("memory_gist_tokens_per_frame must be >= 0")
        if self.memory_recent_latents < 1:
            raise ValueError("memory_recent_latents must be >= 1")
        if self.memory_prefix_frames < 0:
            raise ValueError("memory_prefix_frames must be >= 0")
        if not self.memory_enabled:
            raise ValueError("The varlen MemoryWAM trainer requires memory.enabled=true")
        self.action_rope_mode = str(self.action_rope_mode)
        if self.action_rope_mode not in {"chunk_local", "video_shared"}:
            raise ValueError(
                "action_rope_mode must be 'chunk_local' or 'video_shared', "
                f"got {self.action_rope_mode!r}"
            )
        self.varlen_attention_backend = str(
            getattr(self, "varlen_attention_backend", "auto") or "auto"
        ).lower()
        if self.varlen_attention_backend not in {
            "auto",
            "sdpa_planned",
            "flash_attn_varlen",
            "fa",
            "flex",
            "flex_attention",
        }:
            raise ValueError(
                f"Unsupported varlen_attention_backend: {self.varlen_attention_backend}"
            )
        if self.video_steps is not None and not self.idm_mode:
            raise ValueError("video_steps requires idm_mode=True for stable action-only phase.")
        self.fixed_chunk_size = int(self.fixed_chunk_size)
        if self.fixed_chunk_size < 1:
            raise ValueError(f"fixed_chunk_size must be >= 1, got {self.fixed_chunk_size}")
        if self.action_mode in {"joint", "rel_joint", "abs_eef_rpy"}:
            self.action_dim = 14
            self.action_state_dim = 14
        elif self.action_mode in {"rel_eef", "rel_eef_wxyz"}:
            self.action_dim = 16
            self.action_state_dim = 16
        elif self.action_mode in {"rel_eef_6d", "abs_eef"}:
            self.action_dim = 20
            self.action_state_dim = 20
        elif self.action_mode == "abs_eef_wxyz":
            self.action_dim = 16
            self.action_state_dim = 16
        elif self.action_mode == "qpos20_rel6d":
            self.action_dim = 58
            self.action_state_dim = 58
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        if self.value_target == "progress":
            self.value_range = (0.0, 1.0)
        elif self.value_target == "vlac":
            self.value_range = (-1.0, 1.0)
        else:
            self.value_range = (-1.0, 0.0)
        W = self.max_latent_window
        self.actions_per_latent = 4 * self.video_action_freq_ratio * self.action_upsample_ratio
        self.action_tokens_per_frame = self.actions_per_latent
        self.state_tokens_per_frame = 1 if self.use_state else 0
        self.noisy_action_tokens_per_frame = (
            self.value_tokens_per_frame +
            self.state_tokens_per_frame +
            self.action_tokens_per_frame
        )


# ---------------------------------------------------------------------------
# Motus Model
# ---------------------------------------------------------------------------

class Motus(nn.Module):
    """AR model: parallel chunk-wise training + streaming inference."""

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16

        # --- Video model (WAN) ---
        load_bb = config.load_pretrained_backbones is None or bool(config.load_pretrained_backbones)
        if load_bb:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path if config.vae_path else None,
                config_path=config.wan_config_path if config.wan_config_path else None,
                device=config.device,
                precision=config.video_precision,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        else:
            # When not loading backbones (resume/finetune), still need config + VAE for structure.
            # Diffusers format: config in transformer/, vae in vae/; standard: config.json at root.
            ckpt = config.wan_checkpoint_path or ""
            cfg_path = config.wan_config_path or ckpt
            vae_path = config.vae_path
            if not vae_path and ckpt:
                if os.path.exists(os.path.join(ckpt, "model_index.json")):
                    cfg_path = os.path.join(ckpt, "transformer")
                    vae_path = os.path.join(ckpt, "vae")
                else:
                    vae_path = os.path.join(ckpt, "Wan2.2_VAE.pth")
            self.video_model = WanVideoModel.from_config(
                config_path=cfg_path,
                vae_path=vae_path or "",
                device=config.device,
                precision=config.video_precision,
                gradient_checkpointing=config.gradient_checkpointing,
            )

        wan_dim = getattr(self.video_model.wan_model.config, "dim", 3072)
        wan_heads = getattr(self.video_model.wan_model.config, "num_heads", 24)
        wan_hd = wan_dim // wan_heads
        wan_cfg = {"dim": wan_dim, "num_heads": wan_heads, "head_dim": wan_hd}
        self.wan_dim = wan_dim

        # --- Action expert ---
        W = config.max_latent_window
        N = W - self._obs_latents()  # noisy side starts after the fixed observation latents
        total_action_tokens = N * config.noisy_action_tokens_per_frame
        act_cfg = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=total_action_tokens,
            video_feature_dim=wan_dim,
            eps=config.action_expert_norm_eps,
            use_mot=False,
            use_state=config.use_state,
            action_mode=config.action_mode,
        )
        self.action_expert = ActionExpert(act_cfg, wan_cfg)

        # --- Hybrid memory (gist) tokens: always present ---
        self.gist_tokens = GistTokens(wan_dim, config.memory_gist_tokens_per_frame)

        # --- Value prediction ---
        if config.value_enabled:
            from .value_head import ValueTokens, ValueHead
            self.value_tokens = ValueTokens(wan_dim, config.value_tokens_per_frame)
            self.value_head = ValueHead(wan_dim, config.value_num_bins, config.value_range)

        # --- Device / dtype ---
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)
        self.gist_tokens.to(device=self.device, dtype=self.dtype)
        if config.value_enabled:
            self.value_tokens.to(device=self.device, dtype=self.dtype)
            self.value_head.to(device=self.device, dtype=self.dtype)

        # --- Pre-compute grid sizes ---
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        self.frame_seqlen = lat_H * lat_W  # tokens per latent frame (after patch_embed)
        self.lat_H = lat_H
        self.lat_W = lat_W

        # --- RoPE frequency tables ---
        d = wan_dim // wan_heads
        self.freqs = self.video_model.wan_model.freqs.clone()
        # Buffer must cover max_latent_start * tokens_per_frame across all episodes.
        # action_upsample_ratio=4 → napf=96, so 1024*10=10240 is too small for
        # episodes with >106 latent frames when train_global_rope=True.
        self.freqs_action = rope_params(1024 * 100, d)
        self.freqs_state = rope_params(1024 * 10, d)

        # --- Flow-Matching schedulers ---
        self.scheduler = FlowMatchScheduler(
            shift=config.video_timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000,
        )
        self.action_scheduler.set_timesteps(1000, training=True)

        # --- Policy timestep distribution (pre-compute for logit_normal) ---
        if config.policy_timestep_method == "logit_normal":
            from utils.common import get_t_distribution, slice_t_distribution
            _cfg = type("_C", (), {"timestep_sample_method": "logit_normal",
                                   "sigmoid_scale": config.policy_sigmoid_scale})()
            t_dist = get_t_distribution(_cfg)
            t_dist = slice_t_distribution(t_dist, config.policy_min_t, config.policy_max_t)
            self.register_buffer("_policy_t_dist", t_dist)
        if config.action_timestep_method == "logit_normal":
            from utils.common import get_t_distribution, slice_t_distribution
            _cfg = type("_C", (), {"timestep_sample_method": "logit_normal",
                                   "sigmoid_scale": config.action_sigmoid_scale})()
            t_dist = get_t_distribution(_cfg)
            t_dist = slice_t_distribution(t_dist, config.action_min_t, config.action_max_t)
            self.register_buffer("_action_t_dist", t_dist)

        # --- Attention mask (cached, created on first forward) ---
        self._ar_mask = None
        self._ar_mask_padded_len: Optional[int] = None

        # --- KV cache for inference ---
        self.kv_caches: Optional[List[torch.Tensor]] = None
        self.current_start_frame: int = 0

        self._log_params()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_params(self):
        """Log detailed parameter counts for each component (same style as Motus1)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        logger.info("Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")

    def _qpos20_wrist_xyz_affine(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_min = getattr(self.config, "qpos20_wrist_xyz_min", None)
        xyz_max = getattr(self.config, "qpos20_wrist_xyz_max", None)
        if xyz_min is None or xyz_max is None:
            scale_value = float(getattr(self.config, "qpos20_wrist_xyz_scale", 0.05))
            if scale_value <= 0:
                raise ValueError(f"qpos20_wrist_xyz_scale must be positive, got {scale_value}")
            xyz_min = [-scale_value] * 6
            xyz_max = [scale_value] * 6
        xyz_min_t = torch.as_tensor(xyz_min, device=device, dtype=dtype).flatten()
        xyz_max_t = torch.as_tensor(xyz_max, device=device, dtype=dtype).flatten()
        if xyz_min_t.numel() == 1:
            xyz_min_t = xyz_min_t.expand(6)
        if xyz_max_t.numel() == 1:
            xyz_max_t = xyz_max_t.expand(6)
        if xyz_min_t.numel() != 6 or xyz_max_t.numel() != 6:
            raise ValueError("qpos20_wrist_xyz_min/max must each contain 6 values: left xyz + right xyz")
        xyz_range = (xyz_max_t - xyz_min_t).clamp(min=1e-6)
        scale = 1.0 / xyz_range
        offset = -scale * xyz_min_t
        return scale, offset, xyz_min_t, xyz_max_t

    def _qpos20_wrist_xyz_norm_mode(self) -> str:
        mode = getattr(self.config, "wrist_xyz_normalization", None)
        return str(mode or "min_max").lower()

    def _qpos20_wrist_xyz_mean_std(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz_mean = getattr(self.config, "qpos20_wrist_xyz_mean", None)
        xyz_std = getattr(self.config, "qpos20_wrist_xyz_std", None)
        if xyz_mean is None or xyz_std is None:
            raise ValueError("qpos20_wrist_xyz_mean/std required for wrist_xyz_normalization=mean_std")
        mean_t = torch.as_tensor(xyz_mean, device=device, dtype=dtype).flatten()
        std_t = torch.as_tensor(xyz_std, device=device, dtype=dtype).flatten()
        if mean_t.numel() == 1:
            mean_t = mean_t.expand(6)
        if std_t.numel() == 1:
            std_t = std_t.expand(6)
        if mean_t.numel() != 6 or std_t.numel() != 6:
            raise ValueError("qpos20_wrist_xyz_mean/std must each contain 6 values: left xyz + right xyz")
        std_min = float(getattr(self.config, "wrist_xyz_std_min", 1e-4))
        return mean_t, std_t.clamp(min=std_min)

    def _qpos20_wrist_xyz_quantile(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q01 = getattr(self.config, "qpos20_wrist_xyz_min", None)
        q99 = getattr(self.config, "qpos20_wrist_xyz_max", None)
        if q01 is None or q99 is None:
            raise ValueError("qpos20_wrist_xyz_min/max required for wrist_xyz_normalization=quantile")
        q01_t = torch.as_tensor(q01, device=device, dtype=dtype).flatten()
        q99_t = torch.as_tensor(q99, device=device, dtype=dtype).flatten()
        if q01_t.numel() == 1:
            q01_t = q01_t.expand(6)
        if q99_t.numel() == 1:
            q99_t = q99_t.expand(6)
        if q01_t.numel() != 6 or q99_t.numel() != 6:
            raise ValueError("qpos20_wrist_xyz_min/max must each contain 6 values for quantile mode")
        return q01_t, (q99_t - q01_t).clamp(min=1e-6)

    def _extract_qpos20_rel_wrist_xyz6(self, rel_actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([rel_actions[..., 20:23], rel_actions[..., 49:52]], dim=-1)

    def _write_qpos20_rel_wrist_xyz6(self, rel_actions: torch.Tensor, xyz6: torch.Tensor) -> torch.Tensor:
        out = rel_actions.clone()
        out[..., 20:23] = xyz6[..., :3]
        out[..., 49:52] = xyz6[..., 3:]
        return out

    def _scale_qpos20_rel_wrist_xyz(self, rel_actions: torch.Tensor) -> torch.Tensor:
        """Normalize qpos20_rel6d relative wrist xyz (meters) for model training."""
        if self.config.action_mode != "qpos20_rel6d":
            return rel_actions
        mode = self._qpos20_wrist_xyz_norm_mode()
        xyz6 = self._extract_qpos20_rel_wrist_xyz6(rel_actions)
        if mode == "mean_std":
            mean, std = self._qpos20_wrist_xyz_mean_std(xyz6.device, xyz6.dtype)
            xyz6 = (xyz6 - mean) / std
        elif mode == "quantile":
            q01, q_range = self._qpos20_wrist_xyz_quantile(xyz6.device, xyz6.dtype)
            xyz6 = torch.minimum(torch.maximum(xyz6, q01), q01 + q_range)
            xyz6 = (xyz6 - q01) / q_range
        else:
            scale, offset, xyz_min, xyz_max = self._qpos20_wrist_xyz_affine(xyz6.device, xyz6.dtype)
            xyz6 = torch.minimum(torch.maximum(xyz6, xyz_min), xyz_max)
            xyz6 = xyz6 * scale + offset
        return self._write_qpos20_rel_wrist_xyz6(rel_actions, xyz6)

    def _unscale_qpos20_rel_wrist_xyz(self, rel_actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized qpos20_rel6d wrist xyz back to meters."""
        if self.config.action_mode != "qpos20_rel6d":
            return rel_actions
        mode = self._qpos20_wrist_xyz_norm_mode()
        xyz6 = self._extract_qpos20_rel_wrist_xyz6(rel_actions)
        if mode == "mean_std":
            mean, std = self._qpos20_wrist_xyz_mean_std(xyz6.device, xyz6.dtype)
            xyz6 = xyz6 * std + mean
        elif mode == "quantile":
            q01, q_range = self._qpos20_wrist_xyz_quantile(xyz6.device, xyz6.dtype)
            xyz6 = xyz6 * q_range + q01
        else:
            scale, offset, _, _ = self._qpos20_wrist_xyz_affine(xyz6.device, xyz6.dtype)
            xyz6 = (xyz6 - offset) / scale
        return self._write_qpos20_rel_wrist_xyz6(rel_actions, xyz6)

    def _slice_noisy_actions_and_states(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        chunk_size: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Slice the noisy suffix for the current chunk size.

        For rel_eef, dataset actions/states are absolute 20D rot6d EEF. Convert to 20D
        relative EEF only after chunk_size has been chosen so each latent chunk uses its
        own aligned state anchor. abs_eef keeps the absolute 20D rot6d representation.
        abs_eef_wxyz keeps the absolute 16D wxyz-quaternion representation.
        """
        # Preserve FP32 geometry even inside packed/outer AMP regions.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            target_dtype = dtype or actions.dtype
            apl = self.config.actions_per_latent

            noisy_actions = actions[:, chunk_size * apl:].float()
            noisy_states = states[:, chunk_size:].float() if states is not None else None

            if self.config.action_mode not in {"rel_joint", "rel_eef", "rel_eef_6d", "rel_eef_wxyz", "qpos20_rel6d"}:
                if not self.config.use_state:
                    noisy_states = None
                return noisy_actions.to(target_dtype), (noisy_states.to(target_dtype) if noisy_states is not None else None)

            if noisy_states is None:
                raise ValueError(f"{self.config.action_mode} mode requires absolute state anchors for conversion")

            B = noisy_actions.shape[0]
            N = noisy_states.shape[1]
            noisy_states_fp32 = noisy_states.float()
            noisy_action_chunks_fp32 = noisy_actions.view(B, N, apl, -1).float()
            if self.config.action_mode == "rel_joint":
                rel_actions = _abs_joint_to_rel_joint(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                ).reshape(B, N * apl, -1)
            elif self.config.action_mode == "rel_eef_6d":
                rel_actions = abs_eef_20d_to_relative_eef_20d_torch(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                    self.config.gripper_min,
                    self.config.gripper_max,
                ).reshape(B, N * apl, -1)
            elif self.config.action_mode == "rel_eef_wxyz":
                rel_actions = abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                    self.config.gripper_min,
                    self.config.gripper_max,
                ).reshape(B, N * apl, -1)
            elif self.config.action_mode == "qpos20_rel6d":
                rel_actions = abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                ).reshape(B, N * apl, -1)
                rel_actions = self._scale_qpos20_rel_wrist_xyz(rel_actions)
            else:
                rel_actions = abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                    self.config.gripper_min,
                    self.config.gripper_max,
                ).reshape(B, N * apl, -1)

            if not self.config.use_state:
                noisy_states = None
            return rel_actions.to(target_dtype), (noisy_states.to(target_dtype) if noisy_states is not None else None)

    def _chunk_latent_counts(self, num_noisy_frames: int, chunk_size: int) -> List[int]:
        """Split the noisy suffix into chunk_size-sized blocks."""
        counts = []
        pos = 0
        while pos < num_noisy_frames:
            counts.append(min(chunk_size, num_noisy_frames - pos))
            pos += chunk_size
        return counts

    def _chunk_end_indices(self, block_latent_counts: List[int]) -> List[int]:
        """Return chunk-local end indices into the flattened noisy latent axis."""
        out = []
        offset = 0
        for count in block_latent_counts:
            offset += count
            out.append(offset - 1)
        return out

    def _fixed_chunk_size(self) -> int:
        return int(self.config.fixed_chunk_size)

    def _memory_anchor_frames(self) -> int:
        return max(0, int(self.config.memory_anchor_latents))

    def _memory_recent_frames(self) -> int:
        return max(1, int(self.config.memory_recent_latents))

    def _memory_gist_tokens(self) -> int:
        return max(0, int(self.config.memory_gist_tokens_per_frame))


    def _obs_latents(self) -> int:
        """Pure-observation latents at the window head (never predicted).

        Always 1: cold start has a single real C0 (no L0→chunk_size repeat).
        Predict chunks of ``fixed_chunk_size`` begin immediately after that obs.
        """
        return 1


    def _num_anchor_frames(self) -> int:
        return 0

    def _max_full_cache_tokens(self) -> int:
        """Capacity (in tokens) of the full-frame portion of the hybrid KV cache:
        anchor frames + the recent sliding window, both at full resolution."""
        full_frames = self._memory_anchor_frames() + self._memory_recent_frames()
        return int(full_frames) * self.frame_seqlen

    def _max_cache_tokens(self) -> int:
        # Legacy name; with hybrid memory the streaming cache holds
        # anchor_full + gist(all) + recent_full and is managed in _fill_cache.
        return self._max_full_cache_tokens()

    def _select_chunk_end_values(
        self,
        noisy_values: torch.Tensor,
        block_latent_counts: List[int],
    ) -> torch.Tensor:
        """Select one value target per chunk: the last latent in that chunk."""
        end_idx = self._chunk_end_indices(block_latent_counts)
        if not end_idx:
            return noisy_values[:, :0]
        idx = torch.tensor(end_idx, device=noisy_values.device, dtype=torch.long)
        return noisy_values.index_select(1, idx)

    def _compute_action_loss_terms(
        self,
        action_pred: torch.Tensor,
        target_action: torch.Tensor,
        action_token_mask: torch.Tensor,
        *,
        global_step: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample action loss using the same component split as training_step.

        ``action_pred`` / ``target_action`` are [B, T, D]. ``action_token_mask`` is
        [B, T] and already encodes WM/VM/irrelevant/prefix/valid-len masking.
        """

        device = action_pred.device
        eef_component_warmup_active = (
            self.config.action_mode in {
                "rel_eef", "rel_eef_6d", "abs_eef_rpy",
                "abs_eef", "abs_eef_wxyz", "rel_eef_wxyz", "qpos20_rel6d",
            }
            and global_step < self.config.uniform_loss_weight_warmup_steps
        )
        eef_pos_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_position_loss_weight
        eef_rot_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_rotation_loss_weight
        eef_grip_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_gripper_loss_weight
        hand_qpos_weight = 1.0 if eef_component_warmup_active else self.config.hand_qpos_loss_weight

        pos_loss_per_token = torch.zeros_like(action_token_mask)
        rot_loss_per_token = torch.zeros_like(action_token_mask)
        grip_loss_per_token = torch.zeros_like(action_token_mask)
        is_eef_mode = self.config.action_mode in {
            "rel_eef", "rel_eef_6d", "abs_eef_rpy",
            "abs_eef", "abs_eef_wxyz", "rel_eef_wxyz", "qpos20_rel6d",
        }

        if self.config.action_mode == "rel_eef":
            # 16D xyzw relative EEF: [xyz, quat_xyzw, grip] x 2.
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 8:11], target_action[..., 8:11], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:7], target_action[..., 3:7], reduction="none"),
                F.mse_loss(action_pred[..., 11:15], target_action[..., 11:15], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 7], target_action[..., 7], reduction="none"),
                F.mse_loss(action_pred[..., 15], target_action[..., 15], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode in {"rel_eef_6d", "abs_eef"}:
            # 20D rot6d EEF: [xyz, rot6d, grip] x 2.
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 10:13], target_action[..., 10:13], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:9], target_action[..., 3:9], reduction="none"),
                F.mse_loss(action_pred[..., 13:19], target_action[..., 13:19], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 9], target_action[..., 9], reduction="none"),
                F.mse_loss(action_pred[..., 19], target_action[..., 19], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode in {"rel_eef_wxyz", "abs_eef_wxyz"}:
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 8:11], target_action[..., 8:11], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:7], target_action[..., 3:7], reduction="none"),
                F.mse_loss(action_pred[..., 11:15], target_action[..., 11:15], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 7], target_action[..., 7], reduction="none"),
                F.mse_loss(action_pred[..., 15], target_action[..., 15], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "abs_eef_rpy":
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 7:10], target_action[..., 7:10], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.stack([
                _angle_mse_loss_wrap(action_pred[..., 3:6], target_action[..., 3:6]),
                _angle_mse_loss_wrap(action_pred[..., 10:13], target_action[..., 10:13]),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 6], target_action[..., 6], reduction="none"),
                F.mse_loss(action_pred[..., 13], target_action[..., 13], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "qpos20_rel6d":
            qpos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:20], target_action[..., 0:20], reduction="none"),
                F.mse_loss(action_pred[..., 29:49], target_action[..., 29:49], reduction="none"),
            ], dim=2).mean(dim=2)
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 20:23], target_action[..., 20:23], reduction="none"),
                F.mse_loss(action_pred[..., 49:52], target_action[..., 49:52], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 23:29], target_action[..., 23:29], reduction="none"),
                F.mse_loss(action_pred[..., 52:58], target_action[..., 52:58], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = qpos_loss_per_token
            action_loss_per_token = (
                hand_qpos_weight * qpos_loss_per_token +
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token
            )
        else:
            action_loss_per_token = F.mse_loss(action_pred, target_action, reduction="none").mean(dim=2)

        denom = action_token_mask.sum(dim=1).clamp(min=1)
        action_loss_per_sample = (action_loss_per_token * action_token_mask).sum(dim=1) / denom
        raw_per_token = F.mse_loss(action_pred, target_action, reduction="none").mean(dim=2)
        raw_per_sample = (raw_per_token * action_token_mask).sum(dim=1) / denom
        if is_eef_mode:
            pos_raw = ((pos_loss_per_token * action_token_mask).sum(dim=1) / denom).mean()
            rot_raw = ((rot_loss_per_token * action_token_mask).sum(dim=1) / denom).mean()
            grip_raw = ((grip_loss_per_token * action_token_mask).sum(dim=1) / denom).mean()
        else:
            zero = torch.tensor(0.0, device=device)
            pos_raw = rot_raw = grip_raw = zero
        return {
            "loss_per_sample": action_loss_per_sample,
            "raw_per_sample": raw_per_sample,
            "pos_raw_loss": pos_raw,
            "rot_raw_loss": rot_raw,
            "grip_raw_loss": grip_raw,
            "eef_component_warmup_active": torch.tensor(float(eef_component_warmup_active), device=device),
            "eef_position_loss_weight": torch.tensor(float(eef_pos_weight), device=device),
            "eef_rotation_loss_weight": torch.tensor(float(eef_rot_weight), device=device),
            "eef_gripper_loss_weight": torch.tensor(float(eef_grip_weight), device=device),
            "hand_qpos_loss_weight": torch.tensor(float(hand_qpos_weight), device=device),
        }

    def _prepare_noisy_action_blocks(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        chunk_size: int,
        dtype: Optional[torch.dtype] = None,
        obs_latents: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Regroup noisy actions/states into per-chunk blocks.

        For use_state=True, each block uses the state of its first latent:
          [state_chunk_start | actions_for_all_latents_in_chunk]

        For rel_eef, each chunk's absolute EEF actions are converted relative to the
        chunk-start state, not per-latent states. abs_eef keeps chunk actions absolute
        in 20D rot6d form. abs_eef_wxyz keeps them absolute in 16D wxyz form.

        ``obs_latents`` is the number of leading observation latents whose actions are
        NOT predicted (defaults to ``chunk_size`` for the legacy offset-cs convention;
defaults to ``chunk_size`` for the legacy offset-cs convention).
        The remaining noisy latents are still grouped into ``chunk_size`` blocks.
        """
        # Preserve FP32 geometry even inside packed/outer AMP regions.
        with torch.autocast(device_type=actions.device.type, enabled=False):
            target_dtype = dtype or actions.dtype
            apl = self.config.actions_per_latent
            obs = chunk_size if obs_latents is None else max(1, int(obs_latents))
            noisy_actions_lat = actions[:, obs * apl:].float()
            noisy_states_lat = states[:, obs:].float() if states is not None else None

            if noisy_actions_lat.numel() == 0:
                return {
                    "flat_actions": noisy_actions_lat,
                    "states": None,
                    "action_chunks": [],
                    "block_latent_counts": [],
                    "block_action_token_lengths": [],
                    "action_token_length": 0,
                }

            B = noisy_actions_lat.shape[0]
            N = noisy_actions_lat.shape[1] // apl
            noisy_actions_lat = noisy_actions_lat.view(B, N, apl, -1)
            block_latent_counts = self._chunk_latent_counts(N, chunk_size)

            block_action_chunks = []
            block_states = []
            block_action_token_lengths = []
            need_block_state = self.config.action_mode in {"rel_joint", "rel_eef", "rel_eef_6d", "rel_eef_wxyz", "qpos20_rel6d"}
            offset = 0
            for count in block_latent_counts:
                abs_block_actions = noisy_actions_lat[:, offset:offset + count].reshape(B, count * apl, -1)
                state_block = noisy_states_lat[:, offset] if noisy_states_lat is not None else None
                if need_block_state and state_block is None:
                    raise ValueError(f"{self.config.action_mode} mode requires chunk-start state anchors for conversion")

                if self.config.action_mode == "rel_joint":
                    block_actions = _abs_joint_to_rel_joint(
                        state_block.float(),
                        abs_block_actions.float(),
                    ).to(target_dtype)
                elif self.config.action_mode == "rel_eef":
                    block_actions = abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch(
                        state_block.float(),
                        abs_block_actions.float(),
                        self.config.gripper_min,
                        self.config.gripper_max,
                    ).to(target_dtype)
                elif self.config.action_mode == "rel_eef_6d":
                    block_actions = abs_eef_20d_to_relative_eef_20d_torch(
                        state_block.float(),
                        abs_block_actions.float(),
                        self.config.gripper_min,
                        self.config.gripper_max,
                    ).to(target_dtype)
                elif self.config.action_mode == "rel_eef_wxyz":
                    block_actions = abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch(
                        state_block.float(),
                        abs_block_actions.float(),
                        self.config.gripper_min,
                        self.config.gripper_max,
                    ).to(target_dtype)
                elif self.config.action_mode == "qpos20_rel6d":
                    block_actions = abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(
                        state_block.float(),
                        abs_block_actions.float(),
                    )
                    block_actions = self._scale_qpos20_rel_wrist_xyz(block_actions).to(target_dtype)
                else:
                    block_actions = abs_block_actions
                block_actions_len = count * apl
                block_actions = block_actions[:, :block_actions_len].contiguous().view(B, block_actions_len, -1)
                block_action_chunks.append(block_actions.to(target_dtype))
                if self.config.use_state and state_block is not None:
                    block_states.append(state_block.to(target_dtype))
                block_action_token_lengths.append(block_actions_len)
                offset += count

            flat_actions = torch.cat(block_action_chunks, dim=1) if block_action_chunks else noisy_actions_lat.new_zeros(B, 0, actions.shape[-1])
            chunk_states = torch.stack(block_states, dim=1) if block_states else None
            return {
                "flat_actions": flat_actions,
                "states": chunk_states,
                "action_chunks": block_action_chunks,
                "block_latent_counts": block_latent_counts,
                "block_action_token_lengths": block_action_token_lengths,
                "action_token_length": int(sum(block_action_token_lengths)),
            }

    def _encode_action_chunk_tokens(
        self,
        action_chunks: List[torch.Tensor],
        states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode per-chunk action tokens using one state token per chunk."""
        if self.config.use_state and states is not None:
            state_mask = None
            if self.training and self.config.state_mask_ratio > 0:
                keep_prob = 1.0 - float(self.config.state_mask_ratio)
                keep_prob = min(max(keep_prob, 0.0), 1.0)
                state_mask = torch.bernoulli(
                    states.new_full(states.shape[:2], keep_prob, dtype=self.dtype)
                )
            return self.action_expert.input_encoder.forward_chunk_list(
                states,
                action_chunks,
                state_mask=state_mask,
            )
        if hasattr(self.action_expert.input_encoder, "forward_chunk_list"):
            return self.action_expert.input_encoder.forward_chunk_list(action_chunks)
        return self.action_expert.input_encoder(None, torch.cat(action_chunks, dim=1))

    def _split_flat_action_chunks(
        self,
        flat_actions: torch.Tensor,
        block_action_token_lengths: List[int],
    ) -> List[torch.Tensor]:
        """Split flattened action suffix back into per-chunk chunks."""
        chunks = []
        offset = 0
        for block_len in block_action_token_lengths:
            chunks.append(flat_actions[:, offset:offset + block_len])
            offset += block_len
        return chunks

    def _build_chunked_action_timesteps(
        self,
        t_values: torch.Tensor,
        block_latent_counts: List[int],
        include_value: bool = False,
    ) -> torch.Tensor:
        """Build per-token timesteps for chunk-grouped action registers."""
        parts = []
        apf = self.config.action_tokens_per_frame
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        for count in block_latent_counts:
            if nvpf > 0:
                parts.append(torch.zeros(t_values.shape[0], nvpf, dtype=t_values.dtype, device=t_values.device))
            if spf > 0:
                parts.append(t_values.view(t_values.shape[0], 1).expand(-1, spf))
            parts.append(t_values.view(t_values.shape[0], 1).expand(-1, count * apf))
        return torch.cat(parts, dim=1) if parts else t_values.new_zeros(t_values.shape[0], 0)

    def _strip_chunk_state_predictions(
        self,
        action_pred: torch.Tensor,
        block_action_token_lengths: List[int],
    ) -> torch.Tensor:
        """Remove one state prediction token per chunk, keeping only action predictions."""
        if not self.config.use_state:
            return action_pred
        parts = []
        offset = 0
        for block_len in block_action_token_lengths:
            total_len = self.config.state_tokens_per_frame + block_len
            block = action_pred[:, offset:offset + total_len]
            parts.append(block[:, self.config.state_tokens_per_frame:])
            offset += total_len
        return torch.cat(parts, dim=1) if parts else action_pred[:, :0]


    def load_checkpoint_weights(self, path: str, strict: bool = True) -> None:
        """Load model weights from checkpoint (for finetune from pretrained).

        Unlike Motus1's load_pretrain_weights, this does a normal full load—no
        filtering of action encoder/decoder. Use for: pretrain done, finetune on
        new dataset (load model only, optimizer/scheduler start fresh).

        Args:
            path: Checkpoint directory (e.g. checkpoint_step_20000) or direct .pt file.
                  Directory expects: pytorch_model/mp_rank_00_model_states.pt
            strict: If False, allow missing/unexpected keys
        """
        p = Path(path)
        if p.is_dir():
            model_file = p / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            path = str(model_file)
        elif not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("module", ckpt.get("state_dict", ckpt))
        # Drop removed buffers from older checkpoints (e.g. sinusoidal pos_embed).
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.endswith(".pos_embed") or "action_expert" not in k
        }
        if not strict:
            model_state = self.state_dict()
            mismatched = {
                k: (tuple(v.shape), tuple(model_state[k].shape))
                for k, v in state_dict.items()
                if k in model_state and hasattr(v, "shape") and tuple(v.shape) != tuple(model_state[k].shape)
            }
            if mismatched:
                drop_prefixes = set()
                action_sensitive_prefixes = (
                    "action_expert.input_encoder.",
                    "action_expert.decoder.",
                )
                if any(k.startswith(action_sensitive_prefixes) for k in mismatched):
                    drop_prefixes.update(action_sensitive_prefixes)
                    logger.warning(
                        "Action/state-token checkpoint mismatch detected; reinitializing "
                        "action encoder/decoder and loading only compatible weights."
                    )

                skipped_mismatched = {}
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if any(k.startswith(prefix) for prefix in drop_prefixes):
                        skipped_mismatched[k] = (
                            tuple(v.shape) if hasattr(v, "shape") else "<non-tensor>",
                            tuple(model_state[k].shape) if k in model_state and hasattr(model_state[k], "shape") else "<missing>",
                        )
                        continue
                    if k in mismatched:
                        skipped_mismatched[k] = mismatched[k]
                        continue
                    filtered_state_dict[k] = v

                if skipped_mismatched:
                    preview = list(sorted(skipped_mismatched.items()))[:20]
                    logger.warning(
                        "Skipping %s mismatched checkpoint tensors during non-strict load. First entries: %s",
                        len(skipped_mismatched),
                        preview,
                    )
                state_dict = filtered_state_dict

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.info(f"Loaded checkpoint weights from {path}")

    def _ensure_freqs_device(self):
        dev = self.device
        if self.freqs.device != dev:
            self.freqs = self.freqs.to(dev)
        if self.freqs_action.device != dev:
            self.freqs_action = self.freqs_action.to(dev)
        if self.freqs_state.device != dev:
            self.freqs_state = self.freqs_state.to(dev)


    def _training_rope_starts(self, latent_start, batch_size: int):
        """Add the per-sample random global-RoPE start used only while training."""
        limit = int(self.config.memory_train_rope_offset_max)
        if not (self.training and limit > 0):
            return latent_start
        shift = torch.randint(0, limit + 1, (int(batch_size),))
        if isinstance(latent_start, list):
            return [int(s) + int(d) for s, d in zip(latent_start, shift.tolist())]
        if latent_start is None:
            return shift.to(self.device)
        return latent_start.to(dtype=torch.long) + shift.to(latent_start.device)


    def _inference_video_start_frame(self, cache_seq_len: int) -> int:
        return self.current_start_frame


    def _create_video_freqs(self, f, start_frame=0):
        """Create video freqs for 3D RoPE.

        Args:
            f: Number of temporal frames
            start_frame: Starting frame index (for inference)

        Returns:
            freqs: [f * lat_H * lat_W, 1, head_dim//2] complex tensor
        """
        self._ensure_freqs_device()
        d = self.wan_dim // self.video_model.wan_model.num_heads
        c = d // 2
        # Split freqs into [freqs_t, freqs_h, freqs_w] like rope_apply_one
        freqs_split = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # Create 3D freqs: [f, lat_H, lat_W, head_dim//2]
        video_freqs = torch.cat([
            freqs_split[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, self.lat_H, self.lat_W, -1),
            freqs_split[1][:self.lat_H].view(1, self.lat_H, 1, -1).expand(f, self.lat_H, self.lat_W, -1),
            freqs_split[2][:self.lat_W].view(1, 1, self.lat_W, -1).expand(f, self.lat_H, self.lat_W, -1),
        ], dim=-1).reshape(f * self.lat_H * self.lat_W, 1, -1)

        return video_freqs

    def _create_video_freqs_batch(self, f: int, start_frames: torch.Tensor) -> torch.Tensor:
        """Create per-sample video freqs for 3D RoPE.

        Args:
            f: Number of temporal frames.
            start_frames: [B] absolute latent start index for each sample.

        Returns:
            freqs: [B, f * lat_H * lat_W, 1, head_dim//2]
        """
        self._ensure_freqs_device()
        start_frames = start_frames.to(device=self.device, dtype=torch.long)
        out = [self._create_video_freqs(f, int(s.item())) for s in start_frames]
        return torch.stack(out, dim=0)

    def _create_gist_freqs_from_clean(
        self, clean_video_freqs: torch.Tensor, num_clean_frames: int,
    ) -> torch.Tensor:
        """Frame-level 3D RoPE for gist tokens (same temporal position as clean frame)."""
        M = self._memory_gist_tokens()
        vpf = self.frame_seqlen
        n_frames = int(num_clean_frames)
        if M <= 0 or n_frames <= 0:
            return clean_video_freqs.new_zeros(0, *clean_video_freqs.shape[1:])
        parts = []
        for f in range(n_frames):
            frame_freq = clean_video_freqs[f * vpf : f * vpf + 1]
            parts.append(frame_freq.expand(M, -1, -1))
        return torch.cat(parts, dim=0)

    def _build_noisy_freqs(self, noisy_video_freqs, num_noisy_frames,
                           block_latent_counts: Optional[List[int]] = None,
                           include_value: bool = True):
        """Build RoPE freqs for noisy part: [noisy_vid | per-chunk (value + state + action)].

        Per-chunk token order: [value(count*K) | state(1) | action(count*24)].
        Current policy semantics:
          - value tokens do NOT use RoPE
          - state tokens do NOT use RoPE
          - action tokens use chunk-local RoPE and reset every chunk

        Args:
            noisy_video_freqs: 3D RoPE for noisy video tokens (positions 1..W-1).
            num_noisy_frames: N = W-1, number of noisy frames.
            include_value: If False, skip value token frequencies (for policy/wm eval
                that doesn't include value tokens in the sequence).
        """
        self._ensure_freqs_device()
        napf = self.config.action_tokens_per_frame
        nspf = self.config.state_tokens_per_frame
        nvpf = self.config.value_tokens_per_frame if include_value else 0

        if noisy_video_freqs.dim() == 3:
            parts = [noisy_video_freqs]
            counts = block_latent_counts or [1] * num_noisy_frames
            for count in counts:
                if nvpf > 0:
                    parts.append(torch.ones((nvpf, 1, self.freqs_state.shape[-1]), device=self.device, dtype=self.freqs_state.dtype))
                if nspf > 0:
                    parts.append(torch.ones((nspf, 1, self.freqs_state.shape[-1]), device=self.device, dtype=self.freqs_state.dtype))
                action_len = count * napf
                parts.append(self.freqs_action[:action_len].view(action_len, 1, -1))
            return torch.cat(parts, dim=0)

        if noisy_video_freqs.dim() == 4:
            B = noisy_video_freqs.shape[0]
            out = []
            for b in range(B):
                out.append(self._build_noisy_freqs(
                    noisy_video_freqs[b],
                    num_noisy_frames,
                    block_latent_counts=block_latent_counts,
                    include_value=include_value,
                ))
            return torch.stack(out, dim=0)

        raise ValueError(f"Unsupported noisy_video_freqs shape: {tuple(noisy_video_freqs.shape)}")

    # ------------------------------------------------------------------
    # Action-first hybrid-memory helpers (clean_video | gist | per-chunk(state?, action, noisy_video))
    # ------------------------------------------------------------------
    def _interleave_memory_chunk_ar_tokens(
        self, action_tokens, noisy_video_tokens, block_latent_counts, value_tokens=None,
    ):
        """Pack the noisy suffix action-first per chunk: [state?, action, noisy_video, value?]."""
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        vpf = self.frame_seqlen
        nvpf = self.config.value_tokens_per_frame if value_tokens is not None else 0
        parts = []
        a_off = 0
        v_off = 0
        val_off = 0
        for count in block_latent_counts:
            a_len = spf + count * apf
            v_len = count * vpf
            if a_len > 0:
                parts.append(action_tokens[:, a_off:a_off + a_len])
            if v_len > 0:
                parts.append(noisy_video_tokens[:, v_off:v_off + v_len])
            if nvpf > 0:
                parts.append(value_tokens[:, val_off:val_off + nvpf])
                val_off += nvpf
            a_off += a_len
            v_off += v_len
        return torch.cat(parts, dim=1) if parts else noisy_video_tokens[:, :0]

    def _split_memory_chunk_ar_hidden(self, full_noisy_hidden, block_latent_counts, include_value=False):
        """Unpack action-first [state?, action, noisy_video, value?] -> (action, video[, value])."""
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        vpf = self.frame_seqlen
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        action_parts = []
        video_parts = []
        value_parts = []
        offset = 0
        for count in block_latent_counts:
            a_len = spf + count * apf
            v_len = count * vpf
            if a_len > 0:
                action_parts.append(full_noisy_hidden[:, offset:offset + a_len])
                offset += a_len
            if v_len > 0:
                video_parts.append(full_noisy_hidden[:, offset:offset + v_len])
                offset += v_len
            if nvpf > 0:
                value_parts.append(full_noisy_hidden[:, offset:offset + nvpf])
                offset += nvpf
        action_hidden = torch.cat(action_parts, dim=1) if action_parts else full_noisy_hidden[:, :0]
        video_hidden = torch.cat(video_parts, dim=1) if video_parts else full_noisy_hidden[:, :0]
        if include_value:
            value_hidden = torch.cat(value_parts, dim=1) if value_parts else full_noisy_hidden[:, :0]
            return action_hidden, video_hidden, value_hidden
        return action_hidden, video_hidden

    def _build_memory_chunk_ar_timesteps(self, action_t, video_t, block_latent_counts, include_value=False):
        """Per-token timesteps for action-first [state?, action, noisy_video, value?]."""
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        vpf = self.frame_seqlen
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        B = action_t.shape[0]
        parts = []
        for count in block_latent_counts:
            a_len = spf + count * apf
            if a_len > 0:
                parts.append(action_t.view(B, 1).expand(-1, a_len))
            v_len = count * vpf
            if v_len > 0:
                parts.append(video_t.view(B, 1).expand(-1, v_len))
            if nvpf > 0:
                parts.append(torch.zeros(B, nvpf, dtype=action_t.dtype, device=action_t.device))
        return torch.cat(parts, dim=1) if parts else action_t.new_zeros(B, 0)

    def _action_freqs_video_basis(
        self,
        chunk_start_latent: float,
        action_len: int,
        *,
        actions_per_frame: int,
    ) -> torch.Tensor:
        """Place action-register tokens on the absolute video latent clock."""
        self._ensure_freqs_device()
        head_dim = self.wan_dim // self.video_model.wan_model.num_heads
        offsets = torch.arange(
            int(action_len), device=self.device, dtype=torch.float64
        )
        positions = float(chunk_start_latent) + offsets / float(actions_per_frame)
        return rope_video_basis_freqs(
            positions,
            head_dim,
            device=self.device,
            dtype=self.freqs.dtype,
        )

    def _maybe_action_video_basis_override(
        self,
        num_action_tokens: int,
        chunk_start_latent: int,
    ) -> Optional[torch.Tensor]:
        if self.config.action_rope_mode != "video_shared" or num_action_tokens <= 0:
            return None
        return self._action_freqs_video_basis(
            chunk_start_latent,
            num_action_tokens,
            actions_per_frame=self.config.action_tokens_per_frame,
        )

    def _build_memory_full_freqs(
        self,
        clean_video_freqs,
        noisy_video_freqs,
        block_latent_counts,
        num_anchor_frames: int = 0,
        noisy_start=0,
    ):
        """Full-sequence RoPE freqs for [anchor? | clean_video | gist | per-chunk(state?, action, noisy_video)].

        Gist + state tokens get identity freqs; action tokens get chunk-local 1D action freqs; clean/noisy
        video get 3D video freqs. Handles shared ([L,1,C]) and per-sample
        ([B,L,1,C]) video freqs (global RoPE).
        """
        self._ensure_freqs_device()
        if clean_video_freqs.dim() == 4:
            starts = noisy_start
            if torch.is_tensor(starts) and starts.ndim == 0:
                starts = [int(starts.detach().cpu().item())] * clean_video_freqs.shape[0]
            elif not torch.is_tensor(starts) and not isinstance(starts, (list, tuple)):
                starts = [int(starts)] * clean_video_freqs.shape[0]
            return torch.stack([
                self._build_memory_full_freqs(
                    clean_video_freqs[b], noisy_video_freqs[b], block_latent_counts,
                    num_anchor_frames=num_anchor_frames,
                    noisy_start=(
                        int(starts[b].detach().cpu().item())
                        if torch.is_tensor(starts)
                        else int(starts[b])
                    ),
                )
                for b in range(clean_video_freqs.shape[0])
            ], dim=0)

        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        vpf = self.frame_seqlen
        nvpf = self.config.value_tokens_per_frame
        gist_per_frame = self._memory_gist_tokens()
        num_clean_frames = sum(block_latent_counts)
        C = self.freqs_state.shape[-1]
        dt = self.freqs_state.dtype

        def ident(length):
            return torch.ones((length, 1, C), device=self.device, dtype=dt)

        parts = []
        # Global-L0 anchor: identity RoPE (position-independent landmark).
        if num_anchor_frames > 0:
            parts.append(ident(int(num_anchor_frames) * vpf))
        parts.append(clean_video_freqs)
        if gist_per_frame > 0 and num_clean_frames > 0:
            parts.append(self._create_gist_freqs_from_clean(clean_video_freqs, num_clean_frames))
        video_shared_actions = self.config.action_rope_mode == "video_shared"
        v_off = 0
        latent_offset = 0
        for count in block_latent_counts:
            if spf > 0:
                parts.append(ident(spf))
            a_len = count * apf
            if a_len > 0:
                if video_shared_actions:
                    parts.append(
                        self._action_freqs_video_basis(
                            float(noisy_start) + latent_offset,
                            a_len,
                            actions_per_frame=apf,
                        )
                    )
                else:
                    parts.append(self.freqs_action[:a_len].view(a_len, 1, -1))
            v_len = count * vpf
            parts.append(noisy_video_freqs[v_off:v_off + v_len])
            v_off += v_len
            latent_offset += int(count)
            if nvpf > 0:
                parts.append(ident(nvpf))
        return torch.cat(parts, dim=0)


    def _forward_layer_memory(self, x, e0, layer_idx, t5_ctx, ar_mask, full_freqs):
        """Single WAN layer for the action-first hybrid-memory training forward.

        full_freqs already encodes per-token video/action/identity RoPE, so a
        single rope_action_apply + flex-attention with the memory BlockMask
        reproduces the streaming KV visibility exactly.
        """
        blk = self.video_model.wan_model.blocks[layer_idx]
        mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
        B, L, H_dim = norm_x.shape
        n = self.video_model.wan_model.num_heads
        d = H_dim // n
        q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, L, n, d)
        k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, L, n, d)
        v = blk.self_attn.v(norm_x).view(B, L, n, d)
        q = rope_action_apply(q, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
        k = rope_action_apply(k, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
        y = BLHD_flex_with_mask(q, k, v, ar_mask)
        y = y.flatten(2)
        y = blk.self_attn.o(y)
        x = x + y * mod[2].squeeze(2)
        if t5_ctx is not None:
            x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
        ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
        x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x

    def _training_retention(self, window_lengths, obs, chunk_size):
        """Per-sample clean frames kept visible beyond the recent window.

        ``_retention_override`` (one entry per sample: frame list or None) pins the
        sets for frozen-checkpoint probes; otherwise sets are drawn only while
        training with retention enabled.
        """
        override = getattr(self, "_retention_override", None)
        if override is not None:
            if len(override) != len(window_lengths):
                raise ValueError(
                    f"_retention_override has {len(override)} entries for {len(window_lengths)} samples"
                )
            return [None if r is None else [int(f) for f in r] for r in override]
        max_frames = int(self.config.memory_retention_max_frames)
        if max_frames <= 0 or not self.training:
            return None
        full_prob = float(self.config.memory_retention_full_prob)
        empty_prob = float(self.config.memory_retention_empty_prob)
        cs = max(1, int(chunk_size))
        out = []
        for W in window_lengths:
            N = max(0, ((int(W) - obs) // cs) * cs)
            u = float(torch.rand(()))
            if u < full_prob:
                out.append(list(range(N)))
            elif u < full_prob + empty_prob or N == 0:
                out.append(None)
            else:
                k = min(int(torch.randint(1, max_frames + 1, ())), N)
                out.append(sorted(torch.randperm(N)[:k].tolist()))
        return out

    def _create_training_mask(self, L_F, chunk_size):
        """Action-first hybrid-memory training mask (single source of truth)."""
        attention_mode = self.config.mode
        if attention_mode == "loop":
            attention_mode = "video_idm" if torch.rand((), device="cpu").item() < 0.5 else "action_only"
        mask, padded = create_memory_window_mask(
            num_latent_frames=L_F,
            video_tokens_per_frame=self.frame_seqlen,
            gist_tokens_per_frame=self._memory_gist_tokens(),
            action_tokens_per_frame=self.config.action_tokens_per_frame,
            chunk_size=chunk_size,
            anchor_frames=self._memory_anchor_frames(),
            recent_frames=self._memory_recent_frames(),
            use_state=self.config.use_state,
            state_tokens_per_frame=self.config.state_tokens_per_frame,
            value_tokens_per_frame=self.config.value_tokens_per_frame,
            attention_mode=attention_mode,
            device=self.device,
            separate_anchor=False,
            obs_latents=self._obs_latents(),
        )
        return mask, padded


    def _video_patch_embed(self, latent: torch.Tensor) -> torch.Tensor:
        """Patch-embed video latent -> tokens [B, L, dim]."""
        x = self.video_model.wan_model.patch_embedding(latent)
        return x.flatten(2).transpose(1, 2)

    def _video_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute the WAN time embedding used by both AdaLN and the output head."""
        B, L = timesteps.shape
        t_flat = timesteps.flatten()
        freq_dim = self.video_model.wan_model.freq_dim
        # Use autocast to ensure float32 computation
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat).unflatten(0, (B, L)).float()
            )
        return e

    def _video_time_embed(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute WAN time embedding + 6-param AdaLN projection."""
        e = self._video_time_embedding(timesteps)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0 = self.video_model.wan_model.time_projection(e).unflatten(2, (6, self.wan_dim))
        return e, e0

    def _action_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute the action time embedding without an unused AdaLN projection."""
        B, L = timesteps.shape
        freq_dim = self.action_expert.freq_dim
        # Use autocast to ensure float32 computation
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(freq_dim, timesteps.flatten()).unflatten(0, (B, L)).float()
            )
        return e

    def _action_time_embed(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action time embedding + 6-param projection."""
        e = self._action_time_embedding(timesteps)
        dim = self.action_expert.config.dim
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0 = self.action_expert.time_projection(e).unflatten(2, (6, dim))
        return e, e0

    def _t5_embed(self, language_embeddings) -> torch.Tensor:
        """Process T5 embeddings -> WAN text context."""
        text_len = self.video_model.wan_model.text_len
        if isinstance(language_embeddings, list):
            padded = []
            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded.append(torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])]))
                else:
                    padded.append(emb[:text_len])
            ctx = torch.stack(padded)
        else:
            ctx = language_embeddings
        return self.video_model.wan_model.text_embedding(ctx)

    # ------------------------------------------------------------------
    # Forward: single WAN layer (training, use_mot=False)
    # ------------------------------------------------------------------

    def _forward_layer_simple(self, x, e0, layer_idx, t5_ctx, ar_mask=None,
                              kv_cache=None, v_seq_len=None,
                              is_training=True, action_register_length=None,
                              half_seq_len=None, video_freqs=None, noisy_freqs=None):
        """Forward one WAN layer (use_mot=False). Unified for training and inference.

        Training layout:  [clean_vid | noisy_vid | noisy_act | pad]
        Inference layout: [Z_k + Y_k]  (query attends to KV cache)

        Args:
            video_freqs: Pre-computed 3D RoPE for video tokens (training: W frames, inference: K frames).
            noisy_freqs: Pre-computed RoPE for noisy_vid + noisy_act (training only).
        """
        blk = self.video_model.wan_model.blocks[layer_idx]
        mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)

        norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
        B, L, H_dim = norm_x.shape
        n = self.video_model.wan_model.num_heads
        d = H_dim // n

        q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, L, n, d)
        k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, L, n, d)
        v = blk.self_attn.v(norm_x).view(B, L, n, d)

        if is_training:
            if half_seq_len is None:
                half_seq_len = (L - (action_register_length or 0)) // 2
            action_seq_len = (action_register_length or 0)

            # RoPE: clean part gets video_freqs, noisy part gets noisy_freqs
            q_clean = rope_action_apply(q[:, :half_seq_len], video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            k_clean = rope_action_apply(k[:, :half_seq_len], video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            q_noisy = rope_action_apply(q[:, half_seq_len:half_seq_len + half_seq_len + action_seq_len],
                                        noisy_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            k_noisy = rope_action_apply(k[:, half_seq_len:half_seq_len + half_seq_len + action_seq_len],
                                        noisy_freqs, self.freqs_action, self.freqs_state, None).type_as(v)

            pad_len = L - half_seq_len - half_seq_len - action_seq_len
            if pad_len > 0:
                q_full = torch.cat([q_clean, q_noisy, q[:, -pad_len:]], dim=1)
                k_full = torch.cat([k_clean, k_noisy, k[:, -pad_len:]], dim=1)
            else:
                q_full = torch.cat([q_clean, q_noisy], dim=1)
                k_full = torch.cat([k_clean, k_noisy], dim=1)
            v_full = v
        else:
            assert v_seq_len is not None
            a_seq_len = L - v_seq_len

            infer_latent_frames = v_seq_len // self.frame_seqlen
            cache_seq_len = kv_cache[0].shape[1] if kv_cache is not None else 0
            inf_video_freqs = self._create_video_freqs(
                infer_latent_frames,
                start_frame=self._inference_video_start_frame(cache_seq_len),
            )

            if action_register_length is not None and a_seq_len > 0:
                napb = infer_latent_frames * self.config.action_tokens_per_frame
                nspb = self.config.state_tokens_per_frame
                nvpb = infer_latent_frames * self.config.value_tokens_per_frame
                roped_q = causal_rope_action_apply(q, inf_video_freqs, self.freqs_action, self.freqs_state,
                                                   action_register_length, napb, nspb,
                                                   num_value_per_block=nvpb).type_as(v)
                roped_k = causal_rope_action_apply(k, inf_video_freqs, self.freqs_action, self.freqs_state,
                                                   action_register_length, napb, nspb,
                                                   num_value_per_block=nvpb).type_as(v)
                q_full = torch.cat([roped_q[:, :v_seq_len], roped_q[:, v_seq_len:]], dim=1)
                k_full = torch.cat([roped_k[:, :v_seq_len], roped_k[:, v_seq_len:]], dim=1)
            else:
                q_full = rope_action_apply(q, inf_video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
                k_full = rope_action_apply(k, inf_video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            v_full = v

        # Attention
        if is_training:
            y = BLHD_flex_with_mask(q_full, k_full, v_full, ar_mask)
            y = y.flatten(2)
        else:
            if kv_cache is not None:
                cached_k, cached_v = kv_cache
                attn_k = torch.cat([cached_k, k_full], dim=1)
                attn_v = torch.cat([cached_v, v_full], dim=1)
            else:
                attn_k, attn_v = k_full, v_full
            from wan.modules.attention import flash_attention
            y = flash_attention(q_full, attn_k, attn_v, k_lens=None)
            y = y.flatten(2)

        y = blk.self_attn.o(y)
        x = x + y * mod[2].squeeze(2)
        if t5_ctx is not None:
            x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
        ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
        x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x


    def _forward_layer_memory_packed(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        layer_idx: int,
        t5_ctx,
        ar_mask,
        full_freqs,
        varlen_meta: VarlenBatchMeta,
    ):
        """Memory layer for a packed single-row batch with multiple trajectory segments."""
        blk = self.video_model.wan_model.blocks[layer_idx]
        mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
        B, L, H_dim = norm_x.shape
        n = self.video_model.wan_model.num_heads
        d = H_dim // n
        q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, L, n, d)
        k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, L, n, d)
        v = blk.self_attn.v(norm_x).view(B, L, n, d)
        q = rope_action_apply(q, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
        k = rope_action_apply(k, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)

        # Backend selection: MOTUS2_VARLEN_ATTN=flex keeps the legacy Flex path;
        # otherwise use per-segment factored attention (flash/sdpa).
        env_attn = os.environ.get("MOTUS2_VARLEN_ATTN", "").strip().lower()
        cfg_backend = str(getattr(self.config, "varlen_attention_backend", "auto") or "auto").lower()
        use_flex = (
            env_attn in {"flex", "flex_attention"}
            or cfg_backend in {"flex", "flex_attention"}
            or not varlen_meta.factored_plans
            or not varlen_meta.factored_layouts
        )
        if use_flex:
            y = varlen_block_attention(
                q, k, v,
                flex_block_mask=ar_mask,
                logical_block_mask=varlen_meta.logical_block_mask,
                q_block_indices=varlen_meta.q_block_indices,
                kv_block_indices=varlen_meta.kv_block_indices,
            ).flatten(2)
        else:
            backend = cfg_backend
            if len(varlen_meta.factored_plans) != len(varlen_meta.segments):
                raise ValueError(
                    f"factored_plans({len(varlen_meta.factored_plans)}) != "
                    f"segments({len(varlen_meta.segments)})"
                )
            if len(varlen_meta.factored_layouts) != len(varlen_meta.segments):
                raise ValueError(
                    f"factored_layouts({len(varlen_meta.factored_layouts)}) != "
                    f"segments({len(varlen_meta.segments)})"
                )
            out = torch.zeros_like(q)
            for i, seg in enumerate(varlen_meta.segments):
                offset = int(seg.offset)
                seg_len = int(seg.token_length)
                layout_i = varlen_meta.factored_layouts[i]
                plans_i = varlen_meta.factored_plans[i]
                if int(layout_i.total_len) != seg_len:
                    raise ValueError(
                        f"segment {i}: layout.total_len({layout_i.total_len}) != "
                        f"token_length({seg_len})"
                    )
                q_i = q[:, offset:offset + seg_len]
                k_i = k[:, offset:offset + seg_len]
                v_i = v[:, offset:offset + seg_len]
                y_i = factored_memory_attention(
                    q_i, k_i, v_i,
                    layout_meta=layout_i,
                    plans=plans_i,
                    backend=backend,
                )
                out[:, offset:offset + seg_len] = y_i
            # Pad region (if any) stays zeros — pad tokens are not in the loss.
            y = out.flatten(2)

        y = blk.self_attn.o(y)
        x = x + y * mod[2].squeeze(2)

        if t5_ctx is not None:
            # Packed self-attention is one row, but language conditioning remains
            # per trajectory. Split only for cross-attn, preserve vanilla's full
            # padded-context contract, then restore the packed layout.
            pieces = []
            offset = 0
            lengths = varlen_meta.segment_token_lengths
            for seg_idx, seg_len in enumerate(lengths):
                seg_x = x[:, offset:offset + seg_len]
                seg_ctx = t5_ctx[seg_idx:seg_idx + 1]
                seg_x = seg_x + blk.cross_attn(
                    blk.norm3(seg_x),
                    seg_ctx,
                    context_lens=None,
                )
                pieces.append(seg_x)
                offset += seg_len
            if offset < x.shape[1]:
                pieces.append(x[:, offset:])
            x = torch.cat(pieces, dim=1)

        ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
        x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x

    def training_step_packed_varlen(
        self,
        latents,
        actions,
        states=None,
        language_embeddings=None,
        valid_len=None,
        action_valid_mask=None,
        video_valid_mask=None,
        latent_start=None,
        values=None,
        is_task_irrelevant=None,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Full-trajectory varlen training with packed attention segments.

        This path is intentionally scoped to the action-first memory model used by
        this worktree. It does not pad trajectories before attention; samples are
        concatenated into one packed attention row and separated by segment ids in
        the BlockMask.

        Per-layer activation memory scales ~linearly with the packed token count
        (~2.3GB/T_lat on an 8xB200 ZeRO2 setup, measured empirically), and since
        full-trajectory episodes have no length cap, `gradient_checkpointing`
        below is what keeps that linear-in-depth-times-length cost from being
        30x (num_layers) larger. Without it, a single ~65+ latent-frame outlier
        episode OOMs even though most episodes (10-40 latents) fit comfortably.
        """

        B = len(latents)
        device = self.device
        apl = self.config.actions_per_latent
        chunk_size = self._fixed_chunk_size()
        obs = self._obs_latents()
        nvpf = self.config.value_tokens_per_frame
        value_enabled = self.config.value_enabled and nvpf > 0

        irrelevant_mask = (
            is_task_irrelevant.to(device=device, dtype=torch.bool)
            if is_task_irrelevant is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        irrelevant_start_step = max(
            int(self.config.warmup_steps),
            int(self.config.uniform_loss_weight_warmup_steps),
        )
        active_irrelevant = irrelevant_mask & (global_step >= irrelevant_start_step)
        inactive_irrelevant = irrelevant_mask & ~active_irrelevant
        main_mask = ~irrelevant_mask

        use_wm = torch.zeros(B, dtype=torch.bool, device=device)
        use_vm = torch.zeros(B, dtype=torch.bool, device=device)
        if main_mask.any():
            main_idx = main_mask.nonzero(as_tuple=True)[0]
            rand_mode_main = torch.rand(len(main_idx), device=device)
            if value_enabled and self.config.vm_ratio > 0:
                use_vm_main = rand_mode_main < self.config.vm_ratio
                use_wm_main = (~use_vm_main) & (rand_mode_main < self.config.vm_ratio + self.config.wm_ratio)
                use_vm[main_idx] = use_vm_main
                use_wm[main_idx] = use_wm_main
            else:
                use_wm[main_idx] = rand_mode_main < self.config.wm_ratio
        if active_irrelevant.any():
            irr_idx = active_irrelevant.nonzero(as_tuple=True)[0]
            if value_enabled and self.config.vm_ratio > 0:
                wm_mass = max(float(self.config.wm_ratio), 0.0)
                vm_mass = max(float(self.config.vm_ratio), 0.0)
                total_mass = wm_mass + vm_mass
                if total_mass > 0:
                    irr_rand = torch.rand(len(irr_idx), device=device)
                    use_vm[irr_idx] = irr_rand < (vm_mass / total_mass)
                    use_wm[irr_idx] = ~use_vm[irr_idx]
                else:
                    use_wm[irr_idx] = True
            else:
                use_wm[active_irrelevant.nonzero(as_tuple=True)[0]] = True
        wm_idx = use_wm.nonzero(as_tuple=True)[0]
        vm_idx = use_vm.nonzero(as_tuple=True)[0]

        video_shift = float(self.config.video_timestep_shift)
        action_shift = float(self.config.action_timestep_shift)

        def _shifted_sigma(raw, shift):
            return shift * raw / (1 + (shift - 1) * raw)

        sigma_v = torch.zeros(B, device=device, dtype=self.dtype)
        sigma_a = torch.zeros(B, device=device, dtype=self.dtype)
        pol_idx = (~inactive_irrelevant & ~use_wm & ~use_vm).nonzero(as_tuple=True)[0]
        if len(pol_idx) > 0:
            if self.config.policy_timestep_method == "beta":
                eta = Beta(self.config.beta_alpha, self.config.beta_beta).sample((len(pol_idx),))
                raw_v = eta.to(device)
            else:
                from utils.common import sample_t
                t_pol = sample_t(self._policy_t_dist, len(pol_idx)).to(device=device, dtype=self.dtype)
                raw_v = (1.0 - t_pol).clamp(0, 1)
            sigma_v[pol_idx] = _shifted_sigma(raw_v, video_shift).to(self.dtype)
            if self.config.action_timestep_method == "beta":
                eta_a = Beta(self.config.beta_alpha, self.config.beta_beta).sample((len(pol_idx),))
                raw_a = eta_a.to(device)
            else:
                from utils.common import sample_t
                t_act = sample_t(self._action_t_dist, len(pol_idx)).to(device=device, dtype=self.dtype)
                raw_a = (1.0 - t_act).clamp(0, 1)
            sigma_a[pol_idx] = _shifted_sigma(raw_a, action_shift).to(self.dtype)
        if len(wm_idx) > 0:
            u = torch.randn(len(wm_idx), device=device)
            raw_v_wm = (1.0 - torch.sigmoid(u)).clamp(0, 1)
            sigma_v[wm_idx] = _shifted_sigma(raw_v_wm, video_shift).to(self.dtype)

        noisy_vid_t = (sigma_v * self.scheduler.num_train_timesteps).to(self.dtype)
        noisy_act_t = (sigma_a * self.action_scheduler.num_train_timesteps).to(self.dtype)
        window_lengths_input = [int(sample.shape[1]) for sample in latents]

        def _int_meta_list(value, defaults):
            if value is None:
                return [int(v) for v in defaults]
            if isinstance(value, torch.Tensor):
                return [int(v) for v in value.detach().cpu().view(-1).tolist()]
            out = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    out.append(int(item.detach().cpu().item()))
                else:
                    out.append(int(item))
            return out

        valid_lens_l = _int_meta_list(valid_len, window_lengths_input)
        latent_starts_l = self._training_rope_starts(_int_meta_list(latent_start, [0] * B), B)

        def _caption_lengths(value):
            if value is None:
                return [0] * B, [0] * B
            text_len = self.video_model.wan_model.text_len
            if isinstance(value, list):
                caps = [min(int(v.shape[0]), text_len) for v in value]
                return caps, [text_len] * len(caps)
            # Collate pads language embeddings with zero rows; recover true lengths.
            nonzero = value.detach().abs().sum(dim=-1) > 0
            caps = [min(int(v), text_len) for v in nonzero.sum(dim=1).cpu().tolist()]
            return caps, [int(value.shape[1])] * int(value.shape[0])

        caption_lengths_l, caption_padded_lengths_l = _caption_lengths(language_embeddings)
        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

        # Per-segment named-stream tensors (anchor | clean_video | gist | chunk_ar).
        # Packing is driven by VarlenBatchMeta stream lengths via pack_named_streams.
        stream_list = []
        t_stream_list = []
        freq_parts = []  # only populated for the one-time RoPE self-check
        segment_infos = []
        window_lengths = []
        pred_targets = []
        # One-time guard: verify the metadata-driven RoPE assembler reproduces the
        # legacy per-sample full_freqs on the real config/data, then disable.
        rope_self_check = bool(getattr(self, "_varlen_rope_self_check", True))

        sample_preprocessor = MotusVarlenSamplePreprocessor(self)
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for i in range(B):
                prepared = sample_preprocessor.prepare(
                    latents_i=latents[i],
                    actions_i=actions[i],
                    action_valid_mask_i=(
                        None
                        if action_valid_mask is None
                        else action_valid_mask[i]
                    ),
                    video_valid_mask_i=(
                        None
                        if video_valid_mask is None
                        else video_valid_mask[i]
                    ),
                    states_i=None if states is None else states[i],
                    values_i=None if values is None else values[i],
                    window_length=window_lengths_input[i],
                    valid_len=valid_lens_l[i],
                    latent_start=latent_starts_l[i],
                    sigma_v=sigma_v[i],
                    sigma_a=sigma_a[i],
                    noisy_vid_t=noisy_vid_t[i],
                    noisy_act_t=noisy_act_t[i],
                    rope_self_check=rope_self_check,
                )
                stream_list.append(prepared.stream)
                t_stream_list.append(prepared.timestep_stream)
                if prepared.freq_part is not None:
                    freq_parts.append(prepared.freq_part)
                window_lengths.append(prepared.window_length)
                segment_infos.append(prepared.segment_info)
                pred_targets.append(prepared.pred_targets)

            retained = self._training_retention(window_lengths, obs, chunk_size)
            layout = build_memory_varlen_batch(
                window_lengths,
                video_tokens_per_frame=self.frame_seqlen,
                gist_tokens_per_frame=self._memory_gist_tokens(),
                action_tokens_per_frame=self.config.action_tokens_per_frame,
                chunk_size=chunk_size,
                anchor_frames=self._memory_anchor_frames(),
                recent_frames=self._memory_recent_frames(),
                use_state=self.config.use_state,
                state_tokens_per_frame=self.config.state_tokens_per_frame,
                value_tokens_per_frame=self.config.value_tokens_per_frame,
                attention_mode=self.config.mode,
                device=device,
                separate_anchor=False,
                obs_latents=obs,
                caption_lengths=caption_lengths_l,
                caption_padded_lengths=caption_padded_lengths_l,
                valid_lens=valid_lens_l,
                latent_starts=latent_starts_l,
                latent_hw=(self.lat_H, self.lat_W),
                retained=retained,
            )
            ar_mask = layout.block_mask
            padded_len = layout.padded_total
            # Stream lengths in `layout` drive packing (true multi-stream packing).
            x = pack_named_streams(stream_list, layout, pad_to=padded_len, output_batch_dim=True).tensor
            combined_t_vals = pack_named_streams(t_stream_list, layout, pad_to=padded_len, output_batch_dim=True).tensor
            # RoPE freqs assembled from metadata offsets instead of inline materialization.
            full_freqs = assemble_packed_freqs(
                layout,
                video_freq_fn=self._create_video_freqs,
                action_freqs=self.freqs_action,
                action_freq_fn=(
                    self._action_freqs_video_basis
                    if self.config.action_rope_mode == "video_shared"
                    else None
                ),
                identity_template=self.freqs_state,
                pad_to=padded_len,
                output_batch_dim=False,
                gist_use_frame_rope=True,
            ).tensor
            if rope_self_check:
                legacy_full_freqs = pack_streams(
                    freq_parts, pad_to=padded_len, output_batch_dim=False, pad_value=1.0,
                ).tensor
                # One-time parity check on CPU: torch.allclose on complex CUDA
                # tensors JIT-compiles a jiterator kernel that fails to build on
                # some GPU/toolkit combos (e.g. Blackwell + cu128 nvrtc). The
                # comparison is one-time and tiny, so run it on host tensors.
                if not torch.allclose(full_freqs.detach().cpu(), legacy_full_freqs.detach().cpu()):
                    raise RuntimeError(
                        "packed varlen RoPE assembler diverged from legacy full_freqs; "
                        "metadata-driven RoPE is unsafe for this config"
                    )
                self._varlen_rope_self_check = False
            _, combined_e0 = self._video_time_embed(combined_t_vals)

            for layer_idx in range(self.config.num_layers):
                if torch.is_grad_enabled() and self.video_model.wan_model.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        self._forward_layer_memory_packed,
                        x, combined_e0, layer_idx, t5_ctx, ar_mask, full_freqs, layout,
                        use_reentrant=False,
                    )
                else:
                    x = self._forward_layer_memory_packed(
                        x, combined_e0, layer_idx, t5_ctx, ar_mask, full_freqs, layout
                    )

            pred_noisy_list = []
            target_video_list = []
            action_pred_list = []
            target_action_list = []
            value_hidden_list = []
            noisy_lengths = []
            # Metadata-driven unpacking (vars_split analogue): slice each segment's
            # named streams from the packed row using layout offsets.
            seg_views = unpack_named_streams(x, layout)
            for seg_idx, (info, targets) in enumerate(zip(segment_infos, pred_targets)):
                views = seg_views[seg_idx]
                chunk_ar_out = views["chunk_ar"]
                clean_video_hidden = views["clean_video"]
                if value_enabled:
                    sa_hidden, noisy_video_hidden, value_hidden_i = self._split_memory_chunk_ar_hidden(
                        chunk_ar_out, info["block_latent_counts"], include_value=True,
                    )
                    num_chunks = int(info["num_chunks"])
                    value_hidden_list.append(
                        value_hidden_i.reshape(1, num_chunks, nvpf, -1).to(self.dtype)
                    )
                else:
                    sa_hidden, noisy_video_hidden = self._split_memory_chunk_ar_hidden(
                        chunk_ar_out, info["block_latent_counts"],
                    )
                video_out_i = torch.cat([clean_video_hidden, noisy_video_hidden], dim=1)
                action_out_i = self.action_expert.projection.project_out(sa_hidden)
                video_pred_i = self.video_model.wan_model.head(video_out_i, info["v_emb_full"])
                grid_size = torch.tensor([[2 * info["N"], self.lat_H, self.lat_W]], dtype=torch.long, device=device)
                video_pred_i = self.video_model.wan_model.unpatchify(video_pred_i, grid_size)
                if isinstance(video_pred_i, list):
                    video_pred_i = torch.stack(video_pred_i)
                action_pred_i = self.action_expert.decoder(action_out_i, info["a_emb"])
                action_pred_i = self._strip_chunk_state_predictions(
                    action_pred_i, info["block_action_token_lengths"]
                )
                N_i = info["N"]
                pred_noisy_list.append(video_pred_i[:, :, N_i:, :, :].to(self.dtype))
                target_video_list.append(targets[0].to(self.dtype))
                action_pred_list.append(action_pred_i.to(self.dtype))
                target_action_list.append(targets[1].to(self.dtype))
                noisy_lengths.append(N_i)

        bsmntw_v = self.scheduler.training_weight(
            noisy_vid_t.detach().cpu()
        ).to(device=device, dtype=self.dtype)
        bsmntw_a = self.action_scheduler.training_weight(
            noisy_act_t.detach().cpu()
        ).to(device=device, dtype=self.dtype)

        video_losses = []
        action_losses = []
        action_raw_losses = []
        action_pos_raw_losses = []
        action_rot_raw_losses = []
        action_grip_raw_losses = []
        value_losses = []
        prefix_frames = int(getattr(self.config, "memory_prefix_frames", 0))
        for i, N_i in enumerate(noisy_lengths):
            valid_N_i = int(segment_infos[i].get("valid_N", N_i))
            prefix_noisy = min(max(prefix_frames - obs, 0), N_i)
            valid_start = min(prefix_noisy, N_i)
            valid_end = min(max(valid_N_i, valid_start), N_i)

            v_mask = torch.zeros((1, 1, N_i, 1, 1), device=device, dtype=self.dtype)
            if valid_end > valid_start:
                v_mask[:, :, valid_start:valid_end] = 1.0
            source_video_valid = segment_infos[i].get("video_valid_mask")
            if source_video_valid is not None:
                source_video_valid = source_video_valid.to(
                    device=device,
                    dtype=self.dtype,
                ).reshape(1, 1, N_i, 1, 1)
                v_mask = v_mask * source_video_valid
            if use_vm[i]:
                v_mask = v_mask * 0.0
            mse_v = F.mse_loss(pred_noisy_list[i], target_video_list[i], reduction="none") * v_mask
            denom_v = v_mask.sum().clamp(min=1) * pred_noisy_list[i].shape[1] * pred_noisy_list[i].shape[3] * pred_noisy_list[i].shape[4]
            v_loss_i = mse_v.sum() / denom_v
            if inactive_irrelevant[i]:
                v_loss_i = v_loss_i * 0.0
            video_losses.append(v_loss_i * bsmntw_v[i])

            a_start = valid_start * apl
            a_end = valid_end * apl
            total_action_len = N_i * apl
            action_token_mask = torch.zeros((1, total_action_len), device=device, dtype=self.dtype)
            if a_end > a_start and not (use_wm[i] or use_vm[i] or inactive_irrelevant[i]):
                action_token_mask[:, a_start:a_end] = 1.0
            source_action_valid = segment_infos[i].get("action_valid_mask")
            if source_action_valid is not None:
                source_action_valid = source_action_valid.to(
                    device=device,
                    dtype=self.dtype,
                ).reshape(1, total_action_len)
                action_token_mask = action_token_mask * source_action_valid
            action_terms = self._compute_action_loss_terms(
                action_pred_list[i],
                target_action_list[i],
                action_token_mask,
                global_step=global_step,
            )
            capture = getattr(self, "_capture_action_tokens", None)
            if capture is not None:
                raw_tokens = F.mse_loss(
                    action_pred_list[i].float(), target_action_list[i].float(), reduction="none"
                ).mean(dim=2)
                capture.append({
                    "raw_per_token": raw_tokens[0].detach().cpu(),
                    "mask": action_token_mask[0].detach().float().cpu(),
                    "obs": obs,
                    "actions_per_latent": apl,
                })
            action_loss_i = action_terms["loss_per_sample"].mean()
            action_raw_i = action_terms["raw_per_sample"].mean()
            action_losses.append(action_loss_i * bsmntw_a[i])
            action_raw_losses.append(action_raw_i)
            action_pos_raw_losses.append(action_terms["pos_raw_loss"])
            action_rot_raw_losses.append(action_terms["rot_raw_loss"])
            action_grip_raw_losses.append(action_terms["grip_raw_loss"])

            if value_enabled and value_hidden_list and segment_infos[i].get("values") is not None:
                from .value_head import discretize
                value_hidden_i = value_hidden_list[i]
                value_logits = self.value_head(value_hidden_i.to(self.dtype))
                values_i = segment_infos[i]["values"].to(device=device, dtype=self.dtype)
                noisy_values = values_i[obs:].unsqueeze(0)
                block_latent_counts = segment_infos[i]["block_latent_counts"]
                chunk_values = self._select_chunk_end_values(noisy_values, block_latent_counts)
                value_targets = discretize(
                    chunk_values, self.config.value_num_bins, *self.config.value_range,
                )
                value_targets_expanded = value_targets.unsqueeze(2).expand(-1, -1, nvpf)
                value_ce = F.cross_entropy(
                    value_logits.reshape(-1, self.config.value_num_bins),
                    value_targets_expanded.reshape(-1),
                    reduction="none",
                ).reshape(1, len(block_latent_counts), nvpf).mean(dim=2)
                value_loss_mask = use_vm[i].float().view(1, 1)
                if inactive_irrelevant[i]:
                    value_loss_mask = value_loss_mask * 0.0
                chunk_end_idx = torch.tensor(
                    self._chunk_end_indices(block_latent_counts), device=device, dtype=torch.long,
                ).unsqueeze(0)
                chunk_mask = (chunk_end_idx < valid_N_i).float()
                value_loss_mask = value_loss_mask * chunk_mask
                v_loss_i = (value_ce * value_loss_mask).sum() / value_loss_mask.sum().clamp(min=1)
                value_losses.append(v_loss_i)
            elif value_enabled:
                value_losses.append(torch.tensor(0.0, device=device))

        video_loss = torch.stack(video_losses).mean()
        action_loss = torch.stack(action_losses).mean()
        action_raw_loss = torch.stack(action_raw_losses).mean()
        action_pos_raw_loss = torch.stack(action_pos_raw_losses).mean()
        action_rot_raw_loss = torch.stack(action_rot_raw_losses).mean()
        action_grip_raw_loss = torch.stack(action_grip_raw_losses).mean()
        value_loss_weight = self.config.value_loss_weight
        total_loss = self.config.video_loss_weight * video_loss + self.config.action_loss_weight * action_loss
        if value_enabled and value_losses:
            value_loss = torch.stack(value_losses).mean()
            total_loss = total_loss + value_loss_weight * value_loss
        else:
            value_loss = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)
        return {
            "total_loss": total_loss,
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "action_raw_loss": action_raw_loss.detach(),
            "action_pos_raw_loss": action_pos_raw_loss.detach(),
            "action_rot_raw_loss": action_rot_raw_loss.detach(),
            "action_grip_raw_loss": action_grip_raw_loss.detach(),
            "action_hand_raw_loss": (
                action_grip_raw_loss.detach()
                if self.config.action_mode == "qpos20_rel6d"
                else zero
            ),
            "value_loss": value_loss.detach(),
            "wm_ratio": use_wm.float().mean().detach(),
            "vm_ratio": use_vm.float().mean().detach(),
            "avg_sigma_video": sigma_v.mean().detach(),
            "avg_sigma_action": sigma_a.mean().detach(),
            "video_loss_weight": self.config.video_loss_weight,
            "action_loss_weight": self.config.action_loss_weight,
            "value_loss_weight": value_loss_weight,
            "num_vm_samples": float(len(vm_idx)),
        }

    def _apply_self_forcing_clean_latents(self, clean_latents: torch.Tensor) -> torch.Tensor:
        """Mix clean conditioning latents with Gaussian noise (MemoryWAM §4.1 self-forcing)."""
        prob = float(self.config.noisy_history_prob)
        if prob <= 0.0:
            return clean_latents
        B, _, N, _, _ = clean_latents.shape
        device = clean_latents.device
        sigma = torch.tensor(
            conditioning_noise_sigma(
                N,
                float(self.config.noisy_history_sigma_max),
                int(self.config.noisy_history_clean_prefix),
            ),
            device=device,
            dtype=self.dtype,
        ).view(1, N)
        if not bool((sigma > 0).any()):
            return clean_latents
        alpha = torch.rand(B, N, device=device, dtype=self.dtype) * sigma
        if prob < 1.0:
            keep = (torch.rand(B, N, device=device) < prob).to(self.dtype)
            alpha = alpha * keep
        noise = torch.randn_like(clean_latents)
        s = alpha.view(B, 1, N, 1, 1)
        return (1 - s) * clean_latents + s * noise

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        latents: torch.Tensor,            # [B, C, W, H_lat, W_lat] pre-computed VAE latents
        actions: torch.Tensor,             # [B, W * actions_per_latent, action_dim]
        states: torch.Tensor = None,       # [B, W, state_dim]
        language_embeddings=None,
        valid_len: torch.Tensor = None,    # [B] int, actual latent frames before padding
        action_valid_mask: torch.Tensor = None,  # [B, W * actions_per_latent]
        video_valid_mask: torch.Tensor = None,   # [B, W]
        latent_start: torch.Tensor = None, # [B] int, absolute start index in phase latent trajectory
        values: torch.Tensor = None,       # [B, W] MC returns per frame
        is_task_irrelevant: torch.Tensor = None,  # [B] {0,1}
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Sliding-window AR training step with pre-computed latents."""
        if isinstance(latents, (list, tuple)):
            return self.training_step_packed_varlen(
                latents=latents,
                actions=actions,
                states=states,
                language_embeddings=language_embeddings,
                valid_len=valid_len,
                action_valid_mask=action_valid_mask,
                video_valid_mask=video_valid_mask,
                latent_start=latent_start,
                values=values,
                is_task_irrelevant=is_task_irrelevant,
                global_step=global_step,
            )
        B = latents.shape[0]
        W = latents.shape[2]  # latent frames in window
        device = latents.device
        apl = self.config.actions_per_latent
        napf = self.config.noisy_action_tokens_per_frame
        nvpf = self.config.value_tokens_per_frame  # K value tokens per chunk
        latent_start = self._training_rope_starts(latent_start, B)

        # === 1. Fixed chunk size + random chunk-window for this iteration ===
        chunk_size = self._fixed_chunk_size()

        # === 2. Three-way mode selection: Policy / WM / Value ===
        use_wm = torch.zeros(B, dtype=torch.bool, device=device)
        use_vm = torch.zeros(B, dtype=torch.bool, device=device)
        irrelevant_mask = (
            is_task_irrelevant.to(device=device, dtype=torch.bool)
            if is_task_irrelevant is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        irrelevant_start_step = max(
            int(self.config.warmup_steps),
            int(self.config.uniform_loss_weight_warmup_steps),
        )
        active_irrelevant = irrelevant_mask & (global_step >= irrelevant_start_step)
        inactive_irrelevant = irrelevant_mask & ~active_irrelevant
        main_mask = ~irrelevant_mask

        if main_mask.any():
            main_idx = main_mask.nonzero(as_tuple=True)[0]
            rand_mode_main = torch.rand(len(main_idx), device=device)
            if self.config.value_enabled and self.config.vm_ratio > 0:
                use_vm_main = rand_mode_main < self.config.vm_ratio
                use_wm_main = (~use_vm_main) & (rand_mode_main < self.config.vm_ratio + self.config.wm_ratio)
                use_vm[main_idx] = use_vm_main
                use_wm[main_idx] = use_wm_main
            else:
                use_wm[main_idx] = rand_mode_main < self.config.wm_ratio

        if active_irrelevant.any():
            irr_idx = active_irrelevant.nonzero(as_tuple=True)[0]
            if self.config.value_enabled and self.config.vm_ratio > 0:
                wm_mass = max(float(self.config.wm_ratio), 0.0)
                vm_mass = max(float(self.config.vm_ratio), 0.0)
                total_mass = wm_mass + vm_mass
                if total_mass > 0:
                    irr_rand = torch.rand(len(irr_idx), device=device)
                    use_vm[irr_idx] = irr_rand < (vm_mass / total_mass)
                    use_wm[irr_idx] = ~use_vm[irr_idx]
                else:
                    use_wm[irr_idx] = True
            else:
                use_wm[irr_idx] = True

        vm_idx = use_vm.nonzero(as_tuple=True)[0]
        wm_idx = use_wm.nonzero(as_tuple=True)[0]
        pol_idx = (~inactive_irrelevant & ~use_wm & ~use_vm).nonzero(as_tuple=True)[0]

        # === 2b. Sample sigma directly (WAN-style: sigma is the true physical variable) ===
        # Convention: sigma=0 → clean, sigma=1 → pure noise.
        # Timestep = sigma * num_train_timesteps (WAN convention, fm.py:54).
        # This avoids the old t∈[0,1] → idx → scheduler.sigmas[idx] indirection
        # which had reversed semantics (t=0 mapped to sigmas[0]=max_noise).
        video_shift = float(self.config.video_timestep_shift)
        action_shift = float(self.config.action_timestep_shift)

        def _shifted_sigma(raw, shift):
            """Apply scheduler's shift transform: raw∈[0,1] → shifted sigma."""
            return shift * raw / (1 + (shift - 1) * raw)

        sigma_v = torch.zeros(B, device=device, dtype=self.dtype)
        sigma_a = torch.zeros(B, device=device, dtype=self.dtype)

        # Policy mode: both video and action are noisy
        if len(pol_idx) > 0:
            if self.config.policy_timestep_method == "beta":
                eta = Beta(self.config.beta_alpha, self.config.beta_beta).sample((len(pol_idx),))
                raw_v = eta.to(device)  # Beta(7,1) concentrated near 1 → high sigma
            else:
                from utils.common import sample_t
                t_pol = sample_t(self._policy_t_dist, len(pol_idx)).to(device=device, dtype=self.dtype)
                raw_v = (1.0 - t_pol).clamp(0, 1)
            sigma_v[pol_idx] = _shifted_sigma(raw_v, video_shift).to(self.dtype)
            if self.config.action_timestep_method == "beta":
                eta_a = Beta(self.config.beta_alpha, self.config.beta_beta).sample((len(pol_idx),))
                raw_a = eta_a.to(device)
            else:
                from utils.common import sample_t
                t_act = sample_t(self._action_t_dist, len(pol_idx)).to(device=device, dtype=self.dtype)
                raw_a = (1.0 - t_act).clamp(0, 1)
            sigma_a[pol_idx] = _shifted_sigma(raw_a, action_shift).to(self.dtype)

        # WM mode: noisy video, clean action (sigma_a stays 0)
        if len(wm_idx) > 0:
            u = torch.randn(len(wm_idx), device=device)
            raw_v_wm = (1.0 - torch.sigmoid(u)).clamp(0, 1)
            sigma_v[wm_idx] = _shifted_sigma(raw_v_wm, video_shift).to(self.dtype)
            # sigma_a[wm_idx] = 0 already (clean action condition)

        # VM mode: sigma_v=0, sigma_a=0 (clean everything, only value loss)
        # (already initialized to 0)

        # === 3. Split: clean 0..W-obs-1, noisy obs..W-1 ===
        # obs = observation latents (never predicted); defaults to chunk_size.
        # obs == 1 (only L0 is observed), so the first predicted chunk conditions on
        # exactly L0 — matching a cold deploy start that only has L0.
        cs = chunk_size
        obs = self._obs_latents()
        from .memory_rope import extend_valid_mask, pad_window_to_chunk, repeat_last
        W_real = W
        W, N = pad_window_to_chunk(W_real, obs, cs)
        if N < cs:
            raise ValueError(f"training window too short for one chunk: W={W_real}, obs={obs}, cs={cs}")
        apl = self.config.actions_per_latent
        pad = W - W_real
        if pad:
            latents = repeat_last(latents, 2, pad)
            actions = repeat_last(actions, 1, pad * apl)
            states = repeat_last(states, 1, pad)
            if values is not None:
                values = repeat_last(values, 1 if values.dim() > 1 else 0, pad)
            if action_valid_mask is not None:
                action_valid_mask = extend_valid_mask(action_valid_mask, pad * apl)
            if video_valid_mask is not None:
                video_valid_mask = extend_valid_mask(video_valid_mask, pad)
        if valid_len is None:
            valid_len = torch.full((B,), W_real, device=device, dtype=torch.long)
        valid_len = valid_len.clamp(max=W_real)
        clean_latents = latents[:, :, :N].to(self.dtype)       # [B, C, N, H, W] phys 0..W-obs-1
        noisy_latents = latents[:, :, obs:W].to(self.dtype)    # [B, C, N, H, W] phys obs..W-1
        action_block_info = self._prepare_noisy_action_blocks(
            actions, states, cs, dtype=self.dtype, obs_latents=obs
        )
        noisy_actions = action_block_info["flat_actions"]
        noisy_states = action_block_info["states"]
        block_latent_counts = action_block_info["block_latent_counts"]
        block_action_token_lengths = action_block_info["block_action_token_lengths"]
        num_noisy_chunks = len(block_latent_counts)
        target_actions_raw = noisy_actions.clone()

        # === 3.5 Self-forcing on clean conditioning latents (MemoryWAM §4.1) ===
        clean_latents = self._apply_self_forcing_clean_latents(clean_latents)

        # === 4. Noise + interpolate (noisy side only) ===
        noise_v = torch.randn_like(noisy_latents)
        sv = sigma_v.view(B, 1, 1, 1, 1)
        noisy_video = (1 - sv) * noisy_latents + sv * noise_v
        target_video = noise_v - noisy_latents

        noise_a = torch.randn_like(noisy_actions)
        sa = sigma_a.view(B, 1, 1)
        noisy_action = (1 - sa) * noisy_actions + sa * noise_a
        target_action = noise_a - target_actions_raw

        # === 5. Token embedding ===
        clean_video_tokens = self._video_patch_embed(clean_latents)
        noisy_video_tokens = self._video_patch_embed(noisy_video)
        half_seq_len = clean_video_tokens.shape[1]

        noisy_action_chunks = self._split_flat_action_chunks(noisy_action, block_action_token_lengths)
        action_tokens = self._encode_action_chunk_tokens(noisy_action_chunks, noisy_states)
        sa_action_seq_len = action_tokens.shape[1]

        action_tokens = self.action_expert.projection.project_in(action_tokens)
            # action_tokens: [B, N*(state+action), wan_dim]

        # === 5b. Value query tokens (optional; legacy MoT path only) ===
        value_hidden = None  # set after transformer if value_enabled
        value_query = None   # separate tensor for use_mot=True
        if self.config.value_enabled and nvpf > 0:
            value_query = self.value_tokens.expand(B, num_noisy_chunks)

        # === 7. Timesteps (sigma -> WAN timestep) ===
        noisy_vid_t = (sigma_v * self.scheduler.num_train_timesteps).to(self.dtype)
        noisy_act_t = (sigma_a * self.action_scheduler.num_train_timesteps).to(self.dtype)
        clean_t_vals = torch.zeros(B, half_seq_len, dtype=self.dtype, device=device)
        noisy_v_t_vals = noisy_vid_t.view(B, 1).expand(-1, half_seq_len)

        # === 9. Time embeddings (shared by both paths) ===
        # Video head: time embedding over [clean_video | noisy_video] (2N frames).
        v_emb_full, v_e0 = self._video_time_embed(torch.cat([clean_t_vals, noisy_v_t_vals], dim=1))
        # Action decoder: time embedding over [state?, action] per chunk.
        sa_t_vals = self._build_chunked_action_timesteps(noisy_act_t, block_latent_counts, include_value=False)
        a_emb, a_e0 = self._action_time_embed(sa_t_vals)

        # === 11. T5 cross-attention context ===
        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            if self.config.value_enabled and nvpf > 0:
                raise NotImplementedError(
                    "value tokens are not supported in the action-first memory path; "
                    "this MemoryWAM implementation supports policy/world-model training."
                )
            gist_tokens = self.gist_tokens.expand(B, N)
            chunk_ar_tokens = self._interleave_memory_chunk_ar_tokens(
                action_tokens, noisy_video_tokens, block_latent_counts,
            )
            anchor_len = 0
            x = torch.cat([clean_video_tokens, gist_tokens, chunk_ar_tokens], dim=1)
            clean_region_len = anchor_len + clean_video_tokens.shape[1] + gist_tokens.shape[1]
            chunk_ar_len = chunk_ar_tokens.shape[1]
            total_seq_len = x.shape[1]

            ar_mask, padded_len = self._create_training_mask(W, chunk_size)

            gist_t_vals = torch.zeros(B, gist_tokens.shape[1], dtype=self.dtype, device=device)
            chunk_ar_t_vals = self._build_memory_chunk_ar_timesteps(
                noisy_act_t, noisy_vid_t, block_latent_counts,
            )
            anchor_t_vals = torch.zeros(B, anchor_len, dtype=self.dtype, device=device)
            combined_t_vals = torch.cat([anchor_t_vals, clean_t_vals, gist_t_vals, chunk_ar_t_vals], dim=1)
            if total_seq_len < padded_len:
                pad_len = padded_len - total_seq_len
                x = torch.cat([x, x.new_zeros(B, pad_len, x.shape[2])], dim=1)
                combined_t_vals = torch.cat([combined_t_vals, combined_t_vals.new_zeros(B, pad_len)], dim=1)
            _, combined_e0 = self._video_time_embed(combined_t_vals)

            if latent_start is not None:
                latent_start = latent_start.to(device=device, dtype=torch.long)
                clean_video_freqs = self._create_video_freqs_batch(N, latent_start)
                noisy_video_freqs = self._create_video_freqs_batch(N, latent_start + obs)
                noisy_rope_start = latent_start + obs
            else:
                clean_video_freqs = self._create_video_freqs(N, start_frame=0)
                noisy_video_freqs = self._create_video_freqs(N, start_frame=obs)
                noisy_rope_start = obs
            full_freqs = self._build_memory_full_freqs(
                clean_video_freqs, noisy_video_freqs, block_latent_counts,
                noisy_start=noisy_rope_start,
            )

            for layer_idx in range(self.config.num_layers):
                if torch.is_grad_enabled() and self.video_model.wan_model.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        self._forward_layer_memory, x, combined_e0, layer_idx, t5_ctx, ar_mask, full_freqs,
                        use_reentrant=False,
                    )
                else:
                    x = self._forward_layer_memory(x, combined_e0, layer_idx, t5_ctx, ar_mask, full_freqs)

            clean_video_hidden = x[:, anchor_len:anchor_len + clean_video_tokens.shape[1]]
            chunk_ar_out = x[:, clean_region_len:clean_region_len + chunk_ar_len]
            sa_hidden, noisy_video_hidden = self._split_memory_chunk_ar_hidden(
                chunk_ar_out, block_latent_counts,
            )
            video_out = torch.cat([clean_video_hidden, noisy_video_hidden], dim=1)
            action_out = self.action_expert.projection.project_out(sa_hidden)

            # === 14. Heads ===
            video_pred = self.video_model.wan_model.head(video_out, v_emb_full)
            grid_size = torch.tensor([[2 * N, self.lat_H, self.lat_W]], dtype=torch.long).expand(B, -1)
            video_pred = self.video_model.wan_model.unpatchify(video_pred, grid_size)
            if isinstance(video_pred, list):
                video_pred = torch.stack(video_pred)

            action_pred = self.action_expert.decoder(action_out, a_emb)

        # Extract action predictions (skip state tokens when use_state)
        # napf here is the full noisy_action_tokens_per_frame (including value), but decoder
        # only receives state+action tokens. Use the state+action count for reshaping.
        action_pred = self._strip_chunk_state_predictions(action_pred, block_action_token_lengths)

        # === 15. Loss ===
        pred_noisy = video_pred[:, :, N:, :, :].to(self.dtype)  # noisy half of unpatchified output
        noisy_valid_len = (valid_len - obs).clamp(min=0) if valid_len is not None else None
        # Prefix-prefill: the first ``memory_prefix_frames`` latent frames are
        # warm-up history. They condition the anchor/gist/recent memory (via the
        # hybrid mask) but contribute no loss → truncated supervision on the
        # target segment only. A predicted frame at physical position cs+j is
        # warm-up when cs+j < P, i.e. noisy index j < P - cs.
        prefix_frames = int(getattr(self.config, "memory_prefix_frames", 0))
        prefix_noisy = min(max(prefix_frames - obs, 0), N)
        loss_mask_v = torch.ones(B, 1, N, 1, 1, device=device, dtype=self.dtype)
        if prefix_noisy > 0:
            loss_mask_v[:, :, :prefix_noisy] = 0.0
        if noisy_valid_len is not None:
            t_arange = torch.arange(N, device=device)
            invalid = t_arange.unsqueeze(0) >= noisy_valid_len.unsqueeze(1)
            loss_mask_v = loss_mask_v * (1.0 - invalid.to(self.dtype).unsqueeze(1).unsqueeze(3).unsqueeze(4))
        if video_valid_mask is not None:
            noisy_video_valid = video_valid_mask[
                :,
                obs : obs + N,
            ].to(device=device, dtype=self.dtype)
            loss_mask_v = loss_mask_v * noisy_video_valid.view(B, 1, N, 1, 1)
        if inactive_irrelevant.any():
            loss_mask_v[inactive_irrelevant] = 0.0
        # Value mode: zero out video loss (value mode only trains value head)
        if len(vm_idx) > 0:
            loss_mask_v[vm_idx] = 0.0

        bsmntw_v = self.scheduler.training_weight(
            noisy_vid_t.detach().cpu()
        ).to(device=device, dtype=self.dtype)
        bsmntw_a = self.action_scheduler.training_weight(
            noisy_act_t.detach().cpu()
        ).to(device=device, dtype=self.dtype)

        mse_v = F.mse_loss(pred_noisy, target_video, reduction="none") * loss_mask_v
        C_lat, H_lat, W_lat = pred_noisy.shape[1], pred_noisy.shape[3], pred_noisy.shape[4]
        video_loss_per_sample = mse_v.sum(dim=(1, 2, 3, 4)) / (loss_mask_v.sum(dim=(1, 2, 3, 4)).clamp(min=1) * C_lat * H_lat * W_lat)
        video_loss = (video_loss_per_sample * bsmntw_v).mean()

        loss_mask_a = torch.ones(B, device=device, dtype=self.dtype)
        if len(wm_idx) > 0:
            loss_mask_a[wm_idx] = 0.0
        if len(vm_idx) > 0:
            loss_mask_a[vm_idx] = 0.0  # Value mode samples don't contribute to action loss
        if inactive_irrelevant.any():
            loss_mask_a[inactive_irrelevant] = 0.0
        action_pos_raw_loss = torch.tensor(0.0, device=device)
        action_rot_raw_loss = torch.tensor(0.0, device=device)
        action_grip_raw_loss = torch.tensor(0.0, device=device)
        action_raw_loss = torch.tensor(0.0, device=device)
        eef_component_warmup_active = (
            self.config.action_mode in {
                "rel_eef", "rel_eef_6d", "abs_eef_rpy",
                "abs_eef", "abs_eef_wxyz", "rel_eef_wxyz", "qpos20_rel6d",
            }
            and global_step < self.config.uniform_loss_weight_warmup_steps
        )
        eef_pos_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_position_loss_weight
        eef_rot_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_rotation_loss_weight
        eef_grip_weight = 1.0 if eef_component_warmup_active else self.config.rel_eef_gripper_loss_weight
        hand_qpos_weight = 1.0 if eef_component_warmup_active else self.config.hand_qpos_loss_weight

        if self.config.action_mode == "rel_eef":
            # 16D xyzw relative EEF: [xyz, quat_xyzw, grip] x 2.
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 8:11], target_action[..., 8:11], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:7], target_action[..., 3:7], reduction="none"),
                F.mse_loss(action_pred[..., 11:15], target_action[..., 11:15], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 7], target_action[..., 7], reduction="none"),
                F.mse_loss(action_pred[..., 15], target_action[..., 15], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode in {"rel_eef_6d", "abs_eef"}:
            # 20D rot6d EEF: [xyz, rot6d, grip] x 2.
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 10:13], target_action[..., 10:13], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:9], target_action[..., 3:9], reduction="none"),
                F.mse_loss(action_pred[..., 13:19], target_action[..., 13:19], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 9], target_action[..., 9], reduction="none"),
                F.mse_loss(action_pred[..., 19], target_action[..., 19], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "rel_eef_wxyz":
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 8:11], target_action[..., 8:11], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:7], target_action[..., 3:7], reduction="none"),
                F.mse_loss(action_pred[..., 11:15], target_action[..., 11:15], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 7], target_action[..., 7], reduction="none"),
                F.mse_loss(action_pred[..., 15], target_action[..., 15], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "abs_eef_wxyz":
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 8:11], target_action[..., 8:11], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 3:7], target_action[..., 3:7], reduction="none"),
                F.mse_loss(action_pred[..., 11:15], target_action[..., 11:15], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 7], target_action[..., 7], reduction="none"),
                F.mse_loss(action_pred[..., 15], target_action[..., 15], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "abs_eef_rpy":
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:3], target_action[..., 0:3], reduction="none"),
                F.mse_loss(action_pred[..., 7:10], target_action[..., 7:10], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.stack([
                _angle_mse_loss_wrap(action_pred[..., 3:6], target_action[..., 3:6]),
                _angle_mse_loss_wrap(action_pred[..., 10:13], target_action[..., 10:13]),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = torch.stack([
                F.mse_loss(action_pred[..., 6], target_action[..., 6], reduction="none"),
                F.mse_loss(action_pred[..., 13], target_action[..., 13], reduction="none"),
            ], dim=2).mean(dim=2)
            action_loss_per_token = (
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token +
                eef_grip_weight * grip_loss_per_token
            )
        elif self.config.action_mode == "qpos20_rel6d":
            qpos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 0:20], target_action[..., 0:20], reduction="none"),
                F.mse_loss(action_pred[..., 29:49], target_action[..., 29:49], reduction="none"),
            ], dim=2).mean(dim=2)
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 20:23], target_action[..., 20:23], reduction="none"),
                F.mse_loss(action_pred[..., 49:52], target_action[..., 49:52], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., 23:29], target_action[..., 23:29], reduction="none"),
                F.mse_loss(action_pred[..., 52:58], target_action[..., 52:58], reduction="none"),
            ], dim=2).mean(dim=2)
            grip_loss_per_token = qpos_loss_per_token
            action_loss_per_token = (
                hand_qpos_weight * qpos_loss_per_token +
                eef_pos_weight * pos_loss_per_token +
                eef_rot_weight * rot_loss_per_token
            )
        else:
            action_loss_per_token = F.mse_loss(action_pred, target_action, reduction="none").mean(dim=2)
        total_action_len = N * apl
        if noisy_valid_len is not None:
            valid_action_len = (noisy_valid_len * apl).clamp(max=total_action_len)
            action_token_mask = (torch.arange(total_action_len, device=device).unsqueeze(0) < valid_action_len.unsqueeze(1)).to(self.dtype)
            action_token_mask = action_token_mask * loss_mask_a.unsqueeze(1)
        else:
            action_token_mask = loss_mask_a.unsqueeze(1).expand(-1, total_action_len)
        if action_valid_mask is not None:
            noisy_action_valid = action_valid_mask[
                :,
                obs * apl : obs * apl + total_action_len,
            ].to(device=device, dtype=self.dtype)
            action_token_mask = action_token_mask * noisy_action_valid
        if prefix_noisy > 0:
            # Drop warm-up frames' action tokens (frame-major, apl tokens/frame).
            keep_action = (
                torch.arange(total_action_len, device=device) >= prefix_noisy * apl
            ).to(self.dtype)
            action_token_mask = action_token_mask * keep_action.unsqueeze(0)
        if self.config.action_mode in {
            "rel_eef", "rel_eef_6d", "abs_eef_rpy",
            "abs_eef", "abs_eef_wxyz", "rel_eef_wxyz", "qpos20_rel6d",
        }:
            denom = action_token_mask.sum(dim=1).clamp(min=1)
            action_pos_raw_per_sample = (pos_loss_per_token * action_token_mask).sum(dim=1) / denom
            action_rot_raw_per_sample = (rot_loss_per_token * action_token_mask).sum(dim=1) / denom
            action_grip_raw_per_sample = (grip_loss_per_token * action_token_mask).sum(dim=1) / denom
            action_raw_per_token = F.mse_loss(action_pred, target_action, reduction="none").mean(dim=2)
            action_raw_per_sample = (action_raw_per_token * action_token_mask).sum(dim=1) / denom
            action_pos_raw_loss = action_pos_raw_per_sample.mean()
            action_rot_raw_loss = action_rot_raw_per_sample.mean()
            action_grip_raw_loss = action_grip_raw_per_sample.mean()
            action_raw_loss = action_raw_per_sample.mean()
        action_loss_per_sample = (action_loss_per_token * action_token_mask).sum(dim=1) / action_token_mask.sum(dim=1).clamp(min=1)
        action_loss = (action_loss_per_sample * bsmntw_a).mean()

        video_loss_weight = self.config.video_loss_weight
        action_loss_weight = self.config.action_loss_weight
        value_loss_weight = self.config.value_loss_weight

        total_loss = video_loss_weight * video_loss + action_loss_weight * action_loss

        # === 16. Value loss ===
        value_loss = torch.tensor(0.0, device=device)
        if self.config.value_enabled and value_hidden is not None and values is not None:
            from .value_head import discretize
            value_logits = self.value_head(value_hidden.to(self.dtype))  # [B, M, K, num_bins]

            # GT chunk values: take the last latent value from each noisy chunk.
            noisy_values = values[:, cs:].to(self.dtype)  # [B, N]
            chunk_values = self._select_chunk_end_values(noisy_values, block_latent_counts)  # [B, M]
            value_targets = discretize(chunk_values, self.config.value_num_bins,
                                       *self.config.value_range)  # [B, N] long

            # CE loss: average over K query tokens per chunk
            value_targets_expanded = value_targets.unsqueeze(2).expand(-1, -1, nvpf)  # [B, M, K]
            value_ce = F.cross_entropy(
                value_logits.reshape(-1, self.config.value_num_bins),
                value_targets_expanded.reshape(-1),
                reduction='none'
            ).reshape(B, num_noisy_chunks, nvpf).mean(dim=2)

            # Only compute value loss for Value mode samples
            value_loss_mask = use_vm.float().unsqueeze(1)  # [B, 1]
            if inactive_irrelevant.any():
                value_loss_mask[inactive_irrelevant] = 0.0
            if noisy_valid_len is not None:
                chunk_end_idx = torch.tensor(
                    self._chunk_end_indices(block_latent_counts), device=device, dtype=torch.long
                ).unsqueeze(0)
                chunk_mask = (chunk_end_idx < noisy_valid_len.unsqueeze(1)).float()
                value_loss_mask = value_loss_mask * chunk_mask

            value_loss = (value_ce * value_loss_mask).sum() / value_loss_mask.sum().clamp(min=1)
            total_loss = total_loss + value_loss_weight * value_loss

        return {
            "total_loss": total_loss,
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "action_raw_loss": action_raw_loss.detach(),
            "action_pos_raw_loss": action_pos_raw_loss.detach(),
            "action_rot_raw_loss": action_rot_raw_loss.detach(),
            "action_grip_raw_loss": action_grip_raw_loss.detach(),
            "action_hand_raw_loss": (
                action_grip_raw_loss.detach()
                if self.config.action_mode == "qpos20_rel6d"
                else torch.tensor(0.0, device=device)
            ),
            "value_loss": value_loss.detach(),
            "video_loss_weight": video_loss_weight,
            "action_loss_weight": action_loss_weight,
            "value_loss_weight": value_loss_weight,
            "eef_component_warmup_active": float(eef_component_warmup_active),
            "eef_position_loss_weight": eef_pos_weight,
            "eef_rotation_loss_weight": eef_rot_weight,
            "eef_gripper_loss_weight": eef_grip_weight,
            "hand_qpos_loss_weight": hand_qpos_weight,
            "num_vm_samples": len(vm_idx),
            "num_wm_samples": len(wm_idx),
            "num_pol_samples": len(pol_idx),
        }

    # ------------------------------------------------------------------
    # Inference: KV cache
    # ------------------------------------------------------------------

    def init_kv_cache(self, batch_size: int = 1, start_frame_abs: int = 0):
        """Initialize empty KV caches for all layers."""
        n = self.video_model.wan_model.num_heads
        d = self.wan_dim // n
        self.kv_caches = []
        for _ in range(self.config.num_layers):
            self.kv_caches.append(
                torch.zeros(2, batch_size, 0, n, d, device=self.device, dtype=self.dtype)
            )
        self.current_start_frame = int(start_frame_abs)
        self._last_infer_K = self._fixed_chunk_size()
        # Hybrid-memory cache bookkeeping (shared across layers): per-token absolute
        # frame id and whether the token is a gist (permanent) token.
        self._cache_frame_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self._cache_is_gist = torch.zeros(0, dtype=torch.bool, device=self.device)
        self._cache_first_frame = int(start_frame_abs)
        self._reset_memory_selector(start_frame_abs)
        logger.info("KV cache initialized at start_frame=%s", self.current_start_frame)

    def _memory_selector_runtime(self):
        kind = str(getattr(self.config, "memory_selector", "none") or "none")
        if kind == "none":
            return None
        selector = getattr(self, "_memory_selector", None)
        if selector is None:
            from .memory_selector import MemorySelector, load_write_gate

            gate = None
            if kind == "gate":
                gate = load_write_gate(
                    self.config.memory_selector_gate_path,
                    self.config.memory_selector_gate_sha256,
                )
            selector = MemorySelector(
                kind,
                self.config.memory_selector_budget,
                seed=self.config.memory_selector_seed,
                gate=gate,
            )
            self._memory_selector = selector
        return selector

    def _reset_memory_selector(self, start_frame_abs: int = 0) -> None:
        selector = self._memory_selector_runtime()
        if selector is not None:
            selector.reset(int(start_frame_abs))

    def clear_kv_cache(self, start_frame_abs: int = 0):
        self.kv_caches = None
        self.current_start_frame = int(start_frame_abs)
        self._last_infer_K = self._fixed_chunk_size()
        self._cache_frame_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self._cache_is_gist = torch.zeros(0, dtype=torch.bool, device=self.device)
        self._cache_first_frame = int(start_frame_abs)
        self._reset_memory_selector(start_frame_abs)

    def set_current_start_frame(self, start_frame_abs: int) -> None:
        self.current_start_frame = int(start_frame_abs)

    def export_inference_state(self) -> Dict[str, Any]:
        """Return a cloneable snapshot of streaming inference state."""
        kv_caches = None
        if self.kv_caches is not None:
            kv_caches = [cache.clone() for cache in self.kv_caches]
        return {
            "kv_caches": kv_caches,
            "current_start_frame": int(self.current_start_frame),
            "last_infer_K": int(getattr(self, "_last_infer_K", self._fixed_chunk_size())),
            "cache_frame_ids": self._cache_frame_ids.clone() if getattr(self, "_cache_frame_ids", None) is not None else None,
            "cache_is_gist": self._cache_is_gist.clone() if getattr(self, "_cache_is_gist", None) is not None else None,
            "cache_first_frame": int(getattr(self, "_cache_first_frame", self.current_start_frame)),
        }

    def load_inference_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore streaming inference state from export_inference_state()."""
        if state is None:
            self.clear_kv_cache()
            return
        kv_caches = state.get("kv_caches")
        if kv_caches is None:
            self.kv_caches = None
        else:
            self.kv_caches = [cache.clone().to(device=self.device, dtype=self.dtype) for cache in kv_caches]
        self.current_start_frame = int(state.get("current_start_frame", 0))
        self._last_infer_K = int(state.get("last_infer_K", self._fixed_chunk_size()))
        cfi = state.get("cache_frame_ids")
        cig = state.get("cache_is_gist")
        self._cache_frame_ids = (
            cfi.clone().to(device=self.device) if cfi is not None
            else torch.zeros(0, dtype=torch.long, device=self.device)
        )
        self._cache_is_gist = (
            cig.clone().to(device=self.device) if cig is not None
            else torch.zeros(0, dtype=torch.bool, device=self.device)
        )
        self._cache_first_frame = int(state.get("cache_first_frame", self.current_start_frame))

    @torch.no_grad()
    def fill_kv_cache(self, obs_frames: torch.Tensor, is_initial: bool = False,
                      language_embeddings=None):
        """Encode observation frames and fill KV cache (conditioning frame).

        Hybrid-memory version: each call writes the new frame's full clean KV +
        gist KV, then evicts non-anchor full frames that have left the recent
        window (gist + optional anchor + recent_full are retained). See
        ``_fill_cache_from_latent``. No frame replication, no DreamZero reset.

        Closed-loop inference must call this with real observation frames, not
        lasting ``fill_kv_cache_from_latent`` on predicted latents.

        Args:
            obs_frames: [B, C, T, H, W] raw observation frames.
                        is_initial=True: T=1 (single frame, uses encode_chunk_first)
                        is_initial=False: T=4 (4 frames, uses encode_chunk)
            language_embeddings: T5 embeddings for cross-attention.
        """
        if is_initial:
            # Single real C0 only — no L0 repeat to fixed_chunk_size (obs_latents=1).
            obs_norm = (obs_frames * 2.0 - 1.0).to(self.dtype)
            self.video_model.vae.clear_cache()
            latent = self.video_model.vae.encode_chunk_first(obs_norm).to(self.dtype)
        else:
            B, C, T, H, W = obs_frames.shape
            assert T == 4, f"Streaming inference expects exactly 4 frames per chunk, got {T}"
            obs_norm = (obs_frames * 2.0 - 1.0).to(self.dtype)
            latent = self.video_model.vae.encode_chunk(obs_norm).to(self.dtype)

        self._fill_cache_from_latent(latent, language_embeddings)

    def _fill_cache_from_latent(self, latent: torch.Tensor, language_embeddings=None):
        """Shared logic: patchify latent → forward through all layers → update KV cache.

        Used by both fill_kv_cache (real obs → VAE encode → latent) and
        fill_kv_cache_from_latent (imagined latent, skip VAE).
        """
        latent = latent.to(self.dtype)
        K = int(latent.shape[2])             # new clean frames this call
        B = int(latent.shape[0])
        M = self._memory_gist_tokens()
        vbs = self.frame_seqlen
        new_frame_start = int(self.current_start_frame)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # Sequence written this step: [clean_video[f] | gist[f]] for the K new frames.
            clean_tokens = self._video_patch_embed(latent)        # [B, K*vbs, dim]
            gist_tokens = (
                self.gist_tokens.expand(B, K) if M > 0 else clean_tokens[:, :0]
            )
            x = torch.cat([clean_tokens, gist_tokens], dim=1)
            clean_len = clean_tokens.shape[1]
            gist_len = gist_tokens.shape[1]
            seq_len = x.shape[1]

            t_zero = torch.zeros(B, seq_len, device=self.device)
            _, e0 = self._video_time_embed(t_zero)
            t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

            self._ensure_freqs_device()
            # RoPE: clean video gets global absolute 3D freqs; gist shares frame temporal RoPE.
            video_freqs = self._create_video_freqs(K, start_frame=new_frame_start)  # [K*vbs,1,C]
            if gist_len > 0:
                gist_freqs = self._create_gist_freqs_from_clean(video_freqs, K)
                fill_freqs = torch.cat([video_freqs, gist_freqs], dim=0)
            else:
                fill_freqs = video_freqs

            n = self.video_model.wan_model.num_heads
            d = self.wan_dim // n
            cache_len = int(self._cache_frame_ids.numel())
            selector = self._memory_selector_runtime()
            retained_frames = selector.retained(self.device) if selector is not None else None
            fill_mask = build_fill_cache_mask(
                self._cache_frame_ids.to(self.device) if cache_len else torch.zeros(0, dtype=torch.long, device=self.device),
                self._cache_is_gist.to(self.device) if cache_len else torch.zeros(0, dtype=torch.bool, device=self.device),
                new_frame_start,
                K,
                video_tokens_per_frame=vbs,
                gist_tokens_per_frame=M,
                anchor_frames=self._memory_anchor_frames(),
                recent_frames=self._memory_recent_frames(),
                obs_latents=self._obs_latents(),
                chunk_size=self._fixed_chunk_size(),
                cache_first_frame=int(self._cache_first_frame),
                separate_anchor=False,
                device=self.device,
                retained_frames=retained_frames,
            )
            for layer_idx in range(self.config.num_layers):
                blk = self.video_model.wan_model.blocks[layer_idx]
                mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)

                q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, seq_len, n, d)
                k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, seq_len, n, d)
                v = blk.self_attn.v(norm_x).view(B, seq_len, n, d)
                q = rope_action_apply(q, fill_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
                k = rope_action_apply(k, fill_freqs, self.freqs_action, self.freqs_state, None).type_as(v)

                cached_k, cached_v = self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]
                full_k = torch.cat([cached_k, k], dim=1)
                full_v = torch.cat([cached_v, v], dim=1)
                y = F.scaled_dot_product_attention(
                    q.transpose(1, 2), full_k.transpose(1, 2), full_v.transpose(1, 2),
                    attn_mask=fill_mask,
                )
                y = y.transpose(1, 2).reshape(B, seq_len, n * d)
                y = blk.self_attn.o(y)
                x = x + y * mod[2].squeeze(2)
                if t5_ctx is not None:
                    x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
                ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
                self.kv_caches[layer_idx] = torch.stack([full_k, full_v])

        # --- Update per-token metadata, then hybrid-evict non-anchor/non-recent full frames ---
        device = self.device
        frame_range = torch.arange(new_frame_start, new_frame_start + K, device=device)
        clean_ids = torch.repeat_interleave(frame_range, vbs)
        new_ids = [clean_ids]
        new_gist_flags = [torch.zeros(clean_len, dtype=torch.bool, device=device)]
        if gist_len > 0:
            new_ids.append(torch.repeat_interleave(frame_range, M))
            new_gist_flags.append(torch.ones(gist_len, dtype=torch.bool, device=device))
        self._cache_frame_ids = torch.cat([self._cache_frame_ids.to(device)] + new_ids)
        self._cache_is_gist = torch.cat([self._cache_is_gist.to(device)] + new_gist_flags)

        fmax = new_frame_start + K - 1
        recent_lo = fmax - self._memory_recent_frames() + 1
        keep = self._cache_is_gist | (self._cache_frame_ids >= recent_lo)
        if self._memory_anchor_frames() > 0:
            anchor_hi = int(self._cache_first_frame) + self._memory_anchor_frames()
            keep = keep | (self._cache_frame_ids < anchor_hi)
        if selector is not None:
            selector.observe(new_frame_start, latent, language_embeddings)
            selector.advance(recent_lo)
            retained_now = selector.retained(device)
            if retained_now.numel() > 0:
                keep = keep | (torch.isin(self._cache_frame_ids, retained_now) & ~self._cache_is_gist)
        if not bool(keep.all()):
            idx = keep.nonzero(as_tuple=True)[0]
            for layer_idx in range(self.config.num_layers):
                self.kv_caches[layer_idx] = self.kv_caches[layer_idx][:, :, idx, :, :].contiguous()
            self._cache_frame_ids = self._cache_frame_ids[idx]
            self._cache_is_gist = self._cache_is_gist[idx]

        self.current_start_frame += K
        logger.debug(
            "KV cache filled: start_frame=%s, cache_tokens=%s",
            self.current_start_frame, int(self._cache_frame_ids.numel()),
        )

    @torch.no_grad()
    def fill_kv_cache_from_latent(self, video_latent: torch.Tensor,
                                   language_embeddings=None):
        """Fill KV cache from an already-encoded video latent (for imagination rollout).

        Same as fill_kv_cache but skips VAE encoding — used when the latent
        comes from inference_step_streaming (imagined future) rather than real obs.

        Args:
            video_latent: [B, C, K, H_lat, W_lat] from inference_step_streaming.
            language_embeddings: T5 embeddings for cross-attention.
        """
        self._fill_cache_from_latent(video_latent, language_embeddings)

    def _build_frozen_video_layer_kv(
        self,
        z_tokens: torch.Tensor,
        t_vid: torch.Tensor,
        t5_ctx,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build per-layer frozen video K/V once for the IDM action-only phase."""
        B, v_seq_len, _ = z_tokens.shape
        n = self.video_model.wan_model.num_heads
        d = self.wan_dim // n
        x = z_tokens
        v_t_vals = torch.full((B, v_seq_len), t_vid.item(), device=self.device)
        _, e0 = self._video_time_embed(v_t_vals)
        infer_latent_frames = v_seq_len // self.frame_seqlen
        video_kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(self.config.num_layers):
            blk = self.video_model.wan_model.blocks[layer_idx]
            mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
            q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, v_seq_len, n, d)
            k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, v_seq_len, n, d)
            v = blk.self_attn.v(norm_x).view(B, v_seq_len, n, d)

            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
            cache_seq_len = kv_cache[0].shape[1] if kv_cache is not None else 0
            video_freqs = self._create_video_freqs(
                infer_latent_frames,
                start_frame=self._inference_video_start_frame(cache_seq_len),
            )
            q = rope_action_apply(q, video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            k = rope_action_apply(k, video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)

            video_kv_per_layer.append((k, v))
            if kv_cache is not None:
                attn_k = torch.cat([kv_cache[0], k], dim=1)
                attn_v = torch.cat([kv_cache[1], v], dim=1)
            else:
                attn_k, attn_v = k, v
            from wan.modules.attention import flash_attention
            y = flash_attention(q, attn_k, attn_v, k_lens=None).flatten(2)
            y = blk.self_attn.o(y)
            x = x + y * mod[2].squeeze(2)
            if t5_ctx is not None:
                x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
            ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
            x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return video_kv_per_layer

    def _forward_action_only_cached(
        self,
        action_tokens: torch.Tensor,
        t_act: torch.Tensor,
        frozen_video_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        t5_ctx,
        K: int,
    ) -> torch.Tensor:
        """Action-only forward using frozen current-chunk video K/V."""
        B, a_seq_len, _ = action_tokens.shape
        n = self.video_model.wan_model.num_heads
        d = self.wan_dim // n
        a_t_vals = torch.full((B, a_seq_len), t_act.item(), device=self.device)
        _, a_e0 = self._video_time_embed(a_t_vals)
        infer_latent_frames = int(K)
        x = action_tokens
        action_override = self._maybe_action_video_basis_override(
            a_seq_len - self.config.state_tokens_per_frame,
            self.current_start_frame,
        )

        for layer_idx in range(self.config.num_layers):
            blk = self.video_model.wan_model.blocks[layer_idx]
            mod = (blk.modulation.unsqueeze(0) + a_e0).chunk(6, dim=2)
            norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
            q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, a_seq_len, n, d)
            k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, a_seq_len, n, d)
            v = blk.self_attn.v(norm_x).view(B, a_seq_len, n, d)

            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
            action_only_freqs = self.freqs_action.new_empty((0, 1, self.freqs_action.shape[-1]))
            q = causal_rope_action_apply(
                q,
                action_only_freqs,
                self.freqs_action,
                self.freqs_state,
                action_register_length=a_seq_len,
                num_action_per_block=infer_latent_frames * self.config.action_tokens_per_frame,
                num_state_per_block=self.config.state_tokens_per_frame,
                num_value_per_block=0,
                action_freqs_override=action_override,
            ).type_as(v)
            k = causal_rope_action_apply(
                k,
                action_only_freqs,
                self.freqs_action,
                self.freqs_state,
                action_register_length=a_seq_len,
                num_action_per_block=infer_latent_frames * self.config.action_tokens_per_frame,
                num_state_per_block=self.config.state_tokens_per_frame,
                num_value_per_block=0,
                action_freqs_override=action_override,
            ).type_as(v)

            frozen_k, frozen_v = frozen_video_kv[layer_idx]
            if kv_cache is not None:
                attn_k = torch.cat([kv_cache[0], frozen_k, k], dim=1)
                attn_v = torch.cat([kv_cache[1], frozen_v, v], dim=1)
            else:
                attn_k = torch.cat([frozen_k, k], dim=1)
                attn_v = torch.cat([frozen_v, v], dim=1)
            from wan.modules.attention import flash_attention
            y = flash_attention(q, attn_k, attn_v, k_lens=None).flatten(2)
            y = blk.self_attn.o(y)
            x = x + y * mod[2].squeeze(2)
            if t5_ctx is not None:
                x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
            ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
            x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x

    # ------------------------------------------------------------------
    # Action-first streaming inference (current chunk over hybrid KV cache)
    # ------------------------------------------------------------------
    def _build_infer_chunk_freqs(self, noisy_video_freqs: torch.Tensor, K: int) -> torch.Tensor:
        """Per-token RoPE for the action-first current chunk.

        Layout [state?, action, noisy_video]: state gets identity, action follows
        ``action_rope_mode``, and noisy_video gets global 3D video freqs.
        Mirrors the per-chunk portion of :meth:`_build_memory_full_freqs`.
        """
        self._ensure_freqs_device()
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        C = self.freqs_state.shape[-1]
        dt = self.freqs_state.dtype
        parts = []
        if spf > 0:
            parts.append(torch.ones((spf, 1, C), device=self.device, dtype=dt))
        a_len = int(K) * apf
        if a_len > 0:
            action_override = self._maybe_action_video_basis_override(
                a_len,
                self.current_start_frame,
            )
            parts.append(
                action_override
                if action_override is not None
                else self.freqs_action[:a_len].view(a_len, 1, -1)
            )
        parts.append(noisy_video_freqs)
        return torch.cat(parts, dim=0)

    def _build_infer_chunk_mask(self, K: int, cache_len: int, device) -> torch.Tensor:
        """FDM visibility for the action-first current chunk over [cache | chunk].

        The retained hybrid cache (anchor + recent full + all gist, all strictly
        past frames) is fully visible. Intra-chunk visibility reproduces
        :func:`build_memory_visibility`: under action_only/fdm, action queries
        see only chunk action(+state), noisy_video queries see action+video.
        Returns a bool mask [Lq, cache_len + Lq] (True = attend).
        """
        spf = self.config.state_tokens_per_frame if self.config.use_state else 0
        apf = self.config.action_tokens_per_frame
        vbs = self.frame_seqlen
        La = spf + int(K) * apf
        Lv = int(K) * vbs
        Lq = La + Lv
        mode = str(self.config.mode or "joint").lower()
        chunk = torch.zeros(Lq, Lq, dtype=torch.bool, device=device)
        is_action = torch.zeros(Lq, dtype=torch.bool, device=device)
        is_action[:La] = True
        is_video = ~is_action
        if mode in ("action_only", "fdm"):
            chunk[:La] = is_action.view(1, -1)   # action q -> action(+state) only
            chunk[La:] = True                    # video q -> action + video
        elif mode == "video_idm":
            chunk[:La] = True                    # action q -> action + video
            chunk[La:] = is_video.view(1, -1)    # video q -> video only
        else:  # joint
            chunk[:] = True
        chunk |= torch.eye(Lq, dtype=torch.bool, device=device)
        if int(cache_len) > 0:
            cache_part = torch.ones(Lq, int(cache_len), dtype=torch.bool, device=device)
            return torch.cat([cache_part, chunk], dim=1)
        return chunk

    def _forward_layer_memory_infer(self, x_chunk, e0, layer_idx, t5_ctx,
                                    chunk_freqs, kv_cache, attn_mask):
        """One WAN layer for action-first streaming inference (current chunk).

        Current chunk x = [state?, action, noisy_video] attends to
        [hybrid KV cache | current chunk] via SDPA with ``attn_mask``. Mirrors
        :meth:`_forward_layer_memory` (training) but reads history from the cache
        instead of in-sequence clean/gist tokens, so train/infer stay aligned.
        """
        blk = self.video_model.wan_model.blocks[layer_idx]
        mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        norm_x = (blk.norm1(x_chunk) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
        B, L, H_dim = norm_x.shape
        n = self.video_model.wan_model.num_heads
        d = H_dim // n
        q = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, L, n, d)
        k = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, L, n, d)
        v = blk.self_attn.v(norm_x).view(B, L, n, d)
        q = rope_action_apply(q, chunk_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
        k = rope_action_apply(k, chunk_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
        if kv_cache is not None and kv_cache[0].shape[1] > 0:
            attn_k = torch.cat([kv_cache[0], k], dim=1)
            attn_v = torch.cat([kv_cache[1], v], dim=1)
        else:
            attn_k, attn_v = k, v
        # SDPA expects [B, n, L, d]; attn_mask broadcasts over [B, n].
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), attn_k.transpose(1, 2), attn_v.transpose(1, 2),
            attn_mask=attn_mask,
        )
        y = y.transpose(1, 2).reshape(B, L, H_dim)
        y = blk.self_attn.o(y)
        x_chunk = x_chunk + y * mod[2].squeeze(2)
        if t5_ctx is not None:
            x_chunk = x_chunk + blk.cross_attn(blk.norm3(x_chunk), t5_ctx, context_lens=None)
        ffn_in = (blk.norm2(x_chunk) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
        x_chunk = x_chunk + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x_chunk

    @torch.no_grad()
    def inference_step_streaming(
        self,
        state: torch.Tensor = None,
        language_embeddings=None,
        num_inference_steps: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next chunk Z_k, Y_k using KV cache.

        Returns:
            video_latent: [B, C, K, H_lat, W_lat]
            action_pred: [B, action_chunk_size, action_dim]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_timesteps

        K = self._fixed_chunk_size()
        B = self.kv_caches[0].shape[1]

        # Initialize noise
        C_lat = self.video_model.wan_model.in_dim
        z = torch.randn(B, C_lat, K, self.lat_H * 2, self.lat_W * 2, device=self.device, dtype=self.dtype)
        action_chunk_len = K * self.config.actions_per_latent
        y = torch.randn(B, action_chunk_len, self.config.action_dim, device=self.device, dtype=self.dtype)

        # Two separate schedulers (decouple inference)
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched_action = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched_video.set_timesteps(
            num_inference_steps,
            device=self.device,
            shift=float(self.config.video_timestep_shift),
        )
        sched_action.set_timesteps(
            num_inference_steps,
            device=self.device,
            shift=float(self.config.action_timestep_shift),
        )

        # Decouple: rescale video schedule so it ends at video_inference_final_noise instead of 0
        if self.config.decouple_inference_noise:
            sigma_max = sched_video.sigmas[0].item()
            final_noise = self.config.video_inference_final_noise
            sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
            sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)

        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None
        total_steps = len(sched_video.timesteps)
        joint_steps = total_steps
        if self.config.video_steps is not None:
            joint_steps = max(0, min(int(self.config.video_steps), total_steps))
        use_action_only_phase = bool(self.config.idm_mode and joint_steps < total_steps)
        frozen_z = None
        frozen_video_kv = None

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for step_idx in range(total_steps):
                t_vid = sched_video.timesteps[step_idx]
                t_act = sched_action.timesteps[step_idx]
                action_only_phase = use_action_only_phase and (step_idx >= joint_steps)
                if action_only_phase and frozen_z is None:
                    frozen_z = z.detach()
                z_input = frozen_z if action_only_phase else z

                # Prepare video tokens
                z_tokens = self._video_patch_embed(z_input)  # [B, K*frame_seqlen, wan_dim]
                v_seq_len = z_tokens.shape[1]

                # Prepare action tokens
                if state is not None and self.config.use_state:
                    state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                    a_tokens = self.action_expert.input_encoder(state_chunk, y)
                else:
                    a_tokens = self.action_expert.input_encoder(None, y)

                a_tokens = self.action_expert.projection.project_in(a_tokens)
                a_seq_len = a_tokens.shape[1]

                # Time embeddings: video and action use DIFFERENT timesteps
                v_t_vals = torch.full((B, v_seq_len), t_vid.item(), device=self.device)
                a_t_vals = torch.full((B, a_seq_len), t_act.item(), device=self.device)
                # action-first layer order: [state?, action | noisy_video]
                combined_t_vals = torch.cat([a_t_vals, v_t_vals], dim=1)
                _, e0_combined = self._video_time_embed(combined_t_vals)
                v_emb, _ = self._video_time_embed(v_t_vals)
                a_emb, _ = self._action_time_embed(a_t_vals)

                # Forward through layers
                self._ensure_freqs_device()

                if action_only_phase:
                    if frozen_video_kv is None:
                        frozen_video_kv = self._build_frozen_video_layer_kv(
                            z_tokens=z_tokens,
                            t_vid=t_vid,
                            t5_ctx=t5_ctx,
                        )
                    act_hidden = self._forward_action_only_cached(
                        action_tokens=a_tokens,
                        t_act=t_act,
                        frozen_video_kv=frozen_video_kv,
                        t5_ctx=t5_ctx,
                        K=K,
                    )
                    vid_out = None
                    act_out = self.action_expert.projection.project_out(act_hidden)
                else:
                    # Action-first current chunk: [state?, action, noisy_video]
                    # attends to [hybrid KV cache | current chunk] with FDM.
                    block_latent_counts = [K]
                    x_chunk = self._interleave_memory_chunk_ar_tokens(
                        a_tokens, z_tokens, block_latent_counts,
                    )
                    noisy_video_freqs = self._create_video_freqs(
                        K, start_frame=self.current_start_frame,
                    )
                    chunk_freqs = self._build_infer_chunk_freqs(noisy_video_freqs, K)
                    cache_len = self.kv_caches[0][0].shape[1] if self.kv_caches else 0
                    attn_mask = self._build_infer_chunk_mask(K, cache_len, x_chunk.device)

                    for layer_idx in range(self.config.num_layers):
                        kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                        x_chunk = self._forward_layer_memory_infer(
                            x_chunk, e0_combined, layer_idx, t5_ctx,
                            chunk_freqs, kv_cache, attn_mask,
                        )

                    sa_hidden, noisy_video_hidden = self._split_memory_chunk_ar_hidden(
                        x_chunk, block_latent_counts,
                    )
                    vid_out = noisy_video_hidden
                    act_out = self.action_expert.projection.project_out(sa_hidden)

                # Predict velocities
                v_pred_unpatched = None
                if not action_only_phase:
                    v_pred = self.video_model.wan_model.head(vid_out, v_emb)
                    grid = torch.tensor([[K, self.lat_H, self.lat_W]], dtype=torch.long).expand(B, -1)
                    v_pred_unpatched = self.video_model.wan_model.unpatchify(v_pred, grid)
                    if isinstance(v_pred_unpatched, list):
                        v_pred_unpatched = torch.stack(v_pred_unpatched)

                a_pred = self.action_expert.decoder(act_out, a_emb)
                # Extract action only (skip state token when use_state)
                if self.config.use_state:
                    a_pred = a_pred[:, 1:, :]  # [B, tokens_per_chunk, dim] -> [B, action_chunk_size, dim]

                # Scheduler step: video and action use their respective schedulers
                if not action_only_phase:
                    z = sched_video.step(v_pred_unpatched.transpose(1, 2), t_vid, z.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
                y = sched_action.step(a_pred, t_act, y, return_dict=False)[0]

        self._last_infer_K = K
        return z, y

    def forward_action_velocity_from_latent(
        self,
        video_latent: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma_action: torch.Tensor,
        state: torch.Tensor = None,
        language_embeddings=None,
        restore_inference_state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Predict action velocity for a noisy action chunk conditioned on a video latent.

        This is the action-only streaming forward used by MBRL/NFT training.
        The current chunk video latent is treated as a clean condition (t=0), while
        the action branch receives the provided action noise level.
        """
        if self.kv_caches is None and restore_inference_state is None:
            raise RuntimeError("forward_action_velocity_from_latent requires a pre-filled KV cache")

        saved_state = None
        if restore_inference_state is not None:
            saved_state = self.export_inference_state()
            self.load_inference_state(restore_inference_state)

        try:
            K = int(video_latent.shape[2])
            B = int(video_latent.shape[0])
            sigma_action = sigma_action.to(device=self.device, dtype=self.dtype).reshape(-1)
            if sigma_action.numel() == 1 and B > 1:
                sigma_action = sigma_action.expand(B)
            if sigma_action.shape[0] != B:
                raise ValueError(f"sigma_action batch mismatch: got {sigma_action.shape[0]}, expected {B}")

            t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                z_tokens = self._video_patch_embed(video_latent.to(self.dtype))
                v_seq_len = z_tokens.shape[1]

                if state is not None and self.config.use_state:
                    state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                    a_tokens = self.action_expert.input_encoder(state_chunk.to(self.dtype), noisy_action.to(self.dtype))
                else:
                    a_tokens = self.action_expert.input_encoder(None, noisy_action.to(self.dtype))

                a_tokens = self.action_expert.projection.project_in(a_tokens)
                a_seq_len = a_tokens.shape[1]

                action_timestep = (
                    sigma_action * self.action_scheduler.num_train_timesteps
                ).to(self.dtype)

                v_t_vals = torch.zeros(B, v_seq_len, device=self.device, dtype=self.dtype)
                a_t_vals = action_timestep.view(B, 1).expand(-1, a_seq_len)
                combined_t_vals = torch.cat([v_t_vals, a_t_vals], dim=1)
                _, e0_combined = self._video_time_embed(combined_t_vals)
                a_emb, _ = self._action_time_embed(a_t_vals)

                x = torch.cat([z_tokens, a_tokens], dim=1)
                action_register_length = a_seq_len if a_seq_len > 0 else None
                for layer_idx in range(self.config.num_layers):
                    kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                    x = self._forward_layer_simple(
                        x,
                        e0_combined,
                        layer_idx,
                        t5_ctx,
                        ar_mask=None,
                        kv_cache=kv_cache,
                        v_seq_len=v_seq_len,
                        is_training=False,
                        action_register_length=action_register_length,
                    )
                act_out = self.action_expert.projection.project_out(x[:, v_seq_len:])

                action_pred = self.action_expert.decoder(act_out, a_emb)
                if self.config.use_state:
                    action_pred = action_pred[:, 1:, :]

            self._last_infer_K = K
            return action_pred
        finally:
            if saved_state is not None:
                self.load_inference_state(saved_state)

    @torch.no_grad()
    def inference_value_streaming(
        self,
        video_latent: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        language_embeddings=None,
    ) -> torch.Tensor:
        """Predict chunk-level value for the current chunk using learnable value queries.

        Runs trunk forward (video + state + action, no value tokens), then at each
        layer does a separate value attention where value Q attends to:
          - clean KV cache (history)
          - full current chunk trunk KV
          - current chunk value KV

        Args:
            video_latent: [B, C, K, H_lat, W_lat] denoised video latent for current chunk
            actions: [B, K*apl, action_dim] predicted/gt actions for current chunk
            state: [B, K, state_dim] or [B, state_dim] states (optional)
            language_embeddings: T5 embeddings for cross-attention

        Returns:
            value_pred: [B, 1] predicted value for the current chunk
        """
        if not self.config.value_enabled or self.config.value_tokens_per_frame == 0:
            return None

        from wan.modules.attention import flash_attention
        from models.value_head import expected_value

        K = self._last_infer_K
        nvpf = self.config.value_tokens_per_frame
        B = video_latent.shape[0]
        apl = self.config.actions_per_latent

        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # === Prepare trunk tokens (same as inference_step_streaming but with t=0) ===
            z_tokens = self._video_patch_embed(video_latent)
            v_seq_len = z_tokens.shape[1]

            if state is not None and self.config.use_state:
                state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                a_tokens = self.action_expert.input_encoder(state_chunk, actions)
            else:
                a_tokens = self.action_expert.input_encoder(None, actions)

            a_tokens = self.action_expert.projection.project_in(a_tokens)
            a_seq_len = a_tokens.shape[1]

            # Chunk-level value query tokens: one group per current chunk.
            val_tokens = self.value_tokens.expand(B, 1)  # [B, nvpf, wan_dim]

            # Time embeddings: t=0 for all (value readout from clean/denoised context)
            self._ensure_freqs_device()
            v_t_vals = torch.zeros(B, v_seq_len, device=self.device)
            a_t_vals = torch.zeros(B, a_seq_len, device=self.device)
            combined_t_vals = torch.cat([v_t_vals, a_t_vals], dim=1)
            _, e0_combined = self._video_time_embed(combined_t_vals)

            # Value AdaLN: t=0
            val_t_vals = torch.zeros(B, nvpf, device=self.device)
            _, val_e0 = self._video_time_embed(val_t_vals)

            x = torch.cat([z_tokens, a_tokens], dim=1)
            action_register_length = a_seq_len if a_seq_len > 0 else None

            # Tokens per physical frame (video + state + action)
            vbs = self.frame_seqlen  # video tokens per frame
            spf = self.config.state_tokens_per_frame if self.config.use_state else 0
            apf = self.config.action_tokens_per_frame
            for layer_idx in range(self.config.num_layers):
                blk = self.video_model.wan_model.blocks[layer_idx]

                # --- Trunk forward (same as _forward_layer_simple inference) ---
                mod = (blk.modulation.unsqueeze(0) + e0_combined).chunk(6, dim=2)
                norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
                n = self.video_model.wan_model.num_heads
                d = self.wan_dim // n
                q_trunk = blk.self_attn.norm_q(blk.self_attn.q(norm_x)).view(B, -1, n, d)
                k_trunk = blk.self_attn.norm_k(blk.self_attn.k(norm_x)).view(B, -1, n, d)
                v_trunk = blk.self_attn.v(norm_x).view(B, -1, n, d)

                # RoPE for trunk
                cache_seq_len = self.kv_caches[layer_idx][0].shape[1] if self.kv_caches is not None else 0
                inf_video_freqs = self._create_video_freqs(
                    K,
                    start_frame=self._inference_video_start_frame(cache_seq_len),
                )
                if action_register_length is not None and a_seq_len > 0:
                    napb = K * self.config.action_tokens_per_frame
                    nspb = self.config.state_tokens_per_frame
                    q_trunk = causal_rope_action_apply(
                        q_trunk, inf_video_freqs, self.freqs_action, self.freqs_state,
                        action_register_length, napb, nspb,
                    ).type_as(v_trunk)
                    k_trunk = causal_rope_action_apply(
                        k_trunk, inf_video_freqs, self.freqs_action, self.freqs_state,
                        action_register_length, napb, nspb,
                    ).type_as(v_trunk)
                else:
                    q_trunk = rope_action_apply(
                        q_trunk, inf_video_freqs, self.freqs_action, self.freqs_state, None
                    ).type_as(v_trunk)
                    k_trunk = rope_action_apply(
                        k_trunk, inf_video_freqs, self.freqs_action, self.freqs_state, None
                    ).type_as(v_trunk)

                # Trunk attention with KV cache
                kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                if kv_cache is not None:
                    cached_k, cached_v = kv_cache
                    attn_k_trunk = torch.cat([cached_k, k_trunk], dim=1)
                    attn_v_trunk = torch.cat([cached_v, v_trunk], dim=1)
                else:
                    attn_k_trunk, attn_v_trunk = k_trunk, v_trunk
                y_trunk = flash_attention(q_trunk, attn_k_trunk, attn_v_trunk, k_lens=None).flatten(2)

                y_trunk = blk.self_attn.o(y_trunk)
                x = x + y_trunk * mod[2].squeeze(2)
                if t5_ctx is not None:
                    x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
                ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)

                # --- Value forward (separate chunk-level attention) ---
                val_mod = (blk.modulation.unsqueeze(0) + val_e0).chunk(6, dim=2)
                norm_val = (blk.norm1(val_tokens) * (1 + val_mod[1].squeeze(2)) + val_mod[0].squeeze(2)).to(self.dtype)
                q_val = blk.self_attn.norm_q(blk.self_attn.q(norm_val)).view(B, nvpf, n, d)
                k_val = blk.self_attn.norm_k(blk.self_attn.k(norm_val)).view(B, nvpf, n, d)
                v_val = blk.self_attn.v(norm_val).view(B, nvpf, n, d)

                # Chunk-level attention: value sees clean cache + full current chunk trunk + own value KV.
                if kv_cache is not None:
                    attn_k_val = torch.cat([cached_k, k_trunk, k_val], dim=1)
                    attn_v_val = torch.cat([cached_v, v_trunk, v_val], dim=1)
                else:
                    attn_k_val = torch.cat([k_trunk, k_val], dim=1)
                    attn_v_val = torch.cat([v_trunk, v_val], dim=1)

                y_val = flash_attention(q_val, attn_k_val, attn_v_val, k_lens=None)
                y_val = y_val.reshape(B, nvpf, -1)

                y_val = blk.self_attn.o(y_val)
                val_tokens = val_tokens + y_val * val_mod[2].squeeze(2)
                # Value tokens skip cross-attention
                ffn_val_in = (blk.norm2(val_tokens) * (1 + val_mod[4].squeeze(2)) + val_mod[3].squeeze(2)).to(self.dtype)
                val_tokens = val_tokens + blk.ffn(ffn_val_in) * val_mod[5].squeeze(2)

            # Value prediction: current chunk only.
            val_hidden = val_tokens.reshape(B, 1, nvpf, -1)
            val_logits = self.value_head(val_hidden.to(self.dtype).reshape(B * nvpf, -1))
            val_logits = val_logits.reshape(B, 1, nvpf, -1).mean(dim=2)
            value_pred = expected_value(
                val_logits, self.config.value_num_bins,
                self.config.value_range[0], self.config.value_range[1]
            )  # [B, 1]

        return value_pred
