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

from data.sample_class import SampleClass
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .rope import rope_params, rope_action_apply, causal_rope_action_apply
from .block_attention import (
    create_sliding_window_mask,
    BLHD_flex_with_mask,
)
from utils.action_mode import (
    is_rel_hand_qpos_mode,
    qpos_family_action_dim,
    qpos_family_hand_dim,
    qpos_family_layout,
)
from utils.action_temporal_loss import (
    masked_chunk_second_difference_mse,
    validate_action_d2_loss_weight,
)
from utils.relative_eef import (
    abs_eef_20d_to_relative_eef_20d_torch,
    abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch,
    abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch,
    abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
    abs_qpos20_wrist9_to_relqpos20_rel6d_torch,
)
from utils.value_cross_attention import apply_value_cross_attention
from utils.training_qk_rope import (
    prepare_training_qk_frequencies,
    try_training_qkv_with_rope,
)

logger = logging.getLogger(__name__)


def _configure_training_compile(config):
    """Allow the finite grad/no-grad and VM sub-batch specializations.

    The static flex callable shares a Dynamo cache across training and the
    on-model rollout. Its default eight entries are insufficient when the
    sampled window and VM sub-batch sizes vary. Keep fullgraph and all guards;
    enlarge the cache for the supported training shapes.
    """
    dynamo_config = torch._dynamo.config
    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
    logger.info(
        "static FlexAttention: recompile_limit=%d",
        dynamo_config.recompile_limit,
    )


SAMPLE_CLASS_SUBOPTIMAL = SampleClass.SUBOPTIMAL


# Action modes whose loss is split into position / rotation / gripper-or-hand
# components (and therefore share the uniform-weight warmup).
_EEF_COMPONENT_MODES = frozenset({
    "rel_eef", "rel_eef_6d", "abs_eef_rpy", "abs_eef", "abs_eef_wxyz", "rel_eef_wxyz",
})


def _is_eef_component_mode(action_mode: str) -> bool:
    return (
        action_mode in _EEF_COMPONENT_MODES
        or qpos_family_hand_dim(action_mode) is not None
    )

# === NVTX profiling helpers (env-gated, zero overhead when MOTUS2_NVTX!=1) ===
_MOTUS2_NVTX = bool(int(os.environ.get("MOTUS2_NVTX", "0")))
def _nvpush(name):
    if _MOTUS2_NVTX:
        torch.cuda.nvtx.range_push(name)
def _nvpop():
    if _MOTUS2_NVTX:
        torch.cuda.nvtx.range_pop()


# ============================================================================
# t1v11a: qkv_fuse (math-equivalent QKV Linear weight merge for inference).
# Active when harness calls install_wan_qkv_fuse(model) after load.
# Without install, _wan_self_qkv falls back to the 3-Linear path -> bit-exact.
# ============================================================================
@torch.no_grad()
def install_wan_qkv_fuse(model) -> int:
    """Concatenate Q/K/V Linear weights per block (math-equivalent, bit-exact)."""
    blocks = model.video_model.wan_model.blocks
    n_installed = 0
    for block in blocks:
        sa = block.self_attn
        if not all(hasattr(sa, x) for x in ("q", "k", "v")):
            continue
        if sa.q.weight.shape != sa.k.weight.shape or sa.q.weight.shape != sa.v.weight.shape:
            continue
        sa._motus2_qkv_fused_weight = torch.cat(
            [sa.q.weight.detach(), sa.k.weight.detach(), sa.v.weight.detach()], dim=0
        ).contiguous()
        has_bias = any(L.bias is not None for L in (sa.q, sa.k, sa.v))
        if has_bias:
            biases = []
            for L in (sa.q, sa.k, sa.v):
                biases.append(L.bias.detach() if L.bias is not None
                              else torch.zeros(L.out_features, dtype=L.weight.dtype, device=L.weight.device))
            sa._motus2_qkv_fused_bias = torch.cat(biases, dim=0).contiguous()
        else:
            sa._motus2_qkv_fused_bias = None
        sa._motus2_qkv_fused = True
        n_installed += 1
    logger.info("[motus2.accel] qkv_fuse installed on %d blocks", n_installed)
    return n_installed


def _wan_self_qkv(self_attn, norm_x, B: int, L: int, n: int, d: int):
    """Compute (q, k, v) each [B, L, n, d]; uses fused weight if installed.

    F-S7: when both QKV is fused (t1v11a) AND FlashVLA fvk is available, use
    qkv_split_norm_rope_bf16 (seq_len=0 → RoPE disabled) to fuse chunk + RMSNorm.
    Bitwise-equivalent to the manual path (BF16 rounding). Compile-safe: under
    torch.compile, falls back to eager chunk + norm.
    """
    if getattr(self_attn, "_motus2_qkv_fused", False):
        qkv = F.linear(norm_x, self_attn._motus2_qkv_fused_weight, self_attn._motus2_qkv_fused_bias)
        # F-S7 fast path
        if not torch.compiler.is_compiling():
            try:
                from accel.wan_qkv_split_norm_fuse import try_fused_split_norm
                fused = try_fused_split_norm(qkv, self_attn.norm_q, self_attn.norm_k, B, L, n, d)
                if fused is not None:
                    return fused
            except (ImportError, RuntimeError):
                pass
        q_raw, k_raw, v_raw = (t.contiguous() for t in qkv.chunk(3, dim=-1))
    else:
        q_raw = self_attn.q(norm_x)
        k_raw = self_attn.k(norm_x)
        v_raw = self_attn.v(norm_x)
    q = self_attn.norm_q(q_raw).view(B, L, n, d)
    k = self_attn.norm_k(k_raw).view(B, L, n, d)
    v = v_raw.view(B, L, n, d)
    return q, k, v


# ============================================================================
# t1v11b: bf16_norm (WanRMSNorm BF16-only, no FP32 upcast).
# Monkey-patches WanRMSNorm.forward globally; ULP-level vs original.
# ============================================================================
def install_wan_bf16_norm() -> int:
    """Replace WanRMSNorm.forward with a BF16-only variant. ULP-level deviation."""
    from wan.modules.model import WanRMSNorm
    if getattr(WanRMSNorm, "_motus2_bf16_patched", False):
        return 0

    def _bf16_forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

    WanRMSNorm.forward = _bf16_forward
    WanRMSNorm._motus2_bf16_patched = True
    logger.info("[motus2.accel] bf16_norm installed (WanRMSNorm.forward replaced)")
    return 1


# ============================================================================
# t1v11l: Euler scheduler (1阶 ODE solver, alternative to UniPC).
# CRITICAL: use affine rescale for final_sigma (NOT direct sigmas[-1] overwrite,
# which would cause final step to go backward and yield zeros).
# ============================================================================
class _FlowEulerScheduler:
    """Graph-friendly 1-order Euler sampler for flow-prediction inference."""

    def __init__(self, num_inference_steps: int, *, device, shift: float = 5.0,
                 num_train_timesteps: int = 1000, final_sigma: float = 0.0) -> None:
        base = torch.linspace(
            (num_train_timesteps - 1) / num_train_timesteps, 0.0,
            int(num_inference_steps) + 1,
            device=device, dtype=torch.float32,
        )
        sigmas = shift * base / (1 + (shift - 1) * base)
        sigmas[-1] = float(final_sigma)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(torch.int64)
        self.num_inference_steps = int(num_inference_steps)
        self._step_index = 0

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(self, model_output, timestep, sample, return_dict: bool = True):
        del timestep
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=sample.dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=sample.dtype)
        prev_sample = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return {"prev_sample": prev_sample}


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


def wm_language_row_gate(use_wm: torch.Tensor, t5_ctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """[B, 1, 1] multiplier on the cross-attention output: 0 on WM rows, 1 elsewhere."""
    if t5_ctx is None:
        return None
    return (~use_wm.bool()).to(dtype=t5_ctx.dtype).view(-1, 1, 1)


def gated_cross_attn(blk, x: torch.Tensor, t5_ctx: torch.Tensor, row_gate: Optional[torch.Tensor]) -> torch.Tensor:
    cross = blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
    return cross if row_gate is None else cross * row_gate

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
    num_views: int = 1

    # Sliding-window AR params
    video_action_freq_ratio: int = 6   # action 30Hz, video 5Hz
    action_upsample_ratio: int = 1     # 1 = no upsampling, 4 = interpolate to 120Hz
    max_latent_window: int = 10        # W: training window in latent frames
    fixed_chunk_size: int = 2          # fixed latent chunk size for training and inference
    chunk_window_range: tuple = (1, 4) # random visible-chunk window per iteration
    chunk_window_size: int = 4         # inference/cache visible chunk window
    video_rope_mode: str = "global"    # "global" or "window" (video only)
    video_view_rope_mode: str = "shared"  # "shared" or "hw_offset" for multiview 3D RoPE

    # Action expert
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 3072
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-6

    # AR modes
    use_mot: bool = False
    use_state: bool = False
    state_mask_ratio: float = 0.0
    idm_mode: bool = False
    action_mode: str = "joint"        # "joint", "rel_joint", "rel_eef"(16D xyzw), "rel_eef_6d"(20D rot6d), or "abs_eef_rpy"(14D)
    mode: str = "joint"               # "joint", "video_idm", "action_only", or "loop"
    # Attention kernel used by _fixed_cache_sdpa. "sdpa" keeps the
    # F.scaled_dot_product_attention path (compile-friendly via fixed-shape
    # cache + boolean mask, but the backend dispatch under bf16 + mask drifts
    # ULP-per-layer vs WAN's flash_attention). "flash_varlen" routes through
    # flash_attn.flash_attn_varlen_func over a compacted prefix, matching the
    # Wan flash_attention path while
    # remaining compile-traceable under dynamic shape support.
    attention_backend: str = "sdpa"   # "sdpa", "flash_varlen", or "flex"

    # Dual-t sampling
    policy_timestep_method: str = "beta"  # "beta" or "logit_normal"
    beta_alpha: float = 7.0
    beta_beta: float = 1.0
    policy_sigmoid_scale: float = 1.0
    policy_min_t: float = 0.0
    policy_max_t: float = 1.0
    video_timestep_shift: float = 5.0
    action_timestep_shift: float = 1.0
    action_timestep_method: str = "logit_normal"  # "beta" or "logit_normal"
    action_sigmoid_scale: float = 1.0
    action_min_t: float = 0.0
    action_max_t: float = 1.0
    wm_ratio: float = 0.2
    # WM rows (clean action -> video) train without T5 cross-attention on every
    # token of the row, history included, so the instruction cannot stand in for
    # the action. Policy and VM rows keep language. Inference must match: WM
    # rollouts pass language_embeddings=None and read a history cache that was
    # filled without language.
    wm_mask_language: bool = False
    # Training
    batch_size: int = 8
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    action_d2_loss_weight: float = 0.0
    warmup_steps: int = 0
    uniform_loss_weight_warmup_steps: int = 0  # Warm component weights to 1/1/1 for first N steps.
    # First step at which fail/random samples reach the loss; None keeps the
    # legacy gate max(warmup_steps, uniform_loss_weight_warmup_steps).
    off_policy_start_step: Optional[int] = None
    rel_eef_position_loss_weight: float = 1.0
    rel_eef_rotation_loss_weight: float = 1.0
    rel_eef_gripper_loss_weight: float = 1.0  # Legacy EEF gripper modes only.
    hand_qpos_loss_weight: float = 1.0  # qpos{N}_rel6d hand qpos component.
    # Prefix-conditioned RTC (the `train` family of arXiv 2608.01880): under
    # asynchronous deployment the first rtc_committed_steps actions of the chunk
    # being generated were already emitted by the previous chunk and cannot be
    # revised. Training reproduces that by supplying them clean (timestep 0) and
    # leaving them unsupervised, so the model learns to continue from a
    # commitment instead of re-deciding it.
    rtc_training_enabled: bool = False
    rtc_committed_steps: int = 0       # d_est: committed prefix length, in actions.
    rtc_replan_action_index: int = 0   # s: deployment replans after this action.
    rtc_training_prob: float = 0.5     # Fraction of samples that get a prefix.
    # Only committed_steps enters the loss; replan_action_index records the
    # deployment schedule and bounds it. See __post_init__ for the relation.
    # Wrist relative-eef xyz normalization (orthogonal to action_mode layout).
    wrist_xyz_normalization: str = "min_max"  # mean_std | min_max [0,1] | quantile
    qpos20_wrist_xyz_min: tuple = (-0.05, -0.05, -0.05, -0.05, -0.05, -0.05)
    qpos20_wrist_xyz_max: tuple = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    qpos20_wrist_xyz_mean: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    qpos20_wrist_xyz_std: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    wrist_xyz_std_min: float = 1e-4

    # Noisy history augmentation: per-frame independent noise on clean conditioning latents
    noisy_history_prob: float = 0.5       # probability of adding noise to each clean frame
    noisy_history_sigma_max: float = 0.1  # max sigma for augmentation, Uniform(0, sigma_max)

    # Inference
    num_inference_timesteps: int = 10
    num_action_inference_timesteps: Optional[int] = None
    video_steps: Optional[int] = None  # joint denoise steps before action-only IDM phase
    # Decoupled inference: video stays noisy, action fully denoises
    decouple_inference_noise: bool = False
    video_inference_final_noise: float = 0.0
    qkv_fuse: bool = False           # t1v11a: math-equivalent QKV Linear weight merge
    bf16_norm: bool = False          # t1v11b: WanRMSNorm BF16-only (no FP32 upcast)
    t5_kv_caching: bool = False      # t1v11c: T5 cross-attn K/V cache (bit-exact)
    cache_inference_time_tensors: bool = False  # t1v11d: memoize per-step time embeddings
    cache_inference_rope_tensors: bool = False  # t1v11e: memoize video RoPE freqs grid
    prewarm_inference_caches: bool = False  # t1v11f: prewarm time/rope caches after KV prefill
    inference_scheduler: str = "unipc"      # t1v11l: "unipc" (default high-order) or "euler" (1-order)

    # Value prediction
    value_enabled: bool = False
    value_tokens_per_frame: int = 0       # K value query tokens per chunk (0 = disabled)
    value_num_bins: int = 201
    # Value targets and decoding in float32. bf16 targets land in a neighbouring
    # bin for ~20% of values and bf16 bin centers are off by up to 0.003.
    value_fp32: bool = False
    value_range: tuple = (0.0, 1.0)
    value_target: str = "progress"        # "progress", "ttc", or "vlac"
    value_loss_weight: float = 1.0
    vm_ratio: float = 0.0                 # Value mode ratio (0 = disabled)
    progress_c: float = 0.01              # Adaptive gamma parameter for progress targets
    # On-model value: score the WM's own rollout instead of the clean ground truth.
    #
    # VM samples run at sigma_v=0, so the value head is trained on real future
    # frames, while MBRL scores latents the WM generated. Those differ by an L1 of
    # 7.8-9.4 against the real frame, and the candidates the value must separate
    # differ from each other by 0.3-1.2 -- the scoring input sits 7-30x further
    # out of distribution than the difference being resolved. Dreamer and MuZero
    # both train the critic on generated states for exactly this reason.
    # Default False keeps every existing run's bytes unchanged.
    value_on_model: bool = False
    value_temporal_sampling: str = "none"
    # Denoise steps for that rollout. The count lives in the MBRL config, not
    # this one, and three different keys claim the name:
    #   model.inference.video_steps (2)   only bounds video_idm; never on this path
    #   mbrl.video_inference_steps (8)    the production joint planner
    #   mbrl.wm_num_inference_steps (10)  wm_rollout_step, which produced E41/E43
    # The default follows the last one: that rollout is what measures whether
    # the real action beats the degenerate candidates, and it already generates
    # the way this does -- euler, chunk by chunk, generated history fed forward,
    # conditioned on a clean action. Override with value.on_model_steps.
    value_on_model_steps: int = 10
    # On-model generation and training validation both receive the configured
    # inference.inference_scheduler. The separate field preserves the on-model
    # helper's interface and its independently configured generation step count.
    value_on_model_scheduler: str = "unipc"
    # VM share of the sub-optimal samples; None = same split as task-irrelevant.
    suboptimal_vm_share: Optional[float] = None
    # VM share of the random-pool samples; None = same split as task-irrelevant.
    # 0 makes the random pool WM-only (e.g. simulator counterfactual branches,
    # whose expert prefix must not get a random-pool value target).
    fail_vm_share: Optional[float] = None
    random_vm_share: Optional[float] = None
    # Train the value head on policy-mode samples too, instead of value-mode
    # samples only. A policy sample already carries noisy video and action, so
    # its value target is learned under a condition the pure value mode never
    # sees; value_on_policy_max_sigma keeps only the low-noise part of that
    # distribution (sigma of the *action* draw), which is the part closest to
    # the clean condition. None = every policy sample. Default off keeps every
    # existing run's bytes unchanged.
    value_on_policy: bool = False
    value_on_policy_max_sigma: Optional[float] = None

    # rel_eef gripper normalization range (must match dataset)
    gripper_min: float = 0.0
    gripper_max: float = 1.0

    # Misc
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        self.action_d2_loss_weight = validate_action_d2_loss_weight(
            self.action_d2_loss_weight
        )
        self.inference_scheduler = str(self.inference_scheduler).lower()
        if self.inference_scheduler not in {"euler", "unipc"}:
            raise ValueError(f"Unsupported inference_scheduler: {self.inference_scheduler!r}")
        if self.video_rope_mode not in {"global", "window"}:
            raise ValueError(f"Unsupported video_rope_mode: {self.video_rope_mode}")
        if self.video_rope_mode == "window":
            cs = int(self.fixed_chunk_size)
            if cs < 1 or self.max_latent_window % cs or self.chunk_window_size < 1:
                raise ValueError("training_window reference requires complete positive chunks")
            if self.chunk_window_size * cs > self.max_latent_window - cs:
                raise ValueError("training_window reference cache exceeds the training clean prefix")
        if self.video_view_rope_mode not in {"shared", "hw_offset"}:
            raise ValueError(f"Unsupported video_view_rope_mode: {self.video_view_rope_mode}")
        self.num_views = int(self.num_views)
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if self.value_target not in {"progress", "ttc", "vlac"}:
            raise ValueError(f"Unsupported value_target: {self.value_target}")
        if self.value_temporal_sampling not in {"none", "random_chunks"}:
            raise ValueError(f"Unsupported value_temporal_sampling: {self.value_temporal_sampling}")
        if self.value_temporal_sampling != "none" and (not self.value_enabled or self.value_on_model):
            raise ValueError("VM temporal sampling requires value enabled and on_model disabled")
        self.wm_mask_language = bool(self.wm_mask_language)
        if self.wm_mask_language and self.value_on_model:
            raise ValueError("wm_mask_language does not cover the on-model value rollout")
        if self.suboptimal_vm_share is not None:
            self.suboptimal_vm_share = float(self.suboptimal_vm_share)
            if not 0.0 <= self.suboptimal_vm_share <= 1.0:
                raise ValueError(
                    f"suboptimal_vm_share must be in [0, 1], got {self.suboptimal_vm_share}"
                )
        if self.fail_vm_share is not None:
            self.fail_vm_share = float(self.fail_vm_share)
            if not 0.0 <= self.fail_vm_share <= 1.0:
                raise ValueError(f"fail_vm_share must be in [0, 1], got {self.fail_vm_share}")
        if self.random_vm_share is not None:
            self.random_vm_share = float(self.random_vm_share)
            if not 0.0 <= self.random_vm_share <= 1.0:
                raise ValueError(f"random_vm_share must be in [0, 1], got {self.random_vm_share}")
        raw_mode = str(self.mode).lower()
        supported_modes = {"joint", "video_idm", "action_only", "loop"}
        if raw_mode not in supported_modes:
            raise ValueError(
                f"Unsupported mode: {raw_mode}. Expected one of "
                "{'joint', 'video_idm', 'action_only', 'loop'}."
            )
        self.mode = raw_mode
        if self.video_steps is not None and self.mode == "video_idm" and not self.idm_mode:
            raise ValueError("mode=video_idm with video_steps requires idm_mode=True for action-only phase.")
        raw_backend = str(self.attention_backend).lower()
        supported_backends = {"sdpa", "flash_varlen", "flex"}
        if raw_backend not in supported_backends:
            raise ValueError(
                f"Unsupported attention_backend: {raw_backend}. Expected one of "
                "{'sdpa', 'flash_varlen', 'flex'}."
            )
        self.attention_backend = raw_backend
        self.wrist_xyz_normalization = str(self.wrist_xyz_normalization or "min_max")
        # bool comes in straight from yaml/cli; coerce via _as_bool-style logic.
        if self.num_action_inference_timesteps is None:
            self.num_action_inference_timesteps = self.num_inference_timesteps
        self.fixed_chunk_size = int(self.fixed_chunk_size)
        if self.fixed_chunk_size < 1:
            raise ValueError(f"fixed_chunk_size must be >= 1, got {self.fixed_chunk_size}")
        max_chunk_window = max(1, math.ceil(max(self.max_latent_window - self.fixed_chunk_size, 0) / self.fixed_chunk_size))
        lo, hi = (int(self.chunk_window_range[0]), int(self.chunk_window_range[1]))
        if lo < 1 or hi < lo:
            raise ValueError(f"Invalid chunk_window_range: {self.chunk_window_range}")
        if hi > max_chunk_window:
            raise ValueError(
                f"chunk_window_range={self.chunk_window_range} exceeds the maximum possible "
                f"{max_chunk_window} chunks for max_latent_window={self.max_latent_window} "
                f"and fixed_chunk_size={self.fixed_chunk_size}"
            )
        if int(self.chunk_window_size) < 1 or int(self.chunk_window_size) > max_chunk_window:
            raise ValueError(
                f"chunk_window_size={self.chunk_window_size} exceeds the maximum possible "
                f"{max_chunk_window} chunks for max_latent_window={self.max_latent_window} "
                f"and fixed_chunk_size={self.fixed_chunk_size}"
            )
        self.chunk_window_range = (lo, hi)
        self.chunk_window_size = int(self.chunk_window_size)
        self.attn_window = self.chunk_window_size * self.fixed_chunk_size
        if self.action_mode in {"joint", "rel_joint", "abs_eef_rpy"}:
            self.action_dim = 14
            self.action_state_dim = 14
        elif self.action_mode == "abs_eef52":
            self.action_dim = 52
            self.action_state_dim = 52
        elif self.action_mode in {"rel_eef", "rel_eef_wxyz"}:
            self.action_dim = 16
            self.action_state_dim = 16
        elif self.action_mode in {"rel_eef_6d", "abs_eef"}:
            self.action_dim = 20
            self.action_state_dim = 20
        elif self.action_mode == "abs_eef_wxyz":
            self.action_dim = 16
            self.action_state_dim = 16
        elif qpos_family_hand_dim(self.action_mode) is not None:
            self.action_dim = qpos_family_action_dim(qpos_family_hand_dim(self.action_mode))
            self.action_state_dim = self.action_dim
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        if self.value_target == "progress":
            self.value_range = (0.0, 1.0)
        elif self.value_target == "vlac":
            self.value_range = (-1.0, 1.0)
        else:
            self.value_range = (-1.0, 0.0)
        W = self.max_latent_window
        self.frame_chunk_size = self.fixed_chunk_size
        self.actions_per_latent = 4 * self.video_action_freq_ratio * self.action_upsample_ratio
        self.action_tokens_per_frame = self.actions_per_latent
        self.action_register_tokens_per_frame = self.action_tokens_per_frame
        self.action_tokens_per_timestep = 1
        self.state_tokens_per_frame = 1 if self.use_state else 0
        self.noisy_action_tokens_per_frame = (
            self.value_tokens_per_frame +
            self.state_tokens_per_frame +
            self.action_register_tokens_per_frame
        )
        self.chunk_action_length = self.fixed_chunk_size * self.actions_per_latent
        if self.rtc_training_enabled:
            # H is the chunk action horizon, s is where deployment replans, and the
            # committed prefix is the delay region at the head of the two chunks'
            # overlap. So it has to fit inside that overlap, H - s. Equality is the
            # no-hold case, where the new chunk takes over exactly as the old one
            # runs out; a shorter prefix leaves slack for delay jitter.
            H = self.chunk_action_length
            s = self.rtc_replan_action_index
            if not 0 < s < H:
                raise ValueError(
                    f"rtc_training.replan_action_index must lie in (0, {H}) "
                    f"(= fixed_chunk_size {self.fixed_chunk_size} x actions_per_latent "
                    f"{self.actions_per_latent} actions per chunk), got {s}"
                )
            if not 0 < self.rtc_committed_steps < H:
                raise ValueError(
                    f"rtc_training.committed_steps must lie in (0, {H}) so that every "
                    f"chunk keeps supervised actions, got {self.rtc_committed_steps}"
                )
            if self.rtc_committed_steps > H - s:
                raise ValueError(
                    "rtc_training.committed_steps cannot exceed the overlap between "
                    f"consecutive chunks, H - replan_action_index = {H} - {s} = {H - s}, "
                    f"got {self.rtc_committed_steps}"
                )
            if not 0.0 <= self.rtc_training_prob <= 1.0:
                raise ValueError(
                    f"rtc_training.probability must lie in [0, 1], got {self.rtc_training_prob}"
                )


# ---------------------------------------------------------------------------
# Motus Model
# ---------------------------------------------------------------------------

class Motus(nn.Module):
    """AR model: parallel chunk-wise training + streaming inference."""

    supports_unequal_joint_inference = True

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config
        _configure_training_compile(config)
        self.dtype = torch.bfloat16
        # Cache the layout flag as a plain bool at construction time. dynamo
        # treats a Python attribute set in __init__ as a trace-time constant,
        # so the layout-conditional branches below compile down to a single
        # static-shape graph per layout without graph breaks or recompiles.
        # t1v11c: T5 cross-attn K/V cache state (lazy install on first _t5_embed)
        self._t5_kv_hooks_installed = False
        # t1v11d: time embedding memoization cache
        self._inference_time_cache = {}
        # Plan Z: step_idx-indexed list cache (replaces .item()-dependent dict cache)
        self._step_emb_cache_key = None
        self._step_emb_cache_v_e0 = []
        self._step_emb_cache_v_emb = []
        self._step_emb_cache_a_emb = []
        # F-beta: chunk-scoped real-paired freqs cache
        self._b15_full_freqs_real = None
        # t1v11e: video RoPE freqs cache
        self._inference_rope_cache = {}

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
            )

        # The training config is authoritative for both initialization paths.
        self.video_model.wan_model.gradient_checkpointing = bool(config.gradient_checkpointing)

        wan_dim = getattr(self.video_model.wan_model.config, "dim", 3072)
        wan_heads = getattr(self.video_model.wan_model.config, "num_heads", 24)
        wan_hd = wan_dim // wan_heads
        wan_cfg = {"dim": wan_dim, "num_heads": wan_heads, "head_dim": wan_hd}
        self.wan_dim = wan_dim

        # --- Action expert ---
        W = config.max_latent_window
        N = W - self._fixed_chunk_size()  # noisy side starts after the fixed observation chunk
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
            use_mot=config.use_mot,
            use_state=config.use_state,
            action_mode=config.action_mode,
        )
        self.action_expert = ActionExpert(act_cfg, wan_cfg)

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
        if config.value_enabled:
            self.value_tokens.to(device=self.device, dtype=self.dtype)
            self.value_head.to(device=self.device, dtype=self.dtype)

        # --- Pre-compute grid sizes ---
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        self.per_view_frame_seqlen = lat_H * lat_W
        self.frame_seqlen = config.num_views * self.per_view_frame_seqlen  # tokens per latent frame after patch_embed
        self.lat_H = lat_H
        self.lat_W = lat_W
        if config.num_views > 1:
            self.view_embedding = nn.Parameter(torch.zeros(config.num_views, wan_dim))
            self.view_embedding.data = self.view_embedding.data.to(device=self.device, dtype=self.dtype)

        # --- RoPE frequency tables ---
        d = wan_dim // wan_heads
        # Open-source training uses the native WAN per-axis spectrum throughout.
        self.freqs = self.video_model.wan_model.freqs.clone()
        # Buffer must cover max_latent_start * tokens_per_frame across all episodes.
        # action_upsample_ratio=4 → napf=96, so 1024*10=10240 is too small for
        # episodes with >106 latent frames when train_global_rope=True.
        self.freqs_action = rope_params(1024 * 100, d)
        self.freqs_state = rope_params(1024 * 10, d)

        # --- Flow-Matching schedulers ---
        self.scheduler = FlowMatchScheduler(shift=config.video_timestep_shift, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler = FlowMatchScheduler(shift=config.action_timestep_shift, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
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
        self.kv_cache_valid_len: int = 0
        self.kv_cache_valid_len_tensor = torch.zeros((), device=self.device, dtype=torch.long)
        self.current_start_frame: int = 0

        self._log_params()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mode(self) -> str:
        return str(getattr(self.config, "mode", "joint") or "joint").lower()

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

    def _qpos20_wrist_xyz_norm_mode(self) -> str:
        mode = getattr(self.config, "wrist_xyz_normalization", None)
        return str(mode or "min_max")

    def _qpos20_wrist_xyz_affine(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Min-max affine params mapping relative wrist xyz (m) to [0, 1]."""
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

    def _qpos20_wrist_xyz_mean_std(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _qpos20_wrist_xyz_quantile(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @property
    def _qpos_layout(self):
        """Slice offsets for the active qpos{N}_rel6d layout (N = hand DoF)."""
        return qpos_family_layout(self._qpos_hand_dim)

    @property
    def _qpos_hand_dim(self) -> int:
        hand_dim = qpos_family_hand_dim(self.config.action_mode)
        if hand_dim is None:
            raise ValueError(
                f"action_mode={self.config.action_mode!r} is not in the qpos rel6d family"
            )
        return hand_dim

    def _extract_qpos20_rel_wrist_xyz6(self, rel_actions: torch.Tensor) -> torch.Tensor:
        lay = self._qpos_layout
        return torch.cat(
            [rel_actions[..., lay.left_xyz], rel_actions[..., lay.right_xyz]], dim=-1
        )

    def _write_qpos20_rel_wrist_xyz6(self, rel_actions: torch.Tensor, xyz6: torch.Tensor) -> torch.Tensor:
        lay = self._qpos_layout
        out = rel_actions.clone()
        out[..., lay.left_xyz] = xyz6[..., :3]
        out[..., lay.right_xyz] = xyz6[..., 3:]
        return out

    def _scale_qpos20_rel_wrist_xyz(self, rel_actions: torch.Tensor) -> torch.Tensor:
        """Normalize qpos{N}_rel6d relative wrist xyz (meters) for model training."""
        if qpos_family_hand_dim(self.config.action_mode) is None:
            return rel_actions
        rel_actions = rel_actions.float()
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
        """Convert normalized qpos{N}_rel6d wrist xyz back to meters."""
        if qpos_family_hand_dim(self.config.action_mode) is None:
            return rel_actions
        rel_actions = rel_actions.float()
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
        target_dtype = dtype or actions.dtype
        apl = self.config.actions_per_latent

        noisy_actions = actions[:, chunk_size * apl:].float()
        noisy_states = states[:, chunk_size:].float() if states is not None else None

        if (
            self.config.action_mode not in {"rel_joint", "rel_eef", "rel_eef_6d", "rel_eef_wxyz"}
            and qpos_family_hand_dim(self.config.action_mode) is None
        ):
            if not self.config.use_state:
                noisy_states = None
            return noisy_actions.to(target_dtype), noisy_states.to(target_dtype) if noisy_states is not None else None

        if noisy_states is None:
            raise ValueError(f"{self.config.action_mode} mode requires absolute state anchors for conversion")

        B = noisy_actions.shape[0]
        N = noisy_states.shape[1]
        # Geometry must stay FP32 even when the caller enables AMP.
        with torch.autocast(device_type=actions.device.type, enabled=False):
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
            elif qpos_family_hand_dim(self.config.action_mode) is not None:
                convert = (
                    abs_qpos20_wrist9_to_relqpos20_rel6d_torch
                    if is_rel_hand_qpos_mode(self.config.action_mode)
                    else abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch
                )
                rel_actions = convert(
                    noisy_states_fp32,
                    noisy_action_chunks_fp32,
                    hand_dim=self._qpos_hand_dim,
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
        return rel_actions.to(target_dtype), noisy_states.to(target_dtype) if noisy_states is not None else None

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

    def _max_cache_tokens(self) -> int:
        return int(self.config.chunk_window_size) * self._fixed_chunk_size() * self.frame_seqlen

    def _kv_cache_valid_len_int(self) -> int:
        return int(getattr(self, "kv_cache_valid_len", 0))

    def _kv_cache_valid_len_tensor(self, device: torch.device) -> torch.Tensor:
        value = getattr(self, "kv_cache_valid_len_tensor", None)
        if value is None:
            value = torch.tensor(self._kv_cache_valid_len_int(), device=device, dtype=torch.long)
            self.kv_cache_valid_len_tensor = value
        return value.to(device=device)

    def _set_kv_cache_valid_len(self, value: int) -> None:
        value_i = int(value)
        self.kv_cache_valid_len = value_i
        tensor = getattr(self, "kv_cache_valid_len_tensor", None)
        if tensor is not None:
            device = tensor.device
            self.kv_cache_valid_len_tensor = torch.tensor(value_i, device=device, dtype=torch.long)

    def _kv_cache_key_valid_mask(self, current_len: int, device: torch.device) -> torch.Tensor:
        """Valid key mask for fixed-size cache prefix plus current chunk tokens."""
        max_len = self._max_cache_tokens()
        valid_len = self._kv_cache_valid_len_tensor(device).clamp(min=0, max=max_len)
        cache_valid = torch.arange(max_len, device=device) < valid_len
        current_valid = torch.ones(int(current_len), dtype=torch.bool, device=device)
        return torch.cat([cache_valid, current_valid], dim=0)

    def _kv_cache_attention_mask(
        self,
        query_len: int,
        current_len: int,
        device: torch.device,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_valid = self._kv_cache_key_valid_mask(current_len, device)
        mask = key_valid.view(1, 1, 1, -1)
        if visibility is not None:
            mask = mask & visibility.view(1, 1, int(query_len), -1)
        return mask

    def _fixed_cache_sdpa(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention over a fixed-shape KV cache buffer + current chunk tokens.

        Backend selected by config.attention_backend:

        * "sdpa" (default): F.scaled_dot_product_attention over the
          [k_cache | k_current] concatenation with an explicit attn_mask that
          gates the unused cache slots ([valid_len, max_cache)). Compile-
          friendly because shapes are static (max_cache + L_current is a
          constant), but the SDPA backend dispatch (FLASH -> EFFICIENT -> MATH)
          is mask-dependent: with a boolean attn_mask FLASH declines and
          EFFICIENT/MATH take over. Their bf16 reduction order differs from
          flash-attn's by ~ULP and that drift compounds across 30 transformer
          layers.

        * "flash_varlen": packs the valid prefix + current tokens into a
          contiguous [total_tokens, H, D] tensor and calls
          flash_attn.flash_attn_varlen_func — the same kernel WAN's
          flash_attention wraps in Wan. Numerically bit-equivalent
          to that path. The compaction (k_cache[:, :valid_len]) is a dynamic-
          shape slice; under torch.compile with `dynamic=True` it is traced as
          a SymInt rather than triggering a graph break. Falls back to the
          sdpa branch when `visibility` is non-trivial because varlen cannot
          express arbitrary per-query masks.
        """
        backend = getattr(self.config, "attention_backend", "sdpa")
        if backend == "flash_varlen" and visibility is None and q.shape[0] == 1:
            return self._fixed_cache_flash_varlen(q, k_cache, v_cache, k_current, v_current)
        if backend == "flex" and visibility is None:
            return self._fixed_cache_flex(q, k_cache, v_cache, k_current, v_current)

        attn_k = torch.cat([k_cache, k_current], dim=1)
        attn_v = torch.cat([v_cache, v_current], dim=1)
        attn_mask = self._kv_cache_attention_mask(
            query_len=q.shape[1],
            current_len=k_current.shape[1],
            device=q.device,
            visibility=visibility,
        )
        q_bhld = q.permute(0, 2, 1, 3).contiguous()
        k_bhld = attn_k.permute(0, 2, 1, 3).contiguous()
        v_bhld = attn_v.permute(0, 2, 1, 3).contiguous()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            with sdpa_kernel(backends):
                y = F.scaled_dot_product_attention(
                    q_bhld,
                    k_bhld,
                    v_bhld,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                )
        except (ImportError, AttributeError):
            y = F.scaled_dot_product_attention(
                q_bhld,
                k_bhld,
                v_bhld,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
        return y.permute(0, 2, 1, 3).contiguous()

    def _fixed_cache_flash_varlen(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
    ) -> torch.Tensor:
        """flash_attn_varlen_func over a compacted (cache_prefix | current) packing.

        Returns [B, L_q, H, D], matching the sdpa path's return shape so call
        sites can use either backend interchangeably.

        Caller invariants enforced by _fixed_cache_sdpa dispatch:
            visibility is None (varlen can't represent arbitrary masks)
            B == 1                (single-stream streaming inference)

        Numerical equivalence to Wan:
            A compact KV cache computes:
                full_k = cat(cached_k, k_current); flash_attention(q, full_k, full_v)
            The fixed buffer represents the same valid sequence:
              (i) k_cache is right-padded with zeros past valid_len,
             (ii) we slice it down to k_cache[:, :valid_len] before the cat.
            Both reach flash-attn with the same logical (q, valid_k, valid_v),
            so the kernel produces bit-identical bf16 output.
        """
        from flash_attn import flash_attn_varlen_func

        B, L_q, H, D = q.shape
        L_current = int(k_current.shape[1])
        max_cache_len = int(k_cache.shape[1])
        # Plan A: read valid_len as Python int (dynamo specializes per value).
        # Each unique valid_len yields a separate compiled cudagraph; with 3
        # chunks (vlens 480/960/1440) we get 3 graphs. Avoids the .item()
        # graph break that disables cudagraph in the B15 big-graph compile.
        valid_len = self._kv_cache_valid_len_tensor(q.device).to(torch.int64).item()
        torch._check_is_size(valid_len)
        torch._check(valid_len <= max_cache_len)
        k_valid = k_cache.narrow(1, 0, valid_len)
        v_valid = v_cache.narrow(1, 0, valid_len)
        # === t1v11n: SDPA fast path (shengshu lht_0519 pattern) ===
        import os as _os
        if _os.environ.get('MOTUS_USE_SDPA', '0') == '1':
            k_full = torch.cat([k_valid, k_current], dim=1)
            v_full = torch.cat([v_valid, v_current], dim=1)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k_full.transpose(1, 2),
                    v_full.transpose(1, 2),
                )
            return out.transpose(1, 2).contiguous()
        # === flash_attn_varlen_func path (default, bit-exact to wan flash) ===
        k_packed = torch.cat([k_valid, k_current], dim=1).reshape(-1, H, D)
        v_packed = torch.cat([v_valid, v_current], dim=1).reshape(-1, H, D)
        q_packed = q.reshape(-1, H, D)

        cu_seqlens_q = torch.tensor([0, L_q], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor(
            [0, valid_len + L_current], device=q.device, dtype=torch.int32
        )
        max_seqlen_q = int(L_q)
        max_seqlen_k = max_cache_len + L_current
        y = flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
        )
        return y.view(B, L_q, H, D)

    def _fixed_cache_flex(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
    ) -> torch.Tensor:
        """flex_attention path. Compile-friendly static-shape attention over the
        full `[k_cache | k_current]` buffer with a BlockMask gating the
        unused cache slots (positions in [valid_len, max_cache_len)).

        Trade-offs vs `_fixed_cache_flash_varlen`:
          * Tensor shapes are fully static — no `.item()`, no
            `_check_is_size`, no narrow with tensor length. dynamo traces a
            single static graph; torch.compile (any backend) and CUDA graphs
            can capture it cleanly.
          * Underlying kernel is triton, not flash-attn's cutlass. bf16
            reduction order differs by ULP. Per-call max|d| vs the
            flash_varlen reference is typically 1e-4 ~ 1e-3 in attention
            output; the final action58_rel max|d| vs 0501 is non-zero but
            well below the 1.57 we had before the RoPE fix.
          * BlockMask is constructed per-call because valid_len varies. The
            mask compute itself is cheap (it's a coarse-grained block visibility
            graph, not a per-element mask), and `_compile=True` makes
            subsequent constructions reuse the same compiled mask kernel.

        Caller invariants (enforced by the dispatcher in _fixed_cache_sdpa):
          * visibility is None — flex BlockMask can express arbitrary masks,
            but our current code path only needs the valid_len gate. If we
            later need video_idm/action_only visibility patterns we can
            compose them into mask_fn here.
        """
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        B, L_q, H, D = q.shape
        max_cache_len = k_cache.shape[1]
        L_current = k_current.shape[1]
        L_k_total = max_cache_len + L_current

        # Always feed the FULL buffer to flex; no compaction. Static shapes.
        k_full = torch.cat([k_cache, k_current], dim=1)   # [B, L_k_total, H, D]
        v_full = torch.cat([v_cache, v_current], dim=1)
        # flex_attention wants [B, H, L, D] layout.
        q_bhld = q.transpose(1, 2).contiguous()
        k_bhld = k_full.transpose(1, 2).contiguous()
        v_bhld = v_full.transpose(1, 2).contiguous()

        # Capture valid_len as a python int (specializes BlockMask per value).
        # This is the ONE place we still hit Python int — but it's outside
        # the compiled forward (BlockMask construction is its own compile
        # cache, keyed on the captured value). Since valid_len takes only a
        # handful of discrete values across a deployment (one per chunk
        # boundary), the cache fills quickly and per-call cost amortizes.
        valid_len = int(self._kv_cache_valid_len_int())

        def mask_fn(b, h, q_idx, kv_idx):
            # Visible:
            #   * cache prefix: kv_idx < valid_len
            #   * current tokens: max_cache_len <= kv_idx < L_k_total
            in_cache = kv_idx < valid_len
            in_current = (kv_idx >= max_cache_len) & (kv_idx < L_k_total)
            return in_cache | in_current

        block_mask = create_block_mask(
            mask_fn,
            B=None, H=None,
            Q_LEN=L_q, KV_LEN=L_k_total,
            device=q.device,
            _compile=True,
        )

        y_bhld = flex_attention(q_bhld, k_bhld, v_bhld, block_mask=block_mask)
        return y_bhld.transpose(1, 2)  # [B, L_q, H, D]

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

    def _prepare_noisy_action_blocks(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        chunk_size: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Regroup noisy actions/states into per-chunk blocks.

        For use_state=True, each block uses the state of its first latent:
          [state_chunk_start | actions_for_all_latents_in_chunk]

        For rel_eef, each chunk's absolute EEF actions are converted relative to the
        chunk-start state, not per-latent states. abs_eef keeps chunk actions absolute
        in 20D rot6d form. abs_eef_wxyz keeps them absolute in 16D wxyz form.
        """
        target_dtype = dtype or actions.dtype
        apl = self.config.actions_per_latent
        noisy_actions_lat = actions[:, chunk_size * apl:].float()
        noisy_states_lat = states[:, chunk_size:].float() if states is not None else None

        if noisy_actions_lat.numel() == 0:
            return {
                "flat_actions": noisy_actions_lat.to(target_dtype),
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
        need_block_state = (
            self.config.action_mode in {"rel_joint", "rel_eef", "rel_eef_6d", "rel_eef_wxyz"}
            or qpos_family_hand_dim(self.config.action_mode) is not None
        )
        offset = 0
        for count in block_latent_counts:
            abs_block_actions = noisy_actions_lat[:, offset:offset + count].reshape(B, count * apl, -1)
            state_block = noisy_states_lat[:, offset] if noisy_states_lat is not None else None
            if need_block_state and state_block is None:
                raise ValueError(f"{self.config.action_mode} mode requires chunk-start state anchors for conversion")

            # AMP also downcasts FP32 matmul inputs unless explicitly disabled.
            with torch.autocast(device_type=actions.device.type, enabled=False):
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
                elif qpos_family_hand_dim(self.config.action_mode) is not None:
                    convert = (
                        abs_qpos20_wrist9_to_relqpos20_rel6d_torch
                        if is_rel_hand_qpos_mode(self.config.action_mode)
                        else abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch
                    )
                    block_actions = convert(
                        state_block.float(),
                        abs_block_actions.float(),
                        hand_dim=self._qpos_hand_dim,
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
        if self.config.use_state and states is None:
            raise ValueError("State conditioning requires measured chunk-start states")
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

    def _action_register_timesteps(
        self,
        t_values: torch.Tensor,
        count: int,
        action_token_t: Optional[torch.Tensor],
        offset: int,
    ) -> torch.Tensor:
        """Timesteps for one chunk's action registers.

        `action_token_t` is [B, total_action_tokens] in the flat per-action layout;
        when given, its values replace the per-sample scalar on the action registers
        of this chunk.
        """
        apf = self.config.action_register_tokens_per_frame
        registers = t_values.view(t_values.shape[0], 1).expand(-1, count * apf)
        if action_token_t is None:
            return registers
        block_len = count * self.config.actions_per_latent
        registers = registers.clone()
        registers[:, ::self.config.action_tokens_per_timestep] = action_token_t[
            :, offset:offset + block_len
        ].to(dtype=registers.dtype)
        return registers

    def _build_chunked_action_timesteps(
        self,
        t_values: torch.Tensor,
        block_latent_counts: List[int],
        include_value: bool = False,
        action_token_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build per-token timesteps for chunk-grouped action registers."""
        parts = []
        prefix = self.config.state_tokens_per_frame
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        offset = 0
        for count in block_latent_counts:
            if nvpf > 0:
                parts.append(torch.zeros(t_values.shape[0], nvpf, dtype=t_values.dtype, device=t_values.device))
            if prefix > 0:
                parts.append(t_values.view(t_values.shape[0], 1).expand(-1, prefix))
            parts.append(self._action_register_timesteps(t_values, count, action_token_t, offset))
            offset += count * self.config.actions_per_latent
        return torch.cat(parts, dim=1) if parts else t_values.new_zeros(t_values.shape[0], 0)

    def _strip_chunk_state_predictions(
        self,
        action_pred: torch.Tensor,
        block_action_token_lengths: List[int],
    ) -> torch.Tensor:
        """Remove one state prediction token per chunk, keeping only action predictions."""
        if self.config.state_tokens_per_frame == 0:
            return action_pred
        parts = []
        offset = 0
        for block_len in block_action_token_lengths:
            total_len = self.config.state_tokens_per_frame + block_len
            block = action_pred[:, offset:offset + total_len]
            parts.append(block[:, self.config.state_tokens_per_frame:])
            offset += total_len
        return torch.cat(parts, dim=1) if parts else action_pred[:, :0]



    def _interleave_chunk_ar_tokens(
        self,
        action_tokens: torch.Tensor,
        noisy_video_tokens: torch.Tensor,
        value_tokens: Optional[torch.Tensor],
        block_latent_counts: List[int],
        video_tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """Pack noisy suffix as per-chunk [state/action | noisy_video | value]."""
        nvpf = self.config.value_tokens_per_frame if value_tokens is not None else 0
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        vpf = int(video_tokens_per_frame or self.frame_seqlen)
        expected_action_len = sum(spf + count * apf for count in block_latent_counts)
        expected_video_len = sum(count * vpf for count in block_latent_counts)
        expected_value_len = len(block_latent_counts) * nvpf
        if action_tokens.shape[1] != expected_action_len:
            raise ValueError(
                f"Chunk-AR pack action length mismatch: got {action_tokens.shape[1]}, "
                f"expected {expected_action_len}"
            )
        if noisy_video_tokens.shape[1] != expected_video_len:
            raise ValueError(
                f"Chunk-AR pack video length mismatch: got {noisy_video_tokens.shape[1]}, "
                f"expected {expected_video_len}"
            )
        if value_tokens is not None and value_tokens.shape[1] != expected_value_len:
            raise ValueError(
                f"Chunk-AR pack value length mismatch: got {value_tokens.shape[1]}, "
                f"expected {expected_value_len}"
            )

        parts = []
        action_offset = 0
        video_offset = 0
        value_offset = 0
        for count in block_latent_counts:
            action_len = spf + count * apf
            video_len = count * vpf
            if action_len > 0:
                parts.append(action_tokens[:, action_offset:action_offset + action_len])
            if video_len > 0:
                parts.append(noisy_video_tokens[:, video_offset:video_offset + video_len])
            if nvpf > 0:
                parts.append(value_tokens[:, value_offset:value_offset + nvpf])
                value_offset += nvpf
            action_offset += action_len
            video_offset += video_len
        return torch.cat(parts, dim=1) if parts else noisy_video_tokens[:, :0]

    def _build_chunk_ar_timesteps(
        self,
        action_t: torch.Tensor,
        video_t: torch.Tensor,
        block_latent_counts: List[int],
        include_value: bool = True,
        video_tokens_per_frame: Optional[int] = None,
        action_token_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build timesteps matching per-chunk [state/action | noisy_video | value]."""
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        vpf = int(video_tokens_per_frame or self.frame_seqlen)
        parts = []
        action_offset = 0
        for count in block_latent_counts:
            action_len = spf + count * apf
            if action_len > 0:
                registers = self._action_register_timesteps(
                    action_t, count, action_token_t, action_offset
                )
                if spf > 0:
                    registers = torch.cat(
                        [action_t.view(action_t.shape[0], 1).expand(-1, spf), registers], dim=1
                    )
                parts.append(registers)
            action_offset += count * self.config.actions_per_latent
            video_len = count * vpf
            if video_len > 0:
                parts.append(video_t.view(video_t.shape[0], 1).expand(-1, video_len))
            if nvpf > 0:
                parts.append(torch.zeros(action_t.shape[0], nvpf, dtype=action_t.dtype, device=action_t.device))
        return torch.cat(parts, dim=1) if parts else action_t.new_zeros(action_t.shape[0], 0)

    def _cat_chunk_tokens(self, action_part: torch.Tensor, video_part: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Concatenate action tokens followed by video tokens."""
        return torch.cat([action_part, video_part], dim=dim)

    def _chunk_token_slices(self, a_seq_len: int, v_seq_len: int) -> Tuple[slice, slice]:
        """Return (act_slice, vid_slice) into a concat'd chunk tensor that
        matches the layout-respecting `_cat_chunk_tokens` ordering.
        """
        return slice(0, a_seq_len), slice(a_seq_len, a_seq_len + v_seq_len)


    def _build_chunk_ar_freqs(
        self,
        clean_video_freqs: torch.Tensor,
        noisy_video_freqs: torch.Tensor,
        block_latent_counts: List[int],
        include_value: bool = True,
        video_tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """Build native RoPE freqs for [state/action | noisy_video | value] chunks."""
        self._ensure_freqs_device()
        if clean_video_freqs.dim() == 4:
            return torch.stack(
                [
                    self._build_chunk_ar_freqs(
                        clean_video_freqs[b],
                        noisy_video_freqs[b],
                        block_latent_counts,
                        include_value=include_value,
                        video_tokens_per_frame=video_tokens_per_frame,
                    )
                    for b in range(clean_video_freqs.shape[0])
                ],
                dim=0,
            )
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        vpf = int(video_tokens_per_frame or self.frame_seqlen)

        parts: List[torch.Tensor] = [clean_video_freqs]
        video_offset = 0
        for count in block_latent_counts:
            action_parts: List[torch.Tensor] = []
            prefix = self.config.state_tokens_per_frame
            if prefix > 0:
                action_parts.append(torch.ones((prefix, 1, self.freqs_state.shape[-1]), device=self.device, dtype=self.freqs_state.dtype))
            action_len_no_state = count * apf
            if action_len_no_state > 0:
                action_parts.append(
                    self.freqs_action[: (action_len_no_state + self.config.action_tokens_per_timestep - 1) // self.config.action_tokens_per_timestep]
                    .repeat_interleave(self.config.action_tokens_per_timestep, dim=0)[:action_len_no_state]
                    .view(action_len_no_state, 1, -1)
                )

            action_freqs = torch.cat(action_parts, dim=0) if action_parts else None
            video_len = count * vpf
            video_freqs = noisy_video_freqs[video_offset:video_offset + video_len]
            video_offset += video_len

            if action_freqs is not None:
                parts.append(action_freqs)
            parts.append(video_freqs)

            if nvpf > 0:
                parts.append(torch.ones((nvpf, 1, self.freqs_state.shape[-1]), device=self.device, dtype=self.freqs_state.dtype))

        return torch.cat(parts, dim=0)

    def _split_chunk_ar_hidden(
        self,
        full_noisy_hidden: torch.Tensor,
        block_latent_counts: List[int],
        include_value: bool = True,
        video_tokens_per_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Unpack per-chunk [state/action | noisy_video | value] hidden states."""
        nvpf = self.config.value_tokens_per_frame if include_value else 0
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        vpf = int(video_tokens_per_frame or self.frame_seqlen)
        expected_len = sum((spf + count * apf) + (count * vpf) + nvpf for count in block_latent_counts)
        if full_noisy_hidden.shape[1] < expected_len:
            raise ValueError(
                f"Chunk-AR suffix too short: got {full_noisy_hidden.shape[1]}, expected at least {expected_len}"
            )

        action_parts: List[torch.Tensor] = []
        video_parts: List[torch.Tensor] = []
        value_parts: List[torch.Tensor] = []
        offset = 0
        for count in block_latent_counts:
            action_len = spf + count * apf
            video_len = count * vpf
            if action_len > 0:
                action_parts.append(full_noisy_hidden[:, offset:offset + action_len])
                offset += action_len
            if video_len > 0:
                video_parts.append(full_noisy_hidden[:, offset:offset + video_len])
                offset += video_len
            if nvpf > 0:
                value_parts.append(full_noisy_hidden[:, offset:offset + nvpf])
                offset += nvpf
        action_hidden = torch.cat(action_parts, dim=1) if action_parts else full_noisy_hidden[:, :0]
        video_hidden = torch.cat(video_parts, dim=1) if video_parts else full_noisy_hidden[:, :0]
        value_hidden = None
        if nvpf > 0 and value_parts:
            value_concat = torch.cat(value_parts, dim=1)
            value_hidden = value_concat.reshape(
                full_noisy_hidden.shape[0], len(block_latent_counts), nvpf, -1
            )
        return action_hidden, video_hidden, value_hidden

    def _current_chunk_visibility_mask(
        self,
        mode: str,
        query_len: int,
        key_len: int,
        cache_len: int,
        video_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Bool attention mask for current chunk visibility during inference."""
        mask = torch.ones(query_len, key_len, dtype=torch.bool, device=device)
        cur_key_start = int(cache_len)
        cur_key_len = key_len - cur_key_start
        if cur_key_len <= 0:
            return mask

        mode = str(mode or "joint").lower()
        a_len = max(0, int(query_len) - int(video_len))
        a_q0, a_q1 = 0, a_len
        v_q0, v_q1 = a_q1, int(query_len)
        a_k0, a_k1 = cur_key_start, cur_key_start + a_len
        v_k0, v_k1 = a_k1, key_len

        if mode == "video_idm":
            # action <- video, but video cannot see current action.
            if v_q1 > v_q0 and a_k1 > a_k0:
                mask[v_q0:v_q1, a_k0:a_k1] = False
        elif mode in {"action_only", "fdm"}:
            # video <- action, but action cannot see current video.
            if a_q1 > a_q0 and v_k1 > v_k0:
                mask[a_q0:a_q1, v_k0:v_k1] = False
        return mask

    def _make_inference_video_noise(self, batch_size: int, num_frames: int) -> torch.Tensor:
        c_lat = self.video_model.wan_model.in_dim
        if self.config.num_views > 1:
            return torch.randn(
                batch_size,
                self.config.num_views,
                c_lat,
                num_frames,
                self.lat_H * 2,
                self.lat_W * 2,
                device=self.device,
                dtype=self.dtype,
            )
        return torch.randn(
            batch_size,
            c_lat,
            num_frames,
            self.lat_H * 2,
            self.lat_W * 2,
            device=self.device,
            dtype=self.dtype,
        )

    def _empty_current_video_latent(self, batch_size: int, num_frames: int) -> torch.Tensor:
        c_lat = self.video_model.wan_model.in_dim
        if self.config.num_views > 1:
            return torch.zeros(
                batch_size,
                self.config.num_views,
                c_lat,
                num_frames,
                self.lat_H * 2,
                self.lat_W * 2,
                device=self.device,
                dtype=self.dtype,
            )
        return torch.zeros(
            batch_size,
            c_lat,
            num_frames,
            self.lat_H * 2,
            self.lat_W * 2,
            device=self.device,
            dtype=self.dtype,
        )

    def load_checkpoint_weights(
        self, path: str, strict: bool = True, mmap: bool = True, strict_action: bool = False,
        allow_new_state_encoder: bool = False,
    ) -> None:
        """Load model weights from checkpoint (for finetune from pretrained).

        Unlike Motus1's load_pretrain_weights, this does a normal full load—no
        filtering of action encoder/decoder. Use for: pretrain done, finetune on
        new dataset (load model only, optimizer/scheduler start fresh).

        Args:
            path: Checkpoint directory (e.g. checkpoint_step_20000) or direct .pt file.
                  Directory expects: pytorch_model/mp_rank_00_model_states.pt
            strict: If False, allow missing/unexpected keys
            mmap: Map the file instead of reading it. Right for a large file that
                  many readers share, wrong for a small one that nobody has read
                  yet -- see the note below.
        """
        p = Path(path)
        if p.is_dir():
            model_file = p / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            path = str(model_file)
        elif not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Memory-map compatible checkpoints to share the OS page cache between
        # readers. Disable mmap for workloads that prefer sequential file I/O.
        # Fall back to ordinary loading for unsupported serialization formats.
        if mmap:
            try:
                ckpt = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                )
            except (RuntimeError, ValueError, TypeError):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("module", ckpt.get("state_dict", ckpt))
        # A no-state base can gain measured-state conditioning without resetting
        # any learned action parameters. Only absent state-encoder tensors are
        # allowed; existing tensors (including state tensors) must still match.
        if allow_new_state_encoder and (not strict_action or strict or not self.config.use_state):
            raise ValueError("allow_new_state_encoder requires use_state=True, strict_action=True, strict=False")
        new_state_keys = []
        if strict_action:
            expected = {k: v for k, v in self.named_parameters() if k.startswith('action_expert.')}
            new_state_keys = [k for k in expected
                              if allow_new_state_encoder
                              and not any(n.startswith('action_expert.input_encoder.state_encoder.') for n in state_dict)
                              and k.startswith('action_expert.input_encoder.state_encoder.')
                              and k not in state_dict]
            problems = {k: (tuple(state_dict[k].shape) if k in state_dict else None, tuple(v.shape))
                        for k, v in expected.items()
                        if k not in new_state_keys
                        and (k not in state_dict or tuple(state_dict[k].shape) != tuple(v.shape))}
            if problems:
                raise ValueError(f'Action expert must load without reinitialization: {problems}')
            logger.info('Verified %d existing action-expert parameters are present and shape-compatible',
                        len(expected) - len(new_state_keys))
            if new_state_keys:
                logger.info('Initializing only %d new measured-state encoder tensors: %s',
                            len(new_state_keys), new_state_keys)
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
        # The fused QKV weight is a copy, so it goes stale the moment q/k/v
        # change. Refreshing here rather than at the call sites is what makes it
        # safe: every loader gets it, including the MBRL weight sync that reloads
        # a published reference mid-run, which would otherwise leave the fused
        # copy describing the previous version's weights. Callers that must not
        # fuse (an unfrozen backbone, where training would drift the originals)
        # express that by leaving config.qkv_fuse false.
        if getattr(self.config, "qkv_fuse", False):
            install_wan_qkv_fuse(self)

    def reset_time_embedding_caches(self) -> None:
        """Drop memoized time embeddings; they are functions of the current weights."""
        self._inference_time_cache = {}
        self._step_emb_cache_key = None
        self._step_emb_cache_v_e0 = []
        self._step_emb_cache_v_emb = []
        self._step_emb_cache_a_emb = []

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Every weight load (finetune init, MBRL reference sync) invalidates the
        # memoized time embeddings, which would otherwise keep the old weights' output.
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.reset_time_embedding_caches()
        return result

    def train(self, mode: bool = True):
        # Optimizer steps taken in train mode change the time embedding in place,
        # so an eval pass after them must not reuse embeddings cached before.
        if hasattr(self, "_inference_time_cache"):
            self.reset_time_embedding_caches()
        return super().train(mode)

    def _ensure_freqs_device(self):
        dev = self.device
        if self.freqs.device != dev:
            self.freqs = self.freqs.to(dev)
        if self.freqs_action.device != dev:
            self.freqs_action = self.freqs_action.to(dev)
        if self.freqs_state.device != dev:
            self.freqs_state = self.freqs_state.to(dev)

    def _video_rope_is_global(self) -> bool:
        return self.config.video_rope_mode == "global"

    def _should_adjust_video_rope_on_pop(self) -> bool:
        return not self._video_rope_is_global()

    def _inference_video_start_frame(self, cache_seq_len: int) -> int:
        if self._video_rope_is_global():
            return self.current_start_frame
        return getattr(self, "_window_rope_cache_start", 0) + cache_seq_len // self.frame_seqlen

    def _window_cache_rope_update(self, old_len: int, added_len: int) -> Tuple[int, int]:
        """Return (new cache origin, temporal shift) after a cache append.

        Training resets action/state phases for each chunk but gives video its
        position in the sampled window. Keep streaming video in that same
        coordinate system: advance through the training window, then reuse its
        final prediction block. Merely removing all rebasing would drift beyond
        the trained coordinates on long episodes.
        """
        drop = max(old_len + added_len - self._max_cache_tokens(), 0)
        old_origin = getattr(self, "_window_rope_cache_start", 0)
        retained_frames = (old_len + added_len - drop) // self.frame_seqlen
        end = min(
            old_origin + (old_len + added_len) // self.frame_seqlen,
            self.config.max_latent_window - self._fixed_chunk_size(),
        )
        new_origin = max(end - retained_frames, 0)
        return new_origin, old_origin + drop // self.frame_seqlen - new_origin

    def _shift_cached_video_k_temporal(self, k_cache: torch.Tensor, delta_frames: int) -> torch.Tensor:
        """Shift cached video K by -delta_frames in temporal RoPE space for window mode."""
        if delta_frames <= 0 or k_cache.numel() == 0:
            return k_cache

        self._ensure_freqs_device()
        d = k_cache.shape[-1]
        c = d // 2
        c_t = c - 2 * (c // 3)

        if delta_frames >= self.freqs.shape[0]:
            logger.warning(
                "video_rope_mode=window shift skipped: delta_frames=%d exceeds precomputed freq len=%d",
                delta_frames,
                int(self.freqs.shape[0]),
            )
            return k_cache

        phase_t = self.freqs[delta_frames, :c_t].conj().to(k_cache.device)

        B, L, n, _ = k_cache.shape
        k_complex = torch.view_as_complex(k_cache.to(torch.float64).reshape(B, L, n, -1, 2))
        k_complex[..., :c_t] = k_complex[..., :c_t] * phase_t.view(1, 1, 1, c_t)
        return torch.view_as_real(k_complex).flatten(3).to(k_cache.dtype)

    @staticmethod
    def _latent_grid_from_tensor(latent: torch.Tensor) -> Tuple[int, int, int]:
        """Return (num_views, patch_grid_h, patch_grid_w) for a latent batch."""
        if latent.dim() == 6:
            _, views, _, _, h_lat, w_lat = latent.shape
        elif latent.dim() == 5:
            views = 1
            _, _, _, h_lat, w_lat = latent.shape
        else:
            raise ValueError(f"Expected video latent [B,C,T,H,W] or [B,V,C,T,H,W], got {tuple(latent.shape)}")
        if h_lat % 2 != 0 or w_lat % 2 != 0:
            raise ValueError(f"WAN patch embedding expects even latent H/W, got {(h_lat, w_lat)}")
        return int(views), int(h_lat // 2), int(w_lat // 2)

    def _video_grid_from_latent(self, latent: torch.Tensor) -> Tuple[int, int, int, int, int]:
        """Return (views, lat_H, lat_W, per_view_frame_seqlen, frame_seqlen)."""
        views, lat_h, lat_w = self._latent_grid_from_tensor(latent)
        per_view = lat_h * lat_w
        return views, lat_h, lat_w, per_view, views * per_view


    def _create_video_freqs(
        self,
        f,
        start_frame=0,
        lat_H: Optional[int] = None,
        lat_W: Optional[int] = None,
        num_views: Optional[int] = None,
    ):
        """Create video freqs for 3D RoPE.

        Args:
            f: Number of temporal frames
            start_frame: Starting frame index (for inference)

        Returns:
            freqs: [f * lat_H * lat_W, 1, head_dim//2] complex tensor
        """
        _nvpush("_create_video_freqs")
        self._ensure_freqs_device()
        lat_H = int(self.lat_H if lat_H is None else lat_H)
        lat_W = int(self.lat_W if lat_W is None else lat_W)
        view_count = int(self.config.num_views if num_views is None else num_views)
        # t1v11e: video RoPE freqs cache (bit-exact)
        _use_rope_cache = (not self.training and
                           bool(getattr(self.config, "cache_inference_rope_tensors", False)))
        _rope_cache_key = None
        if _use_rope_cache:
            _rope_cache_key = (int(f), int(start_frame), int(lat_H), int(lat_W),
                               int(view_count), str(self.device))
            _cached = self._inference_rope_cache.get(_rope_cache_key)
            if _cached is not None:
                # Clone on return so downstream mutation can't poison cache.
                return _cached.clone()
        frame_seqlen = view_count * lat_H * lat_W
        d = self.wan_dim // self.video_model.wan_model.num_heads
        c = d // 2
        # Split freqs into [freqs_t, freqs_h, freqs_w] like rope_apply_one
        freqs_split = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        freqs_t, freqs_h, freqs_w = freqs_split
        t_freqs = freqs_t[start_frame:start_frame + f]
        if t_freqs.shape[0] < f:
            raise ValueError(
                f"Video RoPE temporal range [{start_frame}, {start_frame + f}) exceeds "
                f"precomputed length {freqs_t.shape[0]}"
            )

        # Create 3D freqs per view: [f, V, lat_H, lat_W, head_dim//2].
        # For stereo hw_offset we still use F/H/W RoPE; views occupy
        # non-overlapping W coordinate ranges while preserving the same H rows.
        view_freqs = []
        for view_idx in range(view_count):
            if view_count > 1 and self.config.video_view_rope_mode == "hw_offset":
                h_offset = 0
                w_offset = view_idx * lat_W
            else:
                h_offset = 0
                w_offset = 0
            h_end = h_offset + lat_H
            w_end = w_offset + lat_W
            if h_end > freqs_h.shape[0] or w_end > freqs_w.shape[0]:
                raise ValueError(
                    f"video_view_rope_mode={self.config.video_view_rope_mode} needs H/W RoPE "
                    f"up to ({h_end}, {w_end}), but table is ({freqs_h.shape[0]}, {freqs_w.shape[0]})"
                )
            per_view_freqs = torch.cat([
                t_freqs.view(f, 1, 1, -1).expand(f, lat_H, lat_W, -1),
                freqs_h[h_offset:h_end].view(1, lat_H, 1, -1).expand(f, lat_H, lat_W, -1),
                freqs_w[w_offset:w_end].view(1, 1, lat_W, -1).expand(f, lat_H, lat_W, -1),
            ], dim=-1)
            view_freqs.append(per_view_freqs)

        if view_count > 1:
            video_freqs = torch.stack(view_freqs, dim=1).reshape(f * frame_seqlen, 1, -1)
        else:
            video_freqs = view_freqs[0].reshape(f * frame_seqlen, 1, -1)

        if _rope_cache_key is not None:
            # Clone before store to protect against downstream in-place mutation
            # of the cached tensor (e.g., view-based rope_apply that operates on freqs).
            self._inference_rope_cache[_rope_cache_key] = video_freqs.detach().clone()
        _nvpop()  # _create_video_freqs
        return video_freqs

    def _create_video_freqs_batch(
        self,
        f: int,
        start_frames: torch.Tensor,
        lat_H: Optional[int] = None,
        lat_W: Optional[int] = None,
        num_views: Optional[int] = None,
    ) -> torch.Tensor:
        """Create per-sample video freqs for 3D RoPE.

        Args:
            f: Number of temporal frames.
            start_frames: [B] absolute latent start index for each sample.

        Returns:
            freqs: [B, f * lat_H * lat_W, 1, head_dim//2]
        """
        self._ensure_freqs_device()
        lat_H = int(self.lat_H if lat_H is None else lat_H)
        lat_W = int(self.lat_W if lat_W is None else lat_W)
        view_count = int(self.config.num_views if num_views is None else num_views)
        frame_seqlen = view_count * lat_H * lat_W
        d = self.wan_dim // self.video_model.wan_model.num_heads
        c = d // 2
        freqs_t, freqs_h, freqs_w = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        start_frames = start_frames.to(device=self.device, dtype=torch.long)
        frame_offsets = torch.arange(f, device=self.device, dtype=torch.long)
        t_freqs = freqs_t[start_frames[:, None] + frame_offsets[None, :]]

        view_freqs = []
        for view_idx in range(view_count):
            if view_count > 1 and self.config.video_view_rope_mode == "hw_offset":
                h_offset = 0
                w_offset = view_idx * lat_W
            else:
                h_offset = 0
                w_offset = 0
            h_end = h_offset + lat_H
            w_end = w_offset + lat_W
            if h_end > freqs_h.shape[0] or w_end > freqs_w.shape[0]:
                raise ValueError(
                    f"video_view_rope_mode={self.config.video_view_rope_mode} needs H/W RoPE "
                    f"up to ({h_end}, {w_end}), but table is ({freqs_h.shape[0]}, {freqs_w.shape[0]})"
                )
            per_view_freqs = torch.cat([
                t_freqs.view(start_frames.shape[0], f, 1, 1, -1).expand(-1, f, lat_H, lat_W, -1),
                freqs_h[h_offset:h_end].view(1, 1, lat_H, 1, -1).expand(start_frames.shape[0], f, lat_H, lat_W, -1),
                freqs_w[w_offset:w_end].view(1, 1, 1, lat_W, -1).expand(start_frames.shape[0], f, lat_H, lat_W, -1),
            ], dim=-1)
            view_freqs.append(per_view_freqs)

        if view_count > 1:
            return torch.stack(view_freqs, dim=2).reshape(start_frames.shape[0], f * frame_seqlen, 1, -1)
        return view_freqs[0].reshape(start_frames.shape[0], f * frame_seqlen, 1, -1)

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
        napf = self.config.action_register_tokens_per_frame
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
                parts.append(
                    self.freqs_action[: (action_len + self.config.action_tokens_per_timestep - 1) // self.config.action_tokens_per_timestep]
                    .repeat_interleave(self.config.action_tokens_per_timestep, dim=0)[:action_len]
                    .view(action_len, 1, -1)
                )
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

    def _create_training_mask(self, L_F, chunk_size, window_size, video_tokens_per_frame: Optional[int] = None):
        """Create sliding-window mask for this training iteration."""
        attention_mode = self.config.mode
        if attention_mode == "loop":
            # Flex block masks are shared across the batch, so loop training samples
            # either IDM or FDM visibility per iteration.
            attention_mode = "fdm" if torch.rand((), device="cpu").item() < 0.5 else "video_idm"
        mask, padded = create_sliding_window_mask(
            num_latent_frames=L_F,
            video_tokens_per_frame=int(video_tokens_per_frame or self.frame_seqlen),
            action_tokens_per_frame=self.config.action_register_tokens_per_frame,
            chunk_size=chunk_size,
            window_size=window_size,
            use_state=self.config.use_state,
            state_tokens_per_frame=self.config.state_tokens_per_frame,
            value_tokens_per_frame=self.config.value_tokens_per_frame,
            attention_mode=attention_mode,
            device=self.device,
        )
        return mask, padded

    def _video_patch_embed(self, latent: torch.Tensor) -> torch.Tensor:
        """Patch-embed video latent -> tokens [B, L, dim]."""
        if latent.dim() == 6:
            B, V, C, T, H, W = latent.shape
            if V != self.config.num_views:
                raise ValueError(f"Expected {self.config.num_views} views, got {V}")
            x = self.video_model.wan_model.patch_embedding(latent.reshape(B * V, C, T, H, W))
            x = x.flatten(2).transpose(1, 2)
            _, lat_h, lat_w = self._latent_grid_from_tensor(latent)
            per_view = lat_h * lat_w
            x = x.view(B, V, T, per_view, -1)
            if hasattr(self, "view_embedding"):
                x = x + self.view_embedding.view(1, V, 1, 1, -1).to(device=x.device, dtype=x.dtype)
            x = x.permute(0, 2, 1, 3, 4).reshape(B, T * V * per_view, -1)
            return x
        x = self.video_model.wan_model.patch_embedding(latent)
        return x.flatten(2).transpose(1, 2)

    def _latent_num_frames(self, latent: torch.Tensor) -> int:
        """Return latent temporal length for [B,C,T,H,W] or [B,V,C,T,H,W]."""
        return int(latent.shape[3] if latent.dim() == 6 else latent.shape[2])

    def _unpatchify_video_tokens(
        self,
        patch_tokens: torch.Tensor,
        num_frames: int,
        batch_size: int,
        lat_H: Optional[int] = None,
        lat_W: Optional[int] = None,
        num_views: Optional[int] = None,
    ) -> torch.Tensor:
        """Unpatchify WAN head output, preserving explicit stereo view dimension."""
        lat_H = int(self.lat_H if lat_H is None else lat_H)
        lat_W = int(self.lat_W if lat_W is None else lat_W)
        num_views = int(self.config.num_views if num_views is None else num_views)
        if num_views == 1:
            grid_size = torch.tensor(
                [[num_frames, lat_H, lat_W]], dtype=torch.long
            ).expand(batch_size, -1)
            video = self.video_model.wan_model.unpatchify(patch_tokens, grid_size)
            return torch.stack(video) if isinstance(video, list) else video

        V = num_views
        per_view = lat_H * lat_W
        expected_len = num_frames * V * per_view
        if patch_tokens.shape[1] != expected_len:
            raise ValueError(
                f"Stereo unpatchify length mismatch: got {patch_tokens.shape[1]}, "
                f"expected {expected_len} for frames={num_frames}, views={V}, per_view={per_view}"
            )
        x = patch_tokens.view(batch_size, num_frames, V, per_view, -1)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * V, num_frames * per_view, -1)
        grid_size = torch.tensor(
            [[num_frames, lat_H, lat_W]], dtype=torch.long
        ).expand(batch_size * V, -1)
        video = self.video_model.wan_model.unpatchify(x, grid_size)
        if isinstance(video, list):
            video = torch.stack(video)
        _, C, T, H, W = video.shape
        return video.view(batch_size, V, C, T, H, W)

    def _step_video_scheduler(self, scheduler, model_output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        """Apply UniPC step while keeping the scheduler's expected channel axis."""
        if sample.dim() == 6:
            B, V, C, T, H, W = sample.shape
            pred = model_output.reshape(B, V * C, T, H, W).transpose(1, 2)
            cur = sample.reshape(B, V * C, T, H, W).transpose(1, 2)
            nxt = scheduler.step(pred, timestep, cur, return_dict=False)[0].transpose(1, 2)
            return nxt.reshape(B, V, C, T, H, W)
        return scheduler.step(
            model_output.transpose(1, 2),
            timestep,
            sample.transpose(1, 2),
            return_dict=False,
        )[0].transpose(1, 2)

    def _video_time_embed(self, timesteps: torch.Tensor, cache_key_extra: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute WAN time embedding + 6-param AdaLN projection (memoized when cache_inference_time_tensors=True)."""
        _nvpush("_video_time_embed")
        # t1v11d: cache by (shape, first_val, last_val, device). Bit-exact iff input has <=2 uniform regions.
        _use_t_cache = (not self.training and
                        bool(getattr(self.config, "cache_inference_time_tensors", False)))
        _t_cache_key = None
        # The key describes row 0 only, so a batch whose rows sit at different
        # noise levels (MBRL scores candidates at per-row t) must not use it.
        if (_use_t_cache and timesteps.dim() == 2 and timesteps.shape[0] >= 1
                and bool((timesteps == timesteps[:1]).all())):
            _row = timesteps[0]
            # Plan Y2: CONTRACT — callers must pass <=2-uniform tensors.
            # All current callsites satisfy this (torch.full / cat([full, full])).
            # Eager-mode assert catches contract violations; dynamo constant-folds
            # this away at trace time so no tensor-bool branch in compiled graph.
            if __debug__ and not torch.compiler.is_compiling():
                assert _row.unique().numel() <= 2, (
                    f"_video_time_embed cache requires <=2-uniform input, got "
                    f"{_row.unique().numel()} unique values. Caller broke contract."
                )
            _t_cache_key = ("v", tuple(timesteps.shape),
                            int(_row[0].item()), int(_row[-1].item()),
                            str(timesteps.device), int(cache_key_extra))
            _cached = self._inference_time_cache.get(_t_cache_key)
            if _cached is not None:
                return _cached
        B, L = timesteps.shape
        t_flat = timesteps.flatten()
        freq_dim = self.video_model.wan_model.freq_dim
        dim = self.wan_dim
        # Use autocast to ensure float32 computation
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat).unflatten(0, (B, L)).float()
            )
            e0 = self.video_model.wan_model.time_projection(e).unflatten(2, (6, dim))
        if _t_cache_key is not None:
            self._inference_time_cache[_t_cache_key] = (e.detach(), e0.detach())
        _nvpop()  # _video_time_embed
        return e, e0

    def _action_time_embed(self, timesteps: torch.Tensor, cache_key_extra: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action time embedding + 6-param projection (memoized when cache_inference_time_tensors=True)."""
        _nvpush("_action_time_embed")
        _use_t_cache = (not self.training and
                        bool(getattr(self.config, "cache_inference_time_tensors", False)))
        _t_cache_key = None
        # The key describes row 0 only, so a batch whose rows sit at different
        # noise levels (MBRL scores candidates at per-row t) must not use it.
        if (_use_t_cache and timesteps.dim() == 2 and timesteps.shape[0] >= 1
                and bool((timesteps == timesteps[:1]).all())):
            _row = timesteps[0]
            # Plan Y2: see _video_time_embed for the contract rationale.
            if __debug__ and not torch.compiler.is_compiling():
                assert _row.unique().numel() <= 2, (
                    f"_action_time_embed cache requires <=2-uniform input, got "
                    f"{_row.unique().numel()} unique values. Caller broke contract."
                )
            _t_cache_key = ("a", tuple(timesteps.shape),
                            int(_row[0].item()), int(_row[-1].item()),
                            str(timesteps.device), int(cache_key_extra))
            _cached = self._inference_time_cache.get(_t_cache_key)
            if _cached is not None:
                return _cached
        B, L = timesteps.shape
        freq_dim = self.action_expert.freq_dim
        dim = self.action_expert.config.dim
        # Use autocast to ensure float32 computation
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(freq_dim, timesteps.flatten()).unflatten(0, (B, L)).float()
            )
            e0 = self.action_expert.time_projection(e).unflatten(2, (6, dim))
        if _t_cache_key is not None:
            self._inference_time_cache[_t_cache_key] = (e.detach(), e0.detach())
        _nvpop()  # _action_time_embed
        return e, e0

    # ====================================================================
    # Plan Z: step_idx-indexed list cache for time embeddings.
    # Replaces dict-cache (keyed by .item()-derived Python ints) which
    # required ~30 .item() calls/chunk for cache key construction.
    # ====================================================================

    def _video_time_embed_compute(self, timesteps: torch.Tensor):
        """Pure compute path for video time embedding (no cache, no .item)."""
        B, L = timesteps.shape
        t_flat = timesteps.flatten()
        freq_dim = self.video_model.wan_model.freq_dim
        dim = self.wan_dim
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat).unflatten(0, (B, L)).float()
            )
            e0 = self.video_model.wan_model.time_projection(e).unflatten(2, (6, dim))
        return e, e0

    def _action_time_embed_compute(self, timesteps: torch.Tensor):
        """Pure compute path for action time embedding (no cache, no .item)."""
        B, L = timesteps.shape
        freq_dim = self.action_expert.freq_dim
        dim = self.action_expert.config.dim
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(freq_dim, timesteps.flatten()).unflatten(0, (B, L)).float()
            )
            e0 = self.action_expert.time_projection(e).unflatten(2, (6, dim))
        return e, e0

    def _expand_training_action_time(
        self,
        noisy_row: torch.Tensor,
        clean_row: torch.Tensor,
        block_latent_counts: List[int],
        rtc_committed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Expand compact time rows into state/action registers, including RTC.

        Expand/cat gives backward a reduction over each repeated region instead
        of an indexed gather's contended scatter into a small embedding bank.
        State and extra action registers retain the sampled action timestep.
        """
        B = noisy_row.shape[0]
        tail = noisy_row.shape[2:]
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        parts = []
        action_offset = 0
        for count in block_latent_counts:
            if spf > 0:
                parts.append(noisy_row.expand(B, spf, *tail))
            registers = noisy_row.expand(B, count * apf, *tail)
            block_len = count * self.config.actions_per_latent
            if rtc_committed_mask is not None:
                committed = torch.zeros(
                    B, count * apf, dtype=torch.bool, device=noisy_row.device,
                )
                committed[:, ::self.config.action_tokens_per_timestep] = (
                    rtc_committed_mask[:, action_offset:action_offset + block_len].bool()
                )
                registers = torch.where(
                    committed.reshape(B, count * apf, *([1] * len(tail))),
                    clean_row,
                    registers,
                )
            parts.append(registers)
            action_offset += block_len
        return torch.cat(parts, dim=1) if parts else noisy_row[:, :0]

    def _dense_training_time_embeddings(
        self,
        noisy_vid_t: torch.Tensor,
        noisy_act_t: torch.Tensor,
        block_latent_counts: List[int],
        half_seq_len: int,
        frame_seqlen: int,
        padded_len: int,
        rtc_committed_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep TF32 GEMM shapes/gradient reductions; skip unused projections.

        Reducing repeated MLP rows changes where TF32 rounds their upstream
        gradients. AdamW can amplify that difference for near-zero gradients.
        This path retains the original linear calls and backward arithmetic;
        only the two projection results with no consumers are removed.
        """
        B = noisy_vid_t.shape[0]
        wan = self.video_model.wan_model
        clean_t = noisy_vid_t.new_zeros(B, half_seq_len)
        noisy_t = noisy_vid_t[:, None].expand(-1, half_seq_len)
        rtc_action_t = (
            None if rtc_committed_mask is None
            else noisy_act_t[:, None] * (1 - rtc_committed_mask)
        )
        action_t = self._build_chunked_action_timesteps(
            noisy_act_t, block_latent_counts, include_value=False,
            action_token_t=rtc_action_t,
        )
        include_value = self.config.value_enabled and self.config.value_tokens_per_frame > 0
        suffix_t = self._build_chunk_ar_timesteps(
            noisy_act_t, noisy_vid_t, block_latent_counts,
            include_value=include_value, video_tokens_per_frame=frame_seqlen,
            action_token_t=rtc_action_t,
        )
        combined_t = torch.cat([clean_t, suffix_t], dim=1)
        if combined_t.shape[1] < padded_len:
            combined_t = torch.cat([
                combined_t, combined_t.new_zeros(B, padded_len - combined_t.shape[1]),
            ], dim=1)
        with torch.autocast(
            device_type=noisy_vid_t.device.type, dtype=torch.float32,
            enabled=noisy_vid_t.is_cuda,
        ):
            video_t = torch.cat([clean_t, noisy_t], dim=1)
            v_emb = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, video_t.flatten())
                .unflatten(0, (B, video_t.shape[1])).float()
            )
            a_emb = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, action_t.flatten())
                .unflatten(0, (B, action_t.shape[1])).float()
            )
            combined_emb = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, combined_t.flatten())
                .unflatten(0, (B, combined_t.shape[1])).float()
            )
            combined_e0 = wan.time_projection(combined_emb).unflatten(2, (6, self.wan_dim))
        return v_emb, a_emb, combined_e0

    def _training_time_embeddings(
        self,
        noisy_vid_t: torch.Tensor,
        noisy_act_t: torch.Tensor,
        block_latent_counts: List[int],
        half_seq_len: int,
        frame_seqlen: int,
        padded_len: int,
        rtc_committed_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Avoid unused projections; compress MLP rows when precision permits.

        This is a per-forward computation, never an inference/optimizer cache.
        TF32 or reduced-precision parameters require the original GEMM shapes
        and separate gradient rounding. FP32 parameters with IEEE GEMMs can
        compute three structural timesteps per sample, then expand
        their pointwise MLP outputs. Clean, value, padding and committed RTC
        tokens use zero. No global precision setting or user toggle is changed.
        """
        wan = self.video_model.wan_model
        # BF16 leaf gradients round at each original MLP call. Combining those
        # calls changes AdamW updates even when the GEMMs themselves use FP32.
        dense_parameters = any(
            parameter.dtype != torch.float32
            for module in (wan.time_embedding, wan.time_projection, self.action_expert.time_embedding)
            for parameter in module.parameters()
        )
        if dense_parameters or (noisy_vid_t.is_cuda and torch.backends.cuda.matmul.allow_tf32):
            return self._dense_training_time_embeddings(
                noisy_vid_t, noisy_act_t, block_latent_counts, half_seq_len,
                frame_seqlen, padded_len, rtc_committed_mask,
            )
        B = noisy_vid_t.shape[0]
        wan = self.video_model.wan_model
        # Keep the original FP64 sinusoid -> FP32 MLP path under outer BF16 AMP.
        with torch.autocast(
            device_type=noisy_vid_t.device.type, dtype=torch.float32,
            enabled=noisy_vid_t.is_cuda,
        ):
            video_t = torch.stack(
                [torch.zeros_like(noisy_vid_t), noisy_vid_t, noisy_act_t], dim=1,
            )
            video_rows = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, video_t.flatten())
                .unflatten(0, (B, 3)).float()
            )
            modulation_rows = wan.time_projection(video_rows).unflatten(2, (6, self.wan_dim))
            action_t = torch.stack([torch.zeros_like(noisy_act_t), noisy_act_t], dim=1)
            action_rows = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, action_t.flatten())
                .unflatten(0, (B, 2)).float()
            )
            # The simple backbone consumes WAN modulation only. Computing
            # action_expert.time_projection here would have no consumer.

        v_emb_full = torch.cat([
            video_rows[:, :1].expand(B, half_seq_len, self.wan_dim),
            video_rows[:, 1:2].expand(B, half_seq_len, self.wan_dim),
        ], dim=1)
        a_emb = self._expand_training_action_time(
            action_rows[:, 1:2], action_rows[:, :1],
            block_latent_counts, rtc_committed_mask,
        )
        action_e0 = self._expand_training_action_time(
            modulation_rows[:, 2:3], modulation_rows[:, :1],
            block_latent_counts, rtc_committed_mask,
        )
        clean_e0 = modulation_rows[:, :1].expand(B, half_seq_len, 6, self.wan_dim)
        noisy_e0 = modulation_rows[:, 1:2].expand(B, half_seq_len, 6, self.wan_dim)
        value_e0 = None
        if self.config.value_enabled and self.config.value_tokens_per_frame > 0:
            value_e0 = modulation_rows[:, :1].expand(
                B, len(block_latent_counts) * self.config.value_tokens_per_frame, 6, self.wan_dim,
            )
        suffix = self._interleave_chunk_ar_tokens(
            action_e0, noisy_e0, value_e0, block_latent_counts,
            video_tokens_per_frame=frame_seqlen,
        )
        parts = [clean_e0, suffix]
        total_len = sum(part.shape[1] for part in parts)
        if total_len < padded_len:
            parts.append(modulation_rows[:, :1].expand(B, padded_len - total_len, 6, self.wan_dim))
        return v_emb_full, a_emb, torch.cat(parts, dim=1)

    def _ensure_step_embeddings(self, sched_video, sched_action,
                                 num_steps: int, B: int,
                                 v_seq_len: int, a_seq_len: int) -> None:
        """Fill list cache (indexed by step_idx) with (e0_combined, v_emb, a_emb).

        All inputs to _video_time_embed_compute / _action_time_embed_compute
        are 0-d → broadcast tensors — no .item() calls.

        Idempotent: skips if cache key (num_steps, B, v_len, a_len) matches.
        """
        cache_key = (
            int(num_steps),
            len(sched_video.timesteps),
            len(sched_action.timesteps),
            int(B),
            int(v_seq_len),
            int(a_seq_len),
            str(self.device),
        )
        if getattr(self, "_step_emb_cache_key", None) == cache_key:
            return
        v_e0_list = []
        v_emb_list = []
        a_emb_list = []
        for step_idx in range(num_steps):
            # 0-d tensors stay as tensors — no .item()
            t_vid_scalar = sched_video.timesteps[step_idx]
            t_act_scalar = (
                sched_action.timesteps[step_idx]
                if step_idx < len(sched_action.timesteps)
                else torch.zeros_like(t_vid_scalar)
            )
            v_t_vals = t_vid_scalar.view(1, 1).expand(B, v_seq_len).to(device=self.device)
            a_t_vals = t_act_scalar.view(1, 1).expand(B, a_seq_len).to(device=self.device)
            combined_t_vals = self._cat_chunk_tokens(a_t_vals, v_t_vals, dim=1)
            _, e0_combined = self._video_time_embed_compute(combined_t_vals)
            v_emb, _ = self._video_time_embed_compute(v_t_vals)
            a_emb, _ = self._action_time_embed_compute(a_t_vals)
            v_e0_list.append(e0_combined.detach())
            v_emb_list.append(v_emb.detach())
            a_emb_list.append(a_emb.detach())
        self._step_emb_cache_key = cache_key
        self._step_emb_cache_v_e0 = v_e0_list
        self._step_emb_cache_v_emb = v_emb_list
        self._step_emb_cache_a_emb = a_emb_list


    def prewarm_streaming_inference_caches(self, batch_size: int = 1,
                                            num_inference_steps=None) -> dict:
        """t1v11f: pre-populate time/rope caches so the first inference call hits.
        Bit-exact (only changes timing). Idempotent across calls.
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_timesteps
        want_time = bool(getattr(self.config, "cache_inference_time_tensors", False))
        want_rope = bool(getattr(self.config, "cache_inference_rope_tensors", False))
        if not (want_time or want_rope):
            return {"time_added": 0, "rope_added": 0}

        K = int(self.config.fixed_chunk_size)
        B = int(batch_size)
        v_seq_len = K * self.lat_H * self.lat_W * self.config.num_views
        action_chunk_len = K * self.config.actions_per_latent
        # spf is the single source of truth (qpos state prefix count).
        state_tokens = self.config.state_tokens_per_frame
        a_seq_len = action_chunk_len + state_tokens

        # Same scheduler setup as inference_step_streaming would use.
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sched_v = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched_a = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched_v.set_timesteps(num_inference_steps, device=self.device,
                              shift=float(getattr(self.config, "video_timestep_shift", 5.0)))
        sched_a.set_timesteps(num_inference_steps, device=self.device,
                              shift=float(getattr(self.config, "action_timestep_shift", 5.0)))
        if self.config.decouple_inference_noise:
            # B-sched: keep sigma_max as 0-d tensor to avoid .item() sync (matches
            # the UniPC branch form in inference_step_streaming).
            sigma_max = sched_v.sigmas[0]
            final_noise = float(self.config.video_inference_final_noise)
            sched_v.sigmas = sched_v.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
            sched_v.timesteps = (sched_v.sigmas[:-1] * 1000).to(torch.int64)

        before_time = len(getattr(self, "_inference_time_cache", {}))
        before_rope = len(getattr(self, "_inference_rope_cache", {}))

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            if want_time:
                for i in range(len(sched_v.timesteps)):
                    t_vid = int(sched_v.timesteps[i].item())
                    t_act = int(sched_a.timesteps[i].item())
                    v_t_vals = torch.full((B, v_seq_len), t_vid, device=self.device, dtype=torch.int64)
                    a_t_vals = torch.full((B, a_seq_len), t_act, device=self.device, dtype=torch.int64)
                    self._video_time_embed(v_t_vals)
                    self._action_time_embed(a_t_vals)
                    combined = self._cat_chunk_tokens(a_t_vals, v_t_vals, dim=1)
                    self._video_time_embed(combined)
            if want_rope:
                # Prewarm rope for start_frame=0 (chunk 0). Subsequent chunks miss
                # by design since start_frame is monotonic.
                self._create_video_freqs(K, start_frame=0)

        added_time = len(getattr(self, "_inference_time_cache", {})) - before_time
        added_rope = len(getattr(self, "_inference_rope_cache", {})) - before_rope
        logger.info("[motus2.accel] prewarm filled time=%d rope=%d entries", added_time, added_rope)
        return {"time_added": added_time, "rope_added": added_rope}


    def _t5_embed(self, language_embeddings) -> torch.Tensor:
        """Process T5 embeddings -> WAN text context."""
        _nvpush("_t5_embed")
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
            # Training pads every caption to text_len with zeros before the
            # collate (data/dataset.py: _process_language_embeddings_batch) and
            # cross-attention always runs with context_lens=None, so the trained
            # regime has text_len keys of which most are padding. A caller that
            # hands over an unpadded tensor changes the softmax normalisation
            # over keys and silently moves every block's cross-attn output.
            if torch.is_tensor(ctx) and ctx.dim() >= 2 and ctx.shape[-2] != text_len:
                if ctx.shape[-2] < text_len:
                    ctx = F.pad(ctx, (0, 0, 0, text_len - ctx.shape[-2]))
                else:
                    ctx = ctx[..., :text_len, :]
        if torch.is_tensor(ctx) and (ctx.dim() < 2 or ctx.shape[-2] != text_len):
            raise ValueError(
                f"T5 context must carry {text_len} tokens, got {tuple(ctx.shape)}"
            )
        ctx_out = self.video_model.wan_model.text_embedding(ctx)
        # t1v11c: optional T5 cross-attn K/V cache (bit-exact)
        if getattr(self.config, "t5_kv_caching", False) and not self.training:
            self._ensure_t5_kv_cache(ctx_out, language_embeddings)
        _nvpop()  # _t5_embed
        return ctx_out

    def _ensure_t5_kv_cache(self, t5_ctx: torch.Tensor, language_embeddings=None) -> None:
        """Refresh once per generated context, then reuse within its denoise loop.

        Do not key by input id: ids can be reused, tensors can be edited in
        place, and reference weights may have changed since the last chunk.
        """
        from .t5_kv_cache import install_t5_kv_cache, populate_t5_kv_cache
        if not getattr(self, "_t5_kv_hooks_installed", False):
            install_t5_kv_cache(self.video_model.wan_model)
            self._t5_kv_hooks_installed = True
        populate_t5_kv_cache(self.video_model.wan_model, t5_ctx)


    # ------------------------------------------------------------------
    # Forward: single WAN layer (training, use_mot=False)
    # ------------------------------------------------------------------

    def _forward_layer_simple(self, x, e0, layer_idx, t5_ctx, ar_mask=None,
                              kv_cache=None, v_seq_len=None,
                              is_training=True, action_register_length=None,
                              half_seq_len=None, video_freqs=None, noisy_freqs=None,
                              full_freqs=None, attention_mode: Optional[str] = None,
                              precomputed_real_freqs=None, training_real_freqs=None,
                              t5_row_gate=None):
        _nvpush(f"layer_{layer_idx}")
        """Forward one WAN layer (use_mot=False). Unified for training and inference.

        Training layout:  [clean_vid | action | noisy_vid | value | pad]
        Inference layout: [action | video]  (query attends to KV cache)

        Args:
            video_freqs: Pre-computed 3D RoPE for video tokens (training: W frames, inference: K frames).
            noisy_freqs: Pre-computed RoPE for noisy_vid + noisy_act (training only).
        """
        blk = self.video_model.wan_model.blocks[layer_idx]
        _nvpush("modulation")
        mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        _nvpop()  # modulation

        _nvpush("norm1")
        norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
        B, L, H_dim = norm_x.shape
        n = self.video_model.wan_model.num_heads
        d = H_dim // n
        _nvpop()  # norm1

        _nvpush("qkv_proj")
        training_qkv = None
        if is_training and full_freqs is not None:
            training_qkv = try_training_qkv_with_rope(
                blk.self_attn, norm_x, training_real_freqs, n
            )
        if training_qkv is None:
            q, k, v = _wan_self_qkv(blk.self_attn, norm_x, B, L, n, d)
        else:
            q, k, v = training_qkv
        _nvpop()  # qkv_proj

        if is_training:
            _nvpush("rope")
            if training_qkv is not None:
                q_full, k_full = q, k
            elif full_freqs is not None:
                q_full = rope_action_apply(q, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
                k_full = rope_action_apply(k, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            else:
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
            _nvpop()  # rope (training)
        else:
            assert v_seq_len is not None
            a_seq_len = L - v_seq_len

            _nvpush("rope")
            if precomputed_real_freqs is not None:
                # F-gamma: real-only multiply with pre-computed (eager) freqs.
                from .rope import _rope_apply_real_only
                q_full = _rope_apply_real_only(q, precomputed_real_freqs).type_as(v)
                k_full = _rope_apply_real_only(k, precomputed_real_freqs).type_as(v)
            elif full_freqs is not None:
                q_full = rope_action_apply(q, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
                k_full = rope_action_apply(k, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            elif v_seq_len == 0:
                # History-only action prediction: sequence is all action tokens (no current video).
                # Must not call _create_video_freqs(0, ...); use the same RoPE path as _forward_action_only_cached.
                if not (action_register_length is not None and a_seq_len > 0):
                    raise ValueError("Inference with v_seq_len=0 requires non-empty action tokens and action_register_length")
                infer_latent_frames = self._fixed_chunk_size()
                action_only_freqs = self.freqs_action.new_empty((0, 1, self.freqs_action.shape[-1]))
                q_full = causal_rope_action_apply(
                    q,
                    action_only_freqs,
                    self.freqs_action,
                    self.freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=infer_latent_frames * self.config.action_register_tokens_per_frame,
                    num_state_per_block=self.config.state_tokens_per_frame,
                    num_value_per_block=0,
                    action_tokens_per_timestep=self.config.action_tokens_per_timestep,
                ).type_as(v)
                k_full = causal_rope_action_apply(
                    k,
                    action_only_freqs,
                    self.freqs_action,
                    self.freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=infer_latent_frames * self.config.action_register_tokens_per_frame,
                    num_state_per_block=self.config.state_tokens_per_frame,
                    num_value_per_block=0,
                    action_tokens_per_timestep=self.config.action_tokens_per_timestep,
                ).type_as(v)
            else:
                infer_latent_frames = v_seq_len // self.frame_seqlen
                cache_seq_len = self._kv_cache_valid_len_int() if kv_cache is not None else 0
                inf_video_freqs = self._create_video_freqs(
                    infer_latent_frames,
                    start_frame=self._inference_video_start_frame(cache_seq_len),
                )

                if action_register_length is not None and a_seq_len > 0:
                    napb = infer_latent_frames * self.config.action_register_tokens_per_frame
                    nspb = self.config.state_tokens_per_frame
                    nvpb = infer_latent_frames * self.config.value_tokens_per_frame
                    roped_q = causal_rope_action_apply(q, inf_video_freqs, self.freqs_action, self.freqs_state,
                                                       action_register_length, napb, nspb,
                                                       num_value_per_block=nvpb,
                                                       action_tokens_per_timestep=self.config.action_tokens_per_timestep).type_as(v)
                    roped_k = causal_rope_action_apply(k, inf_video_freqs, self.freqs_action, self.freqs_state,
                                                       action_register_length, napb, nspb,
                                                       num_value_per_block=nvpb,
                                                       action_tokens_per_timestep=self.config.action_tokens_per_timestep).type_as(v)
                    q_full = torch.cat([roped_q[:, :v_seq_len], roped_q[:, v_seq_len:]], dim=1)
                    k_full = torch.cat([roped_k[:, :v_seq_len], roped_k[:, v_seq_len:]], dim=1)
                else:
                    q_full = rope_action_apply(q, inf_video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
                    k_full = rope_action_apply(k, inf_video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            v_full = v
            _nvpop()  # rope (inference)

        # Attention
        if is_training:
            _nvpush("attn_train_flex")
            y = BLHD_flex_with_mask(
                q_full,
                k_full,
                v_full,
                ar_mask,
            )
            y = y.flatten(2)
            _nvpop()  # attn_train_flex
        else:
            _nvpush("attn_dispatch")
            if kv_cache is not None:
                cached_k, cached_v = kv_cache
                cache_len = cached_k.shape[1]
            else:
                cache_len = 0
            mode = str(attention_mode or "joint").lower()
            visibility = None
            if mode in {"video_idm", "action_only", "fdm"}:
                visibility = self._current_chunk_visibility_mask(
                    mode=mode,
                    query_len=q_full.shape[1],
                    key_len=cache_len + k_full.shape[1],
                    cache_len=cache_len,
                    video_len=int(v_seq_len),
                    device=q_full.device,
                )
            if kv_cache is not None:
                y = self._fixed_cache_sdpa(q_full, cached_k, cached_v, k_full, v_full, visibility=visibility)
                y = y.flatten(2)
            else:
                from wan.modules.attention import flash_attention
                y = flash_attention(q_full, k_full, v_full, k_lens=None)
                y = y.flatten(2)
            _nvpop()  # attn_dispatch

        _nvpush("attn_o")
        y = blk.self_attn.o(y)
        x = x + y * mod[2].squeeze(2)
        _nvpop()  # attn_o
        if t5_ctx is not None:
            _nvpush("cross_attn")
            x = x + gated_cross_attn(blk, x, t5_ctx, t5_row_gate)
            _nvpop()  # cross_attn
        _nvpush("norm2_mod")
        ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
        _nvpop()  # norm2_mod
        _nvpush("ffn")
        x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        _nvpop()  # ffn
        _nvpop()  # layer_{layer_idx}
        return x

    def _forward_layer_mot(self, video_tokens, action_tokens, video_e0, action_e0,
                           layer_idx, t5_ctx, ar_mask=None, kv_cache=None,
                           v_seq_len=None, is_training=True,
                           video_freqs=None, noisy_freqs=None, padded_len=None,
                           value_tokens=None, value_e0=None,
                           block_latent_counts: Optional[List[int]] = None,
                           full_freqs=None, attention_mode: Optional[str] = None, t5_row_gate=None):
        """Forward one layer in MoT mode: joint attention with separate video/action tokens.

        Value tokens (if present) go through video QKV/O/FFN projections.
        The attention sequence follows the non-MoT action-first layout:
        training [clean_video | per-chunk(state/action | noisy_video | value)], inference [action | video].

        Returns:
            video_tokens, action_tokens, value_tokens (value_tokens is None if not enabled)
        """
        blk = self.video_model.wan_model.blocks[layer_idx]
        action_blk = self.action_expert.blocks[layer_idx]

        # Compute AdaLN modulation for video and action separately
        v_mod = (blk.modulation.unsqueeze(0) + video_e0).chunk(6, dim=2)
        a_mod = (action_blk.modulation.unsqueeze(0) + action_e0).chunk(6, dim=2)

        # Pre-attention normalization with AdaLN
        norm_video = (blk.norm1(video_tokens) * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)).to(self.dtype)
        norm_action = (action_blk.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)).to(self.dtype)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Video QKV (WAN space)
        q_v, k_v, v_v = _wan_self_qkv(blk.self_attn, norm_video, B, L_v, n, d)

        # Action QKV: project to WAN head space
        # action_blk.wan_action_qkv: [3, n_heads, action_dim, head_dim]
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_blk.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]  # [B, L_a, n, d]
        a_q = action_blk.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_blk.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Value QKV: use video projections (value tokens are wan_dim)
        L_val = 0
        if value_tokens is not None:
            val_mod = (blk.modulation.unsqueeze(0) + value_e0).chunk(6, dim=2)
            norm_value = (blk.norm1(value_tokens) * (1 + val_mod[1].squeeze(2)) + val_mod[0].squeeze(2)).to(self.dtype)
            L_val = norm_value.shape[1]
            q_val, k_val, v_val = _wan_self_qkv(blk.self_attn, norm_value, B, L_val, n, d)

        counts = block_latent_counts or [self._fixed_chunk_size()]
        spf = self.config.state_tokens_per_frame
        apf = self.config.action_register_tokens_per_frame
        nvpf = self.config.value_tokens_per_frame if value_tokens is not None else 0
        action_register_length = L_a if L_a > 0 else None

        # Apply RoPE using rope_action_apply (unified for video and action)
        if is_training:
            half_seq_len = L_v // 2
            q_clean, k_clean, v_clean = q_v[:, :half_seq_len], k_v[:, :half_seq_len], v_v[:, :half_seq_len]
            q_noisy_video = q_v[:, half_seq_len:]
            k_noisy_video = k_v[:, half_seq_len:]
            v_noisy_video = v_v[:, half_seq_len:]

            video_tokens_per_frame = half_seq_len // max(1, sum(counts))
            q_parts = [q_clean]
            k_parts = [k_clean]
            v_parts = [v_clean]
            action_offset = 0
            video_offset = 0
            value_offset = 0
            for count in counts:
                action_len = spf + count * apf
                video_len = count * video_tokens_per_frame
                if action_len > 0:
                    q_parts.append(a_q[:, action_offset:action_offset + action_len])
                    k_parts.append(a_k[:, action_offset:action_offset + action_len])
                    v_parts.append(a_v[:, action_offset:action_offset + action_len])
                q_parts.append(q_noisy_video[:, video_offset:video_offset + video_len])
                k_parts.append(k_noisy_video[:, video_offset:video_offset + video_len])
                v_parts.append(v_noisy_video[:, video_offset:video_offset + video_len])
                if nvpf > 0:
                    q_parts.append(q_val[:, value_offset:value_offset + nvpf])
                    k_parts.append(k_val[:, value_offset:value_offset + nvpf])
                    v_parts.append(v_val[:, value_offset:value_offset + nvpf])
                    value_offset += nvpf
                action_offset += action_len
                video_offset += video_len

            q_full = torch.cat(q_parts, dim=1)
            k_full = torch.cat(k_parts, dim=1)
            v_full = torch.cat(v_parts, dim=1)
            if full_freqs is None:
                raise ValueError("MoT action-first training requires full_freqs")
            q_full = rope_action_apply(q_full, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v_full)
            k_full = rope_action_apply(k_full, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v_full)
        else:
            assert v_seq_len is not None, "v_seq_len must be provided for inference"
            infer_latent_frames = v_seq_len // self.frame_seqlen
            cache_seq_len = self._kv_cache_valid_len_int() if kv_cache is not None else 0
            q_full = torch.cat([a_q, q_v], dim=1)
            k_full = torch.cat([a_k, k_v], dim=1)
            v_full = torch.cat([a_v, v_v], dim=1)
            if full_freqs is None:
                video_freqs = self._create_video_freqs(
                    infer_latent_frames,
                    start_frame=self._inference_video_start_frame(cache_seq_len),
                )
                full_freqs = self._build_chunk_ar_freqs(
                    clean_video_freqs=q_v.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                    noisy_video_freqs=video_freqs,
                    block_latent_counts=[infer_latent_frames],
                    include_value=False,
                )
            q_full = rope_action_apply(q_full, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v_full)
            k_full = rope_action_apply(k_full, full_freqs, self.freqs_action, self.freqs_state, None).type_as(v_full)

        # Attention: training uses flex_attention with mask, inference uses flash_attention with KV cache
        if is_training:
            # Training: flex_attention with AR mask (mask expects padded_len)
            if padded_len is not None and q_full.shape[1] < padded_len:
                pad_n = padded_len - q_full.shape[1]
                q_full = torch.cat([q_full, q_full.new_zeros(B, pad_n, n, d)], dim=1)
                k_full = torch.cat([k_full, k_full.new_zeros(B, pad_n, n, d)], dim=1)
                v_full = torch.cat([v_full, v_full.new_zeros(B, pad_n, n, d)], dim=1)
            attn_out = BLHD_flex_with_mask(
                q_full,
                k_full,
                v_full,
                ar_mask,
            )
            attn_out = attn_out.flatten(2)
        else:
            # Inference: flash_attention with KV cache
            visibility = None
            if kv_cache is not None:
                cached_k, cached_v = kv_cache
                mode = str(attention_mode or "joint").lower()
                if mode in {"video_idm", "action_only", "fdm"}:
                    cache_len = cached_k.shape[1]
                    visibility = self._current_chunk_visibility_mask(
                        mode=mode,
                        query_len=q_full.shape[1],
                        key_len=cache_len + k_full.shape[1],
                        cache_len=cache_len,
                        video_len=int(v_seq_len),
                        device=q_full.device,
                    )
                attn_out = self._fixed_cache_sdpa(q_full, cached_k, cached_v, k_full, v_full, visibility=visibility)
            else:
                from wan.modules.attention import flash_attention
                attn_out = flash_attention(q_full, k_full, v_full, k_lens=None)
            attn_out = attn_out.flatten(2)

        # Split outputs back to the separate video/action/value streams.
        value_out_h = None
        if is_training:
            half_seq_len = L_v // 2
            clean_video_out_h = attn_out[:, :half_seq_len]
            offset = half_seq_len
            video_tokens_per_frame = half_seq_len // max(1, sum(counts))
            action_parts = []
            noisy_video_parts = []
            value_parts = []
            for count in counts:
                action_len = spf + count * apf
                video_len = count * video_tokens_per_frame
                if action_len > 0:
                    action_parts.append(attn_out[:, offset:offset + action_len])
                offset += action_len
                noisy_video_parts.append(attn_out[:, offset:offset + video_len])
                offset += video_len
                if value_tokens is not None:
                    value_parts.append(attn_out[:, offset:offset + nvpf])
                    offset += nvpf
            action_out_h = torch.cat(action_parts, dim=1) if action_parts else attn_out[:, :0]
            noisy_video_out_h = torch.cat(noisy_video_parts, dim=1) if noisy_video_parts else attn_out[:, :0]
            video_out_h = torch.cat([clean_video_out_h, noisy_video_out_h], dim=1)
            if value_tokens is not None:
                value_out_h = torch.cat(value_parts, dim=1) if value_parts else attn_out[:, :0]
        video_out_h = attn_out[:, :L_v]
        action_out_h = attn_out[:, L_v:L_v + L_a]
        if value_tokens is not None:
            value_out_h = attn_out[:, L_v + L_a:L_v + L_a + L_val]

        # Project video output
        video_out = blk.self_attn.o(video_out_h)
        video_tokens = video_tokens + video_out * v_mod[2].squeeze(2)

        # Project value output through video O projection
        if value_tokens is not None:
            value_out = blk.self_attn.o(value_out_h)
            value_tokens = value_tokens + value_out * val_mod[2].squeeze(2)

        # Project action output back to action space
        action_out = action_blk.wan_action_o(action_out_h)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)

        # Cross-attention (video only; value tokens skip cross-attn)
        if t5_ctx is not None:
            video_tokens = video_tokens + gated_cross_attn(blk, video_tokens, t5_ctx, t5_row_gate)

        # FFN: separate for video, value, and action
        # Video FFN
        ffn_v_in = (blk.norm2(video_tokens) * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)).to(self.dtype)
        video_tokens = video_tokens + blk.ffn(ffn_v_in) * v_mod[5].squeeze(2)

        # Value FFN (through video FFN)
        if value_tokens is not None:
            ffn_val_in = (blk.norm2(value_tokens) * (1 + val_mod[4].squeeze(2)) + val_mod[3].squeeze(2)).to(self.dtype)
            value_tokens = value_tokens + blk.ffn(ffn_val_in) * val_mod[5].squeeze(2)

        # Action FFN
        ffn_a_in = (action_blk.norm2(action_tokens) * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)).to(self.dtype)
        action_tokens = action_tokens + action_blk.ffn(ffn_a_in) * a_mod[5].squeeze(2)

        return video_tokens.to(self.dtype), action_tokens.to(self.dtype), (value_tokens.to(self.dtype) if value_tokens is not None else None)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _rtc_committed_action_mask(
        self,
        batch: int,
        block_action_token_lengths: List[int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Pick which action tokens are a committed prefix of their own chunk.

        Returns the [B, total_action_tokens] mask (1.0 on committed tokens) and the
        [B] bool of which samples were selected. The prefix is taken per chunk, not
        per window: at deployment the commitment belongs to the chunk currently
        being generated, and every subsequent chunk faces the same situation. A
        chunk shorter than the prefix (the ragged tail of a window) is skipped
        rather than blanked, so no chunk is ever left without supervision.
        """
        selected = torch.zeros(batch, device=device, dtype=torch.bool)
        committed_steps = int(self.config.rtc_committed_steps)
        total = int(sum(block_action_token_lengths))
        if not self.config.rtc_training_enabled or committed_steps <= 0 or total == 0:
            return None, selected
        selected = torch.rand(batch, device=device) < float(self.config.rtc_training_prob)
        positions = torch.zeros(total, device=device, dtype=self.dtype)
        offset = 0
        for block_len in block_action_token_lengths:
            if block_len > committed_steps:
                positions[offset:offset + committed_steps] = 1.0
            offset += block_len
        return selected.to(self.dtype).unsqueeze(1) * positions.unsqueeze(0), selected

    def _build_value_on_model_scheduler(self, device):
        """Same sampler the MBRL rollout builds, selected the same way.

        _joint_policy_video_rollout_step is the path that runs: every MBRL config
        sets rollout_pipeline to joint and the default is joint too. It picks
        euler or unipc off the scheduler name, so this does as well -- the two
        disagree about where the sigmas sit, and generating on one grid to score
        frames produced on the other would reintroduce the gap this is closing.
        """
        steps = max(1, int(self.config.value_on_model_steps))
        shift = float(self.config.video_timestep_shift)
        if str(self.config.value_on_model_scheduler).lower() == "euler":
            return _FlowEulerScheduler(
                steps, device=device, shift=shift, final_sigma=0.0,
            )
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(steps, device=device, shift=shift)
        return scheduler

    @torch.no_grad()
    def _rollout_vm_future_latents(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        language_embeddings,
        valid_len: Optional[torch.Tensor],
        latent_start: Optional[torch.Tensor],
        vm_idx: torch.Tensor,
        chunk_size: int,
        window_size: int,
    ) -> torch.Tensor:
        """Overwrite the VM samples' future latents with the WM's own rollout.

        One chunk at a time, committing each back into the latents before the
        next is drawn. At deployment chunk m is generated from a history that
        already holds the generated chunk m-1; generating all chunks against the
        real history instead would hand the value head a future contradicting
        its own past, which is not a state the planner can ever reach.
        """
        cs = int(chunk_size)
        is6d = latents.dim() == 6
        W = latents.shape[3] if is6d else latents.shape[2]
        N = W - cs
        if N <= 0 or vm_idx.numel() == 0:
            return latents

        def _sub(t):
            if t is None:
                return None
            if isinstance(t, list):
                return [t[int(i)] for i in vm_idx.tolist()]
            return t[vm_idx]

        out = latents.clone()
        sub_actions = _sub(actions)
        sub_states = _sub(states)
        sub_lang = _sub(language_embeddings)
        sub_valid = _sub(valid_len)
        sub_start = _sub(latent_start)

        offset = 0
        for count in self._chunk_latent_counts(N, cs):
            lo, hi = cs + offset, cs + offset + count

            scheduler = self._build_value_on_model_scheduler(latents.device)
            if getattr(self.config, "decouple_inference_noise", False):
                sigma_max = scheduler.sigmas[0]
                final_noise = float(self.config.video_inference_final_noise)
                scheduler.sigmas = (
                    scheduler.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
                )
                scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64)

            noisy_region = out[vm_idx][:, :, :, cs:] if is6d else out[vm_idx][:, :, cs:]
            noisy = torch.randn_like(noisy_region.to(self.dtype))
            for step_idx, timestep in enumerate(scheduler.timesteps):
                velocity = self.training_step(
                    latents=out[vm_idx],
                    actions=sub_actions,
                    states=sub_states,
                    language_embeddings=sub_lang,
                    valid_len=sub_valid,
                    latent_start=sub_start,
                    _gen_video_only=True,
                    _gen_window_size=window_size,
                    _gen_sigma_v=scheduler.sigmas[step_idx].reshape(1),
                    _gen_noisy_video=noisy,
                )["video_velocity"]
                noisy = self._step_video_scheduler(scheduler, velocity, timestep, noisy)

            # Only this chunk is trusted: the others were denoised against a
            # history that has not been rolled forward to them yet.
            if is6d:
                out[vm_idx, :, :, lo:hi] = noisy[:, :, :, offset:offset + count].to(out.dtype)
            else:
                out[vm_idx, :, lo:hi] = noisy[:, :, offset:offset + count].to(out.dtype)
            offset += count

        return out

    def training_step(
        self,
        latents: torch.Tensor,            # [B, C, W, H_lat, W_lat] pre-computed VAE latents
        actions: torch.Tensor,             # [B, W * actions_per_latent, action_dim]
        states: torch.Tensor = None,       # [B, W, state_dim]
        language_embeddings=None,
        valid_len: torch.Tensor = None,    # [B] int, actual latent frames before padding
        latent_start: torch.Tensor = None, # [B] int, absolute start index in phase latent trajectory
        values: torch.Tensor = None,       # [B, W] MC returns per frame
        sample_class: torch.Tensor = None,  # [B] SampleClass code
        global_step: int = 0,
        vm_observations: Optional[Dict[str, torch.Tensor]] = None,
        action_valid_mask: torch.Tensor = None,  # [B, W * actions_per_latent] real action rows
        video_valid_mask: torch.Tensor = None,   # [B, W] latents with a real video target
        # --- World-model generation for value_on_model ---
        # Set only by _rollout_vm_future_latents. They turn this method into a
        # single velocity evaluation: every sample is forced to WM mode, no loss
        # is built, and the video velocity is returned. Keeping it inside this
        # method is what makes the generated latents come from the same masked
        # attention the training pass uses, rather than a second implementation.
        _gen_video_only: bool = False,
        _gen_window_size: Optional[int] = None,
        _gen_sigma_v: Optional[torch.Tensor] = None,
        _gen_noisy_video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sliding-window AR training step with pre-computed latents."""
        B = latents.shape[0]
        W = latents.shape[3] if latents.dim() == 6 else latents.shape[2]  # latent frames in window
        device = latents.device
        apl = self.config.actions_per_latent
        nvpf = self.config.value_tokens_per_frame  # K value tokens per chunk
        num_views, lat_H, lat_W, per_view_frame_seqlen, frame_seqlen = self._video_grid_from_latent(latents)

        def _debug_model_forward_enabled() -> bool:
            raw = os.environ.get("MOTUS2_DEBUG_STEP_RANGE")
            if not raw:
                return False
            text = str(raw).strip()
            try:
                if ":" in text:
                    start_s, end_s = text.split(":", 1)
                    start, end = int(start_s), int(end_s)
                else:
                    start = end = int(text)
            except ValueError:
                return False
            if end < start:
                start, end = end, start
            step = int(global_step) + 1
            return start <= step <= end

        debug_model_forward = _debug_model_forward_enabled()

        def _dbg(stage: str) -> None:
            if not debug_model_forward:
                return
            if os.environ.get("MOTUS2_DEBUG_MODEL_SYNC", "1").lower() not in {"0", "false", "no", "off"} and torch.cuda.is_available():
                torch.cuda.synchronize()
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            logger.info("[debug-model step %s R%s] %s", int(global_step) + 1, rank, stage)

        _dbg(
            "start "
            f"latents={tuple(latents.shape)} actions={tuple(actions.shape)} "
            f"B={B} W={W} grid=({num_views},{lat_H},{lat_W}) frame_seqlen={frame_seqlen}"
        )

        # === 1. Fixed chunk size + random chunk-window for this iteration ===
        import random as _random
        chunk_size = self._fixed_chunk_size()
        # The generation passes must see the same visible-chunk window as the
        # training pass that will score their output; drawing a fresh one would
        # generate from a different context than the value head then reads.
        window_size = (
            int(_gen_window_size)
            if _gen_window_size is not None
            else _random.randint(*self.config.chunk_window_range)
        )

        # === 2. Three-way mode selection: Policy / WM / Value ===
        use_wm = torch.zeros(B, dtype=torch.bool, device=device)
        use_vm = torch.zeros(B, dtype=torch.bool, device=device)
        sample_class = (
            sample_class.to(device=device)
            if sample_class is not None
            else torch.zeros(B, dtype=torch.long, device=device)
        )
        # Every class above MAIN is off the policy loss, so one comparison
        # answers "is this policy-trainable" without enumerating the classes.
        off_policy_mask = sample_class > SampleClass.MAIN
        # Sub-optimal episodes reach the goal but by a path not worth imitating.
        # They are already excluded from the policy by the line above; the only
        # thing left to decide is their WM/VM split. Their value target is the
        # success curve and is fixed dataset-side.
        suboptimal_mask = sample_class == SampleClass.SUBOPTIMAL
        off_policy_start_step = (
            int(self.config.off_policy_start_step)
            if self.config.off_policy_start_step is not None
            else max(
                int(self.config.warmup_steps),
                int(self.config.uniform_loss_weight_warmup_steps),
            )
        )
        off_policy_active = off_policy_mask & (global_step >= off_policy_start_step)
        off_policy_pending = off_policy_mask & ~off_policy_active
        main_mask = ~off_policy_mask

        rand_mode = torch.rand(B, device=device)
        if self.config.value_enabled and (
            self.config.vm_ratio > 0 or self.config.fail_vm_share is not None
            or self.config.random_vm_share is not None
        ):
            main_use_vm = main_mask & (rand_mode < self.config.vm_ratio)
            main_use_wm = main_mask & ~main_use_vm & (rand_mode < self.config.vm_ratio + self.config.wm_ratio)
            use_vm = use_vm | main_use_vm
            use_wm = use_wm | main_use_wm

            wm_mass = max(float(self.config.wm_ratio), 0.0)
            vm_mass = max(float(self.config.vm_ratio), 0.0)
            total_mass = wm_mass + vm_mass
            if total_mass > 0 or self.config.fail_vm_share is not None or self.config.random_vm_share is not None:
                irr_rand = torch.rand(B, device=device)
                irr_vm_share = vm_mass / total_mass if total_mass > 0 else 0.0
                if self.config.fail_vm_share is not None:
                    irr_vm_share = torch.where(
                        sample_class == SampleClass.TASK_IRRELEVANT,
                        torch.full_like(irr_rand, float(self.config.fail_vm_share)),
                        torch.full_like(irr_rand, irr_vm_share),
                    )
                if self.config.suboptimal_vm_share is not None:
                    irr_vm_share = torch.where(
                        suboptimal_mask,
                        torch.full_like(irr_rand, float(self.config.suboptimal_vm_share)),
                        torch.as_tensor(irr_vm_share, dtype=irr_rand.dtype, device=device).expand_as(irr_rand),
                    )
                if self.config.random_vm_share is not None:
                    irr_vm_share = torch.where(
                        sample_class == SampleClass.RANDOM,
                        torch.full_like(irr_rand, float(self.config.random_vm_share)),
                        torch.as_tensor(irr_vm_share, dtype=irr_rand.dtype, device=device).expand_as(irr_rand),
                    )
                irr_use_vm = off_policy_active & (irr_rand < irr_vm_share)
                if self.config.random_vm_share is not None and int(global_step) % 100 == 0:
                    is_random = sample_class == SampleClass.RANDOM
                    logger.info(
                        "[pool-split step %s] random wm=%d vm=%d | other off-policy wm=%d vm=%d",
                        int(global_step) + 1,
                        int((off_policy_active & is_random & ~irr_use_vm).sum()),
                        int((is_random & irr_use_vm).sum()),
                        int((off_policy_active & ~is_random & ~irr_use_vm).sum()),
                        int((~is_random & irr_use_vm).sum()),
                    )
                use_vm = use_vm | irr_use_vm
                use_wm = use_wm | (off_policy_active & ~irr_use_vm)
            else:
                use_wm = use_wm | off_policy_active
        else:
            use_wm = use_wm | (main_mask & (rand_mode < self.config.wm_ratio))
            use_wm = use_wm | off_policy_active

        vm_idx = use_vm.nonzero(as_tuple=True)[0]
        wm_idx = use_wm.nonzero(as_tuple=True)[0]
        pol_idx = (~off_policy_pending & ~use_wm & ~use_vm).nonzero(as_tuple=True)[0]

        if _gen_video_only:
            # One velocity evaluation of "clean action -> predict video" for every
            # sample handed in. No value target, no policy target, and no action
            # perturbation: the caller wants what the WM would render, not a
            # training signal.
            use_wm = torch.ones(B, dtype=torch.bool, device=device)
            use_vm = torch.zeros(B, dtype=torch.bool, device=device)
            vm_idx = use_vm.nonzero(as_tuple=True)[0]
            wm_idx = use_wm.nonzero(as_tuple=True)[0]
            pol_idx = vm_idx

        if self.config.value_temporal_sampling != "none" and not _gen_video_only:
            from data.vm_temporal_sampling import replace_vm_inputs
            latents, actions, states, values, valid_len, latent_start = replace_vm_inputs(
                latents, actions, states, values, valid_len, latent_start, vm_idx, vm_observations,
            )
            if len(vm_idx) > 0 and (debug_model_forward or int(global_step) % 100 == 0):
                logger.info("[vm-temporal step %s] source_latent_indices=%s", int(global_step) + 1,
                            vm_observations["source_indices"].index_select(0, vm_idx).tolist())

        # On-model value: the frames the value head scores become the frames the
        # WM renders. Full replacement, not a mixture -- the clean future is the
        # distribution we are trying to stop training on. Generate before the
        # clean/noisy split, so the generated frames also become
        # the visible history of the chunks that follow them.
        if (
            getattr(self.config, "value_on_model", False)
            and not _gen_video_only
            and len(vm_idx) > 0
        ):
            latents = self._rollout_vm_future_latents(
                latents=latents,
                actions=actions,
                states=states,
                language_embeddings=language_embeddings,
                valid_len=valid_len,
                latent_start=latent_start,
                vm_idx=vm_idx,
                chunk_size=chunk_size,
                window_size=window_size,
            ).detach()

        num_suboptimal = int(suboptimal_mask.sum())
        num_suboptimal_vm = int((suboptimal_mask & use_vm).sum())
        _dbg(
            f"mode selected pol={len(pol_idx)} wm={len(wm_idx)} vm={len(vm_idx)} "
            f"subopt={num_suboptimal}(vm={num_suboptimal_vm}) "
            f"chunk={chunk_size} window={window_size}"
        )

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

        if _gen_video_only:
            # The caller owns the denoise schedule, so it owns sigma too.
            sigma_v = _gen_sigma_v.to(device=device, dtype=self.dtype).expand(B).clone()
            sigma_a = torch.zeros(B, device=device, dtype=self.dtype)

        # === 3. Split: clean 0..W-cs-1, noisy cs..W-1 ===
        cs = chunk_size
        N = W - cs  # frames per side (observation = first cs frames)
        if latents.dim() == 6:
            clean_latents = latents[:, :, :, :N].to(self.dtype)       # [B, V, C, N, H, W]
            noisy_latents = latents[:, :, :, cs:].to(self.dtype)      # [B, V, C, N, H, W]
        else:
            clean_latents = latents[:, :, :N].to(self.dtype)          # [B, C, N, H, W]
            noisy_latents = latents[:, :, cs:].to(self.dtype)         # [B, C, N, H, W]
        action_block_info = self._prepare_noisy_action_blocks(
            actions, states, cs, dtype=self.dtype
        )
        noisy_actions = action_block_info["flat_actions"]
        noisy_states = action_block_info["states"]
        block_latent_counts = action_block_info["block_latent_counts"]
        block_action_token_lengths = action_block_info["block_action_token_lengths"]
        num_noisy_chunks = len(block_latent_counts)
        target_actions_raw = noisy_actions.clone()

        _dbg(
            "split/noisy prepared "
            f"clean={tuple(clean_latents.shape)} noisy_latents={tuple(noisy_latents.shape)} "
            f"noisy_actions={tuple(noisy_actions.shape)} blocks={block_latent_counts}"
        )

        # === 3.5 Noisy history augmentation (per-frame independent) ===
        aug_sigma_per_frame = torch.zeros(B, N, device=device, dtype=self.dtype)
        # Generation reproduces deployment, where history is never augmented.
        if self.config.noisy_history_sigma_max > 0 and not _gen_video_only:
            aug_mask = torch.rand(B, N, device=device) < self.config.noisy_history_prob
            rand_sigma = torch.rand(B, N, device=device, dtype=self.dtype) * self.config.noisy_history_sigma_max
            aug_sigma_per_frame = torch.where(aug_mask, rand_sigma, aug_sigma_per_frame)
            noise_clean = torch.randn_like(clean_latents)
            s = aug_sigma_per_frame.view(B, 1, 1, N, 1, 1) if clean_latents.dim() == 6 else aug_sigma_per_frame.view(B, 1, N, 1, 1)
            clean_latents = (1 - s) * clean_latents + s * noise_clean

        # === 4. Noise + interpolate (noisy side only) ===
        noise_v = torch.randn_like(noisy_latents)
        sv = sigma_v.view(B, 1, 1, 1, 1, 1) if noisy_latents.dim() == 6 else sigma_v.view(B, 1, 1, 1, 1)
        noisy_video = (1 - sv) * noisy_latents + sv * noise_v
        target_video = noise_v - noisy_latents
        if _gen_video_only:
            # Mid-trajectory state of the caller's denoise loop. Interpolating
            # toward the real latent here would feed the answer back in and the
            # rollout would converge on the ground truth it is meant to replace.
            noisy_video = _gen_noisy_video.to(device=device, dtype=self.dtype)

        noise_a = torch.randn_like(noisy_actions)
        sa = sigma_a.view(B, 1, 1)
        rtc_committed_mask, rtc_selected = self._rtc_committed_action_mask(
            B, block_action_token_lengths, device
        )
        # Committed actions are handed to the model exactly as the previous chunk
        # emitted them: sigma 0 here, timestep 0 below, no gradient in the loss.
        sa_tokens = sa if rtc_committed_mask is None else sa * (1 - rtc_committed_mask).unsqueeze(-1)
        noisy_action = (1 - sa_tokens) * noisy_actions + sa_tokens * noise_a
        target_action = noise_a - target_actions_raw

        # === 5. Token embedding ===
        clean_video_tokens = self._video_patch_embed(clean_latents)
        noisy_video_tokens = self._video_patch_embed(noisy_video)
        half_seq_len = clean_video_tokens.shape[1]
        _dbg(f"video patch embed clean_tokens={tuple(clean_video_tokens.shape)} noisy_tokens={tuple(noisy_video_tokens.shape)}")

        noisy_action_chunks = self._split_flat_action_chunks(noisy_action, block_action_token_lengths)
        action_tokens = self._encode_action_chunk_tokens(noisy_action_chunks, noisy_states)
        sa_action_seq_len = action_tokens.shape[1]
        _dbg(f"action encode action_tokens={tuple(action_tokens.shape)}")

        if not self.config.use_mot:
            action_tokens = self.action_expert.projection.project_in(action_tokens)
            # action_tokens: [B, N*(state+action), wan_dim]
            _dbg(f"action project_in action_tokens={tuple(action_tokens.shape)}")

        # === 5b. Value token construction ===
        value_hidden = None  # will be set after transformer if value_enabled
        value_query = None   # separate tensor for use_mot=True
        chunk_ar_tokens = None
        if self.config.value_enabled and nvpf > 0:
            # Generate chunk-level value query tokens: [B, num_chunks*K, wan_dim]
            value_query = self.value_tokens.expand(B, num_noisy_chunks)

            # use_mot=True: value_query stays separate (wan_dim), will be passed to _forward_layer_mot

        action_seq_len = action_tokens.shape[1]

        # === 6. Build sequence ===
        # use_mot=False: [clean_vid | action | noisy_vid | value] all in WAN dim
        # use_mot=True: action stays in action_expert dim (1024), only concat video tokens
        if not self.config.use_mot:
            chunk_ar_tokens = self._interleave_chunk_ar_tokens(
                action_tokens,
                noisy_video_tokens,
                value_query,
                block_latent_counts,
                video_tokens_per_frame=frame_seqlen,
            )
            x = torch.cat([clean_video_tokens, chunk_ar_tokens], dim=1)
        else:
            x = torch.cat([clean_video_tokens, noisy_video_tokens], dim=1)
        total_seq_len = x.shape[1]
        _dbg(f"sequence built x={tuple(x.shape)} total_seq_len={total_seq_len} action_seq_len={action_seq_len}")

        # === 7. Timesteps (derived from sigma, WAN convention: timestep = sigma * T) ===
        noisy_vid_t = (sigma_v * self.scheduler.num_train_timesteps).to(self.dtype)
        noisy_act_t = (sigma_a * self.action_scheduler.num_train_timesteps).to(self.dtype)

        # === 8. Create sliding-window mask ===
        ar_mask, padded_len = self._create_training_mask(
            W,
            chunk_size,
            window_size,
            video_tokens_per_frame=frame_seqlen,
        )

        # === 9. Time embeddings ===
        val_e0 = None
        if not self.config.use_mot:
            v_emb_full, a_emb, combined_e0 = self._training_time_embeddings(
                noisy_vid_t, noisy_act_t, block_latent_counts, half_seq_len,
                frame_seqlen, padded_len, rtc_committed_mask,
            )
            if total_seq_len < padded_len:
                pad_len = padded_len - total_seq_len
                x = torch.cat([x, x.new_zeros(B, pad_len, x.shape[2])], dim=1)
        else:
            # MoT has separate video/action modulation consumers. Preserve its
            # existing compute path, including per-token RTC conditioning.
            clean_t_vals = torch.zeros(B, half_seq_len, dtype=self.dtype, device=device)
            noisy_v_t_vals = noisy_vid_t.view(B, 1).expand(-1, half_seq_len)
            v_emb_full, v_e0 = self._video_time_embed(torch.cat([clean_t_vals, noisy_v_t_vals], dim=1))
            rtc_action_token_t = (
                None if rtc_committed_mask is None
                else noisy_act_t.view(B, 1) * (1 - rtc_committed_mask)
            )
            sa_t_vals = self._build_chunked_action_timesteps(
                noisy_act_t, block_latent_counts, include_value=False,
                action_token_t=rtc_action_token_t,
            )
            a_emb, a_e0 = self._action_time_embed(sa_t_vals)
            if value_query is not None:
                value_t_vals = torch.zeros(B, num_noisy_chunks * nvpf, dtype=self.dtype, device=device)
                _, val_e0 = self._video_time_embed(value_t_vals)
        _dbg(f"mask/time prepared padded_len={padded_len}")

        # === 10. RoPE ===
        if self._video_rope_is_global() and latent_start is not None:
            # Global absolute RoPE training: align train coordinates with inference accumulation.
            latent_start = latent_start.to(device=device, dtype=torch.long)
            clean_start = latent_start
            noisy_start = latent_start + cs
            clean_video_freqs = self._create_video_freqs_batch(
                N, clean_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )
            noisy_video_freqs = self._create_video_freqs_batch(
                N, noisy_start, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )
            noisy_freqs = None
            full_freqs = self._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=(self.config.value_enabled and nvpf > 0),
                video_tokens_per_frame=frame_seqlen,
            )
        else:
            clean_video_freqs = self._create_video_freqs(
                N, start_frame=0, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )    # positions 0..W-cs-1
            noisy_video_freqs = self._create_video_freqs(
                N, start_frame=cs, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )   # positions cs..W-1
            noisy_freqs = None
            full_freqs = self._build_chunk_ar_freqs(
                clean_video_freqs,
                noisy_video_freqs,
                block_latent_counts,
                include_value=(self.config.value_enabled and nvpf > 0),
                video_tokens_per_frame=frame_seqlen,
            )

        training_real_freqs = None
        if not self.config.use_mot:
            training_real_freqs = prepare_training_qk_frequencies(full_freqs, B, padded_len)

        # === 11. T5 context ===
        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None
        _dbg(f"rope/t5 prepared t5_ctx={'none' if t5_ctx is None else tuple(t5_ctx.shape)}")
        t5_row_gate = (
            wm_language_row_gate(use_wm, t5_ctx)
            if self.config.wm_mask_language and len(wm_idx) > 0
            else None
        )

        # === 12. Forward through layers ===
        action_register_length = action_seq_len if action_seq_len > 0 else None
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            if not self.config.use_mot:
                for layer_idx in range(self.config.num_layers):
                    _dbg(f"layer {layer_idx} begin simple x={tuple(x.shape)}")
                    if torch.is_grad_enabled() and self.video_model.wan_model.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(
                            self._forward_layer_simple, x, combined_e0, layer_idx, t5_ctx, ar_mask,
                            None, None, True, action_register_length, half_seq_len,
                            clean_video_freqs, noisy_freqs, full_freqs,
                            training_real_freqs=training_real_freqs,
                            t5_row_gate=t5_row_gate,
                            use_reentrant=False,
                        )
                    else:
                        x = self._forward_layer_simple(
                            x, combined_e0, layer_idx, t5_ctx, ar_mask,
                            is_training=True, action_register_length=action_register_length,
                            half_seq_len=half_seq_len, video_freqs=clean_video_freqs,
                            noisy_freqs=noisy_freqs, full_freqs=full_freqs,
                            training_real_freqs=training_real_freqs,
                            t5_row_gate=t5_row_gate,
                        )
                    _dbg(f"layer {layer_idx} end simple x={tuple(x.shape)}")
            else:
                video_tokens = x
                mot_action_tokens = action_tokens
                mot_value_tokens = value_query  # None if value not enabled
                for layer_idx in range(self.config.num_layers):
                    _dbg(
                        f"layer {layer_idx} begin mot video={tuple(video_tokens.shape)} "
                        f"action={tuple(mot_action_tokens.shape)}"
                    )
                    if torch.is_grad_enabled() and self.video_model.wan_model.gradient_checkpointing:
                        video_tokens, mot_action_tokens, mot_value_tokens = torch.utils.checkpoint.checkpoint(
                            self._forward_layer_mot, video_tokens, mot_action_tokens, v_e0, a_e0,
                            layer_idx, t5_ctx, ar_mask,
                            None, None, True, clean_video_freqs, noisy_freqs,
                            padded_len, mot_value_tokens, val_e0, block_latent_counts,
                            full_freqs,
                            t5_row_gate=t5_row_gate,
                            use_reentrant=False,
                        )
                    else:
                        video_tokens, mot_action_tokens, mot_value_tokens = self._forward_layer_mot(
                            video_tokens, mot_action_tokens, v_e0, a_e0,
                            layer_idx, t5_ctx, ar_mask,
                            is_training=True, video_freqs=clean_video_freqs, noisy_freqs=noisy_freqs,
                            padded_len=padded_len,
                            value_tokens=mot_value_tokens, value_e0=val_e0,
                            block_latent_counts=block_latent_counts,
                            full_freqs=full_freqs,
                            t5_row_gate=t5_row_gate,
                        )
                    _dbg(
                        f"layer {layer_idx} end mot video={tuple(video_tokens.shape)} "
                        f"action={tuple(mot_action_tokens.shape)}"
                    )
                # MoT: video and action have different dims, keep separate (no concat)
                video_out = video_tokens
                action_out = mot_action_tokens.to(self.dtype)
                # Value hidden from MoT: already separate tensor
                if mot_value_tokens is not None:
                    value_hidden = mot_value_tokens.reshape(B, num_noisy_chunks, nvpf, -1)

            if not self.config.use_mot:
                clean_video_out = x[:, :half_seq_len]
                full_noisy_out = x[:, half_seq_len:half_seq_len + chunk_ar_tokens.shape[1]]
                sa_hidden, noisy_video_hidden, value_hidden = self._split_chunk_ar_hidden(
                    full_noisy_out,
                    block_latent_counts,
                    include_value=(self.config.value_enabled and nvpf > 0),
                    video_tokens_per_frame=frame_seqlen,
                )
                video_out = torch.cat([clean_video_out, noisy_video_hidden], dim=1)
                action_out = self.action_expert.projection.project_out(sa_hidden)

            # === 13. Outputs extracted (video_out, action_out, value_hidden) ===

            # === 14. Heads ===
            _dbg(f"heads begin video_out={tuple(video_out.shape)} action_out={tuple(action_out.shape)}")
            video_pred = self.video_model.wan_model.head(video_out, v_emb_full)
            video_pred = self._unpatchify_video_tokens(
                video_pred, 2 * N, B, lat_H=lat_H, lat_W=lat_W, num_views=num_views
            )

            action_pred = self.action_expert.decoder(action_out, a_emb)
            _dbg(f"heads end video_pred={tuple(video_pred.shape)} action_pred={tuple(action_pred.shape)}")

        # Extract action predictions (skip state tokens when use_state)
        # napf here is the full noisy_action_tokens_per_frame (including value), but decoder
        # only receives state+action tokens. Use the state+action count for reshaping.
        action_pred = self._strip_chunk_state_predictions(action_pred, block_action_token_lengths)

        # === 15. Loss ===
        _dbg("loss begin")
        if video_pred.dim() == 6:
            pred_noisy = video_pred[:, :, :, N:, :, :].to(self.dtype)  # [B,V,C,N,H,W]
        else:
            pred_noisy = video_pred[:, :, N:, :, :].to(self.dtype)  # [B,C,N,H,W]
        if _gen_video_only:
            return {"video_velocity": pred_noisy}
        noisy_valid_len = (valid_len - cs).clamp(min=0) if valid_len is not None else None
        if pred_noisy.dim() == 6:
            loss_mask_v = torch.ones(B, 1, 1, N, 1, 1, device=device, dtype=self.dtype)
        else:
            loss_mask_v = torch.ones(B, 1, N, 1, 1, device=device, dtype=self.dtype)
        if noisy_valid_len is not None:
            t_arange = torch.arange(N, device=device)
            invalid = t_arange.unsqueeze(0) >= noisy_valid_len.unsqueeze(1)
            if pred_noisy.dim() == 6:
                loss_mask_v = loss_mask_v * (
                    1.0 - invalid.to(self.dtype).view(B, 1, 1, N, 1, 1)
                )
            else:
                loss_mask_v = loss_mask_v * (
                    1.0 - invalid.to(self.dtype).unsqueeze(1).unsqueeze(3).unsqueeze(4)
                )
        if video_valid_mask is not None:
            real_video = video_valid_mask[:, cs:cs + N].to(device=device, dtype=self.dtype)
            view = (B, 1, 1, N, 1, 1) if pred_noisy.dim() == 6 else (B, 1, N, 1, 1)
            loss_mask_v = loss_mask_v * real_video.reshape(view)
        if pred_noisy.dim() == 6:
            loss_mask_v = loss_mask_v * (~off_policy_pending).to(self.dtype).view(B, 1, 1, 1, 1, 1)
        else:
            loss_mask_v = loss_mask_v * (~off_policy_pending).to(self.dtype).view(B, 1, 1, 1, 1)
        # Value mode: zero out video loss (value mode only trains value head)
        if len(vm_idx) > 0:
            loss_mask_v[vm_idx] = 0.0
        mse_v = F.mse_loss(pred_noisy, target_video, reduction="none") * loss_mask_v
        if pred_noisy.dim() == 6:
            V_lat, C_lat, H_lat, W_lat = pred_noisy.shape[1], pred_noisy.shape[2], pred_noisy.shape[4], pred_noisy.shape[5]
            video_loss_per_sample = mse_v.sum(dim=(1, 2, 3, 4, 5)) / (
                loss_mask_v.sum(dim=(1, 2, 3, 4, 5)).clamp(min=1) * V_lat * C_lat * H_lat * W_lat
            )
        else:
            C_lat, H_lat, W_lat = pred_noisy.shape[1], pred_noisy.shape[3], pred_noisy.shape[4]
            video_loss_per_sample = mse_v.sum(dim=(1, 2, 3, 4)) / (
                loss_mask_v.sum(dim=(1, 2, 3, 4)).clamp(min=1) * C_lat * H_lat * W_lat
            )
        video_loss = video_loss_per_sample.mean()

        loss_mask_a = torch.ones(B, device=device, dtype=self.dtype)
        if len(wm_idx) > 0:
            loss_mask_a[wm_idx] = 0.0
        if len(vm_idx) > 0:
            loss_mask_a[vm_idx] = 0.0  # Value mode samples don't contribute to action loss
        loss_mask_a = loss_mask_a * (~off_policy_pending).to(self.dtype)
        action_pos_raw_loss = torch.tensor(0.0, device=device)
        action_rot_raw_loss = torch.tensor(0.0, device=device)
        action_grip_raw_loss = torch.tensor(0.0, device=device)
        action_raw_loss = torch.tensor(0.0, device=device)
        eef_component_warmup_active = (
            _is_eef_component_mode(self.config.action_mode)
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
        elif qpos_family_hand_dim(self.config.action_mode) is not None:
            lay = self._qpos_layout
            qpos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., lay.left_qpos], target_action[..., lay.left_qpos], reduction="none"),
                F.mse_loss(action_pred[..., lay.right_qpos], target_action[..., lay.right_qpos], reduction="none"),
            ], dim=2).mean(dim=2)
            pos_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., lay.left_xyz], target_action[..., lay.left_xyz], reduction="none"),
                F.mse_loss(action_pred[..., lay.right_xyz], target_action[..., lay.right_xyz], reduction="none"),
            ], dim=2).mean(dim=2)
            rot_loss_per_token = torch.cat([
                F.mse_loss(action_pred[..., lay.left_rot6d], target_action[..., lay.left_rot6d], reduction="none"),
                F.mse_loss(action_pred[..., lay.right_rot6d], target_action[..., lay.right_rot6d], reduction="none"),
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
            real_actions = action_valid_mask[:, cs * apl:cs * apl + total_action_len]
            action_token_mask = action_token_mask * real_actions.to(device=device, dtype=self.dtype)
        if rtc_committed_mask is not None:
            action_token_mask = action_token_mask * (1 - rtc_committed_mask)
        if _is_eef_component_mode(self.config.action_mode):
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
        action_loss = action_loss_per_sample.mean()
        action_d2_loss = torch.tensor(0.0, device=device)
        if self.config.action_d2_loss_weight > 0:
            if self.config.action_mode != "joint":
                raise ValueError(
                    "action_d2_loss_weight currently supports action_mode=joint only"
                )
            arm_indices = torch.tensor(
                [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
                device=device,
                dtype=torch.long,
            )
            predicted_clean_action = noisy_action - sa_tokens * action_pred
            action_d2_loss = masked_chunk_second_difference_mse(
                predicted_clean_action.index_select(2, arm_indices),
                target_actions_raw.index_select(2, arm_indices),
                action_token_mask,
                block_action_token_lengths,
            )

        video_loss_weight = self.config.video_loss_weight
        action_loss_weight = self.config.action_loss_weight
        action_d2_loss_weight = self.config.action_d2_loss_weight
        value_loss_weight = self.config.value_loss_weight

        total_loss = (
            video_loss_weight * video_loss
            + action_loss_weight * action_loss
            + action_d2_loss_weight * action_d2_loss
        )
        # === 16. Value loss ===
        value_loss = torch.tensor(0.0, device=device)
        if (self.config.value_enabled and value_hidden is not None and values is not None
                and (len(vm_idx) > 0 or (self.config.value_on_policy and len(pol_idx) > 0))):
            from .value_head import discretize
            value_logits = self.value_head(value_hidden.to(self.dtype))  # [B, M, K, num_bins]

            # GT chunk values: take the last latent value from each noisy chunk.
            noisy_values = values[:, cs:].to(torch.float32 if self.config.value_fp32 else self.dtype)  # [B, N]
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

            # Only compute value loss for Value mode samples, plus policy
            # samples when value_on_policy asks for them.
            value_keep = use_vm
            if self.config.value_on_policy and len(pol_idx) > 0:
                value_keep = value_keep | ~off_policy_mask
            value_loss_mask = value_keep.float().unsqueeze(1)  # [B, 1]
            value_loss_mask = value_loss_mask * (~off_policy_pending).float().unsqueeze(1)
            if self.config.value_on_policy and self.config.value_on_policy_max_sigma is not None:
                _cap = float(self.config.value_on_policy_max_sigma)
                value_loss_mask = value_loss_mask * (sigma_a <= _cap).to(self.dtype).unsqueeze(1)
            if noisy_valid_len is not None:
                chunk_end_idx = torch.tensor(
                    self._chunk_end_indices(block_latent_counts), device=device, dtype=torch.long
                ).unsqueeze(0)
                chunk_mask = (chunk_end_idx < noisy_valid_len.unsqueeze(1)).float()
                value_loss_mask = value_loss_mask * chunk_mask

            value_loss = (value_ce * value_loss_mask).sum() / value_loss_mask.sum().clamp(min=1)
            total_loss = total_loss + value_loss_weight * value_loss
        _dbg("loss end")

        return {
            "total_loss": total_loss,
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "action_d2_loss": action_d2_loss.detach(),
            "action_raw_loss": action_raw_loss.detach(),
            "action_pos_raw_loss": action_pos_raw_loss.detach(),
            "action_rot_raw_loss": action_rot_raw_loss.detach(),
            "action_grip_raw_loss": action_grip_raw_loss.detach(),
            "action_hand_raw_loss": (
                action_grip_raw_loss.detach()
                if qpos_family_hand_dim(self.config.action_mode) is not None
                else torch.tensor(0.0, device=device)
            ),
            "value_loss": value_loss.detach(),
            "video_loss_weight": video_loss_weight,
            "action_loss_weight": action_loss_weight,
            "action_d2_loss_weight": action_d2_loss_weight,
            "value_loss_weight": value_loss_weight,
            "eef_component_warmup_active": float(eef_component_warmup_active),
            "eef_position_loss_weight": eef_pos_weight,
            "eef_rotation_loss_weight": eef_rot_weight,
            "eef_gripper_loss_weight": eef_grip_weight,
            "hand_qpos_loss_weight": hand_qpos_weight,
            "rtc_training_ratio": rtc_selected.to(self.dtype).mean().detach(),
            "num_vm_samples": len(vm_idx),
            "num_wm_samples": len(wm_idx),
            "num_pol_samples": len(pol_idx),
            "num_suboptimal_samples": num_suboptimal,
            "num_suboptimal_vm_samples": num_suboptimal_vm,
        }

    # ------------------------------------------------------------------
    # Inference: KV cache
    # ------------------------------------------------------------------

    def init_kv_cache(self, batch_size: int = 1, start_frame_abs: int = 0):
        """Initialize empty KV caches for all layers."""
        self._pending_observation_latents = None
        n = self.video_model.wan_model.num_heads
        d = self.wan_dim // n
        max_cache_len = self._max_cache_tokens()
        self.kv_caches = []
        for _ in range(self.config.num_layers):
            self.kv_caches.append(
                torch.zeros(2, batch_size, max_cache_len, n, d, device=self.device, dtype=self.dtype)
            )
        self._set_kv_cache_valid_len(0)
        self._window_rope_cache_start = 0
        self.current_start_frame = int(start_frame_abs)
        self._last_infer_K = self._fixed_chunk_size()
        logger.info("KV cache initialized at start_frame=%s", self.current_start_frame)

    def clear_kv_cache(self, start_frame_abs: int = 0):
        self._pending_observation_latents = None
        self.kv_caches = None
        self._set_kv_cache_valid_len(0)
        self._window_rope_cache_start = 0
        self.current_start_frame = int(start_frame_abs)
        self._last_infer_K = self._fixed_chunk_size()

    def set_current_start_frame(self, start_frame_abs: int) -> None:
        self.current_start_frame = int(start_frame_abs)

    def export_inference_state(self) -> Dict[str, Any]:
        """Return a cloneable snapshot of streaming inference state."""
        kv_caches = None
        if self.kv_caches is not None:
            kv_caches = [cache.clone() for cache in self.kv_caches]
        return {
            "kv_caches": kv_caches,
            "pending_observation_latents": (
                self._pending_observation_latents.clone()
                if getattr(self, "_pending_observation_latents", None) is not None else None
            ),
            "kv_cache_valid_len": self._kv_cache_valid_len_int(),
            "current_start_frame": int(self.current_start_frame),
            "window_rope_cache_start": int(getattr(self, "_window_rope_cache_start", 0)),
            "video_rope_window_reference": "training_window",
            "last_infer_K": int(getattr(self, "_last_infer_K", self._fixed_chunk_size())),
        }

    def load_inference_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore streaming inference state from export_inference_state()."""
        if state is None:
            self.clear_kv_cache()
            return
        kv_caches = state.get("kv_caches")
        if kv_caches is not None and state.get("video_rope_window_reference") != "training_window":
            raise ValueError("Cannot restore a KV cache with a different video RoPE reference")
        self._window_rope_cache_start = int(state.get("window_rope_cache_start", 0))
        if kv_caches is None:
            self.kv_caches = None
            self._set_kv_cache_valid_len(0)
        else:
            max_cache_len = self._max_cache_tokens()
            loaded = []
            inferred_valid_len = int(kv_caches[0].shape[2]) if kv_caches else 0
            for cache in kv_caches:
                cache = cache.to(device=self.device, dtype=self.dtype)
                if cache.shape[2] == max_cache_len:
                    loaded.append(cache.clone())
                else:
                    fixed = cache.new_zeros(cache.shape[0], cache.shape[1], max_cache_len, cache.shape[3], cache.shape[4])
                    copy_len = min(int(cache.shape[2]), max_cache_len)
                    if copy_len > 0:
                        fixed[:, :, :copy_len].copy_(cache[:, :, :copy_len])
                    loaded.append(fixed)
            self.kv_caches = loaded
            self._set_kv_cache_valid_len(min(int(state.get("kv_cache_valid_len", inferred_valid_len)), max_cache_len))
        self.current_start_frame = int(state.get("current_start_frame", 0))
        self._last_infer_K = int(state.get("last_infer_K", self._fixed_chunk_size()))
        pending = state.get("pending_observation_latents")
        self._pending_observation_latents = (
            pending.to(device=self.device, dtype=self.dtype).clone()
            if pending is not None else None
        )

    @torch.no_grad()
    def fill_kv_cache(self, obs_frames: torch.Tensor, is_initial: bool = False,
                      language_embeddings=None):
        """Encode observation frames and fill KV cache (conditioning frame).
        NVTX: outer "fill_kv_cache" with vae_encode + _fill_cache_from_latent sub-ranges.

        VAE encoding remains streaming; transformer history is filled in complete
        training chunks so clean tokens within a chunk can attend one another.

        Args:
            obs_frames: [B, C, T, H, W] or [B, V, C, T, H, W] raw observation frames.
                        is_initial=True: T=1 (single frame, uses encode_chunk_first)
                        is_initial=False: T=4 (4 frames, uses encode_chunk)
            language_embeddings: T5 embeddings for cross-attention.
        """
        _nvpush("fill_kv_cache")
        is_stereo = obs_frames.dim() == 6
        if is_stereo:
            B, V, C, T, H, W = obs_frames.shape
            if V != self.config.num_views:
                raise ValueError(f"Expected {self.config.num_views} views, got {V}")
            obs_flat = obs_frames.reshape(B * V, C, T, H, W)
        else:
            obs_flat = obs_frames

        _nvpush("vae_encode")
        if is_initial:
            obs_norm = (obs_flat * 2.0 - 1.0).to(self.dtype)
            self.video_model.vae.clear_cache()
            latent = self.video_model.vae.encode_chunk_first(obs_norm).to(self.dtype)
            from utils.common import repeat_initial_latent_to_chunk_size
            latent = repeat_initial_latent_to_chunk_size(latent, int(self._fixed_chunk_size()))
        else:
            B_flat, C, T, H, W = obs_flat.shape
            assert T == 4, f"Streaming inference expects exactly 4 frames per chunk, got {T}"
            obs_norm = (obs_flat * 2.0 - 1.0).to(self.dtype)
            latent = self.video_model.vae.encode_chunk(obs_norm).to(self.dtype)
        _nvpop()  # vae_encode
        if is_stereo:
            latent = latent.view(B, V, *latent.shape[1:])

        if is_initial:
            self._pending_observation_latents = None
        else:
            time_dim = 3 if is_stereo else 2
            pending = getattr(self, "_pending_observation_latents", None)
            if pending is not None:
                latent = torch.cat([pending, latent], dim=time_dim)
            if self._latent_num_frames(latent) < self._fixed_chunk_size():
                self._pending_observation_latents = latent
                _nvpop()  # fill_kv_cache
                return
            self._pending_observation_latents = None
        self._fill_cache_from_latent(latent, language_embeddings)
        _nvpop()  # fill_kv_cache

    def _fill_cache_from_latent(self, latent: torch.Tensor, language_embeddings=None):
        """Shared logic: patchify latent → forward through all layers → update KV cache.

        Used by both fill_kv_cache (real obs → VAE encode → latent) and
        fill_kv_cache_from_latent (imagined latent, skip VAE).
        """
        _nvpush("_fill_cache_from_latent")
        latent = latent.to(self.dtype)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            _nvpush("patch_embed")
            tokens = self._video_patch_embed(latent)  # [B, T_lat*frame_seqlen, wan_dim]
            seq_len = tokens.shape[1]
            _nvpop()  # patch_embed

            # Time embedding (t=0 for conditioning)
            B = tokens.shape[0]
            t_zero = torch.zeros(B, seq_len, device=self.device)
            _, e0 = self._video_time_embed(t_zero)

            # T5 context
            t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

            # Forward through layers, updating KV cache
            self._ensure_freqs_device()
            x = tokens
            max_cache_len = self._max_cache_tokens()
            old_valid_len = self._kv_cache_valid_len_int()
            if seq_len > max_cache_len:
                raise ValueError(f"KV cache chunk length {seq_len} exceeds max_cache_tokens={max_cache_len}")
            drop_len = max(old_valid_len + seq_len - max_cache_len, 0)
            keep_len = old_valid_len - drop_len
            write_start = keep_len
            write_end = write_start + seq_len
            new_rope_origin, rope_shift = self._window_cache_rope_update(old_valid_len, seq_len)
            self._set_kv_cache_valid_len(old_valid_len)
            for layer_idx in range(self.config.num_layers):
                _nvpush(f"fill_layer_{layer_idx}")
                blk = self.video_model.wan_model.blocks[layer_idx]
                _nvpush("modulation")
                mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                _nvpop()  # modulation

                _nvpush("norm1")
                norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
                n = self.video_model.wan_model.num_heads
                d = self.wan_dim // n
                _nvpop()  # norm1

                _nvpush("qkv_proj")
                q, k, v = _wan_self_qkv(blk.self_attn, norm_x, B, seq_len, n, d)
                _nvpop()  # qkv_proj

                cache = self.kv_caches[layer_idx]
                cache_k, cache_v = cache[0], cache[1]

                _nvpush("rope_freqs")
                n_lat = self._latent_num_frames(latent)
                video_freqs = self._create_video_freqs(
                    n_lat,
                    start_frame=self._inference_video_start_frame(old_valid_len),
                )
                q = rope_action_apply(
                    x=q,
                    freqs=video_freqs,
                    freqs_action=self.freqs_action,
                    freqs_state=self.freqs_state,
                    action_register_length=None,
                ).type_as(v)
                k = rope_action_apply(
                    x=k,
                    freqs=video_freqs,
                    freqs_action=self.freqs_action,
                    freqs_state=self.freqs_state,
                    action_register_length=None,
                ).type_as(v)
                _nvpop()  # rope_freqs (in fill layer)

                _nvpush("attn_dispatch")
                y = self._fixed_cache_sdpa(q, cache_k, cache_v, k, v)
                y = y.flatten(2)
                _nvpop()  # attn_dispatch
                _nvpush("attn_o")
                y = blk.self_attn.o(y)
                x = x + y * mod[2].squeeze(2)
                _nvpop()  # attn_o

                # Cross attention
                if t5_ctx is not None:
                    _nvpush("cross_attn")
                    x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
                    _nvpop()  # cross_attn

                # FFN
                _nvpush("norm2_mod")
                ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                _nvpop()  # norm2_mod
                _nvpush("ffn")
                x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
                _nvpop()  # ffn

                _nvpush("cache_write")
                retained_k = torch.cat([cache_k[:, :old_valid_len], k], dim=1)[:, drop_len:]
                retained_v = torch.cat([cache_v[:, :old_valid_len], v], dim=1)[:, drop_len:]
                if rope_shift > 0 and self._should_adjust_video_rope_on_pop():
                    retained_k = self._shift_cached_video_k_temporal(retained_k, rope_shift)
                cache_k[:, :write_end].copy_(retained_k)
                cache_v[:, :write_end].copy_(retained_v)
                _nvpop()  # cache_write
                _nvpop()  # fill_layer_{layer_idx}

        self._set_kv_cache_valid_len(write_end)
        self._window_rope_cache_start = new_rope_origin
        self.current_start_frame += self._latent_num_frames(latent)
        logger.debug(f"KV cache filled: start_frame={self.current_start_frame}")
        _nvpop()  # _fill_cache_from_latent

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
        v_t_vals = t_vid.to(device=self.device).reshape(1, 1).expand(B, v_seq_len)
        _, e0 = self._video_time_embed(v_t_vals)
        infer_latent_frames = v_seq_len // self.frame_seqlen
        video_kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx in range(self.config.num_layers):
            blk = self.video_model.wan_model.blocks[layer_idx]
            mod = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
            q, k, v = _wan_self_qkv(blk.self_attn, norm_x, B, v_seq_len, n, d)

            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
            cache_seq_len = self._kv_cache_valid_len_int() if kv_cache is not None else 0
            video_freqs = self._create_video_freqs(
                infer_latent_frames,
                start_frame=self._inference_video_start_frame(cache_seq_len),
            )
            q = rope_action_apply(q, video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)
            k = rope_action_apply(k, video_freqs, self.freqs_action, self.freqs_state, None).type_as(v)

            video_kv_per_layer.append((k, v))
            if kv_cache is not None:
                y = self._fixed_cache_sdpa(q, kv_cache[0], kv_cache[1], k, v).flatten(2)
            else:
                from wan.modules.attention import flash_attention
                y = flash_attention(q, k, v, k_lens=None).flatten(2)
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
        cache_key_extra: int = 0,
    ) -> torch.Tensor:
        """Action-only forward using frozen current-chunk video K/V."""
        B, a_seq_len, _ = action_tokens.shape
        n = self.video_model.wan_model.num_heads
        d = self.wan_dim // n
        t_act = t_act.to(device=self.device, dtype=self.dtype)
        if t_act.dim() == 2 and t_act.shape[-1] == a_seq_len:
            a_t_vals = t_act  # Already per-token (e.g. an RTC committed prefix at t=0).
        else:
            t_act_vals = t_act.reshape(-1)
            if t_act_vals.numel() == 1:
                a_t_vals = t_act_vals.reshape(1, 1).expand(B, a_seq_len)
            else:
                a_t_vals = t_act_vals.reshape(B, 1).expand(-1, a_seq_len)
        if self.config.use_mot:
            _, a_e0 = self._action_time_embed(a_t_vals, cache_key_extra=cache_key_extra)
        else:
            _, a_e0 = self._video_time_embed(a_t_vals, cache_key_extra=cache_key_extra)
        infer_latent_frames = int(K)
        x = action_tokens

        for layer_idx in range(self.config.num_layers):
            blk = self.video_model.wan_model.blocks[layer_idx]
            if self.config.use_mot:
                action_blk = self.action_expert.blocks[layer_idx]
                mod = (action_blk.modulation.unsqueeze(0) + a_e0).chunk(6, dim=2)
                norm_x = (action_blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
                a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_x, action_blk.wan_action_qkv)
                q = action_blk.wan_action_norm_q(a_qkv[0].flatten(-2)).view(B, a_seq_len, n, d)
                k = action_blk.wan_action_norm_k(a_qkv[1].flatten(-2)).view(B, a_seq_len, n, d)
                v = a_qkv[2].view(B, a_seq_len, n, d)
            else:
                mod = (blk.modulation.unsqueeze(0) + a_e0).chunk(6, dim=2)
                norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
                q, k, v = _wan_self_qkv(blk.self_attn, norm_x, B, a_seq_len, n, d)

            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
            action_only_freqs = self.freqs_action.new_empty((0, 1, self.freqs_action.shape[-1]))
            q = causal_rope_action_apply(
                q,
                action_only_freqs,
                self.freqs_action,
                self.freqs_state,
                action_register_length=a_seq_len,
                num_action_per_block=infer_latent_frames * self.config.action_register_tokens_per_frame,
                num_state_per_block=self.config.state_tokens_per_frame,
                num_value_per_block=0,
                action_tokens_per_timestep=self.config.action_tokens_per_timestep,
            ).type_as(v)
            k = causal_rope_action_apply(
                k,
                action_only_freqs,
                self.freqs_action,
                self.freqs_state,
                action_register_length=a_seq_len,
                num_action_per_block=infer_latent_frames * self.config.action_register_tokens_per_frame,
                num_state_per_block=self.config.state_tokens_per_frame,
                num_value_per_block=0,
                action_tokens_per_timestep=self.config.action_tokens_per_timestep,
            ).type_as(v)

            frozen_k, frozen_v = frozen_video_kv[layer_idx]
            if kv_cache is not None:
                current_k = torch.cat([frozen_k, k], dim=1)
                current_v = torch.cat([frozen_v, v], dim=1)
                y = self._fixed_cache_sdpa(q, kv_cache[0], kv_cache[1], current_k, current_v).flatten(2)
            else:
                attn_k = torch.cat([frozen_k, k], dim=1)
                attn_v = torch.cat([frozen_v, v], dim=1)
                from wan.modules.attention import flash_attention
                y = flash_attention(q, attn_k, attn_v, k_lens=None).flatten(2)
            if self.config.use_mot:
                y = action_blk.wan_action_o(y)
                x = x + y * mod[2].squeeze(2)
                ffn_in = (action_blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                x = x + action_blk.ffn(ffn_in) * mod[5].squeeze(2)
            else:
                y = blk.self_attn.o(y)
                x = x + y * mod[2].squeeze(2)
                if t5_ctx is not None:
                    x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
                ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)
        return x

    def _encode_current_action_tokens(
        self,
        action_chunk: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        project_in: bool = True,
    ) -> torch.Tensor:
        if self.config.use_state and state is None:
            raise ValueError("State conditioning requires the current measured state")
        state_chunk = None
        if state is not None and self.config.use_state:
            state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
            state_chunk = state_chunk.to(device=self.device, dtype=self.dtype)
        if state_chunk is not None:
            tokens = self.action_expert.input_encoder(state_chunk, action_chunk.to(device=self.device, dtype=self.dtype))
        else:
            tokens = self.action_expert.input_encoder(None, action_chunk.to(device=self.device, dtype=self.dtype))
        if project_in and not self.config.use_mot:
            tokens = self.action_expert.projection.project_in(tokens)
        return tokens

    def _decode_current_action_tokens(
        self,
        act_out: torch.Tensor,
        a_emb: torch.Tensor,
        action_chunk_len: int,
    ) -> torch.Tensor:
        action_pred = self.action_expert.decoder(act_out, a_emb)
        if self.config.state_tokens_per_frame > 0:
            action_pred = action_pred[:, self.config.state_tokens_per_frame:]
        return action_pred

    def _build_action_inference_timesteps(self, num_action_inference_steps: int) -> torch.Tensor:
        steps = max(int(num_action_inference_steps), 1)
        raw = torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=self.dtype)
        shift = float(self.config.action_timestep_shift)
        if shift != 1.0:
            raw = shift * raw / (1.0 + (shift - 1.0) * raw)
        return raw

    def _expand_action_sigma(self, sigma_action: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        sigma_action = sigma_action.to(device=action_tensor.device, dtype=action_tensor.dtype)
        while sigma_action.dim() < action_tensor.dim():
            sigma_action = sigma_action.unsqueeze(-1)
        return sigma_action

    def _integrate_action_velocity_euler(
        self,
        noisy_action: torch.Tensor,
        action_velocity: torch.Tensor,
        timestep_curr: torch.Tensor,
        timestep_next: torch.Tensor,
    ) -> torch.Tensor:
        dt = self._expand_action_sigma(timestep_next - timestep_curr, noisy_action)
        return noisy_action + dt * action_velocity

    def _predict_action_velocity_from_history(
        self,
        noisy_action: torch.Tensor,
        sigma_action: torch.Tensor,
        t5_ctx: Optional[torch.Tensor],
        state: torch.Tensor = None,
        committed_steps: int = 0,
    ) -> torch.Tensor:
        """Predict action velocity using only history KV cache plus current noisy action."""
        B = noisy_action.shape[0]
        action_chunk_len = noisy_action.shape[1]
        a_tokens = self._encode_current_action_tokens(
            noisy_action, state, project_in=True,
        )
        a_seq_len = a_tokens.shape[1]
        a_t_vals = (
            sigma_action.to(device=self.device, dtype=self.dtype).view(B, 1).expand(-1, a_seq_len)
            * self.action_scheduler.num_train_timesteps
        )
        if committed_steps > 0:
            # Mirror training: the committed prefix carries timestep 0. Tokens are
            # [state | action registers], so action i sits at spf + i*atpt.
            spf = self.config.state_tokens_per_frame
            atpt = self.config.action_tokens_per_timestep
            a_t_vals = a_t_vals.clone()
            a_t_vals[:, spf:spf + committed_steps * atpt:atpt] = 0
        a_emb, _ = self._action_time_embed(a_t_vals, cache_key_extra=committed_steps)

        if not self.config.use_mot:
            _, e0_combined = self._video_time_embed(a_t_vals, cache_key_extra=committed_steps)
            x = a_tokens
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
                    v_seq_len=0,
                    is_training=False,
                    action_register_length=action_register_length,
                    attention_mode="joint",
                )
            act_out = self.action_expert.projection.project_out(x)
        else:
            n = self.video_model.wan_model.num_heads
            d = self.wan_dim // n
            empty_video_kv = [
                (
                    a_tokens.new_zeros(B, 0, n, d),
                    a_tokens.new_zeros(B, 0, n, d),
                )
                for _ in range(self.config.num_layers)
            ]
            act_out = self._forward_action_only_cached(
                action_tokens=a_tokens,
                t_act=a_t_vals,
                frozen_video_kv=empty_video_kv,
                t5_ctx=t5_ctx,
                K=self._fixed_chunk_size(),
                cache_key_extra=committed_steps,
            )

        return self._decode_current_action_tokens(act_out, a_emb, action_chunk_len)

    def _sample_action_chunk_from_history(
        self,
        t5_ctx: Optional[torch.Tensor],
        num_action_inference_steps: int,
        state: torch.Tensor = None,
        committed_prefix: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.kv_caches is None or len(self.kv_caches) == 0:
            raise RuntimeError("Action streaming inference requires a pre-filled KV cache")
        K = self._fixed_chunk_size()
        B = self.kv_caches[0].shape[1]
        action_chunk_len = K * self.config.actions_per_latent
        y = torch.randn(B, action_chunk_len, self.config.action_dim, device=self.device, dtype=self.dtype)
        committed_steps = 0
        prefix = None
        if committed_prefix is not None and committed_prefix.shape[1] > 0:
            prefix = self._scale_qpos20_rel_wrist_xyz(
                committed_prefix.to(device=self.device, dtype=torch.float32)
            )
            committed_steps = min(int(prefix.shape[1]), action_chunk_len)
            prefix = prefix[:, :committed_steps]
            y[:, :committed_steps] = prefix
        action_timesteps = self._build_action_inference_timesteps(num_action_inference_steps)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # mode=action_only policy loop: sample actions from history KV only.
            for step_idx in range(len(action_timesteps) - 1):
                timestep_curr = action_timesteps[step_idx].expand(B)
                timestep_next = action_timesteps[step_idx + 1].expand(B)
                action_velocity = self._predict_action_velocity_from_history(
                    y,
                    timestep_curr,
                    t5_ctx,
                    state=state,
                    committed_steps=committed_steps,
                )
                y = self._integrate_action_velocity_euler(y, action_velocity, timestep_curr, timestep_next)
                if committed_steps > 0:
                    y[:, :committed_steps] = prefix
        return y

    @torch.no_grad()
    def inference_action_streaming(
        self,
        state: torch.Tensor = None,
        language_embeddings=None,
        num_action_inference_steps: int = None,
        committed_prefix: torch.Tensor = None,
    ) -> torch.Tensor:
        if num_action_inference_steps is None:
            num_action_inference_steps = self.config.num_action_inference_timesteps
        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None
        y = self._sample_action_chunk_from_history(
            t5_ctx, num_action_inference_steps, state=state,
            committed_prefix=committed_prefix,
        )
        self._last_infer_K = self._fixed_chunk_size()
        return self._unscale_qpos20_rel_wrist_xyz(y)

    def _predict_video_velocity_given_actions(
        self,
        noisy_video: torch.Tensor,
        actions: torch.Tensor,
        timestep_video: torch.Tensor,
        t5_ctx: Optional[torch.Tensor],
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict video velocity for action_only WM/value path: current action keys precede video keys."""
        B = noisy_video.shape[0]
        K = self._latent_num_frames(noisy_video)
        z_tokens = self._video_patch_embed(noisy_video.to(self.dtype))
        v_seq_len = z_tokens.shape[1]
        a_tokens = self._encode_current_action_tokens(
            actions, state, project_in=True,
        )
        a_seq_len = a_tokens.shape[1]
        video_freqs = self._create_video_freqs(
            K,
            start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
        )

        v_t_vals = timestep_video.to(device=self.device, dtype=self.dtype).view(B, 1).expand(-1, v_seq_len)
        v_emb, v_e0 = self._video_time_embed(v_t_vals)
        a_t_vals = torch.zeros(B, a_seq_len, device=self.device, dtype=self.dtype)
        action_freqs = self._build_chunk_ar_freqs(
            clean_video_freqs=z_tokens.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
            noisy_video_freqs=video_freqs,
            block_latent_counts=[K],
            include_value=False,
        )
        if self.config.use_mot:
            _, a_e0 = self._action_time_embed(a_t_vals)
            video_tokens = z_tokens
            action_tokens = a_tokens
            for layer_idx in range(self.config.num_layers):
                kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                video_tokens, action_tokens, _ = self._forward_layer_mot(
                    video_tokens,
                    action_tokens,
                    v_e0,
                    a_e0,
                    layer_idx,
                    t5_ctx,
                    ar_mask=None,
                    kv_cache=kv_cache,
                    v_seq_len=v_seq_len,
                    is_training=False,
                    full_freqs=action_freqs,
                    attention_mode="fdm",
                )
            vid_out = video_tokens
            v_pred = self.video_model.wan_model.head(vid_out, v_emb)
            return self._unpatchify_video_tokens(v_pred, K, B)

        combined_t_vals = self._cat_chunk_tokens(a_t_vals, v_t_vals, dim=1)
        _, e0_combined = self._video_time_embed(combined_t_vals)

        # mode=action_only optional video/value branch: current chunk layout is [action | video].
        x = self._cat_chunk_tokens(a_tokens, z_tokens, dim=1)
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
                full_freqs=action_freqs,
                attention_mode="fdm",
            )
        _, vid_slice = self._chunk_token_slices(a_seq_len, v_seq_len)
        vid_out = x[:, vid_slice]
        v_pred = self.video_model.wan_model.head(vid_out, v_emb)
        return self._unpatchify_video_tokens(v_pred, K, B)

    def _denoise_step_core(
        self,
        z_input: torch.Tensor,
        state: Optional[torch.Tensor],
        y: torch.Tensor,
        t5_ctx: Optional[torch.Tensor],
        step_idx: int,
        K: int,
        B: int,
        mode: str,
        precomputed_real_freqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """B15: compile-friendly per-step core. Returns (v_pred_patch, a_pred).
        v_pred stays in PATCH space; caller unpatchifies in eager (B15 ❌).
        """
        # patch_embed
        z_tokens = self._video_patch_embed(z_input)
        v_seq_len = z_tokens.shape[1]

        # encode_action + project_in
        if state is not None and self.config.use_state:
            state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
            a_tokens = self.action_expert.input_encoder(state_chunk, y)
        else:
            a_tokens = self.action_expert.input_encoder(None, y)
        a_tokens = self.action_expert.projection.project_in(a_tokens)
        a_seq_len = a_tokens.shape[1]

        # time_embed lookup (Plan Z list cache)
        e0_combined = self._step_emb_cache_v_e0[step_idx]
        v_emb = self._step_emb_cache_v_emb[step_idx]
        a_emb = self._step_emb_cache_a_emb[step_idx]

        # rope_freqs
        x = self._cat_chunk_tokens(a_tokens, z_tokens, dim=1)
        # F-gamma: complex freqs construction lifted to inference_step_streaming (eager).
        # The compile region only sees the real-paired tensor passed in.
        current_freqs = None
        action_register_length = a_seq_len if a_seq_len > 0 else None

        # 30-layer forward (M9 compiled region when wrapped)
        for layer_idx in range(self.config.num_layers):
            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
            x = self._forward_layer_simple(
                x, e0_combined, layer_idx, t5_ctx, ar_mask=None,
                kv_cache=kv_cache, v_seq_len=v_seq_len,
                is_training=False,
                action_register_length=action_register_length,
                full_freqs=current_freqs,
                attention_mode=("video_idm" if mode in {"video_idm", "loop"} else "joint"),
                precomputed_real_freqs=precomputed_real_freqs,
            )

        # split + project_out + head (still patch space) + action_decoder
        act_slice, vid_slice = self._chunk_token_slices(a_seq_len, v_seq_len)
        act_out = self.action_expert.projection.project_out(x[:, act_slice])
        vid_out = x[:, vid_slice]
        v_pred = self.video_model.wan_model.head(vid_out, v_emb)  # patch space
        a_pred = self.action_expert.decoder(act_out, a_emb)
        if self.config.state_tokens_per_frame > 0:
            a_pred = a_pred[:, self.config.state_tokens_per_frame:, :]
        return v_pred, a_pred

    @torch.no_grad()
    def inference_step_streaming(
        self,
        state: torch.Tensor = None,
        language_embeddings=None,
        num_inference_steps: int = None,
        num_action_inference_steps: int = None,
        predict_video: Optional[bool] = None,
        predict_value: bool = False,
        mode_override: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next chunk Z_k, Y_k using KV cache.

        Returns:
            video_latent: [B, C, K, H_lat, W_lat]
            action_pred: [B, action_chunk_size, action_dim]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_timesteps
        if num_action_inference_steps is None:
            num_action_inference_steps = self.config.num_action_inference_timesteps

        K = self._fixed_chunk_size()
        B = self.kv_caches[0].shape[1]
        mode = (
            str(mode_override).lower()
            if mode_override is not None
            else self._mode()
        )
        if mode not in {"joint", "video_idm", "action_only", "loop"}:
            raise ValueError(f"Unsupported inference mode override: {mode!r}")
        t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

        # mode=action_only does not enter the joint/video_idm step loop below.
        # It samples actions first, then predicts video/value only when requested.
        if mode == "action_only":
            y_model = self._sample_action_chunk_from_history(
                t5_ctx,
                num_action_inference_steps,
                state=state,
            )
            self._last_infer_K = K

            do_video = bool(predict_video) if predict_video is not None else bool(predict_value)
            if do_video:
                z = self._make_inference_video_noise(B, K)
                from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
                sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
                sched_video.set_timesteps(num_inference_steps, device=self.device, shift=float(self.config.video_timestep_shift))
                if self.config.decouple_inference_noise:
                    sigma_max = sched_video.sigmas[0]
                    final_noise = self.config.video_inference_final_noise
                    sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
                    sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    for t_vid in sched_video.timesteps:
                        timestep_video = t_vid.to(device=self.device, dtype=torch.float32).reshape(1).expand(B)
                        video_velocity = self._predict_video_velocity_given_actions(
                            z,
                            y_model,
                            timestep_video,
                            t5_ctx,
                            state=state,
                        )
                        z = self._step_video_scheduler(sched_video, video_velocity, t_vid, z)
            else:
                z = self._empty_current_video_latent(B, K)

            y = self._unscale_qpos20_rel_wrist_xyz(y_model)
            if predict_value and self.config.value_enabled and do_video:
                value_pred = self.inference_value_streaming(
                    video_latent=z,
                    actions=y,
                    state=state,
                    language_embeddings=language_embeddings,
                )
                return z, y, value_pred
            return z, y

        if mode == "loop":
            z = self._make_inference_video_noise(B, K)
            action_chunk_len = K * self.config.actions_per_latent
            y = torch.randn(B, action_chunk_len, self.config.action_dim, device=self.device, dtype=self.dtype)

            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sched_video.set_timesteps(num_inference_steps, device=self.device, shift=float(self.config.video_timestep_shift))

            action_sigmas = self._build_action_inference_timesteps(num_action_inference_steps)
            max_fdm_steps = len(sched_video.timesteps)
            if self.config.video_steps is not None:
                max_fdm_steps = min(max_fdm_steps, max(0, int(self.config.video_steps)))
            fdm_budget = max_fdm_steps
            fdm_step = 0

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                for step_idx in range(len(action_sigmas) - 1):
                    sigma_curr = action_sigmas[step_idx].expand(B)
                    sigma_next = action_sigmas[step_idx + 1].expand(B)

                    if step_idx == 0:
                        action_velocity = self._predict_action_velocity_from_history(
                            y,
                            sigma_curr,
                            t5_ctx,
                            state=state,
                        )
                    else:
                        if fdm_step < fdm_budget:
                            t_vid = sched_video.timesteps[fdm_step]
                            timestep_video = t_vid.to(device=self.device, dtype=torch.float32).reshape(1).expand(B)
                            video_velocity = self._predict_video_velocity_given_actions(
                                z,
                                y,
                                timestep_video,
                                t5_ctx,
                                state=state,
                            )
                            z = self._step_video_scheduler(sched_video, video_velocity, t_vid, z)
                            fdm_step += 1

                        action_velocity = self.forward_action_velocity_from_latent(
                            video_latent=z,
                            noisy_action=y,
                            sigma_action=sigma_curr,
                            state=state,
                            language_embeddings=language_embeddings,
                        )
                    y = self._integrate_action_velocity_euler(y, action_velocity, sigma_curr, sigma_next)

            self._last_infer_K = K
            y_out = self._unscale_qpos20_rel_wrist_xyz(y)
            if predict_value and self.config.value_enabled:
                value_pred = self.inference_value_streaming(
                    video_latent=z,
                    actions=y_out,
                    state=state,
                    language_embeddings=language_embeddings,
                )
                return z, y_out, value_pred
            return z, y_out

        # Initialize noise
        z = self._make_inference_video_noise(B, K)
        action_chunk_len = K * self.config.actions_per_latent
        y = torch.randn(B, action_chunk_len, self.config.action_dim, device=self.device, dtype=self.dtype)

        # Two separate schedulers (decouple inference)
        _nvpush("inference_step_streaming")
        _nvpush("scheduler_setup")
        # t1v11l: scheduler dispatch (UniPC default, Euler optional)
        _scheduler_name = str(getattr(self.config, "inference_scheduler", "unipc")).lower()
        if _scheduler_name == "euler":
            sched_video = _FlowEulerScheduler(num_inference_steps, device=self.device,
                                              shift=float(self.config.video_timestep_shift),
                                              final_sigma=0.0)
            sched_action = _FlowEulerScheduler(num_action_inference_steps, device=self.device,
                                               shift=float(self.config.action_timestep_shift),
                                               final_sigma=0.0)
            if self.config.decouple_inference_noise:
                # B-sched: keep sigma_max as 0-d tensor (matches UniPC branch below).
                sigma_max = sched_video.sigmas[0]
                final_noise = float(self.config.video_inference_final_noise)
                sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
                sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)
        else:
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            sched_video = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sched_action = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sched_video.set_timesteps(num_inference_steps, device=self.device, shift=float(self.config.video_timestep_shift))
            sched_action.set_timesteps(num_action_inference_steps, device=self.device, shift=float(self.config.action_timestep_shift))
            # Decouple: rescale video schedule so it ends at video_inference_final_noise instead of 0
            if self.config.decouple_inference_noise:
                sigma_max = sched_video.sigmas[0]
                final_noise = self.config.video_inference_final_noise
                sched_video.sigmas = sched_video.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
                sched_video.timesteps = (sched_video.sigmas[:-1] * 1000).to(torch.int64)

        _nvpop()  # scheduler_setup
        if mode == "video_idm":
            total_steps = len(sched_action.timesteps)
        elif mode == "joint" and len(sched_video.timesteps) >= len(sched_action.timesteps):
            # Support fused unequal schedules such as action=4/video=8:
            # jointly denoise until action is clean, then keep the clean action
            # register at t=0 while finishing the video schedule.
            total_steps = len(sched_video.timesteps)
        else:
            total_steps = min(
                len(sched_video.timesteps),
                len(sched_action.timesteps),
            )
        joint_steps = total_steps
        if mode == "video_idm" and self.config.video_steps is not None:
            joint_steps = max(0, min(int(self.config.video_steps), len(sched_video.timesteps), total_steps))
        use_action_only_phase = bool(mode == "video_idm" and self.config.idm_mode and joint_steps < total_steps)
        frozen_z = None
        frozen_video_kv = None

        self._b15_full_freqs_real = None

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            # mode=joint: every step updates video and action together.
            # mode=video_idm: prefix steps update both with masked [action | video];
            #                 suffix steps freeze video and continue action denoising.
            for step_idx in range(total_steps):
                _nvpush(f"denoise_step_{step_idx}")
                t_vid = sched_video.timesteps[min(step_idx, len(sched_video.timesteps) - 1)]
                action_active = step_idx < len(sched_action.timesteps)
                t_act = (
                    sched_action.timesteps[step_idx]
                    if action_active
                    else torch.zeros_like(t_vid)
                )
                action_only_phase = use_action_only_phase and (step_idx >= joint_steps)
                if action_only_phase and frozen_z is None:
                    frozen_z = z.detach()
                z_input = frozen_z if action_only_phase else z

                # === Optimization A: hoist B15 fast path BEFORE outer eager work ===
                # If B15 path is eligible, skip the redundant outer patch_embed/encode_action/
                # time_embed lookup (they are re-done inside _denoise_step_core).
                _use_b15_core_early = (
                    not self.config.use_mot
                    and not (use_action_only_phase and step_idx >= joint_steps)
                )
                if _use_b15_core_early:
                    # Config-derived seq lens (no patch_embed/encode_action needed)
                    _v_seq_len_cfg = int(K) * int(self.frame_seqlen)
                    # spf = state_tokens_per_frame is the single source of truth.
                    _state_tokens = int(self.config.state_tokens_per_frame)
                    _a_seq_len_cfg = _state_tokens + int(K) * int(self.config.action_tokens_per_frame)
                    self._ensure_freqs_device()
                    # Ensure Plan Z step embeddings populated (prewarm should have done this)
                    if step_idx == 0:
                        self._ensure_step_embeddings(
                            sched_video, sched_action, total_steps, B, _v_seq_len_cfg, _a_seq_len_cfg,
                        )
                    # F-beta precompute real-paired freqs once per chunk
                    if step_idx == 0 and self._b15_full_freqs_real is None:
                        _video_freqs_cplx = self._create_video_freqs(
                            K,
                            start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
                        )
                        _full_freqs_cplx = self._build_chunk_ar_freqs(
                            clean_video_freqs=_video_freqs_cplx.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                            noisy_video_freqs=_video_freqs_cplx,
                            block_latent_counts=[K],
                            include_value=False,
                        )
                        self._b15_full_freqs_real = torch.view_as_real(_full_freqs_cplx.contiguous())
                    # Closure for DiTCache; calls _denoise_step_core which does ALL pre-core eager
                    # work (patch_embed, encode_action, project_in, time-embed lookup) internally.
                    def _predict_velocities():
                        v_pred_patch_local, a_pred_local = self._denoise_step_core(
                            z_input, state, y, t5_ctx, step_idx, K, B, mode,
                            self._b15_full_freqs_real,
                        )
                        v_pred_unpatched_local = self._unpatchify_video_tokens(v_pred_patch_local, K, B)
                        return v_pred_unpatched_local, a_pred_local
                    v_pred_unpatched, a_pred = _predict_velocities()
                    # === Optimization B: inline Euler scheduler step ===
                    _scheduler_name = str(getattr(self.config, "inference_scheduler", "euler")).lower()
                    _nvpush("scheduler_step")
                    if _scheduler_name == "euler":
                        _sigma_v = sched_video.sigmas[step_idx].to(z.dtype)
                        _sigma_v_next = sched_video.sigmas[step_idx + 1].to(z.dtype)
                        z = z + (_sigma_v_next - _sigma_v) * v_pred_unpatched
                        if action_active:
                            _sigma_a = sched_action.sigmas[step_idx].to(y.dtype)
                            _sigma_a_next = sched_action.sigmas[step_idx + 1].to(y.dtype)
                            y = y + (_sigma_a_next - _sigma_a) * a_pred
                    else:
                        z = self._step_video_scheduler(sched_video, v_pred_unpatched, t_vid, z)
                        if action_active:
                            y = sched_action.step(a_pred, t_act, y, return_dict=False)[0]
                    _nvpop()  # scheduler_step
                    _nvpop()  # denoise_step_{step_idx}
                    continue  # skip outer eager body

                # === Non-B15 path (original code): outer eager pre-core compute ===
                # Prepare video tokens
                _nvpush("patch_embed")
                z_tokens = self._video_patch_embed(z_input)  # [B, K*frame_seqlen, wan_dim]
                v_seq_len = z_tokens.shape[1]
                _nvpop()  # patch_embed

                # Prepare action register tokens
                _nvpush("encode_action")
                if state is not None and self.config.use_state:
                    state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                    a_tokens = self.action_expert.input_encoder(state_chunk, y)
                else:
                    a_tokens = self.action_expert.input_encoder(None, y)

                if not self.config.use_mot:
                    a_tokens = self.action_expert.projection.project_in(a_tokens)
                a_seq_len = a_tokens.shape[1]
                _nvpop()  # encode_action

                # Time embeddings: video and action use DIFFERENT timesteps
                # Plan Z: pre-fill step_idx-indexed list cache on step 0;
                # subsequent steps do O(1) list lookup (no .item, no dict.get).
                _nvpush("time_embed")
                if not self.config.use_mot:
                    if step_idx == 0:
                        self._ensure_step_embeddings(
                            sched_video, sched_action, total_steps, B, v_seq_len, a_seq_len,
                        )
                    e0_combined = self._step_emb_cache_v_e0[step_idx]
                    v_emb = self._step_emb_cache_v_emb[step_idx]
                    a_emb = self._step_emb_cache_a_emb[step_idx]
                else:
                    v_t_val = t_vid.to(device=self.device).reshape(1, 1).expand(B, v_seq_len)
                    v_emb, v_e0 = self._video_time_embed(v_t_val)
                    a_t_val = t_act.to(device=self.device).reshape(1, 1).expand(B, a_seq_len)
                    a_emb, a_e0 = self._action_time_embed(a_t_val)
                _nvpop()  # time_embed

                # Forward through layers
                self._ensure_freqs_device()

                if not self.config.use_mot:
                    # Simple mode: concatenated tokens (no value tokens in policy/wm inference)
                    if action_only_phase:
                        # mode=video_idm suffix branch: frozen current video KV + current action tokens.
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
                        # mode=joint/video_idm prefix branch: current chunk layout is [action | video].
                        _nvpush("rope_freqs")
                        x = self._cat_chunk_tokens(a_tokens, z_tokens, dim=1)
                        current_video_freqs = self._create_video_freqs(
                            K,
                            start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
                        )
                        current_freqs = self._build_chunk_ar_freqs(
                            clean_video_freqs=z_tokens.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                            noisy_video_freqs=current_video_freqs,
                            block_latent_counts=[K],
                            include_value=False,
                        )
                        # Calculate action_register_length (state + action tokens)
                        action_register_length = a_seq_len if a_seq_len > 0 else None
                        _nvpop()  # rope_freqs

                        _nvpush("transformer")
                        for layer_idx in range(self.config.num_layers):
                            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                            x = self._forward_layer_simple(
                                x, e0_combined, layer_idx, t5_ctx, ar_mask=None,
                                kv_cache=kv_cache, v_seq_len=v_seq_len,
                                is_training=False,
                                action_register_length=action_register_length,
                                full_freqs=current_freqs,
                                attention_mode=("video_idm" if mode in {"video_idm", "loop"} else "joint"),
                            )
                        _nvpop()  # transformer

                        # Split output (layout-aware: act_out / vid_out come
                        # from the appropriate slices of the concat'd chunk).
                        _nvpush("split_outputs")
                        act_slice, vid_slice = self._chunk_token_slices(a_seq_len, v_seq_len)
                        act_out = self.action_expert.projection.project_out(x[:, act_slice])
                        vid_out = x[:, vid_slice]
                        _nvpop()  # split_outputs
                else:
                    if action_only_phase:
                        # mode=video_idm suffix branch: frozen current video KV + current action tokens.
                        if frozen_video_kv is None:
                            frozen_video_kv = self._build_frozen_video_layer_kv(
                                z_tokens=z_tokens,
                                t_vid=t_vid,
                                t5_ctx=t5_ctx,
                            )
                        act_out = self._forward_action_only_cached(
                            action_tokens=a_tokens,
                            t_act=t_act,
                            frozen_video_kv=frozen_video_kv,
                            t5_ctx=t5_ctx,
                            K=K,
                        )
                        vid_out = None
                    else:
                        # mode=joint/video_idm prefix branch: separate video/action tokens.
                        # MoT mode: separate tokens (no value tokens in policy/wm inference)
                        video_tokens = z_tokens
                        action_tokens = a_tokens
                        current_video_freqs = self._create_video_freqs(
                            K,
                            start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
                        )
                        current_freqs = self._build_chunk_ar_freqs(
                            clean_video_freqs=z_tokens.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                            noisy_video_freqs=current_video_freqs,
                            block_latent_counts=[K],
                            include_value=False,
                        )

                        for layer_idx in range(self.config.num_layers):
                            kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                            video_tokens, action_tokens, _ = self._forward_layer_mot(
                                video_tokens, action_tokens, v_e0, a_e0, layer_idx, t5_ctx, ar_mask=None,
                                kv_cache=kv_cache, v_seq_len=v_seq_len,
                                is_training=False,
                                full_freqs=current_freqs,
                                attention_mode=("video_idm" if mode in {"video_idm", "loop"} else "joint"),
                            )

                        vid_out = video_tokens
                        act_out = action_tokens

                # Predict velocities
                v_pred_unpatched = None
                if not action_only_phase:
                    _nvpush("video_head")
                    v_pred = self.video_model.wan_model.head(vid_out, v_emb)
                    v_pred_unpatched = self._unpatchify_video_tokens(v_pred, K, B)
                    _nvpop()  # video_head

                _nvpush("action_decoder")
                a_pred = self.action_expert.decoder(act_out, a_emb)
                if self.config.state_tokens_per_frame > 0:
                    a_pred = a_pred[:, self.config.state_tokens_per_frame:, :]
                _nvpop()  # action_decoder

                # Scheduler step: video and action use their respective schedulers
                _nvpush("scheduler_step")
                if not action_only_phase:
                    z = self._step_video_scheduler(sched_video, v_pred_unpatched, t_vid, z)
                if action_active:
                    y = sched_action.step(
                        a_pred,
                        t_act,
                        y,
                        return_dict=False,
                    )[0]
                _nvpop()  # scheduler_step
                _nvpop()  # denoise_step_{step_idx}

        self._last_infer_K = K
        _nvpop()  # inference_step_streaming
        y = self._unscale_qpos20_rel_wrist_xyz(y)
        return z, y

    def forward_action_velocity_from_latent(
        self,
        video_latent: torch.Tensor,
        noisy_action: torch.Tensor,
        sigma_action: torch.Tensor,
        state: torch.Tensor = None,
        language_embeddings=None,
        restore_inference_state: Optional[Dict[str, Any]] = None,
        history_only: bool = False,
        video_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict action velocity for a noisy action chunk conditioned on a video latent.

        This is the action-only streaming forward used by MBRL/NFT training.
        By default the current chunk video latent is treated as a clean condition
        (t=0), while the action branch receives the provided action noise level.
        ``video_sigma`` instead tells the video tokens their noise level, for a
        caller that already noised ``video_latent``: a mode=joint model is only
        trained with video and action noised together, never clean video beside
        a noisy action.
        """
        if self.kv_caches is None and restore_inference_state is None:
            raise RuntimeError("forward_action_velocity_from_latent requires a pre-filled KV cache")

        saved_state = None
        if restore_inference_state is not None:
            saved_state = self.export_inference_state()
            self.load_inference_state(restore_inference_state)

        try:
            # [B,C,T,H,W] -> T at dim 2; [B,V,C,T,H,W] -> T at dim 3 (do not use shape[2] for 6D).
            K = self._latent_num_frames(video_latent)
            B = int(video_latent.shape[0])
            sigma_action = sigma_action.to(device=self.device, dtype=self.dtype).reshape(-1)
            if sigma_action.numel() == 1 and B > 1:
                sigma_action = sigma_action.expand(B)
            if sigma_action.shape[0] != B:
                raise ValueError(f"sigma_action batch mismatch: got {sigma_action.shape[0]}, expected {B}")

            t5_ctx = self._t5_embed(language_embeddings) if language_embeddings is not None else None

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                mode = self._mode()
                if history_only or mode == "action_only":
                    action_pred = self._predict_action_velocity_from_history(
                        noisy_action.to(device=self.device, dtype=self.dtype),
                        sigma_action,
                        t5_ctx,
                        state=state,
                    )
                    self._last_infer_K = K
                    return action_pred

                z_tokens = self._video_patch_embed(video_latent.to(self.dtype))
                v_seq_len = z_tokens.shape[1]

                if state is not None and self.config.use_state:
                    state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                    a_tokens = self.action_expert.input_encoder(state_chunk.to(self.dtype), noisy_action.to(self.dtype))
                else:
                    a_tokens = self.action_expert.input_encoder(None, noisy_action.to(self.dtype))

                if not self.config.use_mot:
                    a_tokens = self.action_expert.projection.project_in(a_tokens)
                a_seq_len = a_tokens.shape[1]

                action_timestep = (sigma_action * self.scheduler.num_train_timesteps).to(self.dtype)

                if video_sigma is None:
                    video_timestep = torch.zeros(B, device=self.device, dtype=self.dtype)
                else:
                    video_timestep = (
                        video_sigma.to(device=self.device, dtype=self.dtype).reshape(-1).expand(B)
                        * self.scheduler.num_train_timesteps
                    ).to(self.dtype)

                if not self.config.use_mot:
                    v_t_vals = video_timestep.view(B, 1).expand(-1, v_seq_len)
                    a_t_vals = action_timestep.view(B, 1).expand(-1, a_seq_len)
                    combined_t_vals = self._cat_chunk_tokens(a_t_vals, v_t_vals, dim=1)
                    _, e0_combined = self._video_time_embed(combined_t_vals)
                    a_emb, _ = self._action_time_embed(a_t_vals)

                    x = self._cat_chunk_tokens(a_tokens, z_tokens, dim=1)
                    current_video_freqs = self._create_video_freqs(
                        K,
                        start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
                    )
                    current_freqs = self._build_chunk_ar_freqs(
                        clean_video_freqs=z_tokens.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                        noisy_video_freqs=current_video_freqs,
                        block_latent_counts=[K],
                        include_value=False,
                    )
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
                            full_freqs=current_freqs,
                            attention_mode=("video_idm" if mode in {"video_idm", "loop"} else "joint"),
                        )
                    act_slice, _ = self._chunk_token_slices(a_seq_len, v_seq_len)
                    act_out = self.action_expert.projection.project_out(x[:, act_slice])
                else:
                    v_t_vals = video_timestep.view(B, 1).expand(-1, v_seq_len)
                    v_emb, v_e0 = self._video_time_embed(v_t_vals)
                    a_t_vals = action_timestep.view(B, 1).expand(-1, a_seq_len)
                    a_emb, a_e0 = self._action_time_embed(a_t_vals)
                    current_video_freqs = self._create_video_freqs(
                        K,
                        start_frame=self._inference_video_start_frame(self._kv_cache_valid_len_int() if self.kv_caches else 0),
                    )
                    current_freqs = self._build_chunk_ar_freqs(
                        clean_video_freqs=z_tokens.new_zeros((0, 1, self.freqs_action.shape[-1]), dtype=self.freqs_action.dtype),
                        noisy_video_freqs=current_video_freqs,
                        block_latent_counts=[K],
                        include_value=False,
                    )

                    video_tokens = z_tokens
                    action_tokens = a_tokens
                    for layer_idx in range(self.config.num_layers):
                        kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                        video_tokens, action_tokens, _ = self._forward_layer_mot(
                            video_tokens,
                            action_tokens,
                            v_e0,
                            a_e0,
                            layer_idx,
                            t5_ctx,
                            ar_mask=None,
                            kv_cache=kv_cache,
                            v_seq_len=v_seq_len,
                            is_training=False,
                            full_freqs=current_freqs,
                            attention_mode=("video_idm" if mode in {"video_idm", "loop"} else "joint"),
                        )
                    act_out = action_tokens

                action_pred = self.action_expert.decoder(act_out, a_emb)
                if self.config.state_tokens_per_frame > 0:
                    action_pred = action_pred[:, self.config.state_tokens_per_frame:, :]

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
            actions = self._scale_qpos20_rel_wrist_xyz(actions.to(device=self.device, dtype=torch.float32)).to(self.dtype)

            if state is not None and self.config.use_state:
                state_chunk = state[:, :1] if state.dim() == 3 else state.unsqueeze(1)
                a_tokens = self.action_expert.input_encoder(state_chunk, actions)
            else:
                a_tokens = self.action_expert.input_encoder(None, actions)

            if not self.config.use_mot:
                a_tokens = self.action_expert.projection.project_in(a_tokens)
            a_seq_len = a_tokens.shape[1]

            # Chunk-level value query tokens: one group per current chunk.
            val_tokens = self.value_tokens.expand(B, 1)  # [B, nvpf, wan_dim]

            # Time embeddings: t=0 for all (value readout from clean/denoised context)
            self._ensure_freqs_device()
            if not self.config.use_mot:
                # === use_mot=False: trunk = [video | state+action], value separate ===
                v_t_vals = torch.zeros(B, v_seq_len, device=self.device)
                a_t_vals = torch.zeros(B, a_seq_len, device=self.device)
                mode = self._mode()
                combined_t_vals = self._cat_chunk_tokens(a_t_vals, v_t_vals, dim=1)
                _, e0_combined = self._video_time_embed(combined_t_vals)

                # Value AdaLN: t=0
                val_t_vals = torch.zeros(B, nvpf, device=self.device)
                _, val_e0 = self._video_time_embed(val_t_vals)

                x = self._cat_chunk_tokens(a_tokens, z_tokens, dim=1)
                action_register_length = a_seq_len if a_seq_len > 0 else None

                # Tokens per physical frame (video + state + action)
                vbs = self.frame_seqlen  # video tokens per frame
                spf = self.config.state_tokens_per_frame
                apf = self.config.action_tokens_per_frame
                for layer_idx in range(self.config.num_layers):
                    blk = self.video_model.wan_model.blocks[layer_idx]

                    # --- Trunk forward (same as _forward_layer_simple inference) ---
                    mod = (blk.modulation.unsqueeze(0) + e0_combined).chunk(6, dim=2)
                    norm_x = (blk.norm1(x) * (1 + mod[1].squeeze(2)) + mod[0].squeeze(2)).to(self.dtype)
                    n = self.video_model.wan_model.num_heads
                    d = self.wan_dim // n
                    q_trunk, k_trunk, v_trunk = _wan_self_qkv(blk.self_attn, norm_x, B, norm_x.shape[1], n, d)

                    # RoPE for trunk
                    cache_seq_len = self._kv_cache_valid_len_int() if self.kv_caches is not None else 0
                    inf_video_freqs = self._create_video_freqs(
                        K,
                        start_frame=self._inference_video_start_frame(cache_seq_len),
                    )
                    full_freqs = self._build_chunk_ar_freqs(
                        clean_video_freqs=self.freqs_action.new_zeros(
                            (0, 1, self.freqs_action.shape[-1])
                        ),
                        noisy_video_freqs=inf_video_freqs,
                        block_latent_counts=[K],
                        include_value=False,
                    )
                    q_trunk = rope_action_apply(
                        q_trunk, full_freqs, self.freqs_action, self.freqs_state, None
                    ).type_as(v_trunk)
                    k_trunk = rope_action_apply(
                        k_trunk, full_freqs, self.freqs_action, self.freqs_state, None
                    ).type_as(v_trunk)

                    # Trunk attention with KV cache
                    kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None
                    if kv_cache is not None:
                        cached_k, cached_v = kv_cache
                        cache_len = cached_k.shape[1]
                    else:
                        cache_len = 0
                    visibility = None
                    if mode in {"video_idm", "action_only", "loop"}:
                        visibility_mode = "video_idm" if mode in {"video_idm", "loop"} else "fdm"
                        visibility = self._current_chunk_visibility_mask(
                            mode=visibility_mode,
                            query_len=q_trunk.shape[1],
                            key_len=cache_len + k_trunk.shape[1],
                            cache_len=cache_len,
                            video_len=v_seq_len,
                            device=q_trunk.device,
                        )
                    if kv_cache is not None:
                        y_trunk = self._fixed_cache_sdpa(
                            q_trunk, cached_k, cached_v, k_trunk, v_trunk, visibility=visibility
                        ).flatten(2)
                    else:
                        y_trunk = flash_attention(q_trunk, k_trunk, v_trunk, k_lens=None).flatten(2)

                    y_trunk = blk.self_attn.o(y_trunk)
                    x = x + y_trunk * mod[2].squeeze(2)
                    if t5_ctx is not None:
                        x = x + blk.cross_attn(blk.norm3(x), t5_ctx, context_lens=None)
                    ffn_in = (blk.norm2(x) * (1 + mod[4].squeeze(2)) + mod[3].squeeze(2)).to(self.dtype)
                    x = x + blk.ffn(ffn_in) * mod[5].squeeze(2)

                    # --- Value forward (separate chunk-level attention) ---
                    val_mod = (blk.modulation.unsqueeze(0) + val_e0).chunk(6, dim=2)
                    norm_val = (blk.norm1(val_tokens) * (1 + val_mod[1].squeeze(2)) + val_mod[0].squeeze(2)).to(self.dtype)
                    q_val, k_val, v_val = _wan_self_qkv(blk.self_attn, norm_val, B, nvpf, n, d)

                    # Chunk-level attention: value sees clean cache + full current chunk trunk + own value KV.
                    if kv_cache is not None:
                        current_k_val = torch.cat([k_trunk, k_val], dim=1)
                        current_v_val = torch.cat([v_trunk, v_val], dim=1)
                        y_val = self._fixed_cache_sdpa(q_val, cached_k, cached_v, current_k_val, current_v_val)
                    else:
                        attn_k_val = torch.cat([k_trunk, k_val], dim=1)
                        attn_v_val = torch.cat([v_trunk, v_val], dim=1)
                        y_val = flash_attention(q_val, attn_k_val, attn_v_val, k_lens=None)

                    y_val = y_val.reshape(B, nvpf, -1)

                    y_val = blk.self_attn.o(y_val)
                    val_tokens = val_tokens + y_val * val_mod[2].squeeze(2)
                    # Non-MoT training interleaves value tokens into x, so the
                    # whole sequence receives this residual before the FFN.
                    val_tokens = apply_value_cross_attention(
                        blk, val_tokens, t5_ctx
                    )
                    ffn_val_in = (blk.norm2(val_tokens) * (1 + val_mod[4].squeeze(2)) + val_mod[3].squeeze(2)).to(self.dtype)
                    val_tokens = val_tokens + blk.ffn(ffn_val_in) * val_mod[5].squeeze(2)

            else:
                # === use_mot=True: video/action separate, value separate ===
                v_t_val = torch.zeros(B, v_seq_len, device=self.device)
                v_emb, v_e0 = self._video_time_embed(v_t_val)
                a_t_val = torch.zeros(B, a_seq_len, device=self.device)
                a_emb, a_e0 = self._action_time_embed(a_t_val)
                val_t_vals = torch.zeros(B, nvpf, device=self.device, dtype=self.dtype)
                _, val_e0 = self._video_time_embed(val_t_vals)

                video_tokens = z_tokens
                action_tokens = a_tokens

                vbs = self.frame_seqlen
                spf = self.config.state_tokens_per_frame
                apf = self.config.action_tokens_per_frame

                for layer_idx in range(self.config.num_layers):
                    kv_cache = (self.kv_caches[layer_idx][0], self.kv_caches[layer_idx][1]) if self.kv_caches else None

                    # --- Compute trunk K/V BEFORE layer forward (for value attention) ---
                    blk = self.video_model.wan_model.blocks[layer_idx]
                    action_blk = self.action_expert.blocks[layer_idx]
                    n = self.video_model.wan_model.num_heads
                    d = self.wan_dim // n

                    v_mod = (blk.modulation.unsqueeze(0) + v_e0).chunk(6, dim=2)
                    a_mod_l = (action_blk.modulation.unsqueeze(0) + a_e0).chunk(6, dim=2)
                    norm_video_pre = (blk.norm1(video_tokens) * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)).to(self.dtype)
                    norm_action_pre = (action_blk.norm1(action_tokens) * (1 + a_mod_l[1].squeeze(2)) + a_mod_l[0].squeeze(2)).to(self.dtype)

                    # Video K/V via WAN projections (wan_dim -> n*d)
                    k_vid = blk.self_attn.norm_k(blk.self_attn.k(norm_video_pre)).view(B, -1, n, d)
                    v_vid = blk.self_attn.v(norm_video_pre).view(B, -1, n, d)

                    # Action K/V via MoT wan_action_qkv (action_dim -> n*d head space)
                    a_kv = torch.einsum("BTD,KNDE->KBTNE", norm_action_pre, action_blk.wan_action_qkv)
                    a_k_h = action_blk.wan_action_norm_k(a_kv[1].flatten(-2)).view(B, -1, n, d)
                    a_v_h = a_kv[2].view(B, -1, n, d)

                    # Concatenate in head space: [video | action]
                    k_trunk = torch.cat([k_vid, a_k_h], dim=1)
                    v_trunk_layer = torch.cat([v_vid, a_v_h], dim=1)

                    # --- Trunk forward (MoT, no value tokens) ---
                    video_tokens, action_tokens, _ = self._forward_layer_mot(
                        video_tokens, action_tokens, v_e0, a_e0, layer_idx, t5_ctx, ar_mask=None,
                        kv_cache=kv_cache, v_seq_len=v_seq_len,
                        is_training=False,
                    )

                    # --- Value forward (separate chunk-level attention) ---
                    val_mod = (blk.modulation.unsqueeze(0) + val_e0).chunk(6, dim=2)
                    norm_val = (blk.norm1(val_tokens) * (1 + val_mod[1].squeeze(2)) + val_mod[0].squeeze(2)).to(self.dtype)
                    q_val, k_val, v_val = _wan_self_qkv(blk.self_attn, norm_val, B, nvpf, n, d)

                    if kv_cache is not None:
                        cached_k, cached_v = kv_cache
                        current_k_val = torch.cat([k_trunk, k_val], dim=1)
                        current_v_val = torch.cat([v_trunk_layer, v_val], dim=1)
                        y_val = self._fixed_cache_sdpa(q_val, cached_k, cached_v, current_k_val, current_v_val)
                    else:
                        attn_k_val = torch.cat([k_trunk, k_val], dim=1)
                        attn_v_val = torch.cat([v_trunk_layer, v_val], dim=1)
                        y_val = flash_attention(q_val, attn_k_val, attn_v_val, k_lens=None)

                    y_val = y_val.reshape(B, nvpf, -1)

                    y_val = blk.self_attn.o(y_val)
                    val_tokens = val_tokens + y_val * val_mod[2].squeeze(2)
                    ffn_val_in = (blk.norm2(val_tokens) * (1 + val_mod[4].squeeze(2)) + val_mod[3].squeeze(2)).to(self.dtype)
                    val_tokens = val_tokens + blk.ffn(ffn_val_in) * val_mod[5].squeeze(2)

            # Value prediction: current chunk only.
            val_hidden = val_tokens.reshape(B, 1, nvpf, -1)
            val_logits = self.value_head(val_hidden.to(self.dtype).reshape(B * nvpf, -1))
            val_logits = val_logits.reshape(B, 1, nvpf, -1).mean(dim=2)
            value_pred = expected_value(
                val_logits, self.config.value_num_bins,
                self.config.value_range[0], self.config.value_range[1],
                fp32=self.config.value_fp32,
            )  # [B, 1]

        return value_pred
