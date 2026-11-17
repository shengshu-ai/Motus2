"""Autograd-capable WAN training Q/K normalization and rotary fusion.

The public helpers return ``None`` outside the validated implementation scope.
Callers then retain their normal QKV/RoPE path. This is an implementation choice,
not a recipe option; in particular the raw-pointer inference kernels are unused.
"""

from functools import lru_cache
import logging

import torch
from torch.nn.modules import module as nn_module


logger = logging.getLogger(__name__)
_failed_signatures = set()


@lru_cache(maxsize=1)
def _compiler_supports_precision_casts():
    # Earlier Inductor versions can eliminate the intermediate BF16 rounding.
    try:
        return "emulate_precision_casts" in torch._inductor.list_options()
    except (AttributeError, ImportError):
        return False


def _supported_device(tensor):
    return (
        tensor.is_cuda
        and torch.cuda.get_device_capability(tensor.device) in ((10, 0), (10, 3))
        and _compiler_supports_precision_casts()
    )


def _real_frequencies(freqs, batch, length):
    """Mirror rope_action_apply's full-frequency broadcasting and padding."""
    if freqs.ndim == 3:
        if freqs.shape[0] < length:
            freqs = torch.cat([
                freqs,
                freqs[-1:].expand(length - freqs.shape[0], *freqs.shape[1:]).clone(),
            ])
        freqs = freqs.unsqueeze(0)
    elif freqs.ndim != 4:
        return None
    if freqs.shape[0] != batch and freqs.shape[1] == batch:
        freqs = freqs.permute(1, 0, 2, 3).contiguous()
    if freqs.shape[1] < length:
        freqs = torch.cat([
            freqs,
            freqs[:, -1:].expand(
                freqs.shape[0], length - freqs.shape[1], *freqs.shape[2:]
            ).clone(),
        ], dim=1)
    if freqs.shape[0] not in (1, batch) or freqs.shape[1] != length:
        return None
    return freqs.real.float().contiguous(), freqs.imag.float().contiguous()


def prepare_training_qk_frequencies(freqs, batch, length):
    """Prepare full RoPE once per training batch; never cache across batches."""
    if torch.compiler.is_compiling():
        return None
    if (
        freqs is None or not freqs.is_complex() or freqs.numel() == 0
        or not _supported_device(freqs) or freqs.requires_grad
    ):
        return None
    return _real_frequencies(freqs, batch, length)


def _norm_rope_bhld(raw, weight, real, imag, heads, eps):
    # WAN normalizes over the entire model width, not separately per head.
    x32 = raw.float()
    norm = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(raw.dtype)
    weighted = norm * weight
    pairs = weighted.view(
        *raw.shape[:2], heads, raw.shape[-1] // heads
    ).float().unflatten(-1, (raw.shape[-1] // heads // 2, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    out = torch.stack([a * real - b * imag, a * imag + b * real], -1)
    return out.flatten(-2).to(raw.dtype).permute(0, 2, 1, 3).contiguous()


def _qk_norm_rope(q, k, wq, wk, real, imag, heads, eps_q, eps_k):
    return (
        _norm_rope_bhld(q, wq, real, imag, heads, eps_q),
        _norm_rope_bhld(k, wk, real, imag, heads, eps_k),
    )


@lru_cache(maxsize=1)
def _compiled_qk_norm_rope():
    # Keep the FP32 norm -> BF16 -> weight -> FP32 RoPE -> BF16 boundaries.
    # AOTAutograd supplies the backward; no detached projections or weights.
    return torch.compile(
        _qk_norm_rope, dynamic=False, fullgraph=True,
        options={"emulate_precision_casts": True},
    )


def try_training_qkv_with_rope(self_attn, norm_x, real_freqs, heads):
    """Return BLHD views whose Q/K storage is already contiguous BHLD.

    Only ordinary WAN Q/K/V projections and unmodified WanRMSNorm are handled.
    The existing BLHD Flex wrapper then obtains contiguous Q/K without copying.
    """
    if torch.compiler.is_compiling():
        return None
    if (
        real_freqs is None or not _supported_device(norm_x)
        or norm_x.dtype != torch.bfloat16 or norm_x.ndim != 3
        or norm_x.shape[-1] != 3072 or heads != 24
        or norm_x.shape[1] % 128 or norm_x.shape[0] < 1
        or getattr(self_attn, "_motus2_qkv_fused", False)
    ):
        return None
    if any(getattr(nn_module, name, None) for name in (
        "_global_forward_pre_hooks", "_global_forward_hooks",
        "_global_backward_pre_hooks", "_global_backward_hooks",
    )):
        # Fusion bypasses norm.__call__, and compiler fallback retries QKV.
        # Global hooks must observe the original module calls exactly once.
        return None
    projections = (self_attn.q, self_attn.k, self_attn.v)
    if any(
        type(projection) is not torch.nn.Linear
        # All-BF16 WAN with FP32 master updates did not pass the original
        # optimizer gate. Keep its original QKV/RoPE implementation.
        or projection.weight.dtype != torch.float32
        or (projection.bias is not None and projection.bias.dtype != torch.float32)
        or "forward" in projection.__dict__
        or projection._forward_hooks or projection._forward_pre_hooks
        or projection._backward_pre_hooks or projection._backward_hooks
        for projection in projections
    ):
        # A compiler fallback can retry ordinary deterministic linear calls.
        # It must not execute stochastic adapters or module hooks twice.
        return None
    # Import only on the selected path: the model has already loaded WAN here.
    from wan.modules.model import WanRMSNorm
    norms = (self_attn.norm_q, self_attn.norm_k)
    if any(
        type(norm) is not WanRMSNorm
        or getattr(norm, "_motus2_bf16_patched", False)
        or "forward" in norm.__dict__
        or "_norm" in norm.__dict__
        or norm._forward_hooks or norm._forward_pre_hooks or norm._backward_hooks
        or norm._backward_pre_hooks
        # FP32 norm weights amplify backward rounding in AdamW; retain eager.
        or norm.weight.dtype != torch.bfloat16
        or norm.weight.device != norm_x.device
        or tuple(norm.weight.shape) != (3072,)
        for norm in norms
    ):
        return None
    real, imag = real_freqs
    if (
        real.shape != imag.shape or real.ndim != 4
        or real.shape[0] not in (1, norm_x.shape[0])
        or real.shape[1:] != (norm_x.shape[1], 1, 64)
        or real.dtype != torch.float32 or imag.dtype != torch.float32
        or real.device != norm_x.device or imag.device != norm_x.device
    ):
        return None
    signature = (
        norm_x.device.index, tuple(norm_x.shape), tuple(norm_x.stride()),
        norms[0].weight.dtype, norms[1].weight.dtype, tuple(real.shape),
    )
    if signature in _failed_signatures:
        return None
    q_raw, k_raw, v_raw = self_attn.q(norm_x), self_attn.k(norm_x), self_attn.v(norm_x)
    if q_raw.dtype != torch.bfloat16 or k_raw.dtype != torch.bfloat16 or v_raw.dtype != torch.bfloat16:
        return None
    from torch._dynamo.exc import BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported
    try:
        q, k = _compiled_qk_norm_rope()(
            q_raw, k_raw, norms[0].weight, norms[1].weight, real, imag,
            heads, norms[0].eps, norms[1].eps,
        )
    except (BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported) as exc:
        # A compiler limitation must not make a supported model untrainable.
        # No CUDA/memory/data errors are swallowed by this fallback.
        _failed_signatures.add(signature)
        logger.warning("Training Q/K fusion unavailable for %s: %s", signature, type(exc).__name__)
        return None
    batch, length, width = norm_x.shape
    return (
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
        v_raw.view(batch, length, heads, width // heads),
    )
