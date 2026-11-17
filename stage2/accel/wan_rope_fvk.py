"""FlashVLA fvk rope_apply_bf16_to_bf16 fast path for Motus2 RoPE.

Replaces the upstream x.to(float64) -> view_as_complex -> multiply -> view_as_real
chain with a single fused bf16 kernel call when shapes/dtypes match. Verified
diff=0.0 vs the upstream complex math (the kernel implements exactly the same
rotation math in fp32 internally).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import torch

logger = logging.getLogger(__name__)

_KERNELS: Any = None
_KERNEL_ERROR: str | None = None
_FREQ_CACHE: dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _try_load_fvk() -> Any:
    global _KERNELS, _KERNEL_ERROR
    if _KERNELS is not None or _KERNEL_ERROR is not None:
        return _KERNELS
    if os.environ.get("MOTUS2_DISABLE_FLASHVLA", "0").lower() in {"1", "true", "yes", "on"}:
        _KERNEL_ERROR = "disabled by MOTUS2_DISABLE_FLASHVLA"
        return None
    try:
        from .flashvla_kernels import load_flashvla_kernels

        kernels, _ = load_flashvla_kernels()
        if not hasattr(kernels, "rope_apply_bf16_to_bf16"):
            raise RuntimeError("rope_apply_bf16_to_bf16 missing")
        _KERNELS = kernels
    except Exception as exc:
        _KERNEL_ERROR = str(exc)
        logger.warning("FlashVLA kernels unavailable for rope swap: %s", exc)
    return _KERNELS


def _stream_ptr() -> int:
    return int(torch.cuda.current_stream().cuda_stream)


def _split_complex(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache fp32 real/imag of a complex freqs tensor by (id, shape)."""
    if freqs.dim() == 3 and freqs.shape[1] == 1:
        f = freqs.squeeze(1)
    elif freqs.dim() == 2:
        f = freqs
    else:
        f = freqs.reshape(-1, freqs.shape[-1])
    key = (int(f.data_ptr()), int(f.shape[0]), int(f.shape[1]), int(f.numel()))
    hit = _FREQ_CACHE.get(key)
    if hit is not None:
        return hit
    re = f.real.to(torch.float32).contiguous()
    im = f.imag.to(torch.float32).contiguous()
    _FREQ_CACHE[key] = (re, im)
    return re, im


def fvk_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor | None:
    """Apply RoPE via the FlashVLA bf16 kernel; return None to signal fallback.

    Args:
        x: input tensor of shape [B, L, n_heads, head_dim] in any dtype.
        freqs: complex64 tensor of shape [seq_len, 1, head_dim/2] or
            [seq_len, head_dim/2] (or batched with dim 0 = B).

    Returns:
        bf16 tensor with the same shape as x, or None if the kernel cannot be
        used (caller should run the slow complex-multiply path instead).
    """
    kernels = _try_load_fvk()
    if kernels is None:
        return None
    if x.dim() != 4 or not x.is_cuda:
        return None
    B, L, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        return None
    # freqs supported shapes: [seq_len, 1, c_complex] or [seq_len, c_complex];
    # [1, seq_len, 1, c_complex] also accepted (drop batch).
    f = freqs
    if f.dim() == 4 and f.shape[0] == 1:
        f = f[0]
    if f.dim() == 3 and f.shape[1] == 1:
        f = f.squeeze(1)
    if f.dim() != 2 or f.shape[1] != head_dim // 2:
        return None
    if f.dtype not in (torch.complex64, torch.complex128):
        return None
    seq_len = int(f.shape[0])
    if seq_len > L:
        return None
    x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
    re, im = _split_complex(f)
    out = torch.empty_like(x_c)
    kernels.rope_apply_bf16_to_bf16(
        int(x_c.data_ptr()),
        int(re.data_ptr()),
        int(im.data_ptr()),
        int(out.data_ptr()),
        int(B),
        int(L),
        int(n_heads),
        int(head_dim),
        int(seq_len),
        _stream_ptr(),
    )
    return out


def is_available() -> bool:
    return _try_load_fvk() is not None


def clear_freq_cache() -> None:
    _FREQ_CACHE.clear()
