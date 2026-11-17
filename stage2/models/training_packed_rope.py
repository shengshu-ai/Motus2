"""Packed-layout RoPE dispatch with a general-layout fallback.
"""
import torch
from .training_rope import apply_training_real_rope


def eligibility(x, freqs, dtype, *, allow_cpu=False):
    if x.ndim != 4 or tuple(x.shape[2:]) != (24, 128):
        return 'requires BLH24D128'
    if x.shape[0] not in (1, 2, 4) or x.shape[1] <= 0:
        return 'requires nonempty B1/B2/B4'
    if x.numel() > 2**31 - 1:
        return 'requires positive int32 numel; larger shapes use original fallback'
    if x.dtype not in (torch.float32, torch.bfloat16):
        return 'requires FP32/BF16 input'
    if dtype not in (torch.float32, torch.bfloat16):
        return 'requires FP32/BF16 output'
    if not x.is_contiguous():
        return 'requires original contiguous BLHD'
    if freqs.requires_grad:
        return 'learnable frequencies use original autograd'
    if freqs.dtype != torch.float32 or not freqs.is_contiguous():
        return 'requires original contiguous FP32 real frequencies'
    if freqs.ndim != 5 or tuple(freqs.shape[1:]) != (x.shape[1], 1, 64, 2):
        return 'requires normalized frequency pairs'
    if freqs.shape[0] not in (1, x.shape[0]) or freqs.device != x.device:
        return 'frequency batch/device mismatch'
    if not allow_cpu and not x.is_cuda:
        return 'CUDA only outside explicit interpreter checks'
    return None


def launch_forward(x, freqs, dtype, block=1024, warps=4):
    import triton
    from .training_packed_rope_kernels import packed_rope_forward
    y = torch.empty(x.shape, device=x.device, dtype=dtype)
    packed_rope_forward[(triton.cdiv(x.numel(), block),)](
        x, freqs, y, x.numel(), x.shape[1], freqs.shape[0] == 1, block,
        num_warps=warps, enable_fp_fusion=False)
    return y


def launch_backward(grad, freqs, dtype, block=1024, warps=4):
    import triton
    from .training_packed_rope_kernels import packed_rope_backward
    grad = grad.contiguous()
    dx = torch.empty(grad.shape, device=grad.device, dtype=dtype)
    packed_rope_backward[(triton.cdiv(grad.numel(), block),)](
        grad, freqs, dx, grad.numel(), grad.shape[1], freqs.shape[0] == 1, block,
        num_warps=warps, enable_fp_fusion=False)
    return dx


class _PackedRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs, dtype, block, warps):
        ctx.save_for_backward(freqs)
        ctx.input_dtype = x.dtype
        ctx.block, ctx.warps = block, warps
        return launch_forward(x, freqs, dtype, block, warps)

    @staticmethod
    def backward(ctx, grad):
        freqs, = ctx.saved_tensors
        dx = launch_backward(grad, freqs, ctx.input_dtype, ctx.block, ctx.warps)
        return dx, None, None, None, None


def packed_training_rope(x, freqs, dtype, *, block=1024, warps=4,
                         allow_cpu_interpreter=False):
    if block not in (512, 1024, 2048, 4096) or warps not in (4, 8):
        raise ValueError('Unprepared tile/warps')
    if eligibility(x, freqs, dtype, allow_cpu=allow_cpu_interpreter):
        return apply_training_real_rope(x, freqs, dtype)
    if not x.is_cuda:
        import os
        if os.environ.get('TRITON_INTERPRET') != '1':
            raise RuntimeError('Explicit CPU interpreter required')
    return _PackedRotation.apply(x, freqs, dtype, block, warps)
