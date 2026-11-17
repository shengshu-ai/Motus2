"""Opt-in, autograd-capable regions for the actually called training layer.

No detached/copied parameters, altered math mode, attention, or task selection.
Keep the six-way modulation unmaterialized: each region reads only its slices.
"""
from functools import lru_cache
import torch


def _modulation_part(block, e0, index):
    from .training_compact_e0 import CompactE0, modulation_part
    if isinstance(e0, CompactE0):
        return modulation_part(block, e0, index)
    # Preserve the original addition order/dtype, before the 1+scale operation.
    part = e0[index] if isinstance(e0, tuple) else e0[:, :, index, :]
    return block.modulation[:, index, :].unsqueeze(0) + part


def training_pre_attention(block, x, e0, dtype):
    shift = _modulation_part(block, e0, 0)
    scale = _modulation_part(block, e0, 1)
    return (block.norm1(x) * (1 + scale) + shift).to(dtype)


def training_attention_residual(block, x, y, e0):
    gate = _modulation_part(block, e0, 2)
    return x + block.self_attn.o(y) * gate


def training_ffn_residual(block, x, e0, dtype):
    shift = _modulation_part(block, e0, 3)
    scale = _modulation_part(block, e0, 4)
    gate = _modulation_part(block, e0, 5)
    ffn_in = (block.norm2(x) * (1 + scale) + shift).to(dtype)
    return x + block.ffn(ffn_in) * gate


def make_training_region_runners(backend='inductor'):
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    # AOTAutograd retains true parameter gradients. Its default partitioner may
    # rematerialize pointwise operations; no policy asks it to rerun GEMMs.
    options = {'triton.cudagraphs': False, 'emulate_precision_casts': True}
    kwargs = dict(backend=backend, dynamic=True, fullgraph=True)
    if backend == 'inductor':
        kwargs['options'] = options
    return tuple(torch.compile(fn, **kwargs) for fn in (
        training_pre_attention, training_attention_residual, training_ffn_residual,
    ))


@lru_cache(maxsize=1)
def training_region_runners():
    return make_training_region_runners()
