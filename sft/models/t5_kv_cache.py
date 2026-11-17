"""T5 cross-attention K/V cache for Motus2 Wan blocks.

Within a generated chunk every WanCrossAttention block sees the same context
tensor across denoise steps. Refresh on each new processed context. The K/V
projections (`norm_k(k(ctx))`, `v(ctx)`) are recomputed every call. Caching
them is bit-exact: same input, same weights -> same output.

Usage:
    install_t5_kv_cache(model.video_model.wan_model)   # patch each cross-attn once
    populate_t5_kv_cache(model.video_model.wan_model, t5_ctx)  # call when t5_ctx changes
"""
from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def _is_wan_cross_attention(module) -> bool:
    return type(module).__name__ == "WanCrossAttention"


import os as _os
import torch.nn.functional as _F
import types as _types


def _tensor_stamp(tensor):
    # Inference tensors have no mutation counter. They must use the original
    # forward rather than a cache whose validity we cannot establish.
    if torch.is_inference(tensor):
        return None
    return (tensor._version, tensor.data_ptr(), tensor.shape, tensor.stride(),
            tensor.device, tensor.dtype)


def _autocast_stamp(device_type):
    enabled = torch.is_autocast_enabled(device_type)
    return (enabled, torch.get_autocast_dtype(device_type) if enabled else None)


def _kv_parameters(module):
    return tuple(p for layer in (module.k, module.v, module.norm_k)
                 for p in layer.parameters())


def _cache_matches(module, context):
    if context is not module._motus2_t5_context:
        return False
    if _tensor_stamp(context) != module._motus2_t5_context_stamp:
        return False
    if _autocast_stamp(context.device.type) != module._motus2_t5_autocast:
        return False
    params = _kv_parameters(module)
    old = module._motus2_t5_parameters
    return (len(params) == len(old)
            and all(p is previous and _tensor_stamp(p) == stamp
                    for p, (previous, stamp) in zip(params, old)))


def _t5_cached_cross_attn_forward(self, x, context, context_lens):
    """Method binding (NOT closure) — same function object shared across all
    WanCrossAttention modules. Avoids dynamo closure-cell specialization that
    would cause one recompile per block.
    """
    # Never detach training/gradient-enabled work through an old inference
    # cache. Python mutation guards also stay outside compiled/captured graphs.
    if (self.training or torch.is_grad_enabled() or torch.compiler.is_compiling()
            or (x.is_cuda and torch.cuda.is_current_stream_capturing())):
        return self._motus2_original_forward(x, context, context_lens)
    k_cache = self._motus2_t5_k_cache
    v_cache = self._motus2_t5_v_cache
    if k_cache is None or v_cache is None or not self._motus2_t5_cache_valid:
        return self._motus2_original_forward(x, context, context_lens)
    if not _cache_matches(self, context):
        self._motus2_t5_cache_valid = False
        return self._motus2_original_forward(x, context, context_lens)
    if x.shape[0] != k_cache.shape[0]:
        return self._motus2_original_forward(x, context, context_lens)
    if context_lens is not None and _os.environ.get("MOTUS_USE_SDPA", "0") == "1":
        # The unmasked SDPA shortcut cannot represent variable text lengths.
        return self._motus2_original_forward(x, context, context_lens)

    b = x.size(0)
    q = self.norm_q(self.q(x)).view(b, -1, self.num_heads, self.head_dim)
    if _os.environ.get("MOTUS_USE_SDPA", "0") == "1":
        out = _F.scaled_dot_product_attention(
            q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2)
        ).transpose(1, 2).contiguous()
    else:
        from wan.modules.attention import flash_attention
        out = flash_attention(q, k_cache, v_cache, k_lens=context_lens)
    return self.o(out.flatten(2))


def install_t5_kv_cache(model) -> dict:
    """Patch all WanCrossAttention modules to optionally read cached T5 K/V."""
    counts = {"patched": 0, "skipped": 0}
    for _name, module in model.named_modules():
        if not _is_wan_cross_attention(module):
            continue
        if getattr(module, "_motus2_t5_kv_patched", False):
            counts["skipped"] += 1
            continue
        module._motus2_t5_k_cache = None
        module._motus2_t5_v_cache = None
        module._motus2_t5_cache_valid = False
        module._motus2_t5_context = None
        module._motus2_t5_parameters = ()
        module._motus2_original_forward = module.forward
        module.forward = _types.MethodType(_t5_cached_cross_attn_forward, module)
        module._motus2_t5_kv_patched = True
        counts["patched"] += 1

    if counts["patched"] + counts["skipped"] == 0:
        raise RuntimeError("T5 K/V cache installed zero WanCrossAttention modules")
    # M18 fix: gate logger.info with torch.compiler.is_compiling() so dynamo
    # doesn't trip on the logger call during trace (this hook is also called
    # from _ensure_t5_kv_cache inside _t5_embed, which is in the compile region
    # when inference_step_streaming is compiled).
    if not torch.compiler.is_compiling():
        logger.info("[motus2.accel] T5 K/V cache hooks: %s", counts)
    return counts


@torch.no_grad()
def populate_t5_kv_cache(model, t5_ctx: torch.Tensor | None) -> dict:
    """Populate pointer-stable K/V buffers from a processed Wan T5 context."""
    invalidate_t5_kv_cache(model)
    if (t5_ctx is None or torch.compiler.is_compiling()
            or (t5_ctx.is_cuda and torch.cuda.is_current_stream_capturing())):
        return {"populated": 0, "skipped": 0}

    counts = {"populated": 0, "skipped": 0}
    saw_hook = False
    for _name, module in model.named_modules():
        if not _is_wan_cross_attention(module) or not getattr(module, "_motus2_t5_kv_patched", False):
            continue
        saw_hook = True

        context_stamp = _tensor_stamp(t5_ctx)
        parameters = tuple((p, _tensor_stamp(p)) for p in _kv_parameters(module))
        if context_stamp is None or any(stamp is None for _, stamp in parameters):
            counts["skipped"] += 1
            continue

        b = int(t5_ctx.shape[0])
        n = int(module.num_heads)
        d = int(module.head_dim)
        k = module.norm_k(module.k(t5_ctx)).view(b, -1, n, d).contiguous()
        v = module.v(t5_ctx).view(b, -1, n, d).contiguous()

        if (module._motus2_t5_k_cache is None or module._motus2_t5_k_cache.shape != k.shape
                or module._motus2_t5_k_cache.dtype != k.dtype
                or module._motus2_t5_k_cache.device != k.device):
            module._motus2_t5_k_cache = torch.empty_like(k)
            module._motus2_t5_v_cache = torch.empty_like(v)
        module._motus2_t5_k_cache.copy_(k)
        module._motus2_t5_v_cache.copy_(v)
        module._motus2_t5_context = t5_ctx
        module._motus2_t5_context_stamp = context_stamp
        module._motus2_t5_parameters = parameters
        module._motus2_t5_autocast = _autocast_stamp(t5_ctx.device.type)
        module._motus2_t5_cache_valid = True
        counts["populated"] += 1

    if not saw_hook:
        counts["skipped"] += 1
    return counts


def invalidate_t5_kv_cache(model) -> None:
    """Force re-population on next inference call (e.g. on language change)."""
    for _name, module in model.named_modules():
        if _is_wan_cross_attention(module) and getattr(module, "_motus2_t5_kv_patched", False):
            module._motus2_t5_cache_valid = False
            module._motus2_t5_context = None
            module._motus2_t5_parameters = ()
