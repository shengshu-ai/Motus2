"""T5 cross-attention K/V cache for Motus2 Wan blocks.

For one episode the language instruction is fixed, so every WanCrossAttention
block sees the same context tensor for ~30 chunks * ~10 denoise steps. The K/V
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


def _t5_cached_cross_attn_forward(self, x, context, context_lens):
    """Method binding (NOT closure) — same function object shared across all
    WanCrossAttention modules. Avoids dynamo closure-cell specialization that
    would cause one recompile per block.
    """
    k_cache = self._motus2_t5_k_cache
    v_cache = self._motus2_t5_v_cache
    if k_cache is None or v_cache is None or not self._motus2_t5_cache_valid:
        return self._motus2_original_forward(x, context, context_lens)
    if context is not None and (context.shape[0] != k_cache.shape[0] or context.shape[1] != k_cache.shape[1]):
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
    if t5_ctx is None:
        return {"populated": 0, "skipped": 0}

    counts = {"populated": 0, "skipped": 0}
    saw_hook = False
    for _name, module in model.named_modules():
        if not _is_wan_cross_attention(module) or not getattr(module, "_motus2_t5_kv_patched", False):
            continue
        saw_hook = True

        b = int(t5_ctx.shape[0])
        n = int(module.num_heads)
        d = int(module.head_dim)
        k = module.norm_k(module.k(t5_ctx)).view(b, -1, n, d).contiguous()
        v = module.v(t5_ctx).view(b, -1, n, d).contiguous()

        if module._motus2_t5_k_cache is None or module._motus2_t5_k_cache.shape != k.shape:
            module._motus2_t5_k_cache = torch.empty_like(k)
            module._motus2_t5_v_cache = torch.empty_like(v)
        module._motus2_t5_k_cache.copy_(k)
        module._motus2_t5_v_cache.copy_(v)
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
