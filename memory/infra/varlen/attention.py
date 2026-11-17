"""Open varlen block-attention dispatch with a pluggable backend registry.

Built-in backends:

* ``flex_attention`` (default) - PyTorch FlexAttention over a compiled BlockMask.
* ``sdpa`` - dense ``scaled_dot_product_attention`` reference (small tests / CPU).

Additional backends (a Triton block-sparse kernel, or an FA4
wrapper) plug in via :func:`register_attention_backend` *without editing this
file*. They keep the same ``logical_block_mask`` + ``q_block_indices`` /
``kv_block_indices`` interface: a backend receives a :class:`VarlenAttentionInputs`
carrying BHLD tensors plus lazy mask/block-index materializers, and returns a
BHLD output. Additional backends must be installed and registered separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Static specializations cover grad/no-grad, VM sub-batches and packed lengths.
# Support both the older and newer Dynamo configuration names.
for _cache_key in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _cache_key):
        setattr(torch._dynamo.config, _cache_key,
                max(getattr(torch._dynamo.config, _cache_key), 64))

flex_attention_compiled = torch.compile(
    flex_attention, dynamic=False, fullgraph=True,
)

def BHLD_flex_with_mask(q, k, v, attention_mask=None):
    """Attend to contiguous [B, H, L, D] tensors without a layout round trip."""
    kernel_options = {}
    if (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and torch.cuda.get_device_capability(q.device) in {(10, 0), (10, 3)}
    ):
        # Blackwell's newer four-warp heuristic regresses this backward kernel.
        # Eight warps retain the same tiles and arithmetic (including mask),
        # and match the earlier compiler's choice for these training heads.
        kernel_options["bwd_num_warps"] = 8
    return flex_attention_compiled(
        q, k, v, block_mask=attention_mask, kernel_options=kernel_options,
    )

compiled_create_block_mask = torch.compile(create_block_mask)


def _normalize_block_sizes(block_sizes: int | Sequence[int] | None, n_blocks: int, name: str) -> list[int]:
    if block_sizes is None:
        raise ValueError(f"{name} must be provided for logical block-mask attention")
    if isinstance(block_sizes, int):
        return [int(block_sizes)] * int(n_blocks)
    out = [int(v) for v in block_sizes]
    if len(out) != int(n_blocks):
        raise ValueError(f"{name} length {len(out)} does not match block count {n_blocks}")
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} values must be positive, got {out}")
    return out


def block_indices_from_sizes(block_sizes: Sequence[int], *, device) -> torch.Tensor:
    sizes = torch.tensor([int(v) for v in block_sizes], device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(block_sizes), device=device), sizes)


def _resolve_block_indices(logical_block_mask, q_block_sizes, kv_block_sizes, q_block_indices, kv_block_indices, *, device):
    if q_block_indices is None:
        q_sizes = _normalize_block_sizes(q_block_sizes, logical_block_mask.shape[0], "q_block_sizes")
        q_block_indices = block_indices_from_sizes(q_sizes, device=device)
    else:
        q_block_indices = q_block_indices.to(device=device, dtype=torch.long)
    if kv_block_indices is None:
        kv_sizes = _normalize_block_sizes(kv_block_sizes or q_block_sizes, logical_block_mask.shape[1], "kv_block_sizes")
        kv_block_indices = block_indices_from_sizes(kv_sizes, device=device)
    else:
        kv_block_indices = kv_block_indices.to(device=device, dtype=torch.long)
    return q_block_indices, kv_block_indices


def _dense_mask_from_logical(
    logical_block_mask: torch.Tensor,
    q_block_sizes: Sequence[int] | None,
    kv_block_sizes: Sequence[int] | None,
    *,
    q_block_indices: torch.Tensor | None = None,
    kv_block_indices: torch.Tensor | None = None,
    device,
) -> torch.Tensor:
    q_idx, kv_idx = _resolve_block_indices(logical_block_mask, q_block_sizes, kv_block_sizes, q_block_indices, kv_block_indices, device=device)
    return logical_block_mask.to(device=device, dtype=torch.bool)[q_idx[:, None], kv_idx[None, :]]


def flex_block_mask_from_logical(
    logical_block_mask: torch.Tensor,
    q_block_sizes: int | Sequence[int] | None = None,
    kv_block_sizes: int | Sequence[int] | None = None,
    *,
    q_block_indices: torch.Tensor | None = None,
    kv_block_indices: torch.Tensor | None = None,
    device,
):
    """Convert a logical block matrix into a FlexAttention BlockMask."""

    q_idx, kv_idx = _resolve_block_indices(logical_block_mask, q_block_sizes, kv_block_sizes, q_block_indices, kv_block_indices, device=device)
    logical = logical_block_mask.to(device=device, dtype=torch.bool)

    def mask_mod(b, h, q_pos, kv_pos):
        return logical[q_idx[q_pos], kv_idx[kv_pos]]

    return compiled_create_block_mask(
        mask_mod,
        None,
        None,
        Q_LEN=int(q_idx.shape[0]),
        KV_LEN=int(kv_idx.shape[0]),
        device=device,
        _compile=True,
    )


# ---------------------------------------------------------------------------
# Backend interface + registry
# ---------------------------------------------------------------------------


@dataclass
class VarlenAttentionInputs:
    """Inputs handed to an attention backend.

    ``q/k/v`` are BHLD (batch, heads, length, dim). Masks are materialized lazily
    so a backend only pays for the representation it actually consumes:

    * sparse/structured backends call :meth:`flex_mask` or use
      :meth:`block_indices` + ``logical_block_mask`` directly;
    * dense backends call :meth:`dense_mask`.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    logical_block_mask: torch.Tensor | None = None
    q_block_sizes: int | Sequence[int] | None = None
    kv_block_sizes: int | Sequence[int] | None = None
    q_block_indices: torch.Tensor | None = None
    kv_block_indices: torch.Tensor | None = None
    flex_block_mask: object | None = None

    @property
    def device(self):
        return self.q.device

    def block_indices(self):
        if self.logical_block_mask is None:
            raise ValueError("block_indices() requires logical_block_mask")
        return _resolve_block_indices(
            self.logical_block_mask,
            self.q_block_sizes,
            self.kv_block_sizes,
            self.q_block_indices,
            self.kv_block_indices,
            device=self.q.device,
        )

    def dense_mask(self) -> torch.Tensor:
        if self.logical_block_mask is None:
            raise ValueError("dense_mask() requires logical_block_mask and block sizes/indices")
        return _dense_mask_from_logical(
            self.logical_block_mask,
            self.q_block_sizes,
            self.kv_block_sizes,
            q_block_indices=self.q_block_indices,
            kv_block_indices=self.kv_block_indices,
            device=self.q.device,
        )

    def flex_mask(self):
        if self.flex_block_mask is not None:
            return self.flex_block_mask
        if self.logical_block_mask is None:
            raise ValueError("flex_mask() requires flex_block_mask or logical_block_mask")
        return flex_block_mask_from_logical(
            self.logical_block_mask,
            self.q_block_sizes,
            self.kv_block_sizes,
            q_block_indices=self.q_block_indices,
            kv_block_indices=self.kv_block_indices,
            device=self.q.device,
        )


# Backend signature: (VarlenAttentionInputs) -> BHLD output tensor.
AttentionBackend = Callable[[VarlenAttentionInputs], torch.Tensor]


@dataclass(frozen=True)
class _BackendSpec:
    fn: AttentionBackend
    requires_stream_major: bool = False
    description: str = ""


_BACKENDS: dict[str, _BackendSpec] = {}
_BACKEND_ALIASES = {"flex": "flex_attention"}


def register_attention_backend(
    name: str,
    fn: AttentionBackend,
    *,
    requires_stream_major: bool = False,
    description: str = "",
    overwrite: bool = False,
) -> None:
    """Register a varlen attention backend under ``name``.

    A backend that needs cross-sample stream interleave declares
    ``requires_stream_major=True``; the caller is then responsible for wiring
    :mod:`infra.varlen.stream_major` (reorder q/k/v/mask, invert before decode).
    """

    name = str(name)
    if not callable(fn):
        raise TypeError("attention backend must be callable")
    if name in _BACKENDS and not overwrite:
        raise ValueError(f"attention backend '{name}' already registered (pass overwrite=True)")
    _BACKENDS[name] = _BackendSpec(fn=fn, requires_stream_major=bool(requires_stream_major), description=str(description))


def available_attention_backends(*, detail: bool = False):
    """Return registered backend names (or name -> spec dict when ``detail``)."""

    if detail:
        return {
            name: {
                "requires_stream_major": spec.requires_stream_major,
                "description": spec.description,
            }
            for name, spec in sorted(_BACKENDS.items())
        }
    return sorted(_BACKENDS)


def backend_requires_stream_major(name: str) -> bool:
    name = _BACKEND_ALIASES.get(str(name), str(name))
    spec = _BACKENDS.get(name)
    return bool(spec and spec.requires_stream_major)


def _resolve_backend(name: str) -> _BackendSpec:
    name = _BACKEND_ALIASES.get(str(name), str(name))
    spec = _BACKENDS.get(name)
    if spec is not None:
        return spec
    if name == "fa4":
        raise RuntimeError(
            "backend='fa4' requires an externally installed "
            "implementation. Register it with "
            "register_attention_backend('fa4', fn) to enable it."
        )
    raise ValueError(
        f"Unsupported varlen attention backend: {name!r}. "
        f"Available: {available_attention_backends()}. "
        "Add one via register_attention_backend(name, fn)."
    )


def _flex_backend(inp: VarlenAttentionInputs) -> torch.Tensor:
    return BHLD_flex_with_mask(inp.q, inp.k, inp.v, inp.flex_mask())


def _sdpa_backend(inp: VarlenAttentionInputs) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(inp.q, inp.k, inp.v, attn_mask=inp.dense_mask())


register_attention_backend("flex_attention", _flex_backend, description="PyTorch FlexAttention over a compiled BlockMask (default)")
register_attention_backend("sdpa", _sdpa_backend, description="Dense scaled_dot_product_attention reference path")


def varlen_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    flex_block_mask=None,
    logical_block_mask: torch.Tensor | None = None,
    q_block_sizes: int | Sequence[int] | None = None,
    kv_block_sizes: int | Sequence[int] | None = None,
    q_block_indices: torch.Tensor | None = None,
    kv_block_indices: torch.Tensor | None = None,
    backend: str = "flex_attention",
) -> torch.Tensor:
    """Apply packed varlen attention to BLHD tensors and return BLHD tensors."""

    spec = _resolve_backend(backend)

    inp = VarlenAttentionInputs(
        q=q.transpose(1, 2).contiguous(),
        k=k.transpose(1, 2).contiguous(),
        v=v.transpose(1, 2).contiguous(),
        logical_block_mask=logical_block_mask,
        q_block_sizes=q_block_sizes,
        kv_block_sizes=kv_block_sizes,
        q_block_indices=q_block_indices,
        kv_block_indices=kv_block_indices,
        flex_block_mask=flex_block_mask,
    )
    out = spec.fn(inp)
    return out.transpose(1, 2)
