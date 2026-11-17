"""Factored memory attention: planned-KV gather → SDPA / Flash-Attn → scatter.

Backends
--------
- ``sdpa_planned``: mask-free SDPA per Q group (oracle / CPU).
- ``flash_attn_varlen``: Option B — pack all Q groups for one sample into a
  single ``flash_attn_varlen_func`` call via ``cu_seqlens`` (preferred when FA
  is available). Falls back is not automatic; use ``backend="auto"``.
- ``auto``: ``flash_attn_varlen`` if ``HAS_FLASH_ATTN`` else ``sdpa_planned``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .factored_plans import SegmentFactoredPlans

try:
    from flash_attn import flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except Exception:  # ImportError, CUDA ABI mismatch, missing .so, etc.
    flash_attn_varlen_func = None  # type: ignore[assignment]
    HAS_FLASH_ATTN = False


@dataclass
class SampleMajorLayoutMeta:
    """Token index ranges for one sample-major memory window.

    Layout matches ``build_memory_visibility`` emit order when
    ``separate_anchor=False`` and ``value_tokens_per_frame=0``::

        [ clean_0..N-1 | gist_0..N-1 | per-chunk(state?, action, noisy) ]
    """

    clean_ranges: list[tuple[int, int]]
    gist_ranges: list[tuple[int, int]]
    action_ranges: list[tuple[int, int]]
    noisy_ranges: list[tuple[int, int]]
    total_len: int


def build_sample_major_layout_meta(
    *,
    num_clean_frames: int,
    obs_latents: int,
    chunk_size: int,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    action_tokens_per_frame: int,
    state_tokens_per_frame: int = 0,
    value_tokens_per_frame: int = 0,
) -> SampleMajorLayoutMeta:
    """Build per-frame / per-chunk token ranges matching visibility layout."""
    N = int(num_clean_frames)
    obs, cs = int(obs_latents), max(1, int(chunk_size))
    vbs = int(video_tokens_per_frame)
    gbs = int(gist_tokens_per_frame)
    abs_ = int(action_tokens_per_frame)
    sbs = int(state_tokens_per_frame)
    nvpf = int(value_tokens_per_frame)
    if nvpf != 0:
        raise NotImplementedError(
            "value_tokens_per_frame != 0 is not supported in factored layout meta yet"
        )

    clean_ranges: list[tuple[int, int]] = []
    pos = 0
    for _ in range(N):
        clean_ranges.append((pos, pos + vbs))
        pos += vbs

    gist_ranges: list[tuple[int, int]] = []
    for _ in range(N):
        gist_ranges.append((pos, pos + gbs))
        pos += gbs

    action_ranges: list[tuple[int, int]] = []
    noisy_ranges: list[tuple[int, int]] = []
    offset = 0
    while offset < N:
        count = min(cs, N - offset)
        alen = sbs + count * abs_
        nlen = count * vbs
        action_ranges.append((pos, pos + alen))
        pos += alen
        noisy_ranges.append((pos, pos + nlen))
        pos += nlen
        offset += count

    return SampleMajorLayoutMeta(
        clean_ranges=clean_ranges,
        gist_ranges=gist_ranges,
        action_ranges=action_ranges,
        noisy_ranges=noisy_ranges,
        total_len=pos,
    )


def _range_indices(ranges: list[tuple[int, int]], phys_list: list[int]) -> list[int]:
    idx: list[int] = []
    for p in phys_list:
        s, e = ranges[p]
        idx.extend(range(s, e))
    return idx


def _iter_planned_groups(
    layout_meta: SampleMajorLayoutMeta,
    plans: SegmentFactoredPlans,
) -> list[tuple[list[int], list[int]]]:
    """Build (q_idx, kv_idx) groups for A / B / C1 / C2 in layout order."""
    clean_r = layout_meta.clean_ranges
    gist_r = layout_meta.gist_ranges
    action_r = layout_meta.action_ranges
    noisy_r = layout_meta.noisy_ranges
    groups: list[tuple[list[int], list[int]]] = []

    for i, (qs, qe) in enumerate(clean_r):
        q_idx = list(range(qs, qe))
        kv_idx = _range_indices(clean_r, plans.clean_kv_clean_phys[i])
        kv_idx += _range_indices(gist_r, plans.clean_kv_gist_phys[i])
        groups.append((q_idx, kv_idx))

    for i, (qs, qe) in enumerate(gist_r):
        q_idx = list(range(qs, qe))
        kv_idx = _range_indices(clean_r, plans.gist_kv_clean_phys[i])
        kv_idx += _range_indices(gist_r, plans.gist_kv_gist_phys[i])
        groups.append((q_idx, kv_idx))

    n_chunks = len(action_r)
    if n_chunks != len(noisy_r) or n_chunks != len(plans.action_kv_clean_phys):
        raise ValueError("chunk plan / layout range count mismatch")
    for c in range(n_chunks):
        as_, ae = action_r[c]
        ns, ne = noisy_r[c]
        hist = _range_indices(clean_r, plans.action_kv_clean_phys[c])
        hist += _range_indices(gist_r, plans.action_kv_gist_phys[c])
        action_self = list(range(as_, ae))
        noisy_self = list(range(ns, ne))

        # C1 action: hist + action self (no same-chunk noisy)
        if as_ < ae:
            groups.append((action_self, hist + action_self))
        # C2 noisy: hist + action + noisy self
        if ns < ne:
            groups.append((noisy_self, hist + action_self + noisy_self))

    return groups


@dataclass
class _MaterializedGroups:
    """Device-side gather/scatter indices for one segment's planned groups.

    ``_iter_planned_groups`` costs ``O(planned_kv)`` Python list operations, and
    converting each group to a tensor costs another pass. Both depend only on the
    layout and the plans, which are fixed for a batch, while the attention itself
    runs once per transformer layer. Materializing here lets layer 0 pay and every
    later layer reuse.

    Per-group entries are views into the flat gather tensors, so the whole struct
    is two index tensors plus the ``cu_seqlens`` pair.
    """

    layout: SampleMajorLayoutMeta
    device: torch.device
    q_groups: list[torch.Tensor]
    kv_groups: list[torch.Tensor]
    q_gather: torch.Tensor
    kv_gather: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_q: int
    max_k: int


def _materialize_groups(
    layout_meta: SampleMajorLayoutMeta,
    plans: SegmentFactoredPlans,
    device: torch.device,
) -> _MaterializedGroups:
    """Return cached gather indices for ``(layout_meta, plans)`` on ``device``."""
    cached = plans.materialized_groups
    if (
        isinstance(cached, _MaterializedGroups)
        and cached.layout is layout_meta
        and cached.device == device
    ):
        return cached

    # Empty Q groups contribute no output and would give Flash-Attn a zero-length
    # sequence, so they are dropped once here instead of per layer.
    groups = [g for g in _iter_planned_groups(layout_meta, plans) if g[0]]

    q_flat: list[int] = []
    kv_flat: list[int] = []
    cu_q = [0]
    cu_k = [0]
    max_q = 0
    max_k = 0
    for q_idx, kv_idx in groups:
        q_flat.extend(q_idx)
        kv_flat.extend(kv_idx)
        cu_q.append(len(q_flat))
        cu_k.append(len(kv_flat))
        max_q = max(max_q, len(q_idx))
        max_k = max(max_k, len(kv_idx))

    q_gather = torch.tensor(q_flat, dtype=torch.long, device="cpu").to(device)
    kv_gather = torch.tensor(kv_flat, dtype=torch.long, device="cpu").to(device)
    materialized = _MaterializedGroups(
        layout=layout_meta,
        device=device,
        q_groups=[q_gather[cu_q[i] : cu_q[i + 1]] for i in range(len(groups))],
        kv_groups=[kv_gather[cu_k[i] : cu_k[i + 1]] for i in range(len(groups))],
        q_gather=q_gather,
        kv_gather=kv_gather,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device="cpu").to(device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device="cpu").to(device),
        max_q=max_q,
        max_k=max_k,
    )
    plans.materialized_groups = materialized
    return materialized


def _run_sdpa_planned(
    qb: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    mat: _MaterializedGroups,
    out_b: torch.Tensor,
) -> None:
    """Scatter SDPA outputs for one batch row into ``out_b`` [1, L, H, D]."""
    for q_t, kv_t in zip(mat.q_groups, mat.kv_groups):
        qq = qb.index_select(1, q_t)
        kk = kb.index_select(1, kv_t)
        vv = vb.index_select(1, kv_t)
        yy = F.scaled_dot_product_attention(
            qq.transpose(1, 2),
            kk.transpose(1, 2),
            vv.transpose(1, 2),
        ).transpose(1, 2)
        out_b.index_copy_(1, q_t, yy)


def _run_flash_attn_varlen(
    qb: torch.Tensor,
    kb: torch.Tensor,
    vb: torch.Tensor,
    mat: _MaterializedGroups,
    out_b: torch.Tensor,
) -> None:
    """Option B: pack all groups into one ``flash_attn_varlen_func`` call.

    ``qb/kb/vb`` are ``[1, L, H, D]``. Outputs are scattered back by Q index.
    """
    if not HAS_FLASH_ATTN or flash_attn_varlen_func is None:
        raise RuntimeError(
            "backend='flash_attn_varlen' requires a working flash_attn install"
        )
    if mat.q_gather.numel() == 0:
        return

    # Squeeze batch: work in [L, H, D]. Gathering with the concatenated index is
    # equivalent to gathering per group and concatenating.
    q_pack = qb[0].index_select(0, mat.q_gather)  # [total_q, H, D]
    k_pack = kb[0].index_select(0, mat.kv_gather)
    v_pack = vb[0].index_select(0, mat.kv_gather)

    out_pack = flash_attn_varlen_func(
        q_pack,
        k_pack,
        v_pack,
        mat.cu_seqlens_q,
        mat.cu_seqlens_k,
        mat.max_q,
        mat.max_k,
        causal=False,
    )  # [total_q, H, D]

    # Q ranges are disjoint across groups (clean / gist / action / noisy occupy
    # disjoint token spans), so one scatter replaces the per-group loop.
    out_b[0].index_copy_(0, mat.q_gather, out_pack)


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "flash_attn_varlen" if HAS_FLASH_ATTN else "sdpa_planned"
    if backend in ("flash_attn_varlen", "fa"):
        return "flash_attn_varlen"
    if backend == "sdpa_planned":
        return "sdpa_planned"
    raise ValueError(
        f"unknown backend={backend!r}; use 'sdpa_planned', "
        f"'flash_attn_varlen'/'fa', or 'auto'"
    )


def factored_memory_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout_meta: SampleMajorLayoutMeta,
    plans: SegmentFactoredPlans,
    backend: str = "sdpa_planned",
) -> torch.Tensor:
    """Run A/B/C1/C2 planned-KV attention and scatter into ``[B, L, H, D]``.

    ``q/k/v`` are sample-major packed rows. v1 supports ``B>=1`` (loops batch).

    Backends: ``sdpa_planned``, ``flash_attn_varlen`` (alias ``fa``), ``auto``.
    Flash path uses Option B (one varlen launch packing all Q groups per sample).
    """
    resolved = _resolve_backend(backend)
    if backend == "auto" and not q.is_cuda:
        resolved = "sdpa_planned"
    if resolved == "flash_attn_varlen" and not HAS_FLASH_ATTN:
        raise RuntimeError(
            "backend='flash_attn_varlen' requested but flash_attn is unavailable; "
            "use backend='sdpa_planned' or 'auto'"
        )
    if resolved == "flash_attn_varlen" and not q.is_cuda:
        raise RuntimeError("backend='flash_attn_varlen' requires CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [B, L, H, D]")
    B, L, H, D = q.shape
    if L != layout_meta.total_len:
        raise ValueError(
            f"sequence length {L} != layout_meta.total_len {layout_meta.total_len}"
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v shapes must match")

    out = torch.zeros_like(q)
    mat = _materialize_groups(layout_meta, plans, q.device)

    for b in range(B):
        qb, kb, vb = q[b : b + 1], k[b : b + 1], v[b : b + 1]
        out_b = out[b : b + 1]
        if resolved == "sdpa_planned":
            _run_sdpa_planned(qb, kb, vb, mat, out_b)
        else:
            _run_flash_attn_varlen(qb, kb, vb, mat, out_b)

    return out
