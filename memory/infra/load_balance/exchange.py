"""Sample packing and P2P exchange for load-balanced batches."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist


class PackedSample:
    """One training sample serialized as flat GPU tensors for P2P exchange."""

    def __init__(
        self,
        sample_id: int,
        data_list: list[torch.Tensor],
        *,
        meta: dict[str, Any] | None = None,
        src_rank: int | None = None,
    ):
        self.sample_id = sample_id
        self.data_list = data_list
        self.shapes = tuple(tuple(int(x) for x in t.shape) for t in data_list)
        self.dtypes = tuple(str(t.dtype).removeprefix("torch.") for t in data_list)
        self._meta = dict(meta or {})
        self.src_rank = src_rank

    @property
    def meta(self) -> dict[str, Any]:
        out = dict(self._meta)
        out["src_rank"] = self.src_rank
        out["sample_id"] = self.sample_id
        out["tensor_dtypes"] = list(self.dtypes)
        return out

    @property
    def buffer_size(self) -> int:
        return sum(int(np.prod(shape)) for shape in self.shapes)

    def pack(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t in self.data_list:
            flat = t.reshape(-1)
            if dtype is not None and flat.is_floating_point():
                flat = flat.to(dtype=dtype)
            parts.append(flat.contiguous())
        return torch.cat(parts, dim=0)

    @staticmethod
    def unpack(
        packed: torch.Tensor,
        shapes: Sequence[Sequence[int]],
        dtypes: Sequence[str] | None = None,
    ) -> list[torch.Tensor]:
        expected = sum(int(np.prod(shape)) for shape in shapes)
        if int(packed.numel()) != expected:
            raise ValueError(f"packed numel {packed.numel()} != expected {expected}")
        out: list[torch.Tensor] = []
        start = 0
        for i, shape in enumerate(shapes):
            count = int(np.prod(shape))
            end = start + count
            tensor = packed[start:end].view(tuple(shape))
            if dtypes is not None and i < len(dtypes):
                target = getattr(torch, dtypes[i], None)
                if target is not None and tensor.dtype != target:
                    tensor = tensor.to(dtype=target)
            out.append(tensor)
            start = end
        return out


def _global_peer(dp_peer: int, rank_to_global: Callable[[int], int] | None) -> int:
    if rank_to_global is not None:
        return int(rank_to_global(dp_peer))
    return int(dp_peer)


def p2p_exchange_samples(
    samples: list[PackedSample],
    recv_plan: dict[int, list[tuple[int, tuple[tuple[int, ...], ...], dict[str, Any]]]],
    send_plan: dict[int, list[int]],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    group: Any = None,
    rank_to_global: Callable[[int], int] | None = None,
    batch_per_peer: bool = True,
) -> list[PackedSample]:
    """Exchange packed sample payloads across DP ranks.

    When ``batch_per_peer`` is true (default), all samples bound for the same
    peer are concatenated into one P2P message per direction to cut latency.
    """
    if not dist.is_initialized():
        return []

    ops: list[dist.P2POp] = []
    recv_storage: list[dict[str, Any]] = []
    send_refs: list[torch.Tensor] = []

    for src_dp_rank in sorted(recv_plan.keys()):
        entries = recv_plan[src_dp_rank]
        src_global = _global_peer(src_dp_rank, rank_to_global)
        if batch_per_peer and len(entries) > 1:
            parts = [sum(int(np.prod(shape)) for shape in shapes) for _, shapes, _ in entries]
            total = sum(parts)
            buffer = torch.zeros(total, dtype=dtype, device=device)
            ops.append(dist.P2POp(dist.irecv, buffer, src_global, group=group))
            recv_storage.append(
                {
                    "batched": True,
                    "src_rank": src_dp_rank,
                    "buffer": buffer,
                    "parts": parts,
                    "entries": entries,
                }
            )
        else:
            for sample_id, shapes, meta in entries:
                buf_size = sum(int(np.prod(shape)) for shape in shapes)
                buffer = torch.zeros(buf_size, dtype=dtype, device=device)
                ops.append(dist.P2POp(dist.irecv, buffer, src_global, group=group))
                recv_storage.append(
                    {
                        "batched": False,
                        "src_rank": src_dp_rank,
                        "sample_id": sample_id,
                        "data": buffer,
                        "shapes": shapes,
                        "meta": meta,
                    }
                )

    for tgt_dp_rank in sorted(send_plan.keys()):
        indices = send_plan[tgt_dp_rank]
        tgt_global = _global_peer(tgt_dp_rank, rank_to_global)
        if batch_per_peer and len(indices) > 1:
            tensor = torch.cat(
                [samples[idx].pack(dtype=dtype) for idx in indices],
                dim=0,
            )
            send_refs.append(tensor)
            ops.append(dist.P2POp(dist.isend, tensor, tgt_global, group=group))
        else:
            for idx in indices:
                tensor = samples[idx].pack(dtype=dtype)
                send_refs.append(tensor)
                ops.append(dist.P2POp(dist.isend, tensor, tgt_global, group=group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    del send_refs

    received: list[PackedSample] = []
    for item in recv_storage:
        if item.get("batched"):
            buffer = item["buffer"]
            offset = 0
            for part, (sample_id, shapes, meta) in zip(item["parts"], item["entries"]):
                chunk = buffer[offset : offset + part]
                dtypes = meta.get("tensor_dtypes")
                data_list = PackedSample.unpack(chunk, shapes, dtypes=dtypes)
                received.append(
                    PackedSample(
                        sample_id=int(sample_id),
                        data_list=data_list,
                        meta={k: v for k, v in meta.items() if k != "tensor_dtypes"},
                        src_rank=int(item["src_rank"]),
                    )
                )
                offset += part
        else:
            dtypes = item["meta"].get("tensor_dtypes")
            data_list = PackedSample.unpack(item["data"], item["shapes"], dtypes=dtypes)
            received.append(
                PackedSample(
                    sample_id=int(item["sample_id"]),
                    data_list=data_list,
                    meta={k: v for k, v in item["meta"].items() if k != "tensor_dtypes"},
                    src_rank=int(item["src_rank"]),
                )
            )
    return received
