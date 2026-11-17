"""Training-loop batch load balancer (the closed stack LoadBalancePreprocessor port)."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch
import torch.distributed as dist

from .exchange import PackedSample, p2p_exchange_samples
from .groups import DataParallelContext, get_data_parallel_context, init_data_parallel_group
from .lpt import build_lpt_plan
from .types import SampleCost

logger = logging.getLogger(__name__)

# Per-sample tensor fields exchanged via P2P (order matters for pack/unpack).
_TENSOR_KEYS = (
    "latents",
    "actions",
    "states",
    "anchor_latents",
    "values",
    "language_embedding",
)

# Scalar / small metadata carried in SampleCost.meta (not flattened into P2P payload).
_SCALAR_META_KEYS = (
    "valid_len",
    "latent_start",
    "is_task_irrelevant",
)


def compute_sample_cost(
    *,
    valid_len: int | None,
    latents: torch.Tensor,
    actions_per_latent: int = 6,
) -> int:
    """Estimate per-sample compute cost for LPT assignment."""
    if valid_len is not None and int(valid_len) > 0:
        t_lat = int(valid_len)
    elif latents.ndim >= 3:
        t_lat = int(latents.shape[-3] if latents.ndim == 5 else latents.shape[1])
    else:
        t_lat = 1
    # Attention cost scales roughly with trajectory length; include action tokens.
    return max(1, t_lat * max(1, int(actions_per_latent)))


def _default_exchange_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if use_bf16 else torch.float32


class MotusBatchBalancePreprocessor:
    """Cross-rank LPT rebalancer for Motus training batches.

    Complements ``TokenBudgetBatchSampler`` (local OOM guard) by redistributing
    samples across DP ranks each step to reduce stragglers when trajectory
    lengths differ. No-op when disabled, ``dp_size == 1``, or fewer than two
    global samples exist.

    Uses the **data-parallel** process group only (see ``groups.py``), so CP/TP
    ranks that share a model shard do not exchange samples.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        dp_rank: int | None = None,
        dp_size: int | None = None,
        dp_context: DataParallelContext | None = None,
        device: torch.device | str = "cuda",
        actions_per_latent: int = 6,
        exchange_bf16: bool = True,
        batch_per_peer: bool = True,
    ):
        self.enabled = bool(enabled)
        self._dp_context = dp_context
        if dp_context is not None:
            self.dp_rank = int(dp_context.rank)
            self.dp_size = int(dp_context.size)
        else:
            self.dp_rank = int(dp_rank if dp_rank is not None else 0)
            self.dp_size = int(dp_size if dp_size is not None else 1)
        self.device = torch.device(device)
        self.actions_per_latent = int(actions_per_latent)
        self.exchange_dtype = _default_exchange_dtype(exchange_bf16)
        self.batch_per_peer = bool(batch_per_peer)

    @property
    def dp_context(self) -> DataParallelContext:
        if self._dp_context is not None:
            return self._dp_context
        return get_data_parallel_context()

    def _rank_to_global(self) -> Callable[[int], int]:
        ctx = self.dp_context
        if ctx.group is None:
            return lambda r: r
        return ctx.global_rank

    def __call__(self, batch: dict[str, Any] | None) -> dict[str, Any] | None:
        if batch is None or not self.enabled or self.dp_size <= 1:
            return batch
        if not dist.is_initialized():
            return batch
        return self.rebalance(batch)

    def rebalance(self, batch: dict[str, Any]) -> dict[str, Any]:
        samples = self._batch_to_samples(batch)
        if not samples:
            return batch

        local_costs = [
            SampleCost(
                src_rank=self.dp_rank,
                sample_id=s.sample_id,
                cost=compute_sample_cost(
                    valid_len=s.meta.get("valid_len"),
                    latents=s.data_list[0],
                    actions_per_latent=self.actions_per_latent,
                ),
                shapes=s.shapes,
                meta={k: v for k, v in s.meta.items() if k not in ("src_rank", "sample_id", "tensor_dtypes")},
            )
            for s in samples
        ]

        dp_group = self.dp_context.group
        gathered: list[list[SampleCost] | None] = [None for _ in range(self.dp_size)]
        dist.all_gather_object(gathered, local_costs, group=dp_group)
        gathered_costs = [list(x) for x in gathered if x is not None]

        total_samples = sum(len(x) for x in gathered_costs)
        if total_samples <= 1:
            return batch

        plan = build_lpt_plan(
            local_costs,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            gathered_costs=gathered_costs,
        )

        if not plan.send_plan and not plan.recv_plan:
            logger.debug(
                "Load balance: rank=%d LPT no-op (kept=%d global_samples=%d)",
                self.dp_rank,
                len(plan.kept_indices),
                total_samples,
            )
            return batch

        logger.info(
            "Load balance P2P: rank=%d send=%s recv=%s kept=%d global_samples=%d",
            self.dp_rank,
            {k: len(v) for k, v in plan.send_plan.items()},
            {k: len(v) for k, v in plan.recv_plan.items()},
            len(plan.kept_indices),
            total_samples,
        )
        dist.barrier(group=dp_group)
        received = p2p_exchange_samples(
            samples,
            plan.recv_plan,
            plan.send_plan,
            device=self.device,
            dtype=self.exchange_dtype,
            group=dp_group,
            rank_to_global=self._rank_to_global(),
            batch_per_peer=self.batch_per_peer,
        )
        kept = [samples[i] for i in plan.kept_indices]
        balanced = kept + received
        return self._samples_to_batch(balanced, template=batch)

    def _batch_to_samples(self, batch: dict[str, Any]) -> list[PackedSample]:
        latents = batch.get("latents")
        if latents is None:
            return []

        if isinstance(latents, (list, tuple)):
            return self._list_batch_to_samples(batch)

        return self._stacked_batch_to_samples(batch)

    def _list_batch_to_samples(self, batch: dict[str, Any]) -> list[PackedSample]:
        latents = batch["latents"]
        n = len(latents)
        samples: list[PackedSample] = []
        for i in range(n):
            data_list: list[torch.Tensor] = []
            for key in _TENSOR_KEYS:
                value = batch.get(key)
                if value is None:
                    continue
                if key == "language_embedding":
                    data_list.append(value[i].to(self.device))
                else:
                    data_list.append(value[i].to(self.device))

            meta = self._scalar_meta_for_index(batch, i)
            samples.append(PackedSample(i, data_list, meta=meta, src_rank=self.dp_rank))
        return samples

    def _stacked_batch_to_samples(self, batch: dict[str, Any]) -> list[PackedSample]:
        latents = batch["latents"]
        n = int(latents.shape[0])
        samples: list[PackedSample] = []
        for i in range(n):
            data_list = []
            for key in _TENSOR_KEYS:
                value = batch.get(key)
                if value is None:
                    continue
                if key == "language_embedding":
                    data_list.append(value[i].to(self.device))
                else:
                    data_list.append(value[i].to(self.device))
            meta = self._scalar_meta_for_index(batch, i)
            samples.append(PackedSample(i, data_list, meta=meta, src_rank=self.dp_rank))
        return samples

    @staticmethod
    def _scalar_meta_for_index(batch: dict[str, Any], index: int) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for key in _SCALAR_META_KEYS:
            value = batch.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                meta[key] = int(value[index].item())
            elif isinstance(value, (list, tuple)):
                meta[key] = value[index]
            else:
                meta[key] = value
        return meta

    def _samples_to_batch(self, samples: list[PackedSample], template: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy({k: v for k, v in template.items() if k not in _TENSOR_KEYS and k not in _SCALAR_META_KEYS})

        list_mode = isinstance(template.get("latents"), (list, tuple))
        per_key_tensors: dict[str, list[torch.Tensor]] = {k: [] for k in _TENSOR_KEYS if template.get(k) is not None}
        scalars: dict[str, list[Any]] = {k: [] for k in _SCALAR_META_KEYS if template.get(k) is not None}

        key_order = [k for k in _TENSOR_KEYS if template.get(k) is not None]
        for sample in samples:
            if len(sample.data_list) != len(key_order):
                raise ValueError(
                    f"sample {sample.sample_id} has {len(sample.data_list)} tensors, expected {len(key_order)}"
                )
            for key, tensor in zip(key_order, sample.data_list):
                per_key_tensors[key].append(tensor)

            for key in scalars:
                if key in sample.meta:
                    scalars[key].append(sample.meta[key])

        if list_mode:
            for key, tensors in per_key_tensors.items():
                if key == "language_embedding":
                    out[key] = torch.stack(tensors, dim=0)
                else:
                    out[key] = tensors
        else:
            for key, tensors in per_key_tensors.items():
                out[key] = torch.stack(tensors, dim=0)

        for key, values in scalars.items():
            if not values:
                continue
            if torch.is_tensor(template.get(key)):
                out[key] = torch.tensor(values, device=self.device, dtype=template[key].dtype)
            else:
                out[key] = values

        return out


def maybe_build_batch_balance_preprocessor(
    config: Any,
    *,
    accelerator: Any | None = None,
    dp_rank: int | None = None,
    dp_size: int | None = None,
    device: torch.device | str,
) -> MotusBatchBalancePreprocessor:
    """Factory from OmegaConf / dict training.load_balance section."""
    lb_cfg = getattr(getattr(config, "training", None), "load_balance", None)
    enabled = False
    actions_per_latent = 6
    exchange_bf16 = True
    batch_per_peer = True
    cp_size = None
    tp_size = None
    if lb_cfg is not None:
        enabled = bool(getattr(lb_cfg, "enabled", False))
        actions_per_latent = int(getattr(lb_cfg, "actions_per_latent", actions_per_latent))
        exchange_bf16 = bool(getattr(lb_cfg, "exchange_bf16", exchange_bf16))
        batch_per_peer = bool(getattr(lb_cfg, "batch_per_peer", batch_per_peer))
        if getattr(lb_cfg, "cp_size", None) is not None:
            cp_size = int(lb_cfg.cp_size)
        if getattr(lb_cfg, "tp_size", None) is not None:
            tp_size = int(lb_cfg.tp_size)

    dp_context = init_data_parallel_group(
        accelerator=accelerator,
        cp_size=cp_size,
        tp_size=tp_size,
    )

    return MotusBatchBalancePreprocessor(
        enabled=enabled,
        dp_context=dp_context,
        dp_rank=dp_rank,
        dp_size=dp_size,
        device=device,
        actions_per_latent=actions_per_latent,
        exchange_bf16=exchange_bf16,
        batch_per_peer=batch_per_peer,
    )
