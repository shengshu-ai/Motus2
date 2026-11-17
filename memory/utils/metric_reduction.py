"""Materialize training scalars once, with one packed distributed reduction."""

from collections import defaultdict
from typing import Any, Dict, Mapping

import torch
import torch.distributed as dist


# These are counts in Trainer's existing logging contract. A num_* name alone
# does not make an arbitrary metric a count.
SUM_METRICS = frozenset({
    "num_vm_samples", "num_wm_samples", "num_pol_samples",
})


def reduce_training_metrics(
    metrics: Mapping[str, Any], device: torch.device, dist_module=dist,
) -> Dict[str, Any]:
    """Preserve per-rank scalar averages and sample-count sums.

    All ranks must supply the same numeric keys, as with the former per-key
    collectives. Sorting makes the packed wire order independent of dict order.
    Tensors are detached here; no metric participates in the training graph.
    """
    result = dict(metrics)
    tensor_keys = []
    number_keys = []
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"metric {key!r} must be scalar, got {tuple(value.shape)}")
            tensor_keys.append(key)
        elif isinstance(value, (int, float)):
            number_keys.append(key)

    distributed = dist_module.is_available() and dist_module.is_initialized()
    if not distributed:
        # Preserve local scalar types, including integer counts and float64 metrics.
        groups = defaultdict(list)
        for key in tensor_keys:
            value = metrics[key]
            groups[(value.device, value.dtype)].append(key)
        for keys in groups.values():
            values = torch.stack([metrics[key].detach().reshape(()) for key in keys])
            result.update(zip(keys, values.cpu().tolist()))
        return result

    numeric_keys = set(tensor_keys) | set(number_keys)
    average_keys = sorted(numeric_keys - SUM_METRICS)
    keys = average_keys + sorted(numeric_keys & SUM_METRICS)
    if not keys:
        return result

    # Reduce distributed metrics in float32 and batch host-to-device transfers.
    scalars = {
        key: metrics[key].detach().reshape(()).to(device=device, dtype=torch.float32)
        for key in tensor_keys
    }
    if number_keys:
        host_values = torch.tensor(
            [metrics[key] for key in number_keys], device=device, dtype=torch.float32,
        )
        scalars.update(zip(number_keys, host_values.unbind()))
    packed = torch.stack([scalars[key] for key in keys])
    world_size = dist_module.get_world_size()
    if world_size > 1:
        dist_module.all_reduce(packed, op=dist_module.ReduceOp.SUM)
        if average_keys:
            packed[:len(average_keys)].div_(world_size)
    result.update(zip(keys, packed.cpu().tolist()))
    return result
