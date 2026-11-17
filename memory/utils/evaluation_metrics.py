"""Effective-sample means for variable-length and distributed evaluation."""
import torch


def append_metrics(storage, prefix, values):
    for key, value in values.items():
        if key.startswith('_'):
            continue
        count = values.get(f'_count/{key}', 0)
        if count > 0:
            storage[f'{prefix}_{key}'].append((float(value), float(count)))


def finalize_metrics(storage, device, distributed=None):
    distributed = torch.distributed if distributed is None else distributed
    keys = sorted(k for k in storage if not k.startswith('_'))
    active = distributed.is_available() and distributed.is_initialized()
    if active:
        gathered = [None] * distributed.get_world_size()
        distributed.all_gather_object(gathered, keys)
        keys = sorted(set().union(*gathered))
    totals = torch.tensor([
        [sum(value * count for value, count in storage.get(key, [])),
         sum(count for _, count in storage.get(key, []))]
        for key in keys
    ], dtype=torch.float64, device=device).reshape(-1, 2)
    if active and keys:
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    final = {}
    for key, (total, count) in zip(keys, totals.tolist()):
        if count > 0:
            final[key] = total / count
            final[f'_count/{key}'] = count
    for key, values in storage.items():
        if key.startswith('_visualization_') and values:
            final[key.replace('_visualization_', 'visualization_', 1)] = values[0]
    return final
