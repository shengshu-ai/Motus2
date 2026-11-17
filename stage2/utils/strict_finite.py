"""Strict per-update finite verification with a handful of reductions."""
import torch


def _all_finite(tensors, device) -> bool:
    """True iff every element of every tensor is finite.

    amax/amin return NaN, +inf or -inf whenever any element is one, so two
    reductions per tensor decide finiteness without a tensor-sized mask.
    """
    extrema = []
    for tensor in tensors:
        if tensor.numel():
            extrema.append(tensor.amax().float().to(device, non_blocking=True))
            extrema.append(tensor.amin().float().to(device, non_blocking=True))
    if not extrema:
        return True
    return bool(torch.isfinite(torch.stack(extrema)).all().item())


def _strict_update_tensors(engine, inner_optimizer):
    """The tensors the strict per-update check must cover, in as few as possible.

    ZeRO-2 keeps each parameter group's 16-bit weights as views into one flat,
    zero-padded buffer, so checking the buffers checks exactly those weights
    with a handful of reductions instead of one per parameter. Parameters not
    owned by a buffer, the fp32 master partitions and all optimizer state are
    checked individually, as before.
    """
    zero = getattr(engine, 'optimizer', None)
    flats = getattr(zero, 'bit16_groups_flat', None) or []
    groups = getattr(zero, 'bit16_groups', None) or []
    tensors, covered = [], set()
    if flats and len(flats) == len(groups) and all(
            flat.numel() >= sum(p.numel() for p in group) for flat, group in zip(flats, groups)):
        tensors.extend(flats)
        covered = {id(p) for group in groups for p in group}
    tensors.extend(p.data for p in engine.parameters() if id(p) not in covered)
    for group in inner_optimizer.param_groups:
        tensors.extend(p.data for p in group['params'])
    for state in inner_optimizer.state.values():
        tensors.extend(v for v in state.values() if torch.is_tensor(v))
    return tensors
