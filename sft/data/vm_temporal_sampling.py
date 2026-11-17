"""Observation sampling for VM only; source chronology remains the label clock."""
from __future__ import annotations

import random

import torch


def sample_chunk_indices(total: int, window: int, chunk: int, *, rng: random.Random,
                         shuffle_probability: float = 0.0,
                         rewind_probability: float = 0.0) -> torch.Tensor:
    """Sample intact chunks, then optionally change their presentation order.

    This is a chunk-based adaptation of RynnValue, not its unpublished sampler.
    A forward-biased walk visits each selected chunk once; when the preferred
    direction has no remaining observations it takes the available direction.
    No global RNG state is read or changed here.
    """
    if chunk < 1 or window < chunk or window % chunk or total < window:
        raise ValueError("VM sampling needs complete chunks and total >= window")
    if not 0 <= shuffle_probability <= 1 or not 0 <= rewind_probability <= 1:
        raise ValueError("VM temporal probabilities must lie in [0, 1]")
    starts = list(range(0, total - chunk + 1, chunk))
    if starts[-1] != total - chunk:
        starts.append(total - chunk)  # Include the terminal observation of odd-length episodes.
    selected = sorted(rng.sample(starts, window // chunk))
    if shuffle_probability > 0 and rng.random() < shuffle_probability:
        rng.shuffle(selected)
    elif rewind_probability > 0:
        remaining = selected.copy()
        current = remaining.pop(rng.randrange(len(remaining)))
        ordered = [current]
        while remaining:
            backward = rng.random() < rewind_probability
            preferred = [i for i in remaining if (i < current) == backward]
            current = rng.choice(preferred or remaining)
            ordered.append(current)
            remaining.remove(current)
        selected = ordered
    return torch.tensor([i + j for i in selected for j in range(chunk)], dtype=torch.long)


def replace_vm_inputs(latents, actions, states, values, valid_len, latent_start,
                      vm_indices, observations):
    """Replace only selected VM rows, without modifying the collated batch."""
    originals = dict(latents=latents, actions=actions, states=states,
                     values=values, valid_len=valid_len, latent_start=latent_start)
    if vm_indices.numel() == 0:
        return tuple(originals.values())
    if observations is None:
        raise ValueError("VM temporal sampling enabled but observation payload is missing")
    outputs = []
    for key, original in originals.items():
        if original is None:
            raise ValueError(f"VM temporal sampling requires {key}")
        replacement = observations[key]
        if replacement.shape != original.shape:
            raise ValueError(f"VM {key} shape mismatch: {replacement.shape} != {original.shape}")
        replacement = replacement.to(device=original.device, dtype=original.dtype)
        outputs.append(original.index_copy(0, vm_indices, replacement.index_select(0, vm_indices)))
    return tuple(outputs)
