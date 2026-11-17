"""Exact action-only supervision masks in original saved-row coordinates."""
from __future__ import annotations

import math

import torch


def validate_metadata(meta):
    if meta.get('schema') != 'robotwin-intervention/1':
        raise ValueError('unsupported intervention metadata schema')
    n = meta.get('num_frames')
    if not isinstance(n, int) or n < 2 or not isinstance(meta.get('success'), bool):
        raise ValueError('intervention needs num_frames and observed success')
    intervals = meta.get('perturbation_intervals')
    if not isinstance(intervals, list) or not intervals:
        raise ValueError('intervention needs exact half-open row intervals')
    end = 0
    for a, b in intervals:
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= n or a < end:
            raise ValueError('invalid or overlapping intervention row interval')
        end = b
    if meta.get('recovery_start_row') != end:
        raise ValueError('recovery boundary disagrees with intervention end')
    return meta


def action_loss_mask(meta, start, count, window, apl, phase, stride, exclude_episode=False):
    validate_metadata(meta)
    result = torch.zeros(window * apl, dtype=torch.bool)
    if exclude_episode:
        return result
    for i in range(count):
        latent = start + i
        first = phase if latent == 0 else phase + 1 + (latent - 1) * 4 * stride
        last = first + (1 if latent == 0 else 4 * stride)
        for j, row in enumerate(range(first, min(last, meta['num_frames']))):
            if j >= apl:
                break
            result[i * apl + j] = not any(a <= row < b for a, b in meta['perturbation_intervals'])
    return result


def recovery_window(meta, phase, stride, total_latents, window, rng):
    """Keep the entire intervention in history of at least one recovery target."""
    validate_metadata(meta)
    first = meta['perturbation_intervals'][0][0]
    recovery = meta['recovery_start_row']
    if phase >= first:
        raise ValueError('latent phase omits the intervention anchor/history')
    apl = 4 * stride
    first_latent = 1 + (first - phase - 1) // apl
    # First latent whose action interval starts at/after recovery.
    recovery_latent = 1 + math.ceil((recovery - phase - 1) / apl)
    if recovery_latent >= total_latents:
        raise ValueError('no recovery latent after the intervention')
    low = max(0, recovery_latent + 1 - window)
    high = min(first_latent, max(0, total_latents - window), recovery_latent - 2)
    if low > high:
        raise ValueError('window cannot retain intervention history and a recovery target')
    return rng.randint(low, high)
