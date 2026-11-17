"""Self-forcing conditioning-noise schedule for the hybrid-memory model.

Kept free of torch and of the model package so the schedule can be tested without the
CUDA-bound Wan import chain.
"""

from typing import List


def conditioning_noise_sigma(
    num_latents: int, sigma_max: float, clean_prefix: int
) -> List[float]:
    """Per-latent upper bound of the self-forcing mix ratio.

    MemoryWAM §4.1 draws one ratio per clean conditioning latent from ``U(0, sigma_max)``.
    A latent's full tokens leave the KV cache once it falls out of the recent window, so for
    the opening of an episode the gist computed from it is the only permanent record of the
    initial scene. ``clean_prefix`` exempts that many leading latents so the record is written
    from unmodified observations; ``0`` reproduces MemoryWAM's uniform schedule.

    Indices are window-local, matching the clean-video physical positions ``0..N-1`` used by
    ``block_attention.build_memory_visibility``.
    """
    n = max(0, int(num_latents))
    prefix = min(max(0, int(clean_prefix)), n)
    return [0.0] * prefix + [float(sigma_max)] * (n - prefix)
