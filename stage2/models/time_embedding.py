"""Time MLP evaluation shared by the video and action training paths."""

from typing import Callable, Optional, Tuple

import torch


def embed_timesteps(
    timesteps: torch.Tensor,
    sinusoidal_embedding: Callable[[torch.Tensor], torch.Tensor],
    embedding: torch.nn.Module,
    projection: torch.nn.Module,
    *,
    compact: bool = False,
    project: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate the pointwise MLP, optionally once per distinct timestep.

    WAN/action time MLPs contain only Linear/SiLU layers: identical timesteps
    have identical outputs. Gather restores every token, so autograd sums all
    of their contributions to the same parameters. No output is detached or
    retained across batches. Compacting can change floating point reduction
    order, as can other GEMM kernel/shape changes.

    Differentiable timestep inputs retain the full timestep path: torch.unique
    does not implement their derivative. Inference callers also use the
    full timestep path unless their caller explicitly opts in.
    """
    if timesteps.ndim != 2:
        raise ValueError("Expected [batch, tokens] timesteps")
    batch, tokens = timesteps.shape
    flat = timesteps.reshape(-1)
    compact = compact and not timesteps.requires_grad and flat.numel() > 0
    if compact:
        values, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        encoded = sinusoidal_embedding(values).float().unsqueeze(0)
    else:
        encoded = sinusoidal_embedding(flat).float().unflatten(0, (batch, tokens))
    embedded = embedding(encoded)
    projected = projection(embedded) if project else None
    if compact:
        embedded = embedded.squeeze(0).index_select(0, inverse).reshape(batch, tokens, -1)
        if projected is not None:
            projected = projected.squeeze(0).index_select(0, inverse).reshape(batch, tokens, -1)
    return embedded, projected
