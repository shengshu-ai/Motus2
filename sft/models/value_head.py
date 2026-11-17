# Value prediction components for Motus2 MBRL.
# ValueTokens: learnable query embeddings (K per chunk).
# ValueHead: map hidden states -> 201-bin categorical distribution.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueTokens(nn.Module):
    """Learnable value query embeddings, K tokens per chunk."""

    def __init__(self, dim: int, num_tokens_per_frame: int):
        super().__init__()
        self.dim = dim
        self.num_tokens_per_frame = num_tokens_per_frame
        self.query = nn.Parameter(torch.randn(num_tokens_per_frame, dim) * 0.02)

    def expand(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Expand query embeddings for a batch.

        Returns:
            [B, num_frames * K, dim]
        """
        K = self.num_tokens_per_frame
        # [K, dim] -> [1, 1, K, dim] -> [B, num_frames, K, dim] -> [B, num_frames*K, dim]
        return self.query.unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1).reshape(
            batch_size, num_frames * K, self.dim
        )


class ValueHead(nn.Module):
    """Map value token hidden states -> 201-bin logits."""

    def __init__(self, dim: int, num_bins: int = 201, value_range: tuple = (0.0, 1.0)):
        super().__init__()
        self.num_bins = num_bins
        self.value_range = value_range
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_bins),
        )
        self.register_buffer(
            "bin_centers",
            torch.linspace(value_range[0], value_range[1], num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim] hidden states
        Returns:
            logits: [..., num_bins]
        """
        return self.head(x)


def discretize(values: torch.Tensor, num_bins: int = 201,
               v_min: float = 0.0, v_max: float = 1.0) -> torch.Tensor:
    """Continuous value -> bin index for CE loss target.

    Args:
        values: arbitrary shape, continuous values in [v_min, v_max]
    Returns:
        Long tensor of same shape, bin indices in [0, num_bins-1]
    """
    clamped = values.clamp(v_min, v_max)
    scaled = (clamped - v_min) / (v_max - v_min) * (num_bins - 1)
    return scaled.round().long()


def expected_value(logits: torch.Tensor, num_bins: int = 201,
                   v_min: float = 0.0, v_max: float = 1.0, fp32: bool = False) -> torch.Tensor:
    """Logits -> continuous expected value (softmax weighted sum of bin centers).

    ``fp32`` decodes in float32. In bf16 the centers are off by up to 0.003, a
    third of a bin, which is the size of the score differences MBRL ranks.

    Args:
        logits: [..., num_bins]
    Returns:
        Continuous values [...] in [v_min, v_max]
    """
    if fp32:
        logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    bin_centers = torch.linspace(v_min, v_max, num_bins, device=logits.device, dtype=logits.dtype)
    return (probs * bin_centers).sum(dim=-1)
