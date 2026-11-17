# Learnable per-frame gist (memory) query tokens for the hybrid-memory model.
# Mirrors value_head.ValueTokens. M gist tokens are appended per history frame;
# they attend to that frame's clean latent (so they have interacted with it) and
# survive in the KV cache after the frame's full tokens are evicted.

import torch
import torch.nn as nn


class GistTokens(nn.Module):
    """Learnable gist query embeddings, M tokens per (history) frame."""

    def __init__(self, dim: int, num_tokens_per_frame: int):
        super().__init__()
        self.dim = dim
        self.num_tokens_per_frame = int(num_tokens_per_frame)
        q = torch.randn(self.num_tokens_per_frame, dim) * 0.02
        if self.num_tokens_per_frame > 0:
            self.query = nn.Parameter(q)
        else:
            # Gist disabled (M=0). A zero-numel nn.Parameter is NOT a harmless no-op:
            # DeepSpeed ZeRO-1/2 init partitions/flattens trainable params and an empty
            # param desyncs the per-rank flat-buffer collectives, hanging one rank inside
            # the optimizer __init__ while the others wait at the next collective. Register
            # it as a non-trainable buffer so it stays out of the optimizer entirely;
            # expand() still returns a [B, 0, dim] tensor with the right device/dtype.
            self.register_buffer("query", q, persistent=False)

    def expand(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Expand gist queries for a batch.

        Returns:
            [B, num_frames * M, dim]
        """
        m = self.num_tokens_per_frame
        return self.query.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_frames, -1, -1
        ).reshape(batch_size, num_frames * m, self.dim)
