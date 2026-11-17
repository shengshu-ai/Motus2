# Action Expert for AR training.
# Supports use_mot (MoT blocks vs simple projection), action_mode, use_state.

import re
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """Fixed sinusoidal positional embedding."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(length, dtype=np.float64)
    out = np.einsum("m,d->md", pos, omega)
    return torch.from_numpy(np.concatenate([np.sin(out), np.cos(out)], axis=1)).float()


def build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
    """Build MLP projector."""
    if projector_type == "linear":
        return nn.Linear(in_features, out_features)
    match = re.match(r"^mlp(\d+)x_silu$", projector_type)
    if match:
        depth = int(match.group(1))
        layers = [nn.Linear(in_features, out_features)]
        for _ in range(1, depth):
            layers += [nn.SiLU(), nn.Linear(out_features, out_features)]
        return nn.Sequential(*layers)
    raise ValueError(f"Unknown projector type: {projector_type}")


# ---------------------------------------------------------------------------
# Lightweight norm (re-use WAN definitions to keep checkpoint compat)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x.float().mul(torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, elementwise_affine=False, eps=eps)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ActionExpertConfig:
    dim: int = 1024                   # Hidden dimension
    ffn_dim: int = 4096               # FFN dimension
    num_layers: int = 30              # Number of layers (matches WAN)
    state_dim: int = 14               # Robot state dim
    action_dim: int = 14              # Robot action dim
    chunk_size: int = 16              # Total action tokens per forward
    video_feature_dim: int = 3072     # WAN hidden dim
    eps: float = 1e-6
    # AR-specific
    use_mot: bool = False             # True: expert blocks + MoT, False: project into WAN
    use_state: bool = True            # True: StateActionEncoder, False: ActionEncoder
    action_mode: str = "joint"        # "joint" (14D), "abs_eef" (20D abs EEF), or "rel_eef" (20D)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class StateActionEncoder(nn.Module):
    """Encode state (1 token) + action (N tokens).

    No sinusoidal pos_embed — chunk-local position is already provided by
    RoPE applied to Q/K in the DiT attention layers.
    """

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.state_encoder = build_mlp("mlp3x_silu", config.state_dim, config.dim)
        self.action_encoder = build_mlp("mlp3x_silu", config.action_dim, config.dim)

    @staticmethod
    def _apply_state_mask(state_tokens, state_mask):
        if state_mask is None:
            return state_tokens
        return state_tokens * state_mask.to(dtype=state_tokens.dtype).unsqueeze(-1)

    def forward(self, state, actions, state_mask=None):
        """state: [B,1,D_s], actions: [B,T,D_a] -> [B,1+T,dim]"""
        state_enc = self._apply_state_mask(self.state_encoder(state), state_mask)
        action_enc = self.action_encoder(actions)
        return torch.cat([state_enc, action_enc], dim=1)

    def forward_batch_chunks(self, states, actions, state_mask=None):
        """Batch encode M chunks. states: [B,M,state_dim], actions: [B,M,T,action_dim] -> [B,M*(1+T),dim]"""
        B, M, T, _ = actions.shape
        states_enc = self.state_encoder(states)  # [B, M, dim]
        if state_mask is not None:
            states_enc = states_enc * state_mask.to(dtype=states_enc.dtype).unsqueeze(-1)
        actions_enc = self.action_encoder(actions.reshape(B * M, T, -1))  # [B*M, T, dim]
        actions_enc = actions_enc.view(B, M, T, -1)
        chunks = torch.cat([states_enc.unsqueeze(2), actions_enc], dim=2)  # [B, M, 1+T, dim]
        enc = chunks.reshape(B, M * (1 + T), -1)
        return enc

    def forward_chunk_list(self, states, action_chunks, state_mask=None):
        """Encode variable-length chunks.

        Args:
            states: [B, M, state_dim]
            action_chunks: list of M tensors, each [B, T_i, action_dim]

        Returns:
            [B, sum_i (1 + T_i), dim]
        """
        parts = []
        for i, actions in enumerate(action_chunks):
            state_i = states[:, i:i + 1]
            state_enc = self.state_encoder(state_i)
            if state_mask is not None:
                state_enc = state_enc * state_mask[:, i:i + 1].to(dtype=state_enc.dtype).unsqueeze(-1)
            action_enc = self.action_encoder(actions)
            parts.append(torch.cat([state_enc, action_enc], dim=1))
        return torch.cat(parts, dim=1)


class ActionEncoder(nn.Module):
    """Encode action-only (no state token)."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.action_encoder = build_mlp("mlp3x_silu", config.action_dim, config.dim)

    def forward(self, state, actions):
        """state: ignored, pass None when not using state. actions: [B,T,D_a] -> [B,T,dim]"""
        return self.action_encoder(actions)

    def forward_chunk_list(self, action_chunks):
        """Encode variable-length chunks and concatenate them."""
        parts = []
        for actions in action_chunks:
            parts.append(self.action_encoder(actions))
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ActionDecoder(nn.Module):
    """Decode action tokens -> action predictions with time modulation."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.norm = LayerNorm(config.dim, eps=config.eps)
        self.head = build_mlp("mlp1x_silu", config.dim, config.action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim ** 0.5)

    def forward(self, x, time_emb):
        """x: [B,T,dim], time_emb: [B,T,dim] -> [B,T,action_dim] in x.dtype (time_emb is float32 for stability)"""
        e0, e1 = (self.modulation.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
        return self.head(z).to(x.dtype)


# ---------------------------------------------------------------------------
# Expert block (only created when use_mot=True)
# ---------------------------------------------------------------------------

class ActionExpertBlock(nn.Module):
    """MoT block: owns action-side QKV/O projections into WAN head space."""

    def __init__(self, config: ActionExpertConfig, wan_config: dict):
        super().__init__()
        self.norm1 = LayerNorm(config.dim, eps=config.eps)
        self.norm2 = LayerNorm(config.dim, eps=config.eps)

        # WAN-space projections for joint attention
        nh = wan_config["num_heads"]
        hd = wan_config["head_dim"]
        wd = wan_config["dim"]
        self.wan_action_qkv = nn.Parameter(torch.randn(3, nh, config.dim, hd) / (config.dim * hd) ** 0.5)
        self.wan_action_o = nn.Linear(wd, config.dim, bias=False)
        self.wan_action_norm_q = RMSNorm(wd, eps=config.eps)
        self.wan_action_norm_k = RMSNorm(wd, eps=config.eps)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, config.dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim ** 0.5)


# ---------------------------------------------------------------------------
# Projection layer (only when use_mot=False)
# ---------------------------------------------------------------------------

class ActionProjection(nn.Module):
    """Project action tokens to/from WAN hidden dim when use_mot=False."""

    def __init__(self, action_dim: int, wan_dim: int):
        super().__init__()
        self.to_wan = nn.Linear(action_dim, wan_dim)
        self.from_wan = nn.Linear(wan_dim, action_dim)

    def project_in(self, x):
        return self.to_wan(x)

    def project_out(self, x):
        return self.from_wan(x)


# ---------------------------------------------------------------------------
# ActionExpert (top-level)
# ---------------------------------------------------------------------------

class ActionExpert(nn.Module):
    """Action expert supporting both MoT (use_mot=True) and simple (use_mot=False) modes."""

    def __init__(self, config: ActionExpertConfig, wan_config: dict = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256

        # Select encoder based on whether state tokens are enabled for this run.
        if config.use_state:
            self.input_encoder = StateActionEncoder(config)
        else:
            self.input_encoder = ActionEncoder(config)

        # Time embedding (shared with WAN structure)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.dim, config.dim * 6)
        )

        # Expert blocks (only for MoT mode)
        if config.use_mot:
            assert wan_config is not None, "wan_config required for MoT mode"
            self.blocks = nn.ModuleList([
                ActionExpertBlock(config, wan_config) for _ in range(config.num_layers)
            ])
        else:
            self.blocks = None
            # Simple mode: project action dim <-> WAN dim
            wan_dim = wan_config["dim"] if wan_config else 3072
            self.projection = ActionProjection(config.dim, wan_dim)

        # Decoder
        self.decoder = ActionDecoder(config)

        self._init_weights()
        logger.info(f"ActionExpert init: use_mot={config.use_mot}, action_mode={config.action_mode}, "
                     f"use_state={config.use_state}, params={sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init decoder head
        nn.init.zeros_(self.decoder.head[-1].weight)
        if self.decoder.head[-1].bias is not None:
            nn.init.zeros_(self.decoder.head[-1].bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
