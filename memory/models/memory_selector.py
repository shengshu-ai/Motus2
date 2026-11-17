"""Deployment-time write selection for frames leaving the recent window.

A frame is offered exactly once, at the moment it would be evicted from the
recent window, while its KV is still in the cache. A selector may keep it or
drop it and may later drop previously kept frames, but it can never revive an
evicted frame or hold more than ``budget`` frames. The retained set is what the
KV cache keeps beyond the recent window.
"""

from __future__ import annotations

import hashlib
import heapq
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_POOL = (4, 4)
GATE_CONTEXT = (-1, 0, 1, 2, 3)


def pooled_latent_feature(latent: torch.Tensor) -> torch.Tensor:
    """[C, H, W] VAE latent -> flat pooled feature used by the write gate."""
    pooled = F.adaptive_avg_pool2d(latent.float().unsqueeze(0), GATE_POOL)
    return pooled.flatten()


def pooled_language_feature(language_embeddings) -> torch.Tensor | None:
    if language_embeddings is None:
        return None
    emb = language_embeddings
    if isinstance(emb, list):
        emb = emb[0]
    emb = emb.float()
    if emb.dim() == 3:
        emb = emb[0]
    nonzero = emb.abs().sum(dim=-1) > 0
    if bool(nonzero.any()):
        emb = emb[nonzero]
    return emb.mean(dim=0)


def gate_window_feature(frame_features: dict, frame_id: int, first_frame: int) -> torch.Tensor:
    """Concatenate pooled latents of ``frame_id + GATE_CONTEXT``; missing frames repeat the nearest one."""
    parts = []
    for offset in GATE_CONTEXT:
        target = max(first_frame, frame_id + offset)
        while target not in frame_features and target > first_frame:
            target -= 1
        parts.append(frame_features[target])
    return torch.cat(parts)


TIME_SCALE = 100.0


def gate_time_feature(frame_id: int, first_frame: int) -> torch.Tensor:
    return torch.tensor([(int(frame_id) - int(first_frame)) / TIME_SCALE], dtype=torch.float32)


class WriteGate(nn.Module):
    """Scores a frame for retention from the task text and either its local latent
    window (``feature="content"``) or only its time since the episode began (``"time"``)."""

    def __init__(self, frame_dim: int, language_dim: int, hidden: int = 512, feature: str = "content"):
        super().__init__()
        if feature not in {"content", "time"}:
            raise ValueError(f"Unsupported gate feature: {feature}")
        self.feature = feature
        self.frame_dim = int(frame_dim)
        self.language_dim = int(language_dim)
        window_dim = self.frame_dim * len(GATE_CONTEXT) if feature == "content" else 1
        self.frame_proj = nn.Sequential(nn.LayerNorm(window_dim), nn.Linear(window_dim, hidden))
        self.language_proj = nn.Sequential(nn.LayerNorm(self.language_dim), nn.Linear(self.language_dim, hidden))
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, window: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        return self.head(self.frame_proj(window) + self.language_proj(language)).squeeze(-1)


def load_write_gate(path: str, expected_sha256: str = "") -> WriteGate:
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"write gate {path} sha256 {digest} != expected {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gate = WriteGate(
        payload["frame_dim"], payload["language_dim"], payload.get("hidden", 512),
        feature=payload.get("feature", "content"),
    )
    gate.load_state_dict(payload["state_dict"])
    gate.eval()
    return gate


class MemorySelector:
    def __init__(self, kind: str, budget: int, seed: int = 0, gate: WriteGate | None = None):
        self.kind = str(kind)
        self.budget = int(budget)
        self.seed = int(seed)
        self.gate = gate
        self._resets = 0
        self.reset(0)

    def reset(self, first_frame: int = 0) -> None:
        self.first_frame = int(first_frame)
        self.next_offer = int(first_frame)
        self.kept: list[int] = []
        self.seen = 0
        self.stride = 1
        self.scores: list[tuple[float, int]] = []
        self.frame_features: dict[int, torch.Tensor] = {}
        self.language: torch.Tensor | None = None
        self.history: list[dict] = []
        self._resets += 1
        self.rng = np.random.default_rng([self.seed, self._resets, os.getpid()])

    def observe(self, frame_start: int, latent: torch.Tensor, language_embeddings=None) -> None:
        """Record pooled features of newly written frames ([B, C, K, H, W] latent)."""
        if self.kind != "gate":
            return
        if self.language is None:
            self.language = pooled_language_feature(language_embeddings)
        for k in range(int(latent.shape[2])):
            self.frame_features[int(frame_start) + k] = pooled_latent_feature(latent[0, :, k]).cpu()

    def advance(self, recent_lo: int) -> None:
        """Offer every frame below ``recent_lo`` that has not been offered yet."""
        while self.next_offer < int(recent_lo):
            self._offer(self.next_offer)
            self.next_offer += 1
        if self.kind == "gate":
            stale = self.next_offer + min(GATE_CONTEXT) - 1
            for fid in [f for f in self.frame_features if f < stale]:
                del self.frame_features[fid]

    def _offer(self, frame_id: int) -> None:
        kind, budget = self.kind, self.budget
        self.seen += 1
        score = None
        if kind == "none":
            pass
        elif kind == "all":
            self.kept.append(frame_id)
        elif kind == "fifo":
            self.kept.append(frame_id)
            if len(self.kept) > budget:
                self.kept.pop(0)
        elif kind == "first":
            if len(self.kept) < budget:
                self.kept.append(frame_id)
        elif kind == "random":
            if len(self.kept) < budget:
                self.kept.append(frame_id)
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < budget:
                    self.kept[j] = frame_id
        elif kind == "uniform":
            if (frame_id - self.first_frame) % self.stride == 0:
                self.kept.append(frame_id)
            while len(self.kept) > budget:
                self.stride *= 2
                self.kept = [f for f in self.kept if (f - self.first_frame) % self.stride == 0]
        elif kind == "gate":
            if self.gate.feature == "time":
                window = gate_time_feature(frame_id, self.first_frame)
            else:
                window = gate_window_feature(self.frame_features, frame_id, self.first_frame)
            language = self.language if self.language is not None else torch.zeros(self.gate.language_dim)
            with torch.no_grad():
                score = float(self.gate(window.unsqueeze(0), language.unsqueeze(0))[0])
            heapq.heappush(self.scores, (score, frame_id))
            if len(self.scores) > budget:
                heapq.heappop(self.scores)
            self.kept = sorted(f for _, f in self.scores)
        else:
            raise ValueError(f"Unsupported memory selector: {kind}")
        self.history.append({"frame": int(frame_id), "score": score, "kept": sorted(int(f) for f in self.kept)})

    def retained(self, device) -> torch.Tensor:
        return torch.tensor(sorted(self.kept), dtype=torch.long, device=device)
