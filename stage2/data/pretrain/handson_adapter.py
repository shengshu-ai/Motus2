"""Explicit, fail-closed HandsOn stage2 sidecar contract (legacy sources unchanged)."""
import json
from pathlib import Path

import torch

from .action_pad import pad_or_trim_chunk

SCHEMA = 'handson-stage2/raw54-v1'


def decode_payload(payload, *, hand_bounds=None):
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or payload.get('action_hz') != 30:
        raise ValueError('invalid HandsOn stage2 schema/clock')
    actions, states = payload['commands'], payload['measured_states']
    if (not isinstance(actions, torch.Tensor) or not isinstance(states, torch.Tensor) or actions.ndim != 2
            or actions.shape[-1] != 54 or states.shape != actions.shape
            or not torch.isfinite(actions).all() or not torch.isfinite(states).all()):
        raise ValueError('expected aligned finite raw54 commands and measured states')
    bounds = payload['qpos_stats']
    if bounds.get('statistics_split') != 'train':
        raise ValueError('HandsOn bounds must be train-only')
    lo, hi = (torch.as_tensor(bounds[k], dtype=torch.float32) for k in ('min', 'max'))
    if lo.shape != (40,) or hi.shape != (40,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (hi < lo).any():
        raise ValueError('invalid HandsOn hand bounds')
    span = (hi - lo).clamp_min(1e-6)
    if hand_bounds is not None:
        lo, span = hand_bounds

    def normalize(raw):
        raw = raw.float()
        q = torch.cat([raw[:, :20], raw[:, 27:47]], -1)
        q = (q - lo) / span
        def wrist(offset):
            quat = raw[:, offset + 3:offset + 7]
            if not torch.allclose(quat.norm(dim=-1), torch.ones(len(quat)), rtol=1e-5, atol=1e-6):
                raise ValueError('expected unit xyzw quaternion')
            x, y, z, w = quat.unbind(-1)
            col1 = torch.stack([1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)], -1)
            col2 = torch.stack([2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)], -1)
            return torch.cat([raw[:, offset:offset+3], col1, col2], -1)
        return torch.cat([q[:, :20], wrist(20), q[:, 20:], wrist(47)], -1)

    return normalize(actions), normalize(states)


def load_payload(path, load=None):
    """tjwj.pt is a bare54 tensor; independent measured states never alias commands."""
    root = Path(path).parent
    if load is None:
        load = lambda p: torch.load(p, weights_only=True, map_location='cpu')
    metadata = json.loads((root / 'metadata.json').read_text())
    return dict(metadata, commands=load(str(root / 'tjwj.pt')),
                measured_states=load(str(root / 'measured_states.pt')))


def full_latents(action_frames, phase):
    """Latents needed to cover the action stream, keeping the partial tail.

    Wan's first latent is one frame; each later latent covers 24 commands. A tail
    with fewer than 24 commands still gets its own latent (ceil), rather than
    being cropped away: the encoder already produced that latent, and
    `gather` pads the short tail by repeating the final command.
    """
    if action_frames <= phase:
        return 0
    remaining = int(action_frames) - int(phase) - 1
    if remaining <= 0:
        return 1
    return 1 + (remaining + 23) // 24


def gather(actions, states, latent_start, count, phase):
    """Assemble per-latent action chunks, padding a short tail with the last command.

    Mirrors the generic path's `_gather_per_latent`: when a latent's 24-command
    window runs past the end of the stream, the final real command is repeated to
    fill it. The stored command is still real data; only the pad is synthetic.
    """
    if latent_start < 0 or count < 1 or latent_start + count > full_latents(len(actions), phase):
        raise ValueError('HandsOn request exceeds real action support')
    last_cmd = actions[-1]
    last_state = states[-1]
    chunks, anchors = [], []
    for latent in range(latent_start, latent_start + count):
        if latent == 0:
            # Stage2 bootstrap convention: one initial observation/command.
            chunks.append(actions[phase:phase + 1].expand(24, -1))
            anchors.append(states[phase])
        else:
            begin = phase + 1 + (latent - 1) * 24
            chunk = pad_or_trim_chunk(actions[begin:begin + 24], 24, fill=last_cmd)
            chunks.append(chunk)
            anchors.append(states[begin - 1] if begin - 1 < len(states) else last_state)
    return torch.stack(anchors), torch.cat(chunks)
