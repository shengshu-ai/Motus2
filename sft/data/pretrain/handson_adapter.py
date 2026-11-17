"""Explicit, fail-closed HandsOn stage2 sidecar contract (legacy sources unchanged)."""
import json
from pathlib import Path

import torch

SCHEMA = 'handson-stage2/raw54-v1'


def decode_payload(payload):
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
    def normalize(raw):
        raw = raw.float()
        q = torch.cat([raw[:, :20], raw[:, 27:47]], -1)
        q = (q - lo) / (hi - lo).clamp_min(1e-6)
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
    """Wan first latent is one frame; subsequent latents cover 24 real commands."""
    if action_frames <= phase:
        return 0
    return 1 + (int(action_frames) - int(phase) - 1) // 24


def gather(actions, states, latent_start, count, phase):
    if latent_start < 0 or count < 1 or latent_start + count > full_latents(len(actions), phase):
        raise ValueError('HandsOn request exceeds real action support')
    chunks, anchors = [], []
    for latent in range(latent_start, latent_start + count):
        if latent == 0:
            # Stage2 bootstrap convention: one initial observation/command.
            chunks.append(actions[phase:phase + 1].expand(24, -1))
            anchors.append(states[phase])
        else:
            begin = phase + 1 + (latent - 1) * 24
            chunks.append(actions[begin:begin + 24])
            anchors.append(states[begin - 1])
    return torch.stack(anchors), torch.cat(chunks)
