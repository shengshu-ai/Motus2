"""Native R/L xyz+rotvec+qpos52 to Wuji L/R qpos+xyz+rot6d58.

The model handles chunk-relative wrist conversion. This codec only changes
layout/rotation representation and applies the existing Wuji hand statistics.
It preserves the native palm frame and never retargets arm joints.
"""
import json
from pathlib import Path

import numpy as np
import torch


class Wuji58Codec:
    def __init__(self, stats_path, normalization='min_max'):
        stats = json.loads(Path(stats_path).read_text())
        if stats.get('ready_for_training') is False:
            raise ValueError('HandsOn statistics await new visual manifest acceptance')
        self.manifest_sha256 = stats.get('manifest_sha256')
        self.stat_window = stats.get('stat_window_config', {})
        if stats.get('statistics_split', 'train') != 'train':
            raise ValueError('HandsOn normalization must be computed on train only')
        if stats.get('qpos_order') != ['L_qpos20', 'R_qpos20']:
            raise ValueError('expected canonical left/right Wuji joint statistics')
        q = stats['normalization']['qpos']
        self.normalization = {'q01_q99': 'quantile'}.get(normalization, normalization)
        if self.normalization not in {'min_max', 'mean_std', 'quantile'}:
            raise ValueError(f'unsupported hand normalization: {normalization}')
        low_key, high_key = ('q01', 'q99') if self.normalization == 'quantile' else ('min', 'max')
        self.low = torch.tensor(q[low_key], dtype=torch.float32)
        high = torch.tensor(q[high_key], dtype=torch.float32)
        if self.low.shape != (40,) or high.shape != (40,):
            raise ValueError('expected 40 hand-joint bounds')
        if not torch.isfinite(self.low).all() or not torch.isfinite(high).all():
            raise ValueError('nonfinite hand-joint bounds')
        if (high < self.low).any():
            raise ValueError('reversed hand-joint bounds')
        self.span = (high - self.low).clamp(min=1e-6)
        if self.normalization == 'mean_std':
            self.low = torch.tensor(q['mean'], dtype=torch.float32)
            std = torch.tensor(q['std'], dtype=torch.float32)
            if (self.low.shape != (40,) or std.shape != (40,)
                    or not torch.isfinite(self.low).all() or not torch.isfinite(std).all()
                    or (std < 0).any()):
                raise ValueError('expected finite 40D hand mean/nonnegative std')
            self.span = std.clamp(min=float(q.get('std_min', 0.02)))

    def encode(self, native):
        """Absolute native 52D -> normalized-hand, absolute-pose 58D."""
        from utils.relative_eef.relative_eef import _mat_to_rot6d_torch
        x = torch.as_tensor(native).float()
        if x.shape[-1] != 52 or not torch.isfinite(x).all():
            raise ValueError('expected finite native 52D commands/states')
        low, span = self.low.to(x.device), self.span.to(x.device)
        q = (torch.cat([x[..., 32:52], x[..., 6:26]], -1) - low) / span
        if self.normalization == 'quantile':
            q = q.clamp(0, 1)

        def pose(offset):
            v = x[..., offset + 3:offset + 6]
            a, b, c = v.unbind(-1)
            z = torch.zeros_like(a)
            cross = torch.stack([z, -c, b, c, z, -a, -b, a, z], -1).reshape(*v.shape[:-1], 3, 3)
            angle = torch.linalg.vector_norm(v, dim=-1)[..., None, None]
            eye = torch.eye(3, dtype=x.dtype, device=x.device)
            rot = (eye + torch.sinc(angle / torch.pi) * cross
                   + 0.5 * torch.sinc(angle / (2 * torch.pi)).square() * (cross @ cross))
            return torch.cat([x[..., offset:offset + 3], _mat_to_rot6d_torch(rot)], -1)

        return torch.cat([q[..., :20], pose(26), q[..., 20:], pose(0)], -1)

    def decode(self, absolute):
        """Absolute 58D output -> native 52D, including hand denormalization."""
        from scipy.spatial.transform import Rotation
        from utils.relative_eef.relative_eef import _rot6d_to_mat_torch
        x = torch.as_tensor(absolute).float()
        if x.shape[-1] != 58 or not torch.isfinite(x).all():
            raise ValueError('expected finite absolute 58D model output')
        q = torch.cat([x[..., :20], x[..., 29:49]], -1)
        q = (q * self.span.to(x.device) + self.low.to(x.device)).cpu().numpy()

        def pose(offset):
            rot = _rot6d_to_mat_torch(x[..., offset + 3:offset + 9]).cpu().numpy()
            shape = rot.shape[:-2]
            vec = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)
            return np.concatenate([x[..., offset:offset + 3].cpu().numpy(), vec], -1)

        return np.concatenate([pose(49), q[..., 20:], pose(20), q[..., :20]], -1).astype(np.float32)

    def normalize(self, value, key):
        if key != 'state':
            raise ValueError('58D serving only encodes native states')
        return self.encode(value).cpu().numpy()

    def decode_relative(self, relative, anchor):
        """Streaming returns unscaled relative wrists, not absolute poses."""
        from utils.relative_eef import rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch
        absolute = rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch(
            torch.as_tensor(anchor).float(), torch.as_tensor(relative).float())
        return self.decode(absolute)

    def denormalize(self, value, key):
        if key != 'action':
            raise ValueError('58D serving only decodes absolute model actions')
        return self.decode(value)
