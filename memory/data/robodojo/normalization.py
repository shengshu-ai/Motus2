"""Shared training/deployment joint transforms; statistics are configured in YAML."""
import numpy as np
import torch


class JointNormalizer:
    def __init__(self, config=None):
        config = config or {}
        self.kind = config.get('type', 'none')
        if self.kind not in {'none', 'quantile'}:
            raise ValueError('RoboDojo normalization must be none or quantile')
        self.bounds = {}
        if self.kind == 'quantile':
            for key in ('action', 'state'):
                lo = np.asarray(config[key]['q01'], dtype=np.float32)
                hi = np.asarray(config[key]['q99'], dtype=np.float32)
                if (lo.shape != (14,) or hi.shape != (14,)
                        or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                        or not (hi > lo).all()):
                    raise ValueError(f'invalid {key} q01/q99 bounds')
                self.bounds[key] = (lo, hi)

    def transform(self, value, key, *, inverse=False):
        if self.kind == 'none':
            return value
        lo, hi = self.bounds[key]
        if isinstance(value, torch.Tensor):
            lo = torch.as_tensor(lo, dtype=value.dtype, device=value.device)
            hi = torch.as_tensor(hi, dtype=value.dtype, device=value.device)
            if inverse:
                bounded = torch.where(torch.isfinite(value), value.clamp(-1, 1), value)
                return (bounded + 1) * 0.5 * (hi - lo) + lo
            return 2 * (value.clamp(min=lo, max=hi) - lo) / (hi - lo) - 1
        value = np.asarray(value)
        if inverse:
            bounded = np.where(np.isfinite(value), np.clip(value, -1, 1), value)
            return (bounded + 1) * 0.5 * (hi - lo) + lo
        return 2 * (np.clip(value, lo, hi) - lo) / (hi - lo) - 1
