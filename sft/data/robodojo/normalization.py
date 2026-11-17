"""Shared training/deployment joint transforms; statistics are configured in YAML."""
import numpy as np
import torch


class JointNormalizer:
    def __init__(self, config=None, expected_dim=14):
        config = config or {}
        self.kind = config.get('type', 'none')
        if self.kind not in {'none', 'quantile', 'min_max'}:
            raise ValueError('RoboDojo normalization must be none, quantile or min_max')
        self.bounds = {}
        if self.kind != 'none':
            low_key, high_key = ('min', 'max') if self.kind == 'min_max' else ('q01', 'q99')
            for key in ('action', 'state'):
                lo = np.asarray(config[key][low_key], dtype=np.float32)
                hi = np.asarray(config[key][high_key], dtype=np.float32)
                if (lo.shape != (expected_dim,) or hi.shape != (expected_dim,)
                        or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                        or not (hi > lo).all()):
                    raise ValueError(f'invalid {key} {low_key}/{high_key} bounds')
                self.bounds[key] = (lo, hi)

    def transform(self, value, key, *, inverse=False):
        if self.kind == 'none':
            return value
        lo, hi = self.bounds[key]
        if isinstance(value, torch.Tensor):
            lo = torch.as_tensor(lo, dtype=value.dtype, device=value.device)
            hi = torch.as_tensor(hi, dtype=value.dtype, device=value.device)
            if self.kind == 'min_max':
                return value * (hi - lo) + lo if inverse else (value - lo) / (hi - lo)
            if inverse:
                bounded = torch.where(torch.isfinite(value), value.clamp(-1, 1), value)
                return (bounded + 1) * 0.5 * (hi - lo) + lo
            return 2 * (value.clamp(min=lo, max=hi) - lo) / (hi - lo) - 1
        value = np.asarray(value)
        if self.kind == 'min_max':
            return value * (hi - lo) + lo if inverse else (value - lo) / (hi - lo)
        if inverse:
            bounded = np.where(np.isfinite(value), np.clip(value, -1, 1), value)
            return (bounded + 1) * 0.5 * (hi - lo) + lo
        return 2 * (np.clip(value, lo, hi) - lo) / (hi - lo) - 1
