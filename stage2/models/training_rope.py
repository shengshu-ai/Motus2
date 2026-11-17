"""Fixed-layout training RoPE preparation and compiled FP32 rotation.
"""
from functools import lru_cache
import torch
from .rope import _rope_apply_real_only


def prepare_training_real_freqs(freqs, B, seq_len):
    if freqs.requires_grad:
        raise ValueError('Learnable frequencies must retain original per-call conversion path')
    # Keep the original rope_action_apply normalization/padding order exactly.
    if freqs.dim() == 3:
        if freqs.shape[0] < seq_len:
            pad_len = seq_len - freqs.shape[0]
            freqs_pad = freqs[-1:].expand(pad_len, *freqs.shape[1:]).clone()
            freqs = torch.cat([freqs, freqs_pad], dim=0)
        freqs = freqs.unsqueeze(0)
    elif freqs.dim() != 4:
        raise ValueError(f"Unsupported freqs shape: {tuple(freqs.shape)}")
    if freqs.dim() == 4 and freqs.shape[0] != B:
        if freqs.shape[1] == B:
            freqs = freqs.permute(1, 0, 2, 3).contiguous()
    if freqs.dim() == 4 and freqs.shape[1] < seq_len:
        pad_len = seq_len - freqs.shape[1]
        freqs_pad = freqs[:, -1:, :, :].expand(freqs.shape[0], pad_len, freqs.shape[2], freqs.shape[3]).clone()
        freqs = torch.cat([freqs, freqs_pad], dim=1)
    if freqs.shape[0] not in (1, B) or freqs.shape[1] != seq_len:
        raise ValueError(f"RoPE freqs/x shape mismatch: x batch/length={(B, seq_len)}, freqs={tuple(freqs.shape)}")
    # Resolve a possible conjugate view without detaching its gradient path.
    return torch.view_as_real(freqs.resolve_conj()).float()


def apply_training_real_rope(x, freqs_real_pairs, dtype):
    # Keep all four multiplies and add/sub in FP32, then the original final cast.
    return _rope_apply_real_only(x, freqs_real_pairs).to(dtype)


def make_training_rope_runner(backend='inductor'):
    kwargs = dict(backend=backend, dynamic=True, fullgraph=True)
    if backend == 'inductor':
        kwargs['options'] = {'triton.cudagraphs': False, 'emulate_precision_casts': True}
    return torch.compile(apply_training_real_rope, **kwargs)


_packed_rope_counts = {'packed_calls': 0, 'fallback_calls': 0, 'fallback_reasons': {}}


@lru_cache(maxsize=1)
def _reference_training_rope_runner():
    return make_training_rope_runner()


def _dispatch_training_rope(x, freqs_real_pairs, dtype):
    from .training_packed_rope import eligibility, packed_training_rope
    reason = eligibility(x, freqs_real_pairs, dtype)
    if reason is None:
        _packed_rope_counts['packed_calls'] += 1
        return packed_training_rope(x, freqs_real_pairs, dtype, block=1024, warps=4)
    _packed_rope_counts['fallback_calls'] += 1
    reasons = _packed_rope_counts['fallback_reasons']
    reasons[reason] = reasons.get(reason, 0) + 1
    return _reference_training_rope_runner()(x, freqs_real_pairs, dtype)


def training_rope_summary():
    return {**_packed_rope_counts, 'fallback_reasons': dict(_packed_rope_counts['fallback_reasons'])}


@lru_cache(maxsize=1)
def training_rope_runner():
    return _dispatch_training_rope
