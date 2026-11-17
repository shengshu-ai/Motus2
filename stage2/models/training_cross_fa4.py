"""Guarded fixed-length FA4 text-cross operator for joint training.

Requires B4,Lq7424,Lk512,H24,D128 with BF16 tensors.
All 512 text positions stay present, including the original padded positions.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import torch

INTERFACE_SHA='27aa8985be0464bc872cf773d39c1b3266ce6f1b3dd2ed63e0f9829861b59c9b'

class UnsupportedCrossFA4(ValueError):pass

class FixedCrossFA4:
    def __init__(self,device,*,deterministic=False):
        device=torch.device(device)
        if device.type!='cuda':raise UnsupportedCrossFA4('Requires CUDA')
        if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        if importlib.metadata.version('flash-attn-4')!='4.0.0b14':raise UnsupportedCrossFA4('Pinned b14 only')
        from flash_attn.cute import interface
        if hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest()!=INTERFACE_SHA:
            raise UnsupportedCrossFA4('Incompatible FA4 interface')
        if torch.cuda.get_device_capability(device)!=(10,3):raise UnsupportedCrossFA4('SM103 only')
        self.call=interface.flash_attn_func;self.device=device;self.deterministic=bool(deterministic)
    def rejection(self,q,k,v,*,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),dtype=torch.bfloat16,version=None):
        if q_lens is not None or k_lens is not None:return 'lengths/masking unsupported: retain all512 text positions'
        if dropout_p!=0 or causal or softmax_scale is not None or q_scale is not None:return 'dropout/causal/custom scaling unsupported'
        if tuple(window_size)!=(-1,-1) or dtype!=torch.bfloat16 or version not in (None,2):return 'unsupported window/dtype/version'
        if any(t.ndim!=4 for t in (q,k,v)):return 'BLHD only'
        if tuple(q.shape)!=(4,7424,24,128) or tuple(k.shape)!=(4,512,24,128) or v.shape!=k.shape:return 'requires B4Q7424K512H24D128'
        if any(t.dtype!=torch.bfloat16 or t.device!=self.device or t.stride(-1)!=1 for t in (q,k,v)):return 'BF16/device/last stride mismatch'
        return None
    def __call__(self,q,k,v,*,flex_fallback=None,**kwargs):
        # Use the Wan cross-attention wrapper when this backend is ineligible.
        why=self.rejection(q,k,v,**kwargs)
        if why:
            if flex_fallback is not None:return flex_fallback(q,k,v,**kwargs)
            raise UnsupportedCrossFA4(why)
        out,_lse=self.call(q,k,v,softmax_scale=None,causal=False,pack_gqa=False,
                           deterministic=self.deterministic)
        return out


class TrainingCrossFA4:
    """Per-Motus dispatch and host coverage; no model tensors are cached."""

    def __init__(self):
        self.contexts = {}
        self.unavailable = {}
        self.fast_calls = 0
        self.fallback_calls = 0
        self.fallback_reasons = {}
        self.prepare_calls = 0

    def __call__(self, q, k, v, *, original, q_lens=None, k_lens=None,
                 dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
                 window_size=(-1, -1), deterministic=False,
                 dtype=torch.bfloat16, version=None):
        kwargs = dict(q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                      softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                      window_size=window_size, deterministic=deterministic,
                      dtype=dtype, version=version)

        def fallback(reason):
            self.fallback_calls += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            return original(q, k, v, **kwargs)

        # Preserve the original backend route for any different installation.
        backend = getattr(original, '__globals__', {})
        if (not backend.get('FLASH_ATTN_2_AVAILABLE', False)
                or (version is None and backend.get('FLASH_ATTN_3_AVAILABLE', False))):
            return fallback('original_backend_not_FA2')
        if torch.compiler.is_compiling():
            return fallback('original_compile_path')

        # Run the frozen, exact mathematical/shape guard BEFORE importing FA4.
        guard = FixedCrossFA4.__new__(FixedCrossFA4)
        guard.device = q.device
        guard_kwargs = {k: value for k, value in kwargs.items() if k != 'deterministic'}
        reason = guard.rejection(q, k, v, **guard_kwargs)
        if reason is not None:
            return fallback(reason)
        key = (q.device.type, q.device.index, bool(deterministic))
        if key in self.unavailable:
            return fallback(self.unavailable[key])
        context = self.contexts.get(key)
        if context is None:
            self.prepare_calls += 1
            try:
                context = FixedCrossFA4(q.device, deterministic=deterministic)
            except (UnsupportedCrossFA4, ImportError, importlib.metadata.PackageNotFoundError) as exc:
                reason = 'backend_unavailable:' + type(exc).__name__
                self.unavailable[key] = reason
                return fallback(reason)
            self.contexts[key] = context
        # Do not catch CUDA/operator failures: a failed backend must fail the job.
        out = context(q, k, v, **guard_kwargs)
        self.fast_calls += 1
        return out

    def summary(self):
        return dict(fa4_cross_calls=self.fast_calls,
                    original_cross_fallback_calls=self.fallback_calls,
                    fallback_reasons=dict(self.fallback_reasons),
                    prepare_calls=self.prepare_calls,
                    cached_backends=len(self.contexts),
                    unavailable_devices=len(self.unavailable),
                    supported_shape=dict(q=[4, 7424, 24, 128], kv=[4, 512, 24, 128]),
                    all_512_text_keys_retained=True)
