"""Finite FP64 L2-norm guard for the current DeepSpeed ZeRO2 instance.

No shared site-packages/class mutation. Install after accelerator.prepare and
before first backward, also when constructing an engine to restore a checkpoint.
"""
from __future__ import annotations
import hashlib
import inspect
import math
import numbers
from pathlib import Path
import types
import torch

_AUDITED_STAGE12_SHA256 = '478209eee30499ed06aaa199f8aabe760cc176853e251cf929a0ce1333d68b9f'



def _gradient_clip_value(value, where):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{where} must be a real number, got {type(value).__name__}')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{where} must be finite and >= 0, got {value}')
    return value


def configure_gradient_clipping(training, deepspeed_config=None):
    """Apply an explicit YAML clipping value to this run's plugin configuration.

    An absent YAML field preserves the DS configuration. Zero explicitly disables
    native clipping. Invalid values fail before Accelerator/model construction.
    """
    if 'grad_clip_norm' not in training:
        return None
    value = _gradient_clip_value(training['grad_clip_norm'], 'training.grad_clip_norm')
    if deepspeed_config is not None:
        if not isinstance(deepspeed_config, dict):
            raise TypeError('DeepSpeed plugin configuration must be a dictionary')
        deepspeed_config['gradient_clipping'] = value
    return value


def _require_norm(value, where, *, fp64=False):
    if not torch.is_tensor(value) or value.numel() != 1:
        raise FloatingPointError(f'{where}: expected one scalar norm tensor')
    if fp64 and value.dtype != torch.float64:
        raise FloatingPointError(f'{where}: norm must remain float64, got {value.dtype}')
    if not bool((torch.isfinite(value) & (value >= 0)).all().item()):
        raise FloatingPointError(f'{where}: nonfinite or negative norm: {value.detach().cpu().tolist()}')
    return value


def _make_instance_methods(native_direct, native_scaled, native_clip):
    # Take the already-loaded native runtime dependencies, without importing or
    # monkeypatching DeepSpeed. These exact filters/collectives are source-pinned.
    deps = native_direct.__globals__
    ds_dist = deps['dist']
    pipe_replicated = deps['PIPE_REPLICATED']
    is_model_parallel_parameter = deps['is_model_parallel_parameter']
    get_accelerator = deps['get_accelerator']

    def direct(self, gradients, params, norm_type=2):
        if float(norm_type) != 2.0:
            # This installation changes only active L2 arithmetic. Even native
            # infinity/general-p results must fail closed if invalid.
            return _require_norm(native_direct(self, gradients, params, norm_type), 'ZeRO norm(non-L2)')
        all_norms = []
        for g, p in zip(gradients, params):
            if hasattr(p, pipe_replicated) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or self.model_parallel_rank == 0:
                all_norms.append(torch.linalg.vector_norm(g.data.double().detach(), ord=2.0)
                                 .to(get_accelerator().current_device_name()))
        if all_norms:
            # Keep squares, reductions AND returned group norm in FP64. Casting
            # after sqrt is insufficient: the outer group norm squares again.
            total = torch.stack(all_norms).square().sum()
        else:
            total = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        ds_dist.all_reduce(total, op=ds_dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total, op=ds_dist.ReduceOp.SUM)
        total = total.pow(0.5)
        # Never convert invalid values to the native -1 sentinel: otherwise
        # scaled_global_norm could silently turn a failed group into norm~1.
        return _require_norm(total, 'ZeRO group L2 norm', fp64=True)

    def scaled(self, norm_type=2):
        # Native iteration order, groups, MoE and CPU-offload branches are kept;
        # installer refuses unsupported branches rather than claiming coverage.
        if float(norm_type) != 2.0:
            raise ValueError('Only L2 global norm is supported by this ZeRO2 implementation')
        total = native_scaled(self, norm_type)
        return _require_norm(total, 'ZeRO global scaled L2 norm', fp64=True)

    def clip(self, grad_groups_flat, total_norm):
        _require_norm(total_norm, 'ZeRO pre-Adam clip norm', fp64=True)
        scale = float(self.loss_scale)
        limit = float(self.clip_grad)
        if not math.isfinite(scale) or scale <= 0:
            raise FloatingPointError(f'ZeRO pre-Adam invalid loss_scale={scale}')
        if not math.isfinite(limit) or limit <= 0:
            raise FloatingPointError(f'ZeRO pre-Adam invalid clip_grad={limit}')
        # Exact native clip formula and in-place gradient scaling. No replacement
        # of gradients, no skipped optimizer step, no change to AdamW math/state.
        return native_clip(self, grad_groups_flat, total_norm)

    return direct, scaled, clip


def install_zero2_fp64_norm_guard(engine):
    """Install on this prepared engine, return a source/clip receipt; no new flags."""
    if not callable(getattr(engine, 'zero_optimization_stage', None)):
        return dict(installed=False, reason='not_deepspeed')
    stage = int(engine.zero_optimization_stage())
    if stage != 2:
        return dict(installed=False, reason='not_zero2', zero_stage=stage)
    clip = _gradient_clip_value(engine.gradient_clipping(), 'engine.gradient_clipping')
    if clip == 0:
        return dict(installed=False, reason='gradient_clipping_disabled', zero_stage=stage,
                    gradient_clipping=clip)
    zero = engine.optimizer
    if _gradient_clip_value(zero.clip_grad, 'ZeRO clip_grad') != clip:
        raise RuntimeError('Engine and ZeRO clipping configuration disagree')
    prior = getattr(zero, '_motus_fp64_norm_guard_receipt', None)
    if prior is not None:
        return dict(prior)
    if bool(getattr(zero, 'cpu_offload', False)):
        raise RuntimeError('FP64 norm guard: CPU-offload is not supported')
    if bool(getattr(zero, 'has_moe_layers', False)):
        raise RuntimeError('FP64 norm guard: MoE norm averaging is not supported')
    if not math.isfinite(float(zero.clip_grad)) or float(zero.clip_grad) <= 0:
        raise RuntimeError('FP64 norm guard requires configured positive gradient clipping')
    methods = [zero.get_grad_norm_direct, zero.scaled_global_norm, zero.unscale_and_clip_grads]
    if any(not isinstance(method, types.MethodType) for method in methods):
        raise RuntimeError('FP64 norm guard: native instance methods already replaced')
    functions = [method.__func__ for method in methods]
    source = Path(inspect.getsourcefile(functions[0]))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != _AUDITED_STAGE12_SHA256 or any(Path(inspect.getsourcefile(fn)) != source for fn in functions):
        raise RuntimeError(f'FP64 norm guard: runtime source differs; review first ({source_sha})')
    direct, scaled, clip = _make_instance_methods(*functions)
    # Mutate only this optimizer instance, never its class/shared installed file.
    zero.get_grad_norm_direct = types.MethodType(direct, zero)
    zero.scaled_global_norm = types.MethodType(scaled, zero)
    zero.unscale_and_clip_grads = types.MethodType(clip, zero)
    receipt = dict(schema='motus-zero2-fp64-norm-guard@1', installed=True, zero_stage=2,
                   optimizer_class=type(zero).__module__+'.'+type(zero).__qualname__,
                   installed_source=str(source), installed_source_sha256=source_sha,
                   gradient_clipping=float(zero.clip_grad), loss_scale=float(zero.loss_scale),
                   norm_dtype='float64', fail_closed=True, instance_only=True,
                   unchanged=['AdamW','gradient/master/moment dtypes','DP/MP filter and collective groups',
                              'clip formula','optimizer-step count','checkpoint state format'])
    zero._motus_fp64_norm_guard_receipt = receipt
    return dict(receipt)
