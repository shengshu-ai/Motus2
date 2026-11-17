"""Route clipping to the backend that actually owns the optimizer update."""

import math


def configure_gradient_clipping(accelerator_kwargs, training_config):
    """Set DeepSpeed's clip before Accelerator constructs its engine.

    Accelerate's DeepSpeed backward also performs engine.step(), so clipping
    parameters after accelerator.backward() is too late for that backend.
    The training YAML is authoritative over the auxiliary DeepSpeed JSON.
    """
    max_norm = float(training_config.get("grad_clip_norm", 1.0))
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("training.grad_clip_norm must be finite and nonnegative")
    plugin = accelerator_kwargs.get("deepspeed_plugin")
    if plugin is not None:
        plugin.deepspeed_config["gradient_clipping"] = max_norm
    return max_norm


def clip_gradients(accelerator, parameters, max_norm):
    """Clip non-DeepSpeed gradients before optimizer.step, with AMP unscaling.

    DeepSpeed clips the global ZeRO gradient norm inside engine.step(), before
    the update. Do not clip its already consumed/partitioned gradients again.
    A zero threshold disables clipping on both backends.
    """
    backend = getattr(accelerator.distributed_type, "name", accelerator.distributed_type)
    if backend == "DEEPSPEED" or max_norm == 0:
        return None
    return accelerator.clip_grad_norm_(parameters, max_norm=max_norm)
