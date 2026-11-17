#!/usr/bin/env python3
"""
Learning rate schedulers for joint video-action training.
"""

import torch
from typing import Optional
import math
from diffusers.optimization import get_scheduler as hf_get_scheduler


class LambdaLinearScheduler:
    """
    Linear learning rate scheduler with warmup.
    
    This scheduler implements:
    1. Warmup phase: Linear increase from f_start to f_max over warm_up_steps
    2. Decay phase: Linear decrease from f_max to f_min over remaining steps
    
    Args:
        optimizer: PyTorch optimizer
        warm_up_steps: Number of warmup steps
        cycle_length: Total number of steps in the cycle
        f_max: Maximum learning rate multiplier (reached after warmup)
        f_min: Minimum learning rate multiplier (at end of cycle)
        f_start: Starting learning rate multiplier (at step 0)
    """
    
    def __init__(self, optimizer, warm_up_steps: int, cycle_length: int, 
                 f_max: float = 1.0, f_min: float = 0.1, f_start: float = 1e-6):
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.cycle_length = cycle_length
        self.f_max = f_max
        self.f_min = f_min
        self.f_start = f_start
        # Support per-parameter-group base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        # Backward-compat single base lr (first group's lr)
        self.base_lr = self.base_lrs[0]
        self.step_count = 0
        
        # Validate parameters
        assert warm_up_steps >= 0, "warm_up_steps must be non-negative"
        assert cycle_length > warm_up_steps, "cycle_length must be greater than warm_up_steps"
        assert f_max >= f_min, "f_max must be >= f_min"
        assert f_start >= 0, "f_start must be non-negative"
        
    def step(self):
        """Update learning rates for next step (supports multiple param groups)"""
        self.step_count += 1
        lr_multiplier = self.get_lr_multiplier(self.step_count)

        # Apply per-group base lr scaling
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx] if idx < len(self.base_lrs) else self.base_lr
            param_group['lr'] = base_lr * lr_multiplier
    
    def get_lr_multiplier(self, step: int) -> float:
        """Calculate learning rate multiplier for given step"""
        if step <= 0:
            return self.f_start
        elif step <= self.warm_up_steps:
            # Warmup: linear increase from f_start to f_max
            return self.f_start + (self.f_max - self.f_start) * step / self.warm_up_steps
        elif step < self.cycle_length:
            # Main phase: linear decay from f_max to f_min
            remaining_steps = self.cycle_length - step
            decay_steps = self.cycle_length - self.warm_up_steps
            return self.f_min + (self.f_max - self.f_min) * remaining_steps / decay_steps
        else:
            # After cycle ends, maintain minimum learning rate
            return self.f_min
    
    def get_last_lr(self):
        """Return current learning rates for all parameter groups"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step_count': self.step_count,
            'base_lr': self.base_lr,
            'base_lrs': self.base_lrs,
            'warm_up_steps': self.warm_up_steps,
            'cycle_length': self.cycle_length,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'f_start': self.f_start,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.warm_up_steps = state_dict['warm_up_steps']
        self.cycle_length = state_dict['cycle_length']
        self.f_max = state_dict['f_max']
        self.f_min = state_dict['f_min']
        self.f_start = state_dict['f_start']


class WarmupConstantScheduler:
    """Linear warmup (f_start → f_max) over warmup_steps, then constant multiplier f_max."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        f_start: float = 0.0,
        f_max: float = 1.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.f_start = float(f_start)
        self.f_max = float(f_max)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.base_lr = self.base_lrs[0]
        self.step_count = 0
        assert self.warmup_steps >= 0
        assert self.f_max >= self.f_start >= 0.0
        for i, g in enumerate(self.optimizer.param_groups):
            bl = self.base_lrs[i] if i < len(self.base_lrs) else self.base_lr
            g["lr"] = bl * self.f_start

    def step(self):
        self.step_count += 1
        m = self.get_lr_multiplier(self.step_count)
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx] if idx < len(self.base_lrs) else self.base_lr
            param_group["lr"] = base_lr * m

    def get_lr_multiplier(self, step: int) -> float:
        if step <= 0:
            return self.f_start
        if self.warmup_steps == 0:
            return self.f_max
        if step <= self.warmup_steps:
            return self.f_start + (self.f_max - self.f_start) * (step / self.warmup_steps)
        return self.f_max

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "f_start": self.f_start,
            "f_max": self.f_max,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
        self.warmup_steps = state_dict["warmup_steps"]
        self.f_start = state_dict["f_start"]
        self.f_max = state_dict["f_max"]
        m = self.get_lr_multiplier(self.step_count)
        for i, g in enumerate(self.optimizer.param_groups):
            bl = self.base_lrs[i] if i < len(self.base_lrs) else self.base_lr
            g["lr"] = bl * m


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration object
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = getattr(config.training, 'scheduler_type', 'cosine')
    
    if scheduler_type == "linear":
        return LambdaLinearScheduler(
            optimizer,
            warm_up_steps=config.training.warmup_steps,
            cycle_length=config.training.cycle_length,
            f_max=config.training.f_max,
            f_min=config.training.f_min,
            f_start=getattr(config.training, 'f_start', 1e-6)
        )
    elif scheduler_type == "cosine":
        # 保持向后兼容：原生PyTorch余弦（带min比例）
        T_max = getattr(config.training, 'max_steps', 10000)
        eta_min = float(getattr(config.training, 'min_lr', config.training.learning_rate * getattr(config.training, 'min_lr_ratio', 0.1)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_type == "diffusers_cosine":
        # 使用 diffusers.optimization.get_scheduler 的余弦调度
        total_steps = int(getattr(config.training, 'lr_schedule_steps', getattr(config.training, 'max_steps', 10000)))
        warmup_steps = int(getattr(config.training, 'warmup_steps', 0))
        base = float(config.training.learning_rate)
        min_lr = float(getattr(config.training, 'min_lr', base * getattr(config.training, 'min_lr_ratio', 0.0)))

        inner = hf_get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        class _MinLRWrapper:
            def __init__(self, inner_sched, opt, min_lr_value: float):
                self.inner = inner_sched
                self.optimizer = opt
                self.min_lr = float(min_lr_value)

            def step(self):
                self.inner.step()
                if self.min_lr > 0.0:
                    for g in self.optimizer.param_groups:
                        if g.get('lr', 0.0) < self.min_lr:
                            g['lr'] = self.min_lr

            def get_last_lr(self):
                try:
                    return self.inner.get_last_lr()
                except Exception:
                    return [g['lr'] for g in self.optimizer.param_groups]

            # 兼容 accelerate.save_state / load_state
            def state_dict(self):
                sd = getattr(self.inner, 'state_dict', None)
                return sd() if callable(sd) else {}

            def load_state_dict(self, state):
                ld = getattr(self.inner, 'load_state_dict', None)
                if callable(ld):
                    ld(state)

        return _MinLRWrapper(inner, optimizer, min_lr)
    elif scheduler_type == "warmup_constant":
        return WarmupConstantScheduler(
            optimizer,
            warmup_steps=int(getattr(config.training, "warmup_steps", 0)),
            f_start=float(getattr(config.training, "f_start", 0.0)),
            f_max=float(getattr(config.training, "f_max", 1.0)),
        )
    elif scheduler_type == "constant":
        # 固定学习率：乘子恒为 1，不改变 optimizer 初始 lr
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")