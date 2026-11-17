"""Stage1 trainer metadata independent of Accelerate scheduler type detection."""
import re
from pathlib import Path

import torch


def save_trainer_state(checkpoint_dir, global_step, scheduler):
    path = Path(checkpoint_dir) / "trainer_state.pt"
    state = {
        "format_version": 1,
        "global_step": int(global_step),
        "scheduler_class": type(scheduler).__name__ if scheduler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "next_lrs": [float(g["lr"]) for g in scheduler.optimizer.param_groups]
        if scheduler is not None else None,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_trainer_state(checkpoint_dir, scheduler):
    root = Path(checkpoint_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    path = root / "trainer_state.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = type(scheduler).__name__ if scheduler is not None else None
        if state.get("format_version") != 1 or state.get("scheduler_class") != expected:
            raise ValueError("Trainer checkpoint format or scheduler class differs from this run")
        if not isinstance(state.get("global_step"), int) or state["global_step"] < 0:
            raise ValueError("Trainer checkpoint has an invalid global_step")
        return state
    # When trainer metadata is absent, parse only the leaf checkpoint name,
    # never a parent directory that might itself contain 'step_N'.
    match = re.search(r"checkpoint_step_(\d+)(?:_|$)", root.name)
    if match is None:
        raise ValueError("Legacy checkpoint needs a checkpoint_step_N directory name")
    return {"global_step": int(match.group(1)), "legacy_root": str(root)}


def restore_scheduler(scheduler, state):
    if scheduler is None:
        return
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        lrs = state["next_lrs"]
        if len(lrs) != len(scheduler.optimizer.param_groups):
            raise ValueError("Checkpoint optimizer group count differs")
        for group, lr in zip(scheduler.optimizer.param_groups, lrs):
            group["lr"] = lr
        return
    # Legacy custom schedulers were never registered by Accelerator.prepare.
    # Standard torch schedulers may have a scheduler.bin; prefer its actual state.
    saved = Path(state["legacy_root"]) / "scheduler.bin"
    if saved.exists():
        scheduler.load_state_dict(torch.load(saved, map_location="cpu", weights_only=True))
        return  # optimizer.load_state has already restored the matching LR
    step = state["global_step"]
    if hasattr(scheduler, "step_count") and hasattr(scheduler, "get_lr_multiplier"):
        scheduler.step_count = step
        multiplier = scheduler.get_lr_multiplier(step)
        for group, base in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = base * multiplier
    else:
        inner = getattr(scheduler, "inner", scheduler)
        if not isinstance(inner, torch.optim.lr_scheduler.LRScheduler):
            raise ValueError("Cannot reconstruct legacy scheduler; supply its saved state")
        inner.step(step)
        minimum = getattr(scheduler, "min_lr", 0.0)
        for group in scheduler.optimizer.param_groups:
            group["lr"] = max(group["lr"], minimum)
