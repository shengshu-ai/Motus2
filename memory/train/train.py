#!/usr/bin/env python3

from datetime import timedelta
import json
import os
import re
import shutil
import sys
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator
from accelerate.utils import (
    DeepSpeedPlugin,
    GradientAccumulationPlugin,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))
from utils.gradient_clipping import configure_gradient_clipping, clip_gradients

from models.motus import Motus, MotusConfig
from data.dataset import create_dataset, create_task_irrelevant_dataset, ar_collate_fn
from data.length_bucketing import TokenBudgetBatchSampler, maybe_build_length_bucket_batch_sampler
from data.repeat_sampler import RepeatedRandomSampler
from infra.load_balance import maybe_build_batch_balance_preprocessor
from infra.distributed_timeout import configure_deepspeed_group_timeouts
from utils.action_mode import resolve_action_mode_from_config
from utils.scheduler import create_scheduler
from utils.metric_reduction import reduce_training_metrics
from sample import evaluate_model, log_evaluation_metrics

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}










def _load_deepspeed_zero_stage(ds_config_path: Optional[str]) -> int:
    if not ds_config_path:
        return 0
    try:
        import json

        with open(ds_config_path, "r") as f:
            config = json.load(f)
        zero_cfg = config.get("zero_optimization", {})
        return int(zero_cfg.get("stage", 0) or 0)
    except Exception as e:
        logger.warning("Failed to inspect DeepSpeed config %s: %s", ds_config_path, e)
        return 0


def _is_sample_warning_record(record: logging.LogRecord) -> bool:
    try:
        message = record.getMessage()
    except Exception:
        return False
    return (
        message.startswith("Sample error (")
        or message.startswith("Dropping episode ")
    )


class _ExcludeSampleErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _is_sample_warning_record(record)


class _OnlySampleErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_sample_warning_record(record)


class _AtomicAppendFileHandler(logging.Handler):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            os.write(self._fd, msg.encode("utf-8", errors="replace"))
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if getattr(self, "_fd", None) is not None:
                os.close(self._fd)
                self._fd = None
        finally:
            super().close()


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------

def _to_host_scalars(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return `values` with scalar tensors as Python numbers.

    Calling `.item()` per entry costs one device sync each and blocks the host
    from queuing the next step, so float scalars are copied back together.
    """
    batched = [
        k for k, v in values.items()
        if torch.is_tensor(v) and v.numel() == 1 and v.is_floating_point()
    ]
    out = dict(values)
    if batched:
        stacked = torch.stack([out[k].detach().reshape(()).double() for k in batched])
        out.update(zip(batched, stacked.tolist()))
    for k, v in out.items():
        if torch.is_tensor(v):
            out[k] = v.item()
    return out


def _maybe_set_epoch(obj, epoch: int) -> bool:
    if obj is None:
        return False
    if hasattr(obj, "set_epoch"):
        obj.set_epoch(epoch)
        return True
    for attr in ("batch_sampler", "sampler"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj and _maybe_set_epoch(child, epoch):
            return True
    return False


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _qpos20_wrist_xyz_bounds_from_stats(
    dataset_cfg,
    wrist_xyz_normalization: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    stats_path = _cfg_get(dataset_cfg, "action_stats_path", None)
    if not stats_path:
        return tuple([-0.05] * 6), tuple([0.05] * 6)
    stats = json.loads(_resolve_repo_path(str(stats_path)).read_text(encoding="utf-8"))
    wrist_norm = stats.get("normalization", {}).get("wrist_relative", {})
    mode = str(wrist_xyz_normalization or "min_max").lower()
    if mode in {"quantile", "q01_q99"}:
        low_key, high_key = "xyz_q01", "xyz_q99"
    elif mode == "min_max":
        low_key, high_key = "xyz_min", "xyz_max"
    else:
        return tuple([-0.05] * 6), tuple([0.05] * 6)
    xyz_min = wrist_norm.get(low_key)
    xyz_max = wrist_norm.get(high_key)
    if xyz_min is None or xyz_max is None:
        raise ValueError(
            f"{stats_path} missing normalization.wrist_relative {low_key}/{high_key} "
            f"for wrist_xyz_normalization={mode!r}; "
            "re-run data/tianji_wuji/analyze_tianji_wuji_qpos.py"
        )
    if len(xyz_min) != 6 or len(xyz_max) != 6:
        raise ValueError(f"Expected 6D wrist relative xyz bounds in {stats_path}")
    return tuple(float(x) for x in xyz_min), tuple(float(x) for x in xyz_max)


def _qpos20_wrist_xyz_mean_std_from_stats(
    dataset_cfg,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
    stats_path = _cfg_get(dataset_cfg, "action_stats_path", None)
    if not stats_path:
        return tuple([0.0] * 6), tuple([1.0] * 6), 1e-4
    stats = json.loads(_resolve_repo_path(str(stats_path)).read_text(encoding="utf-8"))
    wrist_norm = stats.get("normalization", {}).get("wrist_relative", {})
    xyz_mean = wrist_norm.get("xyz_mean")
    xyz_std = wrist_norm.get("xyz_std")
    if xyz_mean is None or xyz_std is None:
        raise ValueError(
            f"{stats_path} missing normalization.wrist_relative xyz_mean/xyz_std; "
            "re-run data/tianji_wuji/analyze_tianji_wuji_qpos.py"
        )
    if len(xyz_mean) != 6 or len(xyz_std) != 6:
        raise ValueError(f"Expected 6D wrist relative xyz mean/std in {stats_path}")
    std_min = float(wrist_norm.get("std_min", 1e-4))
    return (
        tuple(float(x) for x in xyz_mean),
        tuple(float(x) for x in xyz_std),
        std_min,
    )


def _resolve_latent_budget(config: Any) -> Optional[int]:
    training_cfg = _cfg_get(config, "training")
    bucket_cfg = _cfg_get(training_cfg, "length_bucketing")
    budget = _cfg_get(bucket_cfg, "latent_budget")
    if budget is None:
        budget = _cfg_get(bucket_cfg, "token_budget")
    if budget is None:
        return None
    try:
        budget = int(budget)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None


def _int_list_from_value(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().view(-1).cpu().tolist()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if torch.is_tensor(item):
                out.append(int(item.detach().cpu().view(-1)[0].item()))
            else:
                out.append(int(item))
        return out
    return [int(value)]


def _latent_length_from_tensor(latent: torch.Tensor) -> Optional[int]:
    if not torch.is_tensor(latent) or latent.ndim == 0:
        return None
    if latent.ndim >= 5:
        # Batched latent layout is [B, C, T_lat, H, W].
        return int(latent.shape[-3])
    if latent.ndim >= 4:
        # Per-sample varlen layout is [C, T_lat, H, W].
        return int(latent.shape[1])
    return int(latent.shape[0])


def _batch_latent_lengths(batch: Dict[str, Any]) -> list[int]:
    latents = batch.get("latents")
    if isinstance(latents, (list, tuple)):
        lengths = [
            length
            for latent in latents
            for length in [_latent_length_from_tensor(latent)]
            if length is not None
        ]
        if lengths:
            return lengths
    elif torch.is_tensor(latents) and latents.ndim > 0:
        length = _latent_length_from_tensor(latents)
        if length is not None:
            batch_size = int(latents.shape[0]) if latents.ndim >= 5 else 1
            return [length] * batch_size

    return _int_list_from_value(batch.get("valid_len"))


def _batch_packing_stats(batch: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(batch, dict):
        return {}

    lengths = _batch_latent_lengths(batch)
    if not lengths:
        return {}
    return {
        "actual_sequences": len(lengths),
        "num_sequences": len(lengths),
        "actual_latents": sum(lengths),
        "T_lat": lengths,
    }


def setup_logging(rank: int = 0, log_level: str = "INFO"):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        f'[R{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    sample_warning_path = os.environ.get("MOTUS2_SAMPLE_WARNING_LOG")
    if sample_warning_path:
        stream_handler.addFilter(_ExcludeSampleErrorFilter())
        sample_handler = _AtomicAppendFileHandler(sample_warning_path)
        sample_handler.setLevel(logging.WARNING)
        sample_handler.setFormatter(formatter)
        sample_handler.addFilter(_OnlySampleErrorFilter())
        root.addHandler(sample_handler)

    root.addHandler(stream_handler)
    warnings.filterwarnings("ignore", message=r"No device id is provided.*", category=UserWarning)


def load_config(config_path: str) -> OmegaConf:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config: {config_path}")
    return config


def save_config_snapshot(config_path: str, config: OmegaConf, checkpoint_dir: str) -> None:
    snapshot_dir = Path(checkpoint_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    src_config_path = Path(config_path).resolve()
    copied_config_path = snapshot_dir / src_config_path.name
    if src_config_path != copied_config_path.resolve():
        shutil.copy2(src_config_path, copied_config_path)

    resolved_config_path = snapshot_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=str(resolved_config_path), resolve=True)

    logger.info(f"Copied config to {copied_config_path}")
    logger.info(f"Saved resolved config to {resolved_config_path}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trainer with Accelerator/DeepSpeed support."""

    def __init__(self, model, train_dataloader, val_dataloader=None,
                 optimizer=None, scheduler=None, device="cuda",
                 rank=0, world_size=1, checkpoint_dir="./checkpoints",
                 log_interval=1, save_interval=5000, val_interval=500,
                 report_to="wandb", tb_writer=None, accelerator=None, config=None,
                 batch_balance_preprocessor=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config
        self.eval_enabled = bool(config.system.get("eval_enabled", True))
        self.batch_balance_preprocessor = batch_balance_preprocessor
        self.global_step = 0
        self.latent_budget = _resolve_latent_budget(config)
        self.batch_stats_first_n = max(0, int(os.environ.get("MOTUS2_BATCH_STATS_FIRST_N", "20")))
        self.batch_stats_every = max(1, int(os.environ.get("MOTUS2_BATCH_STATS_EVERY", "20")))
        self.batch_stats_all_ranks = _env_flag("MOTUS2_BATCH_STATS_ALL_RANKS", False)
        self.profile_timing = _env_flag("MOTUS2_TIMING", False)
        self.timing_cuda_sync = _env_flag("MOTUS2_TIMING_CUDA_SYNC", True)
        self.timing_all_ranks = _env_flag("MOTUS2_TIMING_ALL_RANKS", False)
        self.timing_every = max(1, int(os.environ.get("MOTUS2_TIMING_EVERY", str(log_interval))))
        self._timing_accum = {}
        self._timing_microsteps = 0
        self._timing_active = False

        if self.profile_timing and (rank == 0 or self.timing_all_ranks):
            logger.info(
                "MOTUS2_TIMING enabled: every=%s cuda_sync=%s all_ranks=%s (only sampled steps are synchronized)",
                self.timing_every,
                self.timing_cuda_sync,
                self.timing_all_ranks,
            )

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _sync_for_timing(self) -> None:
        if (
            self.profile_timing
            and self._timing_active
            and self.timing_cuda_sync
            and torch.cuda.is_available()
            and getattr(self.device, "type", None) == "cuda"
        ):
            torch.cuda.synchronize(self.device)

    def _timing_start(self) -> float:
        self._sync_for_timing()
        return time.perf_counter()

    def _timing_elapsed(self, start: float) -> float:
        self._sync_for_timing()
        return time.perf_counter() - start

    def _add_timing(self, timing: Dict[str, float], key: str, value: float) -> None:
        timing[key] = timing.get(key, 0.0) + value

    def _accumulate_timing(self, timing: Dict[str, float]) -> None:
        if not self._timing_active:
            return
        for key, value in timing.items():
            self._timing_accum[key] = self._timing_accum.get(key, 0.0) + value
        self._timing_microsteps += 1

    def _pop_timing(self) -> Dict[str, float]:
        timing = dict(self._timing_accum)
        timing["microsteps"] = float(self._timing_microsteps)
        self._timing_accum = {}
        self._timing_microsteps = 0
        self._timing_active = False
        return timing

    def _format_timing(self, timing: Dict[str, float], step_time: float) -> str:
        ordered_keys = (
            "data",
            "h2d",
            "forward",
            "backward",
            "clip",
            "optimizer",
            "scheduler",
            "metrics_reduce",
        )
        parts = [f"total={step_time:.3f}s"]
        for key in ordered_keys:
            if key in timing:
                parts.append(f"{key}={timing[key]:.3f}s")
        if timing.get("microsteps", 0.0) > 1:
            parts.append(f"microsteps={timing['microsteps']:.0f}")
        if torch.cuda.is_available() and getattr(self.device, "type", None) == "cuda":
            # Allocator peak, not nvidia-smi's reserved figure: the reserved pool
            # only grows, so it cannot answer how much headroom a change leaves.
            parts.append(
                f"peak_alloc={torch.cuda.max_memory_allocated(self.device) / 2**30:.1f}GiB, "
                f"peak_reserved={torch.cuda.max_memory_reserved(self.device) / 2**30:.1f}GiB"
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        return ", ".join(parts)

    def _log_batch_stats_before_forward(self, stats: Dict[str, Any]) -> None:
        if (self.rank != 0 and not self.batch_stats_all_ranks) or not stats:
            return

        t_lat = [int(v) for v in stats.get("T_lat", [])]
        if not t_lat:
            return

        actual_latents = int(stats.get("actual_latents", sum(t_lat)))
        num_sequences = int(stats.get("num_sequences", len(t_lat)))
        latent_budget = self.latent_budget
        over_budget_samples = (
            [length for length in t_lat if length > latent_budget]
            if latent_budget is not None
            else []
        )
        over_budget_batch = (
            actual_latents > latent_budget
            if latent_budget is not None
            else False
        )
        over_budget = bool(over_budget_samples or over_budget_batch)
        step = self.global_step + 1

        if (
            step > self.batch_stats_first_n
            and step % self.batch_stats_every != 0
            and not over_budget
        ):
            return

        logger.info(
            "[batch-stats] step=%s rank=%s num_sequences=%s actual_latents=%s "
            "latent_budget=%s T_lat=%s over_budget=%s "
            "over_budget_samples=%s over_budget_batch=%s",
            step,
            self.rank,
            num_sequences,
            actual_latents,
            latent_budget,
            t_lat,
            over_budget,
            over_budget_samples,
            over_budget_batch,
        )

    def save_checkpoint(self, suffix=""):
        ckpt_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        step_match = re.search(r'step_(\d+)', Path(checkpoint_path).name)
        if step_match:
            self.global_step = int(step_match.group(1))
        self.accelerator.load_state(checkpoint_path)
        self._restore_scheduler_step()
        logger.info(f"Resumed from step {self.global_step}")

    def _restore_scheduler_step(self):
        """Fast-forward the LR schedule to the resumed step.

        Under DeepSpeed, accelerate's checkpoint carries no state for our custom
        schedulers (no scheduler.bin is written), so load_state leaves the counter at
        zero and a resumed run replays the warmup. scheduler.step() is called exactly
        once per optimizer step, so the counter equals global_step.
        """
        if self.scheduler is None or self.global_step <= 0:
            return
        sched = getattr(self.scheduler, "scheduler", self.scheduler)
        if not hasattr(sched, "set_step"):
            logger.warning(
                "Scheduler %s has no set_step(); LR schedule will restart from 0.",
                type(sched).__name__,
            )
            return
        sched.set_step(self.global_step)
        logger.info(
            "LR scheduler fast-forwarded to step %s (lr=%.3e)",
            self.global_step,
            sched.get_last_lr()[0],
        )

    def train_step(self, batch: Dict[str, Any]):
        """Single AR training step. Returns (metrics, did_optimizer_step)."""
        self.model.train()
        did_step = False
        timing = {}

        with self.accelerator.accumulate(self.model):
            # Move batch to device. Varlen full-trajectory batches keep tensors as
            # lists until the model packs them into one attention sequence.
            t0 = self._timing_start() if self._timing_active else None

            def _to_device(x, dtype=None):
                if x is None:
                    return None
                if isinstance(x, list):
                    return [v.to(self.device, dtype=dtype) if dtype is not None else v.to(self.device) for v in x]
                return x.to(self.device, dtype=dtype) if dtype is not None else x.to(self.device)

            latents = _to_device(batch['latents'], self.dtype)
            if logger.isEnabledFor(logging.DEBUG):
                # Cheap visibility into batch composition -- most useful with
                # TokenBudgetBatchSampler (data/length_bucketing.py), where the
                # sample count per step varies instead of being fixed.
                t_lats = [int(t.shape[1]) for t in latents]
                logger.debug(
                    "train_step batch: %d sample(s), T_lat=%s (total=%d)",
                    len(latents), t_lats, sum(t_lats),
                )
            actions = _to_device(batch['actions'], torch.float32)
            states = _to_device(batch.get('states'), torch.float32)
            lang = _to_device(batch.get('language_embedding'), self.dtype)
            valid_len = _to_device(batch.get('valid_len'))
            action_valid_mask = _to_device(batch.get('action_valid_mask'))
            video_valid_mask = _to_device(batch.get('video_valid_mask'))
            latent_start = _to_device(batch.get('latent_start'))
            values = _to_device(batch.get('values'), self.dtype)
            is_task_irrelevant = _to_device(batch.get('is_task_irrelevant'), self.dtype)
            if self._timing_active:
                self._add_timing(timing, "h2d", self._timing_elapsed(t0))

            # Forward (unwrap DDP/DeepSpeed wrapper; Motus uses custom training_step)
            t0 = self._timing_start() if self._timing_active else None
            raw_model = (
                self.accelerator.unwrap_model(self.model)
                if self.accelerator is not None
                else (self.model.module if hasattr(self.model, "module") else self.model)
            )
            loss_dict = raw_model.training_step(
                latents=latents,
                actions=actions,
                states=states,
                language_embeddings=lang,
                valid_len=valid_len,
                action_valid_mask=action_valid_mask,
                video_valid_mask=video_valid_mask,
                latent_start=latent_start,
                values=values,
                is_task_irrelevant=is_task_irrelevant,
                global_step=self.global_step,
            )
            if is_task_irrelevant is not None:
                loss_dict["task_irrelevant_ratio"] = is_task_irrelevant.mean().detach()
            if self._timing_active:
                self._add_timing(timing, "forward", self._timing_elapsed(t0))

            total_loss = loss_dict['total_loss']

            # Backward
            t0 = self._timing_start() if self._timing_active else None
            self.accelerator.backward(total_loss)
            if self._timing_active:
                self._add_timing(timing, "backward", self._timing_elapsed(t0))

            # Gradient clipping only at accumulation boundary
            if self.accelerator.sync_gradients:
                did_step = True
                grad_clip = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                t0 = self._timing_start() if self._timing_active else None
                clip_gradients(self.accelerator, self.model.parameters(), max_norm=grad_clip)
                if self._timing_active:
                    self._add_timing(timing, "clip", self._timing_elapsed(t0))

            # AcceleratedOptimizer auto-skips step/zero_grad on non-boundary mini-batches
            t0 = self._timing_start() if self._timing_active else None
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self._timing_active:
                self._add_timing(timing, "optimizer", self._timing_elapsed(t0))

        if did_step and self.scheduler:
            t0 = self._timing_start() if self._timing_active else None
            self.scheduler.step()
            if self._timing_active:
                self._add_timing(timing, "scheduler", self._timing_elapsed(t0))

        return {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, did_step, timing

    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500):
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting Motus2 training for {max_steps} steps")
        start_time = time.time()
        data_iter = iter(self.train_dataloader)
        epoch = 0

        while self.global_step < max_steps:
            self._timing_active = self.profile_timing and ((self.global_step + 1) % self.timing_every == 0)
            data_t0 = self._timing_start() if self._timing_active else None
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                _maybe_set_epoch(self.train_dataloader, epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            data_time = self._timing_elapsed(data_t0) if self._timing_active else 0.0

            if batch is None:
                continue

            if self.batch_balance_preprocessor is not None:
                batch = self.batch_balance_preprocessor(batch)
                if batch is None:
                    continue
            packing_stats = _batch_packing_stats(batch)
            self._log_batch_stats_before_forward(packing_stats)

            if not hasattr(self, '_step_start'):
                self._step_start = time.time()
            metrics, did_step, timing = self.train_step(batch)
            metrics.update({
                k: packing_stats[k]
                for k in ("actual_sequences", "actual_latents")
                if k in packing_stats
            })
            if self._timing_active:
                timing["data"] = data_time
                self._accumulate_timing(timing)
            if not did_step:
                continue
            step_time = time.time() - self._step_start
            self._step_start = time.time()
            self.global_step += 1
            timing_for_step = self._pop_timing() if self._timing_active else {}

            # Logging (all-reduce metrics for global average across ranks)
            if self.global_step % self.log_interval == 0:
                reduce_t0 = time.perf_counter() if timing_for_step else None
                metrics = reduce_training_metrics(metrics, self.device, dist)
                if reduce_t0 is not None:
                    timing_for_step["metrics_reduce"] = time.perf_counter() - reduce_t0
                if self.rank == 0:
                    lrs = [g['lr'] for g in self.optimizer.param_groups]
                    lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
                    loss_parts = f"video: {metrics['video_loss']:.4f}, action: {metrics['action_loss']:.4f}"
                    if 'action_raw_loss' in metrics and metrics['action_raw_loss'] > 0:
                        loss_parts += (
                            f", action_raw: {metrics['action_raw_loss']:.4f}"
                            f" [pos: {metrics.get('action_pos_raw_loss', 0.0):.4f},"
                            f" rot: {metrics.get('action_rot_raw_loss', 0.0):.4f},"
                            f" grip: {metrics.get('action_grip_raw_loss', 0.0):.4f}]"
                        )
                    if 'value_loss' in metrics:
                        loss_parts += f", value: {metrics['value_loss']:.4f}"
                    mode_str = ""
                    if 'num_vm_samples' in metrics:
                        mode_str = (f", Mode(pol/wm/vm): "
                                    f"{metrics.get('num_pol_samples', 0):.0f}/"
                                    f"{metrics.get('num_wm_samples', 0):.0f}/"
                                    f"{metrics.get('num_vm_samples', 0):.0f}")
                    packing_str = ""
                    if 'actual_sequences' in metrics and 'actual_latents' in metrics:
                        packing_str = (
                            f", actual_sequences: {metrics['actual_sequences']:.0f}, "
                            f"actual_latents: {metrics['actual_latents']:.0f}"
                        )
                    logger.info(
                        f"Step {self.global_step}/{max_steps}, "
                        f"Loss: {metrics['total_loss']:.4f} "
                        f"({loss_parts}), "
                        f"LR: {lr_str}, T: {step_time:.2f}s{packing_str}{mode_str}"
                    )
                    if "wandb" in self.report_to:
                        wandb.log({**metrics, 'step_time': step_time, 'epoch': epoch}, step=self.global_step)
                    if self.tb_writer:
                        for k, v in metrics.items():
                            self.tb_writer.add_scalar(f'train/{k}', v, self.global_step)

            if (
                timing_for_step
                and self.global_step % self.timing_every == 0
                and (self.rank == 0 or self.timing_all_ranks)
            ):
                logger.info(
                    "Timing Step %s/%s: %s",
                    self.global_step,
                    max_steps,
                    self._format_timing(timing_for_step, step_time),
                )

            # Validation
            if (
                self.eval_enabled
                and val_interval > 0
                and self.global_step % val_interval == 0
                and self.val_dataloader is not None
            ):
                if self.rank == 0:
                    # Guard: a rank-0 eval crash must NOT skip the barrier below, or
                    # the other ranks hang until the NCCL watchdog kills the job.
                    try:
                        val_metrics = evaluate_model(self.model, self.val_dataloader, self.accelerator, self.config)
                        log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)
                    except Exception as e:
                        logger.warning(
                            f"Validation eval failed at step {self.global_step}; "
                            f"skipping eval and continuing training: {e}",
                            exc_info=True,
                        )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # Checkpoint (free eval memory first to avoid OOM during NCCL all_reduce)
            if self.global_step % self.save_interval == 0:
                torch.cuda.empty_cache()
                self.save_checkpoint()

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"Training complete: {self.global_step} steps in {total_time:.1f}s")
        # accelerator.save_state() is collective (DeepSpeed ZeRO shards optimizer
        # state across ranks), so every rank must call it -- gating this on
        # rank==0 like the log line above deadlocks: rank 0 blocks forever in
        # the collective gather while the other ranks fall through to
        # destroy_process_group(). Matches the interval-based save above, which
        # is already called unconditionally on all ranks.
        self.save_checkpoint()


# ---------------------------------------------------------------------------
# Model + optimizer factory
# ---------------------------------------------------------------------------

def create_model_and_optimizer(config: OmegaConf, device: torch.device):
    """Create Motus model and optimizer from config."""
    c = config.common
    m = config.model

    resolved_mode = resolve_action_mode_from_config(m, config.dataset)
    action_mode = resolved_mode.action_mode
    wrist_xyz_normalization = resolved_mode.wrist_xyz_normalization
    qpos20_wrist_xyz_min, qpos20_wrist_xyz_max = _qpos20_wrist_xyz_bounds_from_stats(
        config.dataset, wrist_xyz_normalization,
    )
    qpos20_wrist_xyz_mean = (0.0,) * 6
    qpos20_wrist_xyz_std = (1.0,) * 6
    wrist_xyz_std_min = 1e-4
    if wrist_xyz_normalization == "mean_std":
        qpos20_wrist_xyz_mean, qpos20_wrist_xyz_std, wrist_xyz_std_min = (
            _qpos20_wrist_xyz_mean_std_from_stats(config.dataset)
        )
    logger.info(
        "Resolved action config: action_mode=%r -> layout=%s, "
        "normalization.hand=%s, normalization.wrist_xyz=%s",
        resolved_mode.profile,
        action_mode,
        resolved_mode.hand_normalization,
        wrist_xyz_normalization,
    )

    if action_mode in {"joint", "rel_joint", "abs_eef_rpy"}:
        action_dim, action_state_dim = 14, 14
    elif action_mode == "rel_eef":
        action_dim, action_state_dim = 16, 16
    elif action_mode in {"abs_eef", "rel_eef_6d"}:
        action_dim, action_state_dim = 20, 20
    elif action_mode in {"abs_eef_wxyz", "rel_eef_wxyz"}:
        action_dim, action_state_dim = 16, 16
    elif action_mode == "qpos20_rel6d":
        action_dim, action_state_dim = 58, 58
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    memory_cfg = m.get("memory", {})

    model_config = MotusConfig(
        device=str(device),
        wan_checkpoint_path=m.wan.checkpoint_path,
        vae_path=m.wan.get("vae_path", ""),
        wan_config_path=m.wan.get("config_path", ""),
        video_precision=m.wan.precision,
        gradient_checkpointing=m.wan.get("gradient_checkpointing", False),
        video_height=c.video_height,
        video_width=c.video_width,
        video_action_freq_ratio=c.get("video_action_freq_ratio", 6),
        action_upsample_ratio=config.dataset.get("action_upsample_ratio", 1),
        max_latent_window=c.get("max_latent_window", 10),
        fixed_chunk_size=m.get("fixed_chunk_size", c.get("fixed_chunk_size", 2)),
        num_layers=30,
        action_state_dim=action_state_dim,
        action_dim=action_dim,
        action_expert_dim=m.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=m.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=m.action_expert.get("norm_eps", 1e-6),
        use_state=m.get("use_state", True),
        state_mask_ratio=m.get("state_mask_ratio", 0.0),
        idm_mode=m.get("idm_mode", False),
        action_mode=action_mode,
        mode=m.get("mode", "joint"),
        memory_enabled=bool(memory_cfg.get("enabled", True)),
        memory_anchor_latents=memory_cfg.get("anchor_latents", 1),
        memory_gist_tokens_per_frame=memory_cfg.get("gist_tokens_per_frame", 8),
        memory_recent_latents=memory_cfg.get("recent_latents", 4),
        memory_prefix_frames=memory_cfg.get("prefix_frames", 0),
        memory_train_rope_offset_max=memory_cfg.get("train_rope_offset_max", 128),
        memory_retention_max_frames=memory_cfg.get("retention_max_frames", 0),
        memory_retention_full_prob=memory_cfg.get("retention_full_prob", 0.0),
        memory_retention_empty_prob=memory_cfg.get("retention_empty_prob", 0.0),
        memory_selector=memory_cfg.get("selector", "none"),
        memory_selector_budget=memory_cfg.get("selector_budget", 0),
        memory_selector_seed=memory_cfg.get("selector_seed", 0),
        memory_selector_gate_path=memory_cfg.get("selector_gate_path", ""),
        memory_selector_gate_sha256=memory_cfg.get("selector_gate_sha256", ""),
        action_rope_mode=m.get("action_rope_mode", "chunk_local"),
        policy_timestep_method=m.get("policy_timestep_method", "beta"),
        beta_alpha=m.get("beta_alpha", 7.0),
        beta_beta=m.get("beta_beta", 1.0),
        policy_sigmoid_scale=m.get("policy_sigmoid_scale", 1.0),
        policy_min_t=m.get("policy_min_t", 0.0),
        policy_max_t=m.get("policy_max_t", 1.0),
        video_timestep_shift=m.get("video_timestep_shift", 5.0),
        action_timestep_shift=m.get("action_timestep_shift", 1.0),
        action_timestep_method=m.get("action_timestep_method", m.get("policy_timestep_method", "logit_normal")),
        action_sigmoid_scale=m.get("action_sigmoid_scale", m.get("policy_sigmoid_scale", 1.0)),
        action_min_t=m.get("action_min_t", m.get("policy_min_t", 0.0)),
        action_max_t=m.get("action_max_t", m.get("policy_max_t", 1.0)),
        wm_ratio=m.get("wm_ratio", 0.2),
        batch_size=config.training.batch_size,
        video_loss_weight=m.loss_weights.video_loss_weight,
        action_loss_weight=m.loss_weights.action_loss_weight,
        warmup_steps=config.training.get("warmup_steps", 0),
        uniform_loss_weight_warmup_steps=m.loss_weights.get("uniform_loss_weight_warmup_steps", 0),
        rel_eef_position_loss_weight=m.loss_weights.get("rel_eef_position_loss_weight", 1.0),
        rel_eef_rotation_loss_weight=m.loss_weights.get("rel_eef_rotation_loss_weight", 1.0),
        rel_eef_gripper_loss_weight=m.loss_weights.get("rel_eef_gripper_loss_weight", 1.0),
        hand_qpos_loss_weight=m.loss_weights.get("hand_qpos_loss_weight", 1.0),
        wrist_xyz_normalization=wrist_xyz_normalization,
        qpos20_wrist_xyz_min=qpos20_wrist_xyz_min,
        qpos20_wrist_xyz_max=qpos20_wrist_xyz_max,
        qpos20_wrist_xyz_mean=qpos20_wrist_xyz_mean,
        qpos20_wrist_xyz_std=qpos20_wrist_xyz_std,
        wrist_xyz_std_min=wrist_xyz_std_min,
        num_inference_timesteps=m.inference.num_inference_timesteps,
        video_steps=m.inference.get("video_steps", None),
        decouple_inference_noise=m.inference.get("decouple_inference_noise", True),
        video_inference_final_noise=m.inference.get("video_inference_final_noise", 0.8),
        noisy_history_prob=m.get("noisy_history_prob", 0.5),
        noisy_history_sigma_max=m.get("noisy_history_sigma_max", 0.3),
        noisy_history_clean_prefix=m.get("noisy_history_clean_prefix", 0),
        load_pretrained_backbones=getattr(m, 'load_pretrained_backbones', None),
        value_enabled=m.get("value", {}).get("enabled", False),
        value_tokens_per_frame=m.get("value", {}).get("num_value_tokens", 0) if m.get("value", {}).get("enabled", False) else 0,
        value_num_bins=m.get("value", {}).get("num_bins", 201),
        value_target=m.get("value", {}).get("target", "progress"),
        value_loss_weight=m.get("value", {}).get("value_loss_weight", 10.0),
        vm_ratio=m.get("value", {}).get("vm_ratio", 0.0) if m.get("value", {}).get("enabled", False) else 0.0,
        progress_c=m.get("value", {}).get("progress_c", 0.01),
        varlen_attention_backend=m.get("varlen_attention_backend", "auto"),
    )

    model = Motus(model_config)

    # Optimizer with separate LR for WAN backbone
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))

    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_ids = {id(p) for p in wan_params}
    other_params = [p for p in all_trainable if id(p) not in wan_ids]

    param_groups = []
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})
    if wan_params:
        param_groups.append({'params': wan_params, 'lr': wan_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.training.weight_decay, betas=(0.9, 0.95))
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler


def create_dataloaders(config: OmegaConf, rank: int, world_size: int):
    train_ds = create_dataset(config, val=False)
    eval_enabled = bool(config.system.get("eval_enabled", True))
    val_ds = create_dataset(config, val=True) if eval_enabled else None
    irrelevant_train_ds = create_task_irrelevant_dataset(config, val=False)

    if len(train_ds) == 0:
        raise ValueError(
            "Training dataset is empty (len=0). For JSONL + robot filter: catalog "
            "lines often omit 'robot'; ensure code uses task prefix ac_one--... or "
            "set robot: null. Also check task_name / latent_phases=[] skips."
        )

    # NOTE: Do NOT use DistributedSampler here — accelerator.prepare() will
    # add its own BatchSamplerShard. Manual sampler + prepare = double sharding,
    # which can cause empty batches when dataset is small.
    nw = config.system.num_workers
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=config.system.pin_memory,
        persistent_workers=nw > 0,
        collate_fn=ar_collate_fn,
    )
    prefetch_factor = config.system.get("prefetch_factor", None)
    if nw > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    ti_cfg = config.dataset.get("task_irrelevant")
    ti_enabled = bool(
        ti_cfg
        and ti_cfg.get("enabled", False)
        and irrelevant_train_ds is not None
        and len(irrelevant_train_ds) > 0
    )
    sampling_ratio = float(ti_cfg.get("sampling_ratio", 0.0)) if ti_enabled else 0.0
    if ti_enabled and sampling_ratio > 0.0:
        concat_ds = ConcatDataset([train_ds, irrelevant_train_ds])
        main_len = len(train_ds)
        irr_len = len(irrelevant_train_ds)
        main_weight = (1.0 - sampling_ratio) / max(main_len, 1)
        irr_weight = sampling_ratio / max(irr_len, 1)
        weights = torch.full((main_len + irr_len,), main_weight, dtype=torch.double)
        weights[main_len:] = irr_weight
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            concat_ds,
            batch_size=config.training.batch_size,
            sampler=sampler,
            **loader_kwargs,
            drop_last=True,
        )
    else:
        bucket_sampler = maybe_build_length_bucket_batch_sampler(
            config, train_ds, config.training.batch_size
        )
        if bucket_sampler is not None:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=bucket_sampler,
                **loader_kwargs,
            )
        else:
            epoch_repeat = int(config.training.get("epoch_repeat", 1) or 1)
            if epoch_repeat > 1:
                logger.info(
                    "Sampler: %d shuffles per epoch (%d samples), amortizing the "
                    "DataLoader epoch-boundary resync over %dx more steps",
                    epoch_repeat,
                    len(train_ds) * epoch_repeat,
                    epoch_repeat,
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=config.training.batch_size,
                    sampler=RepeatedRandomSampler(train_ds, epoch_repeat),
                    **loader_kwargs,
                    drop_last=True,
                )
            else:
                train_loader = DataLoader(
                    train_ds,
                    batch_size=config.training.batch_size,
                    shuffle=True,
                    **loader_kwargs,
                    drop_last=True,
                )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            # Packed varlen trajectories cannot be stacked into one rectangular
            # tensor. Evaluation materializes this one-sample list batch below.
            batch_size=1,
            shuffle=False,
            **loader_kwargs,
            drop_last=False,
        )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Motus2 Training")
    parser.add_argument("--config", type=str, default="configs/robotwin.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime_output = os.environ.get("CHECKPOINT_DIR")
    if runtime_output:
        # Output routing is operational and lives outside the immutable training
        # parameter YAML. It cannot alter model/data/optimizer semantics.
        config.system.checkpoint_dir = runtime_output
    deepspeed_value = config.system.get("deepspeed_config", None)
    deepspeed_path = None
    if deepspeed_value:
        deepspeed_path = Path(str(deepspeed_value)).expanduser()
        if not deepspeed_path.is_absolute():
            deepspeed_path = (Path.cwd() / deepspeed_path).resolve()
        if not deepspeed_path.is_file():
            raise FileNotFoundError(
                "system.deepspeed_config from the training YAML does not exist: "
                f"{deepspeed_path}"
            )

    try:
        if getattr(config.resume, 'checkpoint_path', None) or \
           (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None)):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass

    dataset_name = os.path.splitext(os.path.basename(args.config))[0]
    config.system.checkpoint_dir = os.path.join(
        config.system.checkpoint_dir, dataset_name
    )
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    # Accelerator
    dist_timeout_sec = int(config.system.get("distributed_timeout_sec", 600))
    grad_accum_steps = int(config.training.get("gradient_accumulation_steps", 1))
    zero_stage = _load_deepspeed_zero_stage(
        str(deepspeed_path) if deepspeed_path else None
    )
    grad_accum_plugin = GradientAccumulationPlugin(
        num_steps=grad_accum_steps,
        sync_with_dataloader=False,
        sync_each_batch=zero_stage >= 2 and grad_accum_steps > 1,
    )
    # Normalize before Accelerator: accelerate rejects the literal string "none".
    report_to = config.logging.get("report_to", "tensorboard")
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
        log_with = report_to
    elif report_to == "none" or report_to is None:
        report_to = []
        log_with = None
    elif isinstance(report_to, str):
        report_to = [report_to]
        log_with = report_to
    else:
        log_with = report_to
    accelerator_kwargs = dict(
        deepspeed_plugin=(
            DeepSpeedPlugin(hf_ds_config=str(deepspeed_path))
            if deepspeed_path
            else None
        ),
        mixed_precision="bf16",
        log_with=log_with,
        project_dir=config.system.checkpoint_dir,
        project_config=ProjectConfiguration(total_limit=20),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_sec))],
    )
    if grad_accum_plugin is not None:
        accelerator_kwargs["gradient_accumulation_plugin"] = grad_accum_plugin
    else:
        accelerator_kwargs["gradient_accumulation_steps"] = grad_accum_steps
    configure_gradient_clipping(accelerator_kwargs, config.training)
    accelerator = Accelerator(**accelerator_kwargs)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, str(config.system.get("log_level", "INFO")))
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.device)
        logger.info(
            "Per-rank CUDA binding: accelerator.device=%s local_process_index=%s current_device=%s",
            accelerator.device,
            accelerator.local_process_index,
            torch.cuda.current_device(),
        )
    logger.info("Distributed process-group timeout: %ss", dist_timeout_sec)
    if zero_stage >= 2 and grad_accum_steps > 1:
        logger.info(
            "DeepSpeed ZeRO stage %s with gradient_accumulation_steps=%s: enabling sync_each_batch to avoid no_sync incompatibility.",
            zero_stage,
            grad_accum_steps,
        )
    stream_timeout_sec = config.system.get("stream_timeout_sec", None)
    if stream_timeout_sec is not None:
        os.environ["MOTUS2_STREAM_TIMEOUT_SEC"] = str(int(stream_timeout_sec))
        logger.info("smart_io stream timeout: %ss", int(stream_timeout_sec))
    stream_max_retries = config.system.get("stream_max_retries", None)
    if stream_max_retries is not None:
        os.environ["MOTUS2_STREAM_MAX_RETRIES"] = str(int(stream_max_retries))
        logger.info("smart_io stream max retries: %s", int(stream_max_retries))

    run_name = config.logging.get('run_name') or f"motus2_{dataset_name}_bs{config.training.batch_size}"
    config.system.checkpoint_dir = os.path.join(
        config.system.checkpoint_dir, run_name
    )
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    if rank == 0:
        save_config_snapshot(args.config, config, config.system.checkpoint_dir)

    # TensorBoard
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        tb_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_dir)
        # Log training config to TensorBoard text tab
        config_yaml = OmegaConf.to_yaml(config, resolve=True)
        tb_writer.add_text("config", f"```yaml\n{config_yaml}\n```", global_step=0)

    # Require an explicit WandB project name within the service length limit.
    if "wandb" in report_to:
        wandb_project = config.logging.get("wandb_project")
        if not wandb_project:
            if rank == 0:
                logger.warning(
                    "logging.report_to includes wandb but logging.wandb_project is "
                    "empty; skipping wandb (use report_to: none|tensorboard, or set "
                    "a short wandb_project)"
                )
            report_to = [item for item in report_to if item != "wandb"]
        elif rank == 0:
            wandb.init(
                project=str(wandb_project),
                config=OmegaConf.to_container(config, resolve=True),
                name=run_name,
            )

    try:
        model, optimizer, scheduler = create_model_and_optimizer(config, accelerator.device)
        first_param = next(model.parameters(), None)
        logger.info(
            "Model init complete: first_param_device=%s",
            first_param.device if first_param is not None else "<no-params>",
        )
        train_loader, val_loader = create_dataloaders(config, rank, world_size)

        # Finetune: load model weights only (no optimizer/scheduler). Use when pretrain done,
        # fine-tuning on new dataset. Mutually exclusive with resume.
        resume_path = getattr(getattr(config, "resume", None), "checkpoint_path", None)
        finetune_path = getattr(getattr(config, "finetune", None), "checkpoint_path", None)
        if finetune_path and not resume_path:
            logger.info(f"Loading finetune weights from {finetune_path} (model only)...")
            try:
                unwrapped = model.module if hasattr(model, "module") else model
                unwrapped.load_checkpoint_weights(finetune_path, strict=False)
                logger.info("Finetune weights loaded.")
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")
                raise

        # TokenBudgetBatchSampler yields variable-size latent-budget batches (no static
        # batch_size attribute), which accelerate's BatchSamplerShard only
        # supports with even_batches=False -- otherwise its default padding
        # path for an uneven final round needs a fixed size to synthesize a
        # filler batch and raises. drop_last=True on the sampler (its
        # default) is what keeps this safe: every rank still drops the same
        # trailing partial round instead of desyncing on step count. See
        # data/length_bucketing.py module docstring for the full rationale.
        if isinstance(getattr(train_loader, "batch_sampler", None), TokenBudgetBatchSampler):
            accelerator.even_batches = False
            logger.info(
                "Latent-budget TokenBudgetBatchSampler in use: "
                "latent_budget=%d, mbs=%d, over_budget_policy=%s; "
                "set accelerator.even_batches=False",
                train_loader.batch_sampler.latent_budget,
                train_loader.batch_sampler.mbs,
                train_loader.batch_sampler.over_budget_policy,
            )

            # DeepSpeed's own bookkeeping (train_batch_size = micro_batch *
            # grad_accum * world_size, used for its internal throughput timer
            # and config validation -- NOT for reshaping/asserting on the
            # actual tensors we hand to engine.backward()) needs a concrete
            # `train_micro_batch_size_per_gpu` instead of "auto", since "auto"
            # resolution reads it off `dataloader.batch_size`, which PyTorch
            # forces to None whenever a DataLoader is built with an explicit
            # `batch_sampler=` (true here, since batches are variable-size by
            # design). We always call `engine.backward()` exactly once per
            # token-budget batch (one dataloader item == one micro-batch
            # forward+backward, regardless of how many packed samples or
            # tokens it holds), so 1 is the semantically correct value here,
            # not a placeholder -- see configs/zero2.json (train_micro_batch_
            # size_per_gpu/train_batch_size/gradient_accumulation_steps are
            # all "auto" there).
            if accelerator.deepspeed_plugin is not None and accelerator.deepspeed_plugin.is_auto(
                "train_micro_batch_size_per_gpu"
            ):
                accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
                logger.info(
                    "Latent-budget TokenBudgetBatchSampler in use: pinned DeepSpeed "
                    "train_micro_batch_size_per_gpu=1 (dataloader has no static "
                    "batch_size for DeepSpeed's 'auto' resolution to read)."
                )

        # Accelerator prepare
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
        configure_deepspeed_group_timeouts(model, timedelta(seconds=dist_timeout_sec))
        prepared_model = model.module if hasattr(model, "module") else model
        prepared_first_param = next(prepared_model.parameters(), None)
        logger.info(
            "Accelerator prepare complete: first_param_device=%s",
            prepared_first_param.device if prepared_first_param is not None else "<no-params>",
        )

        trainer = Trainer(
            model=model, train_dataloader=train_loader, val_dataloader=val_loader,
            optimizer=optimizer, scheduler=scheduler, device=accelerator.device,
            rank=rank, world_size=world_size, checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval, save_interval=config.system.save_interval,
            val_interval=config.system.val_interval, report_to=report_to,
            tb_writer=tb_writer, accelerator=accelerator, config=config,
            batch_balance_preprocessor=maybe_build_batch_balance_preprocessor(
                config,
                accelerator=accelerator,
                device=accelerator.device,
            ),
        )

        trainer.train(
            max_steps=config.training.max_steps,
            resume_from=resume_path,
            val_interval=config.system.val_interval,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer:
            tb_writer.close()


if __name__ == "__main__":
    main()
