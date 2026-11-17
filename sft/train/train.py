#!/usr/bin/env python3

from datetime import timedelta
import bisect
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
from typing import Dict, Any, Optional, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from accelerate import Accelerator, DataLoaderConfiguration
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
from data.dataset import (
    action_stats_paths,
    ar_collate_fn,
    co_dataset_configs,
    create_co_datasets,
    create_dataset,
    create_task_irrelevant_dataset,
)
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from data.samplers import SameBucketVariableBatchSampler, SameResolutionBatchSampler
from utils.action_mode import (
    is_qpos20_rel6d_family,
    qpos_family_action_dim,
    qpos_family_hand_dim,
    resolve_action_mode_from_config,
)
from utils.scheduler import create_scheduler
from utils.metric_reduction import reduce_training_metrics
from utils.co_training import (
    assert_stats_share_action_space,
    co_training_shares,
    qpos20_wrist_xyz_bounds_from_stats,
    qpos20_wrist_xyz_mean_std_from_stats,
)
from sample import evaluate_model, evaluate_scaling_model, log_evaluation_metrics

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


def setup_logging(rank: int = 0, log_level: str = "INFO"):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper()))

    for noisy_logger in ("megfile", "megfile.s3_path", "boto3", "botocore", "s3transfer"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    src_config_path = Path(config_path).resolve()
    copied_config_path = checkpoint_dir / src_config_path.name
    shutil.copy2(src_config_path, copied_config_path)

    resolved_config_path = checkpoint_dir / "resolved_config.yaml"
    OmegaConf.save(config=config, f=str(resolved_config_path), resolve=True)

    logger.info(f"Copied config to {copied_config_path}")
    logger.info(f"Saved resolved config to {resolved_config_path}")


def _env_or_config(config, key: str, env_name: str, default=None):
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return value
    scaling_cfg = config.get("scaling", None)
    if scaling_cfg is not None:
        value = scaling_cfg.get(key, None)
        if value not in (None, ""):
            return value
    return default


def _maybe_number(value):
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and "." not in value:
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


def _dataset_total_episodes(loader_or_dataset) -> Optional[int]:
    dataset = getattr(loader_or_dataset, "dataset", loader_or_dataset)
    if dataset is None:
        return None
    if hasattr(dataset, "total_episodes"):
        try:
            return int(dataset.total_episodes)
        except Exception:
            return None
    if hasattr(dataset, "datasets"):
        total = 0
        found = False
        for child in dataset.datasets:
            child_total = _dataset_total_episodes(child)
            if child_total is not None:
                total += child_total
                found = True
        return total if found else None
    return None


def _load_stereo_source_specs(config: OmegaConf) -> List[Dict[str, Any]]:
    d = config.get("dataset", {})
    specs: List[Dict[str, Any]] = []
    inline_sources = d.get("stereo_latent_sources", []) or []
    for source in inline_sources:
        specs.append(dict(OmegaConf.to_container(source, resolve=True) if OmegaConf.is_config(source) else source))

    source_path = d.get("stereo_latent_sources_path", None)
    if source_path:
        path = Path(str(source_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            with path.open("r") as f:
                payload = json.load(f)
            for source in payload.get("stereo_latent_sources", []) or []:
                specs.append(dict(source))
        except Exception as e:
            logger.warning("Failed to read stereo source specs from %s: %s", path, e)
    return specs


def _build_scaling_metadata(
    config: OmegaConf,
    run_name: str,
    checkpoint_dir: str,
    train_loader=None,
    val_loader=None,
) -> Dict[str, Any]:
    d = config.get("dataset", {})
    t = config.get("training", {})
    clickhouse = d.get("clickhouse", {}) or {}
    batch_size = int(t.get("batch_size", 1))
    grad_accum = int(t.get("gradient_accumulation_steps", 1))
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))

    dataset_sources = []
    for spec in _load_stereo_source_specs(config):
        source = spec.get("source")
        dataset_sources.append({
            "name": spec.get("dataset") or spec.get("name") or (str(source).rstrip("/").rsplit("/", 1)[-1] if source else None),
            "source": source,
            "resolution_bucket": spec.get("resolution_bucket"),
            "left_latent_name": spec.get("left_latent_name"),
            "right_latent_name": spec.get("right_latent_name"),
            "ratio": spec.get("ratio", spec.get("target_hours")),
        })

    train_manifest = d.get("train_manifest_path", None)
    val_manifest = d.get("val_manifest_path", None)
    split_manifest = d.get("split_manifest", None)
    if split_manifest:
        train_manifest = train_manifest or split_manifest.get("train", None)
        val_manifest = val_manifest or split_manifest.get("val", None)

    return {
        "scaling": {
            "data_bucket": _env_or_config(config, "data_bucket", "SCALING_DATA_BUCKET", None),
            "data_hours": _maybe_number(_env_or_config(config, "data_hours", "SCALING_DATA_HOURS", None)),
            "bucket_policy": _env_or_config(config, "bucket_policy", "SCALING_BUCKET_POLICY", "nested_append"),
            "bucket_parent": _env_or_config(config, "bucket_parent", "SCALING_BUCKET_PARENT", None),
        },
        "data": {
            "num_train_episodes": _dataset_total_episodes(train_loader),
            "num_val_episodes": _dataset_total_episodes(val_loader),
            "train_manifest_path": train_manifest,
            "val_manifest_path": val_manifest,
            "val_episode_path_prefix": d.get("val_episode_path_prefix", None),
            "clickhouse_table": clickhouse.get("table", None),
            "dataset_sources": dataset_sources,
            "dataset_mixture": d.get("dataset_mixture", None),
        },
        "training": {
            "run_name": run_name,
            "seed": t.get("seed", 0),
            "global_batch_size": batch_size * grad_accum * world_size,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": t.get("learning_rate", None),
            "scheduler_type": t.get("scheduler_type", None),
            "warmup_steps": t.get("warmup_steps", None),
            "max_steps": t.get("max_steps", None),
            "optimizer": "AdamW",
            "model_size": _env_or_config(config, "model_size", "SCALING_MODEL_SIZE", None),
        },
        "eval_scaling": _plain_omegaconf_value(config.get("eval_scaling", {})),
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_path": None,
            "checkpoint_source": _env_or_config(config, "checkpoint_source", "SCALING_CHECKPOINT_SOURCE", None),
        },
        "metrics": {
            "best_eval_scaling_step": None,
            "best_eval_scaling_action_mse": None,
            "best_metric_name": None,
            "final_eval_scaling_action_mse": None,
        },
    }


def _save_scaling_metadata(metadata: Dict[str, Any], checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir) / "scaling_metadata.yaml"
    OmegaConf.save(config=OmegaConf.create(metadata), f=str(path), resolve=True)
    logger.info("Saved scaling metadata to %s", path)
    return path


def _sync_scaling_metadata_to_wandb(metadata: Dict[str, Any], summary_only: bool = False) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    if not summary_only:
        wandb.config.update({"scaling_metadata": metadata}, allow_val_change=True)
    metrics = metadata.get("metrics", {})
    for key in ("best_eval_scaling_step", "best_eval_scaling_action_mse", "final_eval_scaling_action_mse"):
        if metrics.get(key) is not None:
            wandb.run.summary[key] = metrics[key]


def _update_scaling_metadata_metrics(metadata_path: Optional[Path], step: int, val_metrics: Dict[str, float]) -> None:
    if metadata_path is None or not metadata_path.exists():
        return
    candidates = [
        ("eval_scaling/action_mse", val_metrics.get("eval_scaling/action_mse")),
        ("eval_scaling/hand_qpos_mse", val_metrics.get("eval_scaling/hand_qpos_mse")),
        ("action_mse", val_metrics.get("action_mse")),
        ("policy_full_action_mse_loss", val_metrics.get("policy_full_action_mse_loss")),
        ("policy_decouple_action_mse_loss", val_metrics.get("policy_decouple_action_mse_loss")),
    ]
    metric_name, metric_value = next(((name, value) for name, value in candidates if value is not None), (None, None))
    if metric_value is None:
        return

    metadata = OmegaConf.to_container(OmegaConf.load(str(metadata_path)), resolve=True)
    metrics = metadata.setdefault("metrics", {})
    metric_value = float(metric_value)
    previous_best = metrics.get("best_eval_scaling_action_mse")
    if previous_best is None or metric_value < float(previous_best):
        metrics["best_eval_scaling_action_mse"] = metric_value
        metrics["best_eval_scaling_step"] = int(step)
        metrics["best_metric_name"] = metric_name
    metrics["final_eval_scaling_action_mse"] = metric_value
    _save_scaling_metadata(metadata, str(metadata_path.parent))
    _sync_scaling_metadata_to_wandb(metadata, summary_only=True)


def _add_eval_scaling_metrics(val_metrics: Dict[str, float]) -> Dict[str, float]:
    """Expose the scaling-law validation metric as a first-class scalar tag."""
    if "policy_full_action_mse_loss" in val_metrics:
        val_metrics.setdefault("eval_scaling/action_mse", val_metrics["policy_full_action_mse_loss"])
        val_metrics.setdefault("eval_scaling/policy_full_action_mse", val_metrics["policy_full_action_mse_loss"])
    if "policy_decouple_action_mse_loss" in val_metrics:
        val_metrics.setdefault("eval_scaling/policy_decouple_action_mse", val_metrics["policy_decouple_action_mse_loss"])
    return val_metrics


def _plain_omegaconf_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _apply_eval_scaling_dataset_overrides(val_dataset_cfg, eval_scaling_cfg) -> None:
    """Let scaling-law eval use a fixed validation subset independent of train data."""
    dataset_overrides = eval_scaling_cfg.get("dataset", None)
    if dataset_overrides is not None:
        dataset_overrides = _plain_omegaconf_value(dataset_overrides)
        if not isinstance(dataset_overrides, dict):
            raise ValueError("eval_scaling.dataset must be a mapping when provided")
        for key, value in dataset_overrides.items():
            val_dataset_cfg[key] = value

    alias_to_dataset_key = {
        "val_episode_path_prefix": "episode_path_prefix",
        "val_episode_path_prefixes": "episode_path_prefixes",
        "val_stereo_left_latent_name": "stereo_left_latent_name",
        "val_stereo_right_latent_name": "stereo_right_latent_name",
        "val_stereo_latent_name_pairs": "stereo_latent_name_pairs",
        "val_stereo_latent_sources_path": "stereo_latent_sources_path",
        "val_stereo_latent_sources": "stereo_latent_sources",
    }
    for eval_key, dataset_key in alias_to_dataset_key.items():
        if eval_key in eval_scaling_cfg:
            val_dataset_cfg[dataset_key] = _plain_omegaconf_value(eval_scaling_cfg.get(eval_key))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trainer with Accelerator/DeepSpeed support."""

    def __init__(self, model, train_dataloader, val_dataloader=None,
                 scaling_val_dataloader=None,
                 optimizer=None, scheduler=None, device="cuda",
                 rank=0, world_size=1, checkpoint_dir="./checkpoints",
                 log_interval=1, save_interval=5000, val_interval=500,
                 report_to="wandb", tb_writer=None, accelerator=None, config=None,
                 scaling_metadata_path: Optional[Path] = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scaling_val_dataloader = scaling_val_dataloader
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
        self.scaling_metadata_path = scaling_metadata_path
        self.global_step = 0
        system_cfg = config.get("system", {}) if config is not None else {}
        self.eval_enabled = bool(system_cfg.get("eval_enabled", True))
        scaling_cfg = config.get("eval_scaling", {}) if config is not None else {}
        self.scaling_eval_enabled = bool(scaling_cfg.get("enabled", False))
        self.scaling_val_interval = int(scaling_cfg.get("interval", 0) or 0)
        self.profile_timing = _env_flag("MOTUS2_TIMING", False)
        self.timing_cuda_sync = _env_flag("MOTUS2_TIMING_CUDA_SYNC", True)
        self.timing_all_ranks = _env_flag("MOTUS2_TIMING_ALL_RANKS", False)
        self.timing_every = max(1, int(os.environ.get("MOTUS2_TIMING_EVERY", str(log_interval))))
        self._timing_accum = {}
        self._timing_microsteps = 0
        self._timing_active = False
        self.debug_step_range = self._parse_debug_step_range(os.environ.get("MOTUS2_DEBUG_STEP_RANGE"))

        if self.profile_timing and (rank == 0 or self.timing_all_ranks):
            logger.info(
                "MOTUS2_TIMING enabled: every=%s cuda_sync=%s all_ranks=%s (only sampled steps are synchronized)",
                self.timing_every,
                self.timing_cuda_sync,
                self.timing_all_ranks,
            )

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_debug_step_range(raw: Optional[str]) -> Optional[tuple]:
        if not raw:
            return None
        text = str(raw).strip()
        try:
            if ":" in text:
                start_s, end_s = text.split(":", 1)
                start = int(start_s)
                end = int(end_s)
            else:
                start = end = int(text)
        except ValueError:
            logger.warning("Invalid MOTUS2_DEBUG_STEP_RANGE=%r; expected START:END", raw)
            return None
        if end < start:
            start, end = end, start
        return start, end

    def _debug_step_enabled(self, next_step: int) -> bool:
        if self.debug_step_range is None:
            return False
        start, end = self.debug_step_range
        return start <= int(next_step) <= end

    def _debug_batch_summary(self, batch: Dict[str, Any]) -> str:
        debug_items = batch.get("debug_episode")
        if not debug_items:
            return "<no debug_episode>"
        if isinstance(debug_items, dict):
            debug_items = [debug_items]
        parts = []
        for item in list(debug_items)[:2]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            latent_path = item.get("latent_path") or item.get("left_latent_path") or item.get("right_latent_path")
            parts.append(
                "dataset={dataset} task={task} episode={episode} total_latents={total_latents} "
                "window={window_size} start={start} valid={valid_len} latent={latent}".format(
                    dataset=item.get("dataset"),
                    task=item.get("task"),
                    episode=item.get("episode"),
                    total_latents=item.get("total_latents"),
                    window_size=item.get("window_size"),
                    start=item.get("start"),
                    valid_len=item.get("valid_len"),
                    latent=latent_path,
                )
            )
        return " | ".join(parts)

    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch: int) -> None:
        """Refresh shuffling for eval loaders that use epoch-aware samplers."""
        for sampler in (
            dataloader,
            getattr(dataloader, "batch_sampler", None),
            getattr(dataloader, "sampler", None),
        ):
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(int(epoch))

    def _aggregate_scaling_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not (dist.is_available() and dist.is_initialized()):
            return metrics

        device = self.device if torch.device(self.device).type == "cuda" else torch.device("cpu")
        world_size = dist.get_world_size()
        local_keys = [
            key
            for key, value in metrics.items()
            if str(key).startswith("eval_scaling/") and isinstance(value, (int, float))
        ]
        gathered_keys = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({key for keys in gathered_keys for key in (keys or [])})

        local_windows = float(metrics.get("eval_scaling/eval_windows", 0.0) or 0.0)
        aggregate: Dict[str, float] = {}
        sum_keys = {"eval_scaling/eval_batches", "eval_scaling/eval_windows",
                    "eval_scaling/skipped_windows_extreme_wrist_xyz"}
        passthrough_keys = {"eval_scaling/flow_samples", "eval_scaling/timesteps_per_trajectory"}

        for key in all_keys:
            value = float(metrics.get(key, 0.0) or 0.0)
            if key in passthrough_keys:
                tensor = torch.tensor(value, device=device, dtype=torch.float64)
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                aggregate[key] = float(tensor.item())
            elif key in sum_keys:
                tensor = torch.tensor(value, device=device, dtype=torch.float64)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                aggregate[key] = float(tensor.item())
            else:
                tensor = torch.tensor([value * local_windows, local_windows], device=device, dtype=torch.float64)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                aggregate[key] = float(tensor[0].item() / tensor[1].item()) if tensor[1].item() > 0 else 0.0
        return aggregate

    def _aggregate_validation_metrics(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute count-weighted validation means across all DDP ranks."""
        visualizations = {
            key: value
            for key, value in metrics.items()
            if str(key).startswith("visualization_")
        }
        local_keys = [
            key
            for key, value in metrics.items()
            if not str(key).startswith("_")
            and not str(key).startswith("visualization_")
            and isinstance(value, (int, float))
        ]
        if not (dist.is_available() and dist.is_initialized()):
            return {
                **{key: metrics[key] for key in local_keys},
                **visualizations,
            }

        world_size = dist.get_world_size()
        gathered_keys = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({key for keys in gathered_keys for key in (keys or [])})
        device = self.device if torch.device(self.device).type == "cuda" else torch.device("cpu")
        gathered_sum_keys = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_sum_keys, list(metrics.get("_sum_keys", [])))
        sum_keys = {key for keys in gathered_sum_keys for key in (keys or [])}
        aggregate: Dict[str, Any] = {}
        for key in all_keys:
            if key in sum_keys:
                total = torch.tensor(float(metrics.get(key, 0.0)), device=device, dtype=torch.float64)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                aggregate[key] = float(total.item())
                continue
            count = float(metrics.get(f"_count/{key}", 0.0) or 0.0)
            value = float(metrics.get(key, 0.0) or 0.0)
            weighted = torch.tensor(
                [value * count, count],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            if weighted[1].item() > 0:
                aggregate[key] = float(weighted[0].item() / weighted[1].item())
        if self.rank == 0:
            aggregate.update(visualizations)
        return aggregate

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
        return ", ".join(parts)

    def save_checkpoint(self, suffix=""):
        ckpt_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        self.accelerator.save_state(str(ckpt_dir))
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        step_match = re.search(r'step_(\d+)', Path(checkpoint_path).name)
        if step_match:
            self.global_step = int(step_match.group(1))
        self.accelerator.load_state(checkpoint_path)

        # DeepSpeed checkpoints produced by this training path do not contain
        # the custom LambdaLinearScheduler state (lr_scheduler is None in the
        # model-state file). Reconstruct its deterministic position from the
        # checkpoint step so resume does not restart warmup from step zero.
        scheduler = getattr(self.scheduler, "scheduler", self.scheduler)
        if (
            reset_scheduler
            and scheduler is not None
            and hasattr(scheduler, "step_count")
            and hasattr(scheduler, "get_lr_multiplier")
        ):
            previous_step = int(scheduler.step_count)
            scheduler.step_count = self.global_step
            if hasattr(scheduler, "apply_current_lr"):
                scheduler.apply_current_lr()
            else:
                lr_multiplier = scheduler.get_lr_multiplier(self.global_step)
                base_lrs = getattr(scheduler, "base_lrs", None)
                if base_lrs is None:
                    base_lrs = [scheduler.base_lr]
                for index, param_group in enumerate(scheduler.optimizer.param_groups):
                    base_lr = base_lrs[index] if index < len(base_lrs) else base_lrs[0]
                    param_group["lr"] = base_lr * lr_multiplier
            logger.info(
                "Aligned scheduler from step %s to resumed step %s (LR=%s)",
                previous_step,
                self.global_step,
                "/".join(f"{group['lr']:.2e}" for group in scheduler.optimizer.param_groups),
            )
        logger.info(f"Resumed from step {self.global_step}")

    def train_step(self, batch: Dict[str, Any]):
        """Single AR training step. Returns (metrics, did_optimizer_step)."""
        self.model.train()
        did_step = False
        timing = {}

        with self.accelerator.accumulate(self.model):
            # Move batch to device
            t0 = self._timing_start() if self._timing_active else None
            latents = batch['latents'].to(self.device, dtype=self.dtype)

            actions = batch['actions'].to(self.device, dtype=torch.float32)
            states = batch.get('states')
            if states is not None:
                states = states.to(self.device, dtype=torch.float32)
            lang = batch.get('language_embedding')
            if lang is not None:
                lang = lang.to(self.device, dtype=self.dtype)
            valid_len = batch.get('valid_len')
            if valid_len is not None:
                valid_len = valid_len.to(self.device)
            latent_start = batch.get('latent_start')
            if latent_start is not None:
                latent_start = latent_start.to(self.device)
            values = batch.get('values')
            if values is not None:
                values = values.to(
                    self.device,
                    dtype=torch.float32 if bool(
                        ((self.config.get("model", {}) or {}).get("value", {}) or {}).get("fp32", False)
                    ) else self.dtype,
                )
            sample_class = batch.get(SAMPLE_CLASS_KEY)
            if sample_class is not None:
                sample_class = sample_class.to(self.device, dtype=self.dtype)
            valid_masks = {
                key: batch[key].to(self.device)
                for key in ('action_valid_mask', 'video_valid_mask') if key in batch
            }
            if self._timing_active:
                self._add_timing(timing, "h2d", self._timing_elapsed(t0))

            # Forward
            t0 = self._timing_start() if self._timing_active else None
            debug_step = self._debug_step_enabled(self.global_step + 1)
            if debug_step:
                logger.info("[debug-step %s R%s] forward begin", self.global_step + 1, self.rank)
            vm_kwargs = {}
            if "vm_latents" in batch:
                vm_kwargs["vm_observations"] = {
                    key[3:]: value.to(self.device)
                    for key, value in batch.items() if key.startswith("vm_")
                }
            loss_dict = self.model.training_step(
                latents=latents,
                actions=actions,
                states=states,
                language_embeddings=lang,
                valid_len=valid_len,
                latent_start=latent_start,
                values=values,
                sample_class=sample_class,
                global_step=self.global_step,
                **valid_masks,
                **vm_kwargs,
            )
            if debug_step:
                logger.info(
                    "[debug-step %s R%s] forward end modes pol/wm/vm=%s/%s/%s",
                    self.global_step + 1,
                    self.rank,
                    loss_dict.get("num_pol_samples"),
                    loss_dict.get("num_wm_samples"),
                    loss_dict.get("num_vm_samples"),
                )
            if sample_class is not None:
                loss_dict["off_policy_ratio"] = (
                    (sample_class > SampleClass.MAIN).to(sample_class.dtype).mean().detach()
                )
            if self._timing_active:
                self._add_timing(timing, "forward", self._timing_elapsed(t0))

            total_loss = loss_dict['total_loss']

            # Backward
            t0 = self._timing_start() if self._timing_active else None
            if debug_step:
                logger.info("[debug-step %s R%s] backward begin", self.global_step + 1, self.rank)
            # DeepSpeed may update inside backward; capture the LR beforehand.
            opt = getattr(self.optimizer, "optimizer", self.optimizer)
            self._last_update_lrs = [g["lr"] for g in opt.param_groups]
            self.accelerator.backward(total_loss)
            if debug_step:
                logger.info("[debug-step %s R%s] backward end", self.global_step + 1, self.rank)
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
            if debug_step:
                logger.info("[debug-step %s R%s] optimizer begin", self.global_step + 1, self.rank)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if debug_step:
                logger.info("[debug-step %s R%s] optimizer end", self.global_step + 1, self.rank)
            if self._timing_active:
                self._add_timing(timing, "optimizer", self._timing_elapsed(t0))

        if did_step and self.scheduler:
            t0 = self._timing_start() if self._timing_active else None
            self.scheduler.step()
            if self._timing_active:
                self._add_timing(timing, "scheduler", self._timing_elapsed(t0))

        # Accumulation microsteps and non-logging steps need no host scalars.
        # Detach now so retaining metrics never retains the autograd graph.
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, did_step, timing

    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500):
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting Motus2 training for {max_steps} steps")
        start_time = time.time()
        data_iter = iter(self.train_dataloader)
        epoch = 0

        while self.global_step < max_steps:
            next_step = self.global_step + 1
            debug_step = self._debug_step_enabled(next_step)
            self._timing_active = self.profile_timing and ((self.global_step + 1) % self.timing_every == 0)
            data_t0 = self._timing_start() if self._timing_active else None
            if debug_step:
                logger.info("[debug-step %s R%s] fetching batch", next_step, self.rank)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                _maybe_set_epoch(self.train_dataloader, epoch)
                if self.config.dataset.get("type") == "handson":
                    self.train_dataloader.dataset.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            data_time = self._timing_elapsed(data_t0) if self._timing_active else 0.0

            if batch is None:
                logger.warning(
                    "Step %s local rank %s produced an empty batch before distributed skip",
                    next_step,
                    self.rank,
                )
            local_batch_ok = torch.tensor(
                0 if batch is None else 1,
                device=self.device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_batch_ok, op=dist.ReduceOp.MIN)
            if int(local_batch_ok.item()) == 0:
                if debug_step or self.rank == 0:
                    logger.warning(
                        "Step %s skipped because at least one rank produced an empty batch",
                        next_step,
                    )
                continue
            if debug_step:
                logger.info(
                    "[debug-step %s R%s] batch ready: %s",
                    next_step,
                    self.rank,
                    self._debug_batch_summary(batch),
                )

            if not hasattr(self, '_step_start'):
                self._step_start = time.time()
            if debug_step:
                logger.info("[debug-step %s R%s] train_step begin", next_step, self.rank)
            metrics, did_step, timing = self.train_step(batch)
            if debug_step:
                logger.info(
                    "[debug-step %s R%s] train_step end did_step=%s metrics=%s",
                    next_step,
                    self.rank,
                    did_step,
                    {k: metrics.get(k) for k in ("total_loss", "num_pol_samples", "num_wm_samples", "num_vm_samples")},
                )
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
                # _pop_timing resets _timing_active; use the sampled record to
                # include this segment without synchronizing unsampled steps.
                reduce_t0 = time.perf_counter() if timing_for_step else None
                metrics = reduce_training_metrics(metrics, self.device, dist)
                if reduce_t0 is not None:
                    timing_for_step["metrics_reduce"] = time.perf_counter() - reduce_t0
                if self.rank == 0:
                    lrs = self._last_update_lrs
                    lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
                    loss_parts = f"video: {metrics['video_loss']:.4f}, action: {metrics['action_loss']:.4f}"
                    if metrics.get('action_d2_loss_weight', 0.0) > 0:
                        loss_parts += f", action_d2: {metrics['action_d2_loss']:.4f}"
                    if 'action_raw_loss' in metrics and metrics['action_raw_loss'] > 0:
                        hand_label = "hand" if is_qpos20_rel6d_family(self.config.model.get("action_mode")) else "grip"
                        hand_metric = (
                            metrics.get('action_hand_raw_loss', 0.0)
                            if hand_label == "hand"
                            else metrics.get('action_grip_raw_loss', 0.0)
                        )
                        loss_parts += (
                            f", action_raw: {metrics['action_raw_loss']:.4f}"
                            f" [pos: {metrics.get('action_pos_raw_loss', 0.0):.4f},"
                            f" rot: {metrics.get('action_rot_raw_loss', 0.0):.4f},"
                            f" {hand_label}: {hand_metric:.4f}]"
                        )
                    if 'value_loss' in metrics:
                        loss_parts += f", value: {metrics['value_loss']:.4f}"
                    mode_str = ""
                    if 'num_vm_samples' in metrics:
                        mode_str = (f", Mode(pol/wm/vm): "
                                    f"{metrics.get('num_pol_samples', 0):.0f}/"
                                    f"{metrics.get('num_wm_samples', 0):.0f}/"
                                    f"{metrics.get('num_vm_samples', 0):.0f}")
                    logger.info(
                        f"Step {self.global_step}/{max_steps}, "
                        f"Loss: {metrics['total_loss']:.4f} "
                        f"({loss_parts}), "
                        f"LR: {lr_str}, T: {step_time:.2f}s{mode_str}"
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
                # Run eval on EVERY rank (was rank-0-only). evaluate_model
                # triggers a forward pass that, despite the cleanup wrapping in
                # train/sample.py:_reset_post_eval_module_state, still leaves
                # some rank-asymmetric residue on the inner MotusVLA module
                # (the exact source is not yet pinned down -- candidates include
                # _step_emb_cache_*, _inference_rope_cache, scheduler internal
                # state, RNG state, freqs_state, autocast caching, or DeepSpeed
                # ZeRO bookkeeping). With a rank-0-only call, that residue
                # desyncs rank 0 from peers and the next training-step's
                # DeepSpeed bucket allreduce mismatches on PG 1 while rank 0
                # races ahead to a PG 0 scalar allreduce -- observed as a hang
                # at step 1008 with val_interval=1000 (loop mode) and step ~16
                # with val_interval=10 (loop / joint, cut_banana_easy, 4 nodes
                # x 8 GPUs). Running eval on every rank keeps the state
                # evolution symmetric across ranks, which is the only
                # guaranteed-safe pattern when the cleanup contract is hard to
                # enforce.
                val_metrics = evaluate_model(
                    self.model, self.val_dataloader, self.accelerator, self.config
                )
                val_metrics = self._aggregate_validation_metrics(val_metrics)
                if self.rank == 0:
                    log_evaluation_metrics(
                        val_metrics, self.tb_writer, self.accelerator, self.global_step
                    )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            if (
                self.scaling_eval_enabled
                and self.scaling_val_interval > 0
                and self.global_step % self.scaling_val_interval == 0
                and self.scaling_val_dataloader is not None
            ):
                self._set_dataloader_epoch(self.scaling_val_dataloader, self.global_step)
                scaling_metrics = evaluate_scaling_model(
                    self.model,
                    self.scaling_val_dataloader,
                    self.accelerator,
                    self.config,
                )
                scaling_metrics = self._aggregate_scaling_metrics(scaling_metrics)
                if self.rank == 0:
                    log_evaluation_metrics(scaling_metrics, self.tb_writer, self.accelerator, self.global_step)
                    _update_scaling_metadata_metrics(self.scaling_metadata_path, self.global_step, scaling_metrics)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

            # Checkpoint (free eval memory first to avoid OOM during NCCL all_reduce)
            if self.global_step % self.save_interval == 0:
                torch.cuda.empty_cache()
                self.save_checkpoint()

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"Training complete: {self.global_step} steps in {total_time:.1f}s")
        # save_checkpoint is collective: accelerate's save_state calls DeepSpeed's
        # save_checkpoint, which barriers internally. Calling it under rank == 0 left
        # rank 0 alone in that barrier until the NCCL watchdog aborted the process, so
        # every run that reached its step budget exited 1 after the work was done.
        # Both terms below are identical on every rank, so the branch cannot diverge.
        if self.global_step % self.save_interval != 0:
            torch.cuda.empty_cache()
            self.save_checkpoint()


# ---------------------------------------------------------------------------
# Model + optimizer factory
# ---------------------------------------------------------------------------

def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _derived_action_upsample_ratio(dataset_cfg, fallback: int = 1) -> int:
    video_hz = int(_cfg_get(dataset_cfg, "video_hz", 30))
    action_cfg = _cfg_get(dataset_cfg, "action", None)
    train_action_hz = _cfg_get(dataset_cfg, "train_action_hz", None)
    if train_action_hz is None and action_cfg is not None:
        train_action_hz = _cfg_get(action_cfg, "train_hz", None)
    if train_action_hz is None:
        return int(_cfg_get(dataset_cfg, "action_upsample_ratio", fallback))
    train_action_hz = int(train_action_hz)
    if train_action_hz % video_hz != 0:
        raise ValueError(f"train_action_hz must be divisible by video_hz, got {train_action_hz}/{video_hz}")
    return train_action_hz // video_hz


def _qpos20_wrist_xyz_bounds_from_stats(
    stats_paths: list[str],
    wrist_xyz_normalization: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    bounds = qpos20_wrist_xyz_bounds_from_stats(stats_paths, wrist_xyz_normalization)
    if len(stats_paths) > 1:
        logger.info(
            "Wrist xyz envelope unioned over %d stats files (%s): min=%s max=%s",
            len(stats_paths),
            ", ".join(stats_paths),
            bounds[0],
            bounds[1],
        )
    return bounds


def create_model_and_optimizer(config: OmegaConf, device: torch.device):
    """Create Motus model and optimizer from config."""
    c = config.common
    m = config.model
    from utils.training_mix import resolved_model_mix, training_mix_summary
    mode_mix = resolved_model_mix(m)
    if m.get("training_mode_probabilities") is not None:
        logger.info("Explicit training sample mix: %s", training_mix_summary(config))
    unsupported_options = sorted(str(key) for key in m if str(key).startswith("wm_action_contrast_"))
    if unsupported_options:
        raise ValueError(
            f"Unsupported model configuration options: {unsupported_options}"
        )
    resolved_mode = resolve_action_mode_from_config(m, config.dataset)
    action_mode = resolved_mode.action_mode
    wrist_xyz_normalization = resolved_mode.wrist_xyz_normalization
    stats_paths = action_stats_paths(config)
    assert_stats_share_action_space(stats_paths)
    qpos20_wrist_xyz_min, qpos20_wrist_xyz_max = _qpos20_wrist_xyz_bounds_from_stats(
        stats_paths,
        wrist_xyz_normalization,
    )
    if (
        resolved_mode.profile != action_mode
        or resolved_mode.hand_normalization is not None
        or m.get("normalization") is not None
    ):
        logger.info(
            "Resolved action config: action_mode=%r -> layout=%s, "
            "normalization.hand=%s, normalization.wrist_xyz=%s",
            resolved_mode.profile,
            action_mode,
            resolved_mode.hand_normalization,
            wrist_xyz_normalization,
        )
    qpos20_wrist_xyz_mean = (0.0,) * 6
    qpos20_wrist_xyz_std = (1.0,) * 6
    wrist_xyz_std_min = 1e-4
    if wrist_xyz_normalization == "mean_std":
        qpos20_wrist_xyz_mean, qpos20_wrist_xyz_std, wrist_xyz_std_min = (
            qpos20_wrist_xyz_mean_std_from_stats(stats_paths)
        )

    if action_mode in {"joint", "rel_joint", "abs_eef_rpy"}:
        action_dim, action_state_dim = 14, 14
    elif action_mode == "abs_eef52":
        action_dim, action_state_dim = 52, 52
    elif action_mode == "rel_eef":
        action_dim, action_state_dim = 16, 16
    elif action_mode in {"abs_eef", "rel_eef_6d"}:
        action_dim, action_state_dim = 20, 20
    elif action_mode in {"abs_eef_wxyz", "rel_eef_wxyz"}:
        action_dim, action_state_dim = 16, 16
    elif qpos_family_hand_dim(action_mode) is not None:
        action_dim = qpos_family_action_dim(qpos_family_hand_dim(action_mode))
        action_state_dim = action_dim
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    video_rope_mode = c.get("video_rope_mode", "global")

    model_config = MotusConfig(
        device=str(device),
        wan_checkpoint_path=m.wan.checkpoint_path,
        vae_path=m.wan.get("vae_path", ""),
        wan_config_path=m.wan.get("config_path", ""),
        video_precision=m.wan.precision,
        gradient_checkpointing=m.wan.get("gradient_checkpointing", False),
        video_height=c.video_height,
        video_width=c.video_width,
        num_views=c.get("num_views", 1),
        video_action_freq_ratio=c.get("video_action_freq_ratio", 6),
        action_upsample_ratio=_derived_action_upsample_ratio(config.dataset, 1),
        max_latent_window=c.get("max_latent_window", 10),
        fixed_chunk_size=m.get("fixed_chunk_size", c.get("fixed_chunk_size", 2)),
        chunk_window_range=tuple(c.get("chunk_window_range", [1, 4])),
        chunk_window_size=c.get("chunk_window_size", 4),
        video_rope_mode=video_rope_mode,
        video_view_rope_mode=c.get("video_view_rope_mode", "shared"),
        num_layers=30,
        action_state_dim=action_state_dim,
        action_dim=action_dim,
        action_expert_dim=m.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=m.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=m.action_expert.get("norm_eps", 1e-6),
        use_mot=m.get("use_mot", False),
        use_state=m.get("use_state", False),
        state_mask_ratio=m.get("state_mask_ratio", 0.0),
        idm_mode=m.get("idm_mode", False),
        action_mode=action_mode,
        mode=m.get("mode", "joint"),
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
        wm_ratio=mode_mix["wm_ratio"],
        wm_mask_language=bool(m.get("wm_mask_language", False)),
        batch_size=config.training.batch_size,
        video_loss_weight=m.loss_weights.video_loss_weight,
        action_loss_weight=m.loss_weights.action_loss_weight,
        action_d2_loss_weight=m.loss_weights.get("action_d2_loss_weight", 0.0),
        warmup_steps=config.training.get("warmup_steps", 0),
        uniform_loss_weight_warmup_steps=m.loss_weights.get("uniform_loss_weight_warmup_steps", 0),
        off_policy_start_step=m.loss_weights.get("off_policy_start_step"),
        rel_eef_position_loss_weight=m.loss_weights.get("rel_eef_position_loss_weight", 1.0),
        rel_eef_rotation_loss_weight=m.loss_weights.get("rel_eef_rotation_loss_weight", 1.0),
        rel_eef_gripper_loss_weight=m.loss_weights.get("rel_eef_gripper_loss_weight", 1.0),
        hand_qpos_loss_weight=m.loss_weights.get(
            "hand_qpos_loss_weight",
            m.loss_weights.get("rel_eef_gripper_loss_weight", 1.0)
            if qpos_family_hand_dim(action_mode) is not None
            else 1.0,
        ),
        rtc_training_enabled=m.get("rtc_training", {}).get("enabled", False),
        rtc_committed_steps=m.get("rtc_training", {}).get("committed_steps", 0),
        rtc_replan_action_index=m.get("rtc_training", {}).get("replan_action_index", 0),
        rtc_training_prob=m.get("rtc_training", {}).get("probability", 0.5),
        qpos20_wrist_xyz_min=qpos20_wrist_xyz_min,
        qpos20_wrist_xyz_max=qpos20_wrist_xyz_max,
        wrist_xyz_normalization=wrist_xyz_normalization,
        qpos20_wrist_xyz_mean=qpos20_wrist_xyz_mean,
        qpos20_wrist_xyz_std=qpos20_wrist_xyz_std,
        wrist_xyz_std_min=wrist_xyz_std_min,
        num_inference_timesteps=m.inference.num_inference_timesteps,
        inference_scheduler=str(m.inference.get("inference_scheduler", "unipc")).lower(),
        num_action_inference_timesteps=m.inference.get("num_action_inference_timesteps", None),
        video_steps=m.inference.get("video_steps", None),
        decouple_inference_noise=m.inference.get("decouple_inference_noise", False),
        video_inference_final_noise=m.inference.get("video_inference_final_noise", 0.0),
        noisy_history_prob=m.get("noisy_history_prob", 0.5),
        noisy_history_sigma_max=m.get("noisy_history_sigma_max", 0.3),
        load_pretrained_backbones=getattr(m, 'load_pretrained_backbones', None),
        value_enabled=m.get("value", {}).get("enabled", False),
        value_tokens_per_frame=m.get("value", {}).get("num_value_tokens", 0) if m.get("value", {}).get("enabled", False) else 0,
        value_num_bins=m.get("value", {}).get("num_bins", 201),
        value_fp32=bool(m.get("value", {}).get("fp32", False)),
        value_target=m.get("value", {}).get("target", "progress"),
        value_loss_weight=m.get("value", {}).get("value_loss_weight", 10.0),
        vm_ratio=mode_mix["vm_ratio"],
        progress_c=m.get("value", {}).get("progress_c", 0.01),
        value_on_model=bool(
            m.get("value", {}).get("on_model", False)
            and m.get("value", {}).get("enabled", False)
        ),
        value_temporal_sampling=str(m.get("value", {}).get("temporal_sampling", "none")),
        value_on_policy=bool(m.get("value", {}).get("on_policy", False)),
        value_on_policy_max_sigma=m.get("value", {}).get("on_policy_max_sigma", None),
        # Deliberately not defaulted off m.inference.video_steps: that key only
        # bounds video_idm mode. The step count belongs to the MBRL config; the
        # dataclass default documents which of its keys applies.
        value_on_model_steps=int(
            m.get("value", {}).get("on_model_steps", 0) or 10
        ),
        value_on_model_scheduler=str(
            m.inference.get("inference_scheduler", "unipc")
        ),
        suboptimal_vm_share=m.get("value", {}).get("suboptimal_vm_share", None),
        fail_vm_share=mode_mix["fail_vm_share"],
        random_vm_share=mode_mix["random_vm_share"],
    )

    model = Motus(model_config)

    from utils.optimizer_groups import parameter_groups
    param_groups = parameter_groups(model, config.training)
    for group in param_groups:
        logger.info("Optimizer group %s: lr=%g tensors=%d elements=%d",
                    group.get("name", "legacy"), group["lr"], len(group["params"]),
                    sum(p.numel() for p in group["params"]))

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.training.weight_decay, betas=(0.9, 0.95))
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler


class _ConcatDatasetWithBuckets(ConcatDataset):
    """ConcatDataset that keeps per-sample bucket keys reachable.

    `SameResolutionBatchSampler` falls back to a single "default" bucket when the
    dataset has no `get_bucket_key`, so a plain ConcatDataset would not fail — it
    would quietly allow one batch to mix token grid sizes.
    """

    def get_bucket_key(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        child = self.datasets[dataset_idx]
        if hasattr(child, "get_bucket_key"):
            return child.get_bucket_key(sample_idx)
        return "default"


def create_dataloaders(config: OmegaConf, rank: int, world_size: int):
    train_ds = create_dataset(config, val=False)
    co_train_ds = create_co_datasets(config, val=False)
    eval_scaling_cfg = config.get("eval_scaling", {})
    val_ds = create_dataset(config, val=True) if bool(config.get("system", {}).get("eval_enabled", True)) else None
    if val_ds is not None and co_train_ds:
        # Each corpus contributes its own 2% split, per robotwin_episode_split_v1,
        # and they are reported as one number so the value stays comparable with
        # single-corpus runs.
        co_val_ds = [ds for ds in create_co_datasets(config, val=True) if len(ds) > 0]
        if co_val_ds:
            val_ds = _ConcatDatasetWithBuckets([val_ds, *co_val_ds])
    scaling_val_ds = None
    if bool(eval_scaling_cfg.get("enabled", False)):
        scaling_val_dataset_cfg = OmegaConf.create(OmegaConf.to_container(config.dataset, resolve=True))
        _apply_eval_scaling_dataset_overrides(scaling_val_dataset_cfg, eval_scaling_cfg)
        if config.dataset.get("type", "robotwin") in {"pretrain", "pretrain_sc"}:
            scaling_val_dataset_cfg.samples_per_episode = int(eval_scaling_cfg.get("val_samples_per_episode", 10))
            if scaling_val_dataset_cfg.get("refine_spatial_bucket_keys") is None:
                scaling_val_dataset_cfg.refine_spatial_bucket_keys = bool(
                    eval_scaling_cfg.get("refine_spatial_bucket_keys", True)
                )
        scaling_val_ds = create_dataset(config, val=True, dataset_cfg=scaling_val_dataset_cfg)
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
    bucket_by_resolution = bool(config.training.get("bucket_by_resolution", config.dataset.get("bucket_by_resolution", False)))
    bucket_by_temporal = bool(config.training.get("bucket_by_temporal", config.dataset.get("bucket_by_temporal", False)))
    weighted_bucket_sampling = bool(config.training.get("sample_with_replacement", False)) and hasattr(train_ds, "get_sample_weight")
    variable_bucket_batches = bucket_by_temporal or weighted_bucket_sampling

    def _make_eval_loader(dataset, batch_size: int, seed_offset: int):
        if dataset is None:
            return None
        if bucket_by_resolution:
            sampler = SameResolutionBatchSampler(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                seed=int(config.training.get("seed", 0)) + seed_offset,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                **loader_kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
            drop_last=False,
        )

    if co_train_ds or (ti_enabled and sampling_ratio > 0.0):
        if bucket_by_resolution:
            logger.warning("bucket_by_resolution is disabled for weighted sampling in this run")
        policy_pool = [train_ds, *co_train_ds]
        policy_lengths = [len(ds) for ds in policy_pool]
        # The main dataset never carries an explicit share; it absorbs whatever
        # the co-datasets do not pin.
        explicit = [None] + [ratio for _name, _cfg, ratio in co_dataset_configs(config)]
        shares = co_training_shares(policy_lengths, explicit)

        pool_weights = []
        for length, share in zip(policy_lengths, shares):
            pool_weights.append((1.0 - sampling_ratio) * share / max(length, 1))
        if ti_enabled and sampling_ratio > 0.0:
            members = [*policy_pool, irrelevant_train_ds]
            member_weights = [*pool_weights, sampling_ratio / max(len(irrelevant_train_ds), 1)]
        else:
            members = policy_pool
            member_weights = pool_weights

        concat_ds = ConcatDataset(members)
        weights = torch.empty(sum(len(ds) for ds in members), dtype=torch.double)
        cursor = 0
        for dataset, weight in zip(members, member_weights):
            weights[cursor : cursor + len(dataset)] = weight
            cursor += len(dataset)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=sum(policy_lengths),
            replacement=True,
        )
        if co_train_ds:
            logger.info(
                "Co-training pool: %s",
                ", ".join(
                    f"{name}={length} ({share:.1%})"
                    for name, length, share in zip(
                        ["main", *[n for n, _c, _r in co_dataset_configs(config)]],
                        policy_lengths,
                        shares,
                    )
                ),
            )
        train_loader = DataLoader(
            concat_ds,
            batch_size=config.training.batch_size,
            sampler=sampler,
            **loader_kwargs,
            drop_last=True,
        )
    elif bucket_by_resolution:
        if variable_bucket_batches:
            sampler = SameBucketVariableBatchSampler(
                train_ds,
                batch_size=config.training.batch_size,
                drop_last=True,
                seed=int(config.training.get("seed", 0)),
                replacement=bool(config.training.get("sample_with_replacement", True)),
                num_batches=config.dataset.get("num_batches_per_epoch", None),
            )
        else:
            sampler = SameResolutionBatchSampler(
                train_ds,
                batch_size=config.training.batch_size,
                drop_last=True,
                seed=int(config.training.get("seed", 0)),
            )
        bucket_summary = {
            key: len(indices)
            for key, indices in sorted(sampler.buckets.items(), key=lambda item: item[0])
        }
        if variable_bucket_batches:
            logger.info(
                "Using weighted same-bucket batches: counts=%s batch_sizes=%s replacement=%s",
                bucket_summary,
                getattr(sampler, "batch_sizes", {}),
                getattr(sampler, "replacement", False),
            )
        else:
            logger.info("Using same-resolution bucket batches: %s", bucket_summary)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            **loader_kwargs,
            drop_last=True,
        )

    val_loader = _make_eval_loader(val_ds, int(config.training.batch_size), 10000)
    scaling_val_loader = _make_eval_loader(
        scaling_val_ds,
        int(eval_scaling_cfg.get("batch_size", config.training.batch_size)),
        20000,
    )
    return train_loader, val_loader, scaling_val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Motus2 Training")
    parser.add_argument("--config", type=str, default="configs/robotwin.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Override: resume from checkpoint (full state)")
    parser.add_argument("--finetune", type=str, default=None, help="Override: load model weights only for finetune")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--report_to", type=str, default=None, choices=["wandb", "tensorboard", "all", "none"])
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint_dir:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.report_to:
        config.logging.report_to = args.report_to
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.run_name:
        config.logging.run_name = args.run_name
    if args.resume is not None:
        config.resume = OmegaConf.create({"checkpoint_path": args.resume})
    if args.finetune is not None:
        config.finetune = OmegaConf.create({"checkpoint_path": args.finetune})

    try:
        if getattr(config.resume, 'checkpoint_path', None) or \
           (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None)):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass

    dataset_name = os.path.splitext(os.path.basename(args.config))[0]
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, dataset_name)
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    # Accelerator / process-group timeout.
    #
    # InitProcessGroupKwargs only reaches the default process group. DeepSpeed
    # ZeRO also creates secondary groups with torch.distributed.new_group(), and
    # PyTorch's default NCCL timeout there is 600s unless a timeout is supplied.
    # Fill in that missing timeout so all gradient-reduction groups use the same
    # configured budget.
    dist_timeout_sec = int(os.environ.get("DISTRIBUTED_TIMEOUT_SEC", config.system.get("distributed_timeout_sec", 600)))
    dist_timeout = timedelta(seconds=dist_timeout_sec)
    if dist.is_available() and not getattr(dist.new_group, "_motus2_timeout_patched", False):
        _orig_new_group = dist.new_group

        def _new_group_with_default_timeout(*args, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = dist_timeout
            return _orig_new_group(*args, **kwargs)

        _new_group_with_default_timeout._motus2_timeout_patched = True
        dist.new_group = _new_group_with_default_timeout
    grad_accum_steps = int(config.training.get("gradient_accumulation_steps", 1))
    zero_stage = _load_deepspeed_zero_stage(args.deepspeed)
    ds_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed else None
    if ds_plugin is not None:
        ds_cfg = getattr(ds_plugin, "deepspeed_config", None)
        if isinstance(ds_cfg, dict):
            micro_bs = ds_cfg.get("train_micro_batch_size_per_gpu", None)
            grad_accum_cfg = ds_cfg.get("gradient_accumulation_steps", None)
            if micro_bs == "auto" or micro_bs is None:
                ds_cfg["train_micro_batch_size_per_gpu"] = int(config.training.batch_size)
            if grad_accum_cfg == "auto" or grad_accum_cfg is None:
                ds_cfg["gradient_accumulation_steps"] = grad_accum_steps

    # This loop is budgeted in optimizer steps and restarts the data iterator.
    # An epoch boundary must not flush a partial accumulation window.
    grad_accum_plugin = GradientAccumulationPlugin(
        num_steps=grad_accum_steps,
        sync_with_dataloader=False,
        sync_each_batch=zero_stage >= 2 and grad_accum_steps > 1,
    )
    accelerator_kwargs = dict(
        deepspeed_plugin=ds_plugin,
        mixed_precision="bf16",
        # "none" is our CLI/YAML sentinel; Accelerate expects Python None.
        log_with=(None if config.logging.get('report_to') == 'none'
                  else config.logging.get('report_to', 'tensorboard')),
        project_dir=config.system.checkpoint_dir,
        project_config=ProjectConfiguration(total_limit=20),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=dist_timeout)],
        # Custom batch samplers here do not expose a fixed batch_size.
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    accelerator_kwargs["gradient_accumulation_plugin"] = grad_accum_plugin
    configure_gradient_clipping(accelerator_kwargs, config.training)
    accelerator = Accelerator(**accelerator_kwargs)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)
    if config.training.get('seed_all', False):
        from accelerate.utils import set_seed
        set_seed(int(config.training.seed), device_specific=True)
        logger.info('Training RNG seed=%s (device_specific=True)', config.training.seed)
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
    if ds_plugin is not None and isinstance(getattr(ds_plugin, "deepspeed_config", None), dict):
        logger.info(
            "DeepSpeed runtime batch fallback: train_micro_batch_size_per_gpu=%s, gradient_accumulation_steps=%s",
            ds_plugin.deepspeed_config.get("train_micro_batch_size_per_gpu"),
            ds_plugin.deepspeed_config.get("gradient_accumulation_steps"),
        )
    stream_timeout_sec = config.system.get("stream_timeout_sec", None)
    if stream_timeout_sec is not None:
        os.environ["MOTUS2_STREAM_TIMEOUT_SEC"] = str(int(stream_timeout_sec))
        logger.info("smart_io stream timeout: %ss", int(stream_timeout_sec))
    stream_max_retries = config.system.get("stream_max_retries", None)
    if stream_max_retries is not None:
        os.environ["MOTUS2_STREAM_MAX_RETRIES"] = str(int(stream_max_retries))
        logger.info("smart_io stream max retries: %s", int(stream_max_retries))

    report_to = config.logging.report_to
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
    elif report_to == "none":
        report_to = []
    elif isinstance(report_to, str):
        report_to = [report_to]

    run_name = config.logging.get('run_name') or f"motus2_{dataset_name}_bs{config.training.batch_size}"
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    if rank == 0:
        save_config_snapshot(args.config, config, config.system.checkpoint_dir)

    # TensorBoard
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard logging requested but `tensorboard` is not installed in this Python environment.")
        tb_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_dir)
        # Log training config to TensorBoard text tab
        config_yaml = OmegaConf.to_yaml(config, resolve=True)
        tb_writer.add_text("config", f"```yaml\n{config_yaml}\n```", global_step=0)
        tb_writer.flush()

    # WandB
    wandb_enabled = False
    if rank == 0 and "wandb" in report_to:
        if wandb is None:
            logger.warning("WandB logging requested but `wandb` is not installed; continuing without WandB.")
            report_to = [x for x in report_to if x != "wandb"]
        else:
            try:
                wandb.init(
                    project=config.logging.wandb_project,
                    config=OmegaConf.to_container(config, resolve=True),
                    name=run_name,
                    dir=config.system.checkpoint_dir,
                )
                wandb_enabled = True
            except Exception as e:
                logger.warning("WandB init failed; continuing without WandB. Error: %s", e)
                report_to = [x for x in report_to if x != "wandb"]

    try:
        model, optimizer, scheduler = create_model_and_optimizer(config, accelerator.device)
        first_param = next(model.parameters(), None)
        logger.info(
            "Model init complete: first_param_device=%s",
            first_param.device if first_param is not None else "<no-params>",
        )
        train_loader, val_loader, scaling_val_loader = create_dataloaders(config, rank, world_size)
        scaling_metadata_path = None
        if rank == 0:
            scaling_metadata = _build_scaling_metadata(
                config=config,
                run_name=run_name,
                checkpoint_dir=config.system.checkpoint_dir,
                train_loader=train_loader,
                val_loader=scaling_val_loader if scaling_val_loader is not None else val_loader,
            )
            scaling_metadata_path = _save_scaling_metadata(scaling_metadata, config.system.checkpoint_dir)
            _sync_scaling_metadata_to_wandb(scaling_metadata)

        # Finetune: load model weights only (no optimizer/scheduler). Use when pretrain done,
        # fine-tuning on new dataset. Mutually exclusive with resume.
        resume_path = getattr(getattr(config, "resume", None), "checkpoint_path", None)
        finetune_path = getattr(getattr(config, "finetune", None), "checkpoint_path", None)
        if finetune_path and not resume_path:
            logger.info(f"Loading finetune weights from {finetune_path} (model only)...")
            try:
                unwrapped = model.module if hasattr(model, "module") else model
                unwrapped.load_checkpoint_weights(
                    finetune_path, strict=False,
                    strict_action=bool(config.finetune.get('require_action_compatibility', False)),
                    allow_new_state_encoder=bool(config.finetune.get('allow_new_state_encoder', False)),
                )
                logger.info("Finetune weights loaded.")
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")
                raise

        # Accelerator prepare
        if scaling_val_loader is not None:
            model, optimizer, train_loader, val_loader, scaling_val_loader, scheduler = accelerator.prepare(
                model,
                optimizer,
                train_loader,
                val_loader,
                scaling_val_loader,
                scheduler,
            )
        else:
            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model,
                optimizer,
                train_loader,
                val_loader,
                scheduler,
            )
        prepared_model = model.module if hasattr(model, "module") else model
        prepared_first_param = next(prepared_model.parameters(), None)
        logger.info(
            "Accelerator prepare complete: first_param_device=%s",
            prepared_first_param.device if prepared_first_param is not None else "<no-params>",
        )

        trainer = Trainer(
            model=model, train_dataloader=train_loader, val_dataloader=val_loader,
            scaling_val_dataloader=scaling_val_loader,
            optimizer=optimizer, scheduler=scheduler, device=accelerator.device,
            rank=rank, world_size=world_size, checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval, save_interval=config.system.save_interval,
            val_interval=config.system.val_interval, report_to=report_to,
            tb_writer=tb_writer, accelerator=accelerator, config=config,
            scaling_metadata_path=scaling_metadata_path,
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
        if rank == 0 and wandb_enabled:
            if wandb is not None:
                wandb.finish()
        if tb_writer:
            tb_writer.close()


if __name__ == "__main__":
    main()
