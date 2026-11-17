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
    set_seed,
)
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from models.motus import Motus, MotusConfig
from data.dataset import (
    action_stats_paths,
    wrist_action_stats_paths,
    ar_collate_fn,
    co_dataset_configs,
    create_co_datasets,
    create_dataset,
    create_task_irrelevant_dataset,
)
from data.sample_class import SAMPLE_CLASS_KEY, SampleClass
from data.samplers import (
    SameBucketVariableBatchSampler,
    SameResolutionBatchSampler,
    WeightedExpandedBatchSampler,
)
from utils.action_mode import (
    is_qpos20_rel6d_family,
    qpos_family_action_dim,
    qpos_family_hand_dim,
    resolve_action_mode_from_config,
)
from utils.scheduler import create_scheduler
from utils.deepspeed_norm import configure_gradient_clipping, install_zero2_fp64_norm_guard
from utils.metric_reduction import reduce_training_metrics
from utils.strict_finite import _all_finite, _strict_update_tensors
from utils.co_training import (
    assert_stats_share_action_space,
    build_wrist_bounds_by_source,
    co_training_shares,
    mixed_pretrain_wrist_per_dataset_enabled,
    qpos20_wrist_xyz_bounds_from_stats,
    qpos20_wrist_xyz_mean_std_from_stats,
    wrist_stats_paths_by_source,
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

def _iter_sampler_chain(obj):
    """Walk wrapped dataloader/sampler objects once, including Accelerate shards."""
    pending = [obj]
    seen = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("batch_sampler", "sampler"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                pending.append(child)


def _maybe_set_epoch(obj, epoch: int) -> bool:
    """Set outer iteration and inner sampler epoch without redundant resets."""
    epoch = int(epoch)
    applied = False
    for current in _iter_sampler_chain(obj):
        setter = getattr(current, "set_epoch", None)
        if not callable(setter):
            continue
        # An outer setter may already have forwarded to this child.  A sampler
        # resume() also sets epoch before storing its pending skip cursor.
        value = getattr(current, "epoch", getattr(current, "iteration", None))
        if value != epoch:
            setter(epoch)
        applied = True
    return applied



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
    if isinstance(loader_or_dataset, dict):
        totals = [_dataset_total_episodes(loader) for loader in loader_or_dataset.values()]
        return sum(totals) if totals and all(value is not None for value in totals) else None
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
            # Match PretrainDataset._load_source_definitions: relative source
            # tables belong to this checkout even when the launcher uses a
            # separate data working directory.
            path = Path(__file__).resolve().parents[1] / path
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
    return val_metrics


def _plain_omegaconf_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _isolate_eval_index_cache(train_dataset_cfg, eval_dataset_cfg, explicit_overrides, label):
    """Do not let a changed held-out index overwrite its inherited train cache."""
    index_keys = (
        "type", "episode_index_source", "jsonl_path", "dataset_dir", "clickhouse",
        "catalog_path", "episode_path_prefix", "episode_path_prefixes",
        "latent_dir", "latent_root", "latent_file_name", "latent_file_name_by_dataset",
        "stereo_left_latent_name", "stereo_right_latent_name", "stereo_latent_name_pairs",
        "stereo_latent_sources_path", "stereo_latent_sources",
        "stereo_latent_name_pairs_by_dataset", "latent_subpath", "lang_root",
        "lang_subpath", "lang_dir", "task_mode", "task_name", "max_episodes",
        "latent_phases", "dataset_include", "dataset_exclude", "robot", "action_mode",
        "exclude_explicit_fail",
    )
    changed = [key for key in index_keys if
        _plain_omegaconf_value(train_dataset_cfg.get(key)) !=
        _plain_omegaconf_value(eval_dataset_cfg.get(key))]
    if not changed:
        return
    train_cache = train_dataset_cfg.get("index_cache_path")
    if "index_cache_path" in explicit_overrides:
        eval_cache = eval_dataset_cfg.get("index_cache_path")
        if train_cache and eval_cache and Path(str(train_cache)).expanduser().resolve() == Path(str(eval_cache)).expanduser().resolve():
            raise ValueError(f"{label} changed index inputs but explicitly reuses the training index_cache_path")
        return
    if train_cache:
        eval_dataset_cfg["index_cache_path"] = None
        logger.info("%s changed index inputs %s; disabled inherited training index cache", label, changed)


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
        # Port the bounded diagnostic from e9f0a05 without enabling it by
        # default. It profiles one complete optimizer update, including GA.
        self.torch_profile_step = int(system_cfg.get("torch_profile_step", 0) or 0)
        self.torch_profile_rank = int(system_cfg.get("torch_profile_rank", 0) or 0)
        self.torch_profile_record_shapes = bool(system_cfg.get("torch_profile_record_shapes", False))
        self.torch_profile_with_flops = bool(system_cfg.get("torch_profile_with_flops", False))
        self.torch_profile_top_k = max(1, int(system_cfg.get("torch_profile_top_k", 200) or 200))
        self._torch_profiler = None
        self._torch_profile_completed = False
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
        sum_keys = {"eval_scaling/eval_batches", "eval_scaling/eval_windows"}
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
        dist.all_gather_object(gathered_keys, (local_keys, list(metrics.get("_sum_keys", []))))
        all_keys = sorted({key for keys, _ in gathered_keys for key in (keys or [])})
        sum_keys = {key for _, keys in gathered_keys for key in (keys or [])}
        device = self.device if torch.device(self.device).type == "cuda" else torch.device("cpu")
        aggregate: Dict[str, Any] = {}
        for key in all_keys:
            if key in sum_keys:
                total = torch.tensor(float(metrics.get(key, 0.0) or 0.0), device=device, dtype=torch.float64)
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

    def _fast_finite_check(self) -> bool:
        """Whether the reduction-based strict finite check is exact here.

        Decided once, on the training device itself: an injected NaN, +inf and
        -inf must each be caught in both float32 and bfloat16, and a clean
        tensor must pass. Otherwise the per-tensor check is kept.
        """
        if getattr(self, '_fast_finite', None) is None:
            try:
                caught = []
                for dtype in (torch.float32, torch.bfloat16):
                    for bad in (float('nan'), float('inf'), float('-inf')):
                        probe = torch.ones(4099, device=self.device, dtype=dtype)
                        probe[1234] = bad
                        caught.append(not _all_finite([probe], self.device))
                clean = _all_finite([torch.ones(4099, device=self.device, dtype=torch.bfloat16)], self.device)
                self._fast_finite = all(caught) and clean
            except Exception as exc:
                logger.warning('Strict finite self-test raised %s; keeping the per-tensor check', exc)
                self._fast_finite = False
            logger.info('Strict finite check: %s', 'reductions over ZeRO flat buffers (self-test passed)'
                        if self._fast_finite else 'per-tensor (self-test failed)')
        return self._fast_finite

    def _pop_timing(self) -> Dict[str, float]:
        timing = dict(self._timing_accum)
        timing["microsteps"] = float(self._timing_microsteps)
        self._timing_accum = {}
        self._timing_microsteps = 0
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
            "strict_verify",
            "strict_commit",
        )
        parts = [f"total={step_time:.3f}s"]
        for key in ordered_keys:
            if key in timing:
                parts.append(f"{key}={timing[key]:.3f}s")
        if timing.get("microsteps", 0.0) > 1:
            parts.append(f"microsteps={timing['microsteps']:.0f}")
        return ", ".join(parts)

    def _maybe_start_torch_profiler(self, next_step: int) -> None:
        if (
            self.torch_profile_step <= 0
            or self.rank != self.torch_profile_rank
            or self._torch_profile_completed
            or self._torch_profiler is not None
            or int(next_step) != self.torch_profile_step
        ):
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._torch_profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.torch_profile_record_shapes,
            profile_memory=False,
            with_stack=False,
            with_flops=self.torch_profile_with_flops,
        )
        self._torch_profiler.start()
        logger.info("torch.profiler started for optimizer step %s on rank %s", next_step, self.rank)

    @staticmethod
    def _profile_event_time_us(event, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                return float(value)
        return 0.0

    def _finish_torch_profiler(self, completed_step: int) -> None:
        profiler = self._torch_profiler
        if profiler is None or int(completed_step) != self.torch_profile_step:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        profiler.stop()
        self._torch_profiler = None
        self._torch_profile_completed = True

        output_dir = self.checkpoint_dir / "runtime" / "profiles"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{completed_step:06d}_rank_{self.rank:03d}"
        trace_path = output_dir / f"{stem}.trace.json"
        summary_path = output_dir / f"{stem}.summary.json"
        profiler.export_chrome_trace(str(trace_path))
        rows = []
        for event in profiler.key_averages(group_by_input_shape=False):
            rows.append({
                "op": event.key,
                "count": int(event.count),
                "self_device_time_us": self._profile_event_time_us(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_time_total_us": self._profile_event_time_us(
                    event, "device_time_total", "cuda_time_total"
                ),
                "self_cpu_time_us": self._profile_event_time_us(event, "self_cpu_time_total"),
                "flops": int(getattr(event, "flops", 0) or 0),
            })
        rows.sort(key=lambda row: (-row["self_device_time_us"], -row["self_cpu_time_us"], row["op"]))
        summary_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "complete",
            "optimizer_step": int(completed_step),
            "rank": int(self.rank),
            "record_shapes": self.torch_profile_record_shapes,
            "with_flops": self.torch_profile_with_flops,
            "trace": str(trace_path),
            "events": rows[:self.torch_profile_top_k],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("torch.profiler completed: trace=%s summary=%s", trace_path, summary_path)

    def _close_torch_profiler(self) -> None:
        """Release a probe interrupted before its optimizer boundary."""
        profiler = self._torch_profiler
        self._torch_profiler = None
        if profiler is not None:
            try:
                profiler.stop()
            except Exception as exc:
                logger.warning("Could not close interrupted torch.profiler: %s", exc)

    def save_checkpoint(self, suffix=""):
        ckpt_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        if self.accelerator.is_main_process:
            (ckpt_dir / "checkpoint_complete.json").unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        strict_binding = None
        if getattr(self.train_dataloader, "is_strict_window_loader", False):
            strict_binding = self.train_dataloader.save_state(ckpt_dir, self.global_step)
        if self.accelerator.is_main_process:
            # Publish only after every rank's optimizer and RNG files exist.
            # A supervisor can select this marker instead of an incomplete dir.
            manifest = {
                "schema": "motus-complete-checkpoint@1",
                "global_step": int(self.global_step),
                "world_size": int(self.accelerator.num_processes),
                "files": {
                    str(path.relative_to(ckpt_dir)): path.stat().st_size
                    for path in sorted(ckpt_dir.rglob("*"))
                    if path.is_file() and path.name not in (
                        "checkpoint_complete.json", ".checkpoint_complete.tmp"
                    )
                },
            }
            if strict_binding is not None:
                manifest['strict_window'] = strict_binding
            temporary = ckpt_dir / ".checkpoint_complete.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, ckpt_dir / "checkpoint_complete.json")
        self.accelerator.wait_for_everyone()
        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _resume_data_position(self) -> int:
        """Resume the inner batch sampler and synchronize its wrapped epoch."""
        if getattr(self.train_dataloader, "is_strict_window_loader", False):
            if self.train_dataloader.committed_step != self.global_step:
                raise RuntimeError('Strict consumption and trainer step differ')
            return 0  # The loader restores its exact quota-plan position itself.
        samplers = []
        owners = set()
        for candidate in _iter_sampler_chain(self.train_dataloader):
            resume_fn = getattr(candidate, "resume", None)
            if not callable(resume_fn):
                continue
            owner = getattr(resume_fn, "__self__", candidate)
            if id(owner) not in owners:
                samplers.append((owner, resume_fn))
                owners.add(id(owner))
        if not samplers:
            dataset_cfg = getattr(getattr(self, "config", None), "dataset", None)
            dataset_type = str(getattr(dataset_cfg, "type", "") or "").lower()
            weighted_pretrain = (
                dataset_type in {"pretrain", "pretrain_sc"}
                and bool(getattr(dataset_cfg, "source_weighted_sampling", False))
            )
            if weighted_pretrain:
                raise RuntimeError("Weighted pretrain resume requires a resumable batch sampler")
            # Preserve legacy resume behavior for ordinary robot/random loaders.
            return 0
        if len(samplers) > 1:
            raise RuntimeError(
                "Cannot resume data stream: expected exactly one resumable "
                f"batch sampler behind loader wrappers, found {len(samplers)}"
            )
        sampler, resume_fn = samplers[0]
        try:
            epoch = int(resume_fn(self.global_step, int(self.accelerator.num_processes)))
            # DataLoaderShard.__iter__ calls set_epoch(iteration).  Align that
            # outer iteration as well, so it cannot replace the restored epoch.
            _maybe_set_epoch(self.train_dataloader, epoch)
        except Exception as exc:
            raise RuntimeError("Could not realign the data stream on resume") from exc
        logger.info(
            "Data stream resumed at step %d (epoch %d, skipped %d batches in this epoch)",
            self.global_step, epoch, getattr(sampler, "_skip_batches", 0),
        )
        return epoch


    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        complete = Path(checkpoint_path) / "checkpoint_complete.json"
        if complete.exists():
            manifest = json.loads(complete.read_text(encoding="utf-8"))
            for name, size in manifest["files"].items():
                path = Path(checkpoint_path) / name
                if not path.is_file() or path.stat().st_size != size:
                    raise RuntimeError(f"Incomplete checkpoint file: {path}")
        step_match = re.search(r'step_(\d+)', Path(checkpoint_path).name)
        if step_match:
            self.global_step = int(step_match.group(1))
        self.accelerator.load_state(checkpoint_path)
        if getattr(self.train_dataloader, "is_strict_window_loader", False):
            receipt = self.train_dataloader.load_state(checkpoint_path, self.global_step,
                                                       saved_source_root=self.config.training.get('strict_saved_source_root', None))
            logger.info('Strict data stream restored: %s', receipt)

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
                base_lrs = getattr(scheduler, "base_lrs", [scheduler.base_lr])
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

            # Relative wrist subtraction/rotation must happen before BF16
            # quantization; the model casts the resulting action tokens.
            actions = batch['actions'].to(self.device, dtype=torch.float32)
            states = batch.get('states')
            if states is not None:
                states = states.to(self.device, dtype=torch.float32)
            action_valid_mask = batch.get('action_valid_mask')
            state_valid_mask = batch.get('state_valid_mask')
            if action_valid_mask is not None:
                action_valid_mask = action_valid_mask.to(self.device, dtype=torch.bool)
            if state_valid_mask is not None:
                state_valid_mask = state_valid_mask.to(self.device, dtype=torch.bool)
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
                values = values.to(self.device, dtype=self.dtype)
            sample_class = batch.get(SAMPLE_CLASS_KEY)
            if sample_class is not None:
                sample_class = sample_class.to(self.device, dtype=self.dtype)
            if self._timing_active:
                self._add_timing(timing, "h2d", self._timing_elapsed(t0))

            # Forward
            t0 = self._timing_start() if self._timing_active else None
            debug_step = self._debug_step_enabled(self.global_step + 1)
            if debug_step:
                logger.info("[debug-step %s R%s] forward begin", self.global_step + 1, self.rank)
            wrist_stats_source = batch.get("wrist_stats_source")
            if debug_step and wrist_stats_source is not None:
                logger.info("[debug-step %s R%s] wrist stats sources=%s",
                            self.global_step + 1, self.rank, wrist_stats_source)
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
                action_valid_mask=action_valid_mask,
                state_valid_mask=state_valid_mask,
                wrist_stats_source=wrist_stats_source,
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
            # Accelerate/DeepSpeed updates inside backward, and the scheduler
            # below prepares the next update. Capture the LR actually used.
            update_lrs = [group['lr'] for group in self.optimizer.param_groups]
            t0 = self._timing_start() if self._timing_active else None
            if debug_step:
                logger.info("[debug-step %s R%s] backward begin", self.global_step + 1, self.rank)
            self.accelerator.backward(total_loss)
            if debug_step:
                logger.info("[debug-step %s R%s] backward end", self.global_step + 1, self.rank)
            if self._timing_active:
                self._add_timing(timing, "backward", self._timing_elapsed(t0))

            # DeepSpeed already clips and updates inside accelerator.backward.
            # Keep the original clipping path for non-DeepSpeed training only.
            if self.accelerator.sync_gradients:
                did_step = True
                self._last_update_lrs = update_lrs
                if not callable(getattr(self.model, "zero_optimization_stage", None)):
                    grad_clip = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                    t0 = self._timing_start() if self._timing_active else None
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
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

        # Keep reporting off the critical microstep path without retaining its
        # autograd graph. Host scalars are materialized at the logging boundary.
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}, did_step, timing

    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500):
        if resume_from:
            self.load_checkpoint(resume_from)
        if getattr(self.train_dataloader, "is_strict_window_loader", False):
            if self.global_step != self.train_dataloader.committed_step:
                raise RuntimeError('Strict migration requires the configured complete checkpoint')

        logger.info(f"Starting Motus2 training for {max_steps} steps")
        start_time = time.time()
        epoch = self._resume_data_position() if resume_from else 0
        data_iter = iter(self.train_dataloader)

        while self.global_step < max_steps:
            next_step = self.global_step + 1
            debug_step = self._debug_step_enabled(next_step)
            self._timing_active = self.profile_timing and ((self.global_step + 1) % self.timing_every == 0)
            self._maybe_start_torch_profiler(next_step)
            data_t0 = self._timing_start() if self._timing_active else None
            if debug_step:
                logger.info("[debug-step %s R%s] fetching batch", next_step, self.rank)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                _maybe_set_epoch(self.train_dataloader, epoch)
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
            strict_windows = getattr(self.train_dataloader, 'is_strict_window_loader', False)
            engine_step_before = getattr(self.model, 'global_steps', None) if strict_windows else None
            skipped_before = getattr(self.model, 'skipped_steps', None) if strict_windows else None
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
            if strict_windows:
                finite = bool(did_step)
                try:
                    finite = finite and bool(torch.isfinite(metrics['total_loss']).all().item())
                    if engine_step_before is not None:
                        finite = finite and self.model.global_steps == engine_step_before + 1
                    if skipped_before is not None:
                        finite = finite and self.model.skipped_steps == skipped_before
                    if int(self.config.training.get('strict_optimizer_finite_interval', 1)) != 1:
                        raise ValueError('Strict migration requires optimizer finite checks on every update')
                    engine_optimizer = getattr(self.model, 'optimizer', None)
                    inner = getattr(engine_optimizer, 'optimizer', self.optimizer)
                    verify_t0 = time.perf_counter()
                    if self._fast_finite_check():
                        finite = finite and _all_finite(_strict_update_tensors(self.model, inner), self.device)
                    else:
                        tensors = [p.data for p in self.model.parameters()]
                        for group in inner.param_groups:
                            tensors.extend(p.data for p in group['params'])
                        for state in inner.state.values():
                            tensors.extend(v for v in state.values() if torch.is_tensor(v))
                        if tensors:
                            checks = [torch.isfinite(t).all().to(self.device) for t in tensors]
                            finite = finite and bool(torch.stack(checks).all().item())
                    if self._timing_active:
                        self._timing_accum['strict_verify'] = (self._timing_accum.get('strict_verify', 0.0)
                                                               + time.perf_counter() - verify_t0)
                except Exception as exc:
                    finite = False
                    logger.error('Strict update verification failed on rank %s: %s', self.rank, exc)
                commit_t0 = time.perf_counter()
                agreed = False
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    # Every rank must have verified its update before the step
                    # is committed; one int all-reduce replaces an object gather.
                    flag = torch.tensor(1 if finite else 0, device=self.device, dtype=torch.int32)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    finite, agreed = bool(flag.item()), True
                self.train_dataloader.commit_after_update(next_step, succeeded=finite, agreed=agreed)
                if self._timing_active:
                    # The whole commit: waiting for the next batch's background
                    # preparation, the cross-rank agreement and the ledger update.
                    self._timing_accum['strict_commit'] = (self._timing_accum.get('strict_commit', 0.0)
                                                           + time.perf_counter() - commit_t0)
            if not did_step:
                continue
            step_time = time.time() - self._step_start
            self._step_start = time.time()
            self.global_step += 1
            timing_for_step = self._pop_timing() if self._timing_active else {}

            # Logging (all-reduce metrics for global average across ranks)
            if self.global_step % self.log_interval == 0:
                t0 = self._timing_start() if self._timing_active else None
                metrics = reduce_training_metrics(metrics, self.device, dist)
                if self._timing_active:
                    timing_for_step["metrics_reduce"] = self._timing_elapsed(t0)
                if self.rank == 0:
                    lrs = self._last_update_lrs
                    lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
                    loss_parts = f"video: {metrics['video_loss']:.4f}, action: {metrics['action_loss']:.4f}"
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
            self._timing_active = False
            self._finish_torch_profiler(self.global_step)

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
    resolved_mode = resolve_action_mode_from_config(m, config.dataset)
    action_mode = resolved_mode.action_mode
    wrist_xyz_normalization = resolved_mode.wrist_xyz_normalization
    wrist_xyz_per_dataset = mixed_pretrain_wrist_per_dataset_enabled(config)
    stats_paths = wrist_action_stats_paths(config)
    assert_stats_share_action_space(stats_paths, allow_untyped=wrist_xyz_per_dataset)
    wrist_bounds_by_source = None
    if wrist_xyz_per_dataset:
        wrist_bounds_by_source = build_wrist_bounds_by_source(config, wrist_xyz_normalization)
        by_source = wrist_stats_paths_by_source(config)
        logger.info(
            "Per-dataset wrist xyz: sources=%s stats_files=%s",
            sorted(by_source),
            {k: by_source[k] for k in sorted(by_source)},
        )
        for source, bounds in wrist_bounds_by_source.items():
            logger.info("Wrist xyz source=%s mode=%s stats=%s bounds=%s",
                        source, wrist_xyz_normalization, by_source[source], bounds)
        default_stats = wrist_bounds_by_source["_default"]
        qpos20_wrist_xyz_min, qpos20_wrist_xyz_max = default_stats["min"], default_stats["max"]
    else:
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
    if wrist_xyz_normalization == "mean_std" and not wrist_xyz_per_dataset:
        qpos20_wrist_xyz_mean, qpos20_wrist_xyz_std, wrist_xyz_std_min = (
            qpos20_wrist_xyz_mean_std_from_stats(stats_paths)
        )
    elif wrist_xyz_normalization == "mean_std" and wrist_bounds_by_source:
        default_stats = wrist_bounds_by_source["_default"]
        qpos20_wrist_xyz_mean = default_stats.get("mean", qpos20_wrist_xyz_mean)
        qpos20_wrist_xyz_std = default_stats.get("std", qpos20_wrist_xyz_std)

    if action_mode in {"joint", "rel_joint", "abs_eef_rpy"}:
        action_dim, action_state_dim = 14, 14
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
        flex_attention_static_shapes=m.get(
            "flex_attention_static_shapes", False
        ),
        training_compact_time_embeddings=m.get(
            "training_compact_time_embeddings", False
        ),
        training_skip_unused_time_projections=m.get(
            "training_skip_unused_time_projections", False
        ),
        training_cache_block_masks=m.get("training_cache_block_masks", False),
        training_blackwell_flex_backward=m.get("training_blackwell_flex_backward", False),
        training_fuse_qk_rope=m.get("training_fuse_qk_rope", False),
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
        wrist_bounds_by_source=wrist_bounds_by_source,
        num_inference_timesteps=m.inference.num_inference_timesteps,
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
        value_target=m.get("value", {}).get("target", "progress"),
        value_loss_weight=m.get("value", {}).get("value_loss_weight", 10.0),
        vm_ratio=m.get("value", {}).get("vm_ratio", 0.0) if m.get("value", {}).get("enabled", False) else 0.0,
        progress_c=m.get("value", {}).get("progress_c", 0.01),
        suboptimal_vm_share=m.get("value", {}).get("suboptimal_vm_share", None),
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
    strict_windows = bool(config.training.get('strict_window_sampling', False))
    if strict_windows and int(config.training.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('Strict window protocol currently requires GA1')
    train_ds = create_dataset(config, val=False)
    co_train_ds = create_co_datasets(config, val=False)
    eval_scaling_cfg = config.get("eval_scaling", {})
    val_ds = None
    if bool(config.get("system", {}).get("eval_enabled", True)):
        # PretrainDataset's val flag fixes sampling; it does not split the input
        # JSONL. Give sample/demo evaluation its own held-out corpus when set.
        sample_dataset_cfg = None
        sample_overrides = config.get("eval_sample", {}).get("dataset", None)
        if sample_overrides is not None:
            sample_overrides = _plain_omegaconf_value(sample_overrides)
            if not isinstance(sample_overrides, dict):
                raise ValueError("eval_sample.dataset must be a mapping when provided")
            sample_dataset_cfg = OmegaConf.create(
                OmegaConf.to_container(config.dataset, resolve=True)
            )
            for key, value in sample_overrides.items():
                sample_dataset_cfg[key] = value
            _isolate_eval_index_cache(
                config.dataset, sample_dataset_cfg, sample_overrides, "eval_sample.dataset"
            )
        val_ds = create_dataset(config, val=True, dataset_cfg=sample_dataset_cfg)
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
        _isolate_eval_index_cache(
            config.dataset, scaling_val_dataset_cfg,
            _plain_omegaconf_value(eval_scaling_cfg.get("dataset", {})) or {},
            "eval_scaling.dataset",
        )
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
    source_weighted_sampling = bool(config.dataset.get("source_weighted_sampling", False))
    has_sample_weights = hasattr(train_ds, "get_sample_weight")
    weighted_expand_sampling = source_weighted_sampling and has_sample_weights
    # Set to false to reproduce runs before 2026-09, where source ratios were
    # normalized inside each resolution bucket instead of globally.
    weighted_bucket_sampling = (
        bool(config.training.get("sample_with_replacement", False))
        and has_sample_weights
        and not weighted_expand_sampling
    )
    variable_bucket_batches = (
        bucket_by_temporal or weighted_bucket_sampling or weighted_expand_sampling
    )

    def _make_eval_loader(dataset, batch_size: int, seed_offset: int):
        if dataset is None:
            return None
        eval_loader_kwargs = dict(loader_kwargs)
        eval_workers = int(config.get('eval_sample', {}).get('num_workers', nw))
        eval_loader_kwargs.update(num_workers=eval_workers, persistent_workers=eval_workers > 0)
        if eval_workers == 0:
            eval_loader_kwargs.pop('prefetch_factor', None)
        seed = int(config.training.get("seed", 0)) + seed_offset
        if bucket_by_resolution:
            sampler = SameResolutionBatchSampler(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                seed=seed,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                **eval_loader_kwargs,
            )
        # accelerator.prepare_data_loader() shards this loader by rank
        # (BatchSamplerShard with even_batches=False), so the ranks already cover
        # disjoint batches; adding another stride here would shard twice.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **eval_loader_kwargs,
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
        if weighted_expand_sampling:
            sampler = WeightedExpandedBatchSampler(
                train_ds,
                batch_size=config.training.batch_size,
                drop_last=True,
                seed=int(config.training.get("seed", 0)),
                window_start_sampling=bool(
                    getattr(train_ds, "window_start_sampling", False)
                ) and not strict_windows,
            )
        elif variable_bucket_batches:
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
        if weighted_expand_sampling:
            logger.info(
                "Using weighted expanded same-bucket batches: counts=%s expanded=%s "
                "batch_sizes=%s global_balance=%s",
                bucket_summary,
                dict(sorted(getattr(sampler, "expanded_counts", {}).items())),
                getattr(sampler, "batch_sizes", {}),
                True,
            )
        elif variable_bucket_batches:
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

    eval_groups = config.get('eval_sample', {}).get('groups', None)
    eval_batch_size = int(config.get('eval_sample', {}).get('batch_size', config.training.batch_size))
    if val_ds is not None and eval_groups:
        from data.evaluation_groups import evaluation_subsets
        grouped, val_loader = {}, {}
        for i, (name, spec) in enumerate(_plain_omegaconf_value(eval_groups).items()):
            group_cfg = OmegaConf.create(OmegaConf.to_container(
                sample_dataset_cfg if sample_dataset_cfg is not None else config.dataset, resolve=True))
            for key, value in spec.get('dataset', {}).items():
                group_cfg[key] = value
            group_ds = create_dataset(config, val=True, dataset_cfg=group_cfg)
            selected = evaluation_subsets(group_ds, {name: spec})[name]
            if len(selected) != len(group_ds):
                raise ValueError(f'Evaluation group contains unexpected sources: {name}')
            # Give each pool its own dataset as well as loader. Legacy loader
            # fallbacks then cannot replace a diagnostic with another pool.
            grouped[name] = group_ds
            val_loader[name] = _make_eval_loader(group_ds, eval_batch_size, 10000 + i)
            # accelerator.prepare_data_loader() gives each rank its own shard of
            # this loader, so the check is that every rank gets at least one
            # batch instead of requiring world_size * modes batches.
            if len(val_loader[name]) < 1:
                raise ValueError(f'Evaluation pool {name} yields no batch on rank {rank}; '
                                 'enlarge the fixed pool or lower eval_sample.batch_size')
        logger.info('Evaluation groups (exposure labels in eval_sample.groups): %s',
                    {name: len(subset) for name, subset in grouped.items()})
    else:
        val_loader = _make_eval_loader(val_ds, eval_batch_size, 10000)
    scaling_val_loader = _make_eval_loader(
        scaling_val_ds,
        int(eval_scaling_cfg.get("batch_size", config.training.batch_size)),
        20000,
    )
    if strict_windows:
        if co_train_ds or ti_enabled or not weighted_expand_sampling or not bucket_by_resolution:
            raise ValueError('Strict windows require the single weighted pretrain bucket planner')
        if not getattr(train_ds, 'window_start_sampling', False) or len(train_ds) != len(train_ds.episode_files):
            raise ValueError('Strict windows require one index slot per episode')
        from data.strict_window_loader import StrictWindowLoader, data_plan_fingerprint
        envelope = [None]
        if rank == 0:
            try:
                envelope[0] = {'ok': True, 'value': data_plan_fingerprint(config)}
            except Exception as exc:
                envelope[0] = {'ok': False, 'error': str(exc)}
        if world_size > 1:
            dist.broadcast_object_list(envelope, src=0)
        if not envelope[0]['ok']:
            raise RuntimeError('Cannot freeze strict data plan: ' + envelope[0]['error'])
        train_loader = StrictWindowLoader(train_ds, sampler, rank=rank, world_size=world_size,
            batch_size=int(config.training.batch_size), fingerprint=envelope[0]['value'],
            skip_fingerprint_check=not bool(config.training.get('strict_fingerprint_check', True)),
            resample_sources=config.training.get('strict_resample_sources'),
            append_sources_on_resume=config.training.get('strict_append_sources_on_resume'),
            migration_start_step=int(config.training.get('strict_migration_start_step', 0)),
            seed=int(config.training.get('seed', 42)),
            num_workers=int(config.training.get('strict_loader_workers', 4)),
            prefill_workers=int(config.training.get('strict_prefill_workers',
                config.training.get('strict_loader_workers', 4))),
            load_timeout=float(config.training.get('strict_load_timeout', 120)),
            max_retries=int(config.training.get('strict_load_retries', 2)),
            max_replacements=int(config.training.get('strict_max_replacements', 32)),
            # Rank zero's per-batch planning (discovery + exact geometry for all
            # 128 slots) runs in these processes, one batch ahead.
            plan_processes=int(config.training.get('strict_plan_processes', 16)),
            # A slot is refilled from its own source/bucket up to this many times,
            # keeping the configured source mixture; only past it does the slot
            # fall back to another source of the same bucket instead of exiting.
            same_source_replacements=int(config.training.get('strict_same_source_replacements',
                config.training.get('strict_max_replacements', 32))),
            timing_every=int(os.environ.get('MOTUS2_TIMING_EVERY', 10)),
            # The keys train_step casts to the trainer's bfloat16 on the way to the
            # device; converting them on the loader's thread makes that copy a DMA.
            device_dtypes={'latents': torch.bfloat16, 'language_embedding': torch.bfloat16,
                           'values': torch.bfloat16, SAMPLE_CLASS_KEY: torch.bfloat16},
            audit_path=Path(config.system.checkpoint_dir) / 'runtime/strict_windows.jsonl')
    return train_loader, val_loader, scaling_val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _latest_complete_checkpoint(run_dir, world_size, strict_windows, minimum_step):
    candidates = []
    for path in Path(run_dir).iterdir():
        match = re.fullmatch(r"checkpoint_step_(\d+)", path.name)
        if match and path.is_dir() and int(match[1]) >= minimum_step:
            candidates.append((int(match[1]), path))

    for step, path in sorted(candidates, reverse=True):
        try:
            manifest = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
            files = manifest["files"]
            if (
                manifest["schema"] != "motus-complete-checkpoint@1"
                or manifest["global_step"] != step
                or manifest["world_size"] != world_size
                or not files
            ):
                raise ValueError("checkpoint manifest does not match step or world size")
            if strict_windows:
                ledger = manifest["strict_window"]
                if (
                    ledger["committed_step"] != step
                    or ledger["file"] != "strict_window_state.json"
                    or not files.get(ledger["file"])
                    or not re.fullmatch(r"[0-9a-f]{64}", ledger["sha256"])
                ):
                    raise ValueError("strict sampling ledger is missing or does not match the checkpoint")
            for name, size in files.items():
                entry = path / name
                if not entry.is_file() or entry.stat().st_size != size:
                    raise ValueError(f"incomplete checkpoint file: {entry}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Skipping checkpoint %s: %s", path, exc)
            continue
        return str(path)

    raise RuntimeError(
        f"No complete checkpoint at or above step {minimum_step} for {world_size} ranks in {run_dir}"
    )


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
    ds_cfg = getattr(ds_plugin, "deepspeed_config", None) if ds_plugin is not None else None
    if ds_plugin is not None and not isinstance(ds_cfg, dict):
        raise TypeError("DeepSpeed plugin configuration must be a dictionary")
    # An explicit YAML value is authoritative, including 0 (disable clipping).
    configure_gradient_clipping(config.training, ds_cfg)
    if ds_plugin is not None:
        if isinstance(ds_cfg, dict):
            micro_bs = ds_cfg.get("train_micro_batch_size_per_gpu", None)
            grad_accum_cfg = ds_cfg.get("gradient_accumulation_steps", None)
            if micro_bs == "auto" or micro_bs is None:
                ds_cfg["train_micro_batch_size_per_gpu"] = int(config.training.batch_size)
            if grad_accum_cfg == "auto" or grad_accum_cfg is None:
                ds_cfg["gradient_accumulation_steps"] = grad_accum_steps

    grad_accum_plugin = None
    if zero_stage >= 2 and grad_accum_steps > 1:
        grad_accum_plugin = GradientAccumulationPlugin(
            num_steps=grad_accum_steps,
            sync_each_batch=True,
        )
    accelerator_kwargs = dict(
        deepspeed_plugin=ds_plugin,
        mixed_precision="bf16",
        log_with=config.logging.get('report_to', 'tensorboard'),
        project_dir=config.system.checkpoint_dir,
        project_config=ProjectConfiguration(total_limit=20),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=dist_timeout)],
        # Custom batch samplers here do not expose a fixed batch_size.
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    if grad_accum_plugin is not None:
        accelerator_kwargs["gradient_accumulation_plugin"] = grad_accum_plugin
    else:
        accelerator_kwargs["gradient_accumulation_steps"] = grad_accum_steps
    accelerator = Accelerator(**accelerator_kwargs)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.device)
    # Explicit seeds bind initialization, dataset randomness and noise sampling.
    # Initialize the random seed only when one is configured.
    if config.training.get("seed") is not None:
        effective_seed = int(config.training.seed) + rank
        set_seed(effective_seed)
        logger.info("Training RNG seed: base=%s rank=%s effective=%s",
                    config.training.seed, rank, effective_seed)
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

    if config.get("resume", {}).get("latest_complete", False) and args.resume is None:
        selection = [None]
        if rank == 0:
            try:
                baseline = config.resume.get("checkpoint_path")
                match = re.fullmatch(r"checkpoint_step_(\d+)", Path(baseline).name) if baseline else None
                if baseline and match is None:
                    raise ValueError(f"Invalid checkpoint baseline: {baseline}")
                selection[0] = {"path": _latest_complete_checkpoint(
                    config.system.checkpoint_dir,
                    world_size,
                    bool(config.training.get("strict_window_sampling", False)),
                    int(match[1]) if match else 0,
                )}
            except Exception as exc:
                selection[0] = {"error": str(exc)}
        if world_size > 1:
            dist.broadcast_object_list(selection, src=0)
        if "error" in selection[0]:
            raise RuntimeError("Cannot auto-resume: " + selection[0]["error"])
        config.resume.checkpoint_path = selection[0]["path"]
        config.model.load_pretrained_backbones = False
        if rank == 0:
            logger.info("Auto-resume selected latest complete checkpoint: %s", config.resume.checkpoint_path)

    if rank == 0:
        save_config_snapshot(args.config, config, config.system.checkpoint_dir)

    # TensorBoard
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard logging requested but `tensorboard` is not installed in this Python environment.")
        tb_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        resume_path = getattr(getattr(config, "resume", None), "checkpoint_path", None)
        resume_step = re.fullmatch(r"checkpoint_step_(\d+)", Path(resume_path).name) if resume_path else None
        purge_step = int(resume_step.group(1)) + 1 if resume_step and config.dataset.get("wrist_xyz_per_dataset") else None
        tb_writer = SummaryWriter(log_dir=tb_dir, purge_step=purge_step)
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
                    dir=config.logging.get('wandb_dir', config.system.checkpoint_dir),
                )
                wandb_enabled = True
            except Exception as e:
                logger.warning("WandB init failed; continuing without WandB. Error: %s", e)
                report_to = [x for x in report_to if x != "wandb"]

    trainer = None
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
                unwrapped.load_checkpoint_weights(finetune_path, strict=False)
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
        if isinstance(val_loader, dict):
            # Accelerate leaves a mapping unchanged; shard every real loader
            # once, in the same group order on all ranks.
            val_loader = {name: accelerator.prepare_data_loader(loader)
                          for name, loader in val_loader.items()}
        norm_guard_receipt = install_zero2_fp64_norm_guard(model)
        logger.info("DeepSpeed FP64 norm guard: %s", norm_guard_receipt)
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
        if trainer is not None:
            trainer._close_torch_profiler()
            if getattr(trainer.train_dataloader, 'is_strict_window_loader', False):
                trainer.train_dataloader.close(abort=sys.exc_info()[0] is not None)
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and wandb_enabled:
            if wandb is not None:
                wandb.finish()
        if tb_writer:
            tb_writer.close()


if __name__ == "__main__":
    main()
