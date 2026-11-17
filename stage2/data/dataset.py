# Dataset factory and collate function for latent-space AR training.

import logging
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import ListConfig, OmegaConf

from utils.action_mode import resolve_action_mode_from_config

# ======================================================================
# Dataset factory
# ======================================================================

logger = logging.getLogger(__name__)

def _resolve_dataset_cfg(config: OmegaConf, dataset_cfg=None):
    return config.dataset if dataset_cfg is None else dataset_cfg


def create_dataset(
    config: OmegaConf,
    val: bool = False,
    dataset_cfg=None,
    treat_as_irrelevant: bool = False,
):
    """Create dataset based on config type."""
    dataset_cfg = _resolve_dataset_cfg(config, dataset_cfg)
    dataset_type = dataset_cfg.get("type", "robotwin")
    resolved_mode = None
    if hasattr(config, "model"):
        resolved_mode = resolve_action_mode_from_config(config.model, dataset_cfg)

    def _common_params():
        params = {}
        if hasattr(config, "common"):
            params["max_latent_window"] = config.common.get("max_latent_window", 10)
            params["video_action_freq_ratio"] = config.common.get("video_action_freq_ratio", 6)
        if hasattr(config, "model"):
            params["action_mode"] = (
                resolved_mode.action_mode
                if resolved_mode is not None
                else config.model.get("action_mode", "joint")
            )
            params["use_state"] = config.model.get("use_state", False)
            params["bootstrap_start_window"] = int(config.model.get("fixed_chunk_size", 2)) > 1
            params["bootstrap_chunk_size"] = int(config.model.get("fixed_chunk_size", 2))
        params["val"] = val
        return params

    if dataset_type == "robotwin":
        data_format = dataset_cfg.get("data_format", "latent")
        if data_format != "latent":
            raise ValueError(
                f"Unsupported data_format={data_format}. "
                "This pipeline now uses latent-only training data."
            )
        params = _common_params()
        for key in (
            "dataset_dir", "data_mode", "task_mode", "task_name",
            "max_episodes", "randomized_limit_per_task", "cache_dir",
            "catalog_path", "image_aug", "latent_dir", "latent_phases",
            "task_sampling_weights",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        # Value prediction: pass target semantics from model config.
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
        params["treat_as_irrelevant"] = treat_as_irrelevant

        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        return RobotWinTaskDataset(**params)

    if dataset_type == "realworld":
        params = _common_params()
        for key in (
            "jsonl_path", "latent_subpath", "lang_subpath", "robot",
            "duckdb_path", "duckdb_table",
            "dataset_dir", "catalog_path", "latent_dir", "latent_phases", "latent_max_attempts",
            "task_mode", "task_name", "max_episodes", "cache_dir",
            "action_upsample_ratio", "eef_quat_order",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        from .realworld.realworld_dataset import RealWorldDataset
        return RealWorldDataset(**params)

    if dataset_type in {"pretrain", "pretrain_sc"}:
        params = _common_params()
        common_cfg = getattr(config, "common", {})
        params["bootstrap_chunk_size"] = int(getattr(config, "model", {}).get(
            "fixed_chunk_size", common_cfg.get("fixed_chunk_size", 2)
        ))
        params["bootstrap_start_window"] = params["bootstrap_chunk_size"] > 1
        if hasattr(config, "common") and "wrist_xyz_abs_max" in config.common:
            params["wrist_xyz_abs_max"] = config.common.get("wrist_xyz_abs_max", None)
        if hasattr(config, "common") and "wrist_xyz_relative_max" in config.common:
            params["wrist_xyz_relative_max"] = config.common.get("wrist_xyz_relative_max", None)
        if hasattr(config, "common") and "action_abs_max" in config.common:
            params["action_abs_max"] = config.common.get("action_abs_max", None)
        for key in (
            "jsonl_path", "latent_root", "latent_file_name", "latent_subpath",
            "latent_file_name_by_dataset",
            "lang_root", "lang_subpath", "lang_dir", "robot",
            "clickhouse", "use_index_cache",
            "dataset_dir", "episode_path_prefix", "episode_path_prefixes", "episode_index_source", "catalog_path", "latent_dir", "latent_phases", "latent_max_attempts",
            "index_cache_path",
            "metadata_cache_path",
            "task_mode", "task_name", "max_episodes", "cache_dir",
            "video_hz", "action_upsample_ratio", "action",
            "raw_action_hz", "train_action_hz",
            "empty_lang_path", "eef_quat_order",
            "dataset_include", "dataset_exclude",
            "dataset_allowlist", "dataset_denylist",
            "stereo_left_latent_name", "stereo_right_latent_name",
            "stereo_latent_name_pairs", "stereo_latent_sources_path",
            "stereo_latent_sources", "stereo_latent_name_pairs_by_dataset",
            "action_stats_path", "action_stats_by_dataset", "qpos_normalization",
            "wrist_xyz_per_dataset",
            "samples_per_episode", "window_start_sampling",
            "source_weighted_sampling",
            "subset_fraction", "subset_seed", "temporal_buckets",
            "batch_size_by_window", "sampling_weight", "latent_seconds",
            "num_batches_per_epoch", "dynamic_even_window", "min_latent_window",
            "min_total_latent_frames", "max_total_latent_frames",
            "wrist_xyz_abs_max", "wrist_xyz_relative_max", "action_abs_max",
            "refine_spatial_bucket_keys", "freeze_val_indices",
            "exclude_explicit_fail",
            "mask_invalid_hand_tracking",
            "invalid_origin_wrist_marker",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        if resolved_mode is not None and resolved_mode.hand_normalization is not None:
            params["qpos_normalization"] = resolved_mode.hand_normalization
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
        params["treat_as_irrelevant"] = treat_as_irrelevant
        if dataset_type == "pretrain_sc":
            from .pretrain_sc.pretrain_sc_dataset import PretrainSCDataset
            return PretrainSCDataset(**params)
        from .pretrain.pretrain_dataset import PretrainDataset
        return PretrainDataset(**params)

    if dataset_type in {"tianji_wuji", "tianji_wuji2"}:
        params = _common_params()
        for key in (
            "dataset_dir", "episode_index_source", "episode_path_prefix",
            "clickhouse", "use_index_cache", "index_cache_path",
            "manifest_path", "manifest_sample_type",
            "latent_root", "latent_phases", "latent_resolution", "left_latent_name",
            "right_latent_name", "view_latent_names", "single_latent_name", "language_subpath", "tjwj_subpath", "action_stats_path",
            "instruction_bank_path", "environment_mix_prob", "environment_source",
            "environment_variant", "text_pool_source",
            "task_mode", "task_name", "max_episodes",
            "cache_dir", "video_hz", "action_upsample_ratio", "action",
            "raw_action_hz", "train_action_hz",
            "include_fail", "fail_task_name", "fail_sample_fraction", "include_old",
            "suboptimal",
            "qpos_normalization",
            "action_from_state",
            "validation_fraction",
            "validation_seed",
            "task_episode_skip_first",
            "bootstrap_start_window_prob",
            "view_dropout",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
                params["task_irrelevant_decaying_curve"] = bool(
                    value_cfg.get("task_irrelevant_decaying_curve", False)
                )
        if resolved_mode is not None and resolved_mode.hand_normalization is not None:
            # Forward the canonical hand normalization to the loader.
            params["qpos_normalization"] = resolved_mode.hand_normalization
        # Prefer dataset.action_from_state if set; else accept model.action_from_state
        # (grouped with action_mode / normalization in newer configs).
        if (
            "action_from_state" not in params
            and hasattr(config, "model")
            and config.model.get("action_from_state") is not None
        ):
            params["action_from_state"] = bool(config.model.get("action_from_state"))
        params["treat_as_irrelevant"] = treat_as_irrelevant
        from .tianji_wuji.tianji_wuji_dataset import TianjiWujiDataset
        return TianjiWujiDataset(**params)

    if dataset_type == "tianji_sharpa":
        params = _common_params()
        for key in (
            "dataset_dir", "episode_index_source", "episode_path_prefix",
            "clickhouse", "use_index_cache", "index_cache_path",
            "manifest_path", "manifest_sample_type",
            "latent_root", "latent_phases", "latent_resolution", "left_latent_name",
            "right_latent_name", "view_latent_names", "single_latent_name", "language_subpath",
            "tjwj_subpath", "action_stats_path",
            "text_pool_source",
            "task_mode", "task_name", "max_episodes",
            "cache_dir", "video_hz", "action_upsample_ratio", "action",
            "raw_action_hz", "train_action_hz",
            "include_fail", "fail_task_name", "fail_sample_fraction", "include_old",
            "qpos_normalization",
            "action_from_state",
            "task_episode_skip_first",
            "bootstrap_start_window_prob",
            "view_dropout",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
        if resolved_mode is not None and resolved_mode.hand_normalization is not None:
            params["qpos_normalization"] = resolved_mode.hand_normalization
        if (
            "action_from_state" not in params
            and hasattr(config, "model")
            and config.model.get("action_from_state") is not None
        ):
            params["action_from_state"] = bool(config.model.get("action_from_state"))
        params["treat_as_irrelevant"] = treat_as_irrelevant
        from .tianji_sharpa.tianji_sharpa_dataset import TianjiSharpaDataset
        return TianjiSharpaDataset(**params)

    if dataset_type == "aloha_agilex_2":
        params = _common_params()
        for key in (
            "jsonl_path", "latent_subpath", "lang_subpath", "robot",
            "duckdb_path", "duckdb_table",
            "dataset_dir", "catalog_path", "latent_dir", "latent_phases", "latent_max_attempts",
            "task_mode", "task_name", "max_episodes", "cache_dir",
            "stat_key", "action_upsample_ratio", "eef_quat_order",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        from .aloha_agilex_2.aloha_agilex2_dataset import AlohaAgilex2Dataset
        return AlohaAgilex2Dataset(**params)

    if dataset_type == "ac_one":
        params = _common_params()
        for key in (
            "jsonl_path", "latent_subpath", "lang_subpath", "robot",
            "duckdb_path", "duckdb_table",
            "dataset_dir", "catalog_path", "latent_dir", "latent_phases", "latent_max_attempts",
            "task_mode", "task_name", "max_episodes", "cache_dir",
            "stat_key", "action_upsample_ratio", "eef_quat_order",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        from .ac_one.ac_one_dataset import ACOneDataset
        return ACOneDataset(**params)

    if dataset_type == "robotwin_shard":
        params = _common_params()
        for key in ("shards", "lru_size", "cache_dir"):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)

        from .robotwin2.robotwin_shard_dataset import RobotWinShardDataset
        return RobotWinShardDataset(**params)

    raise ValueError(f"Unknown dataset type: {dataset_type}")


# Keys a co-dataset may restate. Everything else is inherited from the main
# `dataset` block, which gives both corpora the same data configuration.
#
# `action_stats_path` is the reason this list exists: hand qpos is normalized
# inside the dataset, so two robots whose joints are calibrated differently need
# their own envelopes. `qpos_normalization` is deliberately absent — that is the
# normalization *mode*, and mixing modes across a batch would put the action
# expert's inputs on two different scales. `validation_fraction` /
# `validation_seed` are absent for the same reason the split contract exists.
CO_DATASET_OVERRIDABLE = (
    "type",
    "dataset_dir", "episode_index_source", "episode_path_prefix",
    "clickhouse", "use_index_cache", "index_cache_path", "cache_dir",
    "manifest_path", "manifest_sample_type",
    "action_stats_path",
    "task_mode", "task_name", "max_episodes", "task_episode_skip_first",
    "include_fail", "fail_task_name", "fail_sample_fraction", "include_old",
    "suboptimal",
    "latent_root", "latent_phases", "latent_resolution",
    "left_latent_name", "right_latent_name", "view_latent_names",
    "single_latent_name", "language_subpath", "tjwj_subpath",
    # A bank is written per task family, so a corpus that trains on other tasks
    # has to be able to name its own (or null it out) rather than inherit a bank
    # whose task_to_canonical cannot cover it.
    "instruction_bank_path", "environment_mix_prob", "environment_source",
    "environment_variant",
    "video_hz", "action_upsample_ratio", "action",
    "raw_action_hz", "train_action_hz",
)


def co_dataset_configs(config: OmegaConf) -> list:
    """Resolved dataset configs for the co-training corpora, main one excluded."""
    entries = config.dataset.get("co_datasets") or []
    if not isinstance(entries, (list, ListConfig)):
        raise ValueError("dataset.co_datasets must be a list of dataset overrides")

    resolved = []
    for index, entry in enumerate(entries):
        if not entry.get("enabled", True):
            continue
        name = entry.get("name") or f"co_datasets[{index}]"
        dataset_cfg = OmegaConf.create(
            deepcopy(OmegaConf.to_container(config.dataset, resolve=False))
        )
        dataset_cfg.pop("co_datasets", None)
        # A co-dataset is a policy corpus in its own right; the main block's
        # fail-class routing is per-corpus and is restated, not inherited.
        dataset_cfg.pop("task_irrelevant", None)
        for key in CO_DATASET_OVERRIDABLE:
            if key in entry:
                dataset_cfg[key] = entry.get(key)
        if "dataset_dir" in entry:
            dataset_cfg["jsonl_path"] = None
            dataset_cfg["clickhouse"] = None
        unknown = [
            key
            for key in entry
            if key not in CO_DATASET_OVERRIDABLE
            and key not in {"name", "enabled", "sampling_ratio"}
        ]
        if unknown:
            raise ValueError(
                f"{name}: co-dataset cannot override {sorted(unknown)}. "
                f"Overridable keys: {sorted(CO_DATASET_OVERRIDABLE)}"
            )
        resolved.append((name, dataset_cfg, entry.get("sampling_ratio", None)))
    return resolved


def action_stats_paths(config: OmegaConf) -> list:
    """Every stats file whose data reaches the action expert, main one first.

    Hand qpos is normalized per dataset, so the corpora may legitimately
    disagree there. The wrist envelope is a single model parameter, so the
    caller has to cover all of them at once.
    """
    paths = []
    main_path = config.dataset.get("action_stats_path", None)
    if main_path:
        paths.append(str(main_path))
    for _name, dataset_cfg, _ratio in co_dataset_configs(config):
        co_path = dataset_cfg.get("action_stats_path", None)
        if co_path and str(co_path) not in paths:
            paths.append(str(co_path))
    return paths


def wrist_action_stats_paths(config: OmegaConf) -> list:
    """Unique stats files referenced for wrist relative xyz (not a bounds union).

    Mixed pretrain resolves wrist stats per source via
    ```utils.co_training.wrist_stats_paths_by_source``` (human/SC
    ```action_stats_path```, HandsOn/wuji* ```action_stats_by_dataset```).
    Legacy co-training without per-source tables still unions
    ```action_stats_path``` with ```wrist_action_stats_supplement_paths```.
    """
    from utils.co_training import mixed_pretrain_wrist_per_dataset_enabled, wrist_stats_paths_by_source

    if mixed_pretrain_wrist_per_dataset_enabled(config):
        return sorted(set(wrist_stats_paths_by_source(config).values()))

    explicit = config.dataset.get("wrist_action_stats_paths", None)
    if explicit is not None:
        if isinstance(explicit, str):
            explicit = [explicit]
        return [str(p) for p in explicit if p]

    supplement_only = bool(config.dataset.get("wrist_action_stats_supplement_only", False))
    extra = config.dataset.get("wrist_action_stats_supplement_paths", None) or []
    if isinstance(extra, str):
        extra = [extra]
    if supplement_only:
        return [str(path_value) for path_value in extra if path_value]

    paths = action_stats_paths(config)
    for path_value in extra:
        path_str = str(path_value)
        if path_str and path_str not in paths:
            paths.append(path_str)
    return paths


def create_co_datasets(config: OmegaConf, val: bool = False) -> list:
    """Build the co-training corpora declared in `dataset.co_datasets`.

    Each one is a full policy dataset carrying its own `action_stats_path`, so
    hand qpos is normalized against the envelope of the robot that produced it.
    Wrist xyz uses per-source stats when ```action_stats_by_dataset``` is set; otherwise
    the model uses one envelope from ```wrist_action_stats_paths```.
    """
    datasets = []
    for name, dataset_cfg, _ratio in co_dataset_configs(config):
        dataset = create_dataset(config, val=val, dataset_cfg=dataset_cfg)
        logger.info(
            "Co-dataset %s: %d episodes-worth of samples, action_stats_path=%s",
            name,
            len(dataset),
            dataset_cfg.get("action_stats_path"),
        )
        datasets.append(dataset)
    return datasets


def create_task_irrelevant_dataset(config: OmegaConf, val: bool = False):
    """Create the optional task-irrelevant dataset with the robotwin latent loader."""
    ti_cfg = config.dataset.get("task_irrelevant")
    if not ti_cfg or not ti_cfg.get("enabled", False):
        return None

    dataset_cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(config.dataset, resolve=False)))
    # These targets only apply to fail-class records mixed into the main dataset.
    dataset_cfg.pop("fail_sample_fraction", None)
    dataset_cfg.pop("suboptimal", None)
    dataset_cfg.type = ti_cfg.get("type", dataset_cfg.get("type", "robotwin"))

    for key in (
        "dataset_dir", "jsonl_path", "clickhouse", "use_index_cache",
        "catalog_path", "cache_dir", "latent_dir", "latent_phases",
        "task_mode", "task_name", "max_episodes", "latent_root", "latent_file_name", "latent_subpath",
        "episode_index_source", "episode_path_prefix", "episode_path_prefixes",
        "latent_file_name_by_dataset",
        "lang_root", "lang_subpath", "lang_dir", "robot", "empty_lang_path",
        "instruction_bank_path", "environment_mix_prob", "environment_source",
        "environment_variant", "text_pool_source",
        "eef_quat_order", "index_cache_path", "dataset_include", "dataset_exclude",
        "dataset_allowlist", "dataset_denylist", "task_sampling_weights",
        "stereo_latent_name_pairs", "stereo_latent_sources_path",
        "stereo_latent_sources", "stereo_latent_name_pairs_by_dataset",
        "include_fail", "include_old",
        "manifest_path", "manifest_sample_type",
        "view_dropout",
    ):
        if key in ti_cfg:
            dataset_cfg[key] = ti_cfg.get(key)
    if "dataset_dir" in ti_cfg:
        dataset_cfg["jsonl_path"] = None
        dataset_cfg["clickhouse"] = None
    elif "jsonl_path" in ti_cfg:
        dataset_cfg["dataset_dir"] = None
        dataset_cfg["clickhouse"] = None
    elif "clickhouse" in ti_cfg:
        dataset_cfg["dataset_dir"] = None
        dataset_cfg["jsonl_path"] = None
    if dataset_cfg.get("type", "robotwin") == "robotwin":
        dataset_cfg.data_format = "latent"
        if "data_mode" in ti_cfg:
            dataset_cfg.data_mode = ti_cfg.get("data_mode", "randomized")
    if "index_cache_path" not in ti_cfg and dataset_cfg.get("index_cache_path", None):
        base_cache = Path(str(dataset_cfg.get("index_cache_path")))
        dataset_cfg["index_cache_path"] = str(
            base_cache.with_name(f"{base_cache.stem}_task_irrelevant{base_cache.suffix}")
        )

    try:
        return create_dataset(
            config,
            val=val,
            dataset_cfg=dataset_cfg,
            treat_as_irrelevant=True,
        )
    except RuntimeError as exc:
        empty_markers = ("No Tianji/Wuji episodes found", "No Tianji/Sharpa episodes found")
        if bool(ti_cfg.get("allow_empty", False)) and any(m in str(exc) for m in empty_markers):
            logger.warning("Task-irrelevant dataset is empty and allow_empty=true; disabling it for this run")
            return None
        raise


# ======================================================================
# Collate function
# ======================================================================

def _process_language_embeddings_batch(language_embeddings, text_len: int = 512):
    """Process and batch language embeddings with padding (T5 token padding)."""
    padded_embeddings = []
    for emb in language_embeddings:
        if emb.ndim == 3:
            emb = emb[0]
        if emb.ndim == 1:
            # Pooled vector [D] -> treat as single-token [1, D]
            emb = emb.unsqueeze(0)
        if emb.ndim != 2:
            raise ValueError(f"language_embedding must be 2D [T, D], got shape={tuple(emb.shape)}")
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    return torch.stack(padded_embeddings, dim=0)


def ar_collate_fn(batch):
    """Collate function for sliding-window AR training.

    Handles:
    - language_embedding: padded to fixed text_len=512
    - valid_len: kept as int tensor
    - Other tensors: stacked (all have consistent shapes due to max_latent_window padding)
    """
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    keys = batch[0].keys()
    collated = {}

    for k in keys:
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            if k == "language_embedding":
                collated[k] = _process_language_embeddings_batch(vals, text_len=512)
            else:
                collated[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals)
        else:
            collated[k] = vals

    return collated
