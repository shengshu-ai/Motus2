# Dataset factory and collate function for latent-space AR training.

from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.action_mode import resolve_action_mode_from_config

# ======================================================================
# Dataset factory
# ======================================================================

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

    def _common_params():
        params = {}
        if hasattr(config, "common"):
            params["max_latent_window"] = config.common.get("max_latent_window", 10)
            params["video_action_freq_ratio"] = config.common.get("video_action_freq_ratio", 6)
        if hasattr(config, "model"):
            params["action_mode"] = config.model.get("action_mode", "joint")
            params["use_state"] = config.model.get("use_state", False)
            # The current model/deployment cold-start protocol conditions on one
            # real C0 and predicts the first chunk immediately after it. Legacy
            # L0 replication is opt-in only.
            params["bootstrap_start_window"] = bool(
                config.model.get("bootstrap_start_window", False)
            )
            params["bootstrap_chunk_size"] = int(config.model.get("fixed_chunk_size", 2))
        params["val"] = val
        return params

    if dataset_type == "robodojo":
        if config.model.get("action_mode", "joint") != "joint":
            raise ValueError("RoboDojo currently requires absolute joint actions")
        if treat_as_irrelevant or config.model.get("value", {}).get("enabled", False):
            raise ValueError("RoboDojo preparation does not define value/irrelevant targets")
        from .robodojo.dataset import RoboDojoDataset
        return RoboDojoDataset(
            manifest_path=dataset_cfg.manifest_path,
            normalization=dataset_cfg.get("normalization"),
            val=val,
            task_name=dataset_cfg.get("task_name"),
            max_latent_window=config.common.get("max_latent_window", 14),
            video_action_freq_ratio=config.common.get("video_action_freq_ratio", 5),
            use_state=config.model.get("use_state", True),
            sequence_mode=dataset_cfg.get("sequence_mode", "episode"),
            video_height=config.common.get("video_height"),
            video_width=config.common.get("video_width"),
            bootstrap_chunk_size=int(config.model.get("fixed_chunk_size", 2)),
            bootstrap_start_window=bool(config.model.get("bootstrap_start_window", False)),

        )

    if dataset_type == "robotwin":
        data_format = dataset_cfg.get("data_format", "latent")
        if data_format != "latent":
            raise ValueError(
                f"Unsupported data_format={data_format}. "
                "This pipeline now uses latent-only training data."
            )
        params = _common_params()
        # Hybrid-memory prefix-prefill: extend the fixed window with leading
        # warm-up frames (loss-masked downstream). 0 → in-window memory only.
        if hasattr(config, "model") and hasattr(config.model, "memory"):
            params["prefix_frames"] = int(config.model.memory.get("prefix_frames", 0))
        for key in (
            "dataset_dir", "data_mode", "task_mode", "task_name",
            "max_episodes", "randomized_limit_per_task",
            "validation_fraction", "validation_seed", "cache_dir",
            "catalog_path", "image_aug", "latent_dir",
            "language_embedding_dir", "language_embedding_policy", "latent_phases",
                "task_sampling_weights", "sequence_mode", "sequence_max_latents",
                "sequence_overflow_policy", "sequence_phase_policy",
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

        if hasattr(config, "training"):
            lb = config.training.get("length_bucketing", None)
            if lb:
                latent_budget = lb.get("latent_budget", lb.get("token_budget", None))
                if latent_budget is not None:
                    params["latent_budget"] = latent_budget
                params["over_budget_policy"] = lb.get("over_budget_policy", "allow_single")

        # Deterministic length index (episode, phase) -> T_lat. Required for length
        # bucketing. Auto-enable for the train split when bucketing is on, unless the
        # dataset config sets `length_index` explicitly. Never built for the val split
        # or the task-irrelevant aux dataset (neither buckets).
        bucketing_on = False
        if hasattr(config, "training"):
            lb = config.training.get("length_bucketing", None)
            bucketing_on = bool(lb and lb.get("enabled", False))
        length_index = dataset_cfg.get("length_index", None)
        if length_index is None:
            length_index = bucketing_on
        params["length_index"] = bool(length_index) and (not val) and (not treat_as_irrelevant)
        if "length_index_cache" in dataset_cfg:
            params["length_index_cache"] = dataset_cfg.get("length_index_cache")

        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        return RobotWinTaskDataset(**params)

    if dataset_type == "tianji_wuji":
        params = _common_params()
        # This loader has no prefix-prefill path. Accepting the key and dropping
        # it would train on a shorter context than the config asks for.
        if hasattr(config, "model") and hasattr(config.model, "memory"):
            prefix_frames = int(config.model.memory.get("prefix_frames", 0))
            if prefix_frames != 0:
                raise ValueError(
                    "dataset type 'tianji_wuji' does not implement "
                    f"memory.prefix_frames (got {prefix_frames}); set it to 0"
                )
        for key in (
            "dataset_dir", "episode_index_source", "episode_path_prefix", "clickhouse",
            "use_index_cache", "index_cache_path", "manifest_path", "manifest_sample_type",
            "latent_root", "latent_phases", "latent_resolution",
            "single_latent_name", "view_latent_names", "view_merge",
            "language_subpath", "tjwj_subpath",
            "action_stats_path", "qpos_normalization", "action", "action_from_state",
            "action_upsample_ratio", "video_hz",
            "tactile", "tactile_enabled", "tactile_stats_path", "tactile_baseline_dir",
            "tactile_baseline_name", "tactile_baseline_kind", "tactile_baseline_polarity",
            "tactile_baseline_per_source", "task_episode_skip_first", "view_dropout",
            "task_mode", "task_name", "max_episodes", "include_fail", "include_old",
            "cache_dir",
            "sequence_mode", "sequence_max_latents", "sequence_overflow_policy",
            "mask_invalid_replay",
        ):
            if key in dataset_cfg:
                value = dataset_cfg.get(key)
                # The loader is plain Python and indexes these with `in` / `.get`,
                # which behave differently on OmegaConf containers.
                if OmegaConf.is_config(value):
                    value = OmegaConf.to_container(value, resolve=True)
                params[key] = value
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
        if hasattr(config, "model"):
            resolved_mode = resolve_action_mode_from_config(config.model, dataset_cfg)
            if resolved_mode.hand_normalization is not None:
                # Forward the canonical hand normalization to the loader.
                params["qpos_normalization"] = resolved_mode.hand_normalization
        params["treat_as_irrelevant"] = treat_as_irrelevant

        if hasattr(config, "training"):
            lb = config.training.get("length_bucketing", None)
            if lb:
                latent_budget = lb.get("latent_budget", lb.get("token_budget", None))
                if latent_budget is not None:
                    params["latent_budget"] = latent_budget
                params["over_budget_policy"] = lb.get("over_budget_policy", "allow_single")
        # Scanning every latent file for its length takes a while, so the train
        # split caches it. The val split and the aux dataset never bucket.
        if "length_index_cache" in dataset_cfg and not val and not treat_as_irrelevant:
            params["length_index_cache"] = dataset_cfg.get("length_index_cache")

        from .tianji_wuji.tianji_wuji_dataset import TianjiWujiDataset
        return TianjiWujiDataset(**params)

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

    if dataset_type == "pretrain":
        params = _common_params()
        for key in (
            "jsonl_path", "latent_root", "latent_file_name", "latent_subpath",
            "lang_root", "lang_subpath", "lang_dir", "robot",
            "duckdb_path", "duckdb_table",
            "dataset_dir", "episode_index_source", "catalog_path", "latent_dir", "latent_phases", "latent_max_attempts",
            "index_cache_path",
            "metadata_cache_path",
            "task_mode", "task_name", "max_episodes", "cache_dir",
            "action_upsample_ratio", "empty_lang_path", "eef_quat_order",
            "dataset_include", "dataset_exclude",
            "dataset_allowlist", "dataset_denylist",
        ):
            if key in dataset_cfg:
                params[key] = dataset_cfg.get(key)
        if hasattr(config, "model") and hasattr(config.model, "value"):
            value_cfg = config.model.value
            if value_cfg.get("enabled", False):
                params["value_target"] = value_cfg.get("target", "progress")
                params["progress_c"] = value_cfg.get("progress_c", 0.01)
        params["treat_as_irrelevant"] = treat_as_irrelevant
        from .pretrain.pretrain_dataset import PretrainDataset
        return PretrainDataset(**params)

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


def create_task_irrelevant_dataset(config: OmegaConf, val: bool = False):
    """Create the optional task-irrelevant dataset with the robotwin latent loader."""
    ti_cfg = config.dataset.get("task_irrelevant")
    if not ti_cfg or not ti_cfg.get("enabled", False):
        return None

    dataset_cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(config.dataset, resolve=False)))
    dataset_cfg.type = ti_cfg.get("type", dataset_cfg.get("type", "robotwin"))

    for key in (
        "dataset_dir", "jsonl_path", "duckdb_path", "duckdb_table",
        "catalog_path", "cache_dir", "latent_dir", "latent_phases",
        "task_mode", "task_name", "max_episodes", "latent_root", "latent_file_name", "latent_subpath",
        "lang_root", "lang_subpath", "lang_dir", "robot", "empty_lang_path",
        "eef_quat_order", "index_cache_path", "dataset_include", "dataset_exclude",
        "dataset_allowlist", "dataset_denylist", "task_sampling_weights",
    ):
        if key in ti_cfg:
            dataset_cfg[key] = ti_cfg.get(key)
    if "dataset_dir" in ti_cfg:
        dataset_cfg["jsonl_path"] = None
        dataset_cfg["duckdb_path"] = None
    elif "jsonl_path" in ti_cfg:
        dataset_cfg["dataset_dir"] = None
        dataset_cfg["duckdb_path"] = None
    elif "duckdb_path" in ti_cfg:
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

    return create_dataset(
        config,
        val=val,
        dataset_cfg=dataset_cfg,
        treat_as_irrelevant=True,
    )


# ======================================================================
# Collate function
# ======================================================================

def _process_language_embeddings_batch(language_embeddings, text_len: int = 512):
    """Process and batch language embeddings with padding (T5 token padding)."""
    padded_embeddings = []
    for emb in language_embeddings:
        if emb.ndim == 3:
            emb = emb[0]
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    return torch.stack(padded_embeddings, dim=0)


def ar_collate_fn(batch):
    """Collate full-trajectory varlen samples without padding.

    The model consumes the per-sample lists and builds a packed attention sequence
    from the original trajectory lengths.
    """
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    keys = batch[0].keys()
    collated = {"packed_varlen": True}
    for k in keys:
        vals = [s[k] for s in batch]
        if k == "language_embedding":
            collated[k] = _process_language_embeddings_batch(vals, text_len=512)
        elif isinstance(vals[0], torch.Tensor):
            collated[k] = vals
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals)
        else:
            collated[k] = vals
    return collated


def materialize_varlen_eval_batch(batch):
    """Turn a packed-varlen, one-sample list batch into regular batched tensors."""
    if not batch.get("packed_varlen", False):
        return batch

    latents = batch.get("latents")
    if not isinstance(latents, list) or len(latents) != 1:
        size = len(latents) if isinstance(latents, list) else "unknown"
        raise ValueError(
            "Varlen evaluation requires DataLoader(batch_size=1); "
            f"received {size} samples"
        )

    materialized = {}
    for key, value in batch.items():
        if key == "packed_varlen":
            continue
        if isinstance(value, list):
            item = value[0]
            materialized[key] = item.unsqueeze(0) if torch.is_tensor(item) else [item]
        else:
            # language_embedding and scalar metadata are already collated with
            # a leading batch dimension by ar_collate_fn.
            materialized[key] = value
    return materialized
