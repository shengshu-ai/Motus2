"""
Action normalization functions for robot datasets.
Provides unified [0, 1] range normalization.
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_actions(actions: torch.Tensor, action_min: np.ndarray, action_max: np.ndarray) -> torch.Tensor:
    """
    Normalize actions to [0, 1] range using provided statistics.
    Formula: (action - min) / (max - min)

    Args:
        actions: Action tensor to normalize
        action_min: Minimum values for each action dimension
        action_max: Maximum values for each action dimension

    Returns:
        Normalized action tensor in [0, 1] range
    """
    if action_min is None or action_max is None:
        raise ValueError("Normalization stats (action_min/action_max) are None - cannot normalize actions")

    # Convert to numpy for calculation
    actions_np = actions.numpy() if isinstance(actions, torch.Tensor) else actions

    # Normalize to [0, 1] range: (action - min) / (max - min)
    action_range = action_max - action_min
    # Avoid division by zero
    action_range = np.where(action_range == 0, 1.0, action_range)

    normalized = (actions_np - action_min) / action_range

    return torch.from_numpy(normalized).float()


def denormalize_actions(normalized_actions: torch.Tensor, action_min: np.ndarray, action_max: np.ndarray) -> torch.Tensor:
    """
    Denormalize actions from [0, 1] back to original scale.
    Formula: normalized * (max - min) + min

    Args:
        normalized_actions: Normalized action tensor in [0, 1] range
        action_min: Minimum values for each action dimension
        action_max: Maximum values for each action dimension

    Returns:
        Denormalized action tensor in original scale
    """
    if action_min is None or action_max is None:
        raise ValueError("Normalization stats (action_min/action_max) are None - cannot denormalize actions")

    # Convert to numpy for calculation
    norm_actions_np = normalized_actions.numpy() if isinstance(normalized_actions, torch.Tensor) else normalized_actions

    # Denormalize from [0, 1] back to original range: norm * (max - min) + min
    action_range = action_max - action_min
    denormalized = norm_actions_np * action_range + action_min

    return torch.from_numpy(denormalized).float()


def load_normalization_stats(stats_path: str, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load normalization statistics from file for specified dataset.

    Args:
        stats_path: Path to normalization statistics file
        dataset_name: Name of dataset to load stats for (e.g., 'ac_one', 'aloha_agilex_1')

    Returns:
        Tuple of (action_min, action_max) arrays, or (None, None) if loading fails
    """
    try:
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Get stats for specific dataset
        if dataset_name in stats:
            dataset_stats = stats[dataset_name]
            action_min = np.array(dataset_stats['min'], dtype=np.float32)
            action_max = np.array(dataset_stats['max'], dtype=np.float32)
        else:
            raise KeyError(f"Dataset '{dataset_name}' not found in normalization stats file")

        logger.info(f"Loaded normalization stats for {dataset_name} from {stats_path}")
        logger.info(f"  Action min: {action_min}")
        logger.info(f"  Action max: {action_max}")
        logger.info(f"  Action range: {action_max - action_min}")

        return action_min, action_max

    except FileNotFoundError:
        logger.warning(f"Normalization stats file not found: {stats_path}")
        return None, None


def load_quantile_stats(
    stats_path: str,
    dataset_name: str,
    lower_key: str = 'q01',
    upper_key: str = 'q99',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load lower/upper quantile arrays for the specified dataset from a JSON file.

    Expected JSON structure example:
        {
          "latent_action": {"q01": [...], "q99": [...], ...}
        }

    Args:
        stats_path: Path to the quantile stats JSON file
        dataset_name: Dataset name (e.g., 'latent_action')
        lower_key: Lower quantile field name (default 'q01')
        upper_key: Upper quantile field name (default 'q99')

    Returns:
        Tuple (q_low, q_high) as np.ndarray; (None, None) on failure
    """
    try:
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        if dataset_name not in stats:
            raise KeyError(f"Dataset '{dataset_name}' not found in stats file")
        dataset_stats = stats[dataset_name]

        if lower_key not in dataset_stats or upper_key not in dataset_stats:
            raise KeyError(
                f"Keys '{lower_key}'/'{upper_key}' not found in dataset '{dataset_name}' stats"
            )

        q_low = np.array(dataset_stats[lower_key], dtype=np.float32)
        q_high = np.array(dataset_stats[upper_key], dtype=np.float32)

        logger.info(
            f"Loaded quantile stats for {dataset_name} from {stats_path} ({lower_key}/{upper_key})"
        )
        return q_low, q_high

    except FileNotFoundError:
        logger.warning(f"Quantile stats file not found: {stats_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading quantile stats from {stats_path}: {e}")
        return None, None


def normalize_actions_with_quantiles(
    actions: torch.Tensor,
    q_low: np.ndarray,
    q_high: np.ndarray,
    *,
    clip: bool = True,
) -> torch.Tensor:
    """
    Normalize to [0, 1] using quantiles: optional clipping to [q_low, q_high] then linear scaling.

    Formulas:
      1) if clip=True, \( x_{clip} = \min(\max(x, q_{01}), q_{99}) \)
      2) \( x_{norm} = (x_{clip} - q_{01}) / (q_{99} - q_{01}) \)

    Args:
        actions: Action tensor [*, D]
        q_low: Lower quantile array (e.g., q01), shape [D]
        q_high: Upper quantile array (e.g., q99), shape [D]
        clip: Whether to clip actions to [q_low, q_high] before scaling

    Returns:
        Tensor normalized to [0, 1]
    """
    if q_low is None or q_high is None:
        raise ValueError("Quantile stats (q_low/q_high) are None - cannot normalize actions")

    actions_np = actions.numpy() if isinstance(actions, torch.Tensor) else actions

    # Avoid divide-by-zero
    q_range = q_high - q_low
    q_range = np.where(q_range == 0, 1.0, q_range)

    x = actions_np
    if clip:
        x = np.clip(x, q_low, q_high)

    normalized = (x - q_low) / q_range
    return torch.from_numpy(normalized).float()

