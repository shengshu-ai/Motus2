# Aloha-Agilex2 dataset — qpos mode uses full [0,1] normalization; EEF mode
# only normalizes the two gripper channels.

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch

from data.realworld.realworld_dataset import RealWorldDataset
from data.utils.norm import normalize_actions, load_normalization_stats

logger = logging.getLogger(__name__)


class AlohaAgilex2Dataset(RealWorldDataset):
    """Aloha-Agilex2 with qpos normalization and EEF gripper-only normalization.

    Inherits episode loading from RealWorldDataset (JSONL or dataset_dir).
    Only overrides ``_transform_sample()`` for robot-specific normalization.
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        robot: Optional[str] = "aloha2",
        stat_key: str = "aloha_agilex_2",
        **kwargs,
    ):
        super().__init__(jsonl_path=jsonl_path, robot=robot, **kwargs)

        stat_path = Path(__file__).parent.parent / "utils" / "stat.json"
        self.action_min, self.action_max = load_normalization_stats(
            str(stat_path), stat_key
        )
        self.left_gripper_min = float(self.action_min[6])
        self.left_gripper_max = float(self.action_max[6])
        self.right_gripper_min = float(self.action_min[13])
        self.right_gripper_max = float(self.action_max[13])
        logger.info(
            f"AlohaAgilex2Dataset: loaded stats (key={stat_key}), "
            f"{self.total_episodes} episodes"
        )

    @staticmethod
    def _normalize_column(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        denom = hi - lo
        if denom <= 0:
            return values
        return ((values - lo) / denom).clamp_(0.0, 1.0)

    def _normalize_eef_gripper(self, values: torch.Tensor) -> torch.Tensor:
        out = values.clone()
        if out.shape[-1] == 20:
            left_idx, right_idx = 9, 19
        elif out.shape[-1] == 16:
            left_idx, right_idx = 7, 15
        else:
            raise ValueError(f"Expected 16 or 20 dims for EEF, got {out.shape[-1]}")

        out[..., left_idx] = self._normalize_column(
            out[..., left_idx], self.left_gripper_min, self.left_gripper_max
        )
        out[..., right_idx] = self._normalize_column(
            out[..., right_idx], self.right_gripper_min, self.right_gripper_max
        )
        return out

    def _transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "eef" in self.action_mode:
            sample["actions"] = self._normalize_eef_gripper(sample["actions"])
        else:
            sample["actions"] = normalize_actions(
                sample["actions"], self.action_min, self.action_max
            )
        if "states" in sample:
            if "eef" in self.action_mode:
                sample["states"] = self._normalize_eef_gripper(sample["states"])
            else:
                sample["states"] = normalize_actions(
                    sample["states"], self.action_min, self.action_max
                )
        return sample

if __name__ == "__main__":
    jsonl_path = "/path/to/real_world_data_all_processed.jsonl"
    dataset = AlohaAgilex2Dataset(jsonl_path=jsonl_path)
    import pdb; pdb.set_trace()
    print(1)
    # latents actions language_embedding valid_len latent_start states
