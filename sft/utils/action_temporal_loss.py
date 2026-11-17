"""Lightweight temporal action-loss utilities shared by training and tests."""
from typing import List

import math
import torch


def validate_action_d2_loss_weight(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"action_d2_loss_weight must be finite and non-negative, got {value}"
        )
    return value


def masked_chunk_second_difference_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    block_token_lengths: List[int],
) -> torch.Tensor:
    """Match temporal second differences without crossing chunk boundaries."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target actions must share [B,T,D]")
    if token_mask.shape != predicted.shape[:2]:
        raise ValueError("action token mask must be [B,T]")
    if sum(block_token_lengths) != predicted.shape[1]:
        raise ValueError("block token lengths do not cover the action sequence")
    per_sample_sum = predicted.new_zeros(predicted.shape[0], dtype=torch.float32)
    per_sample_count = predicted.new_zeros(predicted.shape[0], dtype=torch.float32)
    offset = 0
    for length in block_token_lengths:
        if length >= 3:
            block_predicted = predicted[:, offset:offset + length].float()
            block_target = target[:, offset:offset + length].float()
            predicted_d2 = (
                block_predicted[:, 2:]
                - 2 * block_predicted[:, 1:-1]
                + block_predicted[:, :-2]
            )
            target_d2 = (
                block_target[:, 2:]
                - 2 * block_target[:, 1:-1]
                + block_target[:, :-2]
            )
            triple_mask = (
                token_mask[:, offset:offset + length - 2]
                * token_mask[:, offset + 1:offset + length - 1]
                * token_mask[:, offset + 2:offset + length]
            ).float()
            per_triple = (predicted_d2 - target_d2).square().mean(dim=2)
            per_sample_sum = per_sample_sum + (per_triple * triple_mask).sum(dim=1)
            per_sample_count = per_sample_count + triple_mask.sum(dim=1)
        offset += length
    return (per_sample_sum / per_sample_count.clamp(min=1)).mean()
