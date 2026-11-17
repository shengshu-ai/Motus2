import pytest
import torch
from data.pretrain.pretrain_dataset import PretrainDataset


def fixture():
    obj = PretrainDataset.__new__(PretrainDataset)
    obj.wrist_xyz_relative_max = 5.0
    obj.actions_per_latent = 2
    obj.bootstrap_chunk_size = 2
    actions = torch.zeros(12, 58)
    states = torch.zeros(6, 58)
    # A far-away world origin should not affect relative movement.
    actions[:, 20:23] = torch.tensor([100., -20., 1.])
    actions[:, 49:52] = torch.tensor([-80., 10., 2.])
    states[:, 20:23] = actions[0, 20:23]
    states[:, 49:52] = actions[0, 49:52]
    return obj, actions, states


def test_relative_threshold_translation_and_chunk_anchors():
    obj, actions, states = fixture()
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6)
    # Last chunk has a different absolute anchor; match it rather than chunk 1.
    states[4, 20] += 20
    actions[8:12, 20] += 20
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6)
    actions[9, 20:23] += torch.tensor([3., 4., 0.])
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6)
    actions[9, 22] += 0.1
    assert not obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6)


def test_invalid_hands_and_nonfinite():
    obj, actions, states = fixture()
    target_mask = torch.ones(12, 2, dtype=torch.bool)
    anchor_mask = torch.ones(6, 2, dtype=torch.bool)
    actions[4, 20] += 100
    target_mask[4, 0] = False
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6, target_mask, anchor_mask)
    target_mask[4, 0] = True
    assert not obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6, target_mask, anchor_mask)
    anchor_mask[2, 0] = False
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6, target_mask, anchor_mask)
    actions[4, 20] = float('nan')
    assert not obj._actions_pass_wrist_xyz_relative_filter(actions, states, 6, target_mask, anchor_mask)


def test_padding_excluded_and_missing_anchor_rejected():
    obj, actions, states = fixture()
    actions[10, 20] += 100
    assert obj._actions_pass_wrist_xyz_relative_filter(actions, states, 5)
    with pytest.raises(ValueError):
        obj._actions_pass_wrist_xyz_relative_filter(actions, None, 6)
