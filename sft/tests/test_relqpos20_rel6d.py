import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.relative_eef import (
    abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
    abs_qpos20_wrist9_to_relqpos20_rel6d_torch,
    rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch,
    relqpos20_rel6d_to_abs_qpos20_wrist9_torch,
)


def _make_state_and_actions():
    torch.manual_seed(0)
    state = torch.randn(2, 58)
    action = torch.randn(2, 12, 58)
    identity_rot6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
    state[:, 23:29] = identity_rot6d
    state[:, 52:58] = identity_rot6d
    action[..., 23:29] = identity_rot6d
    action[..., 52:58] = identity_rot6d
    return state, action


def test_qpos20_rel6d_roundtrip_keeps_absolute_hand_qpos():
    state, action = _make_state_and_actions()

    rel = abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(state, action)
    recovered = rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch(state, rel)

    assert torch.allclose(rel[..., :20], action[..., :20])
    assert torch.allclose(rel[..., 29:49], action[..., 29:49])
    assert torch.allclose(recovered[..., :23], action[..., :23], atol=1e-5)
    assert torch.allclose(recovered[..., 29:52], action[..., 29:52], atol=1e-5)


def test_relqpos20_rel6d_roundtrip_uses_hand_qpos_delta():
    state, action = _make_state_and_actions()

    rel = abs_qpos20_wrist9_to_relqpos20_rel6d_torch(state, action)
    recovered = relqpos20_rel6d_to_abs_qpos20_wrist9_torch(state, rel)

    assert torch.allclose(rel[..., :20], action[..., :20] - state[:, None, :20])
    assert torch.allclose(rel[..., 29:49], action[..., 29:49] - state[:, None, 29:49])
    assert torch.allclose(recovered[..., :23], action[..., :23], atol=1e-5)
    assert torch.allclose(recovered[..., 29:52], action[..., 29:52], atol=1e-5)
