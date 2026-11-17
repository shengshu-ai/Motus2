"""Public wrist min-max endpoints, clipping, inverse and untouched channels."""
from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.motus import Motus
from utils.action_mode import resolve_action_mode_from_config

@pytest.mark.parametrize("shape", [(2, 58), (2, 3, 4, 58)])
def test_public_min_max_endpoints_and_inverse(shape):
    model = Motus.__new__(Motus)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(action_mode="qpos20_rel6d", wrist_xyz_normalization="min_max",
        qpos20_wrist_xyz_min=[-2.] * 6, qpos20_wrist_xyz_max=[4.] * 6)
    raw = torch.randn(shape, dtype=torch.float32)
    raw[..., 20:23] = -2.
    raw[..., 49:52] = 4.
    got = model._scale_qpos20_rel_wrist_xyz(raw)
    assert torch.equal(got[..., 20:23], torch.zeros_like(got[..., 20:23]))
    assert torch.allclose(got[..., 49:52], torch.ones_like(got[..., 49:52]))
    assert torch.allclose(model._unscale_qpos20_rel_wrist_xyz(got), raw)
    keep = list(range(20)) + list(range(23, 49)) + list(range(52, 58))
    assert torch.equal(got[..., keep], raw[..., keep])
    raw[..., 20:23] = -20.
    raw[..., 49:52] = 40.
    clipped = model._scale_qpos20_rel_wrist_xyz(raw)
    assert clipped[..., 20:23].min() == 0
    assert clipped[..., 49:52].max() == 1

def test_public_defaults_and_single_mode_name():
    resolved = resolve_action_mode_from_config({"action_mode": "qpos20_rel6d"}, {})
    assert resolved.hand_normalization == resolved.wrist_xyz_normalization == "min_max"
    with pytest.raises(ValueError):
        resolve_action_mode_from_config({"action_mode": "qpos20_rel6d", "normalization": {"wrist_xyz": "min_max_0_1"}}, {})
