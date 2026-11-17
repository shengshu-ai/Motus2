"""Tests for native video/action phases and millimeter-scale relative targets."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.motus import Motus
from models.rope import rope_params, rope_video_basis_freqs
from utils.relative_eef import abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch


class Geometry:
    _prepare_noisy_action_blocks = Motus._prepare_noisy_action_blocks
    _slice_noisy_actions_and_states = Motus._slice_noisy_actions_and_states
    _chunk_latent_counts = Motus._chunk_latent_counts
    _scale_qpos20_rel_wrist_xyz = staticmethod(lambda x: x)

    def __init__(self, state):
        self.config = SimpleNamespace(action_mode='qpos20_rel6d', use_state=state,
                                      actions_per_latent=2)


@pytest.mark.parametrize('state', [False, True])
def test_relative_actions_keep_millimeter_motion_until_token_cast(state):
    model = Geometry(state)
    states = torch.zeros(1, 8, 58)
    states[..., 20:23] = 1.
    states[..., 49:52] = -1.
    for part in (slice(23, 29), slice(52, 58)):
        states[..., part] = torch.tensor([1., 0., 0., 0., 1., 0.])
    actions = states.repeat_interleave(2, 1)
    actions[..., 20] += .001
    actions[..., 49] -= .001
    result = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.bfloat16)
    expected = torch.cat([abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(
        states[:, start], actions[:, start * 2:(start + 2) * 2])
        for start in (2, 4, 6)], 1).bfloat16()
    torch.testing.assert_close(result['flat_actions'], expected, rtol=0, atol=0)
    assert (result['flat_actions'][..., 20] > 0).all()
    assert result['states'] is None if not state else result['states'].dtype == torch.bfloat16
    per_frame, _ = model._slice_noisy_actions_and_states(actions, states, 2, dtype=torch.bfloat16)
    assert (per_frame[..., 20] > 0).all()
    reference = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        under_amp = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.float32)
        per_frame_amp, _ = model._slice_noisy_actions_and_states(actions, states, 2, dtype=torch.float32)
    torch.testing.assert_close(under_amp['flat_actions'], reference['flat_actions'], rtol=0, atol=0)
    torch.testing.assert_close(per_frame_amp, reference['flat_actions'], rtol=0, atol=0)


def test_shared_action_clock_uses_native_temporal_spectrum_at_fractional_positions():
    dim = 128
    temporal_channels = dim // 2 - 2 * (dim // 2 // 3)
    positions = torch.tensor([0., .25, 1., 2.5], dtype=torch.float64)
    actual = rope_video_basis_freqs(positions, dim)
    base = rope_params(4, temporal_channels * 2)
    expected = torch.polar(torch.ones(4, temporal_channels, dtype=torch.float64),
                           positions[:, None] * torch.angle(base[1])[None])
    torch.testing.assert_close(actual[:, 0, :temporal_channels], expected)
    assert (actual[:, 0, temporal_channels:] == 1).all()


def test_training_and_packed_preparation_keep_absolute_geometry_fp32():
    root = Path(__file__).resolve().parents[1]
    # Check the actual ingress sites, not only the helper's numerical behavior.
    train = (root / 'train/train.py').read_text()
    assert "actions = _to_device(batch['actions'], torch.float32)" in train
    assert "states = _to_device(batch.get('states'), torch.float32)" in train
    tree = ast.parse((root / 'models/varlen_sample_preprocessor.py').read_text())
    assignments = [(target.id, ast.unparse(node.value))
                   for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)]
    assert any(name == 'act_i' and 'dtype=torch.float32' in value for name, value in assignments)
    assert any(name == 'state_i' and 'dtype=torch.float32' in value for name, value in assignments)


def test_model_clones_native_wan_axes_without_frequency_switch():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'models/motus.py').read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == 'freqs' for t in node.targets)
                      and 'clone' in ast.unparse(node.value))
    wan = torch.randn(10, 64, dtype=torch.complex128)
    model = SimpleNamespace(video_model=SimpleNamespace(wan_model=SimpleNamespace(freqs=wan)))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'native-init', 'exec'), {'self': model})
    assert torch.equal(model.freqs, wan)
    assert model.freqs.data_ptr() != wan.data_ptr()
