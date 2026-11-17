"""CPU regressions for pose precision and missing-hand supervision.

Load the actual small production functions/statement region without importing
WAN, constructing a model, or initializing CUDA on a login/CPU test process.
"""
import ast
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.action_mode import is_rel_hand_qpos_mode, qpos_family_hand_dim, qpos_family_layout
from utils.relative_eef import (
    abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
    abs_qpos20_wrist9_to_relqpos20_rel6d_torch,
)


def _definition(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    return next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)


def _function(node, namespace=None):
    namespace = dict(globals(), **(namespace or {}))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, '<production-cpu-function>', 'exec'), namespace)
    return namespace[node.name]


MODEL = ROOT / 'models/motus.py'
TRAIN = ROOT / 'train/train.py'
masked_losses = _function(_definition(MODEL, '_masked_qpos_action_losses'))


class ActionBlocks:
    _prepare_noisy_action_blocks = _function(_definition(MODEL, '_prepare_noisy_action_blocks', 'Motus'))
    _chunk_latent_counts = _function(_definition(MODEL, '_chunk_latent_counts', 'Motus'))
    _qpos_hand_dim = 20
    _qpos_layout = qpos_family_layout(20)

    def __init__(self, apl=2, use_state=False):
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            actions_per_latent=apl, action_mode='qpos20_rel6d', use_state=use_state,
            uniform_loss_weight_warmup_steps=0, hand_qpos_loss_weight=1.,
            rel_eef_position_loss_weight=1., rel_eef_rotation_loss_weight=1.,
            rel_eef_gripper_loss_weight=1.,
        )

    @staticmethod
    def _scale_qpos20_rel_wrist_xyz(value, **kwargs):
        # Unit [-1,1] affine bounds leave the geometric conversion unchanged.
        return value


def _poses(batch=1, frames=6, apl=2):
    states = torch.zeros(batch, frames, 58)
    states[..., 20:23] = 1.
    states[..., 49:52] = 1.
    states[..., 23:29] = torch.tensor([1., 0., 0., 0., 1., 0.])
    states[..., 52:58] = torch.tensor([1., 0., 0., 0., 1., 0.])
    actions = states.repeat_interleave(apl, dim=1).clone()
    return actions, states


def _production_action_loss():
    step = _definition(MODEL, 'training_step', 'Motus')
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    begin = next(i for i, node in enumerate(step.body) if assignment(node, 'loss_mask_a'))
    end = next(i for i, node in enumerate(step.body) if assignment(node, 'video_loss_weight'))
    signature = ast.parse('def loss(self, action_pred, target_action, noisy_action_hand_valid=None, noisy_valid_len=None, rtc_committed_mask=None): pass').body[0]
    prefix = ast.parse('''
B = action_pred.shape[0]
device = action_pred.device
apl = self.config.actions_per_latent
N = action_pred.shape[1] // apl
wm_idx = []
vm_idx = []
off_policy_pending = torch.zeros(B, dtype=torch.bool, device=device)
global_step = 20000
''').body
    suffix = ast.parse('return dict(loss=action_loss, hand=action_grip_raw_loss, position=action_pos_raw_loss, rotation=action_rot_raw_loss, raw=action_raw_loss)').body
    signature.body = prefix + step.body[begin:end] + suffix
    return _function(signature, {
        '_masked_qpos_action_losses': masked_losses,
        '_is_eef_component_mode': lambda mode: qpos_family_hand_dim(mode) is not None,
    })


production_action_loss = _production_action_loss()


def test_absolute_pose_keeps_small_motion_until_relative_conversion():
    model = ActionBlocks()
    actions, states = _poses()
    actions[:, 4:, 20] += .001
    result = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.bfloat16)
    # BF16(1.001)-BF16(1.0) was exactly zero; delta-first retains the motion.
    expected = torch.tensor(.001, dtype=torch.float32).bfloat16()
    assert torch.all(result['flat_actions'][..., 20] == expected)
    assert torch.all(result['flat_actions'][..., 20] > 0)
    old = model._prepare_noisy_action_blocks(actions.bfloat16(), states.bfloat16(), 2, dtype=torch.bfloat16)
    assert torch.count_nonzero(old['flat_actions'][..., 20]) == 0
    assert result['flat_action_valid_mask'] is None


def test_no_mask_bf16_inputs_preserve_block_results():
    model = ActionBlocks(use_state=True)
    actions, states = _poses()
    actions[:, 4:, 20:23] += .125
    actions, states = actions.bfloat16(), states.bfloat16()
    got = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.bfloat16)
    expected = torch.cat([
        abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(states[:, start].float(), actions[:, start * 2:(start + 2) * 2].float()).bfloat16()
        for start in (2, 4)
    ], dim=1)
    assert torch.equal(got['flat_actions'], expected)
    assert got['states'].dtype == torch.bfloat16
    assert got['flat_action_valid_mask'] is None


def test_formal_bootstrap_keeps_first_predicted_chunk_pose_precision():
    common = ROOT / 'utils/common.py'
    prepend = _function(_definition(common, '_prepend_first_valid_block'))
    bootstrap = _function(_definition(common, 'maybe_bootstrap_start_window_sample'), {
        '_prepend_first_valid_block': prepend,
    })
    actions, states = _poses(frames=8, apl=24)
    # Real start-window convention moves raw latent1 behind the duplicated L0.
    actions[:, 24:48, 20] += .001
    _, sampled_actions, sampled_states, _, valid_len = bootstrap(
        torch.zeros(48, 8, 1, 1), actions[0], 8, 0,
        actions_per_latent=24, states=states[0], chunk_size=2,
        enabled=True, prob=1.,
    )
    assert valid_len == 8
    assert torch.equal(sampled_actions[48:72], actions[0, 24:48])
    got = ActionBlocks(apl=24)._prepare_noisy_action_blocks(
        sampled_actions.unsqueeze(0), sampled_states.unsqueeze(0), 2,
        dtype=torch.bfloat16,
    )
    assert torch.all(got['flat_actions'][:, :24, 20] == torch.tensor(.001).bfloat16())
    assert torch.count_nonzero(got['flat_actions'][:, 24:, 20]) == 0


def test_target_and_chunk_anchor_masks_are_combined_per_hand():
    model = ActionBlocks()
    actions, states = _poses()
    targets = torch.ones(1, 12, 2, dtype=torch.bool)
    anchors = torch.ones(1, 6, 2, dtype=torch.bool)
    targets[:, 4, 1] = False
    anchors[:, 2, 0] = False
    got = model._prepare_noisy_action_blocks(actions, states, 2, action_valid_mask=targets, state_valid_mask=anchors)
    expected = torch.ones(1, 8, 2, dtype=torch.bool)
    expected[:, :4, 0] = False
    expected[:, 0, 1] = False
    assert torch.equal(got['flat_action_valid_mask'], expected)
    assert got['block_action_token_lengths'] == [4, 4]


def test_partial_last_chunk_uses_its_own_anchor():
    model = ActionBlocks()
    actions, states = _poses(frames=5)
    targets = torch.ones(1, 10, 2, dtype=torch.bool)
    anchors = torch.ones(1, 5, 2, dtype=torch.bool)
    anchors[:, 4, 1] = False
    got = model._prepare_noisy_action_blocks(actions, states, 2, action_valid_mask=targets, state_valid_mask=anchors)
    assert got['block_action_token_lengths'] == [4, 2]
    assert got['flat_action_valid_mask'][:, :4].all()
    assert not got['flat_action_valid_mask'][:, 4:, 1].any()
    assert got['flat_action_valid_mask'][:, 4:, 0].all()


def test_missing_companion_mask_fails_closed():
    model = ActionBlocks()
    actions, states = _poses()
    with pytest.raises(ValueError, match='supplied together'):
        model._prepare_noisy_action_blocks(actions, states, 2, action_valid_mask=torch.ones(1, 12, 2, dtype=torch.bool))


def test_loss_excludes_missing_targets_and_renormalizes_visible_hands():
    pred = torch.ones(1, 4, 58, requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    valid[:, 0, 0] = False
    target[:, 0, :29] = 1000.
    result = production_action_loss(ActionBlocks(), pred, target, valid)
    assert result['loss'].item() == 3.
    result['loss'].backward()
    assert torch.count_nonzero(pred.grad[:, 0, :29]) == 0
    assert torch.count_nonzero(pred.grad[:, 1:, :29]) > 0
    assert torch.count_nonzero(pred.grad[:, :, 29:]) > 0


def test_one_hand_fully_missing_gives_finite_zero_gradient_for_that_hand():
    pred = torch.ones(1, 4, 58)
    pred[:, :, :29] = float('nan')
    pred.requires_grad_()
    target = torch.zeros_like(pred)
    valid = torch.ones(1, 4, 2, dtype=torch.bool)
    valid[:, :, 0] = False
    result = production_action_loss(ActionBlocks(), pred, target, valid)
    assert all(torch.isfinite(value) for value in result.values())
    assert result['loss'].item() == 3.
    result['loss'].backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[:, :, :29]) == 0


def test_no_hands_valid_gives_differentiable_zero():
    pred = torch.randn(2, 4, 58, requires_grad=True)
    valid = torch.zeros(2, 4, 2, dtype=torch.bool)
    result = production_action_loss(ActionBlocks(), pred, torch.zeros_like(pred), valid)
    assert all(value.item() == 0. for value in result.values())
    result['loss'].backward()
    assert torch.count_nonzero(pred.grad) == 0


def test_hand_mask_respects_temporal_padding_and_committed_actions():
    pred = torch.ones(1, 4, 58, requires_grad=True)
    truth = torch.zeros_like(pred)
    truth[:, 1:] = 100.
    # Only the first latent (2 tokens) is real; token1 is an RTC prefix.
    result = production_action_loss(ActionBlocks(), pred, truth, torch.ones(1, 4, 2, dtype=torch.bool), torch.tensor([1]), torch.tensor([[0., 1., 0., 0.]]))
    assert result['loss'].item() == 3.
    result['loss'].backward()
    assert torch.count_nonzero(pred.grad[:, 1:]) == 0


def test_no_mask_loss_retains_old_group_and_token_arithmetic():
    generator = torch.Generator().manual_seed(20261030)
    pred = torch.randn(2, 4, 58, generator=generator)
    target = torch.randn(2, 4, 58, generator=generator)
    got = production_action_loss(ActionBlocks(), pred, target, noisy_valid_len=torch.tensor([2, 1]))
    layout = qpos_family_layout(20)
    components = []
    token_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])
    for left, right in ((layout.left_qpos, layout.right_qpos), (layout.left_xyz, layout.right_xyz), (layout.left_rot6d, layout.right_rot6d)):
        per_token = torch.cat([F.mse_loss(pred[..., left], target[..., left], reduction='none'), F.mse_loss(pred[..., right], target[..., right], reduction='none')], dim=-1).mean(-1)
        components.append(per_token)
    expected = ((components[0] + components[1] + components[2]) * token_mask).sum(1) / token_mask.sum(1)
    assert torch.equal(got['loss'], expected.mean())


def test_train_entry_preserves_fp32_pose_and_transmits_masks():
    observed = {}
    class Model:
        def train(self):
            pass
        def training_step(self, **kwargs):
            observed.update(kwargs)
            return {'total_loss': torch.tensor(1., requires_grad=True)}
    accelerator = SimpleNamespace(accumulate=lambda _: contextlib.nullcontext(), backward=lambda loss: loss.backward(), sync_gradients=False)
    trainer = SimpleNamespace(model=Model(), accelerator=accelerator, device=torch.device('cpu'), dtype=torch.bfloat16, _timing_active=False, global_step=0, _debug_step_enabled=lambda _: False, optimizer=SimpleNamespace(param_groups=[{'lr': 2e-5}], step=lambda: None, zero_grad=lambda: None), scheduler=None)
    actions, states = _poses()
    batch = dict(latents=torch.zeros(1, 1, 6, 1, 1), actions=actions, states=states, action_valid_mask=torch.ones(1, 12, 2, dtype=torch.bool), state_valid_mask=torch.ones(1, 6, 2, dtype=torch.bool))
    train_step = _function(_definition(TRAIN, 'train_step', 'Trainer'), {'SAMPLE_CLASS_KEY': 'sample_class'})
    train_step(trainer, batch)
    assert observed['actions'].dtype == observed['states'].dtype == torch.float32
    assert observed['action_valid_mask'].dtype == observed['state_valid_mask'].dtype == torch.bool
    assert observed['latents'].dtype == torch.bfloat16


def test_validation_skip_counts_are_summed_while_errors_are_count_weighted():
    class Dist:
        ReduceOp = SimpleNamespace(SUM='sum')
        is_available = is_initialized = staticmethod(lambda: True)
        get_world_size = staticmethod(lambda: 2)
        @staticmethod
        def all_gather_object(output, local):
            output[:] = [local, (['metric', 'skipped'], ['skipped'])]
        @staticmethod
        def all_reduce(value, op):
            value += torch.tensor([30., 3.]) if value.numel() == 2 else 4.
    aggregate = _function(_definition(TRAIN, '_aggregate_validation_metrics', 'Trainer'), {'dist': Dist})
    result = aggregate(SimpleNamespace(device=torch.device('cpu'), rank=0), {'metric': 2., '_count/metric': 1, 'skipped': 1, '_sum_keys': ['skipped']})
    assert result == {'metric': 8., 'skipped': 5.}
