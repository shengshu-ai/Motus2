"""Tests for checkpoint initialization, geometry precision and metric weighting."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 - AST function annotations

import numpy as np  # noqa: F401 - extracted function globals
import pytest
import torch
import torch.nn.functional as F  # noqa: F401 - extracted function globals
from utils.action_mode import is_rel_hand_qpos_mode, qpos_family_hand_dim, qpos_family_layout  # noqa: F401
from utils.relative_eef import (abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
                                abs_qpos20_wrist9_to_relqpos20_rel6d_torch)  # noqa: F401
from utils.scheduler import LambdaConstantScheduler, LambdaLinearScheduler

ROOT = Path(__file__).parents[1]


def extract(path, name, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(globals())
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


class Geometry:
    _qpos_hand_dim = 20
    _qpos_layout = qpos_family_layout(20)
    _prepare_noisy_action_blocks = extract('models/motus.py', '_prepare_noisy_action_blocks', 'Motus')
    _slice_noisy_actions_and_states = extract('models/motus.py', '_slice_noisy_actions_and_states', 'Motus')
    _chunk_latent_counts = extract('models/motus.py', '_chunk_latent_counts', 'Motus')
    _scale_qpos20_rel_wrist_xyz = staticmethod(lambda x: x)

    def __init__(self, use_state=False):
        self.config = SimpleNamespace(action_mode='qpos20_rel6d', use_state=use_state, actions_per_latent=24)


@pytest.mark.parametrize('use_state', [False, True])
def test_millimeter_motion_and_chunk_start_anchors_survive_bf16_tokens(use_state):
    model = Geometry(use_state)
    states = torch.zeros(1, 8, 58)
    states[..., 20:23] = 1.
    states[..., 49:52] = -1.
    for part in (slice(23, 29), slice(52, 58)):
        states[..., part] = torch.tensor([1., 0., 0., 0., 1., 0.])
    actions = states.repeat_interleave(24, 1)
    # Distinct starts: the second latent in a block must use the first anchor.
    states[:, 3, 20] += .2
    actions[:, 48:, 20] += .001
    actions[:, 48:, 49] -= .001
    expected = torch.cat([abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(
        states[:, j], actions[:, j * 24:(j + 2) * 24]) for j in (2, 4, 6)], 1).bfloat16()
    actual = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.bfloat16)
    torch.testing.assert_close(actual['flat_actions'], expected, rtol=0, atol=0)
    assert (actual['flat_actions'][..., 20] > 0).all()
    assert actual['states'] is None if not use_state else actual['states'].dtype == torch.bfloat16
    # The older per-latent helper also must delay rounding.
    per_latent, token_states = model._slice_noisy_actions_and_states(actions, states, 2, dtype=torch.bfloat16)
    assert (per_latent[:, :24, 20] > 0).all()
    assert token_states is None if not use_state else token_states.dtype == torch.bfloat16


@pytest.mark.parametrize('kind', [LambdaLinearScheduler, LambdaConstantScheduler])
def test_warmup_first_update_and_resume(kind):
    def make():
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=.1)
        kwargs = dict(warm_up_steps=4, f_start=0.)
        if kind is LambdaLinearScheduler:
            kwargs['cycle_length'] = 20
        return opt, kind(opt, **kwargs)
    opt, sched = make()
    assert opt.param_groups[0]['lr'] == pytest.approx(.025)
    sched.step()
    assert opt.param_groups[0]['lr'] == pytest.approx(.05)
    opt2, resumed = make()
    resumed.load_state_dict(sched.state_dict())
    assert opt2.param_groups[0]['lr'] == opt.param_groups[0]['lr']
    resumed.step(); sched.step()
    assert opt2.param_groups[0]['lr'] == opt.param_groups[0]['lr']


def test_metric_counts_exclude_empty_samples_and_pixel_views():
    actions = extract('train/sample.py', '_compute_action_metrics')
    gt = torch.zeros(3, 48, 58)
    pred = gt.clone(); pred[0] += 1; pred[1] += 3
    out = actions(pred, gt, torch.tensor([2, 1, 0]), 24)
    assert out['action_mse_loss'] == 5.
    assert out['_count/action_mse_loss'] == 2
    append = extract('train/sample.py', '_append_counted_metrics')
    buckets = defaultdict(list)
    append(buckets, 'policy', out)
    assert buckets['_count/policy_action_mse_loss'] == [2]
    assert not any('policy__count' in k for k in buckets)


def test_video_frequency_selection_uses_wan_axes_without_changing_action_freqs():
    # Execute the actual initialization region against distinct sentinel tables.
    tree = ast.parse((ROOT / 'models/motus.py').read_text())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__' and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'freqs_action' for t in x.targets) for x in n.body))
    assignments = [n for n in init.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr in {'freqs', 'freqs_action', 'freqs_state'} for t in n.targets)]
    wan = torch.arange(8).reshape(2, 4)
    model = SimpleNamespace(video_model=SimpleNamespace(wan_model=SimpleNamespace(freqs=wan)))
    scope = dict(self=model, d=16, rope_params=lambda n, d: torch.full((1,), n))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), 'frequency-init', 'exec'), scope)
    assert torch.equal(model.freqs, wan) and model.freqs.data_ptr() != wan.data_ptr()
    assert (model.freqs_action == 102400).all() and (model.freqs_state == 10240).all()


def test_full_evaluation_weights_uneven_batches_and_sums_skips():
    from omegaconf import OmegaConf
    import logging
    scope = dict(globals(), logger=logging.getLogger(__name__))
    tree = ast.parse((ROOT / 'train/sample.py').read_text())
    names = {'evaluate_model', '_build_evaluation_schedule', '_compute_action_metrics',
             '_append_counted_metrics', '_aligned_gt_actions', '_action_eval_sample_mask',
             '_latent_time_len', '_latent_slice_time', '_is_loop_mode'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-evaluation>', 'exec'), scope)
    class Model:
        config = SimpleNamespace(mode='joint', actions_per_latent=1, fixed_chunk_size=1)
        training = True
        def parameters(self): return iter([torch.zeros(())])
        def eval(self): self.training = False
        def _prepare_noisy_action_blocks(self, actions, states, chunk_size, dtype):
            assert actions.dtype == torch.float32
            return {'flat_actions': actions[:, chunk_size:].to(dtype)}
    def batch(size, error, skipped):
        actions = torch.zeros(size, 2, 58)
        actions[:skipped, 1:, 20:23] = 20
        return dict(latents=torch.zeros(size, 1, 2, 1, 1), actions=actions,
                    valid_len=torch.full((size,), 2), pred=torch.full((size, 1, 58), error))
    scope.update(inference_sample=lambda model, batch, config, **kw: (batch['latents'][:, :, 1:], batch['pred']),
                 _unpack_sample_output=lambda x: x,
                 _compute_video_metrics=lambda pred, gt, valid, model: {'video_latent_mse': 0., '_count/video_latent_mse': pred.shape[0]},
                 create_video_grid=lambda *a, **kw: None,
                 _snapshot_and_disable_inference_accel=lambda model: {},
                 _reset_post_eval_module_state=lambda model, prev, snap: setattr(model, 'training', prev))
    config = OmegaConf.create(dict(dataset={}, model=dict(wm_ratio=0, value={}), common=dict(wrist_xyz_abs_max=10),
                                   system=dict(validation_batches_per_mode=2)))
    model = Model()
    out = scope['evaluate_model'](model, [batch(4, 0., 0), batch(4, 0., 0), batch(2, 10., 1), batch(3, 10., 2)], None, config)
    for mode, skipped in [('policy_decouple', 1), ('policy_full', 2)]:
        assert out[f'{mode}_action_mse_loss'] == 20
        assert out[f'_count/{mode}_action_mse_loss'] == 5
        assert out[f'{mode}_skipped_windows_extreme_wrist_xyz'] == skipped
        assert f'{mode}_skipped_windows_extreme_wrist_xyz' in out['_sum_keys']
    assert model.training


def test_distributed_validation_uses_effective_counts_and_sums_counters():
    import torch.distributed as dist
    method = extract('train/train.py', '_aggregate_validation_metrics', 'Trainer')
    class Distributed:
        ReduceOp = dist.ReduceOp
        is_available = staticmethod(lambda: True)
        is_initialized = staticmethod(lambda: True)
        get_world_size = staticmethod(lambda: 2)
        @staticmethod
        def all_gather_object(out, values): out[:] = [values, values]
        @staticmethod
        def all_reduce(value, op):
            assert op == dist.ReduceOp.SUM
            if value.numel() == 1: value += 3
            else: value += torch.tensor([30., 3.], dtype=value.dtype)
    method.__globals__['dist'] = Distributed
    trainer = SimpleNamespace(device=torch.device('cpu'), rank=0)
    out = method(trainer, {'mse': 2., '_count/mse': 2,
                           'skipped': 1., '_sum_keys': ['skipped']})
    assert out == {'mse': 6.8, 'skipped': 4.}


def test_scaling_evaluation_weights_windows_instead_of_batches():
    import logging
    from omegaconf import OmegaConf
    scope = dict(globals(), logger=logging.getLogger(__name__))
    tree = ast.parse((ROOT / 'train/sample.py').read_text())
    names = {'evaluate_scaling_model', '_compute_action_component_metrics', '_aligned_gt_actions',
             '_action_eval_sample_mask', '_is_loop_mode'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<scaling-evaluation>', 'exec'), scope)
    class Model:
        config = SimpleNamespace(mode='joint', actions_per_latent=1, fixed_chunk_size=1)
        def parameters(self): return iter([torch.zeros(())])
        def eval(self): pass
        def train(self): pass
        def _prepare_noisy_action_blocks(self, actions, states, chunk_size, dtype):
            return {'flat_actions': actions[:, chunk_size:].to(dtype)}
    batches = [dict(actions=torch.zeros(n, 2, 58), pred=torch.full((n, 1, 58), error))
               for n, error in [(4, 0.), (1, 10.)]]
    scope.update(inference_sample=lambda model, batch, config, **kw: (None, batch['pred']),
                 _unpack_sample_output=lambda x: x)
    config = OmegaConf.create(dict(model={}, eval_scaling=dict(flow_samples=2)))
    out = scope['evaluate_scaling_model'](Model(), batches, None, config)
    assert out['eval_scaling/action_mse'] == 20.
    assert out['eval_scaling/eval_windows'] == 5


def test_video_timesteps_do_not_round_to_bf16_action_dtype():
    method = extract('models/motus.py', '_build_chunk_ar_timesteps', 'Motus')
    model = SimpleNamespace(config=SimpleNamespace(value_tokens_per_frame=0,
        state_tokens_per_frame=0, action_register_tokens_per_frame=1, actions_per_latent=24),
        frame_seqlen=2, _action_register_timesteps=lambda t, count, tokens, offset: t[:, None].expand(-1, count))
    out = method(model, torch.tensor([.5], dtype=torch.bfloat16), torch.tensor([457.]), [2], include_value=False)
    assert out.dtype == torch.float32
    assert torch.equal(out[:, 2:], torch.full((1, 4), 457.))


def test_relative_geometry_ignores_callers_autocast():
    from scipy.spatial.transform import Rotation
    from utils.relative_eef.relative_eef import _mat_to_rot6d_torch
    model = Geometry()
    states = torch.zeros(1, 6, 58)
    rot = torch.tensor(Rotation.from_rotvec([.43, -.28, .71]).as_matrix(), dtype=torch.float32)
    for part in (slice(23, 29), slice(52, 58)):
        states[..., part] = _mat_to_rot6d_torch(rot)
    states[..., 20:23] = torch.tensor([.91, -.73, 1.02])
    actions = states.repeat_interleave(24, 1).clone()
    actions[..., 20:23] += torch.tensor([.0012, -.0021, .0008])
    expected = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.float32)['flat_actions']
    with torch.autocast('cpu', dtype=torch.bfloat16):
        got = model._prepare_noisy_action_blocks(actions, states, 2, dtype=torch.float32)['flat_actions']
        got_slice, _ = model._slice_noisy_actions_and_states(actions, states, 2, dtype=torch.float32)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(got_slice, expected, rtol=0, atol=0)
