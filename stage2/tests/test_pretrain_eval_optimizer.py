"""CPU checks for Stage2 warmup, sample counts and masked action evaluation."""

import ast
from collections import defaultdict
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def _definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


@pytest.fixture
def schedulers():
    return _definitions("utils/scheduler.py", {
        "LambdaConstantScheduler", "LambdaLinearScheduler", "create_scheduler"
    }, {"torch": torch})


def _scheduler(namespace, kind="constant", *, warmup=1000):
    params = [torch.nn.Parameter(torch.zeros(())), torch.nn.Parameter(torch.zeros(()))]
    optimizer = torch.optim.SGD([{"params": [params[0]], "lr": 2e-5},
                                 {"params": [params[1]], "lr": 1e-5}])
    training = SimpleNamespace(scheduler_type=kind, warmup_steps=warmup, cycle_length=2000,
                               f_start=0.0, f_max=1.0, f_min=0.1)
    scheduler = namespace["create_scheduler"](optimizer, SimpleNamespace(training=training))
    return optimizer, scheduler


@pytest.mark.parametrize("kind", ["constant", "linear"])
def test_warmup_initializes_three_actual_updates_and_keeps_group_base_lrs(schedulers, kind):
    optimizer, scheduler = _scheduler(schedulers, kind)
    used = []
    for _ in range(3):
        used.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert np.asarray(used) == pytest.approx(np.array([[2e-8, 1e-8], [4e-8, 2e-8], [6e-8, 3e-8]]))
    assert scheduler.step_count == 3
    assert scheduler.base_lrs == [2e-5, 1e-5]
    assert all(lr > 0 for lr in used[0])


@pytest.mark.parametrize("kind", ["constant", "linear"])
@pytest.mark.parametrize("completed", [100, 1000])
def test_warmup_resume_prepares_exact_next_update(schedulers, kind, completed):
    optimizer, scheduler = _scheduler(schedulers, kind)
    for _ in range(completed):
        optimizer.step()
        scheduler.step()
    saved_optimizer, saved_scheduler = optimizer.state_dict(), scheduler.state_dict()
    expected = scheduler.get_last_lr()
    resumed_optimizer, resumed = _scheduler(schedulers, kind)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)
    assert resumed.step_count == completed
    assert resumed.get_last_lr() == pytest.approx(expected)
    assert resumed.get_last_lr()[0] == pytest.approx(2e-5 * resumed.get_lr_multiplier(completed + 1))
    # Trainer's legacy checkpoint alignment uses the same public method.
    resumed.step_count = completed
    resumed.apply_current_lr()
    assert resumed.get_last_lr() == pytest.approx(expected)
    optimizer.step(); scheduler.step()
    resumed_optimizer.step(); resumed.step()
    assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())


@pytest.mark.parametrize("kind", ["constant", "linear"])
def test_default_warmup_and_resume_at_step_zero(schedulers, kind):
   optimizer, scheduler = _scheduler(schedulers, kind)
   assert scheduler.get_last_lr() == pytest.approx([2e-8, 1e-8])
   _, resumed = _scheduler(schedulers, kind)
   resumed.load_state_dict(scheduler.state_dict())
   assert resumed.get_last_lr() == scheduler.get_last_lr()


@pytest.mark.parametrize("kind", ["constant", "linear"])
def test_zero_warmup(schedulers, kind):
   _, scheduler = _scheduler(schedulers, kind, warmup=0)
   factor = 1.0 if kind == "constant" else 0.99955
   assert scheduler.get_last_lr() == pytest.approx([2e-5 * factor, 1e-5 * factor])


@pytest.fixture
def sample_functions():
    return _definitions("train/sample.py", {
        "_build_evaluation_schedule", "_compute_action_metrics", "_compute_action_component_metrics",
        "_action_coordinate_valid_mask", "_accumulate_metric_batch", "_aligned_gt_actions",
        "_action_prepare_kwargs", "_action_eval_sample_mask", "_latent_time_len",
        "_latent_slice_time", "_is_loop_mode", "evaluate_model",
    }, {"torch": torch, "np": np, "F": F, "Dict": Dict, "Optional": Optional,
        "defaultdict": defaultdict, "logger": logging.getLogger(__name__)})


def test_invalid_hand_targets_never_enter_action_or_component_metrics(sample_functions):
    pred = torch.zeros(3, 2, 58)
    gt = torch.ones_like(pred)
    gt[0, :, 29:] = float("nan")
    gt[1, :, :29] = float("nan")
    gt[2] = float("nan")
    validity = torch.tensor([[[True, False]] * 2, [[False, True]] * 2, [[False, False]] * 2])
    metrics = sample_functions["_compute_action_metrics"](pred, gt, None, 1, action_valid_mask=validity)
    assert metrics["_sample_count"] == 2
    assert metrics["action_mse_loss"] == 1
    components = sample_functions["_compute_action_component_metrics"](
        pred, gt, None, 1, action_valid_mask=validity)
    assert components["_sample_count"] == 2
    assert all(value == 1 for key, value in components.items() if not key.startswith("_"))


def test_all_invalid_labels_have_no_metric_denominator(sample_functions):
    metrics = sample_functions["_compute_action_metrics"](
        torch.zeros(1, 2, 58), torch.full((1, 2, 58), float("nan")), None, 1,
        action_valid_mask=torch.zeros(1, 2, 2, dtype=torch.bool))
    assert metrics == {}


def test_aligned_targets_preserve_small_fp32_relative_motion_and_masks(sample_functions):
    class RelativeModel:
        def _prepare_noisy_action_blocks(self, actions, states, chunk_size, dtype, **kwargs):
            assert actions.dtype == states.dtype == torch.float32
            relative = (actions - states[:, :1]).to(dtype)
            return {"flat_actions": relative, "flat_action_valid_mask": kwargs["action_valid_mask"]}

    batch = {"actions": torch.full((1, 2, 58), 1.0001), "states": torch.ones(1, 2, 58),
             "action_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
             "state_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool)}
    gt, valid = sample_functions["_aligned_gt_actions"](
        RelativeModel(), batch, torch.zeros(1, 1, 58, dtype=torch.bfloat16), 1, return_valid_mask=True)
    assert gt.shape == (1, 1, 58)
    assert gt.float().abs().min() > 9e-5
    assert valid.shape == (1, 1, 2)


def test_missing_one_of_two_validity_masks_is_rejected(sample_functions):
    with pytest.raises(ValueError, match="together"):
        sample_functions["_action_prepare_kwargs"]({"action_valid_mask": torch.ones(1, 2, 2)}, "cpu")


def test_abs_filter_ignores_invalid_hand_coordinates(sample_functions):
    actions = torch.zeros(1, 2, 58)
    actions[:, :, 49:52] = float("nan")
    validity = torch.tensor([[[True, False], [True, False]]])
    mask = sample_functions["_action_eval_sample_mask"](actions, None, 1, 10.0,
                                                        action_valid_mask=validity)
    assert mask.tolist() == [True]


def test_real_evaluate_model_counts_surviving_samples_and_sums_skips(sample_functions):
    class FakeModel:
        config = SimpleNamespace(mode="joint", actions_per_latent=1, fixed_chunk_size=1)
        training = True

        def parameters(self):
            return iter([torch.zeros(())])

        def eval(self):
            self.training = False

        def _prepare_noisy_action_blocks(self, actions, states, chunk_size, dtype):
            assert actions.dtype == torch.float32
            return {"flat_actions": actions[:, chunk_size:].to(dtype)}

    def batch(size, error, skipped):
        actions = torch.zeros(size, 2, 58)
        actions[:skipped, 1:, 20:23] = 20
        return {"latents": torch.zeros(size, 1, 2, 1, 1), "actions": actions,
                "valid_len": torch.full((size,), 2), "pred": torch.full((size, 1, 58), error)}

    sample_functions.update({
        "inference_sample": lambda model, batch, config, **kwargs: (batch["latents"][:, :, 1:], batch["pred"]),
        "_unpack_sample_output": lambda result: result,
        "_compute_video_metrics": lambda pred, gt, valid, model: {"video_latent_mse": 0.0, "_count/video_latent_mse": pred.shape[0]},
        "create_video_grid": lambda *args, **kwargs: None,
        "_snapshot_and_disable_inference_accel": lambda model: {},
        "_reset_post_eval_module_state": lambda model, previous, snapshot: setattr(model, "training", previous),
    })
    # policy_decouple is gone, and batches_per_mode<=0 sweeps the whole pool, so
    # every batch is scored by the one remaining mode instead of half of them
    # being spent on a mode that runs identical code.
    settings = {"system": {"validation_batches_per_mode": 0}, "common": {"wrist_xyz_abs_max": 10}}
    config = SimpleNamespace(model={"wm_ratio": 0, "value": {}}, common=settings["common"],
                             get=settings.get)
    model = FakeModel()
    result = sample_functions["evaluate_model"](
        model, [batch(4, 0.0, 0), batch(2, 10.0, 1), batch(3, 10.0, 2), batch(4, 0.0, 0)], None, config)
    assert result["eval_batches"] == 4.0
    assert result["_count/policy_full_action_mse_loss"] == 10
    assert result["policy_full_skipped_windows_extreme_wrist_xyz"] == 3
    assert "policy_full_skipped_windows_extreme_wrist_xyz" in result["_sum_keys"]
    assert result["policy_full_action_mse_loss"] > 0
    assert not any("policy_decouple" in key for key in result)
    assert model.training is True


def test_whole_pool_eval_scores_every_batch_and_reports_coverage(sample_functions):
    class FakeModel:
        config = SimpleNamespace(mode="joint", actions_per_latent=1, fixed_chunk_size=1)
        training = True

        def parameters(self):
            return iter([torch.zeros(())])

        def eval(self):
            self.training = False

    seen = []
    sample_functions.update({
        "inference_sample": lambda model, batch, config, **kwargs: (seen.append(batch["id"]), (batch["latents"][:, :, 1:], None))[1],
        "_unpack_sample_output": lambda result: result,
        "_compute_video_metrics": lambda pred, gt, valid, model: {"video_latent_mse": 2.0, "_sample_count": pred.shape[0]},
        "create_video_grid": lambda *args, **kwargs: None,
        "_snapshot_and_disable_inference_accel": lambda model: {},
        "_reset_post_eval_module_state": lambda model, previous, snapshot: setattr(model, "training", previous),
    })
    batches = [{"id": i, "latents": torch.zeros(2, 1, 3, 1, 1), "valid_len": torch.tensor([3, 3])} for i in range(6)]
    settings = {"system": {"validation_batches_per_mode": 0}, "common": {}}
    config = SimpleNamespace(model={"wm_ratio": 0, "value": {}}, common={}, get=settings.get)
    result = sample_functions["evaluate_model"](FakeModel(), batches, None, config)
    assert seen == list(range(6)), "batches_per_mode<=0 must sweep the whole pool"
    assert result["eval_batches"] == 6.0


def test_grouped_evaluation_runs_the_single_mode_and_keeps_each_pool_images_and_counts(sample_functions):
    class FakeModel:
        config = SimpleNamespace(mode='joint', actions_per_latent=24, fixed_chunk_size=1)
        training = True
        def parameters(self): return iter([torch.zeros(())])
        def eval(self): self.training = False

    calls, resets = [], []
    image = object()
    def infer(model, batch, config, **kwargs):
        calls.append((batch['pool'], kwargs['full_denoise']))
        return batch['latents'][:, :, 1:], None
    sample_functions.update({
        'inference_sample': infer,
        '_unpack_sample_output': lambda result: result,
        '_compute_video_metrics': lambda pred, gt, valid, model: {'video_latent_mse': 2., '_sample_count': pred.shape[0]},
        'create_video_grid': lambda *args, **kwargs: image,
        '_snapshot_and_disable_inference_accel': lambda model: {},
        '_reset_post_eval_module_state': lambda model, previous, snapshot: (resets.append(previous), setattr(model, 'training', previous)),
    })
    names = ['legacy_validation', 'wuji1_diagnostic', 'wuji2_diagnostic', 'HandsOn_heldout']
    def batch(name):
        return {'pool': name, 'latents': torch.zeros(2, 1, 3, 1, 1), 'valid_len': torch.tensor([3, 3])}
    pools = {name: [batch(name), batch(name)] for name in names}
    settings = {'system': {'validation_batches_per_mode': 1}, 'eval_sample': {'groups': {name: {} for name in names}}}
    config = SimpleNamespace(model={'wm_ratio': 0, 'value': {}}, get=settings.get)
    model = FakeModel()
    values = sample_functions['evaluate_model'](model, pools, None, config)
    assert calls == [(name, True) for name in names]
    assert len(resets) == 4 and all(resets) and model.training
    for name in names:
        assert values[f'{name}/policy_full_video_latent_mse'] == 2.
        assert values[f'_count/{name}/policy_full_video_latent_mse'] == 2
        assert values[f'visualization_{name}_policy_full'] is image
