from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch


MODULE = Path(__file__).parents[1] / "models" / "time_embedding.py"
spec = importlib.util.spec_from_file_location("motus_time_embedding", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
embed_timesteps = module.embed_timesteps


def _sinusoidal(values):
    frequencies = torch.arange(1, 5, dtype=values.dtype, device=values.device)
    phase = values.unsqueeze(-1) * frequencies
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def _mlps():
    embedding = torch.nn.Sequential(
        torch.nn.Linear(8, 11), torch.nn.SiLU(), torch.nn.Linear(11, 11)
    )
    projection = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(11, 66))
    return embedding, projection


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_repeated_timesteps_preserve_outputs_and_parameter_gradients(compact, project):
    torch.manual_seed(42)
    timesteps = torch.tensor([[0., 0., 0.3, 0.8, 0.3, 0.], [0.2, 0., 0.2, 0.9, 0.9, 0.]])
    embedding, projection = _mlps()
    other_embedding, other_projection = copy.deepcopy((embedding, projection))
    expected_e = embedding(_sinusoidal(timesteps.reshape(-1)).reshape(2, 6, 8))
    expected_p = projection(expected_e) if project else None
    actual_e, actual_p = embed_timesteps(
        timesteps, _sinusoidal, other_embedding, other_projection,
        compact=compact, project=project,
    )
    torch.testing.assert_close(actual_e, expected_e, atol=2e-6, rtol=2e-5)
    if project:
        torch.testing.assert_close(actual_p, expected_p, atol=2e-6, rtol=2e-5)
    else:
        assert actual_p is None

    # Different token weights ensure repeat contributions are all accumulated.
    weights_e = torch.randn_like(expected_e)
    expected_loss = (expected_e * weights_e).sum()
    actual_loss = (actual_e * weights_e).sum()
    if project:
        weights_p = torch.randn_like(expected_p)
        expected_loss = expected_loss + (expected_p * weights_p).sum()
        actual_loss = actual_loss + (actual_p * weights_p).sum()
    expected_loss.backward()
    actual_loss.backward()
    for old, new in zip(
        list(embedding.parameters()) + list(projection.parameters()),
        list(other_embedding.parameters()) + list(other_projection.parameters()),
    ):
        if old.grad is None:
            assert new.grad is None
        else:
            torch.testing.assert_close(new.grad, old.grad, atol=5e-6, rtol=5e-5)


def test_compact_mlp_sees_only_unique_values_and_no_cross_batch_cache():
    embedding, projection = _mlps()
    observed = []
    handle = embedding.register_forward_pre_hook(lambda _module, args: observed.append(args[0].shape))
    timesteps = torch.tensor([[0., 0., 0.5, 0.5], [0., 1., 1., 1.]])
    for _ in range(2):
        e, p = embed_timesteps(timesteps, _sinusoidal, embedding, projection, compact=True)
        (e.sum() + p.sum()).backward()
        assert embedding[0].weight.grad is not None
        embedding.zero_grad()
        projection.zero_grad()
    handle.remove()
    assert observed == [torch.Size([1, 3, 8]), torch.Size([1, 3, 8])]


def test_differentiable_timestep_falls_back_and_keeps_input_gradient():
    timesteps = torch.tensor([[0., 0., 0.5]], requires_grad=True)
    embedding, projection = _mlps()
    e, p = embed_timesteps(timesteps, _sinusoidal, embedding, projection, compact=True)
    (e.sum() + p.sum()).backward()
    assert timesteps.grad is not None
    assert torch.isfinite(timesteps.grad).all()


def test_empty_token_axis_preserves_the_output_shape():
    embedding, projection = _mlps()
    e, p = embed_timesteps(torch.empty(2, 0), _sinusoidal, embedding, projection, compact=True)
    assert e.shape == (2, 0, 11)
    assert p.shape == (2, 0, 66)
