"""Output-head time embeddings must match reference outputs and gradients."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _methods():
    source = ast.parse((ROOT / "models/motus.py").read_text())
    names = {"_video_time_embedding", "_video_time_embed", "_action_time_embedding", "_action_time_embed"}
    functions = [node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name in names]
    wan_source = ast.parse((ROOT / "third_party/wan/modules/model.py").read_text())
    sinusoidal = next(node for node in wan_source.body if isinstance(node, ast.FunctionDef) and node.name == "sinusoidal_embedding_1d")
    namespace = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module(body=[sinusoidal, *functions], type_ignores=[]), "<time_embeddings>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("kind", ["video", "action"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_head_embedding_preserves_values_gradients_rng_and_skips_projection(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA worker required")
    ns = _methods()
    dim = 64
    branch = SimpleNamespace(
        freq_dim=256, config=SimpleNamespace(dim=dim),
        time_embedding=torch.nn.Sequential(torch.nn.Linear(256, dim), torch.nn.SiLU(), torch.nn.Linear(dim, dim)).to(device),
        time_projection=torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(dim, 6 * dim)).to(device),
    )
    model = SimpleNamespace(wan_dim=dim, video_model=SimpleNamespace(wan_model=branch), action_expert=branch)
    for name, fn in ns.items():
        if name.startswith("_video_time_") or name.startswith("_action_time_"):
            setattr(model, name, fn.__get__(model))
    params = list(branch.time_embedding.parameters()) + list(branch.time_projection.parameters())
    timesteps = torch.tensor([[0., 0., 327., 713.], [931., 47., 47., 0.]], device=device, requires_grad=True)
    weights = torch.randn(2, 4, dim, device=device)

    # Use an independent expression as the numerical reference.
    with torch.amp.autocast('cuda', dtype=torch.float32):
        expected = branch.time_embedding(ns["sinusoidal_embedding_1d"](256, timesteps.flatten()).unflatten(0, (2, 4)).float())
        projected = branch.time_projection(expected).unflatten(2, (6, dim))
    (expected * weights).sum().backward()
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    timestep_grad = timesteps.grad.clone()
    for p in params:
        p.grad = None
    timesteps.grad = None
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    calls = []
    hook = branch.time_projection.register_forward_hook(lambda *args: calls.append(True))
    actual = getattr(model, f"_{kind}_time_embedding")(timesteps)
    (actual * weights).sum().backward()
    hook.remove()
    assert calls == []
    assert torch.equal(actual, expected)
    assert torch.equal(timesteps.grad, timestep_grad)
    assert all((p.grad is None) if grad is None else torch.equal(p.grad, grad) for p, grad in zip(params, grads))
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    embedding, projection = getattr(model, f"_{kind}_time_embed")(timesteps)
    assert torch.equal(embedding, expected)
    assert torch.equal(projection, projected)
