"""Numerical regressions without loading model weights or importing CUDA WAN."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
DEVICE = os.environ.get("CACHE_TEST_DEVICE", "cpu")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(path, name, scope):
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


cache = load_file("cache_under_test", ROOT / "models/t5_kv_cache.py")


class WanCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = 2, 8
        self.q, self.k, self.v, self.o = [nn.Linear(16, 16) for _ in range(4)]
        self.norm_q, self.norm_k = nn.LayerNorm(16), nn.LayerNorm(16)
        self.original_calls = 0

    def forward(self, x, context, context_lens=None):
        self.original_calls += 1
        b = x.shape[0]
        q = self.norm_q(self.q(x)).view(b, -1, 2, 8).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(b, -1, 2, 8).transpose(1, 2)
        v = self.v(context).view(b, -1, 2, 8).transpose(1, 2)
        mask = None if context_lens is None else (
            torch.arange(context.shape[1], device=x.device)[None, None, None, :]
            < context_lens[:, None, None, None])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).contiguous().flatten(2))


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("MOTUS_USE_SDPA", "1")
    torch.manual_seed(23)
    module = WanCrossAttention().to(DEVICE).eval()
    x, ctx = torch.randn(2, 3, 16, device=DEVICE), torch.randn(2, 5, 16, device=DEVICE)
    cache.install_t5_kv_cache(module)
    return module, x, ctx


@torch.no_grad()
def test_repeated_denoise_reuses_projections_and_matches_output(setup):
    m, x, ctx = setup
    counts = []
    handle = m.k.register_forward_hook(lambda *args: counts.append(1))
    expected = m._motus2_original_forward(x, ctx, None)
    cache.populate_t5_kv_cache(m, ctx)
    before = len(counts)
    for _ in range(3):
        torch.testing.assert_close(m(x, ctx, None), expected)
    assert len(counts) == before
    handle.remove()


@pytest.mark.parametrize("mutation", ["new_text", "inplace_text", "weight", "norm", "replace_weight", "load_state", "invalidate", "none"])
@torch.no_grad()
def test_stale_cache_never_changes_results(setup, mutation):
    m, x, ctx = setup
    cache.populate_t5_kv_cache(m, ctx)
    if mutation == "new_text":
        ctx = ctx + 1.0
    elif mutation == "inplace_text":
        ctx.add_(1.0)
    elif mutation == "weight":
        m.v.weight.add_(0.2)
    elif mutation == "norm":
        m.norm_k.weight.mul_(2.0)
    elif mutation == "replace_weight":
        m.v.weight = nn.Parameter(m.v.weight.clone() + 0.1)
    elif mutation == "load_state":
        state = {k: v.clone() + 0.1 for k, v in m.state_dict().items()}
        m.load_state_dict(state)
    elif mutation == "invalidate":
        cache.invalidate_t5_kv_cache(m)
    elif mutation == "none":
        cache.populate_t5_kv_cache(m, None)
    expected = m._motus2_original_forward(x, ctx, None)
    before = m.original_calls
    torch.testing.assert_close(m(x, ctx, None), expected)
    assert m.original_calls == before + 1


@pytest.mark.parametrize("training", [True, False])
def test_gradients_survive_populated_cache(setup, training):
    m, x, ctx = setup
    cache.populate_t5_kv_cache(m, ctx)
    m.train(training)
    ctx.requires_grad_(True)
    params = tuple(m.parameters()) + (ctx,)
    expected = torch.autograd.grad(m._motus2_original_forward(x, ctx, None).sum(), params)
    actual = torch.autograd.grad(m(x, ctx, None).sum(), params)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@torch.no_grad()
def test_autocast_and_variable_lengths(setup):
    m, x, ctx = setup
    cache.populate_t5_kv_cache(m, ctx)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        torch.testing.assert_close(m(x, ctx, None), m._motus2_original_forward(x, ctx, None))
        cache.populate_t5_kv_cache(m, ctx)
        assert m._motus2_t5_k_cache.dtype == torch.bfloat16
        torch.testing.assert_close(m(x, ctx, None), m._motus2_original_forward(x, ctx, None))
    cache.populate_t5_kv_cache(m, ctx)
    lengths = torch.tensor([2, 4], device=DEVICE)
    torch.testing.assert_close(m(x, ctx, lengths), m._motus2_original_forward(x, ctx, lengths))


def test_inference_tensors_safely_fall_back(setup):
    m, x, ctx = setup
    with torch.inference_mode():
        ctx = ctx.clone()
        assert cache.populate_t5_kv_cache(m, ctx)["populated"] == 0
        ctx.add_(1)
        torch.testing.assert_close(m(x, ctx, None), m._motus2_original_forward(x, ctx, None))


@torch.no_grad()
def test_compiled_forward_does_not_capture_stale_cache(setup):
    m, x, ctx = setup
    cache.populate_t5_kv_cache(m, ctx)
    compiled = torch.compile(m, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, ctx, None), m._motus2_original_forward(x, ctx, None))
    ctx.add_(0.3)
    m.v.weight.add_(0.2)
    torch.testing.assert_close(compiled(x, ctx, None), m._motus2_original_forward(x, ctx, None))


@torch.no_grad()
def test_same_language_object_refreshes_on_each_context(setup, monkeypatch):
    m, x, ctx = setup
    fake_models = ModuleType("models")
    fake_models.__path__ = []
    monkeypatch.setitem(sys.modules, "models", fake_models)
    monkeypatch.setitem(sys.modules, "models.t5_kv_cache", cache)
    scope = {"torch": torch, "__package__": "models"}
    ensure = extract(ROOT / "models/motus.py", "_ensure_t5_kv_cache", scope)
    owner = SimpleNamespace(video_model=SimpleNamespace(wan_model=m))
    language = [ctx]
    for delta in (0.0, 1.0, -2.0):
        new_context = ctx + delta
        ensure(owner, new_context, language)
        assert m._motus2_t5_context is new_context
        torch.testing.assert_close(m(x, new_context, None), m._motus2_original_forward(x, new_context, None))


@pytest.fixture
def solvers(monkeypatch):
    motus = ModuleType("models.motus")
    motus._FlowEulerScheduler = extract(ROOT / "models/motus.py", "_FlowEulerScheduler", {"torch": torch})
    unipc = load_file("solver_under_test", ROOT / "third_party/wan/utils/fm_solvers_unipc.py")
    monkeypatch.setitem(sys.modules, "models.motus", motus)
    monkeypatch.setitem(sys.modules, "wan.utils.fm_solvers_unipc", unipc)
    build = extract(ROOT / "train/sample.py", "_build_validation_scheduler", {})
    return build, motus._FlowEulerScheduler, unipc.FlowUniPCMultistepScheduler


@pytest.mark.parametrize("name", ["euler", "unipc"])
@pytest.mark.parametrize("shift", [1.0, 5.0])
@pytest.mark.parametrize("final_noise", [0.0, 0.2])
def test_solver_trajectory_matches_selected_reference(solvers, name, shift, final_noise):
    build, Euler, UniPC = solvers
    actual = build(name, 4, DEVICE, shift)
    if name == "euler":
        expected = Euler(4, device=DEVICE, shift=shift, final_sigma=0)
    else:
        expected = UniPC(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False, solver_order=1)
        expected.set_timesteps(4, device=DEVICE, shift=shift)
    for scheduler in (actual, expected):
        if final_noise:
            sigma_max = scheduler.sigmas[0].item()
            scheduler.sigmas = scheduler.sigmas * (sigma_max - final_noise) / sigma_max + final_noise
            scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64)
    torch.testing.assert_close(actual.timesteps, expected.timesteps)
    torch.testing.assert_close(actual.sigmas, expected.sigmas)
    a = torch.ones(2, 3, device=DEVICE)
    b = a.clone()
    for ta, tb in zip(actual.timesteps, expected.timesteps):
        a = actual.step(a * 0.1, ta, a, return_dict=False)[0]
        b = expected.step(b * 0.1, tb, b, return_dict=False)[0]
        torch.testing.assert_close(a, b)


def test_invalid_solver_rejected(solvers):
    with pytest.raises(ValueError, match="inference_scheduler"):
        solvers[0]("typo", 4, DEVICE, 5.0)


def test_training_constructor_forwards_explicit_and_omitted_solver():
    tree = ast.parse((ROOT / "train/train.py").read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "MotusConfig")
    expr = next(k.value for k in call.keywords if k.arg == "inference_scheduler")
    for raw, expected in [({}, "unipc"), ({"inference_scheduler": "euler"}, "euler")]:
        assert eval(compile(ast.Expression(expr), "constructor", "eval"),
                    {"m": SimpleNamespace(inference=raw)}) == expected


@pytest.mark.parametrize("stereo", [False, True])
def test_video_scheduler_shape_adapter(solvers, stereo):
    from typing import Tuple
    step = extract(ROOT / "models/motus.py", "_step_video_scheduler", {"torch": torch, "Tuple": Tuple})
    for name in ("euler", "unipc"):
        a, b = [solvers[0](name, 4, DEVICE, 5.0) for _ in range(2)]
        shape = (2, 2, 3, 4, 2, 2) if stereo else (2, 3, 4, 2, 2)
        sample = torch.randn(shape, device=DEVICE)
        expected = sample.clone()
        for ta, tb in zip(a.timesteps, b.timesteps):
            sample = step(None, a, sample * 0.1, ta, sample)
            # UniPC solves over non-batch axes without changing element order.
            if stereo:
                packed = expected.reshape(2, 6, 4, 2, 2).transpose(1, 2)
                expected = b.step(packed * 0.1, tb, packed, return_dict=False)[0].transpose(1, 2).reshape(shape)
            else:
                packed = expected.transpose(1, 2)
                expected = b.step(packed * 0.1, tb, packed, return_dict=False)[0].transpose(1, 2)
            torch.testing.assert_close(sample, expected)


@pytest.mark.skipif(DEVICE != "cuda", reason="real FlashAttention requires CUDA")
@pytest.mark.parametrize("lengths", [False, True])
@torch.no_grad()
def test_real_wan_flash_attention_cache(monkeypatch, lengths):
    monkeypatch.setenv("MOTUS_USE_SDPA", "0")
    attention = load_file("wan_attention_under_test", ROOT / "third_party/wan/modules/attention.py")
    monkeypatch.setitem(sys.modules, "wan.modules.attention", attention)
    scope = {"torch": torch, "nn": nn, "flash_attention": attention.flash_attention}
    for name in ("WanRMSNorm", "WanSelfAttention", "WanCrossAttention"):
        extract(ROOT / "third_party/wan/modules/model.py", name, scope)
    m = scope["WanCrossAttention"](64, 2).cuda().bfloat16().eval()
    x = torch.randn(2, 7, 64, device="cuda", dtype=torch.bfloat16)
    ctx = torch.randn(2, 19, 64, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([11, 19], device="cuda", dtype=torch.int32) if lengths else None
    expected = m(x, ctx, lens)
    cache.install_t5_kv_cache(m)
    cache.populate_t5_kv_cache(m, ctx)
    torch.testing.assert_close(m(x, ctx, lens), expected, rtol=0, atol=0)
    ctx.add_(0.1)
    torch.testing.assert_close(m(x, ctx, lens), m._motus2_original_forward(x, ctx, lens), rtol=0, atol=0)
