"""Random retention mask and deployment selectors (CPU only)."""

import importlib.util
from pathlib import Path

import torch

import infra.varlen.block_mask as bm

ROOT = Path(__file__).resolve().parent.parent


def _load_selector_module():
    spec = importlib.util.spec_from_file_location("memory_selector", ROOT / "models" / "memory_selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skeleton(W, recent, monkeypatch):
    monkeypatch.setattr(bm, "compiled_create_block_mask", lambda *a, **k: None)
    return bm._compute_block_skeleton(
        W, vbs=3, gbs=0, abs_=2, sbs=1, nvpf=0, anchor_frames=0, recent_frames=recent,
        n_anchor_tok=0, mode="fdm", obs=1, cs=2, pad_multiple=8, separate_anchor=False, device="cpu",
    )


def test_retaining_everything_equals_full_history(monkeypatch):
    W = [15, 9]
    full = _skeleton(W, 10000, monkeypatch)
    recent = _skeleton(W, 4, monkeypatch)
    frames = [list(range(s.n_pred_frames)) for s in recent.segments]
    assert torch.equal(bm._retention_logical_mask(recent, frames, "cpu"), full.logical_block_mask)


def test_retaining_nothing_equals_recent_window(monkeypatch):
    recent = _skeleton([15, 9], 4, monkeypatch)
    assert torch.equal(bm._retention_logical_mask(recent, [None, []], "cpu"), recent.logical_block_mask)


def test_retained_frame_only_opens_its_own_column_to_later_queries(monkeypatch):
    recent = _skeleton([15, 9], 4, monkeypatch)
    widened = bm._retention_logical_mask(recent, [[1], None], "cpu")
    ss = recent.segments[0]
    types = torch.tensor(ss.block_types)
    phys = torch.tensor(ss.block_phys)
    key = next(i for i in range(ss.n_blocks) if types[i] == bm.TOKEN_CLEAN_VIDEO and phys[i] == 1)
    for q in range(ss.n_blocks):
        expected = bool(recent.logical_block_mask[q, key]) or bool(phys[q] > 1)
        assert bool(widened[q, key]) == expected
    changed = (widened != recent.logical_block_mask).nonzero().tolist()
    assert changed and all(col == key for _, col in changed)


def test_selectors_respect_budget_and_never_revive():
    ms = _load_selector_module()
    gate = ms.WriteGate(48 * 16, 64)
    for kind in ["none", "all", "fifo", "first", "random", "uniform", "gate"]:
        selector = ms.MemorySelector(kind, 5, seed=1, gate=gate if kind == "gate" else None)
        selector.reset(0)
        dropped, previous = set(), set()
        for fmax in range(60):
            selector.observe(fmax, torch.randn(1, 48, 1, 8, 8), torch.randn(1, 7, 64))
            selector.advance(fmax - 3)
            current = set(selector.kept)
            assert kind == "all" or len(current) <= 5
            assert not (current & dropped)
            assert all(f < fmax - 3 for f in current)
            dropped |= previous - current
            previous = current
        if kind == "none":
            assert not current
        if kind == "all":
            assert current == set(range(56))
