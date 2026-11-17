"""The strict per-update finite check covers every weight with few reductions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.strict_finite import _all_finite, _strict_update_tensors  # noqa: E402


def fake_zero2(sizes=(5, 7, 3), extra=2):
    """Parameters as views into one zero-padded flat buffer, as ZeRO-2 keeps them,
    plus a parameter the buffer does not own and an AdamW-like inner optimizer."""
    flat = torch.zeros(sum(sizes) + 4, dtype=torch.bfloat16)  # alignment padding
    params, offset = [], 0
    for size in sizes:
        p = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
        p.data = flat[offset:offset + size].view(size)
        p.data.fill_(0.5)
        params.append(p)
        offset += size
    outside = torch.nn.Parameter(torch.ones(extra))
    master = torch.ones(8)
    inner = SimpleNamespace(param_groups=[{'params': [master]}],
                            state={0: {'exp_avg': torch.zeros(8), 'exp_avg_sq': torch.zeros(8),
                                       'step': torch.tensor(3.0)}})
    engine = SimpleNamespace(optimizer=SimpleNamespace(bit16_groups_flat=[flat], bit16_groups=[params]),
                             parameters=lambda: iter(params + [outside]))
    return engine, inner, params, outside, master, inner.state[0]


def test_clean_update_is_finite():
    engine, inner, *_ = fake_zero2()
    assert _all_finite(_strict_update_tensors(engine, inner), 'cpu')


def test_flat_buffer_replaces_its_parameters_but_nothing_else():
    engine, inner, params, outside, master, state = fake_zero2()
    tensors = _strict_update_tensors(engine, inner)
    ids = {id(t) for t in tensors}
    assert not any(id(p.data) in ids for p in params), 'owned parameters are checked through the buffer'
    assert sum(t.numel() for t in tensors) == (15 + 4) + 2 + 8 + 8 + 8 + 1


def test_any_nonfinite_element_anywhere_is_caught():
    for poison in ('owned', 'outside', 'master', 'exp_avg_sq'):
        for bad in (float('nan'), float('inf'), float('-inf')):
            engine, inner, params, outside, master, state = fake_zero2()
            target = {'owned': params[1].data, 'outside': outside.data, 'master': master,
                      'exp_avg_sq': state['exp_avg_sq']}[poison]
            target.view(-1)[-1] = bad
            assert not _all_finite(_strict_update_tensors(engine, inner), 'cpu'), (poison, bad)


def test_without_flat_buffers_every_parameter_is_checked_individually():
    engine, inner, params, outside, *_ = fake_zero2()
    engine.optimizer = SimpleNamespace()
    tensors = _strict_update_tensors(engine, inner)
    assert sum(t.numel() for t in tensors) == 15 + 2 + 8 + 8 + 8 + 1
    params[0].data[2] = float('nan')
    assert not _all_finite(tensors, 'cpu')
