import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]

def checkpoint_loader():
    # Load the real loader without importing CUDA-only model dependencies.
    tree = ast.parse((ROOT / 'models/motus.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Motus')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_checkpoint_weights')
    scope = {'torch': torch, 'Path': Path, 'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<real-checkpoint-loader>', 'exec'), scope)
    return scope['load_checkpoint_weights']


@pytest.mark.parametrize('bad', ['missing', 'shape', None])
def test_required_action_weights_never_silently_reinitialize(tmp_path, bad):
    model = torch.nn.Module()
    model.config = SimpleNamespace(qkv_fuse=False)
    model.action_expert = torch.nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: torch.full_like(v, 7.) for k, v in before.items()}
    if bad == 'missing':
        del state['action_expert.bias']
    elif bad == 'shape':
        state['action_expert.weight'] = torch.ones(2, 4)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'module': state}, path)
    load = checkpoint_loader()
    if bad:
        with pytest.raises(ValueError, match='without reinitialization'):
            load(model, str(path), strict=False, strict_action=True)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])
    else:
        load(model, str(path), strict=False, strict_action=True)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, state[k])
