"""State means measured observations, never commanded targets or copied action weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_checkpoint_loading import checkpoint_loader
from tests.test_handson_native_posttrain import Geometry, extract
from tests.test_handson import fixture_manifest
from data.handson.dataset import HandsOnDataset


@pytest.mark.parametrize('flag', [False, True])
def test_loader_only_allows_new_state_branch(tmp_path, flag):
    model = nn.Module()
    model.config = SimpleNamespace(use_state=True, qkv_fuse=False)
    model.action_expert = nn.Module()
    model.action_expert.input_encoder = nn.Module()
    model.action_expert.input_encoder.state_encoder = nn.Linear(3, 3)
    model.action_expert.input_encoder.action_encoder = nn.Linear(3, 3)
    model.action_expert.decoder = nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint = {k: torch.full_like(v, 7.) for k, v in before.items() if '.state_encoder.' not in k}
    path = tmp_path / 'weights.pt'; torch.save({'module': checkpoint}, path)
    load = checkpoint_loader()
    if not flag:
        with pytest.raises(ValueError, match='without reinitialization'):
            load(model, str(path), strict=False, strict_action=True)
        return
    load(model, str(path), strict=False, strict_action=True, allow_new_state_encoder=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key] if '.state_encoder.' in key else checkpoint[key])
    # Losing an action parameter must still fail before any weights mutate.
    del checkpoint['action_expert.decoder.weight']
    torch.save({'module': checkpoint}, path)
    with pytest.raises(ValueError, match='without reinitialization'):
        load(model, str(path), strict=False, strict_action=True, allow_new_state_encoder=True)


@pytest.mark.parametrize('problem', ['shape', 'partial_state'])
def test_state_allowance_does_not_hide_corrupt_parameters(tmp_path, problem):
    model = nn.Module(); model.config = SimpleNamespace(use_state=True, qkv_fuse=False)
    model.action_expert = nn.Module(); model.action_expert.input_encoder = nn.Module()
    model.action_expert.input_encoder.state_encoder = nn.Linear(3, 3)
    model.action_expert.input_encoder.action_encoder = nn.Linear(3, 3)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    if problem == 'shape': state['action_expert.input_encoder.state_encoder.weight'] = torch.ones(2, 2)
    else: del state['action_expert.input_encoder.state_encoder.bias']
    path = tmp_path / 'bad.pt'; torch.save({'module': state}, path)
    with pytest.raises(ValueError, match='without reinitialization'):
        checkpoint_loader()(model, str(path), strict=False, strict_action=True, allow_new_state_encoder=True)


def test_measured_state_reaches_separate_encoder_and_not_action_token(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    sample = HandsOnDataset(manifest, val=True, max_latent_window=4, video_action_freq_ratio=6)[0]
    # Native fixture intentionally has states = commands + 1000.
    assert sample['states'][2, 0] == 1036 and sample['actions'][48, 0] == 36
    model = Geometry(use_state=True)
    model.config.action_mode = 'abs_eef52'
    blocks = model._prepare_noisy_action_blocks(sample['actions'][None], sample['states'][None], 2, dtype=torch.float32)
    assert blocks['states'][0, 0, 0] == 1036
    assert blocks['action_chunks'][0][0, 0, 0] == 36
    tree = ast.parse((Path(__file__).parents[1] / 'models/action_expert.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StateActionEncoder')
    scope = dict(nn=nn, torch=torch, ActionExpertConfig=object, build_mlp=lambda *args: nn.Identity())
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-state-encoder>', 'exec'), scope)
    model.action_expert = SimpleNamespace(input_encoder=scope['StateActionEncoder'](SimpleNamespace(state_dim=52, action_dim=52, dim=52)))
    model.training = False
    method = extract('models/motus.py', '_encode_action_chunk_tokens', 'Motus')
    tokens = method(model, blocks['action_chunks'], blocks['states'])
    assert tokens[0, 0, 0] == 1036 and tokens[0, 1, 0] == 36
    with pytest.raises(ValueError, match='measured'):
        method(model, blocks['action_chunks'], None)
    current = extract('models/motus.py', '_encode_current_action_tokens', 'Motus')
    model.dtype = torch.float32; model.device = 'cpu'
    tokens = current(model, blocks['action_chunks'][0], blocks['states'][:, 0], project_in=False)
    assert tokens[0, 0, 0] == 1036 and tokens[0, 1, 0] == 36
    with pytest.raises(ValueError, match='measured'):
        current(model, blocks['action_chunks'][0], None, project_in=False)
