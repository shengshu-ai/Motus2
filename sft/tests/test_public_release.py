from pathlib import Path
from types import SimpleNamespace
import importlib
import pytest
import torch
from models.motus import Motus, MotusConfig
from utils.action_mode import resolve_action_mode_from_config

@pytest.mark.parametrize("module", ["data.robotwin2.robotwin_agilex_dataset", "data.tianji_wuji.tianji_wuji_dataset", "data.pretrain.pretrain_dataset"])
def test_supported_loader_imports(module):
    importlib.import_module(module)

@pytest.mark.parametrize("use_state", [False, True])
@pytest.mark.parametrize("include_value", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_chunk_packing_and_rope_share_native_order(use_state, include_value, batched):
    model = Motus.__new__(Motus)
    torch.nn.Module.__init__(model)
    spf, apf, vpf, nvpf = int(use_state), 2, 3, int(include_value)
    model.config = SimpleNamespace(state_tokens_per_frame=spf, action_register_tokens_per_frame=apf,
        action_tokens_per_timestep=1, value_tokens_per_frame=nvpf)
    model.device = torch.device("cpu")
    model.frame_seqlen = vpf
    model.freqs = torch.ones(20, 2, dtype=torch.complex128)
    model.freqs_state = model.freqs.clone()
    model.freqs_action = torch.arange(2, 42, dtype=torch.float64).reshape(20, 2).to(torch.complex128)
    counts = [2, 1]
    clean = torch.full((3, 1, 2), 100., dtype=torch.complex128)
    video = torch.arange(200, 218, dtype=torch.float64).reshape(9, 1, 2).to(torch.complex128)
    expected = [clean]
    off = 0
    for count in counts:
        expected.extend([torch.ones(spf, 1, 2, dtype=torch.complex128), model.freqs_action[:count*apf].unsqueeze(1), video[off:off+count*vpf], torch.ones(nvpf, 1, 2, dtype=torch.complex128)])
        off += count*vpf
    expected = torch.cat(expected)
    if batched:
        clean, video = clean.repeat(2, 1, 1, 1), video.repeat(2, 1, 1, 1)
        expected = expected.repeat(2, 1, 1, 1)
    got = model._build_chunk_ar_freqs(clean, video, counts, include_value=include_value)
    assert torch.equal(got, expected)
    a = torch.arange(6).reshape(1, 3, 2)
    v = torch.arange(6, 14).reshape(1, 4, 2)
    packed = model._cat_chunk_tokens(a, v)
    sa, sv = model._chunk_token_slices(3, 4)
    assert torch.equal(packed[:, sa], a)
    assert torch.equal(packed[:, sv], v)


def test_public_config_has_no_old_layout_or_normalization_alias():
    fields = MotusConfig.__dataclass_fields__
    assert "current_layout" not in fields
    assert "use_action_first_layout" not in fields
    assert "wrist_relative_normalization" not in fields
    with pytest.raises(ValueError, match="model.normalization"):
        resolve_action_mode_from_config({"action_mode": "qpos20_rel6d"}, {"qpos_normalization": "min_max"})
