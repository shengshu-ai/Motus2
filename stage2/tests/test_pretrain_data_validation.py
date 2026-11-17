"""Independent regressions for source identity and missing hand observations."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import ar_collate_fn, create_dataset
from data.pretrain import pretrain_dataset as module
from data.pretrain.pretrain_dataset import PretrainDataset


def partial_dataset():
    ds = PretrainDataset.__new__(PretrainDataset)
    ds.action_mode = "qpos20_rel6d"
    ds.action_upsample_ratio = 1
    ds.action_downsample_stride = 1
    ds.actions_per_latent = 24
    ds.cache_dir = None
    ds.qpos_normalization = "min_max"
    ds.qpos_min = torch.zeros(40)
    ds.qpos_range = torch.ones(40)
    ds.qpos_min_by_dataset = {}
    ds.qpos_range_by_dataset = {}
    ds.qpos_mean_by_dataset = {}
    ds.qpos_std_by_dataset = {}
    return ds


def raw54(count=70):
    raw = torch.arange(count, dtype=torch.float64)[:, None].expand(count, 54).clone() / 100
    raw[:, 23:27] = torch.tensor([0, 0, 0, 1])
    raw[:, 50:54] = torch.tensor([0, 0, 0, 1])
    return raw


@pytest.mark.parametrize("alias", ["after_0601", "before_0601", "new", "zedm"])
def test_stats_use_canonical_source_for_wuji1_aliases(alias):
    ds = partial_dataset()
    ds.qpos_min_by_dataset["wuji1"] = torch.full((40,), 0.2)
    ds.qpos_range_by_dataset["wuji1"] = torch.full((40,), 2.0)
    ep = {"dataset_name": f"lingqiaoshou/{alias}", "stereo_source": "wuji1"}
    key = ds._stats_dataset_name(ep)
    assert key == "wuji1"
    assert ds._qpos_stats_for(key)[1] is ds.qpos_min_by_dataset["wuji1"]
    # A deliberately configured alias keeps precedence over the broader source.
    ds.qpos_min_by_dataset[ep["dataset_name"]] = torch.full((40,), 0.3)
    assert ds._stats_dataset_name(ep) == ep["dataset_name"]


@pytest.mark.parametrize("val", [False, True])
def test_fallback_never_borrows_another_source_or_resolution(val):
    ds = partial_dataset()
    ds.val = val
    ds.source_weighted_sampling = True
    ds.episode_files = [
        {"episode_path": "failed", "dataset_name": "alias1", "stereo_source": "A", "bucket_key": "512"},
        {"episode_path": "wrong_source", "dataset_name": "B", "bucket_key": "512"},
        {"episode_path": "wrong_bucket", "dataset_name": "A", "bucket_key": "384"},
        {"episode_path": "valid_alias", "dataset_name": "alias2", "stereo_source": "A", "bucket_key": "512"},
    ]
    ds.total_episodes = len(ds.episode_files)
    ds._bucket_to_episode_indices = {"512": [0, 1, 3], "384": [2]}
    ds._dropped_episode_ids = {ds._episode_identity(ds.episode_files[0])}
    assert all(ds._episode_for_index(0)["episode_path"] == "valid_alias" for _ in range(30))
    ds._dropped_episode_ids.add(ds._episode_identity(ds.episode_files[3]))
    assert ds._episode_for_index(0) is None


def test_raw54_validity_precedes_rot6d_and_never_mutates_payload(monkeypatch):
    ds = partial_dataset()
    raw = raw54()
    raw[[0, 25, 49], 23:27] = 0
    raw[[24, 69], 50:54] = 0
    before = raw.clone()
    monkeypatch.setattr(module, "smart_load_pt", lambda *args, **kwargs: raw)
    legacy = ds._load_qpos20_wrist9_sequence({"tjwj_path": "raw"})
    seq, valid = ds._load_qpos20_wrist9_sequence({"tjwj_path": "raw"}, return_validity=True)
    assert torch.equal(seq, legacy)
    assert torch.equal(raw, before) and raw.dtype == torch.float64
    assert valid.dtype == torch.bool and valid.shape == (70, 2)
    assert (~valid[:, 0]).nonzero().flatten().tolist() == [0, 25, 49]
    assert (~valid[:, 1]).nonzero().flatten().tolist() == [24, 69]
    assert torch.equal(seq[25, 23:29], torch.tensor([1, 0, 0, 0, 1, 0]))


def test_explicit_visibility_only_removes_raw_quaternion_validity():
    raw = raw54(4)
    raw[2, 23:27] = 0
    payload = {"hand_visibility": [[1, 0], [1, 1], [1, 1], [1, 1]],
               "left_hand_valid": [1, 0, 1, 1]}
    actual = PretrainDataset._hand_tracking_valid_mask(payload, 4, raw[:, 23:27], raw[:, 50:54])
    assert actual.tolist() == [[True, False], [False, True], [False, True], [True, True]]
    with pytest.raises(ValueError, match="explicit hand visibility"):
        PretrainDataset._hand_tracking_valid_mask({}, 4)
    with pytest.raises(ValueError, match="explicit hand visibility"):
        PretrainDataset._hand_tracking_valid_mask({"left_hand_valid": [1, 1, 1, 1]}, 4)
    with pytest.raises(ValueError, match="shape"):
        PretrainDataset._hand_tracking_valid_mask({"hand_visibility": [[1, 1]]}, 4)
    with pytest.raises(ValueError, match="boolean/0/1"):
        PretrainDataset._hand_tracking_valid_mask({"left_hand_valid": [1, 2, 1, 1]}, 4)


def test_converted_raw58_needs_explicit_visibility_in_opt_in_mode(monkeypatch):
    ds = partial_dataset()
    raw = torch.zeros(4, 58)
    monkeypatch.setattr(module, "smart_load_pt", lambda *args, **kwargs: raw)
    assert ds._load_qpos20_wrist9_sequence({"tjwj_path": "raw"}).shape == (4, 58)
    with pytest.raises(ValueError, match="explicit hand visibility"):
        ds._load_qpos20_wrist9_sequence({"tjwj_path": "raw"}, return_validity=True)
    payload = {"action": raw, "hand_valid_mask": [[1, 1], [0, 1], [1, 0], [1, 1]]}
    monkeypatch.setattr(module, "smart_load_pt", lambda *args, **kwargs: payload)
    _, valid = ds._load_qpos20_wrist9_sequence({"tjwj_path": "raw"}, return_validity=True)
    assert valid.tolist() == [[True, True], [False, True], [True, False], [True, True]]


def test_handson_initial_state_mask_uses_measured_phase_anchor(monkeypatch):
    from data.pretrain import handson_adapter
    ds = partial_dataset()
    ds.mask_invalid_hand_tracking = True
    raw = raw54()
    flags = torch.ones(70, 2, dtype=torch.bool)
    flags[2, 0] = False
    flags[3, 1] = False
    payload = {"schema": "handson-stage2/raw54-v1", "action_hz": 30,
        "commands": raw, "measured_states": raw.clone(), "hand_valid_mask": flags,
        "qpos_stats": {"statistics_split": "train", "min": [0]*40, "max": [1]*40}}
    monkeypatch.setattr(handson_adapter, "load_payload", lambda *args: payload)
    states, actions, action_mask, state_mask = ds._load_actions_and_states(
        {"action_schema": "handson-stage2/raw54-v1", "tjwj_path": "raw", "handson_action_frames": 70},
        "unused", 0, 3, 6, 3, False,
    )
    assert state_mask[0].tolist() == [True, False]  # state[phase], not generic state[phase-1]
    assert action_mask[:24].tolist() == [[True, False]] * 24
    assert states.shape == (3, 58) and actions.shape == (72, 58)


@pytest.mark.parametrize("start,phase", [(0, 0), (0, 3), (1, 0), (1, 3)])
def test_masks_follow_physical_action_rows_and_previous_observation_anchor(monkeypatch, start, phase):
    ds = partial_dataset()
    ds.mask_invalid_hand_tracking = True
    raw = raw54()
    raw[::3, 23:27] = 0
    raw[1::4, 50:54] = 0
    monkeypatch.setattr(module, "smart_load_pt", lambda *args, **kwargs: raw)
    states, actions, action_mask, state_mask = ds._load_actions_and_states(
        {"tjwj_path": "raw"}, "unused", start, 4, 6, phase, False
    )
    valid = torch.stack([raw[:, 23:27].norm(dim=-1) > 0, raw[:, 50:54].norm(dim=-1) > 0], -1)
    expected_actions, expected_states = [], []
    for latent in range(start, start + 4):
        begin = phase if latent == 0 else phase + 1 + (latent - 1) * 24
        end = begin + 1 if latent == 0 else begin + 24
        ids = list(range(min(begin, 69), min(end, 70)))
        ids += [ids[-1]] * (24 - len(ids))
        expected_actions.extend(ids)
        expected_states.append(max(min(begin, 69) - 1, 0))
    assert torch.equal(action_mask, valid[expected_actions])
    assert torch.equal(state_mask, valid[expected_states])
    assert actions.shape == (96, 58) and states.shape == (4, 58)


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("relative_limit", [None, 5.0])
def test_full_getitem_masks_survive_window_padding_bootstrap_and_collation(tmp_path, bootstrap, relative_limit):
    raw = raw54(112)
    raw[0:30, 23:27] = 0
    raw[105:, 23:27] = 0
    raw[0:30, 50:54] = 0
    raw[106:, 50:54] = 0
    torch.save(raw, tmp_path / "tjwj.pt")
    torch.save({"latents": torch.zeros(48, 6, 2, 2), "frame_stride": 6}, tmp_path / "latent.pt")
    torch.save(torch.ones(3, 4096), tmp_path / "language.pt")
    (tmp_path / "stats.json").write_text(json.dumps({"min": [0]*40, "max": [1]*40}))
    (tmp_path / "index.jsonl").write_text(json.dumps({"dataset": "fixture", "task": "task",
        "episode_bos_path": str(tmp_path), "total_latent_frames": 6}) + "\n")
    kwargs = dict(jsonl_path=str(tmp_path / "index.jsonl"), use_index_cache=False,
        episode_index_source="jsonl", latent_root=None, latent_subpath="latent.pt",
        lang_root=None, lang_subpath="language.pt", latent_phases=[0], max_latent_window=8,
        action_mode="qpos20_rel6d", action_stats_path=str(tmp_path / "stats.json"),
        qpos_normalization="min_max", action={"raw_hz":30,"train_hz":30}, use_state=False,
        bootstrap_start_window=bootstrap, bootstrap_chunk_size=2, samples_per_episode=1,
        wrist_xyz_relative_max=relative_limit)
    legacy = PretrainDataset(**kwargs)[0]
    sample = PretrainDataset(**kwargs, mask_invalid_hand_tracking=True)[0]
    assert legacy is not None and sample is not None
    assert "action_valid_mask" not in legacy and "state_valid_mask" not in legacy
    for key in ("actions", "states", "latents"):
        assert torch.equal(sample[key], legacy[key]) and sample[key].dtype == legacy[key].dtype
    assert sample["valid_len"] == 6 + int(bootstrap)
    assert sample["action_valid_mask"].shape == (192, 2)
    assert sample["state_valid_mask"].shape == (8, 2)
    # The first predicted block contains missing observations and has a missing anchor.
    assert not sample["action_valid_mask"][48, 0]
    assert not sample["state_valid_mask"][2, 0]
    assert not sample["action_valid_mask"][-1].any()  # repeated missing final action
    assert sample["state_valid_mask"][-1].all()  # previous-observation anchor is row 96
    packed = ar_collate_fn([sample, sample])
    assert packed["action_valid_mask"].dtype == torch.bool
    assert packed["action_valid_mask"].shape == (2, 192, 2)
    assert packed["state_valid_mask"].shape == (2, 8, 2)


def test_dataset_factory_forwards_opt_in_mask(monkeypatch):
    from omegaconf import OmegaConf
    recorded = {}
    monkeypatch.setattr(module, "PretrainDataset", lambda **kwargs: recorded.update(kwargs))
    create_dataset(OmegaConf.create({"dataset":{"type":"pretrain","mask_invalid_hand_tracking":True}}))
    assert recorded["mask_invalid_hand_tracking"] is True


def test_origin_identity_wrist_is_a_lost_track_only_when_enabled():
    """chaowei writes (0,0,0) + identity rotation for a lost track. Its unit
    quaternion passes the norm test, so it must be recognised explicitly."""
    import torch
    from data.pretrain.pretrain_dataset import PretrainDataset

    count = 6
    xyz = torch.tensor([[0.3, 0.1, 0.5]] * count)
    quat = torch.tensor([[0.1, 0.2, 0.3, 0.927]] * count)  # a real rotation
    left_xyz, left_q = xyz.clone(), quat.clone()
    left_xyz[1] = 0.0; left_q[1] = torch.tensor([0.0, 0.0, 0.0, 1.0])    # xyzw identity
    left_xyz[2] = 0.0; left_q[2] = torch.tensor([0.0, 0.0, 0.0, -1.0])   # same rotation, negated
    left_xyz[3] = 0.0; left_q[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])    # wxyz identity
    left_xyz[4] = 0.0                                                    # origin, real rotation: kept
    left_q[5] = torch.tensor([0.0, 0.0, 0.0, 1.0])                       # identity, real position: kept

    plain = PretrainDataset._hand_tracking_valid_mask(None, count, left_q, quat, left_xyz, xyz, False)
    assert plain.all(), 'without the flag the norm test alone decides, as before'

    masked = PretrainDataset._hand_tracking_valid_mask(None, count, left_q, quat, left_xyz, xyz, True)
    assert masked[:, 0].tolist() == [True, False, False, False, True, True]
    assert masked[:, 1].all(), 'the other hand is untouched'
