"""Exclude labelled failures without pretending unlabelled human videos succeeded."""
from pathlib import Path
import json
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import create_dataset
from data.pretrain import pretrain_dataset as module
from data.pretrain.pretrain_dataset import PretrainDataset


@pytest.mark.parametrize('entry', [
    {'fail': True}, {'is_fail': 1}, {'is_fail_class': 'true'},
    {'failed': 'yes'}, {'is_failure': '1.0'},
    {'success': False}, {'is_success': 0}, {'episode_success': 0.0},
    {'task_success': 'false'}, {'status': 'FAILED'}, {'outcome': 'failure'},
    {'result': 'fail'}, {'sample_type': 'fail'},
    {'episode_bos_path': '/dataset/task/fail/episode_1'},
    {'episode_path': 'bos://bucket/task/Failed/episode_1'},
    {'success': True, 'is_fail': True},
])
def test_explicit_failure_is_recognized(entry):
    assert PretrainDataset._explicit_failure_reason(entry)


@pytest.mark.parametrize('entry', [
    {}, {'success': None}, {'success': True}, {'is_fail': False},
    {'status': 'unknown'}, {'status': 'running'}, {'sample_class': 0},
    {'sample_class': 1}, {'task': 'failsafe grasping'},
    {'episode_bos_path': '/dataset/task_fails_sometimes/episode_1'},
])
def test_absent_unknown_and_routing_labels_are_not_outcomes(entry):
    assert PretrainDataset._explicit_failure_reason(entry) is None


def dataset():
    ds = PretrainDataset.__new__(PretrainDataset)
    ds.exclude_explicit_fail = True
    ds._explicit_fail_seen = {}
    ds._explicit_fail_excluded = {}
    return ds


def test_failure_is_excluded_before_paths_or_tensors_are_opened():
    ds = dataset()
    assert ds._episode_from_jsonl_entry({'dataset': 'A', 'is_success': False}) == []
    assert ds._explicit_fail_seen == {'A': 1}
    assert ds._explicit_fail_excluded == {'A': 1}


def test_unlabelled_record_is_retained_by_the_real_episode_conversion():
    ds = dataset()
    for name, value in {
        'robot_filter': None, 'min_total_latent_frames': 4,
        'dataset_include': [], 'dataset_exclude': [],
        'task_mode': 'multi', 'task_name': None, 'action_mode': 'qpos20_rel6d',
        'latent_root': 'video_features/wan2.2', 'latent_subpath': None,
        'lang_root': 'text_features/UMT5', 'lang_subpath': None,
        'stereo_latent_sources': [], 'stereo_latent_name_pairs_by_dataset': {},
        'stereo_latent_name_pairs': [], 'latent_file_name_by_dataset': {},
        'stereo_left_latent_name': None, 'stereo_right_latent_name': None,
    }.items():
        setattr(ds, name, value)
    entry = {'dataset': 'A', 'task': 'failsafe_grasp',
             'episode_bos_path': '/dataset/failsafe_grasp/episode_1',
             'total_latent_frames': 8}
    assert len(ds._episode_from_jsonl_entry(entry)) == 1
    assert ds._explicit_fail_excluded == {}
    ds.exclude_explicit_fail = False
    assert len(ds._episode_from_jsonl_entry({**entry, 'success': False})) == 1


def test_factory_really_forwards_the_policy(monkeypatch):
    monkeypatch.setattr(module, 'PretrainDataset', lambda **kwargs: kwargs)
    cfg = OmegaConf.create({'dataset': {'type': 'pretrain', 'exclude_explicit_fail': True}})
    assert create_dataset(cfg)['exclude_explicit_fail'] is True


def test_old_cache_cannot_bypass_the_new_policy(tmp_path):
    ds = dataset()
    ds.index_cache_path = str(tmp_path / 'index.json')
    ds._index_cache_meta = lambda: {'exclude_explicit_fail': True}
    for payload in ([{'episode_name': 'legacy'}], {'episodes': []}, {'meta': {}, 'episodes': []}):
        Path(ds.index_cache_path).write_text(json.dumps(payload))
        assert ds._load_index_payload(silent=True) is None
    payload = {'meta': {'exclude_explicit_fail': True}, 'episodes': []}
    Path(ds.index_cache_path).write_text(json.dumps(payload))
    assert ds._load_index_payload(silent=True) == payload


def test_post_filter_source_weights_keep_configured_relative_mixture():
    ds = dataset()
    ds.source_weighted_sampling = True
    ds.stereo_latent_sources = [{'name': 'A', 'ratio': 15}, {'name': 'B', 'ratio': 10}]
    records = [
        {'dataset_name': 'A'}, {'dataset_name': 'A', 'is_fail': True},
        {'dataset_name': 'A'}, {'dataset_name': 'B'},
    ]
    ds.episode_files = [ep for ep in records if not ds._explicit_failure_reason(ep)]
    ds._configure_source_sampling_weights()
    weights = list(ds._episode_sample_weights)
    assert weights == pytest.approx([7.5, 7.5, 10])
    total = sum(weights)
    assert sum(weights[:2]) / total == pytest.approx(0.6)
    assert weights[-1] / total == pytest.approx(0.4)
