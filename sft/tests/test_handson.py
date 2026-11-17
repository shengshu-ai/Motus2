import json

import pytest
import torch
from omegaconf import OmegaConf

from data.dataset import create_dataset
from data.handson.dataset import HandsOnDataset
from data.handson.dataset import sample_prompt


def fixture_manifest(tmp_path, *, phase=12):
    root = tmp_path / 'episode'
    root.mkdir()
    (root / 'text_features/UMT5').mkdir(parents=True)
    (root / 'episode_manifest.json').write_text(json.dumps(dict(n_video=73,
        action_schema='world_palm_command_eef52_v1', state_action_separate=True)))
    actions = torch.arange(73).float()[:, None].expand(-1, 52).clone()
    torch.save(actions, root / 'native_actions.pt')
    torch.save(actions + 1000, root / 'native_states.pt')
    torch.save(torch.ones(3, 4096), root / 'text_features/UMT5/instruction.pt')
    feature = root / 'feature.pt'
    torch.save(dict(latents=torch.zeros(48, 4, 34, 40),
                    frame_stride=6, video_frame_ids=list(range(phase, 73, 6))), feature)
    contract = dict(schema_version=2, source_format='handson-eef52',
        action_type='world_palm_command_eef52_v1', rotation='rotation_vector_radians',
        side_order=['right', 'left'], frequency=30, frame_stride=6,
        video_height=544, video_width=640)
    entries = [dict(id=split, split=split, task='laptop', phase=phase,
                   num_latents=4, episode_path=str(root), feature_path=str(feature))
               for split in ('train', 'val')]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(contract=contract, episodes=entries)))
    return path, root


def test_phase_anchors_and_commands_are_separate_and_tail_is_masked(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    dataset = HandsOnDataset(manifest, val=True, max_latent_window=5,
                            video_action_freq_ratio=6)
    sample = dataset[0]
    assert sample['actions'].shape == (120, 52)
    assert sample['states'].shape == (5, 52)
    assert sample['actions'][24, 0] == 12
    assert sample['actions'][48, 0] == 36
    assert sample['states'][1, 0] == 1012
    assert sample['states'][2, 0] == 1036
    assert not sample['action_valid_mask'][:24].any()
    assert sample['action_valid_mask'][24:85].all()
    assert not sample['action_valid_mask'][85:].any()
    assert sample['video_valid_mask'].tolist() == [True, True, True, False, False]


def test_bootstrap_preserves_native_state_and_actions(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    dataset = HandsOnDataset(manifest, val=True, max_latent_window=6,
        video_action_freq_ratio=6, bootstrap_start_window=True, bootstrap_chunk_size=2)
    sample = dataset[0]
    assert sample['actions'].shape == (144, 52)
    assert sample['states'].shape == (6, 52)
    assert not sample['action_valid_mask'][:48].any()
    assert sample['actions'][48, 0] == 12


def test_frame_alignment_cannot_silently_drift(tmp_path):
    manifest, root = fixture_manifest(tmp_path)
    payload = torch.load(root / 'feature.pt', weights_only=True)
    payload['video_frame_ids'][0] += 1
    torch.save(payload, root / 'feature.pt')
    dataset = HandsOnDataset(manifest, video_action_freq_ratio=6)
    with pytest.raises(ValueError, match='frame alignment'):
        dataset[0]


def test_factory_requires_explicit_native_mode(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    config = OmegaConf.create(dict(dataset=dict(type='handson', manifest_path=str(manifest)),
        common=dict(max_latent_window=6, video_action_freq_ratio=6),
        model=dict(action_mode='joint', use_state=True, fixed_chunk_size=2)))
    with pytest.raises(ValueError, match='native abs_eef52'):
        create_dataset(config)
    config.model.action_mode = 'abs_eef52'
    assert isinstance(create_dataset(config), HandsOnDataset)


def test_statistics_and_phases_keep_official_split(tmp_path):
    from data.handson.prepare import prepare
    raw = tmp_path / 'raw'
    raw.mkdir()
    output = tmp_path / 'output'
    rows = []
    for index, split in enumerate(('train', 'validation')):
        root = output / split / 'laptop' / f'episode_{index:06d}'
        root.mkdir(parents=True)
        (root / 'episode_manifest.json').write_text(json.dumps(dict(n_video=73,
            action_schema='world_palm_command_eef52_v1', state_action_separate=True)))
        torch.save(torch.full((73, 52), float(index * 100)), root / 'native_actions.pt')
        torch.save(torch.full((73, 52), float(index * 100 + 2)), root / 'native_states.pt')
        rows.append(dict(source_task='laptop', split=split, episode_index=index,
                         collection_session_id=split, episode_uid=str(index)))
    (raw / 'source_manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    destination = tmp_path / 'training'
    prepare(raw, output, destination, 'laptop')
    stats = json.loads((destination / 'action_stats.json').read_text())
    entries = json.loads((destination / 'manifest.json').read_text())['episodes']
    assert stats['episodes'] == 1
    assert max(stats['action']['max']) < 1
    assert stats['state']['min'] == [2.0] * 52
    assert len(entries) == 10
    assert [entry['phase'] for entry in entries if entry['split'] == 'val'] == [0]


def test_official_sampler_matches_reference_golden_values():
    row = dict(canonical='canonical', train=[f'train-{index}' for index in range(32)])
    expected = ['train-29', 'train-24', 'train-15', 'train-10', 'train-31', 'train-12']
    assert [sample_prompt(row, 0, epoch, 'episode-a', 24, 'train')
            for epoch in range(6)] == expected
    assert sample_prompt(row, 0, 0, 'episode-a', 24, 'canonical') == 'canonical'
    with pytest.raises(ValueError, match='held-out'):
        sample_prompt(row, 0, 0, 'episode-a', 24, 'unseen')
    selected = [sample_prompt(row, 0, index, 'episode-a', 24, 'train') for index in range(1000)]
    assert 150 < selected.count('canonical') < 250
    assert set(selected) == {'canonical', *row['train']}


def test_native_layout_dimension_is_opt_in():
    from models.motus import MotusConfig
    assert MotusConfig(action_mode='abs_eef52').action_dim == 52
    assert MotusConfig(action_mode='abs_eef52').action_state_dim == 52
    assert MotusConfig(action_mode='joint').action_dim == 14

