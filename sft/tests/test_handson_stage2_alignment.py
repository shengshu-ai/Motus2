import json

import pytest
import torch

from data.handson.dataset import HandsOnDataset


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



def test_stage2_alignment_keeps_anchor_and_uses_next_command(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    dataset = HandsOnDataset(manifest, val=True, max_latent_window=5,
        video_action_freq_ratio=6, command_alignment='stage2')
    sample = dataset[0]
    assert sample['actions'][:24, 0].tolist() == [12.] * 24
    assert sample['actions'][24:48, 0].tolist() == list(range(13, 37))
    assert sample['actions'][48:72, 0].tolist() == list(range(37, 61))
    assert sample['actions'][72:96, 0].tolist() == list(range(61, 73)) + [72.] * 12
    assert sample['states'][:4, 0].tolist() == [1012., 1012., 1036., 1060.]
    assert not sample['action_valid_mask'][:24].any()
    assert sample['action_valid_mask'][24:96].all()
    assert not sample['action_valid_mask'][96:].any()
    assert sample['video_valid_mask'].tolist() == [True, True, True, True, False]


def test_stage2_alignment_bootstrap_duplicates_observed_command(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    dataset = HandsOnDataset(manifest, val=True, max_latent_window=6,
        video_action_freq_ratio=6, command_alignment='stage2',
        bootstrap_start_window=True, bootstrap_chunk_size=2)
    sample = dataset[0]
    assert sample['actions'][:48, 0].tolist() == [12.] * 48
    assert sample['actions'][48:72, 0].tolist() == list(range(13, 37))
    assert sample['states'][2, 0] == 1012.
    assert not sample['action_valid_mask'][:48].any()


def test_unknown_command_alignment_is_rejected(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match='command_alignment'):
        HandsOnDataset(manifest, command_alignment='guess')
