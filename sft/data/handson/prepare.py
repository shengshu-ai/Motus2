"""按官方 session split 建索引，仅从训练集计算原生命令和观测的独立统计。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


PHASES = (0, 1, 2, 3, 4, 5, 6, 12, 18)
VIDEO_NAME = 'video_concat_mono_544_640_VAE.pt'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(raw, output, destination, task=None, exclude=()):
    """One task, or (task=None) every source task except `exclude`, with shared train-split statistics."""
    source = Path(raw) / 'source_manifest.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if set(exclude) - {row['source_task'] for row in rows}:
        raise ValueError('excluded task is not a source task')
    rows = [row for row in rows if (row['source_task'] == task if task else row['source_task'] not in exclude)]
    if not rows or {row['split'] for row in rows} != {'train', 'validation'}:
        raise ValueError('selection must have official train and validation episodes')
    sessions = {}
    contract = dict(schema_version=2, source_format='handson-eef52',
                    action_type='world_palm_command_eef52_v1',
                    rotation='rotation_vector_radians', side_order=['right', 'left'],
                    embodiment='rokae_wuji_world_palm', frequency=30, frame_stride=6,
                    video_height=544, video_width=640,
                    cameras=['front', 'wrist_left', 'wrist_right'])
    episodes = []
    tensors = {'action': [], 'state': []}
    for row in rows:
        session = row['collection_session_id']
        if sessions.setdefault(session, row['split']) != row['split']:
            raise ValueError('collection session crosses splits')
        root = Path(output) / row['split'] / row['source_task'] / f"episode_{row['episode_index']:06d}"
        meta = json.loads((root / 'episode_manifest.json').read_text())
        if meta['action_schema'] != contract['action_type'] or not meta['state_action_separate']:
            raise ValueError('native command/state contract mismatch')
        if row['split'] == 'train':
            for key, filename in [('action', 'native_actions.pt'), ('state', 'native_states.pt')]:
                value = torch.load(root / filename, map_location='cpu', weights_only=True)
                if value.shape != (meta['n_video'], 52) or not torch.isfinite(value).all():
                    raise ValueError('invalid native tensor')
                tensors[key].append(value)
        for phase in PHASES if row['split'] == 'train' else (0,):
            sampled = len(range(phase, meta['n_video'], 6))
            count = 1 + (sampled + 2) // 4
            if count < 2:
                raise ValueError('episode too short for the declared phase set')
            episodes.append(dict(id=f"{row['episode_uid']}:phase{phase}",
                source_episode=row['episode_uid'], collection_session_id=session,
                split='val' if row['split'] == 'validation' else 'train', task=row['source_task'],
                phase=phase, episode_path=str(root), num_latents=count,
                feature_path=str(root / f'video_features/wan2.2/{phase}phase_5hz' / VIDEO_NAME)))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'manifest.json'
    write_json(manifest, dict(contract=contract, episodes=episodes,
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        split='official collection session; validation phase zero only'))
    scope = task or sorted({row['source_task'] for row in rows})
    stats = dict(type='min_max', manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 scope=scope, excluded=sorted(exclude), split='train', episodes=len(tensors['action']))
    for key, values in tensors.items():
        combined = torch.cat(values).float()
        lower, upper = combined.amin(0), combined.amax(0)
        upper = torch.maximum(upper, lower + 1e-5)
        stats[key] = {'min': lower.tolist(), 'max': upper.tolist()}
    write_json(destination / 'action_stats.json', stats)
    write_json(destination / 'preparation.json', dict(status='indexed', policy_ready=False,
        blocker='all phase features and caption/text receipts must pass verification',
        task=scope, excluded=sorted(exclude), source_episodes=len(rows), phase_samples=len(episodes),
        manifest_sha256=stats['manifest_sha256']))


def verify(destination, language_cache=None):
    from .dataset import HandsOnDataset
    destination = Path(destination)
    counts = {}
    verified_episodes = set()
    for val in (False, True):
        dataset = HandsOnDataset(destination / 'manifest.json', val=val,
            normalization=str(destination / 'action_stats.json'),
            language_cache=language_cache,
            video_action_freq_ratio=6, video_height=544, video_width=640,
            max_latent_window=10, use_state=True)
        for entry in dataset.episodes:
            root = Path(entry['episode_path'])
            receipts = {}
            for name in ('vae_receipt.json', 'caption_receipt.json', 'text_receipt.json'):
                receipts[name] = json.loads((root / name).read_text())
                if receipts[name]['status'] != 'complete':
                    raise ValueError(f'incomplete receipt {root / name}')
            caption_hash = hashlib.sha256((root / 'instructions.json').read_bytes()).hexdigest()
            if (receipts['caption_receipt.json']['instructions_sha256'] != caption_hash
                    or receipts['text_receipt.json']['instructions_sha256'] != caption_hash
                    or receipts['caption_receipt.json']['source_episode'] != entry['source_episode']
                    or receipts['vae_receipt.json']['source_episode'] != entry['source_episode']
                    or receipts['vae_receipt.json']['phases'] != list(PHASES)):
                raise ValueError('caption/latent receipt lineage mismatch')
            if root not in verified_episodes:
                for filename in receipts['text_receipt.json']['files']:
                    text = torch.load(root / 'text_features/UMT5' / filename,
                                      map_location='cpu', weights_only=True)
                    if text.ndim != 2 or text.shape[1] != 4096 or not torch.isfinite(text).all():
                        raise ValueError('invalid caption text embedding')
                verified_episodes.add(root)
            bundle = dataset._load_bundle(entry)
            if bundle['latents'].shape[1] != entry['num_latents']:
                raise ValueError('latent count differs from indexed duration')
        for index in (0, len(dataset) - 1):
            sample = dataset[index]
            if sample['actions'].shape != (240, 52) or sample['states'].shape != (10, 52):
                raise ValueError('sample dimensions invalid')
        counts['validation' if val else 'train'] = len(dataset)
    write_json(destination / 'ready.json', dict(status='complete', policy_ready=True,
        samples=counts, action_mode='abs_eef52',
        language_cache=language_cache,
        language_bank_sha256=dataset.language['bank_sha256'] if dataset.language else None,
        training_language='official 20% canonical + 80% train paraphrases' if dataset.language else 'canonical only'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--raw')
    parser.add_argument('--output')
    parser.add_argument('--task', help='one source task; omit for every task except --exclude-task')
    parser.add_argument('--exclude-task', action='append', default=[])
    parser.add_argument('--destination', required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--language-cache')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.verify:
        verify(args.destination, args.language_cache)
    else:
        prepare(args.raw, args.output, args.destination, args.task, tuple(args.exclude_task))


if __name__ == '__main__':
    main()
