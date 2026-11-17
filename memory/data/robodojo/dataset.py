"""读取 process_task.sh RoboDojo 分支产物；不在线重编码、不替换监督标签。"""
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from .normalization import JointNormalizer


def real_extent(bundle, num_actions, stride):
    """Real action count and last real sampled frame of an encoded episode.

    Schema 2 completes the final frame group with the last sampled frame and holds the
    final action past the recording; schema 1 keeps only complete groups, so there
    every stored row and frame is real.
    """
    schema = bundle['contract']['schema_version']
    if schema == 2 and 'num_real_frames' not in bundle:
        raise ValueError('schema 2 episode lacks num_real_frames')
    real = int(bundle['num_real_frames']) if schema == 2 else num_actions
    return real, ((real - 1) // stride) * stride


class RoboDojoDataset(Dataset):
    def __init__(self, manifest_path, *, val=False, task_name=None,
                 max_latent_window=14, video_action_freq_ratio=5, use_state=True,
                 sequence_mode='window', bootstrap_start_window=False, bootstrap_chunk_size=2,
                 video_height=None, video_width=None, normalization=None, **kwargs):
        if sequence_mode not in {'window', 'episode'}:
            raise ValueError('sequence_mode must be window or episode')
        document = json.loads(Path(manifest_path).read_text())
        self.contract = document['contract']
        self.normalizer = JointNormalizer(normalization)
        if normalization and normalization.get('manifest_sha256'):
            import hashlib
            if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != normalization['manifest_sha256']:
                raise ValueError('normalization statistics belong to a different manifest')
        if self.contract.get('schema_version') not in (1, 2) or self.contract.get('embodiment') != 'arx_x5':
            raise ValueError('unsupported RoboDojo manifest')
        if self.contract['frame_stride'] != video_action_freq_ratio:
            raise ValueError('latent stride differs from training config')
        if self.contract['action_type'] != 'joint' or self.contract['frequency'] != 25:
            raise ValueError('expected absolute joint actions at 25 Hz')
        self.episodes = [e for e in document['episodes']
                         if e['split'] == ('val' if val else 'train')
                         and (task_name is None or e['task'] == task_name)]
        if not self.episodes:
            raise ValueError('no prepared episodes in selected split/task')
        ids = [e['id'] for e in document['episodes']]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate episode identity in manifest')
        self.actions_per_latent = 4 * video_action_freq_ratio
        self.window = int(max_latent_window)
        if self.window < 2:
            raise ValueError('need a context latent and at least one target latent')
        self.bootstrap_start_window = bootstrap_start_window
        self.bootstrap_chunk_size = bootstrap_chunk_size
        for key, expected in [('video_height', video_height), ('video_width', video_width)]:
            if expected is not None and self.contract.get(key) != expected:
                raise ValueError(f'{key} differs from encoded canvas')
        self.sequence_mode = sequence_mode
        self.use_state = use_state
        self.val = val
        self.total_episodes = len(self.episodes)

    def __len__(self):
        return len(self.episodes)

    def get_sample_lengths(self):
        return [e['num_latents'] if self.sequence_mode == 'episode'
                else self.window for e in self.episodes]

    def __getitem__(self, idx):
        entry = self.episodes[idx]
        bundle = torch.load(entry['bundle_path'], map_location='cpu', weights_only=True)
        if bundle['contract'] != self.contract:
            raise ValueError('episode encoding contract differs from manifest')
        latents = bundle['latents']
        actions = bundle['actions'].float()
        states = bundle['states'].float()
        text = bundle['language_embedding']
        n = latents.shape[1]
        if n != entry['num_latents'] or n < 2:
            raise ValueError('invalid latent count')
        if latents.ndim != 4 or actions.shape != states.shape or actions.shape[1:] != (14,):
            raise ValueError('invalid tensor shapes')
        if text.ndim != 2 or text.shape[1] != 4096:
            raise ValueError('expected UMT5 [tokens,4096]')
        apl = self.actions_per_latent
        if (n - 1) * apl > len(actions) - 1:
            raise ValueError('latents extend beyond complete action blocks')
        width = n if self.sequence_mode == 'episode' else self.window
        start = 0 if self.val or n <= width else random.randint(0, n - width)
        valid = min(width, n - start)
        video = latents[:, start:start + valid]
        real, last_frame = real_extent(bundle, len(actions), apl // 4)
        blocks, anchors, masks, frames_real = [], [], [], []
        for latent in range(start, start + valid):
            # L1 covers frames stride..4*stride; its controls are action[0:4*stride].
            # L0 is observed context and has no supervised action block.
            begin = max(0, (latent - 1) * apl)
            block = actions[begin:begin + apl]
            blocks.append(block)
            anchors.append(states[begin])
            masks.append((torch.arange(begin, begin + apl) < real) & (latent > 0))
            frames_real.append(latent * apl <= last_frame)
        if valid < width:
            video = torch.cat([video, video[:, -1:].expand(-1, width - valid, -1, -1)], dim=1)
            for _ in range(width - valid):
                blocks.append(blocks[-1])
                anchors.append(anchors[-1])
                masks.append(torch.zeros(apl, dtype=torch.bool))
                frames_real.append(False)
        result = {'latents': video, 'actions': torch.cat(blocks),
                  'language_embedding': text, 'valid_len': valid,
                  'latent_start': start, 'action_valid_mask': torch.cat(masks),
                  'sample_class': 0, 'is_success': 1}
        if self.use_state:
            result['states'] = torch.stack(anchors)
        frames_real = torch.tensor(frames_real)
        if self.bootstrap_start_window:
            from utils.common import maybe_bootstrap_start_window_sample
            # Per-latent frame validity follows the same prepend as per-latent values.
            video, controls, anchor, frames_real, valid, extra = maybe_bootstrap_start_window_sample(
                result['latents'], result['actions'], valid, start,
                actions_per_latent=apl, states=result.get('states'), values=frames_real,
                chunk_size=self.bootstrap_chunk_size, enabled=True,
                extra_action_sequences={'mask': result['action_valid_mask']},
            )
            result.update(latents=video, actions=controls, valid_len=valid,
                          action_valid_mask=extra['mask'])
            if anchor is not None:
                result['states'] = anchor
        # A latent whose frames include the repeated tail frame has no real video target.
        result['video_valid_mask'] = (torch.arange(width) < valid) & frames_real.bool()
        result['anchor_latents'] = latents[:, :1].clone()
        result['is_task_irrelevant'] = 0
        if self.sequence_mode == 'episode':
            result.update(episode_num_latents=n, sequence_num_latents=width,
                          sequence_chunk_trimmed_latents=0)
        result['actions'] = self.normalizer.transform(result['actions'], 'action')
        if 'states' in result:
            result['states'] = self.normalizer.transform(result['states'], 'state')
        return result
