"""读取 process_task.sh RoboDojo 分支产物；不在线重编码、不替换监督标签。"""
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from .normalization import JointNormalizer
from ..outcome_pool import configure_pool_sampling, pool_label, sample_pool_episode
from ..sample_class import SampleClass

# Contract fields every pooled manifest must share with the success manifest; the rest
# (schema tail rule, source format, rgb decoder, action alignment wording) legitimately
# differ per source and are read from each bundle's own contract.
SHARED_CONTRACT_KEYS = ('embodiment', 'action_type', 'frequency', 'frame_stride',
                        'video_height', 'video_width', 'cameras')
POOL_CLASS = {'success': int(SampleClass.MAIN), 'fail': int(SampleClass.TASK_IRRELEVANT),
              'random': int(SampleClass.RANDOM)}


def value_target_builder(value_target, progress_c, decaying_curve):
    """The Tianji/Wuji value-target methods, so every embodiment shares one definition."""
    from ..tianji_wuji.tianji_wuji_dataset import TianjiWujiDataset
    builder = TianjiWujiDataset.__new__(TianjiWujiDataset)
    builder.value_target = str(value_target)
    builder.progress_c = float(progress_c)
    builder.task_irrelevant_decaying_curve = bool(decaying_curve)
    builder.treat_as_irrelevant = False
    return builder


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
    # ARX X5 is the only RoboDojo embodiment; subclasses covering datasets with
    # other robots set this from their manifest instead.
    action_dim = 14

    def __init__(self, manifest_path, *, val=False, task_name=None,
                 max_latent_window=14, video_action_freq_ratio=5, use_state=True,
                 sequence_mode='window', bootstrap_start_window=False, bootstrap_chunk_size=2,
                 video_height=None, video_width=None, normalization=None,
                 extra_manifests=None, value_target=None, progress_c=0.01,
                 fail_sample_fraction=None, random_sample_fraction=None,
                 task_irrelevant_decaying_curve=False, include_primary=True, **kwargs):
        if sequence_mode not in {'window', 'episode'}:
            raise ValueError('sequence_mode must be window or episode')
        document = json.loads(Path(manifest_path).read_text())
        self.contract = document['contract']
        self.contracts = [self.contract]
        self.normalizer = JointNormalizer(normalization, expected_dim=self.action_dim)
        if normalization and normalization.get('manifest_sha256'):
            import hashlib
            if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != normalization['manifest_sha256']:
                raise ValueError('normalization statistics belong to a different manifest')
        self._check_contract(video_action_freq_ratio)
        self.episodes = [e for e in document['episodes']
                         if e['split'] == ('val' if val else 'train')
                         and (task_name is None or e['task'] == task_name)]
        ids = [e['id'] for e in document['episodes']]
        if any(e.get('pool', 'success') != 'success' for e in self.episodes):
            raise ValueError('the primary manifest must hold success demonstrations only')
        # include_primary=False keeps the primary manifest as the contract and
        # normalization anchor but trains only on the extra manifests' episodes.
        self.episodes = ([dict(e, pool='success', contract_index=0) for e in self.episodes]
                         if include_primary else [])
        for extra in extra_manifests or []:
            extra_doc = json.loads(Path(extra).read_text())
            contract = extra_doc['contract']
            mismatched = [k for k in SHARED_CONTRACT_KEYS if contract.get(k) != self.contract.get(k)]
            if mismatched:
                raise ValueError(f'{extra}: contract differs from the success manifest on {mismatched}')
            self.contracts.append(contract)
            selected = [e for e in extra_doc['episodes']
                        if e['split'] == ('val' if val else 'train')
                        and (task_name is None or e['task'] == task_name)]
            if any(e.get('pool') not in ('success', 'fail', 'random') for e in selected):
                raise ValueError(f'{extra}: pooled manifests hold success/fail/random episodes only')
            self.episodes += [dict(e, contract_index=len(self.contracts) - 1) for e in selected]
            ids += [e['id'] for e in extra_doc['episodes']]
        if not self.episodes:
            raise ValueError('no prepared episodes in selected split/task')
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate episode identity in manifest')
        self.pooled = len(self.contracts) > 1
        if value_target is not None and not self.pooled:
            raise ValueError('RoboDojo value targets need fail/random manifests to define negatives')
        self.values = (value_target_builder(value_target, progress_c, task_irrelevant_decaying_curve)
                       if value_target is not None else None)
        self.fail_sample_fraction = fail_sample_fraction
        self.random_sample_fraction = random_sample_fraction
        self.suboptimal_sample_fraction = None
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
        if self.pooled:
            configure_pool_sampling(self)

    def _check_contract(self, video_action_freq_ratio):
        if self.contract.get('schema_version') not in (1, 2) or self.contract.get('embodiment') != 'arx_x5':
            raise ValueError('unsupported RoboDojo manifest')
        if self.contract['frame_stride'] != video_action_freq_ratio:
            raise ValueError('latent stride differs from training config')
        if self.contract['action_type'] != 'joint' or self.contract['frequency'] != 25:
            raise ValueError('expected absolute joint actions at 25 Hz')

    def __len__(self):
        return len(self.episodes)

    def get_sample_lengths(self):
        prefix = max(0, int(self.bootstrap_chunk_size) - 1) if self.bootstrap_start_window else 0
        return [e['num_latents'] + prefix if self.sequence_mode == 'episode'
                else self.window for e in self.episodes]

    def _control_block(self, actions, states, latent, real):
        apl = self.actions_per_latent
        begin = max(0, (latent - 1) * apl)
        return (actions[begin:begin + apl], states[begin],
                (torch.arange(begin, begin + apl) < real) & (latent > 0))

    def _frame_is_real(self, latent, last_frame):
        return latent * self.actions_per_latent <= last_frame

    def __getitem__(self, idx):
        # Pooled training draws by pool fraction; the index only fixes epoch length.
        entry = sample_pool_episode(self) if self.pooled and not self.val else self.episodes[idx]
        bundle = self._load_bundle(entry)
        if bundle['contract'] != self.contracts[entry['contract_index']]:
            raise ValueError('episode encoding contract differs from manifest')
        pool = pool_label(entry)
        sample_class = POOL_CLASS[pool]
        latents = bundle['latents']
        actions = bundle['actions'].float()
        states = bundle['states'].float()
        text = bundle['language_embedding']
        n = latents.shape[1]
        if n != entry['num_latents'] or n < 2:
            raise ValueError('invalid latent count')
        if latents.ndim != 4 or actions.shape != states.shape or actions.shape[1:] != (self.action_dim,):
            raise ValueError('invalid tensor shapes')
        if text.ndim != 2 or text.shape[1] != 4096:
            raise ValueError('expected UMT5 [tokens,4096]')
        apl = self.actions_per_latent
        if (n - 1) * apl > len(actions) - 1:
            raise ValueError('latents extend beyond complete action blocks')
        prefix = max(0, int(self.bootstrap_chunk_size) - 1) if self.bootstrap_start_window else 0
        width = n + prefix if self.sequence_mode == 'episode' else self.window
        virtual_start = 0 if self.val or n + prefix <= width else random.randint(0, n + prefix - width)
        start = max(0, virtual_start - prefix)
        bootstrap_copies = max(0, prefix - virtual_start)
        valid = min(width, n - start)
        video = latents[:, start:start + valid]
        real, last_frame = real_extent(bundle, len(actions), apl // 4)
        blocks, anchors, masks, frames_real = [], [], [], []
        for latent in range(start, start + valid):
            # L1 covers frames stride..4*stride; its controls are action[0:4*stride].
            # L0 is observed context and has no supervised action block.
            block, anchor, mask = self._control_block(actions, states, latent, real)
            blocks.append(block)
            anchors.append(anchor)
            masks.append(mask)
            frames_real.append(self._frame_is_real(latent, last_frame))
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
                  'sample_class': sample_class, 'is_success': int(pool == 'success')}
        if self.use_state:
            result['states'] = torch.stack(anchors)
        frames_real = torch.tensor(frames_real)
        # Frame validity and value targets are both per latent and must shift together.
        per_latent = frames_real.float().unsqueeze(-1)
        if self.values is not None:
            targets = self.values._build_value_targets(start, width, n, sample_class=sample_class)
            per_latent = torch.cat([per_latent, targets.float().unsqueeze(-1)], dim=-1)
        if bootstrap_copies:
            from utils.common import maybe_bootstrap_start_window_sample
            video, controls, anchor, per_latent, valid, extra = maybe_bootstrap_start_window_sample(
                result['latents'], result['actions'], valid, start,
                actions_per_latent=apl, states=result.get('states'), values=per_latent,
                chunk_size=bootstrap_copies + 1, enabled=True,
                extra_action_sequences={'mask': result['action_valid_mask']},
            )
            result.update(latents=video, actions=controls, valid_len=valid,
                          action_valid_mask=extra['mask'])
            if anchor is not None:
                result['states'] = anchor
        frames_real = per_latent[:, 0] > 0.5
        if self.values is not None:
            result['values'] = per_latent[:, 1].clone()
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

    def _load_bundle(self, entry):
        return torch.load(entry['bundle_path'], map_location='cpu', weights_only=True)
