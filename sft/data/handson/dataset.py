"""HandsOn native commands, optionally encoded for the existing Wuji58 expert."""
import json
import hashlib
from pathlib import Path

import torch

from ..robodojo.dataset import RoboDojoDataset


class HandsOnDataset(RoboDojoDataset):
    action_dim = 52

    def __init__(self, manifest_path, *, normalization=None, language_cache=None,
                 language_seed=0, action_mode='abs_eef52', action_stats_path=None,
                 hand_normalization='min_max', command_alignment='native_next_command', **kwargs):
        if command_alignment not in {'native_next_command', 'stage2'}:
            raise ValueError('unsupported HandsOn command_alignment')
        self.command_alignment = command_alignment
        self.language_path = Path(language_cache) if language_cache else None
        self.language_seed = int(language_seed)
        self.language_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.language = json.loads(self.language_path.read_text()) if self.language_path else None
        if self.language and (self.language['status'] != 'complete'
                              or self.language['canonical_probability'] != 0.2
                              or self.language['unseen_encoded']):
            raise ValueError('invalid official language cache')
        if isinstance(normalization, str):
            normalization = json.loads(Path(normalization).read_text())
        self.codec = None
        if action_mode == 'qpos20_rel6d':
            from .action_codec import Wuji58Codec
            if (normalization or {}).get('type', 'none') != 'none':
                raise ValueError('58D uses Wuji hand stats, not native 52D normalization')
            self.codec = Wuji58Codec(action_stats_path, normalization=hand_normalization)
            if (self.codec.manifest_sha256 is not None
                    and self.codec.manifest_sha256 != hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()):
                raise ValueError('Wuji58 statistics belong to a different HandsOn manifest')
            window = self.codec.stat_window
            if window and (int(window.get('relative_chunk_size', 2)) != int(kwargs.get('bootstrap_chunk_size', 2))
                           or int(window.get('actions_per_latent', 24)) != 4 * int(kwargs.get('video_action_freq_ratio', 6))):
                raise ValueError('Wuji58 statistics have different chunk/stride anchors')
            self.action_dim = 58
            # Relative actions need measured anchors even without a state token.
            kwargs['use_state'] = True
        elif action_mode != 'abs_eef52':
            raise ValueError('unsupported HandsOn model action layout')
        super().__init__(manifest_path, normalization=normalization, **kwargs)

    def set_epoch(self, epoch):
        self.language_epoch.fill_(int(epoch))

    def _control_block(self, actions, states, latent, real):
        if self.command_alignment != 'stage2':
            return super()._control_block(actions, states, latent, real)
        apl = self.actions_per_latent
        if latent == 0:
            return actions[:1].expand(apl, -1), states[0], torch.zeros(apl, dtype=torch.bool)
        begin = 1 + (latent - 1) * apl
        # Stage2 handson_adapter.gather trains repeated-last-command tail rows.
        return actions[begin:begin + apl], states[begin - 1], torch.ones(apl, dtype=torch.bool)

    def _frame_is_real(self, latent, last_frame):
        if self.command_alignment == 'stage2':
            # Every stored VAE latent, including its held-frame tail, is real input.
            return True
        return super()._frame_is_real(latent, last_frame)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.language:
            entry = self.episodes[index]
            source = json.loads((Path(entry['episode_path']) / 'source_receipt.json').read_text())['source']
            matches = [row for row in self.language['prompts'].values()
                       if row['task'] == entry['task'] and row['canonical'] == source['prompt']]
            if len(matches) != 1:
                raise ValueError('official language goal mismatch')
            row = matches[0]
            split = 'canonical' if self.val else 'train'
            frame = entry['phase'] + result['latent_start'] * self.actions_per_latent
            text = sample_prompt(row, self.language_seed, int(self.language_epoch),
                                 entry['source_episode'], frame, split)
            result['language_embedding'] = torch.load(
                self.language_path.parent / self.language['files'][text],
                map_location='cpu', weights_only=True)
        return result

    def _check_contract(self, video_action_freq_ratio):
        if (self.contract.get('source_format') != 'handson-eef52'
                or self.contract.get('schema_version') != 2
                or self.contract.get('action_type') != 'world_palm_command_eef52_v1'
                or self.contract.get('rotation') != 'rotation_vector_radians'
                or self.contract.get('side_order') != ['right', 'left']
                or self.contract.get('frequency') != 30
                or self.contract.get('frame_stride') != video_action_freq_ratio
                or video_action_freq_ratio != 6):
            raise ValueError('unsupported HandsOn native command contract')

    def _load_bundle(self, entry):
        root = Path(entry['episode_path'])
        receipt = json.loads((root / 'episode_manifest.json').read_text())
        if receipt['action_schema'] != self.contract['action_type'] or not receipt['state_action_separate']:
            raise ValueError('HandsOn state/command semantics changed')
        payload = torch.load(entry['feature_path'], map_location='cpu', weights_only=True)
        phase = entry['phase']
        expected_ids = list(range(phase, receipt['n_video'], 6))
        if list(payload['video_frame_ids']) != expected_ids or payload['frame_stride'] != 6:
            raise ValueError('phase frame alignment changed')
        actions = torch.load(root / 'native_actions.pt', map_location='cpu', weights_only=True)
        states = torch.load(root / 'native_states.pt', map_location='cpu', weights_only=True)
        if actions.shape != states.shape or actions.shape != (receipt['n_video'], 52):
            raise ValueError('invalid native action/state dimensions')
        actions, states = actions[phase:], states[phase:]
        real = len(actions)
        needed = (entry['num_latents'] - 1) * self.actions_per_latent + self.actions_per_latent
        if real < needed:
            actions = torch.cat([actions, actions[-1:].expand(needed - real, -1)])
            states = torch.cat([states, states[-1:].expand(needed - real, -1)])
        text = torch.load(root / 'text_features/UMT5/instruction.pt', map_location='cpu', weights_only=True)
        latents = payload['latents']
        if latents.shape[0] != 48 or tuple(latents.shape[2:]) != (self.contract['video_height'] // 16,
                                                               self.contract['video_width'] // 16):
            raise ValueError('unexpected HandsOn latent canvas')
        if not all(torch.isfinite(value).all() for value in (latents, actions, states, text)):
            raise ValueError('nonfinite HandsOn feature')
        if self.codec is not None:
            actions, states = self.codec.encode(actions), self.codec.encode(states)
        return dict(contract=self.contract, latents=latents, actions=actions,
                    states=states, language_embedding=text, num_real_frames=real)


def sample_prompt(row, seed, epoch, episode_uid, frame_index, split):
    if split not in ('train', 'canonical'):
        raise ValueError('held-out paraphrases cannot enter training')
    key = json.dumps(['v16-language-v1', seed, epoch, str(episode_uid),
                      int(frame_index), 0, split], separators=(',', ':')).encode()
    value = hashlib.sha256(key).digest()
    if split == 'canonical' or int.from_bytes(value[:8], 'big') / 2**64 < 0.2:
        return row['canonical']
    return row['train'][int.from_bytes(value[8:16], 'big') % len(row['train'])]
