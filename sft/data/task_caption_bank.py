"""Versioned task captions, lazy mmap tensors and outcome-independent draws."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class TaskCaptionBank:
    """Only task + RNG choose text. No episode text or outcome fallback exists."""

    def __init__(self, path, *, cache_size=2):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        data = json.loads(self.path.read_text())
        if data.get('version') not in (2, 3) or data.get('state') != 'released':
            raise ValueError('Task caption bank must be a released version 2 or 3 manifest')
        if data.get('embedding_dim') != 4096:
            raise ValueError('Task caption bank requires UMT5 dimension 4096')
        self.entries = data['canonical_keys']
        self.task_map = data['task_to_canonical']
        self.index = data['index']
        if not self.entries or not self.task_map or set(self.task_map.values()) - set(self.entries):
            raise ValueError('Invalid caption task coverage')
        self.cache_size = cache_size
        self._cache = OrderedDict()
        for key, entry in self.entries.items():
            self._relative(entry['file'])
            if not isinstance(key, str) or not key:
                raise ValueError('Empty canonical task')
            pending = data['version'] == 3 and entry.get('l2_status') == 'pending'
            if data['version'] == 3:
                if entry.get('l2_status') not in ('ready', 'partial_sources', 'pending'):
                    raise ValueError(f'{key}: invalid L2 availability')
                if pending and (entry['L2'] or entry['L3']):
                    raise ValueError(f'{key}: pending L2 must not contain L2 or L3')
            all_ids = []
            for level in ('L1', 'L2', 'L3'):
                rows = entry[level]
                if not rows and not (pending and level in ('L2', 'L3')):
                    raise ValueError(f'{key}: missing {level}')
                for row in rows:
                    text = row['text']
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('Empty caption')
                    if hashlib.sha256(text.encode()).hexdigest() != row['text_sha256']:
                        raise ValueError('Caption text identity mismatch')
                    if not 1 <= row['tokens'] <= 512 or row.get('review_state') != 'accepted':
                        raise ValueError('Unreviewed or overlong caption')
                    all_ids.append(row['tensor_index'])
            if sorted(all_ids) != list(range(len(all_ids))):
                raise ValueError('Tensor indices must match every caption exactly once')

    def _relative(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Bank reference escapes release directory')
        return path

    def source_records(self, dataset_dir, validation_fraction, validation_seed, index_meta):
        """Pin source-loader eligibility so replacing text cannot add episodes."""
        if str(dataset_dir).rstrip('/') != self.index['dataset_dir'].rstrip('/'):
            raise ValueError('Caption bank index belongs to a different dataset')
        meta_path = self._relative(self.index['meta_file'])
        if file_sha256(meta_path) != self.index['meta_sha256']:
            raise ValueError('Caption bank index metadata changed')
        if json.loads(meta_path.read_text()) != index_meta:
            raise ValueError('Caption bank index does not match requested task and latent configuration')
        path = self._relative(self.index['file'])
        if file_sha256(path) != self.index['sha256']:
            raise ValueError('Caption bank index changed')
        index = json.loads(path.read_text())
        if (validation_fraction != index['validation_fraction'] or
                validation_seed != index['validation_seed']):
            raise ValueError('Caption bank source split differs from training split')
        return index['records']

    def _tensors(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        entry = self.entries[key]
        path = self._relative(entry['file'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError(f'{key}: embedding shard changed')
        payload = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
        tensors = payload['embeddings']
        rows = sorted((r for level in ('L1', 'L2', 'L3') for r in entry[level]),
                      key=lambda r:r['tensor_index'])
        if len(tensors) != len(rows) or payload['texts'] != [r['text'] for r in rows]:
            raise ValueError('Embedding shard text alignment mismatch')
        for row, tensor in zip(rows, tensors):
            if (not isinstance(tensor, torch.Tensor) or
                    tuple(tensor.shape) != (row['tokens'], 4096) or
                    tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all()):
                raise ValueError('Invalid UMT5 tensor')
        self._cache[key] = tensors
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensors

    def draw(self, key, rng, *, val=False, environment_mix_prob=0.0):
        entry = self.entries[key]
        level = 'L1' if val or not entry['L2'] else rng.choice(('L1', 'L2'))
        rows = entry[level]
        row = rows[0] if val else rng.choice(rows)
        tensors = self._tensors(key)
        instruction = tensors[row['tensor_index']].to(torch.float32, copy=True)
        trace = {'level':level, 'text':row['text'], 'scene':None,
                 'instruction_tokens':row['tokens'], 'scene_tokens':0}
        if not val and entry['L3'] and rng.random() < environment_mix_prob:
            scene = rng.choice(entry['L3'])
            budget = 512 - instruction.shape[0]
            if budget > 0:
                environment = tensors[scene['tensor_index']][:budget].to(torch.float32)
                instruction = torch.cat((environment, instruction))
                trace.update(scene=scene['text'], scene_tokens=len(environment))
        return instruction, trace

    def __getstate__(self):
        # Spawned workers reopen shards rather than copying a parent's tensors.
        return {**self.__dict__, '_cache':OrderedDict()}
