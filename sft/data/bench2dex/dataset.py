"""读取 process_task.sh bench2dex 分支产物；不在线重编码、不替换监督标签。

采样、掩码与归一化完全复用 RoboDojo 的实现，差别只在三处：manifest 覆盖多个
embodiment 且动作维度各不相同，因此必须先选定一个 embodiment；split 的原子是轨迹
而不是文件；触觉不在 bundle 里，按需从源 HDF5 读 bundle 记录的 source_frame_ids。
"""
import json
from pathlib import Path

import torch

from ..robodojo.dataset import RoboDojoDataset


class Bench2DexDataset(RoboDojoDataset):
    def __init__(self, manifest_path, *, robot_key=None, with_tactile=False,
                 tactile_sites=None, **kwargs):
        document = json.loads(Path(manifest_path).read_text())
        contract = document['contract']
        embodiments = contract.get('embodiments') or {}

        # 同一个 manifest 里 34/36/46/48/52/58/60 七种维度共存，跨 embodiment 的
        # 关节语义不同，不选定就没有一个自洽的 action_dim。
        present = sorted({e['robot_key'] for e in document['episodes']})
        if robot_key is None:
            if len(present) != 1:
                raise ValueError(
                    f'manifest spans {len(present)} embodiments {present}; '
                    'pass robot_key to select one')
            robot_key = present[0]
        if robot_key not in embodiments:
            raise ValueError(f'unknown robot_key {robot_key}; manifest has {sorted(embodiments)}')
        self.robot_key = robot_key
        self.action_dim = int(embodiments[robot_key]['action_dim'])
        self.joint_names = list(embodiments[robot_key]['joint_names'])
        self.with_tactile = bool(with_tactile)
        self.tactile_sites = list(tactile_sites) if tactile_sites else None

        super().__init__(manifest_path, **kwargs)

        self.episodes = [e for e in self.episodes if e['robot_key'] == robot_key]
        if not self.episodes:
            raise ValueError(f'no episodes for {robot_key} in the selected split/task')
        # 轨迹是 split 的原子；同一轨迹的多个视觉重放不得跨 split。
        if len({e['trajectory'] for e in self.episodes}) > len(self.episodes):
            raise ValueError('trajectory/episode bookkeeping is inconsistent')

    def _check_contract(self, video_action_freq_ratio):
        if self.contract.get('source_format') != 'bench2dex-replay-hdf5':
            raise ValueError('not a Bench2Dex manifest')
        if self.contract.get('schema_version') != 1:
            raise ValueError('unsupported Bench2Dex manifest schema')
        if self.contract['frame_stride'] != video_action_freq_ratio:
            raise ValueError('latent stride differs from training config')
        if self.contract.get('action_type') != 'absolute-joint-position':
            raise ValueError('expected absolute joint position actions')
        if self.contract.get('action_layout') != 'native-per-embodiment':
            raise ValueError('unexpected action layout')

    def read_tactile(self, entry, frame_ids):
        """按帧读官方 240x240 触觉图；bundle 只存引用，不复制像素。"""
        import h5py
        import numpy as np

        source = entry.get('tactile_source') or {}
        sites = self.tactile_sites or source.get('sites') or []
        if not sites:
            raise ValueError('episode carries no tactile sites')
        dataset = source.get('dataset', 'robot/tactile/tacmap')
        with h5py.File(source['hdf5_path'], 'r') as f:
            maps = [np.stack([f[f'{dataset}/{site}'][t] for site in sites]) for t in frame_ids]
        # [frames, sites, 1, H, W] uint8，与官方 ACT-Tactile 的单通道约定一致。
        return torch.from_numpy(np.stack(maps)).unsqueeze(2)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if self.with_tactile:
            entry = self.episodes[idx]
            bundle = torch.load(entry['bundle_path'], map_location='cpu', weights_only=False)
            start = int(result['latent_start'])
            width = int(result['valid_len'])
            ids = bundle['tactile_source']['frame_ids'][start:start + width]
            result['tactile'] = self.read_tactile(bundle, ids)
        result['robot_key'] = self.robot_key
        return result
