"""Dataset samplers shared by training and tests."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import BatchSampler


class SameResolutionBatchSampler(BatchSampler):
    """Build batches within a single resolution bucket.

    Bucket batch order is proportional to the number of samples in each bucket:
    every epoch shuffles all full per-bucket batches together, equivalent to
    sample-proportional training with the constraint that a batch cannot mix
    token grid sizes.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        buckets: Dict[str, List[int]] = {}
        for idx in range(len(dataset)):
            key = dataset.get_bucket_key(idx) if hasattr(dataset, "get_bucket_key") else "default"
            buckets.setdefault(str(key), []).append(idx)
        self.buckets = {k: v for k, v in buckets.items() if v}
        if not self.buckets:
            raise ValueError("SameResolutionBatchSampler found no non-empty buckets")

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        all_batches = []
        for indices in self.buckets.values():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    all_batches.append(batch)
        if all_batches:
            batch_order = torch.randperm(len(all_batches), generator=generator).tolist()
            for i in batch_order:
                yield all_batches[i]

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                n += 1
            total += n
        return total

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


class SameBucketVariableBatchSampler(BatchSampler):
    """Build batches inside dataset-provided buckets with optional per-bucket B.

    This is used by scaling-law pretrain where bucket keys can include both
    resolution and temporal window, for example ``512x512/W4``.  The dataset may
    expose:

    - ``get_bucket_key(idx) -> str``
    - ``get_bucket_batch_size(key) -> int``
    - ``get_sample_weight(idx) -> float``

    When ``replacement=True``, bucket and sample draws are weighted by
    ``get_sample_weight``.  That lets long episodes contribute proportional to
    their latent-frame count without mixing different tensor shapes in one
    batch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        replacement: bool = False,
        num_batches: Optional[int] = None,
    ):
        self.dataset = dataset
        self.default_batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0

        buckets: Dict[str, List[int]] = {}
        for idx in range(len(dataset)):
            key = dataset.get_bucket_key(idx) if hasattr(dataset, "get_bucket_key") else "default"
            buckets.setdefault(str(key), []).append(idx)
        self.buckets = {k: v for k, v in buckets.items() if v}
        if not self.buckets:
            raise ValueError("SameBucketVariableBatchSampler found no non-empty buckets")

        self.batch_sizes = {
            key: self._batch_size_for_bucket(key)
            for key in self.buckets
        }
        self.bucket_weights = {
            key: self._bucket_weight(indices)
            for key, indices in self.buckets.items()
        }

        if num_batches is None:
            total = 0
            for key, indices in self.buckets.items():
                bs = self.batch_sizes[key]
                n = len(indices) // bs
                if not self.drop_last and len(indices) % bs:
                    n += 1
                total += n
            num_batches = max(1, total)
        self.num_batches = int(num_batches)

    def _batch_size_for_bucket(self, key: str) -> int:
        getter = getattr(self.dataset, "get_bucket_batch_size", None)
        if callable(getter):
            return max(1, int(getter(key)))
        return max(1, self.default_batch_size)

    def _sample_weight(self, idx: int) -> float:
        getter = getattr(self.dataset, "get_sample_weight", None)
        if callable(getter):
            try:
                return max(0.0, float(getter(idx)))
            except Exception:
                return 1.0
        return 1.0

    def _bucket_weight(self, indices: List[int]) -> float:
        total = sum(self._sample_weight(idx) for idx in indices)
        return total if total > 0 else float(len(indices))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.replacement:
            keys = list(self.buckets)
            bucket_weights = torch.tensor(
                [self.bucket_weights[k] for k in keys],
                dtype=torch.double,
            )
            if float(bucket_weights.sum().item()) <= 0:
                bucket_weights = torch.ones_like(bucket_weights)

            per_bucket_weights = {}
            for key, indices in self.buckets.items():
                weights = torch.tensor(
                    [self._sample_weight(idx) for idx in indices],
                    dtype=torch.double,
                )
                if float(weights.sum().item()) <= 0:
                    weights = torch.ones_like(weights)
                per_bucket_weights[key] = weights

            for _ in range(self.num_batches):
                bucket_pos = int(torch.multinomial(bucket_weights, 1, replacement=True, generator=generator).item())
                key = keys[bucket_pos]
                indices = self.buckets[key]
                bs = self.batch_sizes[key]
                positions = torch.multinomial(
                    per_bucket_weights[key],
                    bs,
                    replacement=True,
                    generator=generator,
                ).tolist()
                yield [indices[pos] for pos in positions]
            return

        all_batches = []
        for key, indices in self.buckets.items():
            bs = self.batch_sizes[key]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            for start in range(0, len(shuffled), bs):
                batch = shuffled[start:start + bs]
                if len(batch) == bs or (batch and not self.drop_last):
                    all_batches.append(batch)
        if all_batches:
            batch_order = torch.randperm(len(all_batches), generator=generator).tolist()
            for i in batch_order:
                yield all_batches[i]

    def __len__(self):
        if self.replacement:
            return self.num_batches
        total = 0
        for key, indices in self.buckets.items():
            bs = self.batch_sizes[key]
            n = len(indices) // bs
            if not self.drop_last and len(indices) % bs:
                n += 1
            total += n
        return total

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
