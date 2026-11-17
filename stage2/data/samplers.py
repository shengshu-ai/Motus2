"""Dataset samplers shared by training and tests."""

from array import array
from bisect import bisect_right
import hashlib
from typing import Dict, List, Optional

import torch
from torch.utils.data import BatchSampler

from data.window_start import ORDINAL_STRIDE, pack_index

# Epoch-sized batch orders are kept as tensors and converted to Python ints in
# slices, so a 30M-index epoch never materializes a 30M-element list.
_BATCH_ORDER_CHUNK = 8192


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


class WeightedExpandedBatchSampler(BatchSampler):
    """Build same-bucket batches from weight-expanded indices, then shuffle.

    Source weights are applied by replicating dataset indices proportional to
    ``get_sample_weight`` before the usual per-bucket shuffle/batch split.
    This preserves target source ratios while keeping dataloader-friendly
    sequential access, avoiding per-step weighted random draws.

    One replication budget is shared by every
    bucket, so a source that owns a whole resolution bucket still lands on its
    configured ratio, and a source whose per-index replication factor is below
    one keeps a proportional random subset instead of being floored away.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        window_start_sampling: bool = False,
    ):
        self.dataset = dataset
        self.default_batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        # When set, every emitted entry is ``episode_index * ORDINAL_STRIDE +
        # visit_ordinal`` so the dataset can resolve a specific latent window.
        self.window_start_sampling = bool(window_start_sampling)
        self.epoch = 0
        # Batches of this sampler already consumed before the current epoch
        # resumes.  Set by ``resume()``; consumed once by ``__iter__``.
        self._skip_batches = 0

        buckets: Dict[str, List[int]] = {}
        for idx in range(len(dataset)):
            key = dataset.get_bucket_key(idx) if hasattr(dataset, "get_bucket_key") else "default"
            buckets.setdefault(str(key), []).append(idx)
        self.buckets = {k: v for k, v in buckets.items() if v}
        if not self.buckets:
            raise ValueError("WeightedExpandedBatchSampler found no non-empty buckets")

        self.batch_sizes = {
            key: self._batch_size_for_bucket(key)
            for key in self.buckets
        }

        # Grouped views of each bucket, only used by the global-balance path.
        self._group_indices: Dict[str, torch.Tensor] = {}
        self._group_sizes: Dict[str, torch.Tensor] = {}
        self._group_targets: Dict[str, torch.Tensor] = {}
        self._plan_global_expansion()
        self.expanded_counts = {
            key: int(self._group_targets[key].sum())
            for key, indices in self.buckets.items()
        }

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

    def _expand_bucket(self, indices: List[int], generator: torch.Generator) -> List[int]:
        weights = [self._sample_weight(idx) for idx in indices]
        total_weight = sum(weights)
        if total_weight <= 0:
            order = torch.randperm(len(indices), generator=generator).tolist()
            return [indices[i] for i in order]

        target_total = len(indices)
        exact = [weight * target_total / total_weight for weight in weights]
        counts = [int(value) for value in exact]
        remaining = target_total - sum(counts)
        if remaining > 0:
            order = sorted(
                range(len(exact)),
                key=lambda i: (exact[i] - counts[i], weights[i]),
                reverse=True,
            )
            for i in order[:remaining]:
                counts[i] += 1

        expanded: List[int] = []
        for idx, reps in zip(indices, counts):
            expanded.extend([idx] * reps)
        if not expanded:
            order = torch.randperm(len(indices), generator=generator).tolist()
            return [indices[i] for i in order]

        order = torch.randperm(len(expanded), generator=generator).tolist()
        return [expanded[i] for i in order]

    def _plan_global_expansion(self) -> None:
        """Split one global replication budget across every weight group.

        ``get_sample_weight`` returns ``source_ratio / episodes_in_source``, so
        indices that share a weight are one data source and a group's weight
        mass is its configured ratio.  Sizing groups from the global mass keeps
        cross-bucket ratios intact, which per-bucket normalization cannot do:
        there a single-source bucket always expands to its own original size
        and its ratio is ignored.
        """
        budget = sum(len(indices) for indices in self.buckets.values())
        masses: List[float] = []
        sizes: List[int] = []
        for key, indices in self.buckets.items():
            weight_buffer = array("d", (self._sample_weight(idx) for idx in indices))
            sorted_weights, order = torch.sort(
                torch.frombuffer(weight_buffer, dtype=torch.float64),
                stable=True,
            )
            group_weights, group_sizes = torch.unique_consecutive(sorted_weights, return_counts=True)
            index_buffer = array("q", indices)
            self._group_indices[key] = torch.frombuffer(index_buffer, dtype=torch.int64)[order]
            self._group_sizes[key] = group_sizes
            masses.extend((group_weights * group_sizes).tolist())
            sizes.extend(group_sizes.tolist())

        targets = self._allocate_group_targets(masses, sizes, budget)
        start = 0
        for key in self.buckets:
            stop = start + len(self._group_sizes[key])
            self._group_targets[key] = torch.tensor(targets[start:stop], dtype=torch.int64)
            start = stop

    @staticmethod
    def _allocate_group_targets(masses: List[float], sizes: List[int], budget: int) -> List[int]:
        """Turn fractional per-group budgets into integers that sum to ``budget``.

        Largest remainder is safe at group granularity because a group is one
        source: the rounding error is at most one index per source.  Applying it
        per index instead lets whole sources lose every slot, because indices of
        one source share a fractional part and are filled or skipped together.
        """
        total_mass = sum(masses)
        if total_mass <= 0:
            return list(sizes)

        exact = [budget * mass / total_mass for mass in masses]
        targets = [int(value) for value in exact]
        remaining = budget - sum(targets)
        if remaining > 0:
            order = sorted(
                range(len(exact)),
                key=lambda i: exact[i] - targets[i],
                reverse=True,
            )
            for i in order[:remaining]:
                targets[i] += 1

        # Rounding must never drop a source out of the epoch entirely.
        for i, mass in enumerate(masses):
            if mass <= 0 or targets[i] > 0:
                continue
            donor = max(range(len(targets)), key=lambda j: targets[j])
            if targets[donor] <= 1:
                break
            targets[donor] -= 1
            targets[i] = 1
        return targets

    def _expand_bucket_balanced(self, key: str, generator: torch.Generator) -> torch.Tensor:
        """Realize the planned per-group counts for one bucket, then shuffle.

        Each group repeats every index ``target // size`` times and hands the
        ``target % size`` leftover slots to a fresh random subset of the group,
        so a group with a sub-1 replication factor still contributes exactly its
        quota through a rotating subset of its episodes.
        """
        sizes = self._group_sizes[key]
        targets = self._group_targets[key]
        base = torch.div(targets, sizes, rounding_mode="floor")
        extra = targets - base * sizes
        counts = torch.repeat_interleave(base, sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        previous_visits = None
        if self.window_start_sampling:
            counts, previous_visits = self._window_visit_counts(key, sizes, base, extra)
        else:
            for pos in torch.nonzero(extra, as_tuple=False).flatten().tolist():
                picks = torch.randperm(int(sizes[pos]), generator=generator)[:int(extra[pos])]
                counts[int(starts[pos]) + picks] += 1

        indices = self._group_indices[key]
        pool = torch.repeat_interleave(indices, counts)
        if pool.numel() == 0:
            return pool
        if self.window_start_sampling:
            pool = self._pack_window_ordinals(pool, counts, previous_visits)
        return pool[torch.randperm(pool.numel(), generator=generator)]

    def _window_visit_counts(self, key, sizes, base, extra):
        """Allocate remainders without holes in each episode's visit ordinals.

        A source group's remainder walks a seeded permutation cyclically.  Each
        epoch still gets exactly its target budget and at most one extra visit
        per episode.  The number of prior visits is computable directly from
        the epoch, including epochs where that episode received no extra slot;
        resume therefore needs no replay of previous million-entry epochs.
        """
        counts = torch.repeat_interleave(base, sizes)
        previous = torch.repeat_interleave(base * int(self.epoch), sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        for pos in torch.nonzero(extra, as_tuple=False).flatten().tolist():
            size, remainder = int(sizes[pos]), int(extra[pos])
            start = int(starts[pos])
            seed_material = f"{self.seed}|{key}|{pos}|window-remainder-v1".encode()
            seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(size, generator=generator)
            cycles, cursor = divmod(int(self.epoch) * remainder, size)
            previous[start:start + size] += cycles
            previous[start + order[:cursor]] += 1
            picks = order[(torch.arange(remainder) + cursor) % size]
            counts[start + picks] += 1
        return counts, previous

    def _pack_window_ordinals(
        self,
        pool: torch.Tensor,
        counts: torch.Tensor,
        previous_visits: torch.Tensor,
    ) -> torch.Tensor:
        """Replace each episode index with ``index * ORDINAL_STRIDE + ordinal``.

        The ordinal counts how many times the episode has been visited *since
        the start of training*, not merely within this epoch, so the dataset
        keeps handing out unseen latent windows across epochs instead of
        restarting at the same first window every time.

        ``previous_visits`` is the exact count of slots allocated in earlier
        epochs, not an upper bound on those counts.  Upper bounds leave holes
        and can advance select_window_start into a new cycle while starts in
        the preceding cycle remain unseen.  As with the batch sampler itself,
        the guarantee covers allocated slots; downstream dropping/replacing a
        sample does not count as consuming its assigned window. The final
        shuffled batch order is handled by _order_window_ordinals below.
        """
        offsets = torch.cumsum(counts, 0) - counts
        within_epoch = (
            torch.arange(pool.numel(), dtype=torch.int64)
            - torch.repeat_interleave(offsets, counts)
        )
        ordinal = torch.repeat_interleave(previous_visits, counts) + within_epoch
        if ordinal.numel() and int(ordinal.max()) >= ORDINAL_STRIDE:
            raise ValueError("Window visit ordinal exceeds packing capacity; refusing repeated windows")
        return pool * ORDINAL_STRIDE + ordinal

    @staticmethod
    def _order_window_ordinals(pools, batch_sizes, offsets, order):
        """Keep visits monotone after both sample and global-batch shuffles.

        Episode slots and bucket/source budgets stay exactly where the RNG put
        them. Only an episode's assigned ordinals are sorted among its slots,
        so a later window cycle cannot be emitted before an earlier cycle in
        the same epoch. Compute the whole order before applying a resume skip.
        """
        for pos, pool in enumerate(pools):
            bucket_batches = order[(order >= offsets[pos]) & (order < offsets[pos + 1])] - offsets[pos]
            slots = (bucket_batches[:, None] * batch_sizes[pos] + torch.arange(batch_sizes[pos])).flatten()
            slots = slots[slots < pool.numel()]
            if slots.numel() == 0:
                continue
            emitted = pool[slots]
            grouped_slots = torch.argsort(emitted // ORDINAL_STRIDE, stable=True)
            reordered = torch.empty_like(emitted)
            reordered[grouped_slots] = emitted.sort().values
            pool[slots] = reordered


    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        pass

        pools: List[torch.Tensor] = []
        batch_sizes: List[int] = []
        offsets: List[int] = [0]
        for key in self.buckets:
            pool = self._expand_bucket_balanced(key, generator)
            bs = self.batch_sizes[key]
            batches = pool.numel() // bs
            if not self.drop_last and pool.numel() % bs:
                batches += 1
            pools.append(pool)
            batch_sizes.append(bs)
            offsets.append(offsets[-1] + batches)

        total_batches = offsets[-1]
        if total_batches <= 0:
            return
        order = torch.randperm(total_batches, generator=generator)
        if self.window_start_sampling:
            self._order_window_ordinals(pools, batch_sizes, offsets, order)
        skip_batches = int(self._skip_batches)
        self._skip_batches = 0
        if skip_batches:
            # Resuming mid-epoch: the batch order is fully determined by
            # ``seed + epoch``, so re-deriving it and dropping the prefix walks
            # the exact same window-start stream the interrupted run would have.
            order = order[skip_batches:]
        for chunk in order.split(_BATCH_ORDER_CHUNK):
            for batch_id in chunk.tolist():
                pos = bisect_right(offsets, batch_id) - 1
                bs = batch_sizes[pos]
                start = (batch_id - offsets[pos]) * bs
                yield pools[pos][start:start + bs].tolist()

    def __len__(self):
        total = 0
        for key, count in self.expanded_counts.items():
            bs = self.batch_sizes[key]
            total += count // bs
            if not self.drop_last and count % bs:
                total += 1
        return total

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def steps_per_epoch(self, num_processes: int) -> int:
        """Optimizer steps one rank consumes per epoch.

        accelerate shards a custom batch sampler by handing batch ``i`` to rank
        ``i % num_processes``, so a rank sees every ``num_processes``-th batch
        and one epoch costs ``len(self) / num_processes`` steps.
        """
        num_processes = max(1, int(num_processes))
        return max(1, len(self) // num_processes)

    def resume(self, global_step: int, num_processes: int) -> int:
        """Continue the sampler exactly where ``global_step`` left off.

        Returns the epoch the run is now in.  Both the batch order and the packed
        window ordinals are pure functions of ``(seed, epoch)``, so re-deriving
        them and skipping the prefix replays nothing and repeats nothing: the
        resumed run sees the window stream the interrupted run would have seen.
        """
        global_step = max(0, int(global_step))
        per_epoch = self.steps_per_epoch(num_processes)
        epoch = global_step // per_epoch
        within_epoch_steps = global_step % per_epoch
        self.epoch = int(epoch)
        self._skip_batches = int(within_epoch_steps) * max(1, int(num_processes))
        return self.epoch
