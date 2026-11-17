"""Coverage and restart contracts for the corrected window visit allocation."""
from collections import Counter, defaultdict

import torch
import pytest

from data.samplers import WeightedExpandedBatchSampler
from data.window_start import select_window_start, unpack_index


class Episodes:
    def __init__(self, weights, buckets=None):
        self.weights = weights
        self.buckets = buckets or ["same"] * len(weights)

    def __len__(self):
        return len(self.weights)

    def get_bucket_key(self, index):
        return self.buckets[index]

    def get_sample_weight(self, index):
        return self.weights[index]


def sampler(dataset, *, seed=42, drop_last=False, windows=True):
    return WeightedExpandedBatchSampler(dataset, batch_size=4, drop_last=drop_last,
                                        seed=seed,
                                        window_start_sampling=windows)


def test_equal_weight_episode_visits_are_dense_across_epochs():
    batch_sampler = sampler(Episodes([1.] * 8))
    ordinals = defaultdict(list)
    window_starts = defaultdict(list)
    for epoch in range(16):
        batch_sampler.set_epoch(epoch)
        for batch in batch_sampler:
            for packed in batch:
                episode, ordinal = unpack_index(packed)
                ordinals[episode].append(ordinal)
                window_starts[episode].append(select_window_start(ordinal, 15, 8, str(episode)))
    for episode in range(8):
        assert ordinals[episode] == list(range(16))
        assert len(set(window_starts[episode][:8])) == 8
        assert len(set(window_starts[episode][8:16])) == 8


def test_fractional_source_budgets_preserve_dense_visits_and_exact_epoch_quotas():
    # One group has >1 visits/episode and the other <1.  This exercises the
    # remainder's wrap boundary and epochs in which an episode is not visited.
    batch_sampler = sampler(Episodes([7.] * 3 + [1.] * 14))
    ordinals = defaultdict(list)
    for epoch in range(23):
        batch_sampler.set_epoch(epoch)
        epoch_counts = Counter()
        for batch in batch_sampler:
            for packed in batch:
                episode, ordinal = unpack_index(packed)
                ordinals[episode].append(ordinal)
                epoch_counts[episode < 3] += 1
        assert epoch_counts == {True: 10, False: 7}
    for values in ordinals.values():
        assert values == list(range(len(values)))


def test_remainder_wraps_mid_epoch_without_duplicate_extra_visits():
    # 15 total slots: this group's 5 episodes receive 8 slots, i.e. base1/r3.
    batch_sampler = sampler(Episodes([8.] * 5 + [3.5] * 10))
    visits = defaultdict(list)
    for epoch in range(11):
        batch_sampler.set_epoch(epoch)
        per_epoch = Counter()
        for batch in batch_sampler:
            for packed in batch:
                episode, ordinal = unpack_index(packed)
                visits[episode].append(ordinal)
                per_epoch[episode] += 1
        assert sum(v for k, v in per_epoch.items() if k < 5) == 8
        assert all(1 <= per_epoch[k] <= 2 for k in range(5))
    for values in visits.values():
        assert values == list(range(len(values)))


def test_multiple_cycles_in_one_epoch_are_emitted_in_visit_order():
    batch_sampler = sampler(Episodes([100.] + [1.] * 99))
    values = []
    for batch in batch_sampler:
        values.extend(ordinal for episode, ordinal in map(unpack_index, batch) if episode == 0)
    assert len(values) > 16
    assert values == list(range(len(values)))
    starts = [select_window_start(x, 15, 8, "oversampled") for x in values]
    assert len(set(starts[:8])) == 8
    assert len(set(starts[8:16])) == 8


def test_packing_overflow_fails_instead_of_clamping_to_repeated_window():
    from data.window_start import ORDINAL_STRIDE

    batch_sampler = sampler(Episodes([1.] * 8))
    batch_sampler.set_epoch(ORDINAL_STRIDE)
    with pytest.raises(ValueError, match="packing capacity"):
        list(batch_sampler)


def test_direct_late_epoch_reconstruction_matches_continuous_sampler():
    dataset = Episodes([7.] * 3 + [1.] * 14)
    continuous = sampler(dataset)
    for epoch in range(23):
        continuous.set_epoch(epoch)
        expected = list(continuous)
    direct = sampler(dataset)
    direct.set_epoch(22)
    assert list(direct) == expected


def test_resume_prefix_after_multiple_epochs():
    from accelerate.data_loader import BatchSamplerShard

    dataset = Episodes([7.] * 3 + [1.] * 14)
    # Same DP truncation rules as the training loader; reconstructing epoch and
    # offset must reproduce every future packed ID on every rank.
    for rank in range(2):
        full = sampler(dataset, drop_last=True)
        expected = []
        for epoch in range(9):
            full.set_epoch(epoch)
            expected.extend(list(BatchSamplerShard(full, num_processes=2,
                                                    process_index=rank,
                                                    even_batches=False)))
        for step in (1, 3, 8, 12):
            restored = sampler(dataset, drop_last=True)
            epoch = restored.resume(step, 2)
            actual = list(BatchSamplerShard(restored, num_processes=2,
                                           process_index=rank, even_batches=False))
            for next_epoch in range(epoch + 1, 9):
                restored.set_epoch(next_epoch)
                actual.extend(list(BatchSamplerShard(restored, num_processes=2,
                                                     process_index=rank,
                                                     even_batches=False)))
            assert actual == expected[step:]


def test_disabled_window_mode_preserves_legacy_random_remainder_path():
    dataset = Episodes([7.] * 3 + [1.] * 14)
    current = sampler(dataset, windows=False)
    # Reconstruct the legacy expansion exactly using the same RNG operations.
    for epoch in (0, 1, 7):
        current.set_epoch(epoch)
        generator = torch.Generator().manual_seed(42 + epoch)
        sizes = current._group_sizes["same"]
        targets = current._group_targets["same"]
        base = torch.div(targets, sizes, rounding_mode="floor")
        extra = targets - base * sizes
        counts = torch.repeat_interleave(base, sizes)
        starts = torch.cumsum(sizes, 0) - sizes
        for pos in torch.nonzero(extra, as_tuple=False).flatten().tolist():
            picks = torch.randperm(int(sizes[pos]), generator=generator)[:int(extra[pos])]
            counts[int(starts[pos]) + picks] += 1
        pool = torch.repeat_interleave(current._group_indices["same"], counts)
        pool = pool[torch.randperm(pool.numel(), generator=generator)]
        expected_batches = [pool[i:i + 4].tolist() for i in range(0, len(pool), 4)]
        order = torch.randperm(len(expected_batches), generator=generator)
        assert list(current) == [expected_batches[i] for i in order.tolist()]
