"""Length-bucketed batching for variable-length trajectories.

Batches respect both the latent-frame budget and maximum sample count.
``two_pointer`` pairs long and short trajectories. ``lpt_balanced`` assigns
longest trajectories first to the least-loaded admissible batch, balancing
work across data-parallel ranks. Select the strategy in the training config;
different strategies produce different batch groupings and epoch lengths.
"""

from __future__ import annotations

import heapq
import logging
import random
from typing import Optional, Sequence

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class LengthBucketBatchSampler(Sampler):
    """Yield batches of similar-length samples (fairseq-style mega-batch sort).

    Each epoch: shuffle indices, cut into mega-batches of ``batch_size *
    mega_batch_factor``, sort each mega-batch by length, slice into batches, then
    shuffle the batch order so consecutive steps are not monotonically increasing
    in length.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        mega_batch_factor: int = 20,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.mega_batch = max(1, int(mega_batch_factor)) * self.batch_size
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        batches: list[list[int]] = []
        for start in range(0, len(order), self.mega_batch):
            mega = order[start:start + self.mega_batch]
            mega.sort(key=lambda j: self.lengths[j])
            for k in range(0, len(mega), self.batch_size):
                batch = mega[k:k + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class TokenBudgetBatchSampler(Sampler):
    """Yield batches bounded by total packed length, not a fixed sample count.

    Two `pack_strategy` values decide how a pass is partitioned; see the module
    docstring for the cost model that separates them.

    ``two_pointer`` (default): shuffle indices, cut into pools of
    ``max_batch_size * mega_batch_factor``, sort each pool by length, then
    greedily bin-pack it via a two-pointer long+short walk (take the longest
    remaining sample, then the shortest remaining, alternating) so batches fill
    up toward `token_budget` instead of leaving a long sample's batch mostly
    empty.

    ``lpt_balanced``: ignore pooling and partition the whole pass at once, fixing
    the batch count at its lower bound and assigning trajectories longest-first
    to the least-loaded admissible batch. This equalizes batch cost, which is
    what a data-parallel step is actually billed for, and reaches a far higher
    utilization than any fill-first heuristic.

    Passes still differ from one another without any help from `lpt_jitter`: the
    index order is shuffled before sorting and Python's sort is stable, so equal
    lengths land in a different order every pass, and equally-loaded bins are
    tie-broken by a per-pass random tag. Measured over 24 passes of the RMBench
    corpus, no batch is ever repeated even at `lpt_jitter=0`.

    `lpt_jitter` therefore defaults to 0, and raising it is close to all cost.
    What it buys is marginal -- which trajectories meet, 26.7k distinct pairs at
    0 against 28.5k at 0.05 -- and packed samples cannot see each other through
    the block-diagonal mask anyway. What it costs is the block-mask skeleton
    cache: balancing loads drives the whole corpus into 25 distinct length
    signatures, for a 98.8% hit rate on an LRU of 64, and jitter shatters that to
    1120 signatures and 34.7%, below even `two_pointer`'s 51.0%. Every miss
    rebuilds a BlockMask through `torch.compile`. Keep it at 0 unless something
    downstream is provably starved of sample mixing.

    Both strategies finally shuffle the batch order so consecutive steps are not
    monotonically increasing in length. Neither needs cross-rank communication:
    a pass is a pure function of ``(seed, pass_index)``.

    A single sample longer than `latent_budget` is never dropped. With
    `over_budget_policy="allow_single"` it becomes its own over-budget batch,
    same as `LengthBucketBatchSampler` never drops data for `drop_last=False`.
    With `over_budget_policy="random_window"` the dataset crops that sample to
    a random contiguous `latent_budget`-sized window, so the sampler caps its
    effective packing length to the budget.

    Note on `drop_last` here: unlike `LengthBucketBatchSampler` (where it means
    "drop a batch smaller than batch_size"), every batch this sampler produces
    is legitimately variable-size by design, so there is no local notion of an
    "incomplete" batch to drop. Instead `drop_last` is exposed purely so
    `accelerate`'s `BatchSamplerShard` -- which reads `getattr(batch_sampler,
    "drop_last", False)` -- drops the same trailing partial round of batches
    on *every* rank when the epoch's batch count is not a multiple of
    `world_size`, instead of some ranks getting one more batch than others.
    That per-rank imbalance would desync collective ops (gradient
    reduce-scatter, metrics all-reduce) since this sampler also has no static
    `batch_size` for accelerate's `even_batches=True` padding path to fall
    back on -- so callers MUST also set `accelerator.even_batches = False`
    when using this sampler (see `train/train.py`). Keeping `drop_last=True`
    (the default) is what makes `even_batches=False` safe: it guarantees
    every rank sees the exact same number of batches per epoch.

    `endless=True` (what `maybe_build_length_bucket_batch_sampler` builds) starts
    the next shuffled pass inline instead of ending the stream, which removes the
    per-pass DataLoader worker resync entirely. It also makes the `drop_last`
    reasoning above moot rather than wrong: a stream that never ends has no
    trailing partial round for `BatchSamplerShard` to split unevenly, so every
    rank gets exactly one batch per round forever. `even_batches=False` is still
    required, since that follows from having no static `batch_size`, not from
    where the stream ends.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        token_budget: int | None = None,
        *,
        latent_budget: int | None = None,
        mbs: int | None = None,
        max_batch_size: int = 32,
        over_budget_policy: str = "allow_single",
        mega_batch_factor: int = 20,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        endless: bool = False,
        pack_strategy: str = "two_pointer",
        lpt_jitter: float = 0.0,
    ):
        self.endless = bool(endless)
        self.raw_lengths = [int(x) for x in lengths]
        resolved_budget = latent_budget if latent_budget is not None else token_budget
        assert resolved_budget is not None, "latent_budget/token_budget must be set"
        self.latent_budget = int(resolved_budget)
        assert self.latent_budget > 0, f"latent_budget must be positive, got {resolved_budget}"
        self.token_budget = self.latent_budget  # legacy alias
        self.mbs = max(1, int(mbs if mbs is not None else max_batch_size))
        self.max_batch_size = self.mbs  # legacy alias
        self.over_budget_policy = str(over_budget_policy)
        if self.over_budget_policy not in {"allow_single", "random_window"}:
            raise ValueError(
                "TokenBudgetBatchSampler supports over_budget_policy='allow_single' "
                "or 'random_window' "
                f"for now, got {self.over_budget_policy!r}"
            )
        self.pack_strategy = str(pack_strategy)
        if self.pack_strategy not in {"two_pointer", "lpt_balanced"}:
            raise ValueError(
                "TokenBudgetBatchSampler supports pack_strategy='two_pointer' or "
                f"'lpt_balanced', got {self.pack_strategy!r}"
            )
        self.lpt_jitter = float(lpt_jitter)
        if self.lpt_jitter < 0.0:
            raise ValueError(f"lpt_jitter must be >= 0, got {lpt_jitter}")
        self.lengths = (
            [min(length, self.latent_budget) for length in self.raw_lengths]
            if self.over_budget_policy == "random_window"
            else list(self.raw_lengths)
        )
        self.mega_batch = max(1, int(mega_batch_factor)) * self.mbs
        self.shuffle = bool(shuffle)
        # See class docstring: this drives accelerate's cross-rank rounding,
        # not per-batch-size filtering like LengthBucketBatchSampler's flag.
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.over_budget_indices = [
            idx for idx, length in enumerate(self.raw_lengths)
            if length > self.latent_budget
        ]
        self.over_budget_count = len(self.over_budget_indices)
        if self.over_budget_count:
            examples = [self.raw_lengths[idx] for idx in self.over_budget_indices[:5]]
            longest = max(examples) if self.over_budget_count <= 5 else max(self.raw_lengths)
            disposition = (
                "will be cropped by the dataset to random contiguous windows of "
                f"length {self.latent_budget}."
                if self.over_budget_policy == "random_window"
                else "will be trained alone in over-budget batches of size 1, "
                "never dropped or truncated."
            )
            logger.warning(
                "TokenBudgetBatchSampler: %d samples exceed latent_budget=%d "
                "(legacy token_budget=%d; longest sample %d exceeds latent_budget "
                "and exceeds token_budget); "
                "over_budget_policy=%s, examples_T_lat=%s. These samples %s",
                self.over_budget_count,
                self.latent_budget,
                self.token_budget,
                longest,
                self.over_budget_policy,
                examples,
                disposition,
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _pack_pool(self, pool_sorted_ids: list[int]) -> list[list[int]]:
        """Two-pointer greedy bin-pack of one length-sorted pool of indices."""
        lengths = self.lengths
        lo, hi = 0, len(pool_sorted_ids) - 1
        batches: list[list[int]] = []
        current: list[int] = []
        current_len = 0
        take_long = True  # alternate: longest-remaining, then shortest-remaining
        while lo <= hi:
            idx = pool_sorted_ids[hi] if take_long else pool_sorted_ids[lo]
            length = lengths[idx]
            fits = (
                current_len + length <= self.latent_budget
                and len(current) < self.mbs
            )
            if current and not fits:
                # Close the current batch and re-evaluate the same idx fresh --
                # an empty batch always accepts its first sample (see the `or
                # not current` invariant this mirrors), so this cannot loop
                # forever: the retry below is guaranteed to make progress.
                batches.append(current)
                current, current_len = [], 0
                continue
            current.append(idx)
            current_len += length
            if take_long:
                hi -= 1
            else:
                lo += 1
            take_long = not take_long
        if current:
            batches.append(current)
        return batches

    def _lpt_place(
        self, items: list[int], bin_count: int, rng: random.Random
    ) -> Optional[list[list[int]]]:
        """Assign `items` (already ordered) to `bin_count` bins, least-loaded first.

        Returns ``None`` when `bin_count` bins cannot hold everything, so the
        caller can retry with one more bin.

        Only bins that still have room for another sample stay in the heap, so
        every candidate popped satisfies the `mbs` cap by construction and the
        one remaining way to fail is capacity. Since the popped bin is the
        least-loaded one, it is also the roomiest: if the item does not fit
        there, it fits nowhere, and this `bin_count` is infeasible.
        """
        lengths = self.lengths
        # Tie-break tags decide which of two equally-loaded bins wins. Shuffling
        # them per pass keeps repeated passes from resolving every tie the same
        # way, which is what would otherwise make co-occurrence static.
        tags = list(range(bin_count))
        rng.shuffle(tags)
        heap = [(0, tags[b], b) for b in range(bin_count)]
        heapq.heapify(heap)
        bins: list[list[int]] = [[] for _ in range(bin_count)]

        for idx in items:
            length = lengths[idx]
            if not heap:
                return None
            load, tag, target = heapq.heappop(heap)
            if load + length > self.latent_budget:
                return None
            bins[target].append(idx)
            if len(bins[target]) < self.mbs:
                heapq.heappush(heap, (load + length, tag, target))
        return [b for b in bins if b]

    def _lpt_pack(self, order: list[int], rng: random.Random) -> list[list[int]]:
        """Partition one pass by balancing batch load instead of filling batches.

        Trajectories longer than the budget cannot share a batch with anything,
        so they are pulled out as singletons first (the `allow_single` contract)
        and the balance problem is solved over what remains.
        """
        lengths = self.lengths
        oversized = [i for i in order if lengths[i] > self.latent_budget]
        rest = [i for i in order if lengths[i] <= self.latent_budget]
        batches: list[list[int]] = [[i] for i in oversized]
        if not rest:
            return batches

        if self.lpt_jitter > 0.0:
            # Perturb the key rather than only breaking exact ties, so that
            # trajectories of merely *similar* length also swap places between
            # passes. Scaling by the length keeps the order roughly descending,
            # which is what LPT's quality depends on.
            def sort_key(i: int) -> float:
                return -lengths[i] * (1.0 + rng.uniform(-self.lpt_jitter, self.lpt_jitter))
        else:
            def sort_key(i: int) -> float:
                return float(-lengths[i])

        ordered = sorted(rest, key=sort_key)
        total = sum(lengths[i] for i in rest)
        bin_count = max(
            -(-total // self.latent_budget),  # ceil: volume bound
            -(-len(rest) // self.mbs),        # ceil: sample-count bound
            1,
        )
        while True:
            placed = self._lpt_place(ordered, bin_count, rng)
            if placed is not None:
                return batches + placed
            bin_count += 1

    def _build_pass(self, pass_index: int) -> list[list[int]]:
        """Bin-pack one full pass over the index space.

        Seeded by ``pass_index`` alone, so every rank independently reproduces the
        same stream without any cross-rank communication.
        """
        rng = random.Random(self.seed + pass_index)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        batches: list[list[int]] = []
        if self.pack_strategy == "lpt_balanced":
            # Deliberately not pooled: balancing is a global property, and
            # mega-batch pools would cap it at the pool boundary.
            batches = self._lpt_pack(order, rng)
        else:
            for start in range(0, len(order), self.mega_batch):
                pool = order[start:start + self.mega_batch]
                pool.sort(key=lambda j: self.lengths[j])
                batches.extend(self._pack_pool(pool))
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _build_batches(self) -> list[list[int]]:
        return self._build_pass(self.epoch)

    def __iter__(self):
        if not self.endless:
            yield from self._build_pass(self.epoch)
            return
        # Endless mode: start the next pass inline instead of raising
        # StopIteration, so the training loop never re-enters
        # `iter(dataloader)`. Each re-entry costs a DataLoader worker resync
        # (`_MultiProcessingDataLoaderIter._reset()` waits for every worker to
        # drain its prefetch backlog, and that work is then discarded --
        # measured 1.48s on the cluster). One pass over a small corpus is only a
        # few optimizer steps, so paying that per pass dominates step time. The
        # passes are lazy: only the current one is materialized.
        pass_index = self.epoch
        while True:
            yield from self._build_pass(pass_index)
            pass_index += 1

    def __len__(self) -> int:
        # Batches in ONE pass, approximate like LengthBucketBatchSampler.__len__:
        # the exact count depends on how bin-packing lands. In endless mode the
        # stream is longer than this by design; nothing drives iteration off it
        # (accelerate only uses it for progress reporting / even_batches
        # rounding -- see module docstring), and the training loop is bounded by
        # max_steps.
        total_length = sum(self.lengths)
        if total_length == 0:
            return 0
        return max(1, -(-total_length // self.latent_budget))  # ceil division


def resolve_sample_lengths(dataset) -> Optional[list[int]]:
    """Return deterministic per-index lengths if the dataset can provide them."""

    getter = getattr(dataset, "get_sample_lengths", None)
    if callable(getter):
        try:
            values = getter()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dataset.get_sample_lengths() failed: %s", exc)
            values = None
        if values is not None:
            return [int(v) for v in values]
    for attr in ("sample_lengths", "lengths"):
        values = getattr(dataset, attr, None)
        if values is not None and not callable(values):
            try:
                return [int(v) for v in values]
            except (TypeError, ValueError):
                return None
    return None


def maybe_build_length_bucket_batch_sampler(config, dataset, batch_size: int):
    """Build a bucketing batch sampler from config, or ``None`` to use a plain loader.

    Config (under ``training.length_bucketing``)::

        enabled: false            # master switch
        mega_batch_factor: 20     # pool size before sorting (see below for the
                                   # per-sample-count multiplier used in each mode)
        shuffle: true
        drop_last: true           # fixed-count mode only (see below)
        seed: 0
        latent_budget: null       # opt-in: switches to TokenBudgetBatchSampler.
                                   # Total per-batch latent budget in T_lat units.
                                   # Batches hold as many samples as fit under
                                   # the budget (at most mbs) instead
                                   # of a fixed count, so a step can be "one very
                                   # long sample, or several short ones" -- see
                                   # data/length_bucketing.py module docstring.
                                   # Effective even when training.batch_size==1.
                                   # Choose a budget that leaves memory headroom
                                   # for model parameters, optimizer state and gradients.
        mbs: 32                   # latent-budget mode only: hard cap on trajectory
                                   # count per batch regardless of budget headroom.
        over_budget_policy: allow_single
                                  # Single trajectories longer than latent_budget
                                  # are kept intact as single-sample batches.
        pack_strategy: two_pointer
                                  # latent-budget mode only. `two_pointer` fills
                                  # batches toward the budget; `lpt_balanced`
                                  # equalizes batch load instead, which is what a
                                  # data-parallel step is billed for. Switching
                                  # changes which trajectories share a step and
                                  # how many steps a pass holds, so treat it as a
                                  # fork-config variable, not a tuning knob.
        lpt_jitter: 0.0           # lpt_balanced only. Relative perturbation of the
                                  # length sort key. Buys a little co-occurrence
                                  # variety and costs a lot of block-mask cache
                                  # hit rate; see the class docstring before
                                  # raising it.

    Fixed-count mode (`latent_budget`/`token_budget` unset) is unchanged from
    before and remains a no-op at `batch_size<=1`. Latent-budget mode is
    independent of `training.batch_size` (it is meant to replace, not multiply,
    that knob).
    """

    training = getattr(config, "training", None)
    bucket_cfg = training.get("length_bucketing", None) if training is not None else None
    if not bucket_cfg or not bool(bucket_cfg.get("enabled", False)):
        return None
    lengths = resolve_sample_lengths(dataset)
    if lengths is None:
        logger.warning(
            "length_bucketing enabled but dataset exposes no deterministic per-index "
            "lengths (get_sample_lengths/sample_lengths/lengths). Falling back to a "
            "plain shuffled loader."
        )
        return None

    latent_budget = bucket_cfg.get("latent_budget", None)
    token_budget = bucket_cfg.get("token_budget", None)
    resolved_budget = latent_budget if latent_budget is not None else token_budget
    if resolved_budget is not None and int(resolved_budget) > 0:
        mbs = int(bucket_cfg.get("mbs", bucket_cfg.get("max_batch_size", 32)))
        over_budget_policy = str(bucket_cfg.get("over_budget_policy", "allow_single"))
        mega_batch_factor = int(bucket_cfg.get("mega_batch_factor", 20))
        pack_strategy = str(bucket_cfg.get("pack_strategy", "two_pointer"))
        lpt_jitter = float(bucket_cfg.get("lpt_jitter", 0.0))
        sampler = TokenBudgetBatchSampler(
            lengths,
            token_budget=int(token_budget) if token_budget is not None else None,
            latent_budget=int(latent_budget) if latent_budget is not None else None,
            mbs=mbs,
            max_batch_size=mbs,
            over_budget_policy=over_budget_policy,
            mega_batch_factor=mega_batch_factor,
            shuffle=bool(bucket_cfg.get("shuffle", True)),
            drop_last=bool(bucket_cfg.get("drop_last", True)),
            seed=int(bucket_cfg.get("seed", 0)),
            # Always endless here: this factory only ever builds the training
            # loader, and that loop exits on max_steps, not on the stream ending.
            # The total batch count a run consumes is known up front
            # (max_steps * grad_accum * world_size), so there is no reason to let
            # the stream end and pay a worker resync. Deliberately not a config
            # key -- a knob whose only correct value is True is noise.
            endless=True,
            pack_strategy=pack_strategy,
            lpt_jitter=lpt_jitter,
        )
        logger.info(
            "length_bucketing enabled (latent-budget mode): %d samples, "
            "latent_budget=%d, mbs=%d, over_budget_policy=%s, "
            "over_budget_samples=%d, mega_batch_factor=%d, "
            "pack_strategy=%s, lpt_jitter=%s, batches_per_pass~%d, endless=%s",
            len(lengths),
            sampler.latent_budget,
            sampler.mbs,
            sampler.over_budget_policy,
            sampler.over_budget_count,
            mega_batch_factor,
            sampler.pack_strategy,
            sampler.lpt_jitter if sampler.pack_strategy == "lpt_balanced" else "n/a",
            len(sampler),
            sampler.endless,
        )
        return sampler

    if int(batch_size) <= 1:
        logger.info("length_bucketing enabled but batch_size<=1; bucketing is a no-op.")
        return None
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size,
        mega_batch_factor=int(bucket_cfg.get("mega_batch_factor", 20)),
        shuffle=bool(bucket_cfg.get("shuffle", True)),
        drop_last=bool(bucket_cfg.get("drop_last", True)),
        seed=int(bucket_cfg.get("seed", 0)),
    )
    logger.info(
        "length_bucketing enabled: %d samples, batch_size=%d, mega_batch_factor=%d",
        len(lengths), int(batch_size), int(bucket_cfg.get("mega_batch_factor", 20)),
    )
    return sampler
