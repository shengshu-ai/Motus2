"""Global request/repair/commit protocol for strict pretrain window consumption.

The quota sampler runs only on rank zero. It contributes final, full local
batches; workers load explicit requests and never own consumption state.

Batch N+1 is prepared on a background thread while the caller computes on N.
Rank zero's per-batch planning -- candidate discovery and exact latent
geometry for every slot -- runs in a forked process pool, one batch ahead, so
none of it sits on the training step's critical path.
"""
import collections
import copy
import hashlib
import itertools
import json
import logging
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .dataset import ar_collate_fn
from .strict_window_sampling import Coordinator


logger = logging.getLogger(__name__)

_WORKER_DATASET = None
_PLAN_DATASET = None


def _worker_init(dataset):
    global _WORKER_DATASET
    _WORKER_DATASET = dataset
    torch.set_num_threads(1)


def _plan_worker_init(dataset):
    global _PLAN_DATASET
    _PLAN_DATASET = dataset
    torch.set_num_threads(1)


def _plan_resolve(index):
    """Exactly ``DatasetCatalog.resolve(index)``, in a forked planning process.

    A separate process owns its own interpreter lock and its own global RNG, so
    discovery and geometry run fully in parallel and cannot perturb the parent's
    augmentation stream. Errors return None: the parent then resolves serially
    and surfaces the error through the coordinator's normal policy.
    """
    dataset = _PLAN_DATASET
    try:
        template = getattr(dataset, '_window_spec_template', None)
        spec = template(index) if template is not None else dataset.window_spec(index, resolve_geometry=False)
        if not spec.get('request_geometry_known', False):
            resolver = getattr(dataset, '_resolve_window_spec_geometry', None)
            spec = dataset.window_spec(index, resolve_geometry=True) if resolver is None else resolver(spec)
        return index, spec
    except Exception:
        return index, None


def _plan_screen(request):
    """The reader's action-side verdict for one request, in a planning process.

    Seeds exactly as ``_read_request`` does, so any request-owned randomness in
    the action path matches the read. An error means "no verdict": the reader
    then judges the window itself.
    """
    verdict = getattr(_PLAN_DATASET, 'strict_window_verdict', None)
    if verdict is None:
        return None
    seed = int.from_bytes(hashlib.sha256(request['request_id'].encode()).digest()[:8], 'big')
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.random.default_generator.manual_seed(seed)
    try:
        return verdict(request)
    except Exception:
        return None


def _read_request(request):
    # A request, rather than the worker scheduling order, owns all CPU choices.
    seed = int.from_bytes(hashlib.sha256(request['request_id'].encode()).digest()[:8], 'big')
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.random.default_generator.manual_seed(seed)
    return _WORKER_DATASET.read_window(request)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def data_plan_fingerprint(config):
    """Bind data lists and rules; run/eval frequency and max_steps may change."""
    from omegaconf import OmegaConf
    recipe = OmegaConf.to_container(config.dataset, resolve=True)
    source_root = Path(__file__).resolve().parents[1]
    def path_for(value, source_relative=False):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (source_root / path if source_relative else path.resolve())
    def file_identity(value, source_relative=False):
        path = path_for(value, source_relative)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    indices = recipe.get('jsonl_path') or []
    if isinstance(indices, str):
        indices = [indices]
    inputs = [file_identity(path) for path in indices]
    for key in ('stereo_latent_sources_path', 'action_stats_path'):
        if recipe.get(key):
            inputs.append(file_identity(recipe[key], source_relative=True))
    for path in (recipe.get('action_stats_by_dataset') or {}).values():
        inputs.append(file_identity(path, source_relative=True))
    model = config.model
    common = config.common
    return dict(schema='motus-strict-data-plan-v1', dataset=recipe, input_files=inputs,
                common=OmegaConf.to_container(common, resolve=True),
                data_model=OmegaConf.to_container(model, resolve=True),
                chunk_size=int(model.get('fixed_chunk_size', common.get('fixed_chunk_size', 2))),
                action_mode=model.get('action_mode'),
                wrist_relative_max=recipe.get('wrist_xyz_relative_max', common.get('wrist_xyz_relative_max')),
                wrist_absolute_max=recipe.get('wrist_xyz_abs_max', common.get('wrist_xyz_abs_max')),
                world_size=int(dist.get_world_size()) if dist.is_initialized() else 1,
                batch_size=int(config.training.batch_size), gradient_accumulation_steps=1,
                seed=int(config.training.get('seed', 42)),
                window_rule_sha256=hashlib.sha256((source_root / 'data/window_start.py').read_bytes()).hexdigest())


class DatasetCatalog:
    """No duplicate million-entry Python catalog; exact geometry is resolved lazily."""
    def __init__(self, dataset, *, prefill_workers=8):
        self.dataset = dataset
        self.resolved = {}
        self._base_specs = {}
        self._zip_cache = {}
        self.prefill_workers = max(1, int(prefill_workers))
        self._prefill_pool = None
        self._plan_pool = None
        self._ahead = []

    def start_plan_pool(self, processes):
        """Fork the planning processes. Call on the constructing (main) thread."""
        if processes and self._plan_pool is None:
            self._plan_pool = multiprocessing.get_context('fork').Pool(
                int(processes), initializer=_plan_worker_init, initargs=(self.dataset,))

    def __len__(self):
        return len(self.dataset.episode_files)

    def __getitem__(self, index):
        ep = self.dataset.episode_files[index]
        return dict(episode_index=int(index), episode_id=self.dataset._episode_identity(ep),
                    source_id=self.dataset._sampling_source_identity(ep),
                    bucket_id=str(self.dataset.get_bucket_key(index)),
                    total_latents=ep.get('sc_total_latent_frames'),
                    window_size=int(self.dataset._window_size_for_episode(ep, index)),
                    latent_variant={})

    def _base_spec(self, index):
        index = int(index)
        if index not in self._base_specs:
            template = getattr(self.dataset, '_window_spec_template', None)
            if template is not None:
                self._base_specs[index] = template(index)
            else:
                self._base_specs[index] = self.dataset.window_spec(index, resolve_geometry=False)
        return copy.deepcopy(self._base_specs[index])

    def _paths_are_local_torch_zip(self, paths):
        for path in paths:
            key = str(path)
            cached = self._zip_cache.get(key)
            if cached is None:
                try:
                    with open(key, 'rb') as stream:
                        cached = stream.read(4) == b'PK\x03\x04'
                except OSError:
                    cached = False
                self._zip_cache[key] = cached
            if not cached:
                return False
        return True

    def close(self):
        if self._prefill_pool is not None:
            self._prefill_pool.shutdown(wait=False)
            self._prefill_pool = None
        if self._plan_pool is not None:
            self._plan_pool.terminate()
            self._plan_pool.join()
            self._plan_pool = None
        self._ahead = []

    def resolve(self, index):
        if index not in self.resolved:
            spec = self._base_spec(index)
            if not spec.get('request_geometry_known', False):
                resolver = getattr(self.dataset, '_resolve_window_spec_geometry', None)
                if resolver is None:
                    spec = self.dataset.window_spec(index, resolve_geometry=True)
                else:
                    spec = resolver(spec)
            self.resolved[index] = spec
        return self.resolved[index]

    def _missing(self, indices):
        return [index for index in dict.fromkeys(int(i) for i in indices) if index not in self.resolved]

    def _collect_ahead(self, wait_for=()):
        """Apply finished look-ahead results; block only on jobs covering `wait_for`.

        Results are pure functions of the episode, so the first one wins.
        """
        wait_for = set(wait_for)
        remaining = []
        for covered, result in self._ahead:
            if result.ready() or covered & wait_for:
                try:
                    rows = result.get()
                except Exception:
                    continue
                for index, spec in rows:
                    if spec is not None:
                        self.resolved.setdefault(index, spec)
            else:
                remaining.append((covered, result))
        self._ahead = remaining

    def prefill_ahead(self, indices):
        """Start resolving a future batch's geometry; never blocks the caller."""
        if self._plan_pool is None:
            return
        self._collect_ahead()
        missing = self._missing(indices)
        if missing:
            self._ahead.append((set(missing), self._plan_pool.map_async(_plan_resolve, missing, chunksize=1)))

    def prefill_current(self, indices):
        """Resolve these episodes' geometry before any reservation touches them.

        With a planning pool, discovery and geometry for every missing episode
        run in parallel processes -- the serial per-episode discovery and file
        header reads on rank zero were the strict loader's per-batch floor. The
        results are exactly ``resolve``'s; failures fall back to it lazily.
        """
        if self._plan_pool is not None:
            # A look-ahead already resolving these episodes is cheaper to wait
            # for than to duplicate; unrelated look-aheads keep running.
            self._collect_ahead(wait_for=self._missing(indices))
            missing = self._missing(indices)
            if missing:
                for index, spec in self._plan_pool.map(_plan_resolve, missing, chunksize=1):
                    if spec is not None:
                        self.resolved.setdefault(index, spec)
            return
        self._collect_ahead()
        resolver = getattr(self.dataset, '_resolve_window_spec_geometry', None)
        if resolver is None:
            return
        pending = []
        for index in dict.fromkeys(int(i) for i in indices):
            if index in self.resolved:
                continue
            try:
                spec = self._base_spec(index)
            except Exception:
                continue  # The coordinator still owns resolution error policy.
            variant = spec.get('latent_variant', {})
            paths = [variant[k] for k in ('latent_path', 'left_latent_path', 'right_latent_path')
                     if variant.get(k)]
            if not paths or not all(Path(path).is_absolute() for path in paths):
                continue
            if not self._paths_are_local_torch_zip(paths):
                continue
            pending.append((index, spec))
        if not pending:
            return

        def read(item):
            index, spec = item
            try:
                return index, resolver(spec, allow_payload_fallback=False)
            except Exception:
                return index, None

        # Join all threads before a first multiprocessing reader pool is forked.
        # Applying results in the original order preserves reservation order.
        workers = min(self.prefill_workers, len(pending))
        if self._prefill_pool is None:
            self._prefill_pool = ThreadPoolExecutor(max_workers=self.prefill_workers)
        resolved = list(self._prefill_pool.map(read, pending, chunksize=max(1, len(pending) // workers)))
        for index, spec in resolved:
            if spec is not None:
                self.resolved[index] = spec


class StrictWindowLoader:
    """One collective batch in flight, with exactly B samples on every rank."""
    is_strict_window_loader = True

    def __init__(self, dataset, batch_sampler, *, rank, world_size, batch_size,
                 fingerprint, migration_start_step, seed=42, num_workers=4,
                 load_timeout=120, max_retries=2, max_replacements=32, audit_path=None,
                 prefill_workers=None, prefetch=True, plan_processes=0,
                 same_source_replacements=32, max_step_replacements=4096, timing_every=10,
                 pin_memory=True, device_dtypes=None, skip_fingerprint_check=False,
                 resample_sources=None, append_sources_on_resume=()):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.rank, self.world_size, self.batch_size = int(rank), int(world_size), int(batch_size)
        self.fingerprint = fingerprint
        self.migration_start_step = int(migration_start_step)
        self.seed = int(seed)
        self.num_workers = max(0, min(int(num_workers), self.batch_size))
        self.load_timeout, self.max_retries = float(load_timeout), int(max_retries)
        self.max_replacements = int(max_replacements)
        # A rejected window is replaced from its own source and bucket first;
        # after this many rejections in one slot, from another source in the
        # same bucket. Data problems never stop training -- only a step needing
        # more than max_step_replacements does, since that is a broken store.
        self.same_source_replacements = max(1, min(int(same_source_replacements), self.max_replacements))
        self.max_step_replacements = int(max_step_replacements)
        self.audit_path = Path(audit_path) if audit_path else None
        self.prefill_workers = max(1, int(prefill_workers if prefill_workers is not None
                                            else max(num_workers, 8)))
        self.plan_processes = max(0, int(plan_processes))
        self._pool = None
        # Prefetch v2: prepare batch N+1 while the caller computes on N.
        self.prefetch = bool(prefetch)
        self.control_group = None
        self._prefetch_pool = None
        self._prefetch_future = None
        self._prefetch_result = None
        self._prefetch_error = None
        self._prefetch_step = None
        self._pending_step = None
        # Rank zero plans one slot list ahead. Each list keeps the planner
        # position it started from, so a checkpoint can rewind exactly.
        self._planned = collections.deque()
        self._step_positions = {}
        self.last_load_seconds = 0.0
        self.last_commit_wait = 0.0
        self.last_prep_timing = {}
        self.timing_every = max(0, int(timing_every))
        self.skip_fingerprint_check = bool(skip_fingerprint_check)
        self.append_sources_on_resume = tuple(append_sources_on_resume or ())
        if resample_sources is None:
            self.resample_sources = ()
        elif isinstance(resample_sources, str):
            self.resample_sources = (resample_sources,)
        else:
            self.resample_sources = tuple(str(s) for s in resample_sources if s)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        # Batch keys the training step casts on its way to the device, and the
        # dtype it casts them to; staged on the background thread.
        self.device_dtypes = dict(device_dtypes or {})
        self._epoch = self._batch_offset = 0
        self._planner = None
        self._pending = None
        self.committed_step = self.migration_start_step
        self.catalog = None
        self.coordinator = None
        self._restored = False
        if self.world_size < 1 or self.batch_size < 1 or self.load_timeout <= 0:
            raise ValueError('Invalid strict loader dimensions/timeout')
        def initialize():
            self.catalog = DatasetCatalog(dataset, prefill_workers=self.prefill_workers)
            self.coordinator = Coordinator(self.catalog, plan_version=_digest(fingerprint),
                seed=self.seed, world_size=self.world_size, batch_size=self.batch_size,
                migration_start_step=self.migration_start_step, fingerprint=fingerprint)
        self._root(initialize)
        # Loader control traffic gets its own gloo group, so a background
        # prefetch can never interleave with the caller's NCCL backward.
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            try:
                self.control_group = dist.new_group(backend='gloo')
            except Exception:
                self.control_group = None
        self.prefetch = self.prefetch and (self.world_size == 1 or self.control_group is not None)
        if self.prefetch:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='strict-prefetch')
        # Pools are forked on the constructing thread, never from the prefetch
        # thread. Only rank zero plans, so only it owns planning processes.
        if self.rank == 0 and self.catalog is not None:
            self.catalog.start_plan_pool(self.plan_processes)
        if self.num_workers:
            self._pool = multiprocessing.get_context('fork').Pool(
                self.num_workers, initializer=_worker_init, initargs=(self.dataset,))

    def _root(self, fn):
        envelope = None
        if self.rank == 0:
            try:
                envelope = {'ok': True, 'value': fn()}
            except Exception as exc:
                envelope = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        if self.world_size > 1:
            values = [envelope]
            dist.broadcast_object_list(values, src=0, group=self.control_group)
            envelope = values[0]
        if not envelope['ok']:
            raise RuntimeError('Strict window protocol: ' + envelope['error'])
        return envelope['value']

    def _gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.control_group)
        return values

    def __iter__(self):
        return self

    def __len__(self):
        return max(1, len(self.batch_sampler) // self.world_size)

    def _next_slots(self):
        # Drop both bucket tails and global tails before any window reservation.
        for _ in range(2):
            if self._planner is None:
                self.batch_sampler.set_epoch(self._epoch)
                self._planner = itertools.islice(iter(self.batch_sampler), self._batch_offset, None)
            batches = []
            while len(batches) < self.world_size:
                try:
                    batch = next(self._planner)
                    self._batch_offset += 1
                except StopIteration:
                    break
                if len(batch) != self.batch_size:
                    continue
                batches.append(batch)
            if len(batches) == self.world_size:
                return [dict(episode_index=int(index), rank=rank, slot=slot)
                        for rank, batch in enumerate(batches) for slot, index in enumerate(batch)]
            self._epoch += 1
            self._batch_offset = 0
            self._planner = None
        raise RuntimeError('Quota plan cannot form a full global batch')

    def _take_slots(self):
        """Next slot list in planner order, with the position it started from."""
        if self._planned:
            return self._planned.popleft()
        position = (self._epoch, self._batch_offset)
        return position, self._next_slots()

    def _plan_ahead(self):
        """Plan the following slot list now and resolve its geometry in the pool.

        Slot selection reads only the batch sampler, never the ledger, so it can
        run a batch early without changing what any batch receives.
        """
        if self.catalog is None or self.catalog._plan_pool is None or self._planned:
            return
        position = (self._epoch, self._batch_offset)
        try:
            slots = self._next_slots()
        except Exception:
            self._epoch, self._batch_offset = position
            self._planner = None
            return
        self._planned.append((position, slots))
        self.catalog.prefill_ahead([slot['episode_index'] for slot in slots])

    def _load(self, requests):
        if not requests:
            return []
        if self.num_workers:
            if self._pool is None:
                self._pool = multiprocessing.get_context('fork').Pool(
                    self.num_workers, initializer=_worker_init, initargs=(self.dataset,))
            jobs = [self._pool.apply_async(_read_request, (request,)) for request in requests]
            results = []
            for request, job in zip(requests, jobs):
                try:
                    results.append(job.get(timeout=self.load_timeout))
                except Exception as exc:
                    # A timed-out read may still be running. End this pool and
                    # attempt; never issue a concurrent duplicate read request.
                    self._pool.terminate()
                    self._pool.join()
                    self._pool = None
                    return [dict(request_id=r['request_id'], status='FATAL', sample=None,
                                 error_code='WORKER_FAILED', error_detail=str(exc)) for r in requests]
            return results
        saved = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        global _WORKER_DATASET
        previous = _WORKER_DATASET
        _WORKER_DATASET = self.dataset
        try:
            return [_read_request(request) for request in requests]
        finally:
            _WORKER_DATASET = previous
            random.setstate(saved[0]); np.random.set_state(saved[1]); torch.random.set_rng_state(saved[2])

    def _audit_failure(self, request, status, attempt, error_code, detail, observed):
        """Persist a failed window's identity before it is replaced."""
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        failure = dict(step=int(request['step_id']), request=request, status=status, attempt=attempt,
                       error_code=error_code, error_detail=detail or '', observed_geometry=observed or {})
        with self.audit_path.with_name('strict_window_repairs.jsonl').open('a') as stream:
            stream.write(json.dumps(failure, sort_keys=True, allow_nan=False) + '\n')

    def _replace(self, request, error, scope, step, replacements, totals):
        """Release a rejected window and reserve its slot again."""
        slot_key = (request['rank'], request['slot'])
        replacements[slot_key] = replacements.get(slot_key, 0) + 1
        totals[0] += 1
        if totals[0] > self.max_step_replacements:
            raise RuntimeError(
                f'Step {step} needed more than {self.max_step_replacements} replacements; '
                f'the data store is failing, not individual windows (last: {error})')
        # Same source and bucket first; after repeated rejections in this slot,
        # another source of the same bucket. Never stop training over data.
        cross = replacements[slot_key] > self.same_source_replacements
        return self.coordinator.replace(request['request_id'], error_code=error,
                                        quarantine=scope, cross_source=cross).to_dict()

    def _screen(self, requests, step, replacements, totals, stage):
        """Replace, before any read, windows the reader's own filters would reject.

        The planning processes run the action half of the reader -- no latents
        -- for every request. Rejected windows get exactly the replace() the
        post-read repair would issue, in the same rank/slot order, so the batch
        is unchanged while its read needs one round instead of several.
        """
        pool = self.catalog._plan_pool if self.catalog is not None else None
        if pool is None or not requests:
            return requests
        current = {r['request_id']: r for r in requests}
        pending = list(requests)
        while pending:
            mark = time.monotonic()
            verdicts = pool.map(_plan_screen, pending, chunksize=1)
            stage['screen'] += time.monotonic() - mark
            rejected = sorted(((r, code) for r, code in zip(pending, verdicts) if code is not None),
                              key=lambda item: (item[0]['rank'], item[0]['slot']))
            pending = []
            for request, code in rejected:
                stage['screened'] += 1
                self._audit_failure(request, 'SCREENED', 0, code, 'rejected before read by the planning screen', {})
                new = self._replace(request, code, 'window', step, replacements, totals)
                current.pop(request['request_id'])
                current[new['request_id']] = new
                pending.append(new)
        return sorted(current.values(), key=lambda r: (r['rank'], r['slot']))

    def _stage_for_device(self, key, value):
        if not torch.is_tensor(value):
            return value
        dtype = self.device_dtypes.get(key)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype)
        return value if value.is_pinned() else value.pin_memory()

    def _prepare(self, step):
        """Reserve, read and collate exactly one global batch for `step`.

        Only the plan owner holds a coordinator, so every ledger operation below
        is local to rank zero and needs no collective.
        """
        stage = collections.defaultdict(float)
        began = time.monotonic()
        attempts, replacements, totals = {}, {}, [0, 0]

        def reserve():
            mark = time.monotonic()
            position, slots = self._take_slots()
            self._step_positions[int(step)] = position
            self.catalog.prefill_current([slot['episode_index'] for slot in slots])
            stage['plan'] += time.monotonic() - mark
            mark = time.monotonic()
            # Open the region that makes this reservation withdrawable: if the
            # step it serves never verifies, the windows go back untouched.
            if not self.coordinator._spec_open:
                self.coordinator.begin_speculation()
            self.coordinator.record_reservation_cursor()
            requests = [r.to_dict() for r in self.coordinator.reserve_global_batch(slots, step)]
            stage['ledger'] += time.monotonic() - mark
            # Screen before the look-ahead is queued: both share the pool.
            requests = self._screen(requests, step, replacements, totals, stage)
            mark = time.monotonic()
            self._plan_ahead()
            stage['ahead'] += time.monotonic() - mark
            return requests
        mark = time.monotonic()
        plan = self._root(reserve)
        stage['reserve'] += time.monotonic() - mark
        local_samples = {}
        current = {r['request_id']: r for r in plan}
        pending = plan
        started = time.monotonic()
        while pending:
            stage['rounds'] += 1
            local = [r for r in pending if r['rank'] == self.rank]
            mark = time.monotonic()
            try:
                results = self._load(local)
            except Exception as exc:
                if self._pool is not None:
                    self._pool.terminate()
                    self._pool.join()
                    self._pool = None
                results = [dict(request_id=r['request_id'], status='FATAL', sample=None,
                                error_code='LOCAL_LOAD_FAILED', error_detail=str(exc)) for r in local]
            stage['load'] += time.monotonic() - mark
            metadata = []
            for request, result in zip(local, results):
                if result.get('request_id') != request['request_id']:
                    result = dict(request_id=request['request_id'], status='FATAL', error_code='REQUEST_ID_MISMATCH')
                if result['status'] == 'OK':
                    if result.get('sample') is None:
                        result = dict(request_id=request['request_id'], status='FATAL', error_code='EMPTY_OK_SAMPLE')
                    else:
                        local_samples[request['slot']] = (request, result['sample'])
                metadata.append({k: v for k, v in result.items() if k != 'sample'})
            mark = time.monotonic()
            gathered = self._gather(metadata)
            stage['gather'] += time.monotonic() - mark

            def repair():
                retry, failed = [], []
                for result in [r for rows in gathered for r in rows]:
                    request_id = result['request_id']
                    request = current[request_id]
                    status = result['status']
                    if status == 'OK':
                        self.coordinator.mark_loaded([request_id])
                        continue
                    attempts[request_id] = attempts.get(request_id, 0) + 1
                    self._audit_failure(request, status, attempts[request_id], result.get('error_code'),
                                        result.get('error_detail', ''), result.get('observed_geometry', {}))
                    error = result.get('error_code') or 'INVALID_SAMPLE'
                    if status == 'FATAL':
                        # A reader that fails outright (worker crash, timeout)
                        # is infrastructure, not data: retry and replace a few
                        # times, then fail loudly rather than spin.
                        totals[1] += 1
                        if totals[1] > 64:
                            raise RuntimeError(f'Strict reader failing on step {step}: {error}: '
                                               f"{result.get('error_detail', '')[:500]}")
                    persistent = status == 'INVALID_SAMPLE' or any(
                        tag in error for tag in ('GEOMETRY', 'IDENTITY', 'MISMATCH'))
                    if not persistent and attempts[request_id] <= self.max_retries:
                        retry.append(request)  # transient I/O or worker failure: same window again
                        continue
                    if any(tag in error for tag in ('GEOMETRY', 'IDENTITY', 'MISMATCH')):
                        logger.warning('Strict window geometry/identity mismatch, replacing: %s %s',
                                       error, result.get('error_detail', '')[:300])
                    failed.append((request, error, 'window' if persistent else None))
                if failed:
                    # Resolve every replacement candidate's geometry in one
                    # parallel pass instead of one serial read per candidate.
                    self.catalog.prefill_current(self.coordinator.peek_replacements(
                        [request['request_id'] for request, _, _ in failed]))
                for request, error, scope in failed:
                    new = self._replace(request, error, scope, step, replacements, totals)
                    current.pop(request['request_id'])
                    current[new['request_id']] = new
                    retry.append(new)
                return retry

            mark = time.monotonic()
            pending = self._root(repair)
            stage['repair'] += time.monotonic() - mark
            # Root owns this map; other ranks only need their final local plan.
            for request in pending:
                current[request['request_id']] = request

        mark = time.monotonic()
        error = None
        batch = None
        try:
            if set(local_samples) != set(range(self.batch_size)):
                raise RuntimeError('Not all local slots loaded')
            ordered = [local_samples[i] for i in range(self.batch_size)]
            batch = ar_collate_fn([sample for _, sample in ordered])
            if batch is None or int(batch['latents'].shape[0]) != self.batch_size:
                raise RuntimeError('Collation changed strict local sample count')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self._gather(error)
        if any(value is not None for value in errors):
            raise RuntimeError(f'Strict collate failed: {errors}')
        plan = self._root(lambda: sorted(current.values(), key=lambda r: (r['rank'], r['slot'])))
        stage['collate'] += time.monotonic() - mark
        batch['_strict_request_ids'] = [local_samples[i][0]['request_id'] for i in range(self.batch_size)]
        if self.pin_memory:
            # Stage the host side of the step's host-to-device copy here, off the
            # step. A copy that also changes dtype first converts on the CPU into
            # a fresh pageable tensor, so pinning alone would not help: convert
            # to the step's target dtype (the identical CPU conversion), then
            # page-lock, and the step's copy is a plain DMA.
            mark = time.monotonic()
            batch = {key: self._stage_for_device(key, value) for key, value in batch.items()}
            stage['pin'] += time.monotonic() - mark
        self.last_load_seconds = time.monotonic() - started
        stage['total'] = time.monotonic() - began
        self.last_prep_timing = dict(stage, replacements=totals[0])
        if self.rank == 0 and self.timing_every and int(step) % self.timing_every == 0:
            logger.info('Strict prep step %d: total=%.3fs plan=%.3fs ledger=%.3fs screen=%.3fs screened=%d '
                        'ahead=%.3fs reserve(bcast)=%.3fs load=%.3fs gather=%.3fs repair=%.3fs rounds=%d '
                        'replacements=%d collate=%.3fs pin=%.3fs prev_commit_wait=%.3fs',
                        int(step), stage['total'], stage['plan'], stage['ledger'], stage['screen'],
                        int(stage['screened']), stage['ahead'], stage['reserve'], stage['load'],
                        stage['gather'], stage['repair'], int(stage['rounds']), totals[0],
                        stage['collate'], stage['pin'], self.last_commit_wait)
        return batch, plan

    def _start_prefetch(self, step, consume=None):
        """Prepare `step` on a background thread, overlapping the caller's compute.

        Runs on every rank: preparation issues control-plane collectives, so all
        ranks must enter them together. While it runs it is the only writer of
        the catalog, the coordinator and the reader pool. `consume` is the batch
        just handed to the caller; its windows are consumed first, so the ledger
        sees exactly the serial order of operations.
        """
        if not self.prefetch:
            return
        if self._prefetch_future is not None:
            raise RuntimeError('A strict prefetch is already outstanding')
        self._prefetch_step = int(step)
        self._prefetch_result = None
        self._prefetch_error = None
        self._prefetch_future = self._prefetch_pool.submit(self._prefetch_run, int(step), consume)

    def _prefetch_run(self, step, consume=None):
        try:
            if consume is not None:
                self._consume(*consume)
            self._prefetch_result = self._prepare(step)
        except BaseException as exc:  # re-raised verbatim on the caller's thread
            self._prefetch_error = exc

    def _consume(self, plan, step):
        """Consume a delivered batch's windows; committed_step waits for the update."""
        if self.rank != 0:
            return
        if not self.coordinator._spec_open:
            self.coordinator.begin_speculation()
        self.coordinator.speculate_commit([r['request_id'] for r in plan], step)

    def _drain_prefetch(self):
        """Join the background prefetch, keeping its batch for the next call."""
        future = self._prefetch_future
        if future is not None:
            future.result()
            self._prefetch_future = None
        if self._prefetch_error is not None:
            error, self._prefetch_error = self._prefetch_error, None
            raise error

    def _take_prefetch(self, step):
        """Return the batch prepared for `step`, or None when none is outstanding."""
        self._drain_prefetch()
        if self._prefetch_result is None:
            return None
        if self._prefetch_step != int(step):
            raise RuntimeError(f'Strict prefetch step {self._prefetch_step} is not requested step {step}')
        result, self._prefetch_result = self._prefetch_result, None
        self._prefetch_step = None
        return result

    def _withdraw_prefetch(self):
        """Drop prepared-but-undelivered work and the ledger state it produced.

        Called before checkpointing: the saved ledger must describe windows an
        optimizer step really consumed. Reservations are released, the request
        counter and group cursors return to their pre-reservation values, and
        the planner rewinds to where the first undelivered slot list started, so
        a resumed run and this one prepare the next batch identically.
        """
        withdrawn = self._prefetch_step
        self._prefetch_result = None
        self._prefetch_step = None
        if self.rank == 0:
            if withdrawn is not None and withdrawn > self.coordinator.committed_step:
                self.coordinator.withdraw_step(withdrawn)
            if self._step_positions:
                position = self._step_positions[min(self._step_positions)]
            elif self._planned:
                position = self._planned[0][0]
            else:
                position = None
            if position is not None:
                self._epoch, self._batch_offset = position
                self._planner = None
            self._step_positions.clear()
            self._planned.clear()

    def __next__(self):
        if self._pending is not None:
            raise RuntimeError('Previous strict batch has not been globally committed')
        step = self.committed_step + 1
        prepared = self._take_prefetch(step)
        if prepared is None:
            prepared = self._prepare(step)
        batch, plan = prepared
        self._step_positions.pop(int(step), None)
        self._pending = plan
        self._pending_step = int(step)
        if self.prefetch:
            # The background job consumes this batch first, then prepares the
            # next one; the caller returns without touching the ledger.
            self._start_prefetch(step + 1, consume=(plan, int(step)))
        else:
            self._consume(plan, int(step))
        return batch

    def commit_after_update(self, step, *, succeeded, agreed=False):
        """Publish the speculative position once the update is verified.

        `agreed=True` means the caller already reduced `succeeded` across all
        ranks (one tensor collective is far cheaper than an object gather);
        otherwise the ranks agree here.
        """
        started = time.monotonic()
        self._drain_prefetch()
        self.last_commit_wait = time.monotonic() - started
        checks = [bool(succeeded)] if agreed else self._gather(bool(succeeded))
        if not all(checks):
            if self.rank == 0:
                # Rewind to the last optimizer-verified step, so restoring a
                # complete checkpoint resumes exactly there.
                self.coordinator.rollback_speculation()
            self._pending = None
            self._pending_step = None
            raise RuntimeError('A rank failed the optimizer/finite check; restore a complete checkpoint')
        if self._pending is None:
            raise RuntimeError('No strict batch pending')
        if self._pending_step != int(step):
            raise RuntimeError('Strict ledger advanced for a different step than the finished update')
        plan = self._pending
        if self.rank == 0:
            if not self.coordinator._spec_open:
                raise RuntimeError('Strict speculation closed before its step was verified')
            self.coordinator.accept_speculation()
            if self.audit_path:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                record = dict(step=int(step), per_rank_counts=[self.batch_size] * self.world_size,
                              global_count=len(plan), plan_sha256=_digest(plan),
                              replacements=sum(r.get('replacement_of') is not None for r in plan))
                if step <= self.migration_start_step + 5 or step % 100 == 0:
                    record['requests'] = plan
                with self.audit_path.open('a') as stream:
                    stream.write(json.dumps(record, sort_keys=True) + '\n')
        # The loader's own position must follow the published step, exactly as
        # the serial path did when it broadcast the commit result.
        self.committed_step = int(step)
        self._pending = None
        self._pending_step = None

    def save_state(self, checkpoint_dir, step):
        # Withdraw a prepared-but-unverified batch before writing. Rolling it
        # back also restores the request-number allocation, so the batch is
        # rebuilt bit-identically when the loader asks for it again.
        self._drain_prefetch()
        self._withdraw_prefetch()

        def save():
            if self._pending is not None or self.coordinator.committed_step != int(step):
                raise RuntimeError('Checkpoint and strict consumption boundary differ')
            state = dict(schema='motus-strict-window-loader-v1', fingerprint=self.fingerprint,
                         committed_step=int(step), epoch=self._epoch, batch_offset=self._batch_offset,
                         resolved_catalog=self.catalog.resolved, ledger=self.coordinator.state_dict())
            path = Path(checkpoint_dir) / 'strict_window_state.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, sort_keys=True, allow_nan=False) + '\n')
            temporary.replace(path)
            return dict(schema=state['schema'], file=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), committed_step=int(step))
        return self._root(save)

    def load_state(self, checkpoint_dir, step, *, saved_source_root=None):
        def restore():
            path = Path(checkpoint_dir) / 'strict_window_state.json'
            marker_path = Path(checkpoint_dir) / 'checkpoint_complete.json'
            marker = json.loads(marker_path.read_text())
            strict = marker.get('strict_window')
            if not path.exists():
                raise RuntimeError('Strict ledger missing; refusing sampler reset')
            if not strict or strict.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError('Strict ledger hash/manifest mismatch')
            state = json.loads(path.read_text())
            if state['schema'] != 'motus-strict-window-loader-v1':
                raise RuntimeError('Strict data plan schema changed')
            appended_at = None
            if self.append_sources_on_resume:
                appended_at = self.coordinator.append_checkpoint_sources(
                    state['ledger'], self.append_sources_on_resume)
            if appended_at is not None:
                if any(int(index) >= appended_at for index in state['resolved_catalog']):
                    raise RuntimeError('Saved resolved catalog overlaps newly appended sources')
                state['fingerprint'] = self.fingerprint
                logger.info(
                    'Strict append-only resume: preserved %d old catalog entries and %d episode states; '
                    'fresh sources=%s, new catalog entries=%d',
                    appended_at, len(state['ledger']['episodes']), self.append_sources_on_resume,
                    len(self.catalog) - appended_at)
            relocation = None
            if state['fingerprint'] != self.fingerprint:
                if self.skip_fingerprint_check:
                    logger.warning(
                        'strict_fingerprint_check disabled: using saved strict-window fingerprint '
                        '(current data plan differs from checkpoint)'
                    )
                    self.coordinator.fingerprint = state['fingerprint']
                    self.coordinator.plan_version = state['ledger']['binding']['plan_version']
                else:
                    from .strict_fingerprint import reconcile_source_relocation
                    relocation = reconcile_source_relocation(
                        state['fingerprint'], self.fingerprint,
                        saved_source_root=saved_source_root,
                        current_source_root=Path(__file__).resolve().parents[1],
                        saved_plan_version=state['ledger']['binding']['plan_version'])
                    # Keep the saved identity: request IDs own augmentation RNG.
                    self.coordinator.fingerprint = relocation['fingerprint']
                    self.coordinator.plan_version = relocation['plan_version']
            if state['committed_step'] != int(step) or strict['committed_step'] != int(step):
                raise RuntimeError('Model and ledger steps differ')
            self.catalog.resolved = {int(k): v for k, v in state['resolved_catalog'].items()}
            binding = state['ledger']['binding']
            ledger_fp = binding.get('fingerprint')
            top_fp = state.get('fingerprint')
            self.coordinator.plan_version = binding['plan_version']
            self.coordinator.seed = int(binding['seed'])
            self.coordinator.world_size = int(binding['world_size'])
            self.coordinator.batch_size = int(binding['batch_size'])
            self.coordinator.migration_start_step = int(binding['migration_start_step'])
            self.coordinator.catalog_sha256 = binding['catalog_sha256']
            if top_fp is not None:
                self.coordinator.fingerprint = top_fp
            if (
                self.skip_fingerprint_check
                and top_fp is not None
                and ledger_fp is not None
                and ledger_fp != top_fp
            ):
                logger.warning(
                    'Strict checkpoint: ledger binding is stale; using checkpoint top '
                    'fingerprint and rebinding ledger (strict_fingerprint_check=false)'
                )
                self.coordinator.fingerprint = top_fp
                state['ledger']['binding'] = self.coordinator._binding()
            elif (
                not self.skip_fingerprint_check
                and ledger_fp is not None
                and top_fp is not None
                and ledger_fp != top_fp
            ):
                logger.warning(
                    'Strict checkpoint: top-level fingerprint differs from ledger binding; '
                    'using ledger binding for restore'
                )
                self.coordinator.fingerprint = ledger_fp
            if self.resample_sources:
                from .strict_window_sampling import resample_ledger_sources
                binding_seed = int(binding['seed'])
                summary = resample_ledger_sources(
                    state['ledger'], self.catalog, self.resample_sources, binding_seed)
                if self.rank == 0:
                    cleared = summary['episodes'] + summary['requests_dropped']
                    logger.warning(
                        'Strict resample sources: %s — cleared %d windows '
                        '(episodes_reset=%d inflight_dropped=%d group_cursors_reset=%d)',
                        ', '.join(self.resample_sources), cleared,
                        summary['episodes'], summary['requests_dropped'],
                        summary['groups'])
            self.coordinator.load_state_dict(state['ledger'])
            if self.coordinator.committed_step != int(step):
                raise RuntimeError('Coordinator commit boundary differs')
            self._epoch, self._batch_offset = int(state['epoch']), int(state['batch_offset'])
            self._planner = None
            self._planned.clear()
            self._step_positions.clear()
            return dict(committed_step=int(step), migrated=False,
                        source_relocations=relocation['relocations'] if relocation else [],
                        _effective_fingerprint=relocation['fingerprint'] if relocation else None)
        result = self._root(restore)
        effective = result.pop('_effective_fingerprint', None)
        if effective is not None:
            self.fingerprint = effective
        self.committed_step = result['committed_step']
        self._restored = True
        return result

    def close(self, abort=False):
        if self._prefetch_future is not None and not abort:
            try:
                self._prefetch_future.result()
            except Exception:
                pass
        self._prefetch_future = None
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=not abort)
            self._prefetch_pool = None
        if self.catalog is not None:
            self.catalog.close()
        if self._pool is not None:
            if abort:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None
