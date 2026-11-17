import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.samplers import WeightedExpandedBatchSampler
from data.strict_window_loader import StrictWindowLoader
from data.strict_window_checkpoint import validate


class ToyDataset:
    def __init__(self, count=132, inject=False):
        self.episode_files = [{'id': str(i), 'sc_total_latent_frames': 3} for i in range(count)]
        self.inject = inject

    def __len__(self): return len(self.episode_files)
    def get_bucket_key(self, index): return 'same'
    def get_sample_weight(self, index): return 1.
    def _episode_identity(self, ep): return ep['id']
    def _sampling_source_identity(self, ep): return 'source'
    def _window_size_for_episode(self, ep, index): return 2
    def window_spec(self, index, resolve_geometry=True):
        return dict(episode_index=index, episode_id=str(index), source_id='source', bucket_id='same',
                    total_latents=3, window_size=2, latent_variant={'path': str(index)})
    def strict_window_verdict(self, request):
        """The action-side rejection read_window applies, without reading."""
        if self.inject and request['episode_index'] == 0 and request['window_start'] == 0:
            return 'WRIST_RELATIVE_LIMIT'
        return None
    def read_window(self, request):
        if self.inject and request['episode_index'] == 0 and request['window_start'] == 0:
            return dict(request_id=request['request_id'], status='INVALID_SAMPLE', sample=None,
                        error_code='WRIST_RELATIVE_LIMIT', error_detail='injected single-window failure')
        return dict(request_id=request['request_id'], status='OK', error_code=None,
                    observed_geometry={'total_latents': 3},
                    sample={'latents': torch.full((1, 2, 1, 1), float(request['episode_index'])),
                            'actions': torch.zeros(4, 58), 'states': torch.zeros(2, 58), 'valid_len': 2})


def loader(rank=0, world=1, count=132, inject=False, workers=0, prefetch=True, plan_processes=0):
    dataset = ToyDataset(count, inject)
    sampler = WeightedExpandedBatchSampler(dataset, batch_size=4, drop_last=True, seed=42,
                                          window_start_sampling=False)
    return StrictWindowLoader(dataset, sampler, rank=rank, world_size=world, batch_size=4,
                              fingerprint={'test': 'v1', 'world': world}, migration_start_step=10000,
                              num_workers=workers, prefetch=prefetch, plan_processes=plan_processes)


def checkpoint(obj, path, step):
    path.mkdir()
    binding = obj.save_state(path, step)
    p = path / binding['file']
    (path/'checkpoint_complete.json').write_text(json.dumps(dict(global_step=step, world_size=obj.world_size,
        files={p.name: p.stat().st_size}, strict_window=binding)))


def test_complete_checkpoint_restores_exact_next_requests(tmp_path):
    obj = loader(count=12, inject=True)
    for step in range(10001, 10005):
        batch = next(obj)
        assert batch['latents'].shape[0] == 4
        obj.commit_after_update(step, succeeded=True)
    checkpoint(obj, tmp_path/'checkpoint', 10004)
    assert validate(tmp_path/'checkpoint', 10004, 1, 4)['passed']
    expected = next(obj)['_strict_request_ids']
    restored = loader(count=12, inject=True)
    restored.load_state(tmp_path/'checkpoint', 10004)
    assert next(restored)['_strict_request_ids'] == expected
    obj.close(); restored.close()


def test_real_workers_and_no_consumption_before_commit():
    obj = loader(count=8, workers=2)
    assert next(obj)['latents'].shape[0] == 4
    assert obj.coordinator.committed_step == 10000
    with pytest.raises(RuntimeError, match='not been globally committed'):
        next(obj)
    obj.commit_after_update(10001, succeeded=True)
    obj.close()


@pytest.mark.parametrize('budget', [1, 32])
def test_failed_window_identity_is_durable_and_never_stops_training(tmp_path, budget):
    """A rejected window is recorded before repair; running out of retries in
    one source never ends the step -- it only changes where the slot is filled."""
    obj = loader(count=8)
    obj.audit_path = tmp_path / 'strict_windows.jsonl'
    obj.same_source_replacements = budget
    obj._next_slots = lambda: [dict(episode_index=i, rank=0, slot=i) for i in range(4)]
    read = obj.dataset.read_window
    def reject_episode(request):
        if request['episode_index'] == 0:
            return dict(request_id=request['request_id'], status='INVALID_SAMPLE', sample=None,
                        error_code='WRIST_RELATIVE_FILTER', error_detail='known rejected trajectory',
                        observed_geometry={'total_latents': 3})
        return read(request)
    obj.dataset.read_window = reject_episode
    try:
        assert next(obj)['latents'].shape[0] == 4
        failures = [json.loads(line) for line in (tmp_path / 'strict_window_repairs.jsonl').read_text().splitlines()]
        assert failures and failures[0]['request']['episode_id'] == '0'
        assert failures[0]['request']['rank'] == failures[0]['request']['slot'] == 0
        assert failures[0]['step'] == 10001 and failures[0]['error_code'] == 'WRIST_RELATIVE_FILTER'
        assert failures[0]['observed_geometry'] == {'total_latents': 3}
        assert not obj.audit_path.exists() and obj.coordinator.committed_step == 10000
    finally:
        obj.close(abort=True)


class TwoSourceDataset(ToyDataset):
    """Even episodes come from a source whose every window is rejected."""
    def _sampling_source_identity(self, ep):
        return 'bad' if int(ep['id']) % 2 == 0 else 'good'
    def window_spec(self, index, resolve_geometry=True):
        return dict(super().window_spec(index, resolve_geometry), source_id='bad' if index % 2 == 0 else 'good')
    def strict_window_verdict(self, request):
        return 'WRIST_RELATIVE_FILTER' if request['source_id'] == 'bad' else None
    def read_window(self, request):
        if request['source_id'] == 'bad':
            return dict(request_id=request['request_id'], status='INVALID_SAMPLE', sample=None,
                        error_code='WRIST_RELATIVE_FILTER', error_detail='source always rejected',
                        observed_geometry={'total_latents': 3})
        return super().read_window(request)


def test_rejecting_source_falls_back_to_bucket_peers_instead_of_exiting():
    dataset = TwoSourceDataset(64)
    sampler = WeightedExpandedBatchSampler(dataset, batch_size=4, drop_last=True, seed=42,
                                           window_start_sampling=False)
    obj = StrictWindowLoader(dataset, sampler, rank=0, world_size=1, batch_size=4,
                             fingerprint={'test': 'v1', 'world': 1}, migration_start_step=10000,
                             num_workers=0, same_source_replacements=2)
    # Every slot starts in the always-rejected source: only fallback can fill it.
    obj._next_slots = lambda: [dict(episode_index=2 * i, rank=0, slot=i) for i in range(4)]
    try:
        for step in range(10001, 10004):
            batch = next(obj)
            assert batch['latents'].shape[0] == 4
            assert all(r['source_id'] == 'good' for r in obj._pending)
            assert all(int(r['episode_index']) % 2 == 1 for r in obj._pending)
            obj.commit_after_update(step, succeeded=True)
        assert obj.coordinator.committed_step == 10003
    finally:
        obj.close()


def test_missing_ledger_refuses_sampler_reset(tmp_path):
    obj = loader(count=8)
    marker = tmp_path/'checkpoint_complete.json'
    marker.write_text(json.dumps({'global_step':10000,'world_size':1,'files':{}}))
    with pytest.raises(RuntimeError, match='Strict ledger missing'):
        obj.load_state(tmp_path,10000)


def _distributed_case(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world)
    obj = loader(rank, world, count=max(132, world*4+4), inject=True)
    committed = []
    try:
        for step in range(10001,10005):
            batch = next(obj)
            assert batch['latents'].shape[0] == 4
            obj.commit_after_update(step,succeeded=True)
            committed.extend(batch['_strict_request_ids'])
        Path(output, f'rank{rank}.json').write_text(json.dumps(committed))
    finally:
        obj.close()
        dist.destroy_process_group()


def test_actual_gloo_control_flow_and_repair(tmp_path):
    world = int(os.environ.get('STRICT_TEST_WORLD_SIZE', '2'))
    mp.start_processes(_distributed_case, args=(world,str(tmp_path/'rendezvous'),str(tmp_path)),
                       nprocs=world, join=True, start_method='fork')
    rows = [json.loads((tmp_path/f'rank{i}.json').read_text()) for i in range(world)]
    assert all(len(row)==16 for row in rows)
    assert len({request for row in rows for request in row}) == world*16


def _distributed_failure_case(rank, rendezvous, output, case):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    obj = loader(rank, 2, count=16)
    try:
        if case == 'load' and rank == 1:
            def broken(_requests):
                raise RuntimeError('injected rank-local loader failure')
            obj._load = broken
        with pytest.raises(RuntimeError):
            next(obj)
            obj.commit_after_update(10001, succeeded=(rank != 1))
        assert obj.committed_step == 10000
        Path(output, f'{case}-rank{rank}').write_text('FAILED_TOGETHER_WITHOUT_COMMIT')
    finally:
        obj.close(abort=True)
        dist.destroy_process_group()


@pytest.mark.parametrize('case', ['load', 'update'])
def test_rank_local_failure_is_global_and_uncommitted(tmp_path, case):
    mp.start_processes(_distributed_failure_case, args=(str(tmp_path/'rendezvous'),str(tmp_path),case),
                       nprocs=2, join=True, start_method='fork')
    assert all((tmp_path/f'{case}-rank{i}').exists() for i in range(2))


def test_sidecar_tampering_rejected(tmp_path):
    obj = loader(count=8)
    next(obj)
    obj.commit_after_update(10001, succeeded=True)
    checkpoint(obj, tmp_path/'checkpoint', 10001)
    sidecar = tmp_path/'checkpoint/strict_window_state.json'
    sidecar.write_text(sidecar.read_text().replace('10001','10002'))
    with pytest.raises(ValueError):
        validate(tmp_path/'checkpoint',10001,1,4)
    restored = loader(count=8)
    with pytest.raises(RuntimeError):
        restored.load_state(tmp_path/'checkpoint',10001)


def test_prefetch_reproduces_the_serial_stream_exactly():
    """Prefetch may only move work in time; it may not change the stream."""
    def stream(prefetch, plan_processes=0, steps=5):
        obj = loader(count=132, prefetch=prefetch, plan_processes=plan_processes)
        try:
            rows = []
            for step in range(10001, 10001 + steps):
                batch = next(obj)
                rows.append((batch['_strict_request_ids'],
                             float(batch['latents'].sum()),
                             [(r['request_id'], r['episode_index'], r['window_start'])
                              for r in obj._pending]))
                obj.commit_after_update(step, succeeded=True)
            return rows
        finally:
            obj.close()

    serial = stream(prefetch=False)
    assert serial == stream(prefetch=True)
    # Planning in forked processes, one batch ahead, only moves work in time too.
    assert serial == stream(prefetch=True, plan_processes=2)


def test_unverified_batch_is_withdrawn_before_checkpoint(tmp_path):
    """A prepared-but-unverified batch must not enter the saved ledger.

    Saving happens after a verified update, when the batch for the *next* step
    is already prepared. That prepared reservation must be withdrawn, so the
    restored loader re-reserves the same windows instead of finding them
    already occupied.
    """
    obj = loader(count=132)
    for step in range(10001, 10003):
        next(obj)
        obj.commit_after_update(step, succeeded=True)
    assert obj._prefetch_step == 10003          # prepared but unverified
    checkpoint(obj, tmp_path/'checkpoint', 10002)
    saved = json.loads((tmp_path/'checkpoint/strict_window_state.json').read_text())
    assert saved['committed_step'] == 10002
    assert saved['ledger']['requests'] == []     # no reservation in the ledger

    expected = next(obj)['_strict_request_ids']
    obj.close()
    restored = loader(count=132)
    restored.load_state(tmp_path/'checkpoint', 10002)
    assert next(restored)['_strict_request_ids'] == expected
    restored.close()


def test_planning_screen_matches_post_read_repair_and_needs_one_read_round():
    """Screening replaces filter-rejected windows before the read with the same
    replace() calls the repair would make, so the stream is bit-identical."""
    def stream(plan_processes, steps=6):
        dataset = ToyDataset(12, inject=True)
        sampler = WeightedExpandedBatchSampler(dataset, batch_size=4, drop_last=True, seed=42,
                                               window_start_sampling=False)
        obj = StrictWindowLoader(dataset, sampler, rank=0, world_size=1, batch_size=4,
                                 fingerprint={'test': 'v1', 'world': 1}, migration_start_step=10000,
                                 num_workers=0, plan_processes=plan_processes)
        rows, rounds = [], []
        try:
            for step in range(10001, 10001 + steps):
                batch = next(obj)
                rows.append((batch['_strict_request_ids'], float(batch['latents'].sum()),
                             [(r['request_id'], r['episode_index'], r['window_start']) for r in obj._pending]))
                rounds.append(obj.last_prep_timing.get('rounds'))
                obj.commit_after_update(step, succeeded=True)
        finally:
            obj.close()
        return rows, rounds

    repaired, repaired_rounds = stream(0)
    screened, screened_rounds = stream(2)
    assert screened == repaired
    assert max(repaired_rounds) > 1, 'the injected rejection must cost the unscreened run a round'
    assert set(screened_rounds) == {1}, 'screened batches are read in a single round'


def test_rejecting_source_is_screened_out_before_any_read():
    dataset = TwoSourceDataset(64)
    sampler = WeightedExpandedBatchSampler(dataset, batch_size=4, drop_last=True, seed=42,
                                           window_start_sampling=False)
    obj = StrictWindowLoader(dataset, sampler, rank=0, world_size=1, batch_size=4,
                             fingerprint={'test': 'v1', 'world': 1}, migration_start_step=10000,
                             num_workers=0, same_source_replacements=2, plan_processes=2)
    obj._next_slots = lambda: [dict(episode_index=2 * i, rank=0, slot=i) for i in range(4)]
    try:
        for step in range(10001, 10004):
            batch = next(obj)
            assert batch['latents'].shape[0] == 4
            assert all(r['source_id'] == 'good' for r in obj._pending)
            assert obj.last_prep_timing['rounds'] == 1
            obj.commit_after_update(step, succeeded=True)
    finally:
        obj.close()


def test_commit_with_caller_agreed_result_skips_the_gather_but_keeps_its_contract():
    def forbid(value):
        raise AssertionError('agreed commit must not gather')
    for succeeded in (False, True):
        # Without prefetch nothing but the commit can gather after next().
        obj = loader(count=8, prefetch=False)
        try:
            next(obj)
            obj._gather = forbid
            if succeeded:
                obj.commit_after_update(10001, succeeded=True, agreed=True)
                assert obj.coordinator.committed_step == 10001 and obj.committed_step == 10001
            else:
                with pytest.raises(RuntimeError, match='finite check'):
                    obj.commit_after_update(10001, succeeded=False, agreed=True)
                assert obj.coordinator.committed_step == 10000 and obj.committed_step == 10000
        finally:
            obj.close(abort=True)
