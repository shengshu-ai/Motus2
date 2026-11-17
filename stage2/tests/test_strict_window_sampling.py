"""Protocol regression tests: no Torch/CUDA/distributed runtime required."""
from array import array
from copy import deepcopy
import json
import random
import unittest

from data.strict_window_sampling import Coordinator, LedgerMismatchError, WindowExhaustedError, WindowRequest


def catalog(count=8, total=11, window=8):
    return [dict(episode_index=i, episode_id=f"source/episode-{i}", source_id="s", bucket_id="b",
                 total_latents=total, window_size=window, latent_variant={"phase": 0, "views": 2})
            for i in range(count)]


def coordinator(rows=None, world=2, batch=2, **kwargs):
    return Coordinator(rows if rows is not None else catalog(), "test-plan-v1", world_size=world,
                       batch_size=batch, migration_start_step=10000, **kwargs)


def slots(indices, batch=2):
    return [dict(episode_index=ep, rank=i // batch, slot=i % batch) for i, ep in enumerate(indices)]


def load_commit(authority, plan, step):
    ids = [r.request_id for r in plan]
    authority.mark_loaded(ids)
    authority.commit(ids, step)


class StrictWindowTests(unittest.TestCase):
    def test_append_source_keeps_consumption_and_rejects_changed_prefix(self):
        old_rows = catalog(2, total=9, window=8)
        old = coordinator(old_rows, world=1, batch=1)
        plan = old.reserve_global_batch(slots([0], batch=1), 10001)
        load_commit(old, plan, 10001)
        saved = old.state_dict()
        extra = dict(old_rows[-1], episode_index=2, episode_id="new/episode-0",
                     source_id="new", bucket_id="new-bucket")
        new = coordinator(old_rows + [extra], world=1, batch=1)
        migrated = deepcopy(saved)
        self.assertEqual(new.append_checkpoint_sources(migrated, ["new"]), 2)
        new.load_state_dict(migrated)
        restored = new.state_dict()
        self.assertEqual(restored["episodes"], saved["episodes"])
        self.assertEqual(restored["next_request"], saved["next_request"])
        self.assertEqual(restored["committed_step"], saved["committed_step"])
        self.assertEqual([row for row in restored["group_cursors"] if row["source_id"] == "s"],
                         saved["group_cursors"])
        continued = new.reserve_global_batch(slots([0], batch=1), 10002)
        self.assertNotEqual(continued[0].window_start, plan[0].window_start)
        load_commit(new, continued, 10002)
        fresh = new.reserve_global_batch(slots([2], batch=1), 10003)
        self.assertEqual(fresh[0].cycle, 0)

        reordered = [dict(old_rows[1], episode_index=0), dict(old_rows[0], episode_index=1), extra]
        bad = coordinator(reordered, world=1, batch=1)
        with self.assertRaisesRegex(LedgerMismatchError, "catalog prefix changed"):
            bad.append_checkpoint_sources(deepcopy(saved), ["new"])

    def test_request_json_contract(self):
        c = coordinator()
        req = c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)[0]
        self.assertEqual(WindowRequest.from_dict(json.loads(json.dumps(req.to_dict()))), req)

    def test_not_loaded_or_partial_commit_cannot_advance(self):
        c = coordinator()
        plan = c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        before = c.state_dict()
        with self.assertRaises(ValueError):
            c.commit([r.request_id for r in plan], 10001)
        self.assertEqual(before, c.state_dict())
        c.mark_loaded([r.request_id for r in plan])
        with self.assertRaises(ValueError):
            c.commit([r.request_id for r in plan[:-1]], 10001)
        with self.assertRaises(ValueError):
            c.commit([r.request_id for r in plan], 10002)
        self.assertEqual(c.committed_step, 10000)
        c.commit([r.request_id for r in plan], 10001)
        self.assertEqual(c.committed_step, 10001)

    def test_global_32_by_4_and_full_coverage_before_repeat(self):
        rows = catalog(132, total=9, window=8)  # Two legal starts each.
        c = coordinator(rows, world=32, batch=4)
        seen = {}
        # Deliberately exclude the same four tail episodes; no reservation was
        # made for tail slots and their windows must remain untouched.
        for step in range(10001, 10009):
            plan = c.reserve_global_batch(slots(list(range(128)), batch=4), step)
            self.assertEqual(len(plan), 128)
            self.assertEqual(len({(r.episode_id, r.window_start, r.cycle) for r in plan}), 128)
            for req in plan:
                key = (req.episode_id, req.cycle)
                self.assertNotIn(req.window_start, seen.setdefault(key, set()))
                if req.cycle:
                    self.assertEqual(seen[(req.episode_id, req.cycle - 1)], {0, 1})
                seen[key].add(req.window_start)
            load_commit(c, plan, step)
        self.assertFalse(any(str(i) in c.state_dict()["episodes"] for i in range(128, 132)))

    def test_last_loaded_or_reserved_window_blocks_next_cycle(self):
        c = coordinator(catalog(1, total=9, window=8), world=1, batch=1)
        first = c.reserve_global_batch(slots([0], batch=1), 10001)
        second = c.reserve_global_batch(slots([0], batch=1), 10002)
        self.assertEqual({first[0].window_start, second[0].window_start}, {0, 1})
        c.mark_loaded([second[0].request_id])
        before = c.state_dict()
        with self.assertRaises(WindowExhaustedError):
            c.reserve_global_batch(slots([0], batch=1), 10003)
        self.assertEqual(c.state_dict(), before)
        load_commit(c, first, 10001)
        with self.assertRaises(WindowExhaustedError):
            c.reserve_global_batch(slots([0], batch=1), 10003)
        c.commit([second[0].request_id], 10002)
        next_cycle = c.reserve_global_batch(slots([0], batch=1), 10003)
        self.assertEqual(next_cycle[0].cycle, 1)

    def test_replacement_owns_new_window_and_success_slots_stay_reserved(self):
        c = coordinator(catalog(2, total=9, window=8), world=1, batch=2)
        plan = c.reserve_global_batch(slots([0, 1], batch=2), 10001)
        c.mark_loaded([plan[1].request_id])
        new = c.replace(plan[0].request_id, "WRIST_RELATIVE", quarantine="window")
        self.assertEqual(new.replacement_of, plan[0].request_id)
        self.assertNotEqual(new.window_start, plan[0].window_start)
        self.assertEqual((new.source_id, new.bucket_id), (plan[0].source_id, plan[0].bucket_id))
        self.assertEqual(c._requests[plan[1].request_id]["status"], "LOADED")
        plan[0] = new
        load_commit(c, plan, 10001)
        for step in range(10002, 10007):
            plan = c.reserve_global_batch(slots([0, 1], batch=2), step)
            self.assertFalse(any(r.episode_index == 0 and r.window_start == 0 for r in plan))
            load_commit(c, plan, step)

    def test_fallback_does_not_borrow_normal_request_window(self):
        rows = catalog(3, total=8, window=8)
        c = coordinator(rows, world=1, batch=2)
        plan = c.reserve_global_batch(slots([0, 1], batch=2), 10001)
        new = c.replace(plan[0].request_id, "FILE_MISSING", quarantine="episode")
        self.assertEqual(new.episode_index, 2)  # B/0 already reserved normally.
        self.assertNotEqual((new.episode_id, new.window_start), (plan[1].episode_id, plan[1].window_start))

    def test_long_rejected_episode_yields_to_other_episode_before_repair_budget(self):
        c = coordinator(catalog(2, total=80, window=8), world=1, batch=1)
        old = c.reserve_global_batch(slots([0], batch=1), 10001)[0]
        self.assertGreater(c._count(c._entry(0)), 33)
        new = c.replace(old.request_id, "WRIST_RELATIVE_FILTER", quarantine="window")
        self.assertEqual(new.episode_index, 1)
        self.assertEqual((new.source_id, new.bucket_id), (old.source_id, old.bucket_id))
        state = c._states[0]
        self.assertEqual(set(state.quarantined_starts), {old.window_start})
        self.assertIsNone(state.episode_quarantine)
        load_commit(c, [new], 10001)
        # Normal requests still prefer this episode; the failure did not ban
        # its unfailed windows or consume them on the successful replacement.
        later = c.reserve_global_batch(slots([0], batch=1), 10002)[0]
        self.assertEqual(later.episode_index, 0)
        self.assertNotEqual(later.window_start, old.window_start)
        self.assertEqual(later.cycle, 0)

    def test_replacement_same_episode_last_fallback_and_quarantine_survives_cycle(self):
        rows = catalog(3, total=9, window=8)
        rows[1]["source_id"] = "other-source"
        rows[2]["bucket_id"] = "other-bucket"
        c = coordinator(rows, world=1, batch=1)
        old = c.reserve_global_batch(slots([0], batch=1), 10001)[0]
        new = c.replace(old.request_id, "WRIST_RELATIVE_FILTER")
        self.assertEqual(new.episode_index, 0)
        self.assertNotEqual(new.window_start, old.window_start)
        self.assertEqual(new.cycle, old.cycle)
        load_commit(c, [new], 10001)
        state = json.loads(json.dumps(c.state_dict()))
        restored = coordinator(rows, world=1, batch=1)
        restored.load_state_dict(state)
        for step in range(10002, 10006):
            left = c.reserve_global_batch(slots([0], batch=1), step)
            right = restored.reserve_global_batch(slots([0], batch=1), step)
            self.assertEqual([r.to_dict() for r in left], [r.to_dict() for r in right])
            self.assertEqual(left[0].episode_index, 0)
            self.assertNotEqual(left[0].window_start, old.window_start)
            self.assertGreater(left[0].cycle, old.cycle)
            load_commit(c, left, step)
            load_commit(restored, right, step)

    def test_replacement_skips_loaded_and_reserved_other_episode_windows(self):
        c = coordinator(catalog(2, total=9, window=8), world=1, batch=1)
        old = c.reserve_global_batch(slots([0], batch=1), 10001)[0]
        future = c.reserve_global_batch(slots([1], batch=1), 10002)[0]
        later = c.reserve_global_batch(slots([1], batch=1), 10003)[0]
        c.mark_loaded([future.request_id])
        before_other = deepcopy(c._states[1])
        new = c.replace(old.request_id, "WRIST_RELATIVE_FILTER")
        self.assertEqual(new.episode_index, 0)  # Other episode's last cycle is inflight.
        self.assertEqual(c._states[1], before_other)
        self.assertEqual(c._requests[future.request_id]["status"], "LOADED")
        self.assertEqual(c._requests[later.request_id]["status"], "RESERVED")
        state = json.loads(json.dumps(c.state_dict()))
        restored = coordinator(catalog(2, total=9, window=8), world=1, batch=1)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        for authority in (c, restored):
            load_commit(authority, [new], 10001)
            authority.commit([future.request_id], 10002)
            load_commit(authority, [later], 10003)
        next_left = c.reserve_global_batch(slots([1], batch=1), 10004)
        next_right = restored.reserve_global_batch(slots([1], batch=1), 10004)
        self.assertEqual(next_left[0].cycle, 1)
        self.assertEqual([r.to_dict() for r in next_left], [r.to_dict() for r in next_right])

    def test_exhausted_source_falls_back_within_bucket_never_across(self):
        rows = catalog(3, total=8, window=8)
        rows[1]["source_id"] = "other"
        rows[2]["bucket_id"] = "other"
        c = coordinator(rows, world=1, batch=1)
        plan = c.reserve_global_batch(slots([0], batch=1), 10001)
        # The slot's own source has no other window: the bucket's other source
        # serves it, so a spent source never stops training.
        new = c.replace(plan[0].request_id, "BAD_WINDOW")
        self.assertEqual((new.episode_index, new.bucket_id), (1, plan[0].bucket_id))
        before = c.state_dict()
        # Nothing left in the bucket: a different bucket is never used.
        with self.assertRaises(WindowExhaustedError):
            c.replace(new.request_id, "BAD_WINDOW")
        self.assertEqual(before, c.state_dict())

    def test_cancel_does_not_consume_or_leave_gaps(self):
        c = coordinator(catalog(1), world=1, batch=2)
        plan = c.reserve_global_batch(slots([0, 0], batch=2), 10001)
        starts = [r.window_start for r in plan]
        c.mark_loaded([plan[0].request_id])
        c.cancel_all_uncommitted()
        fresh = c.reserve_global_batch(slots([0, 0], batch=2), 10001)
        self.assertEqual([r.window_start for r in fresh], starts)
        self.assertEqual(c.committed_step, 10000)
        self.assertNotEqual(fresh[0].request_id, plan[0].request_id)

    def test_temporary_failure_none_does_not_quarantine_or_immediately_reuse(self):
        c = coordinator(catalog(1, total=9, window=8), world=1, batch=1)
        old = c.reserve_global_batch(slots([0], batch=1), 10001)[0]
        new = c.replace(old.request_id, "IO_RETRY_EXHAUSTED", quarantine=None)
        self.assertNotEqual(new.window_start, old.window_start)
        self.assertFalse(c.state_dict()["episodes"]["0"]["quarantined_starts"])
        load_commit(c, [new], 10001)
        next_req = c.reserve_global_batch(slots([0], batch=1), 10002)[0]
        self.assertEqual(next_req.window_start, old.window_start)
        self.assertEqual(next_req.cycle, old.cycle)

    def test_restore_pending_replacements_and_rng_cursor_sequence(self):
        rows = catalog(8, total=10, window=8)
        c = coordinator(rows, fingerprint={"source_sha": "abc", "window_rule": "relative5-v1"})
        plan = c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        c.mark_loaded([plan[2].request_id])
        plan[0] = c.replace(plan[0].request_id, "WRIST_RELATIVE")
        future = c.reserve_global_batch(slots([0, 1, 2, 3]), 10002)
        state = json.loads(json.dumps(c.state_dict()))
        restored = coordinator(rows, fingerprint={"source_sha": "abc", "window_rule": "relative5-v1"})
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        for authority in (c, restored):
            load_commit(authority, plan, 10001)
            load_commit(authority, future, 10002)
        for step in range(10003, 10020):
            left = c.reserve_global_batch(slots([0, 1, 2, 3]), step)
            right = restored.reserve_global_batch(slots([0, 1, 2, 3]), step)
            self.assertEqual([r.to_dict() for r in left], [r.to_dict() for r in right])
            load_commit(c, left, step)
            load_commit(restored, right, step)

    def test_binding_and_tampered_cursor_fail_closed_without_mutation(self):
        c = coordinator()
        c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        original = c.state_dict()
        for field, value in (("seed", 99), ("world_size", 4), ("batch_size", 4),
                             ("plan_version", "different"), ("catalog_sha256", "wrong"),
                             ("fingerprint", {"x": "new"}), ("window_rule_sha256", "wrong")):
            broken = deepcopy(original)
            broken["binding"][field] = value
            with self.assertRaises(LedgerMismatchError):
                c.load_state_dict(broken)
            self.assertEqual(c.state_dict(), original)
        broken = deepcopy(original)
        broken["requests"][0]["request"]["window_start"] += 1
        with self.assertRaises(LedgerMismatchError):
            c.load_state_dict(broken)
        self.assertEqual(c.state_dict(), original)
        broken = deepcopy(original)
        broken["requests"] = []
        with self.assertRaises(LedgerMismatchError):
            c.load_state_dict(broken)
        self.assertEqual(c.state_dict(), original)

    def test_duplicate_slots_ids_and_unknown_ids_rejected(self):
        c = coordinator()
        with self.assertRaises(ValueError):
            c.reserve_global_batch([dict(rank=0, slot=0, episode_index=0)] * 2, 10001)
        plan = c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        with self.assertRaises(ValueError):
            c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        with self.assertRaises(ValueError):
            c.mark_loaded([plan[0].request_id] * 2)
        with self.assertRaises(ValueError):
            c.mark_loaded(["wrong"])

    def test_lazy_unknown_geometry_is_not_resolved_in_group_build(self):
        class LazyCatalog:
            def __init__(self):
                self.rows = catalog(100)
                for row in self.rows:
                    row["total_latents"] = None
                self.calls = []
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, i):
                return self.rows[i]
            def resolve(self, i):
                self.calls.append(i)
                return dict(self.rows[i], total_latents=11)
        rows = LazyCatalog()
        c = coordinator(rows)
        self.assertEqual(rows.calls, [])
        self.assertIsInstance(c._groups[("s", "b")], array)
        self.assertEqual(len(c._states), 0)
        c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        self.assertEqual(rows.calls, [0, 1, 2, 3])
        self.assertEqual(len(c._states), 4)

    def test_resolver_identity_or_restore_geometry_drift_rejected(self):
        class Resolving(list):
            drift = False
            bad_identity = False
            def resolve(self, i):
                return dict(self[i], total_latents=11 + int(self.drift),
                            source_id="wrong" if self.bad_identity else "s")
        rows = Resolving(catalog())
        for row in rows:
            row["total_latents"] = None
        c = coordinator(rows)
        c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        state = c.state_dict()
        rows.drift = True
        with self.assertRaises(LedgerMismatchError):
            coordinator(rows).load_state_dict(state)
        rows.drift, rows.bad_identity = False, True
        with self.assertRaises(LedgerMismatchError):
            coordinator(rows).reserve_global_batch(slots([0, 1, 2, 3]), 10001)

    def test_resolution_failures_bounded_no_payload_scan(self):
        class Unreadable(list):
            calls = 0
            def resolve(self, i):
                self.calls += 1
                raise OSError("bounded missing file")
        rows = Unreadable(catalog(100))
        c = coordinator(rows)
        with self.assertRaisesRegex(WindowExhaustedError, "resolution_errors"):
            c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        self.assertEqual(rows.calls, 8)
        self.assertEqual(c.state_dict()["requests"], [])

    def test_out_of_order_load_and_checkpoint_after_cancellation(self):
        c = coordinator(catalog(8, total=12, window=8), world=2, batch=2)
        rng = random.Random(42)
        for step in range(10001, 10021):
            plan = c.reserve_global_batch(slots([0, 0, 1, 1]), step)
            rng.shuffle(plan)
            for req in plan:
                c.mark_loaded([req.request_id])
            c.commit([r.request_id for r in plan], step)
            c.reserve_global_batch(slots([0, 0, 1, 1]), step + 1)
            c.cancel_all_uncommitted()
            saved = json.loads(json.dumps(c.state_dict()))
            fresh = coordinator(catalog(8, total=12, window=8), world=2, batch=2)
            fresh.load_state_dict(saved)
            self.assertEqual(saved, fresh.state_dict())
            c = fresh

    def test_randomized_replacement_commit_and_json_resume_no_repeats(self):
        rows = catalog(8, total=15, window=8)
        c = coordinator(rows)
        rng, seen, quarantine = random.Random(1729), {}, {}
        for step in range(10001, 10101):
            picks = [rng.randrange(len(rows)) for _ in range(4)]
            plan = c.reserve_global_batch(slots(picks), step)
            if step % 7 == 0:
                old = plan[0]
                plan[0] = c.replace(old.request_id, "LOCAL_TRACKING_JUMP")
                quarantine.setdefault(old.episode_index, set()).add(old.window_start)
            rng.shuffle(plan)
            for req in plan:
                c.mark_loaded([req.request_id])
                key = req.episode_index, req.cycle
                self.assertNotIn(req.window_start, seen.setdefault(key, set()))
                if req.cycle:
                    covered = seen[(req.episode_index, req.cycle - 1)] | quarantine.get(req.episode_index, set())
                    self.assertEqual(covered, set(range(8)))
                seen[key].add(req.window_start)
            c.commit([r.request_id for r in plan], step)
            if step % 5 == 0:
                state = json.loads(json.dumps(c.state_dict()))
                restored = coordinator(rows)
                restored.load_state_dict(state)
                self.assertEqual(restored.state_dict(), state)
                c = restored

    def test_migration_reset_starts_new_ledger_not_inferred_old_history(self):
        first, second = coordinator(), coordinator()
        a = first.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        b = second.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        self.assertEqual(a, b)
        self.assertEqual({r.cycle for r in a}, {0})
        self.assertEqual(first.committed_step, 10000)

    def test_actual_catalog_or_source_rule_change_refuses_restore(self):
        c = coordinator()
        c.reserve_global_batch(slots([0, 1, 2, 3]), 10001)
        state = c.state_dict()
        changed = catalog()
        changed[0]["episode_id"] = "new-data-version/episode-0"
        with self.assertRaises(LedgerMismatchError):
            coordinator(changed).load_state_dict(state)
        with self.assertRaises(LedgerMismatchError):
            coordinator(fingerprint={"relative_threshold": 1.5}).load_state_dict(state)

    def test_episode_quarantine_with_sibling_inflight_requires_explicit_replan(self):
        c = coordinator(catalog(3), world=1, batch=2)
        plan = c.reserve_global_batch(slots([0, 0], batch=2), 10001)
        before = c.state_dict()
        with self.assertRaisesRegex(ValueError, "cancel/replan"):
            c.replace(plan[0].request_id, "WHOLE_FILE_SCHEMA", quarantine="episode")
        self.assertEqual(c.state_dict(), before)

    def test_lazy_catalog_retained_and_group_storage_linear_in_episode_count(self):
        class VirtualCatalog:
            calls = 0
            def __len__(self):
                return 4096
            def __getitem__(self, index):
                self.calls += 1
                return dict(episode_index=index, episode_id=f"ep{index}", source_id="s", bucket_id="b",
                            total_latents=None, window_size=8, latent_variant={"phase": 0})
        virtual = VirtualCatalog()
        c = coordinator(virtual)
        self.assertIs(c.catalog, virtual)
        self.assertEqual(virtual.calls, len(virtual))
        self.assertEqual(len(c._states), 0)
        self.assertEqual(sum(v.itemsize * len(v) for v in c._groups.values()), 4 * len(virtual))

    def test_duplicate_identity_rejected_even_across_source_labels(self):
        rows = catalog(2)
        rows[1]["episode_id"] = rows[0]["episode_id"]
        with self.assertRaisesRegex(LedgerMismatchError, "Duplicate physical"):
            coordinator(rows)
        rows[1]["source_id"] = "another-label-for-same-recording"
        with self.assertRaisesRegex(LedgerMismatchError, "Duplicate physical"):
            coordinator(rows)
        rows[1]["total_latents"] = 1234
        with self.assertRaisesRegex(LedgerMismatchError, "Duplicate physical"):
            coordinator(rows)  # Conflicting duration must not evade the check.

    def test_same_episode_distinct_bound_variant_or_layout_is_separate_domain(self):
        for field, value in (("latent_variant", {"phase": 1, "views": 2}),
                             ("bucket_id", "different-explicit-layout"), ("window_size", 7)):
            rows = catalog(2)
            rows[1]["episode_id"] = rows[0]["episode_id"]
            rows[1][field] = value
            c = coordinator(rows, world=1, batch=2)
            plan = c.reserve_global_batch(slots([0, 1], batch=2), 10001)
            self.assertEqual({r.episode_index for r in plan}, {0, 1})

    def test_duplicate_unknown_geometry_base_spec_rejected_without_resolve(self):
        class Unknown(list):
            def resolve(self, index):
                raise AssertionError("Constructor must never resolve payloads")
        rows = Unknown(catalog(2))
        for row in rows:
            row["total_latents"], row["latent_variant"] = None, {}
        rows[1]["episode_id"] = rows[0]["episode_id"]
        with self.assertRaisesRegex(LedgerMismatchError, "Duplicate physical"):
            coordinator(rows)

    def test_lazy_resolution_convergence_rejected_and_transaction_rolled_back(self):
        class Converging(list):
            def resolve(self, index):
                return dict(self[index], total_latents=9, window_size=8, latent_variant={"phase": 0})
        rows = Converging(catalog(2))
        rows[1]["episode_id"] = rows[0]["episode_id"]
        rows[0]["window_size"], rows[1]["window_size"] = 16, 32
        c = coordinator(rows, world=1, batch=2)
        initial = c.state_dict()
        with self.assertRaisesRegex(LedgerMismatchError, "converges"):
            c.reserve_global_batch(slots([0, 1], batch=2), 10001)
        self.assertEqual(c.state_dict(), initial)
        self.assertEqual(c._resolved_owners, {})
        # The failed transaction must not leave a ghost owner behind.
        result = c.reserve_global_batch([dict(episode_index=1, rank=0, slot=0)], 10001)
        self.assertEqual(result[0].episode_index, 1)
        self.assertEqual(set(c._resolved_owners.values()), {1})

    def test_resolved_owner_map_rebuilt_after_json_restore_and_refuses_later_collision(self):
        class Resolving(list):
            def resolve(self, index):
                return dict(self[index], total_latents=9, window_size=8, latent_variant={"phase": 0})
        rows = Resolving(catalog(2))
        rows[1]["episode_id"] = rows[0]["episode_id"]
        rows[0]["window_size"], rows[1]["window_size"] = 16, 32
        c = coordinator(rows, world=1, batch=1)
        plan = c.reserve_global_batch(slots([0], batch=1), 10001)
        load_commit(c, plan, 10001)
        checkpoint = json.loads(json.dumps(c.state_dict()))
        restored = coordinator(rows, world=1, batch=1)
        restored.load_state_dict(checkpoint)
        self.assertEqual(c._resolved_owners, restored._resolved_owners)
        before = restored.state_dict()
        with self.assertRaisesRegex(LedgerMismatchError, "converges"):
            restored.reserve_global_batch(slots([1], batch=1), 10002)
        self.assertEqual(restored.state_dict(), before)

    def test_tampered_resolved_state_cannot_introduce_second_owner(self):
        class Resolving(list):
            def resolve(self, index):
                return dict(self[index], total_latents=9, window_size=8, latent_variant={"phase": 0})
        rows = Resolving(catalog(2))
        rows[1]["episode_id"] = rows[0]["episode_id"]
        rows[0]["window_size"], rows[1]["window_size"] = 16, 32
        c = coordinator(rows, world=1, batch=1)
        plan = c.reserve_global_batch(slots([0], batch=1), 10001)
        load_commit(c, plan, 10001)
        original = c.state_dict()
        tampered = deepcopy(original)
        tampered["episodes"]["1"] = deepcopy(tampered["episodes"]["0"])
        with self.assertRaisesRegex(LedgerMismatchError, "converges"):
            c.load_state_dict(json.loads(json.dumps(tampered)))
        self.assertEqual(c.state_dict(), original)
        self.assertEqual(set(c._resolved_owners.values()), {0})


if __name__ == "__main__":
    unittest.main()


def test_resample_ledger_sources_resets_one_source_only():
    catalog = [
        dict(episode_index=0, episode_id='a', source_id='HandsOn', bucket_id='b',
             total_latents=4, window_size=2, latent_variant={}),
        dict(episode_index=1, episode_id='c', source_id='Other', bucket_id='b',
             total_latents=4, window_size=2, latent_variant={}),
    ]
    ledger = dict(
        episodes={
            '0': dict(cycle=1, committed_prefix=2, committed_tail=[], cursor=2,
                      quarantined_starts={}, episode_quarantine=None, geometry={}),
            '1': dict(cycle=2, committed_prefix=1, committed_tail=[], cursor=1,
                      quarantined_starts={}, episode_quarantine=None, geometry={}),
        },
        group_cursors=[
            dict(source_id='HandsOn', bucket_id='b', cursor=1),
            dict(source_id='Other', bucket_id='b', cursor=1),
        ],
        requests=[
            dict(request=dict(source_id='HandsOn', episode_index=0, episode_id='a', bucket_id='b',
                              request_id='x:1', plan_version='p', step_id=290001, rank=0, slot=0,
                              window_start=0, window_size=2, cycle=1, latent_variant={},
                              expected_total_latents=4, replacement_of=None),
                   ordinal=0, status='RESERVED'),
        ],
    )
    from data.strict_window_sampling import resample_ledger_sources
    summary = resample_ledger_sources(ledger, catalog, ['HandsOn'], seed=42)
    assert summary['episodes'] == 1
    assert summary['requests_dropped'] == 1
    assert ledger['episodes']['0']['committed_prefix'] == 0
    assert ledger['episodes']['1']['committed_prefix'] == 1
    assert len(ledger['requests']) == 0
