"""CPU authority for reserved, loaded and globally committed training windows.

This module has no tensor, distributed or worker dependency. The caller owns
quota planning, collective readiness and the optimizer boundary. Workers must
load the exact request and report failures; they must never substitute samples.
The guarantee begins at ``migration_start_step``, not at an inferred old cursor.
"""

from array import array
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .window_start import select_window_start, window_start_count


SCHEMA = "motus-strict-window-ledger-v1"
WINDOW_RULE = "data.window_start.select_window_start-v1"
WINDOW_RULE_SHA256 = hashlib.sha256(Path(__file__).with_name("window_start.py").read_bytes()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value):
    return value if isinstance(value, str) else _json(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


class WindowExhaustedError(RuntimeError):
    """The requested source/bucket cannot provide an unreserved valid window."""


class LedgerMismatchError(ValueError):
    """Checkpoint identity or ledger consistency differs from this authority."""


@dataclass(frozen=True)
class WindowRequest:
    request_id: str
    plan_version: str
    step_id: int
    rank: int
    slot: int
    source_id: str
    bucket_id: str
    episode_index: int
    episode_id: str
    window_start: int
    window_size: int
    cycle: int
    latent_variant: dict
    expected_total_latents: int
    replacement_of: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**deepcopy(value))


@dataclass
class _EpisodeState:
    cycle: int = 0
    # Prefix compression avoids one Python entry per consumed window on the
    # usual ordered path. Only out-of-order completion needs a sparse tail.
    committed_prefix: int = 0
    committed_tail: set = field(default_factory=set)
    cursor: int = 0
    quarantined_starts: dict = field(default_factory=dict)
    episode_quarantine: Optional[dict] = None
    inflight: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)

    def committed(self, ordinal):
        return ordinal < self.committed_prefix or ordinal in self.committed_tail

    def consume(self, ordinal):
        self.committed_tail.add(ordinal)
        while self.committed_prefix in self.committed_tail:
            self.committed_tail.remove(self.committed_prefix)
            self.committed_prefix += 1


class Coordinator:
    """Single-process authority; it is intentionally not a thread-safe service.

    ``catalog`` stays a lazy, immutable Sequence. Its episode_index must equal
    its Sequence position. Only compact group-index arrays and a content digest
    are built eagerly; per-episode ledger state is created on first use.
    Quota weights belong to the caller's final slot planner. Replacement walks
    the same source/bucket deterministically and never borrows an ordinal.
    """

    def __init__(self, catalog: Sequence[dict], plan_version: str, seed: int = 42,
                 world_size: int = 32, batch_size: int = 4,
                 migration_start_step: int = 10000, fingerprint: Optional[dict] = None):
        self.catalog = catalog
        self.plan_version = str(plan_version)
        self.seed = _integer(seed, "seed")
        self.world_size = _integer(world_size, "world_size")
        self.batch_size = _integer(batch_size, "batch_size")
        self.migration_start_step = _integer(migration_start_step, "migration_start_step")
        if not self.plan_version or self.world_size < 1 or self.batch_size < 1 or self.migration_start_step < 0:
            raise ValueError("Invalid coordinator dimensions/version/migration step")
        self.fingerprint = deepcopy(fingerprint or {})
        _json(self.fingerprint)
        self.committed_step = self.migration_start_step
        self._next_request = 0
        self._states: Dict[int, _EpisodeState] = {}
        self._requests: Dict[str, dict] = {}
        self._groups = {}
        self._group_cursors = {}
        self._undo = None
        self._resolved_owners = {}
        self._undo_owners = None
        # A speculation spans one delivered-but-unverified step: consumption may
        # run ahead of the published committed_step, and must be withdrawable.
        self._spec_open = False
        self._spec_snapshot = None
        self._spec_committed_step = None
        self._reserved_snapshot = None
        if len(catalog) == 0:
            raise ValueError("An empty catalog cannot serve a training plan")
        digest = hashlib.sha256()
        identities = set()
        index_type = "I" if len(catalog) <= (1 << 32) else "Q"
        for index in range(len(catalog)):
            entry = self._entry(index)
            # Source labels cannot make the same physical window universe into
            # independent ledgers. Keep only 128-bit hashes during this pass,
            # rather than retaining millions of duplicate identity strings.
            # A hash collision rejects conservatively; it cannot allow repeats.
            identity_digest = self._identity_digest(entry)
            if identity_digest in identities:
                raise LedgerMismatchError(
                    f"Duplicate physical episode/window identity at episode_index={index}, "
                    f"episode_id={entry['episode_id']!r}, identity_sha128={identity_digest.hex()}; "
                    "independent catalog indices cannot own the same window ledger"
                )
            identities.add(identity_digest)
            group = self._group(entry)
            if group not in self._groups:
                self._groups[group] = array(index_type)
            self._groups[group].append(index)
            digest.update(_json(entry).encode("utf-8") + b"\n")
        self.catalog_sha256 = digest.hexdigest()
        for group, indices in self._groups.items():
            key = _json([self.seed, *group]).encode("utf-8")
            self._group_cursors[group] = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(indices)

    def _entry(self, index, resolve=False):
        _integer(index, "episode_index")
        if not 0 <= index < len(self.catalog):
            raise ValueError("episode_index outside catalog")
        base = self.catalog[index]
        raw = base
        if resolve:
            resolver = getattr(self.catalog, "resolve", None)
            if resolver is not None:
                raw = dict(base)
                raw.update(resolver(index))
                for key in ("episode_index", "episode_id", "source_id", "bucket_id"):
                    if _identity(raw[key]) != _identity(base[key]):
                        raise LedgerMismatchError(f"Resolver changed catalog identity {key}")
        if _integer(raw["episode_index"], "catalog episode_index") != index:
            raise ValueError("Catalog episode_index must equal Sequence position")
        if any(raw[key] is None or raw[key] == "" for key in ("episode_id", "source_id", "bucket_id")):
            raise ValueError("Catalog episode/source/bucket identities must be explicit")
        total, window = raw.get("total_latents"), raw.get("window_size")
        variant = raw.get("latent_variant")
        if total is not None:
            total = _integer(total, "total_latents")
        if window is not None:
            window = _integer(window, "window_size")
        if ((total is not None and total < 1) or (window is not None and window < 1)
                or (variant is not None and not isinstance(variant, dict))):
            raise ValueError("Catalog geometry must be positive when known; latent_variant must be a dict")
        if resolve and (total is None or window is None or variant is None):
            raise ValueError("Selected episode has no resolved exact geometry/variant")
        entry = dict(episode_index=index, episode_id=_identity(raw["episode_id"]),
                     source_id=_identity(raw["source_id"]), bucket_id=_identity(raw["bucket_id"]),
                     total_latents=total, window_size=window,
                     latent_variant=deepcopy(variant))
        if "frame_stride" in raw:
            entry["frame_stride"] = raw["frame_stride"]
        if "weight" in raw:
            entry["weight"] = raw["weight"]
        _json(entry)
        return entry

    @staticmethod
    def _group(entry):
        return entry["source_id"], entry["bucket_id"]

    @staticmethod
    def _identity_digest(entry):
        identity = _json([entry["episode_id"], entry["window_size"],
                          entry["latent_variant"], entry["bucket_id"]]).encode("utf-8")
        return hashlib.sha256(identity).digest()[:16]

    @staticmethod
    def _count(entry):
        return window_start_count(entry["total_latents"], entry["window_size"])

    @staticmethod
    def _start(entry, cycle, ordinal):
        return int(select_window_start(cycle * Coordinator._count(entry) + ordinal,
                                       entry["total_latents"], entry["window_size"], entry["episode_id"]))

    @staticmethod
    def _geometry(entry):
        return {k: deepcopy(entry[k]) for k in ("total_latents", "window_size", "latent_variant", "frame_stride") if k in entry}

    def _state(self, index, entry=None):
        if self._undo is not None and index not in self._undo:
            self._undo[index] = deepcopy(self._states.get(index))
        if index not in self._states:
            self._states[index] = _EpisodeState()
        state = self._states[index]
        if entry is not None:
            geometry = self._geometry(entry)
            if state.geometry and state.geometry != geometry:
                raise LedgerMismatchError("Resolved geometry changed after a window was reserved")
            identity = self._identity_digest(entry)
            owner = self._resolved_owners.get(identity)
            if owner is not None and owner != index:
                raise LedgerMismatchError(
                    f"Resolved physical episode/window identity converges across indices {owner} and {index}, "
                    f"identity_sha128={identity.hex()}"
                )
            if self._undo_owners is not None and identity not in self._undo_owners:
                self._undo_owners[identity] = owner
            self._resolved_owners[identity] = index
            state.geometry = geometry
        return state

    @contextmanager
    def _transaction(self):
        if self._spec_open:
            # The enclosing speculation owns the undo log; a nested failure
            # withdraws the whole speculation rather than a single step.
            try:
                yield
            except BaseException:
                self.rollback_speculation()
                raise
            return
        if self._undo is not None:
            raise RuntimeError("Coordinator operations cannot be nested")
        self._undo = {}
        self._undo_owners = {}
        requests_before = dict(self._requests)
        cursors_before = dict(self._group_cursors)
        counter_before = self._next_request
        try:
            yield
        except BaseException:
            for index, before in self._undo.items():
                if before is None:
                    self._states.pop(index, None)
                else:
                    self._states[index] = before
            self._requests = requests_before
            self._group_cursors = cursors_before
            self._next_request = counter_before
            for identity, before in self._undo_owners.items():
                if before is None:
                    self._resolved_owners.pop(identity, None)
                else:
                    self._resolved_owners[identity] = before
            raise
        finally:
            self._undo = None
            self._undo_owners = None

    def _available(self, entry, excluded):
        state = self._state(entry["episode_index"], entry)
        count = self._count(entry)
        if state.episode_quarantine or len(state.quarantined_starts) == count:
            return None
        while True:
            ordinal = max(state.cursor, state.committed_prefix)
            while ordinal < count:
                start = self._start(entry, state.cycle, ordinal)
                unavailable = state.committed(ordinal) or ordinal in state.inflight or start in state.quarantined_starts
                if unavailable:
                    if state.cursor <= ordinal and not any(
                        i == entry["episode_index"] and c == state.cycle and o < ordinal
                        for i, c, o in excluded
                    ):
                        state.cursor = ordinal + 1
                elif (entry["episode_index"], state.cycle, ordinal) not in excluded:
                    return state, ordinal, start
                ordinal += 1
            if state.inflight:
                return None  # The last RESERVED/LOADED window blocks next cycle.
            if state.committed_prefix + len(state.committed_tail) + len(state.quarantined_starts) != count:
                return None  # An explicitly excluded, unconsumed window remains.
            state.cycle += 1
            state.committed_prefix = state.cursor = 0
            state.committed_tail.clear()

    def _reserve(self, preferred, step_id, rank, slot, replacement_of=None, excluded=frozenset(),
                 prefer_other_episode=False, groups=None):
        preferred_entry = self._entry(preferred)
        group = self._group(preferred_entry)
        found, selected, errors = None, None, []
        walk = [group] if groups is None else list(groups)

        def others_of(g):
            # Error retries are bounded; scanning already-resolved blocked states
            # is finite in group size and does not cap the number of legal starts.
            indices, cursor = self._groups[g], self._group_cursors[g]
            for offset in range(len(indices)):
                pos = (cursor + offset) % len(indices)
                if int(indices[pos]) != preferred:
                    yield int(indices[pos]), g, pos

        others = chain.from_iterable(others_of(g) for g in walk)
        # A rejected window does not prove the entire episode invalid. Prefer
        # another episode in the same quota group before spending the repair
        # budget on adjacent, potentially correlated windows of this episode.
        # Its unfailed windows stay eligible as a last fallback and for later
        # normal reservations. All choices still pass _available's barriers.
        candidates = (chain(others, ((preferred, None, None),)) if prefer_other_episode
                      else chain(((preferred, None, None),), others))
        for index, g, pos in candidates:
            try:
                entry = self._entry(index, resolve=True)
                found = self._available(entry, excluded)
            except LedgerMismatchError:
                raise
            except Exception as exc:
                errors.append(f"episode_index={index}: {type(exc).__name__}: {exc}")
                if len(errors) >= 8:
                    break
                continue
            if found is not None:
                selected = entry
                if pos is not None:
                    self._group_cursors[g] = (pos + 1) % len(self._groups[g])
                break
        if found is None:
            raise WindowExhaustedError(f"No available window in source/bucket {walk}; committed/inflight barriers retained; resolution_errors={errors}")
        state, ordinal, start = found
        request_id = f"{self.plan_version}:{self._next_request}"
        self._next_request += 1
        request = WindowRequest(request_id, self.plan_version, step_id, rank, slot,
                                selected["source_id"], selected["bucket_id"], selected["episode_index"],
                                selected["episode_id"], start, selected["window_size"], state.cycle,
                                deepcopy(selected["latent_variant"]), selected["total_latents"], replacement_of)
        state.inflight[ordinal] = request_id
        if state.cursor == ordinal:
            state.cursor += 1
        self._requests[request_id] = dict(request=request, ordinal=ordinal, status="RESERVED")
        return request

    def reserve_global_batch(self, slots: List[dict], step_id: int) -> List[WindowRequest]:
        step_id = _integer(step_id, "step_id")
        if step_id <= self.committed_step or not slots:
            raise ValueError("Reservations need a nonempty future step")
        occupied = {(r["request"].step_id, r["request"].rank, r["request"].slot) for r in self._requests.values()}
        parsed = []
        for value in slots:
            rank, slot = _integer(value["rank"], "rank"), _integer(value["slot"], "slot")
            if not 0 <= rank < self.world_size or not 0 <= slot < self.batch_size:
                raise ValueError("Rank/slot outside global batch")
            key = (step_id, rank, slot)
            if key in occupied:
                raise ValueError("Duplicate reserved step/rank/slot")
            occupied.add(key)
            parsed.append((value["episode_index"], rank, slot))
        with self._transaction():
            return [self._reserve_slot(index, step_id, rank, slot) for index, rank, slot in parsed]

    def _reserve_slot(self, index, step_id, rank, slot):
        """A planned slot; if its source/bucket is spent, the bucket's other sources."""
        try:
            return self._reserve(index, step_id, rank, slot)
        except WindowExhaustedError:
            group = self._group(self._entry(index))
            others = [g for g in self._fallback_groups(group) if g != group]
            if not others:
                raise
            return self._reserve(index, step_id, rank, slot, groups=others)

    def mark_loaded(self, request_ids: List[str]):
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate loaded request IDs")
        records = [self._record(request_id) for request_id in request_ids]
        for record in records:
            record["status"] = "LOADED"

    def _record(self, request_id):
        if request_id not in self._requests:
            raise ValueError(f"Unknown or terminal request ID {request_id}")
        record = self._requests[request_id]
        request = record["request"]
        entry = self._entry(request.episode_index, resolve=True)
        expected = (entry["episode_id"], entry["source_id"], entry["bucket_id"], entry["window_size"],
                    entry["total_latents"], entry["latent_variant"])
        actual = (request.episode_id, request.source_id, request.bucket_id, request.window_size,
                  request.expected_total_latents, request.latent_variant)
        state = self._states[request.episode_index]
        if (actual != expected or self._geometry(entry) != state.geometry
                or self._resolved_owners.get(self._identity_digest(entry)) != request.episode_index
                or request.plan_version != self.plan_version or request.cycle != state.cycle
                or state.inflight.get(record["ordinal"]) != request_id
                or request.window_start != self._start(entry, request.cycle, record["ordinal"])):
            raise LedgerMismatchError("Request no longer matches its exact catalog/window reservation")
        return record

    def _release(self, request_id):
        record = self._record(request_id)
        request, ordinal = record["request"], record["ordinal"]
        state = self._state(request.episode_index)
        del state.inflight[ordinal]
        state.cursor = min(state.cursor, ordinal)
        del self._requests[request_id]
        return request, ordinal

    def _fallback_groups(self, group):
        """Groups a slot may draw from once its own group keeps rejecting.

        Same bucket only -- the bucket is what makes a local batch collate --
        other sources first in a fixed rotation after the slot's own source,
        and the slot's own group last, so a single-source bucket still serves.
        """
        source, bucket = group
        same = sorted(g for g in self._groups if g[1] == bucket and g != group)
        return [g for g in same if g[0] > source] + [g for g in same if g[0] <= source] + [group]

    def peek_replacements(self, request_ids, extra=1):
        """Episodes the next replacements will try first. Read-only, for prefetch."""
        wanted, preferred = {}, {}
        for request_id in request_ids:
            record = self._requests.get(request_id)
            if record is None:
                continue
            index = record["request"].episode_index
            group = self._group(self._entry(index))
            wanted[group] = wanted.get(group, 0) + 1
            preferred.setdefault(group, set()).add(index)
        out = []
        for group, count in wanted.items():
            indices, cursor = self._groups[group], self._group_cursors[group]
            taken = 0
            for offset in range(len(indices)):
                index = int(indices[(cursor + offset) % len(indices)])
                if index in preferred[group]:
                    continue
                out.append(index)
                taken += 1
                if taken >= count + extra:
                    break
        return out

    def replace(self, request_id: str, error_code: str, quarantine="window",
                cross_source=False) -> WindowRequest:
        """Release a failed request and reserve its slot again.

        The replacement comes from the slot's own source and bucket unless
        `cross_source` is set or that group has no window left; then from the
        same bucket's other sources. Running out of one source's data is never
        a reason to stop training.
        """
        if quarantine not in (None, "window", "episode") or not isinstance(error_code, str) or not error_code.strip():
            raise ValueError("Replacement requires an explicit error code and quarantine scope")
        with self._transaction():
            old, ordinal = self._release(request_id)
            state = self._state(old.episode_index)
            reason = dict(error_code=str(error_code), request_id=old.request_id,
                          plan_version=self.plan_version, step_id=old.step_id)
            if quarantine == "window":
                state.quarantined_starts[old.window_start] = reason
            elif quarantine == "episode":
                # The caller may use this scope only for confirmed whole-file
                # or schema errors; a local quality rejection is a window.
                state.episode_quarantine = reason
                if state.inflight:
                    raise ValueError("Episode quarantine has other inflight requests; cancel/replan the batch explicitly")
            excluded = {(old.episode_index, old.cycle, ordinal)}
            group = self._group(self._entry(old.episode_index))
            if cross_source:
                groups = self._fallback_groups(group)
            else:
                try:
                    return self._reserve(old.episode_index, old.step_id, old.rank, old.slot, old.request_id,
                                         excluded, prefer_other_episode=True)
                except WindowExhaustedError:
                    # This source/bucket is spent; draw from the bucket's others.
                    groups = [g for g in self._fallback_groups(group) if g != group]
                    if not groups:
                        raise
            return self._reserve(old.episode_index, old.step_id, old.rank, old.slot, old.request_id,
                                 excluded, prefer_other_episode=True, groups=groups)

    def commit(self, request_ids: List[str], step_id: int):
        step_id = _integer(step_id, "step_id")
        if step_id != self.committed_step + 1:
            raise ValueError("Commit must be the next globally successful optimizer step")
        required = self.world_size * self.batch_size
        if len(request_ids) != required or len(set(request_ids)) != required:
            raise ValueError("Commit requires all distinct world_size * batch_size requests")
        records = [self._record(request_id) for request_id in request_ids]
        expected_ids = {key for key, r in self._requests.items() if r["request"].step_id == step_id}
        positions = {(r["request"].rank, r["request"].slot) for r in records}
        if (set(request_ids) != expected_ids or len(positions) != required
                or any(r["status"] != "LOADED" or r["request"].step_id != step_id for r in records)):
            raise ValueError("Commit contains unloaded, missing, wrong-step or duplicate slots")
        with self._transaction():
            for request_id in request_ids:
                request, ordinal = self._release(request_id)
                self._state(request.episode_index).consume(ordinal)
            self.committed_step = step_id

    def begin_speculation(self):
        """Open a rollback-able region covering every ledger change until accepted."""
        if self._spec_open or self._undo is not None:
            raise RuntimeError("Coordinator operations cannot be nested")
        self._spec_open = True
        self._undo = {}
        self._undo_owners = {}
        self._spec_committed_step = self.committed_step
        self._spec_snapshot = dict(requests=dict(self._requests),
                                   cursors=dict(self._group_cursors),
                                   next_request=self._next_request,
                                   committed_step=self.committed_step)
        self._reserved_snapshot = None

    def speculate_commit(self, request_ids, step_id):
        """Consume a delivered batch without advancing the published step.

        Identical to :meth:`commit` except the new position is pending rather
        than published, so an unverified step can still be withdrawn.
        """
        if not self._spec_open:
            raise RuntimeError("Speculative commit requires an open speculation")
        step_id = _integer(step_id, "step_id")
        if step_id != self._spec_committed_step + 1:
            raise ValueError("Speculative commit must follow the pending position")
        required = self.world_size * self.batch_size
        if len(request_ids) != required or len(set(request_ids)) != required:
            raise ValueError("Commit requires all distinct world_size * batch_size requests")
        records = [self._record(request_id) for request_id in request_ids]
        expected_ids = {key for key, r in self._requests.items() if r["request"].step_id == step_id}
        positions = {(r["request"].rank, r["request"].slot) for r in records}
        if (set(request_ids) != expected_ids or len(positions) != required
                or any(r["status"] != "LOADED" or r["request"].step_id != step_id for r in records)):
            raise ValueError("Commit contains unloaded, missing, wrong-step or duplicate slots")
        for request_id in request_ids:
            request, ordinal = self._release(request_id)
            self._state(request.episode_index).consume(ordinal)
        self._spec_committed_step = step_id

    def accept_speculation(self):
        """Publish the speculative position as the committed one.

        The reservation-cursor snapshot survives: the next step's batch was
        reserved inside this speculation, and a checkpoint taken now must still
        be able to withdraw it exactly.
        """
        if not self._spec_open:
            return
        self.committed_step = self._spec_committed_step
        self._spec_open = False
        self._spec_snapshot = None
        self._spec_committed_step = None
        self._undo = None
        self._undo_owners = None

    def rollback_speculation(self):
        """Restore the ledger exactly as it stood at begin_speculation."""
        if not self._spec_open:
            return
        snapshot = self._spec_snapshot
        for index, before in (self._undo or {}).items():
            if before is None:
                self._states.pop(index, None)
            else:
                self._states[index] = before
        for identity, before in (self._undo_owners or {}).items():
            if before is None:
                self._resolved_owners.pop(identity, None)
            else:
                self._resolved_owners[identity] = before
        self._requests = snapshot["requests"]
        self._group_cursors = snapshot["cursors"]
        self._next_request = snapshot["next_request"]
        self.committed_step = snapshot["committed_step"]
        self._spec_open = False
        self._spec_snapshot = None
        self._spec_committed_step = None
        self._reserved_snapshot = None
        self._undo = None
        self._undo_owners = None

    def record_reservation_cursor(self):
        """Remember the cursor state before the next reservation batch.

        Paired with :meth:`withdraw_step` so checkpointing can drop a reservation
        that no optimizer step has verified, and re-issue identical request IDs.
        """
        self._reserved_snapshot = dict(cursors=dict(self._group_cursors),
                                       next_request=self._next_request)

    def withdraw_step(self, step_id):
        """Release every reservation of `step_id` and restore the cursors.

        The saved ledger must describe consumption by real optimizer steps only.
        Restoring ``next_request`` keeps request IDs stable, and request IDs seed
        each sample's augmentation RNG, so the withdrawn batch is rebuilt
        bit-identically when the loader asks for it again.
        """
        step_id = _integer(step_id, "step_id")
        if step_id <= self.committed_step:
            raise ValueError("Cannot withdraw a step at or before the committed position")
        released = 0
        for request_id, record in list(self._requests.items()):
            if record["request"].step_id == step_id:
                self._release(request_id)
                released += 1
        if self._reserved_snapshot is not None:
            self._group_cursors = self._reserved_snapshot["cursors"]
            self._next_request = self._reserved_snapshot["next_request"]
            self._reserved_snapshot = None
        return released

    def cancel_all_uncommitted(self):
        """Return future/inflight windows without treating them as consumption."""
        with self._transaction():
            for request_id in list(self._requests):
                self._release(request_id)

    def _binding(self):
        return dict(plan_version=self.plan_version, seed=self.seed, world_size=self.world_size,
                    batch_size=self.batch_size, migration_start_step=self.migration_start_step,
                    fingerprint=deepcopy(self.fingerprint), catalog_sha256=self.catalog_sha256,
                    window_rule=WINDOW_RULE, window_rule_sha256=WINDOW_RULE_SHA256)

    def state_dict(self):
        states = {}
        for index, state in self._states.items():
            states[str(index)] = dict(cycle=state.cycle, committed_prefix=state.committed_prefix,
                                     committed_tail=sorted(state.committed_tail), cursor=state.cursor,
                                     quarantined_starts={str(k): deepcopy(v) for k, v in state.quarantined_starts.items()},
                                     episode_quarantine=deepcopy(state.episode_quarantine), geometry=deepcopy(state.geometry))
        return dict(schema=SCHEMA, binding=self._binding(), committed_step=self.committed_step,
                    next_request=self._next_request, episodes=states,
                    group_cursors=[dict(source_id=g[0], bucket_id=g[1], cursor=c)
                                   for g, c in sorted(self._group_cursors.items())],
                    requests=[dict(request=r["request"].to_dict(), ordinal=r["ordinal"], status=r["status"])
                              for _, r in sorted(self._requests.items())])

    def append_checkpoint_sources(self, ledger, sources):
        """Bind an append-only catalog extension without resetting old windows."""
        saved_groups = {(row["source_id"], row["bucket_id"]) for row in ledger["group_cursors"]}
        added = self._groups.keys() - saved_groups
        if not added:
            return None
        if (not saved_groups <= self._groups.keys()
                or {group[0] for group in added} != set(sources)
                or any(group[0] in sources for group in saved_groups)
                or ledger["requests"]):
            raise LedgerMismatchError("Append-only resume requires fresh sources and a quiescent checkpoint")

        first_new = len(self.catalog) - sum(len(self._groups[group]) for group in added)
        if (any(self._groups[group][0] < first_new for group in added)
                or any(int(index) >= first_new for index in ledger["episodes"])):
            raise LedgerMismatchError("New sources must follow all saved episode indices")
        digest = hashlib.sha256()
        for index in range(first_new):
            digest.update(_json(self._entry(index)).encode("utf-8") + b"\n")
        saved_binding = ledger["binding"]
        if digest.hexdigest() != saved_binding["catalog_sha256"]:
            raise LedgerMismatchError("Old catalog prefix changed during append-only resume")
        binding = self._binding()
        for key in ("seed", "world_size", "batch_size", "migration_start_step", "window_rule", "window_rule_sha256"):
            if binding[key] != saved_binding[key]:
                raise LedgerMismatchError(f"Saved {key} changed during append-only resume")
        ledger["binding"] = binding
        ledger["group_cursors"].extend(
            dict(source_id=group[0], bucket_id=group[1], cursor=self._group_cursors[group])
            for group in sorted(added)
        )
        return first_new

    def load_state_dict(self, value):
        """Validate in temporary state before replacing this live authority."""
        try:
            if value["schema"] != SCHEMA or value["binding"] != self._binding():
                raise LedgerMismatchError("Ledger schema/data/plan/geometry/world/B/seed binding differs")
            committed = _integer(value["committed_step"], "committed_step")
            counter = _integer(value["next_request"], "next_request")
            if committed < self.migration_start_step or counter < 0:
                raise LedgerMismatchError("Invalid checkpoint cursor")
            states, requests, cursors, owners = {}, {}, {}, {}
            for key, raw in value["episodes"].items():
                index = int(key)
                if str(index) != key or index in states:
                    raise LedgerMismatchError("Invalid or repeated episode state key")
                entry = self._entry(index, resolve=True)
                count = self._count(entry)
                if raw["geometry"] != self._geometry(entry):
                    raise LedgerMismatchError("Resolved checkpoint geometry differs from catalog overlay")
                identity = self._identity_digest(entry)
                if identity in owners and owners[identity] != index:
                    raise LedgerMismatchError("Restored resolved physical window identity converges across indices")
                owners[identity] = index
                cycle = _integer(raw["cycle"], "cycle")
                prefix = _integer(raw["committed_prefix"], "committed_prefix")
                cursor = _integer(raw["cursor"], "cursor")
                tail_values = raw["committed_tail"]
                tail = {_integer(i, "committed ordinal") for i in tail_values}
                quarantine = {int(k): deepcopy(v) for k, v in raw["quarantined_starts"].items()}
                if (cycle < 0 or not 0 <= prefix <= count or not 0 <= cursor <= count or len(tail) != len(tail_values)
                        or any(i <= prefix or i >= count for i in tail)
                        or any(not 0 <= start < count for start in quarantine)):
                    raise LedgerMismatchError("Invalid per-episode committed/quarantine/cycle state")
                if any(self._start(entry, cycle, i) in quarantine for i in range(prefix)) or any(
                    self._start(entry, cycle, i) in quarantine for i in tail
                ):
                    raise LedgerMismatchError("A window cannot be committed and quarantined in the same cycle")
                states[index] = _EpisodeState(cycle, prefix, tail, cursor, quarantine, deepcopy(raw["episode_quarantine"]), geometry=deepcopy(raw["geometry"]))
            for raw in value["group_cursors"]:
                group = (raw["source_id"], raw["bucket_id"])
                cursor = _integer(raw["cursor"], "group cursor")
                if group in cursors or group not in self._groups or not 0 <= cursor < len(self._groups[group]):
                    raise LedgerMismatchError("Invalid replacement group cursor")
                cursors[group] = cursor
            if set(cursors) != set(self._groups):
                raise LedgerMismatchError("Replacement groups differ")
            positions = set()
            for raw in value["requests"]:
                request = WindowRequest.from_dict(raw["request"])
                ordinal = _integer(raw["ordinal"], "request ordinal")
                state = states.get(request.episode_index)
                entry = self._entry(request.episode_index, resolve=True)
                position = (request.step_id, request.rank, request.slot)
                suffix = request.request_id.removeprefix(self.plan_version + ":")
                if (not suffix.isdigit() or not request.request_id.startswith(self.plan_version + ":")
                        or int(suffix) >= counter or request.request_id in requests or position in positions
                        or state is None or request.step_id <= committed or request.cycle != state.cycle
                        or not 0 <= request.rank < self.world_size or not 0 <= request.slot < self.batch_size
                        or not 0 <= ordinal < self._count(entry) or state.committed(ordinal)
                        or ordinal in state.inflight or state.episode_quarantine
                        or request.window_start in state.quarantined_starts or raw["status"] not in ("RESERVED", "LOADED")):
                    raise LedgerMismatchError("Invalid restored inflight request")
                state.inflight[ordinal] = request.request_id
                requests[request.request_id] = dict(request=request, ordinal=ordinal, status=raw["status"])
                positions.add(position)
            # _record rechecks catalog identity, exact start and variant. Do so
            # against the temporary objects, then restore on any validation error.
            previous = self._states, self._requests, self._resolved_owners
            self._states, self._requests, self._resolved_owners = states, requests, owners
            try:
                for request_id in requests:
                    self._record(request_id)
                for index, state in states.items():
                    entry = self._entry(index, resolve=True)
                    for ordinal in range(state.cursor):
                        if not (state.committed(ordinal) or ordinal in state.inflight
                                or self._start(entry, state.cycle, ordinal) in state.quarantined_starts):
                            raise LedgerMismatchError("Cursor skips an available, unconsumed window")
            finally:
                self._states, self._requests, self._resolved_owners = previous
            _json(value)
        except LedgerMismatchError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise LedgerMismatchError(f"Malformed strict window ledger: {exc}") from exc
        self._states, self._requests, self._group_cursors = states, requests, cursors
        self._resolved_owners = owners
        self.committed_step, self._next_request = committed, counter


def resample_ledger_sources(ledger, catalog, source_ids, seed):
    """Drop committed/inflight window state for specific catalog sources only.

    ``committed_step``, ``next_request``, and non-listed sources stay unchanged so
    resume continues strict consumption for the rest of the plan.
    """
    wanted = frozenset(_identity(s) for s in source_ids)
    if not wanted:
        return dict(episodes=0, requests_dropped=0, groups=0)

    def entry_source(index):
        raw = catalog[index]
        if isinstance(raw, dict):
            return _identity(raw.get("source_id"))
        raise ValueError("catalog entry must be a mapping")

    groups = {}
    for index in range(len(catalog)):
        raw = catalog[index]
        group = (_identity(raw["source_id"]), _identity(raw["bucket_id"]))
        groups[group] = groups.get(group, 0) + 1

    episodes_reset = 0
    for key, raw in list(ledger.get("episodes", {}).items()):
        index = int(key)
        if 0 <= index < len(catalog) and entry_source(index) in wanted:
            ledger["episodes"][key] = dict(
                cycle=0,
                committed_prefix=0,
                committed_tail=[],
                cursor=0,
                quarantined_starts={},
                episode_quarantine=None,
                geometry=deepcopy(raw["geometry"]),
            )
            episodes_reset += 1

    before = len(ledger.get("requests", []))
    ledger["requests"] = [
        raw for raw in ledger.get("requests", [])
        if _identity(raw["request"]["source_id"]) not in wanted
    ]
    requests_dropped = before - len(ledger["requests"])

    groups_reset = 0
    for raw in ledger.get("group_cursors", []):
        source_id = _identity(raw["source_id"])
        if source_id not in wanted:
            continue
        group = (source_id, _identity(raw["bucket_id"]))
        size = groups.get(group)
        if size is None or size < 1:
            raise LedgerMismatchError(f"Resample group missing from catalog: {group!r}")
        key = _json([seed, *group]).encode("utf-8")
        raw["cursor"] = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % size
        groups_reset += 1

    return dict(episodes=episodes_reset, requests_dropped=requests_dropped, groups=groups_reset)




StrictWindowCoordinator = Coordinator
