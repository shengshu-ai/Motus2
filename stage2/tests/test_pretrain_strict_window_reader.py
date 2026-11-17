"""Exact worker reads: failures belong to one requested window, never an episode."""
import json
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.pretrain import pretrain_dataset as module
from data.pretrain.pretrain_dataset import PretrainDataset


def make_dataset(tmp_path, *, total=6, bootstrap=False, tracking=True, relative_limit=None):
    raw = torch.arange(400, dtype=torch.float64)[:, None].expand(400, 54).clone() / 100
    raw[:, 20:23] = 0
    raw[:, 47:50] = 0
    raw[:, 23:27] = torch.tensor([0., 0., 0., 1.])
    raw[:, 50:54] = torch.tensor([0., 0., 0., 1.])
    raw[0:30, 23:27] = 0
    torch.save(raw, tmp_path / "tjwj.pt")
    torch.save({"latents": torch.arange(48 * total * 4).reshape(48, total, 2, 2).float(),
                "frame_stride": 6}, tmp_path / "latent.pt")
    torch.save(torch.ones(3, 4096), tmp_path / "language.pt")
    (tmp_path / "stats.json").write_text(json.dumps({"min": [0] * 40, "max": [1] * 40}))
    (tmp_path / "index.jsonl").write_text(json.dumps({"dataset": "fixture", "task": "task",
        "episode_bos_path": str(tmp_path), "total_latent_frames": total}) + "\n")
    return PretrainDataset(jsonl_path=str(tmp_path / "index.jsonl"), use_index_cache=False,
        episode_index_source="jsonl", latent_root=None, latent_subpath="latent.pt",
        lang_root=None, lang_subpath="language.pt", latent_phases=[0], max_latent_window=8,
        action_mode="qpos20_rel6d", action_stats_path=str(tmp_path / "stats.json"),
        qpos_normalization="min_max", action={"raw_hz": 30, "train_hz": 30}, use_state=False,
        bootstrap_start_window=bootstrap, bootstrap_chunk_size=2, samples_per_episode=1,
        mask_invalid_hand_tracking=tracking, wrist_xyz_relative_max=relative_limit,
        wrist_xyz_abs_max=10., val=True)


def request_for(ds, *, start=0, request_id="request-0"):
    spec = ds.window_spec(0, resolve_geometry=True)
    return {"request_id": request_id, "plan_version": 1, "step_id": 1, "rank": 0, "slot": 0,
            "source_id": spec["source_id"], "bucket_id": spec["bucket_id"],
            "episode_index": spec["episode_index"], "episode_id": spec["episode_id"],
            "window_start": start, "window_size": spec["window_size"], "cycle": 0,
            "latent_variant": spec["latent_variant"], "expected_total_latents": spec["total_latents"],
            "replacement_of": None}


def assert_json_status(result):
    assert set(result) == {"request_id", "status", "sample", "observed_geometry", "error_code", "error_detail"}
    json.dumps({k: v for k, v in result.items() if k != "sample"}, allow_nan=False)


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("tracking", [False, True])
def test_success_preserves_original_loader_padding_bootstrap_masks_values(tmp_path, bootstrap, tracking):
    ds = make_dataset(tmp_path, bootstrap=bootstrap, tracking=tracking)
    ds.value_target = "vlac"
    original = ds[0]
    assert original is not None
    request = request_for(ds)
    result = ds[request]
    assert_json_status(result)
    assert result["status"] == "OK" and result["error_code"] is None
    observed = result["sample"]
    for key in ("latents", "actions", "states", "values", "language_embedding",
                "action_valid_mask", "state_valid_mask"):
        assert (key in observed) == (key in original)
        if key in original:
            assert torch.equal(observed[key], original[key]), key
            assert observed[key].dtype == original[key].dtype
    assert observed["valid_len"] == original["valid_len"]
    assert observed["latent_start"] == 0
    assert observed["strict_window_request"] == request
    assert observed["debug_episode"] == original["debug_episode"]
    assert result["observed_geometry"]["returned_latent_shape"] == list(original["latents"].shape)


def test_relative_failure_is_window_local_and_never_replaces_or_drops(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path, total=16, relative_limit=1.5)
    raw = torch.load(tmp_path / "tjwj.pt", weights_only=True)
    raw[100, 20] = 5.  # finite and under abs10, with a valid chunk anchor and hand observation
    torch.save(raw, tmp_path / "tjwj.pt")
    failed = request_for(ds)
    valid = request_for(ds, start=8, request_id="request-next")
    def forbidden(*args, **kwargs):
        raise AssertionError("strict reader invoked worker fallback/drop/window selection")
    monkeypatch.setattr(ds, "_episode_for_index", forbidden)
    monkeypatch.setattr(ds, "_drop_episode", forbidden)
    monkeypatch.setattr(ds, "_select_latent_window", forbidden)
    bad = ds.read_window(failed)
    assert bad["status"] == "INVALID_SAMPLE" and bad["error_code"] == "WRIST_RELATIVE_FILTER"
    assert bad["sample"] is None and bad["observed_geometry"]["window_start"] == 0
    assert ds._dropped_episode_ids == set()
    good = ds.read_window(valid)
    assert good["status"] == "OK" and good["sample"]["latent_start"] == 8
    assert ds._dropped_episode_ids == set()
    assert_json_status(bad)


def test_legacy_relative_failure_rejects_one_window_without_fallback_or_drop(tmp_path, monkeypatch, caplog):
    ds = make_dataset(tmp_path, total=16, relative_limit=1.5)
    raw = torch.load(tmp_path / "tjwj.pt", weights_only=True)
    raw[100, 20] = 5.
    torch.save(raw, tmp_path / "tjwj.pt")
    original_episode = ds._episode_for_index
    calls = []
    def requested_episode(index, attempt=0):
        calls.append((index, attempt))
        assert attempt == 0, "relative rejection retried in the worker"
        return original_episode(index, attempt)
    def forbidden_drop(*args, **kwargs):
        raise AssertionError("relative rejection dropped an entire episode")
    monkeypatch.setattr(ds, "_episode_for_index", requested_episode)
    monkeypatch.setattr(ds, "_drop_episode", forbidden_drop)
    monkeypatch.setattr(ds, "_select_latent_window", lambda *args, **kwargs: (0, 8))
    assert ds[0] is None
    assert calls == [(0, 0)] and ds._dropped_episode_ids == set()
    assert "Rejecting single window: wrist xyz relative filter failed" in caplog.text
    monkeypatch.setattr(ds, "_select_latent_window", lambda *args, **kwargs: (8, 8))
    sample = ds[0]
    assert sample is not None and sample["latent_start"] == 8
    assert calls == [(0, 0), (0, 0)] and ds._dropped_episode_ids == set()


def test_unknown_geometry_is_not_estimated_or_eagerly_loaded(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    ds.episode_files[0].pop("sc_total_latent_frames")
    ds.episode_files[0]["sc_duration_sec"] = 9999.
    original = module.smart_load_pt
    calls = []
    def tracked(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module, "smart_load_pt", tracked)
    rng = random.getstate()
    cheap = ds.window_spec(0)
    assert cheap["total_latents"] is None and not cheap["geometry_known"]
    assert calls == [] and random.getstate() == rng
    resolved = ds.window_spec(0, resolve_geometry=True)
    assert resolved["total_latents"] == 6 and resolved["geometry_known"]
    assert resolved["latent_shape"] == [48, 6, 2, 2]
    assert calls == [] and random.getstate() == rng  # Real zip metadata, not CPU tensor storage.
    assert resolved["source_id"] == cheap["source_id"] and resolved["bucket_id"] == cheap["bucket_id"]


def test_index_length_remains_advisory_until_payload_read(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path, total=120)
    original = module.smart_load_pt
    def forbidden(*args, **kwargs):
        raise AssertionError("cheap catalog opened a latent tensor")
    monkeypatch.setattr(module, "smart_load_pt", forbidden)
    spec = ds.window_spec(0, resolve_geometry=False)
    assert spec["total_latents"] == 120  # Base pretrain has no max-total cap.
    assert not spec["request_geometry_known"] and not spec["geometry_known"]
    assert spec["latent_variant"]["strict_phase_index"] == 0
    assert spec["latent_variant"]["strict_phase_start_30"] == 0
    assert "strict_expected_shape" not in spec["latent_variant"]
    monkeypatch.setattr(module, "smart_load_pt", original)
    spec = ds.window_spec(0, resolve_geometry=True)
    assert spec["request_geometry_known"] and spec["geometry_known"]
    request = {"request_id": "cheap-known-length", "plan_version": 1, "step_id": 1,
        "rank": 0, "slot": 0, "source_id": spec["source_id"], "bucket_id": spec["bucket_id"],
        "episode_index": 0, "episode_id": spec["episode_id"], "window_start": 96,
        "window_size": 8, "cycle": 0, "latent_variant": spec["latent_variant"],
        "expected_total_latents": spec["total_latents"]}
    result = ds.read_window(request)
    assert result["status"] == "OK" and result["observed_geometry"]["total_latents"] == 120
    request["expected_total_latents"] = 119
    assert ds.read_window(request)["error_code"] == "TOTAL_LATENTS_MISMATCH"


@pytest.mark.parametrize("stale_total", [4, 999])
def test_actual_catalog_resolves_stale_index_length_before_frozen_request(tmp_path, monkeypatch, stale_total):
    from data.strict_window_loader import DatasetCatalog
    ds = make_dataset(tmp_path, total=6)
    ds.episode_files[0]["sc_total_latent_frames"] = stale_total
    cheap = ds.window_spec(0, resolve_geometry=False)
    assert cheap["total_latents"] == stale_total
    assert not cheap["request_geometry_known"]
    original = ds._strict_load_latent_variant
    reads = []
    def actual_payload(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(ds, "_strict_load_latent_variant", actual_payload)
    catalog = DatasetCatalog(ds)
    spec = catalog.resolve(0)
    assert spec["total_latents"] == 6 and spec["request_geometry_known"]
    assert spec["latent_variant"]["strict_expected_shape"] == [48, 6, 2, 2]
    assert reads == [1]
    assert catalog.resolve(0) is spec and reads == [1]
    request = {"request_id": "resolved-stale-index", "plan_version": 1, "step_id": 1,
        "rank": 0, "slot": 0, "source_id": spec["source_id"], "bucket_id": spec["bucket_id"],
        "episode_index": 0, "episode_id": spec["episode_id"], "window_start": 0,
        "window_size": spec["window_size"], "cycle": 0, "latent_variant": spec["latent_variant"],
        "expected_total_latents": spec["total_latents"]}
    result = ds.read_window(request)
    assert result["status"] == "OK" and result["observed_geometry"]["total_latents"] == 6


def test_cheap_catalog_does_not_claim_ambiguous_phase_is_request_ready(tmp_path):
    ds = make_dataset(tmp_path)
    ds.latent_phases_set = {0, 1}
    spec = ds.window_spec(0, resolve_geometry=False)
    assert not spec["request_geometry_known"]
    assert "strict_phase_index" not in spec["latent_variant"]
    assert "strict_phase_start_30" not in spec["latent_variant"]


def test_metadata_geometry_does_not_read_tensor_storage_but_worker_still_does(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    original = torch.serialization._open_zipfile_reader
    storage_reads = []
    class ObservedReader:
        def __init__(self, reader):
            self.reader = reader
        def __getattr__(self, name):
            return getattr(self.reader, name)
        def get_storage_from_record(self, *args, **kwargs):
            storage_reads.append(args[0])
            return self.reader.get_storage_from_record(*args, **kwargs)
        def get_record(self, name):
            assert not name.startswith('data/'), 'geometry unexpectedly read a tensor data record'
            return self.reader.get_record(name)
    class ObservedContext:
        def __init__(self, *args, **kwargs):
            self.context = original(*args, **kwargs)
        def __enter__(self):
            return ObservedReader(self.context.__enter__())
        def __exit__(self, *args):
            return self.context.__exit__(*args)
    monkeypatch.setattr(torch.serialization, '_open_zipfile_reader', ObservedContext)
    request = request_for(ds)
    assert storage_reads == []
    result = ds.read_window(request)
    assert result['status'] == 'OK' and storage_reads
    assert result['sample']['latents'].device.type == 'cpu'
    assert result['observed_geometry']['latent_shape'] == request['latent_variant']['strict_expected_shape']


@pytest.mark.parametrize('format', ['tensor', 'dict', 'phases', 'stereo'])
def test_meta_geometry_matches_full_reader_phase_stride_dtype_and_handson_cap(tmp_path, monkeypatch, format):
    ds = make_dataset(tmp_path, total=12)
    ep = ds.episode_files[0]
    ep.update(action_schema='handson-stage2/raw54-v1', handson_action_frames=119)
    values = torch.arange(48*12*4).reshape(48, 12, 2, 2).to(torch.bfloat16)
    if format == 'tensor':
        payload = values
    elif format == 'dict':
        payload = {'latents': values, 'frame_stride': 6, 'phase_start_30': 0}
    else:
        payload = {'phases': [{'latents': values, 'phase_start_30': 0},
                              {'latents': values+1, 'phase_start_30': 1}], 'frame_stride': 6}
    if format == 'stereo':
        for name in ('left.pt', 'right.pt'):
            torch.save({'latents': values, 'frame_stride': 6}, tmp_path/name)
        candidate = {'left_latent_path': str(tmp_path/'left.pt'),
                     'right_latent_path': str(tmp_path/'right.pt'), 'phase_start_30': 0}
        monkeypatch.setattr(ds, '_resolve_latent_candidates', lambda ep: [candidate])
    else:
        torch.save(payload, tmp_path/'latent.pt')
    spec = ds.window_spec(0, resolve_geometry=True)
    variant = spec['latent_variant']
    meta, ms, mp, mb = ds._strict_load_latent_variant(variant, metadata_only=True)
    actual, fs, fp, fb = ds._strict_load_latent_variant(variant)
    meta, mt = ds._strict_latent_geometry(ep, meta, mp)
    actual, ft = ds._strict_latent_geometry(ep, actual, fp)
    assert meta.is_meta and actual.device.type == 'cpu'
    assert list(meta.shape) == list(actual.shape) == spec['latent_shape']
    assert ms == fs == spec['frame_stride'] == 6
    assert mp == fp == spec['phase_start_30'] == 0 and mb == fb
    assert mt == ft == spec['total_latents'] == 6
    assert meta.dtype == actual.dtype == torch.float32


def test_legacy_nonzip_export_keeps_actual_payload_fallback(tmp_path):
    ds = make_dataset(tmp_path)
    torch.save({'latents': torch.ones(48, 6, 2, 2), 'frame_stride': 6},
               tmp_path/'latent.pt', _use_new_zipfile_serialization=False)
    spec = ds.window_spec(0, resolve_geometry=True)
    actual, stride, phase, _ = ds._strict_load_latent_variant(spec['latent_variant'], metadata_only=True)
    assert not actual.is_meta and list(actual.shape) == spec['latent_shape']
    assert stride == 6 and phase == 0 and spec['total_latents'] == 6


def test_tensor_valued_phase_or_stride_keeps_exact_cpu_metadata_fallback(tmp_path):
    ds = make_dataset(tmp_path)
    torch.save({'latents': torch.ones(48, 6, 2, 2), 'frame_stride': torch.tensor(6),
                'phase_start_30': torch.tensor(0)}, tmp_path/'latent.pt')
    spec = ds.window_spec(0, resolve_geometry=True)
    assert spec['total_latents'] == 6 and spec['frame_stride'] == 6 and spec['phase_start_30'] == 0
    assert ds.read_window(request_for(ds))['status'] == 'OK'


def test_remote_metadata_retains_existing_remote_reader(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    calls = []
    payload = torch.ones(48, 6, 2, 2)
    def original_remote(path, cache_dir, map_location):
        calls.append((path, cache_dir, map_location))
        return payload
    monkeypatch.setattr(module, 'smart_load_pt', original_remote)
    assert ds._load_latent_geometry_metadata('s3://bucket/real.pt') is payload
    assert calls == [('s3://bucket/real.pt', ds.cache_dir, 'cpu')]


def test_bound_geometry_resolver_never_discovers_candidates_or_mutates_input_rng_cache(tmp_path, monkeypatch):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    ds = make_dataset(tmp_path)
    spec = ds.window_spec(0, resolve_geometry=False)
    original_spec = json.loads(json.dumps(spec))
    expected = ds.window_spec(0, resolve_geometry=True)
    caches = {name: dict(value) for name, value in ds.__dict__.items()
              if name.endswith('_cache') and isinstance(value, dict)}
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    def forbidden(*args, **kwargs):
        raise AssertionError('bound geometry resolver invoked candidate discovery')
    monkeypatch.setattr(ds, '_strict_latent_candidates', forbidden)
    monkeypatch.setattr(ds, '_resolve_latent_candidates', forbidden)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda bound: ds._resolve_window_spec_geometry(bound, allow_payload_fallback=False), [spec] * 12))
    assert all(result == expected for result in results) and spec == original_spec
    assert random.getstate() == python_rng and torch.equal(torch.random.get_rng_state(), torch_rng)
    actual_np = np.random.get_state()
    assert actual_np[0] == numpy_rng[0] and np.array_equal(actual_np[1], numpy_rng[1])
    assert actual_np[2:] == numpy_rng[2:]
    for name, value in caches.items():
        assert ds.__dict__[name] == value


def test_bound_geometry_resolver_rejects_changed_catalog_identity(tmp_path):
    ds = make_dataset(tmp_path)
    spec = ds.window_spec(0)
    spec['episode_id'] = 'unrelated-episode'
    with pytest.raises(ValueError, match='changed episode/source/bucket identity'):
        ds._resolve_window_spec_geometry(spec)


@pytest.mark.parametrize('format', ['legacy_nonzip', 'tensor_scalar'])
def test_threaded_geometry_cannot_silently_load_payload_fallback(tmp_path, monkeypatch, format):
    ds = make_dataset(tmp_path)
    payload = {'latents': torch.ones(48, 6, 2, 2), 'frame_stride': 6, 'phase_start_30': 0}
    if format == 'tensor_scalar':
        payload['frame_stride'] = torch.tensor(6)
    torch.save(payload, tmp_path/'latent.pt', _use_new_zipfile_serialization=format != 'legacy_nonzip')
    spec = ds.window_spec(0)
    def forbidden(*args, **kwargs):
        raise AssertionError('threaded resolver fell back to full CPU storage')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'smart_load_pt', forbidden)
        with pytest.raises((RuntimeError, NotImplementedError), match='meta'):
            ds._resolve_window_spec_geometry(spec, allow_payload_fallback=False)
    # Main-thread default still resolves exactly from the original payload.
    resolved = ds._resolve_window_spec_geometry(spec)
    assert resolved['total_latents'] == 6 and resolved['frame_stride'] == 6
    assert resolved['phase_start_30'] == 0


def test_threaded_geometry_does_not_enter_remote_reader(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('threaded metadata invoked remote smart_io')
    monkeypatch.setattr(module, 'smart_load_pt', forbidden)
    with pytest.raises(RuntimeError, match='remote payload fallback'):
        ds._load_latent_geometry_metadata('s3://bucket/real.pt', allow_payload_fallback=False)


@pytest.mark.parametrize("change,code", [({"source_id": "other"}, "IDENTITY_MISMATCH"),
    ({"bucket_id": "other"}, "IDENTITY_MISMATCH"), ({"episode_index": -1}, "INVALID_REQUEST"),
    ({"expected_total_latents": 7}, "TOTAL_LATENTS_MISMATCH"),
    ({"window_start": 1}, "WINDOW_OUT_OF_RANGE"), ({"window_size": 4}, "WINDOW_SIZE_MISMATCH")])
def test_mismatched_request_fails_without_repair(tmp_path, change, code):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    request.update(change)
    result = ds.read_window(request)
    assert result["status"] == "INVALID_SAMPLE" and result["error_code"] == code
    assert result["sample"] is None and not ds._dropped_episode_ids
    assert_json_status(result)


def test_wrong_candidate_does_not_read_another_episode(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    request["latent_variant"]["latent_path"] = "another-episode.pt"
    def forbidden(*args, **kwargs):
        raise AssertionError("unbound file was opened")
    monkeypatch.setattr(module, "smart_load_pt", forbidden)
    result = ds.read_window(request)
    assert result["error_code"] == "LATENT_VARIANT_MISMATCH"


def test_actual_shape_drift_returns_observation_not_false_success(tmp_path):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    torch.save({"latents": torch.zeros(48, 6, 3, 2), "frame_stride": 6}, tmp_path / "latent.pt")
    result = ds.read_window(request)
    assert result["error_code"] == "LATENT_GEOMETRY_MISMATCH"
    assert result["observed_geometry"]["latent_shape"] == [48, 6, 3, 2]
    assert not ds._dropped_episode_ids


def test_io_failure_is_one_retryable_read_and_no_worker_poison(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    original = ds._strict_load_latent_variant
    calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        raise OSError("temporary storage failure")
    monkeypatch.setattr(ds, "_strict_load_latent_variant", failure)
    result = ds.read_window(request)
    assert result["status"] == "RETRYABLE_ERROR" and len(calls) == 1
    assert result["error_code"] == "LATENT_READ_ERROR" and not ds._dropped_episode_ids
    monkeypatch.setattr(ds, "_strict_load_latent_variant", original)
    assert ds.read_window(request)["status"] == "OK"


def test_language_failure_does_not_disable_episode_in_one_worker(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    key = ds._episode_identity(ds.episode_files[0])
    candidates = ds._resolve_language_candidates(ds.episode_files[0])[:]
    original = ds._load_language_embedding
    def disable(*args, **kwargs):
        ds._language_candidates_cache[key] = []
        return None
    monkeypatch.setattr(ds, "_load_language_embedding", disable)
    result = ds.read_window(request)
    assert result["status"] == "RETRYABLE_ERROR" and result["error_code"] == "MISSING_LANGUAGE"
    assert ds._language_candidates_cache[key] == candidates and not ds._dropped_episode_ids
    monkeypatch.setattr(ds, "_load_language_embedding", original)
    assert ds.read_window(request)["status"] == "OK"


def test_successful_language_fallback_cannot_prune_strict_candidate_pool(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    request = request_for(ds)
    key = ds._episode_identity(ds.episode_files[0])
    candidates = ds._resolve_language_candidates(ds.episode_files[0])[:]
    candidates.append("another-language-candidate.pt")
    ds._language_candidates_cache[key] = candidates[:]
    calls = []
    def successful_prune(*args, **kwargs):
        calls.append(ds._language_candidates_cache[key][:])
        ds._language_candidates_cache[key].pop(0)
        return torch.ones(3, 4096)
    monkeypatch.setattr(ds, "_load_language_embedding", successful_prune)
    for _ in range(2):
        result = ds.read_window(request)
        assert result["status"] == "OK"
        assert ds._language_candidates_cache[key] == candidates
        assert not ds._dropped_episode_ids
    assert calls == [candidates, candidates]


def test_multiphase_member_is_explicit_and_never_random(tmp_path):
    ds = make_dataset(tmp_path)
    ds.latent_phases_set = None
    torch.save({"phases": [
        {"latents": torch.ones(48, 6, 2, 2), "phase_start_30": 0},
        {"latents": torch.full((48, 6, 2, 2), 2.), "phase_start_30": 1}],
        "frame_stride": 6}, tmp_path / "latent.pt")
    rng = random.getstate()
    request = request_for(ds)
    assert request["latent_variant"]["strict_phase_index"] == 0
    assert random.getstate() == rng
    good = ds.read_window(request)
    assert good["status"] == "OK" and good["sample"]["latents"][:, :6].eq(1).all()
    request["latent_variant"].pop("strict_phase_index")
    bad = ds.read_window(request)
    assert bad["status"] == "INVALID_SAMPLE" and "explicit strict_phase_index" in bad["error_detail"]
    assert not ds._dropped_episode_ids


def test_stereo_actual_view_geometry_is_bound(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path)
    for name in ("left.pt", "right.pt"):
        torch.save({"latents": torch.ones(48, 6, 2, 2), "frame_stride": 6}, tmp_path / name)
    candidate = {"left_latent_path": str(tmp_path / "left.pt"),
                 "right_latent_path": str(tmp_path / "right.pt"), "phase_start_30": 0}
    monkeypatch.setattr(ds, "_resolve_latent_candidates", lambda ep: [candidate])
    result = ds.read_window(request_for(ds))
    assert result["status"] == "OK"
    assert result["observed_geometry"]["num_views"] == 2
    assert result["observed_geometry"]["latent_shape"] == [2, 48, 6, 2, 2]
    assert result["sample"]["latents"].shape == (2, 48, 8, 2, 2)


def test_non_json_request_cannot_leak_tensor_in_status(tmp_path):
    ds = make_dataset(tmp_path)
    result = ds.read_window({"request_id": torch.tensor(1)})
    assert result["status"] == "INVALID_SAMPLE" and result["sample"] is None
    assert_json_status(result)
