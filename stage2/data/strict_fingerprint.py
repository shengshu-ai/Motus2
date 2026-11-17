"""Narrow compatibility for moving an immutable strict-training source tree.

This module does no file I/O. Its inputs are the already content-hashed data
fingerprints. It permits only source-relative inputs emitted by
``data_plan_fingerprint`` to move between two explicitly supplied source roots.
JSONL paths are cwd-relative in that producer and are NEVER relocatable here.
The saved fingerprint and plan version remain the authority for request IDs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath


class FingerprintMismatch(ValueError):
    """A change is outside an explicitly bound source-root relocation."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FingerprintMismatch("Fingerprint must be finite JSON") from exc


def fingerprint_digest(value):
    """The existing loader's exact plan-version digest, without normalization."""
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _relative_declaration(value):
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        return None
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in (".", "..", "~") for part in path.parts)):
        return None
    return value


def _source_root(value, label):
    if value is None:
        raise FingerprintMismatch(f"Explicit {label} required for a relocated source input")
    text = str(value)
    path = PurePosixPath(text)
    if (not path.is_absolute() or path == PurePosixPath("/")
            or path.as_posix() != text or ".." in path.parts or "\\" in text):
        raise FingerprintMismatch(f"{label} must be a canonical absolute source root")
    return path


def _input_declarations(dataset):
    """Mirror producer append order; do not infer declarations from basenames."""
    if not isinstance(dataset, dict):
        raise FingerprintMismatch("dataset fingerprint must be a mapping")
    indices = dataset.get("jsonl_path") or []
    if isinstance(indices, str):
        indices = [indices]
    if not isinstance(indices, list) or any(not isinstance(x, str) for x in indices):
        raise FingerprintMismatch("Invalid JSONL declarations")
    declarations = [(f"jsonl_path[{i}]", value, False) for i, value in enumerate(indices)]
    for key in ("stereo_latent_sources_path", "action_stats_path"):
        if dataset.get(key):
            declarations.append((key, dataset[key], True))
    by_dataset = dataset.get("action_stats_by_dataset") or {}
    if not isinstance(by_dataset, dict):
        raise FingerprintMismatch("Invalid action_stats_by_dataset declarations")
    declarations.extend((f"action_stats_by_dataset.{key}", value, True)
                        for key, value in by_dataset.items())
    return declarations


def reconcile_source_relocation(saved, current, *, saved_source_root=None,
                                current_source_root=None, saved_plan_version=None):
    """Return saved identity iff the ONLY differences are proven source moves.

    All fields except ``input_files[*].path`` must be identical finite JSON.
    Each changed input must occupy its original producer slot, have an identical
    canonical relative declaration, bytes and SHA, and be EXACTLY each supplied
    source root joined with that declaration. No JSONL/absolute/user-path change
    is accepted. Missing fields, reordering and normalization are not repaired.

    The caller must install the returned *saved* fingerprint and plan_version
    into its new loader/Coordinator BEFORE Coordinator.load_state_dict. This
    function never changes the checkpoint, ledger, request counter, RNG or files.
    ``saved_plan_version`` should be the existing ledger binding's plan_version.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise FingerprintMismatch("Fingerprints must be mappings")
    saved_json, current_json = _canonical(saved), _canonical(current)
    digest = hashlib.sha256(saved_json.encode()).hexdigest()
    if saved_plan_version is not None and saved_plan_version != digest:
        raise FingerprintMismatch("Saved plan_version is not the saved fingerprint digest")
    if saved.get("schema") != "motus-strict-data-plan-v1" or current.get("schema") != saved["schema"]:
        raise FingerprintMismatch("Unknown strict data fingerprint schema")
    old_other = {key: value for key, value in saved.items() if key != "input_files"}
    new_other = {key: value for key, value in current.items() if key != "input_files"}
    if _canonical(old_other) != _canonical(new_other):
        raise FingerprintMismatch("High-level data/model/rule fingerprint changed")
    old_files, new_files = saved.get("input_files"), current.get("input_files")
    if not isinstance(old_files, list) or not isinstance(new_files, list):
        raise FingerprintMismatch("input_files must be ordered lists")
    declarations = _input_declarations(saved.get("dataset"))
    current_declarations = _input_declarations(current.get("dataset"))
    if declarations != current_declarations:
        raise FingerprintMismatch("Input declaration order changed")
    if len(old_files) != len(new_files) or len(old_files) != len(declarations):
        raise FingerprintMismatch("Input file count does not match its declarations")
    relocations = []
    for index, (old, new, declared) in enumerate(zip(old_files, new_files, declarations)):
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise FingerprintMismatch("Input identity must be a mapping")
        if set(old) != {"path", "bytes", "sha256"} or set(new) != set(old):
            raise FingerprintMismatch("Unknown or missing input identity fields")
        if (isinstance(old["bytes"], bool) or not isinstance(old["bytes"], int) or old["bytes"] < 0
                or not isinstance(old["sha256"], str) or len(old["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in old["sha256"])):
            raise FingerprintMismatch("Invalid content identity")
        if _canonical({k: v for k, v in old.items() if k != "path"}) != _canonical(
                {k: v for k, v in new.items() if k != "path"}):
            raise FingerprintMismatch(f"Input content changed at index {index}")
        if not isinstance(old["path"], str) or not isinstance(new["path"], str):
            raise FingerprintMismatch("Input paths must be strings")
        if old["path"] == new["path"]:
            continue
        key, value, source_relative = declared
        logical = _relative_declaration(value) if source_relative else None
        if logical is None:
            raise FingerprintMismatch(f"Non-relocatable JSONL/absolute/noncanonical input changed: {key}")
        old_root = _source_root(saved_source_root, "saved_source_root")
        new_root = _source_root(current_source_root, "current_source_root")
        if (old["path"] != str(old_root / logical)
                or new["path"] != str(new_root / logical)):
            raise FingerprintMismatch(f"Input is not the exact declared source-relative path: {key}")
        if old_root == new_root:
            raise FingerprintMismatch("Changed input has identical source roots")
        relocations.append(dict(index=index, config_key=key, logical_path=logical,
                                saved_path=old["path"], current_path=new["path"],
                                bytes=old["bytes"], sha256=old["sha256"]))
    return dict(fingerprint=deepcopy(saved), plan_version=digest, relocations=relocations,
                relocated=bool(relocations), scope="source-relative content-identical input relocation only")
