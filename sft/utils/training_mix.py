"""Explicit pool probabilities and conditional training-mode probabilities.

Pool probabilities refer to all dataset draws. Mode probabilities refer only
to draws from the named pool. Legacy keys retain their original semantics.
"""

import math


def _probabilities(raw, allowed, name):
    values = {str(k): float(v) for k, v in raw.items()}
    if set(values) != set(allowed):
        raise ValueError(f"{name} must contain exactly {sorted(allowed)}")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError(f"{name} probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} probabilities must sum to 1")
    return values


def pool_sample_probabilities(dataset):
    explicit = dataset.get("pool_sample_probabilities")
    if explicit is None:
        fail = float(dataset.get("fail_sample_fraction") or 0)
        random = float((dataset.get("random") or {}).get("sample_fraction") or 0)
        intervention = float(dataset.get("intervention_sample_fraction") or 0)
        return {
            "success": 1 - fail - random - intervention,
            "fail": fail,
            "random": random,
            "intervention": intervention,
        }
    if (
        dataset.get("fail_sample_fraction") is not None
        or (dataset.get("random") or {}).get("sample_fraction") is not None
        or dataset.get("intervention_sample_fraction") is not None
    ):
        raise ValueError(
            "pool_sample_probabilities cannot be mixed with legacy pool fractions"
        )
    return _probabilities(
        explicit, {"success", "fail", "random"}, "dataset.pool_sample_probabilities"
    )


def training_mode_probabilities(model):
    explicit = model.get("training_mode_probabilities")
    value = model.get("value") or {}
    if explicit is None:
        wm = float(model.get("wm_ratio", 0.2))
        vm = float(value.get("vm_ratio", 0)) if value.get("enabled", False) else 0.0
        share = vm / (wm + vm) if wm + vm > 0 else 0.0
        fail_share = value.get("fail_vm_share")
        random_share = value.get("random_vm_share")
        fail_share = share if fail_share is None else float(fail_share)
        random_share = share if random_share is None else float(random_share)
        return {
            "success": {"policy": 1 - wm - vm, "world_model": wm, "value": vm},
            "fail": {"world_model": 1 - fail_share, "value": fail_share},
            "random": {"world_model": 1 - random_share, "value": random_share},
        }
    if (
        model.get("wm_ratio") is not None
        or value.get("vm_ratio") is not None
        or value.get("fail_vm_share") is not None
        or value.get("random_vm_share") is not None
    ):
        raise ValueError(
            "training_mode_probabilities cannot be mixed with legacy mode ratios"
        )
    if set(explicit) != {"success", "fail", "random"}:
        raise ValueError(
            "training_mode_probabilities must name success, fail and random"
        )
    result = {
        pool: _probabilities(
            modes,
            {"policy", "world_model", "value"}
            if pool == "success"
            else {"world_model", "value"},
            f"model.training_mode_probabilities.{pool}",
        )
        for pool, modes in explicit.items()
    }
    if not value.get("enabled", False) and any(m["value"] > 0 for m in result.values()):
        raise ValueError("positive value probability requires model.value.enabled")
    return result


def resolved_model_mix(model):
    modes = training_mode_probabilities(model)
    explicit = model.get("training_mode_probabilities") is not None
    value = model.get("value") or {}
    return {
        "wm_ratio": modes["success"]["world_model"],
        "vm_ratio": modes["success"]["value"],
        "fail_vm_share": modes["fail"]["value"]
        if explicit
        else value.get("fail_vm_share"),
        "random_vm_share": modes["random"]["value"]
        if explicit
        else value.get("random_vm_share"),
    }


def training_mix_summary(config):
    pools = pool_sample_probabilities(config["dataset"])
    modes = training_mode_probabilities(config["model"])
    if pools.get("intervention", 0):
        return {
            "pool_probabilities": pools,
            "conditional_mode_probabilities": modes,
            "global_probabilities": None,
            "note": "intervention outcomes need their own accounting",
        }
    total = {
        mode: sum(
            pools[pool] * modes[pool].get(mode, 0)
            for pool in ("success", "fail", "random")
        )
        for mode in ("policy", "world_model", "value")
    }
    return {
        "pool_probabilities": pools,
        "conditional_mode_probabilities": modes,
        "global_probabilities": total,
        "note": "steady state after off_policy_start_step; task weights apply inside success",
    }
