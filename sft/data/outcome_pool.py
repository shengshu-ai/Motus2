"""On-disk outcome pools: one directory name, one routing class.

A task directory is expected to look like::

    <task>/success/episode_*
    <task>/fail/episode_*
    <task>/suboptimal/episode_*
    <task>/random/episode_*

Legacy layouts still parse: episodes sitting directly under ``<task>/`` are
success, and ``<task>/fail`` / ``<task>/suboptimal`` keep their names. The
legacy sibling ``random_<task>`` is interpreted as ``<task>/random``. The
parent of an episode, relative to the dataset root, is the only input.
"""

from __future__ import annotations

import fnmatch
import random
from typing import Tuple

POOLS = ("success", "fail", "suboptimal", "random")
POOL_SET = frozenset(POOLS)


def split_task_and_pool(rel_parent: str) -> Tuple[str, str]:
    """Return ``(task, pool)`` from an episode's parent relative to the dataset root.

    ``0812_drink/success`` → ``('0812_drink', 'success')``.
    ``0812_drink/fail`` → ``('0812_drink', 'fail')``.
    ``0812_drink`` (legacy, episodes at the task root) → ``('0812_drink', 'success')``.
    """
    parts = [p for p in str(rel_parent).strip("/").split("/") if p]
    pool = parts.pop().lower() if parts and parts[-1].lower() in POOL_SET else "success"
    # Legacy sibling random_<task> is a logical <task>/random. Keep the
    # physical episode path untouched. Do not guess aliases for other names.
    if parts and parts[-1].startswith("random_") and len(parts[-1]) > len("random_"):
        parts[-1] = parts[-1][len("random_"):]
        pool = "random"
    return "/".join(parts), pool


def strip_pool_suffix(name: str) -> str:
    task, _pool = split_task_and_pool(name)
    return task


def is_pool_component(name: str) -> bool:
    return str(name).strip("/").lower() in POOL_SET


def pool_label(ep):
    explicit = str(ep.get("pool") or "").lower()
    if explicit in POOL_SET:
        return explicit
    for pool in ("random", "suboptimal", "fail"):
        if ep.get(f"is_{pool}_class"):
            return pool
    return "success"


def stamp_pool(rec, pool):
    rec["pool"] = pool
    for name in ("fail", "suboptimal", "random"):
        rec[f"is_{name}_class"] = pool == name
    return rec


def task_prefix_could_match(rel, patterns):
    """Prune directory walks using logical names, including legacy random siblings."""
    if not patterns:
        return True
    task, pool = split_task_and_pool(rel)
    logical = task + ("/" + pool if pool != "success" else "")
    for candidate in (rel, logical):
        parts = candidate.strip("/").split("/")
        for pattern in patterns:
            pat = strip_pool_suffix(pattern).replace("%", "*").strip("/").split("/")
            if all(fnmatch.fnmatchcase(r, p) for r, p in zip(parts, pat)):
                if len(parts) <= len(pat) or is_pool_component(parts[len(pat)]):
                    return True
    return False


def configure_pool_sampling(dataset):
    """Explicit fractions reserve mass; every unspecified pool shares the rest."""
    pools = {name: [] for name in POOLS}
    for ep in dataset.episodes:
        pools[pool_label(ep)].append(ep)
    dataset._main_episodes = pools["success"]
    dataset._fail_episodes = pools["fail"]
    dataset._suboptimal_episodes = pools["suboptimal"]
    dataset._random_episodes = pools["random"]
    dataset._non_suboptimal_episodes = pools["success"] + pools["fail"]
    reserved = []
    for name in ("random", "suboptimal", "fail"):
        fraction = None if dataset.val else getattr(dataset, f"{name}_sample_fraction", None)
        setattr(dataset, f"_{name}_sampling_probability", fraction)
        if fraction is not None:
            if not 0 <= fraction <= 1:
                raise ValueError(f"{name} sample fraction must be in [0, 1]")
            if fraction > 0 and not pools[name]:
                raise ValueError(f"{name} sample_fraction requires {name} episodes, but the {name} pool is empty")
            reserved.append((name, fraction))
    total = sum(fraction for _, fraction in reserved)
    if total > 1:
        raise ValueError("fail_sample_fraction + random.sample_fraction + suboptimal.sample_fraction must be <= 1")
    specified = {name for name, _ in reserved}
    remainder = [ep for ep in dataset.episodes if pool_label(ep) not in specified]
    if total < 1 and not remainder:
        raise ValueError("sample fractions leave positive mass but the remaining success/unspecified pool is empty")
    dataset._sampling_slices = [(fraction, pools[name]) for name, fraction in reserved if fraction > 0]
    dataset._sampling_remainder = remainder


def sample_pool_episode(dataset):
    slices = dataset._sampling_slices
    if not slices:
        return random.choice(dataset._sampling_remainder)
    draw = random.random()
    cursor = 0.0
    for fraction, episodes in slices:
        cursor += fraction
        if draw < cursor:
            return random.choice(episodes)
    # Rounding at a fully allocated boundary must not resurrect an excluded pool.
    return random.choice(dataset._sampling_remainder or slices[-1][1])


def legacy_random_patterns(patterns):
    """Physical sibling names for server-side discovery of logical tasks."""
    result = []
    for pattern in patterns:
        task = strip_pool_suffix(pattern).strip("/")
        prefix, sep, name = task.rpartition("/")
        result.append((prefix + sep if sep else "") + "random_" + name)
    return result
