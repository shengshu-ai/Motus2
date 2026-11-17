"""Planned KV index lists for factored memory attention (§1 / §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SegmentFactoredPlans:
    chunk_q0: list[int]
    adjacent_clean_phys: list[int]
    # per clean/gist frame:
    clean_kv_clean_phys: list[list[int]]
    clean_kv_gist_phys: list[list[int]]
    gist_kv_clean_phys: list[list[int]]
    gist_kv_gist_phys: list[list[int]]
    # per chunk:
    action_kv_clean_phys: list[list[int]]
    action_kv_gist_phys: list[list[int]]
    planned_kv_token_lengths: dict[str, int]  # sums for budget

    # Cache slot owned by factored_attention: the gather/scatter index tensors
    # implied by these plans and a layout. Plans are built once per batch but
    # consumed once per transformer layer, so materializing there would rebuild
    # the same ~10^5 token indices tens of times per step. Excluded from
    # equality and repr so the plans still compare by their semantic content.
    materialized_groups: Any = field(default=None, compare=False, repr=False)


def build_segment_factored_plans(
    *,
    num_clean_frames: int,
    obs_latents: int,
    chunk_size: int,
    video_tokens_per_frame: int,
    gist_tokens_per_frame: int,
    action_tokens_per_frame: int,
    state_tokens_per_frame: int = 0,
    recent_frames: int = 1,
    anchor_frames: int = 0,
) -> SegmentFactoredPlans:
    """Build per-query-class planned KV phys lists for one memory window.

    Semantics match §1 visibility:
    - chunk_q0 from obs..obs+N-1 in cs steps
    - adjacent = q0 - 1 (metadata only)
    - action_kv_clean = retained prediction history (recent + optional anchor)
    - action_kv_gist = range(q0)  # pk < q0
    - gist/clean_kv_clean = own + retained update history, where own counts
      inside recent_frames
    - clean/gist_kv_gist = range(i + 1)  # pk <= i
    """
    N = int(num_clean_frames)
    obs, cs = int(obs_latents), int(chunk_size)
    vbs, gbs = int(video_tokens_per_frame), int(gist_tokens_per_frame)
    recent = int(recent_frames)
    anchor = int(anchor_frames)

    # noisy phys runs obs .. obs+N-1 in cs chunks (same as build_memory_visibility).
    # Clean/gist streams only store phys 0..N-1. When N % cs != 0 and obs=cs,
    # the last chunk can have q0-1 >= N (no in-window clean for that adjacent);
    # dense §1 masks silently drop it — plans must clamp the same way.
    chunk_q0: list[int] = []
    off = 0
    while off < N:
        chunk_q0.append(obs + off)
        off += min(cs, N - off)
    adjacent: list[int] = []
    for q0 in chunk_q0:
        adj = q0 - 1
        adjacent.append(adj if 0 <= adj < N else -1)

    def retained_update_hist(pq: int) -> list[int]:
        history_frames = max(0, recent - 1)
        out: list[int] = []
        for pk in range(pq):
            if pk >= pq - history_frames or pk < anchor:
                out.append(pk)
        return out

    def retained_prediction_hist(pq: int) -> list[int]:
        out: list[int] = []
        for pk in range(min(pq, N)):
            if pk >= pq - recent or pk < anchor:
                out.append(pk)
        return out

    clean_kv_clean: list[list[int]] = []
    clean_kv_gist: list[list[int]] = []
    gist_kv_clean: list[list[int]] = []
    gist_kv_gist: list[list[int]] = []
    for i in range(N):
        cleans = [i] + retained_update_hist(i)
        clean_kv_clean.append(sorted(set(cleans)))
        clean_kv_gist.append(list(range(i + 1)))  # pk <= i
        gist_kv_clean.append(sorted(set([i] + retained_update_hist(i))))
        gist_kv_gist.append(list(range(i + 1)))

    action_clean: list[list[int]] = []
    action_gist: list[list[int]] = []
    for q0 in chunk_q0:
        action_clean.append(retained_prediction_hist(q0))
        # gists only exist for phys 0..N-1; clamp pk < q0 to that range
        action_gist.append(list(range(min(q0, N))))

    # planned lengths (token counts, with duplication across groups)
    planned: dict[str, int] = {
        "clean_q": 0,
        "gist_q": 0,
        "action_q": 0,
        "noisy_q": 0,
        "planned_kv": 0,
    }
    for i in range(N):
        planned["clean_q"] += vbs
        planned["gist_q"] += gbs
        planned["planned_kv"] += vbs * len(clean_kv_clean[i]) + gbs * len(clean_kv_gist[i])
        planned["planned_kv"] += vbs * len(gist_kv_clean[i]) + gbs * len(gist_kv_gist[i])
    for ci, q0 in enumerate(chunk_q0):
        count = min(cs, N - (q0 - obs))
        alen = state_tokens_per_frame + count * action_tokens_per_frame
        nlen = count * vbs
        planned["action_q"] += alen
        planned["noisy_q"] += nlen
        hist = vbs * len(action_clean[ci]) + gbs * len(action_gist[ci])
        planned["planned_kv"] += alen * 0 + hist  # hist once per action group
        planned["planned_kv"] += hist + alen  # noisy also sees action tokens

    return SegmentFactoredPlans(
        chunk_q0=chunk_q0,
        adjacent_clean_phys=adjacent,
        clean_kv_clean_phys=clean_kv_clean,
        clean_kv_gist_phys=clean_kv_gist,
        gist_kv_clean_phys=gist_kv_clean,
        gist_kv_gist_phys=gist_kv_gist,
        action_kv_clean_phys=action_clean,
        action_kv_gist_phys=action_gist,
        planned_kv_token_lengths=planned,
    )
