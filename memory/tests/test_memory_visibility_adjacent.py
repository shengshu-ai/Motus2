# tests/test_memory_visibility_adjacent.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

import torch

_ba_path = Path(__file__).parent.parent / "models" / "block_attention.py"
_ba_spec = importlib.util.spec_from_file_location("block_attention_direct", _ba_path)
_ba = importlib.util.module_from_spec(_ba_spec)
assert _ba_spec.loader is not None
_ba_spec.loader.exec_module(_ba)

build_memory_visibility = _ba.build_memory_visibility
TOKEN_CLEAN_VIDEO = _ba.TOKEN_CLEAN_VIDEO
TOKEN_GIST = _ba.TOKEN_GIST
TOKEN_ACTION = _ba.TOKEN_ACTION
TOKEN_NOISY_VIDEO = _ba.TOKEN_NOISY_VIDEO

# Original import (via models package) requires full CUDA stack; load module directly instead.
# from models.block_attention import (
#     build_memory_visibility,
#     TOKEN_CLEAN_VIDEO,
#     TOKEN_GIST,
#     TOKEN_ACTION,
#     TOKEN_NOISY_VIDEO,
# )


def _vis(num_frames=4, vbs=2, gbs=2, abs_=1, cs=2, obs=2, anchor=0, recent=1, mode="fdm"):
    return build_memory_visibility(
        num_frames=num_frames,
        video_tokens_per_frame=vbs,
        gist_tokens_per_frame=gbs,
        anchor_frames=anchor,
        recent_frames=recent,
        action_tokens_per_frame=abs_,
        state_tokens_per_frame=0,
        value_tokens_per_frame=0,
        chunk_size=cs,
        attention_mode=mode,
        device="cpu",
        obs_latents=obs,
        separate_anchor=False,
    )


def _first_token(info, typ, phys):
    t, p = info["token_types"], info["phys_ids"]
    idx = ((t == typ) & (p == phys)).nonzero(as_tuple=True)[0]
    assert idx.numel() > 0, f"no token type={typ} phys={phys}"
    return int(idx[0])


def test_action_noisy_chunk_l4l5_see_only_adjacent_l3_full():
    # With N=4, obs=2, cs=2: noisy phys 2,3,4,5 in two chunks.
    # Chunk0 q0=2; Chunk1 q0=4 (L4,L5). Cleans 0..3.
    info = _vis(num_frames=4, obs=2, cs=2, anchor=0, recent=1)
    mask = info["mask"]
    q_act = _first_token(info, TOKEN_ACTION, 4)  # chunk phys q0=4
    q_vid = _first_token(info, TOKEN_NOISY_VIDEO, 4)
    for q in (q_act, q_vid):
        for phys, expect in [(0, False), (1, False), (2, False), (3, True)]:
            k = _first_token(info, TOKEN_CLEAN_VIDEO, phys)
            assert bool(mask[q, k]) is expect, f"q phys4 → clean{phys}"
        # gists pk < 4 all visible
        for phys in range(4):
            k = _first_token(info, TOKEN_GIST, phys)
            assert bool(mask[q, k]) is True, f"q → gist{phys}"


def test_l5_tokens_share_q0_phys_not_per_frame():
    info = _vis(num_frames=4, obs=2, cs=2, anchor=0, recent=1)
    t, p = info["token_types"], info["phys_ids"]
    noisy = (t == TOKEN_NOISY_VIDEO) & (p == 4)
    assert noisy.sum() > 0
    # all noisy tokens of this chunk must share phys==4 (q0), never 5
    assert (info["phys_ids"][noisy] == 4).all()


def test_gist_recent_one_counts_own_clean_only():
    info = _vis(num_frames=4, obs=2, cs=2, anchor=0, recent=1)
    mask = info["mask"]
    # The update query's own frame is part of recent=1.
    q = _first_token(info, TOKEN_GIST, 3)
    assert bool(mask[q, _first_token(info, TOKEN_CLEAN_VIDEO, 3)]) is True
    assert bool(mask[q, _first_token(info, TOKEN_CLEAN_VIDEO, 2)]) is False
    assert bool(mask[q, _first_token(info, TOKEN_CLEAN_VIDEO, 1)]) is False
    assert bool(mask[q, _first_token(info, TOKEN_CLEAN_VIDEO, 0)]) is False
    # gist chain
    assert bool(mask[q, _first_token(info, TOKEN_GIST, 2)]) is True


def test_gist_recent_four_is_own_plus_three_previous_fulls():
    info = _vis(num_frames=6, obs=1, cs=2, anchor=0, recent=4)
    mask = info["mask"]
    q = _first_token(info, TOKEN_GIST, 5)
    for phys, expect in [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
        (4, True),
        (5, True),
    ]:
        k = _first_token(info, TOKEN_CLEAN_VIDEO, phys)
        assert bool(mask[q, k]) is expect, f"recent=4 gist5 → clean{phys}"


def test_anchor_zero_accepted():
    info = _vis(anchor=0, recent=1)
    assert info["n_anchor_tokens"] == 0


def test_clean_sees_gists():
    info = _vis(num_frames=4, anchor=0, recent=1)
    mask = info["mask"]
    q = _first_token(info, TOKEN_CLEAN_VIDEO, 3)
    assert bool(mask[q, _first_token(info, TOKEN_GIST, 3)]) is True
    assert bool(mask[q, _first_token(info, TOKEN_GIST, 0)]) is True


def test_pred_sees_sliding_recent_window():
    """With recent=2, q0=4 retains clean physical frames 2 and 3."""
    info = _vis(num_frames=4, obs=2, cs=2, anchor=0, recent=2)
    mask = info["mask"]
    q_act = _first_token(info, TOKEN_ACTION, 4)
    q_vid = _first_token(info, TOKEN_NOISY_VIDEO, 4)
    for q in (q_act, q_vid):
        for phys, expect in [(0, False), (1, False), (2, True), (3, True)]:
            k = _first_token(info, TOKEN_CLEAN_VIDEO, phys)
            assert bool(mask[q, k]) is expect, f"recent=2 q phys4 → clean{phys}"


def test_gist_zero_keeps_sliding_clean_history_without_gist_tokens():
    info = _vis(num_frames=4, obs=2, cs=2, gbs=0, anchor=2, recent=2)
    mask = info["mask"]
    assert int((info["token_types"] == TOKEN_GIST).sum()) == 0

    q_act = _first_token(info, TOKEN_ACTION, 4)
    for phys, expect in [(0, True), (1, True), (2, True), (3, True)]:
        k = _first_token(info, TOKEN_CLEAN_VIDEO, phys)
        assert bool(mask[q_act, k]) is expect, f"a2/r2/g0 q phys4 → clean{phys}"


def test_eviction_keep_formula_anchor0_recent1():
    frame_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    is_gist = torch.tensor([False, True, False, True, False, True])
    fmax, recent, anchor = 2, 1, 0
    recent_lo = fmax - recent + 1
    keep = is_gist | (frame_ids >= recent_lo)
    if anchor > 0:
        keep = keep | (frame_ids < anchor)
    assert keep.tolist() == [False, True, False, True, True, True]


def test_infer_history_set_matches_q0_rule():
    """After eviction with anchor=0,recent=1, retained fulls == adjacent for next q0."""
    # Simulate fills of frames 0..3; after fmax=3, recent_lo=3 → retained full phys {3}
    frame_ids = []
    is_gist = []
    vbs, gbs = 2, 2
    for f in range(4):
        frame_ids += [f] * vbs + [f] * gbs
        is_gist += [False] * vbs + [True] * gbs
    frame_ids = torch.tensor(frame_ids)
    is_gist = torch.tensor(is_gist)
    fmax, recent, anchor = 3, 1, 0
    recent_lo = fmax - recent + 1
    keep = is_gist | (frame_ids >= recent_lo)
    if anchor > 0:
        keep = keep | (frame_ids < anchor)
    retained_full = set(frame_ids[(keep) & (~is_gist)].tolist())
    assert retained_full == {3}
    # Next predict chunk with q0=4 → §1.1 adjacent full is phys 3
    q0 = 4
    assert retained_full == {q0 - 1}
