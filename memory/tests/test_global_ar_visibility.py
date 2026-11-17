# tests/test_global_ar_visibility.py
"""Global-AR arm: no memory tokens, unbounded clean history.

The hybrid-memory visibility is parameterised by anchor/recent/gist, so the
global-AR ablation is expressible without a separate mask: gist=0 removes the
compressed memory, and a recent window larger than any episode makes the
sliding window non-binding. These tests pin that equivalence so the arm cannot
silently drift back into a windowed model.
"""
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

# Must match model.memory.recent_latents in the global-AR training config.
GLOBAL_AR_RECENT = 100000
# Longest find_the_square episode on disk is 51 latents; max_latent_window is 128.
LONGEST_EPISODE_LATENTS = 51


def _vis(num_frames, *, vbs=2, gbs=0, abs_=1, cs=2, obs=1, anchor=0,
         recent=GLOBAL_AR_RECENT, mode="fdm"):
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


def _tokens(info, typ, phys):
    t, p = info["token_types"], info["phys_ids"]
    return ((t == typ) & (p == phys)).nonzero(as_tuple=True)[0]


def _first(info, typ, phys):
    idx = _tokens(info, typ, phys)
    assert idx.numel() > 0, f"no token type={typ} phys={phys}"
    return int(idx[0])


def test_no_memory_tokens_are_emitted():
    info = _vis(LONGEST_EPISODE_LATENTS)
    assert int((info["token_types"] == TOKEN_GIST).sum()) == 0


def test_clean_query_sees_every_earlier_clean_latent():
    n = LONGEST_EPISODE_LATENTS
    info = _vis(n)
    mask = info["mask"]
    for pq in (1, n // 2, n - 1):
        q = _first(info, TOKEN_CLEAN_VIDEO, pq)
        for pk in range(n):
            k = _first(info, TOKEN_CLEAN_VIDEO, pk)
            assert bool(mask[q, k]) is (pk <= pq), f"clean{pq} -> clean{pk}"


def test_prediction_query_sees_every_clean_latent_before_its_chunk():
    n = LONGEST_EPISODE_LATENTS
    info = _vis(n)
    mask = info["mask"]
    q0_values = sorted({int(p) for p in info["phys_ids"][info["token_types"] == TOKEN_ACTION]})
    for q0 in (q0_values[0], q0_values[len(q0_values) // 2], q0_values[-1]):
        for typ in (TOKEN_ACTION, TOKEN_NOISY_VIDEO):
            q = _first(info, typ, q0)
            for pk in range(n):
                k = _first(info, TOKEN_CLEAN_VIDEO, pk)
                assert bool(mask[q, k]) is (pk < q0), f"pred{q0} -> clean{pk}"


def test_window_is_not_binding_for_any_reachable_episode_length():
    """A shorter recent window would change the mask; the global one must not."""
    n = LONGEST_EPISODE_LATENTS
    unbounded = _vis(n)["mask"]
    exact = _vis(n, recent=n)["mask"]
    assert torch.equal(unbounded, exact)
    windowed = _vis(n, recent=4)["mask"]
    assert not torch.equal(unbounded, windowed)


def test_intra_chunk_fdm_rules_are_unchanged():
    info = _vis(8, cs=2, obs=1)
    mask = info["mask"]
    q0 = sorted({int(p) for p in info["phys_ids"][info["token_types"] == TOKEN_ACTION]})[1]
    q_act = _first(info, TOKEN_ACTION, q0)
    q_vid = _first(info, TOKEN_NOISY_VIDEO, q0)
    # action reads same-chunk action only; noisy video reads action and video.
    assert bool(mask[q_act, _first(info, TOKEN_ACTION, q0)]) is True
    assert bool(mask[q_act, _first(info, TOKEN_NOISY_VIDEO, q0)]) is False
    assert bool(mask[q_vid, _first(info, TOKEN_ACTION, q0)]) is True
    assert bool(mask[q_vid, _first(info, TOKEN_NOISY_VIDEO, q0)]) is True


def test_streaming_cache_never_evicts():
    """_fill_cache_from_latent keeps a token when frame_id >= fmax - recent + 1."""
    vbs = 2
    frame_ids = torch.arange(LONGEST_EPISODE_LATENTS).repeat_interleave(vbs)
    is_gist = torch.zeros_like(frame_ids, dtype=torch.bool)
    fmax = int(frame_ids.max())
    keep = is_gist | (frame_ids >= fmax - GLOBAL_AR_RECENT + 1)
    assert bool(keep.all())
