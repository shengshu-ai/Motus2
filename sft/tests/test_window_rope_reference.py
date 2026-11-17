"""Tests for cross-modal video phases across repeated streaming cache pops."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from models.motus import Motus, MotusConfig
from models.rope import rope_params


class Window:
    _fixed_chunk_size = Motus._fixed_chunk_size
    _max_cache_tokens = Motus._max_cache_tokens
    _window_cache_rope_update = Motus._window_cache_rope_update
    _inference_video_start_frame = Motus._inference_video_start_frame
    _video_rope_is_global = Motus._video_rope_is_global
    _shift_cached_video_k_temporal = Motus._shift_cached_video_k_temporal

    def __init__(self, reference="training_window", window=8, history=2):
        self.config = SimpleNamespace(fixed_chunk_size=2, chunk_window_size=history,
                                      max_latent_window=window, video_rope_mode="window",
                                      video_rope_window_reference=reference)
        self.frame_seqlen = 1
        self._window_rope_cache_start = 0
        self.freqs = rope_params(64, 12)

    def _ensure_freqs_device(self):
        pass


@pytest.mark.parametrize("window,history", [(8, 2), (10, 2), (8, 3), (10, 1)])
def test_cache_phases_match_training_and_do_not_drift(window, history):
    w = Window(window=window, history=history)
    cache = torch.empty(1, 0, 1, 12, dtype=torch.float64)
    ct = 6 - 2 * (6 // 3)
    q = torch.randn(1, 1, 1, 12, dtype=torch.float64)
    for step in range(20):
        start = w._inference_video_start_frame(cache.shape[1])
        # Two identical unrotated keys make the expected temporal positions explicit.
        z = torch.ones(1, 2, 1, 6, dtype=torch.complex128)
        z[..., :ct] *= w.freqs[start:start + 2, :ct].view(1, 2, 1, ct)
        added = torch.view_as_real(z).flatten(-2)
        origin, shift = w._window_cache_rope_update(cache.shape[1], 2)
        cache = torch.cat([cache, added], 1)[:, -history * 2:]
        cache = w._shift_cached_video_k_temporal(cache, shift)
        w._window_rope_cache_start = origin
        end = min((step + 1) * 2, window - 2)
        positions = torch.arange(end - cache.shape[1], end)
        expected_z = torch.ones(1, cache.shape[1], 1, 6, dtype=torch.complex128)
        expected_z[..., :ct] *= w.freqs[positions, :ct].view(1, -1, 1, ct)
        expected = torch.view_as_real(expected_z).flatten(-2)
        torch.testing.assert_close(cache, expected, atol=1e-12, rtol=1e-12)
        # Action/state phases remain chunk-local; their logits must see training video phases.
        torch.testing.assert_close((q * cache).sum(-1), (q * expected).sum(-1))
        assert w._inference_video_start_frame(cache.shape[1]) == end


def test_global_start_remains_absolute():
    w = Window()
    w.config.video_rope_mode = "global"
    w.current_start_frame = 123
    assert w._inference_video_start_frame(4) == 123


@pytest.mark.parametrize("kwargs", [
    {"max_latent_window": 7},
    {"max_latent_window": 8, "chunk_window_size": 4},
])
def test_invalid_training_reference_rejected(kwargs):
    args = dict(video_rope_mode="window",
                max_latent_window=8, fixed_chunk_size=2, chunk_window_size=2)
    args.update(kwargs)
    with pytest.raises(ValueError):
        MotusConfig(**args)
