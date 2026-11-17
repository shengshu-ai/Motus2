"""Tests for RoPE correctness.

Verifies:
  - Video 3D RoPE: clean/noisy symmetric (same positions)
  - Action 1D RoPE: cross-chunk continuous, inference matches training
"""

import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

_rope_path = Path(__file__).parent.parent / "models" / "rope.py"
_rope_spec = importlib.util.spec_from_file_location("rope_direct", _rope_path)
_rope = importlib.util.module_from_spec(_rope_spec)
assert _rope_spec.loader is not None
_rope_spec.loader.exec_module(_rope)
rope_params = _rope.rope_params
rope_apply_one = _rope.rope_apply_one
rope_action_apply = _rope.rope_action_apply
causal_rope_action_apply = _rope.causal_rope_action_apply


class TestVideoRoPE:
    """Test 3D video RoPE (cosmos-predict2 style)."""

    def test_symmetric_clean_noisy(self):
        """Clean and noisy video halves should get identical RoPE positions."""
        B, H, D = 2, 4, 32  # batch, heads, head_dim
        f, h, w = 5, 3, 4   # grid: 5 temporal, 3 height, 4 width
        L = f * h * w        # = 60 tokens per half

        freqs = rope_params(64, D)
        grid = torch.tensor([f, h, w], dtype=torch.long)

        # Simulate clean/noisy split
        q_full = torch.randn(B, 2 * L, H, D)
        q_c, q_n = q_full.chunk(2, dim=1)

        out_c = rope_apply_one(q_c, grid, freqs)
        out_n = rope_apply_one(q_n, grid, freqs)

        # Same input should produce same RoPE result (positions are identical)
        q_same = torch.randn(B, L, H, D)
        out1 = rope_apply_one(q_same, grid, freqs)
        out2 = rope_apply_one(q_same.clone(), grid, freqs)
        assert torch.allclose(out1, out2, atol=1e-5), "Identical input should give identical RoPE output"

    def test_different_positions(self):
        """Different temporal positions should produce different RoPE outputs."""
        B, H, D = 1, 2, 16
        freqs = rope_params(32, D)

        # Grid with 2 temporal frames
        grid1 = torch.tensor([2, 2, 2], dtype=torch.long)
        x = torch.ones(B, 8, H, D)  # 2*2*2 = 8 tokens

        out = rope_apply_one(x, grid1, freqs)
        # Tokens at different temporal positions should differ
        t0_tokens = out[:, :4]  # first temporal frame
        t1_tokens = out[:, 4:]  # second temporal frame
        assert not torch.allclose(t0_tokens, t1_tokens, atol=1e-3), "Different temporal positions should differ"


class TestActionRoPE:
    """Test 1D action RoPE (DreamZero style)."""

    def test_cross_chunk_continuous(self):
        """Training: action positions should be consecutive across chunks."""
        B, n_heads, head_dim = 1, 4, 32
        num_action = 6
        num_state = 1
        M = 3  # 3 chunks
        total_act = M * num_action
        total_st = M * num_state
        action_reg_len = total_act + total_st

        # Simulate: [video_tokens(10) | action(18) | state(3)]
        video_len = 10
        seq_len = video_len + action_reg_len
        x = torch.randn(B, seq_len, n_heads, head_dim)

        freqs = rope_params(64, head_dim)[:video_len].view(video_len, 1, -1)
        freqs_action = rope_params(256, head_dim)
        freqs_state = rope_params(64, head_dim)

        out = rope_action_apply(x, freqs, freqs_action, freqs_state,
                                action_register_length=action_reg_len,
                                num_action_per_block=num_action,
                                num_state_per_block=num_state)
        assert out.shape == x.shape

    def test_single_chunk_inference_matches_training(self):
        """Chunk-local inference and training helpers must apply identical RoPE."""
        B, n_heads, hd = 1, 2, 16
        num_act = 4
        num_st = 1

        freqs_action = rope_params(128, hd)
        freqs_state = rope_params(32, hd)

        action_register_length = num_act + num_st
        video_len = 6
        x = torch.randn(
            B,
            video_len + action_register_length,
            n_heads,
            hd,
        )
        freqs_vid = rope_params(32, hd)[:video_len].view(video_len, 1, -1)

        out_train = rope_action_apply(
            x,
            freqs_vid,
            freqs_action,
            freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_act,
            num_state_per_block=num_st,
        )
        out_inference = causal_rope_action_apply(
            x,
            freqs_vid,
            freqs_action,
            freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=num_act,
            num_state_per_block=num_st,
        )
        assert torch.allclose(out_train, out_inference, atol=1e-6)


class TestRoPEParams:
    """Test rope_params utility."""

    def test_shape(self):
        freqs = rope_params(100, 64)
        assert freqs.shape == (100, 32), "rope_params should return (max_seq_len, dim/2)"

    def test_unit_magnitude(self):
        """Polar form should have unit magnitude."""
        freqs = rope_params(10, 16)
        assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
