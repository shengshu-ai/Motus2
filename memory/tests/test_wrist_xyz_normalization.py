"""Wrist normalization regression for the public [0,1] min-max contract."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The wan import chain calls torch.cuda.current_device() while evaluating a class
# body, so importing the model needs a device even though this math is pure CPU.

from models.motus import Motus, MotusConfig  # noqa: E402
from models.rope import (  # noqa: E402
    causal_rope_action_apply,
    rope_params,
    rope_video_basis_freqs,
)
from utils.action_mode import (  # noqa: E402
    resolve_action_mode_from_config,
    resolve_normalization_block,
)

STATS_PATH = ROOT / "tests/fixtures/wrist_stats.json"


class _WristOnly:
    """Binds the wrist scaling methods to a config without building the model."""

    def __init__(self, config):
        self.config = config

    scale = Motus._scale_qpos20_rel_wrist_xyz
    unscale = Motus._unscale_qpos20_rel_wrist_xyz
    _qpos20_wrist_xyz_norm_mode = Motus._qpos20_wrist_xyz_norm_mode
    _qpos20_wrist_xyz_affine = Motus._qpos20_wrist_xyz_affine
    _qpos20_wrist_xyz_mean_std = Motus._qpos20_wrist_xyz_mean_std
    _qpos20_wrist_xyz_quantile = Motus._qpos20_wrist_xyz_quantile
    _extract_qpos20_rel_wrist_xyz6 = Motus._extract_qpos20_rel_wrist_xyz6
    _write_qpos20_rel_wrist_xyz6 = Motus._write_qpos20_rel_wrist_xyz6


class _StreamingRopeOnly:
    """Binds streaming RoPE methods without constructing the 5B model."""

    _action_freqs_video_basis = Motus._action_freqs_video_basis
    _maybe_action_video_basis_override = Motus._maybe_action_video_basis_override
    build = Motus._build_infer_chunk_freqs

    def __init__(self, mode: str):
        self.config = SimpleNamespace(
            action_rope_mode=mode,
            action_tokens_per_frame=2,
            state_tokens_per_frame=0,
            use_state=False,
        )
        self.current_start_frame = 7
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.wan_dim = 8
        self.video_model = SimpleNamespace(
            wan_model=SimpleNamespace(num_heads=1),
        )
        self.freqs = torch.ones(1, 4, dtype=torch.complex128)
        self.freqs_state = torch.ones(8, 4, dtype=torch.complex128)
        self.freqs_action = rope_params(32, 8)

    def _ensure_freqs_device(self):
        pass


def _wrist_stats() -> dict:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return stats["normalization"]["wrist_relative"]


def _config(**overrides) -> MotusConfig:
    base = dict(
        wan_checkpoint_path="",
        vae_path="",
        action_mode="qpos20_rel6d",
        action_dim=58,
        action_state_dim=58,
    )
    base.update(overrides)
    return MotusConfig(**base)


def _action58(xyz6) -> torch.Tensor:
    action = torch.zeros(3, 58, dtype=torch.float64)
    action[:, 20:23] = torch.tensor(xyz6[:3], dtype=torch.float64)
    action[:, 49:52] = torch.tensor(xyz6[3:], dtype=torch.float64)
    # rot6d stays identity so we can assert it is never touched.
    action[:, 23:29] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    action[:, 52:58] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    return action


def test_mean_std_uses_stats_and_not_the_fixed_affine():
    """mean_std divides by the configured standard deviation."""
    wrist = _wrist_stats()
    cfg = _config(
        wrist_xyz_normalization="mean_std",
        qpos20_wrist_xyz_mean=tuple(wrist["xyz_mean"]),
        qpos20_wrist_xyz_std=tuple(wrist["xyz_std"]),
        wrist_xyz_std_min=float(wrist.get("std_min", 1e-4)),
    )
    xyz6 = [0.01, -0.02, 0.03, -0.04, 0.02, -0.01]
    got = _WristOnly(cfg).scale(_action58(xyz6))

    mean = torch.tensor(wrist["xyz_mean"], dtype=torch.float64)
    std = torch.tensor(wrist["xyz_std"], dtype=torch.float64).clamp(
        min=float(wrist.get("std_min", 1e-4))
    )
    expected = (torch.tensor(xyz6, dtype=torch.float64) - mean) / std

    assert torch.allclose(got[0, 20:23], expected[:3], atol=1e-12)
    assert torch.allclose(got[0, 49:52], expected[3:], atol=1e-12)
    # The configured standard deviation differs from a fixed-scale transform.
    assert not torch.allclose(got[0, 20:23], torch.tensor(xyz6[:3], dtype=torch.float64) * 20.0, atol=1e-3)


def test_mean_std_does_not_clip_large_motions():
    """min_max clamps at +-0.05 m; mean_std must pass fast motions through."""
    wrist = _wrist_stats()
    cfg = _config(
        wrist_xyz_normalization="mean_std",
        qpos20_wrist_xyz_mean=tuple(wrist["xyz_mean"]),
        qpos20_wrist_xyz_std=tuple(wrist["xyz_std"]),
    )
    far = [0.30, -0.30, 0.30, -0.30, 0.30, -0.30]
    got = _WristOnly(cfg).scale(_action58(far))
    assert got[0, 20:23].abs().min() > 3.0

    clipped = _WristOnly(_config()).scale(_action58(far))
    assert torch.equal(clipped[0, 20:23], torch.tensor([1., 0., 1.], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["min_max", "mean_std", "quantile"])
def test_roundtrip_and_rot6d_untouched(mode: str):
    wrist = _wrist_stats()
    kwargs = {"wrist_xyz_normalization": mode}
    if mode == "mean_std":
        kwargs.update(
            qpos20_wrist_xyz_mean=tuple(wrist["xyz_mean"]),
            qpos20_wrist_xyz_std=tuple(wrist["xyz_std"]),
        )
    elif mode == "quantile":
        kwargs.update(
            qpos20_wrist_xyz_min=tuple(wrist["xyz_q01"]),
            qpos20_wrist_xyz_max=tuple(wrist["xyz_q99"]),
        )
    model = _WristOnly(_config(**kwargs))

    raw = _action58([0.01, -0.02, 0.03, -0.04, 0.02, -0.01])
    restored = model.unscale(model.scale(raw))

    assert torch.allclose(restored[..., 20:23], raw[..., 20:23], atol=1e-9)
    assert torch.allclose(restored[..., 49:52], raw[..., 49:52], atol=1e-9)
    assert torch.equal(restored[..., 23:29], raw[..., 23:29])
    assert torch.equal(restored[..., 52:58], raw[..., 52:58])


def test_default_config_uses_zero_one_min_max():
    """The public min_max maps the lower bound to zero and upper to one."""
    cfg = _config()
    assert cfg.wrist_xyz_normalization == "min_max"
    xyz6 = [0.01, -0.02, 0.03, -0.04, 0.02, -0.01]
    got = _WristOnly(cfg).scale(_action58(xyz6))
    expected = torch.tensor(xyz6, dtype=torch.float64) * 10.0 + 0.5
    assert torch.allclose(got[0, 20:23], expected[:3], atol=1e-12)
    assert torch.allclose(got[0, 49:52], expected[3:], atol=1e-12)


def test_streaming_video_shared_action_rope_matches_training_clock():
    model = _StreamingRopeOnly("video_shared")
    noisy_video_freqs = torch.ones(2, 1, 4, dtype=torch.complex128)

    got = model.build(noisy_video_freqs, K=2)
    expected_action = rope_video_basis_freqs(
        torch.tensor([7.0, 7.5, 8.0, 8.5]),
        8,
        device="cpu",
        dtype=torch.complex128,
    )

    assert torch.equal(got[:4], expected_action)
    assert torch.equal(got[4:], noisy_video_freqs)


def test_streaming_chunk_local_action_rope_keeps_legacy_semantics():
    model = _StreamingRopeOnly("chunk_local")
    noisy_video_freqs = torch.ones(2, 1, 4, dtype=torch.complex128)

    got = model.build(noisy_video_freqs, K=2)

    assert torch.equal(got[:4], model.freqs_action[:4].view(4, 1, -1))
    assert torch.equal(got[4:], noisy_video_freqs)


def test_action_only_rope_accepts_video_shared_override():
    x = torch.zeros(1, 2, 1, 8, dtype=torch.float32)
    x[..., 0::2] = 1.0
    override = rope_video_basis_freqs(
        torch.tensor([7.0, 7.5]),
        8,
        device="cpu",
        dtype=torch.complex128,
    )

    got = causal_rope_action_apply(
        x,
        freqs=torch.ones(0, 1, 4, dtype=torch.complex128),
        freqs_action=rope_params(8, 8),
        freqs_state=torch.ones(8, 4, dtype=torch.complex128),
        action_register_length=2,
        num_action_per_block=2,
        num_state_per_block=0,
        num_value_per_block=0,
        action_freqs_override=override,
    )
    got_complex = torch.view_as_complex(got.reshape(1, 2, 1, 4, 2))

    assert torch.allclose(got_complex, override.unsqueeze(0))


def test_nested_normalization_block_is_no_longer_a_silent_no_op():
    """A wan-style YAML fragment must resolve, not get dropped."""
    model_cfg = {
        "action_mode": "qpos20_rel6d",
        "normalization": {"hand": "mean_std", "wrist_xyz": "mean_std"},
    }
    resolved = resolve_action_mode_from_config(model_cfg, {})
    assert resolved.action_mode == "qpos20_rel6d"
    assert resolved.hand_normalization == "mean_std"
    assert resolved.wrist_xyz_normalization == "mean_std"


def test_removed_dataset_normalization_alias_is_rejected():
    with pytest.raises(ValueError, match="model.normalization"):
        resolve_action_mode_from_config(
            {"action_mode": "qpos20_rel6d"}, {"qpos_normalization": "mean_std"}
        )


def test_conflicting_hand_normalization_is_rejected():
    with pytest.raises(ValueError):
        resolve_action_mode_from_config(
            {"action_mode": "qpos20_rel6d", "normalization": {"hand": "mean_std"}},
            {"qpos_normalization": "min_max"},
        )


def test_train_stats_loaders_match_the_stats_file():
    sys.path.insert(0, str(ROOT / "train"))
    from train import (  # noqa: E402
        _qpos20_wrist_xyz_bounds_from_stats,
        _qpos20_wrist_xyz_mean_std_from_stats,
    )

    dataset_cfg = {"action_stats_path": str(STATS_PATH)}
    wrist = _wrist_stats()

    mean, std, std_min = _qpos20_wrist_xyz_mean_std_from_stats(dataset_cfg)
    assert list(mean) == pytest.approx(wrist["xyz_mean"])
    assert list(std) == pytest.approx(wrist["xyz_std"])
    assert std_min == pytest.approx(float(wrist.get("std_min", 1e-4)))

    lo, hi = _qpos20_wrist_xyz_bounds_from_stats(dataset_cfg, "quantile")
    assert list(lo) == pytest.approx(wrist["xyz_q01"])
    assert list(hi) == pytest.approx(wrist["xyz_q99"])

    lo, hi = _qpos20_wrist_xyz_bounds_from_stats(dataset_cfg, "min_max")
    assert list(lo) == pytest.approx(wrist["xyz_min"])
    assert list(hi) == pytest.approx(wrist["xyz_max"])
