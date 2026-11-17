"""Numerical checks of pretrain action timestamps, independent of model weights."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pretrain import pretrain_dataset as module
from data.pretrain.pretrain_dataset import PretrainDataset, _resolve_rate_config


def dataset():
    ds = PretrainDataset.__new__(PretrainDataset)
    ds.action_upsample_ratio = 1
    ds.action_downsample_stride = 1
    ds.video_action_freq_ratio = 6
    ds.cache_dir = None
    return ds


def resolve(raw, train=30):
    return _resolve_rate_config(
        video_hz=30, action_cfg={"raw_hz": raw, "train_hz": train},
        raw_action_hz=None, train_action_hz=None, action_upsample_ratio=1,
    )


def test_old_configuration_fails_before_loading_episodes():
    with pytest.raises(ValueError, match="does not resample action streams"):
        resolve(60)
    assert resolve(30) == (30, 30, 30, 1, 1)


@pytest.mark.parametrize("raw,train", [(0, 30), (-30, 30), (30, 0), (30, -30)])
def test_invalid_rates_fail_with_an_explicit_error(raw, train):
    with pytest.raises(ValueError, match="positive"):
        resolve(raw, train)


def test_each_action_target_retains_its_original_30hz_timestamp():
    ds = dataset()
    # Every value is the physical timestamp of one source action sample.
    source = torch.arange(288, dtype=torch.float64).view(-1, 1) / 30
    retained = ds._downsample_action_sequence(source)
    assert retained is source
    result = ds._gather_per_latent(retained, 0, 13, 6, 0, 24).view(13, 24)
    # The causal VAE's first latent represents one video frame; its action is
    # intentionally repeated to preserve the fixed tensor shape.
    torch.testing.assert_close(result[0], torch.zeros(24, dtype=torch.float64))
    for k in range(1, 12):
        start = 1 + (k - 1) * 24
        torch.testing.assert_close(result[k], source[start:start + 24, 0])
    # The final latent reaches the end of this real-world episode-length
    # example: only the missing final sample repeats, rather than half the run.
    torch.testing.assert_close(result[12, :-1], source[265:288, 0])
    assert result[12, -1] == source[-1, 0]
    # Reproduce the old sequence[::2] mistake on the same timestamp-labelled
    # input: it moves targets ahead and exhausts the stream halfway through.
    broken = ds._gather_per_latent(source[::2], 0, 13, 6, 0, 24).view(13, 24)
    assert broken[5, 0] == source[194, 0]
    assert result[5, 0] == source[97, 0]
    torch.testing.assert_close(broken[7], source[286, 0].expand(24))


def test_phase_offset_is_applied_in_the_original_action_timeline():
    ds = dataset()
    source = torch.arange(160).view(-1, 1)
    result = ds._gather_per_latent(source, 0, 3, 6, 4, 24).view(3, 24)
    torch.testing.assert_close(result[0], torch.full((24,), 4))
    torch.testing.assert_close(result[1], torch.arange(5, 29))
    torch.testing.assert_close(result[2], torch.arange(29, 53))
    torch.testing.assert_close(ds._gather_states(source, 1, 2, 6, 4)[:, 0], torch.tensor([4, 28]))


def test_runtime_stride_and_chunk_geometry_cannot_silently_change():
    ds = dataset()
    source = torch.arange(160).view(-1, 1)
    ds.action_downsample_stride = 2
    with pytest.raises(ValueError, match="resampling is disabled"):
        ds._downsample_action_sequence(source)
    with pytest.raises(ValueError, match="geometry mismatch"):
        ds._gather_per_latent(source, 0, 3, 3, 0, 24)
    with pytest.raises(ValueError, match="empty"):
        ds._gather_per_latent(source[:0], 0, 3, 6, 0, 24)


@pytest.mark.parametrize("right_stride,right_phase", [(3, 0), (6, 1)])
def test_stereo_latents_require_the_same_time_coordinates(monkeypatch, right_stride, right_phase):
    payloads = {
        "left": {"latents": torch.zeros(48, 4, 2, 2), "frame_stride": 6, "phase_start_30": 0},
        "right": {"latents": torch.zeros(48, 4, 2, 2), "frame_stride": right_stride, "phase_start_30": right_phase},
    }
    monkeypatch.setattr(module, "smart_load_pt", lambda path, *args, **kwargs: payloads[path])
    with pytest.raises(ValueError, match="stereo latent timeline mismatch"):
        dataset()._load_stereo_latent_candidate({"left_latent_path": "left", "right_latent_path": "right"})
