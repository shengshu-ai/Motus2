"""Per-episode sampling-rate resolution for Tianji data.

Tests cover mixed capture rates, metadata precedence, integer stride
validation, and alignment of action samples to the latent-frame timeline.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.tianji_sharpa.tianji_sharpa_dataset import (  # noqa: E402
    _episode_rate_stride,
    _stream_hz,
)
from data.tianji_wuji.tianji_wuji_dataset import (  # noqa: E402
    _episode_rate_stride as wuji_episode_rate_stride,
    _stream_hz as wuji_stream_hz,
)

def stride(hz, target, fallback=1):
    return _episode_rate_stride(
        hz, target, fallback=fallback, stream="action", episode="ep"
    )


def test_stream_hz_prefers_hz_then_timeline_hz():
    assert _stream_hz({"hz": 60, "timeline_hz": 60.0}) == 60.0
    assert _stream_hz({"timeline_hz": 30.0}) == 30.0
    # Only the pair of rate keys counts; an unrelated payload has no rate.
    assert _stream_hz({"left": [], "right": []}) is None
    assert _stream_hz(None) is None


@pytest.mark.parametrize("bad", [0, -1, "sixty", None])
def test_stream_hz_ignores_unusable_values(bad):
    assert _stream_hz({"hz": bad}) is None


def test_the_rates_actually_on_disk_resolve_to_the_expected_strides():
    """Every (rate, target) pair the Sharpa/Wuji dirs actually contain."""
    # Action streams against a 30 Hz training timeline.
    assert stride(30, 30) == 1          # already converted
    assert stride(60, 30) == 2          # pre-conversion capture
    # Integer-multiple rates still resolve exactly (generic stride math).
    assert stride(90, 30) == 3
    assert stride(120, 30) == 4
    assert stride(90, 90) == 1


def test_mixed_rates_normalize_to_a_common_timeline():
    """The whole point: two rates, one post-stride timeline.

    30 Hz and 60 Hz captures of the same 10 seconds must end up the same length,
    which is what lets them sit in one batch and share `_latent_to_action_range`.
    """
    assert len(range(0, 300, stride(30, 30))) == len(range(0, 600, stride(60, 30)))


def test_non_integer_ratio_raises_instead_of_rounding():
    # 120 Hz against a 90 Hz target needs a 4/3 stride. Rounding would silently
    # read actions from the wrong timestamps.
    with pytest.raises(ValueError, match="not an integer multiple"):
        stride(120, 90)
    with pytest.raises(ValueError, match="not an integer multiple"):
        stride(50, 30)


def test_rate_below_the_training_rate_raises():
    """Striding can only decimate; upsampling has to be an explicit decision."""
    with pytest.raises(ValueError, match="not an integer multiple"):
        stride(15, 30)


def test_error_names_the_episode_and_both_rates():
    with pytest.raises(ValueError) as excinfo:
        _episode_rate_stride(
            120, 90, fallback=1, stream="action", episode="/data/task/episode_007"
        )
    message = str(excinfo.value)
    assert "/data/task/episode_007" in message
    assert "action" in message
    assert "120" in message and "90" in message


def test_missing_rate_field_falls_back_to_the_config_stride():
    """Exports predating the `hz` field keep the old global behaviour."""
    assert stride(None, 30, fallback=2) == 2
    assert stride(None, 30, fallback=1) == 1
    # A nonsensical fallback still yields a usable stride rather than a slice
    # step of 0, which would raise deep inside the tensor indexing.
    assert stride(None, 30, fallback=0) == 1


def test_wuji_and_sharpa_helpers_agree():
    """Both loaders keep the same per-episode hz rules; task names are unused."""
    payload = {"hz": 60, "timeline_hz": 60.0}
    assert wuji_stream_hz(payload) == _stream_hz(payload)
    assert wuji_episode_rate_stride(
        30, 30, fallback=2, stream="action", episode="ep"
    ) == stride(30, 30, fallback=2)
    assert wuji_episode_rate_stride(
        60, 30, fallback=2, stream="action", episode="ep"
    ) == stride(60, 30, fallback=2)
    assert wuji_episode_rate_stride(
        None, 30, fallback=2, stream="action", episode="ep"
    ) == stride(None, 30, fallback=2)
