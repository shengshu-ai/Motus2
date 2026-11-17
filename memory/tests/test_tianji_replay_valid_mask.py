from pathlib import Path

import torch

from data.tianji_wuji.tianji_wuji_dataset import TianjiWujiDataset


def _dataset(payloads, *, enabled=True, stride=2):
    dataset = object.__new__(TianjiWujiDataset)
    dataset.mask_invalid_replay = enabled
    dataset.action_from_state = False
    dataset.adaptive_raw_hz_from_task_name = False
    dataset.action_downsample_stride = stride
    dataset._exists = lambda path: path in payloads
    dataset._load_pt = lambda path: payloads[path]
    return dataset


def test_replay_validity_downsamples_and_intersects_action_sources():
    hand_path = Path("/episode/hand_action.pt")
    wrist_path = Path("/episode/arm_action_endpose.pt")
    dataset = _dataset(
        {
            hand_path: {
                "replay_valid_flags": torch.tensor(
                    [True, True, True, True, False, False]
                )
            },
            wrist_path: {
                "replay_valid_flags": torch.tensor(
                    [True, True, False, False, True, True]
                )
            },
        }
    )

    valid = dataset._load_replay_validity(
        {
            "hand_action_path": hand_path,
            "arm_action_endpose_path": wrist_path,
            "task_name": "example_60hz",
        },
        expected_len=3,
    )

    assert valid.tolist() == [True, False, False]


def test_replay_validity_disabled_preserves_all_rows():
    dataset = _dataset({}, enabled=False)

    valid = dataset._load_replay_validity({}, expected_len=4)

    assert valid.tolist() == [True, True, True, True]


def test_replay_validity_rejects_short_metadata():
    wrist_path = Path("/episode/arm_action_endpose.pt")
    dataset = _dataset(
        {
            wrist_path: {
                "replay_valid_flags": torch.tensor([True, False, True])
            }
        }
    )

    try:
        dataset._load_replay_validity(
            {
                "arm_action_endpose_path": wrist_path,
                "task_name": "example_60hz",
            },
            expected_len=3,
        )
    except ValueError as exc:
        assert "expected at least 3" in str(exc)
    else:
        raise AssertionError("short replay_valid_flags should be rejected")
