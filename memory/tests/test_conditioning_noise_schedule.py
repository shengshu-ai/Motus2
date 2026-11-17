# tests/test_conditioning_noise_schedule.py
import importlib.util
from pathlib import Path

# models/motus.py pulls in the Wan text encoder, which resolves torch.cuda at import time.
# Load the schedule helper directly so the contract is testable on a CPU box.
_path = Path(__file__).parent.parent / "models" / "memory_noise.py"
_spec = importlib.util.spec_from_file_location("memory_noise_direct", _path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

conditioning_noise_sigma = _mod.conditioning_noise_sigma


def test_zero_prefix_reproduces_uniform_schedule():
    """clean_prefix=0 must stay bit-identical to MemoryWAM's uniform augmentation."""
    assert conditioning_noise_sigma(6, 1.0, 0) == [1.0] * 6
    assert conditioning_noise_sigma(6, 0.1, 0) == [0.1] * 6


def test_prefix_latents_are_exempt():
    assert conditioning_noise_sigma(6, 1.0, 4) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_prefix_is_clamped_to_window():
    """Episodes shorter than the prefix are legal: RMBench windows run from 7 to 102 latents."""
    assert conditioning_noise_sigma(3, 1.0, 4) == [0.0, 0.0, 0.0]
    assert conditioning_noise_sigma(0, 1.0, 4) == []


def test_negative_prefix_is_treated_as_no_exemption():
    assert conditioning_noise_sigma(3, 1.0, -2) == [1.0, 1.0, 1.0]


def test_disabled_augmentation_stays_disabled_under_a_prefix():
    assert conditioning_noise_sigma(4, 0.0, 2) == [0.0] * 4
