"""FlashVLA/FlashRT CUDA extension loader. Callers fall back to PyTorch when it fails."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


_REQUIRED_KERNEL_SYMBOLS = (
    # sm_100/sm_120 common subset; fp8 GEMM ops vary by arch so we don't gate on them.
    "bias_rms_norm_bf16",
    "ada_layer_norm_bf16",
    "gate_mul_residual_out_bf16",
    "rope_apply_bf16_to_bf16",
)


def _ensure_flashvla_path() -> None:
    """Make ``flash_rt`` importable: installed package first, else ``FLASHVLA_ROOT``."""
    if importlib.util.find_spec("flash_rt") is not None:
        return
    raw = os.environ.get("FLASHVLA_ROOT")
    if not raw:
        raise ImportError(
            "FlashRT (https://github.com/flashrt-project/FlashRT) is not installed; "
            "install it or set FLASHVLA_ROOT to a built checkout"
        )
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"FLASHVLA_ROOT does not exist: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def load_flashvla_kernels() -> tuple[Any, Any]:
    """Import compiled FlashRT kernel modules. fa2 is optional (disabled on sm_100)."""
    _ensure_flashvla_path()
    kernels = importlib.import_module("flash_rt.flash_rt_kernels")
    try:
        fa2 = importlib.import_module("flash_rt.flash_rt_fa2")
    except ImportError:
        # FA2 in-SO is disabled on sm_100 (Thor SM110+ uses cuBLAS path). Not required for N/R/A.
        fa2 = None
    missing = [name for name in _REQUIRED_KERNEL_SYMBOLS if not hasattr(kernels, name)]
    if missing:
        raise RuntimeError(f"FlashVLA kernels missing required symbols: {missing}")
    return kernels, fa2
