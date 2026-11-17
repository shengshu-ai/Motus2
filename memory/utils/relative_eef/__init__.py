"""Relative EEF conversion utilities."""
import sys
from pathlib import Path

_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))
from relative_eef import (
    abs_eef_14d_to_relative_eef_20d,
    relative_eef_20d_to_abs_eef_14d,
    abs_eef_16d_to_relative_eef_20d,
    abs_eef_16d_to_relative_eef_20d_torch,
    abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch,
    abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch,
    abs_eef_16d_to_abs_eef_20d_torch,
    abs_eef_20d_to_abs_eef_16d_torch,
    abs_eef_20d_to_relative_eef_20d_torch,
    relative_eef_20d_to_abs_eef_20d_torch,
    relative_eef_20d_to_abs_eef_16d,
    relative_eef_20d_to_abs_eef_16d_torch,
    relative_eef_xyzw_16d_to_abs_eef_xyzw_16d_torch,
    relative_eef_wxyz_16d_to_abs_eef_16d_torch,
    abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch,
    rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch,
    eef_quat_wxyz_to_xyzw,
    eef_quat_xyzw_to_wxyz,
)

__all__ = [
    "abs_eef_14d_to_relative_eef_20d",
    "relative_eef_20d_to_abs_eef_14d",
    "abs_eef_16d_to_relative_eef_20d",
    "abs_eef_16d_to_relative_eef_20d_torch",
    "abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch",
    "abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch",
    "abs_eef_16d_to_abs_eef_20d_torch",
    "abs_eef_20d_to_abs_eef_16d_torch",
    "abs_eef_20d_to_relative_eef_20d_torch",
    "relative_eef_20d_to_abs_eef_20d_torch",
    "eef_quat_wxyz_to_xyzw",
    "relative_eef_20d_to_abs_eef_16d",
    "relative_eef_20d_to_abs_eef_16d_torch",
    "relative_eef_xyzw_16d_to_abs_eef_xyzw_16d_torch",
    "relative_eef_wxyz_16d_to_abs_eef_16d_torch",
    "abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch",
    "rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch",
    "eef_quat_xyzw_to_wxyz",
]
