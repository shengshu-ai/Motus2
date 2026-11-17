from typing import Optional

import numpy as np
import torch
from pose_utils import pose_to_mat, mat_to_pose10d, pose10d_to_mat, mat_to_pose, pose7d_to_mat
from pose_repr_utils import convert_pose_mat_rep

# Wrist block appended after each hand in the qpos{N}_rel6d layout: xyz(3) + rot6d(6).
QPOS_WRIST_DIM = 9


def normalize_gripper(gripper, gripper_min, gripper_max):
    """Normalize gripper to [0, 1]"""
    return (gripper - gripper_min) / (gripper_max - gripper_min)


def denormalize_gripper(normalized_gripper, gripper_min, gripper_max):
    """Denormalize gripper from [0, 1]"""
    return normalized_gripper * (gripper_max - gripper_min) + gripper_min


def eef_quat_wxyz_to_xyzw(eef):
    """Convert 16D endpose quaternions from wxyz to xyzw order.

    Input/output layout:
      [left_xyz(3), left_quat(4), left_gripper(1),
       right_xyz(3), right_quat(4), right_gripper(1)]
    """
    if isinstance(eef, torch.Tensor):
        out = eef.clone()
    else:
        out = np.array(eef, copy=True)
    out[..., 3:7] = eef[..., [4, 5, 6, 3]]
    out[..., 11:15] = eef[..., [12, 13, 14, 11]]
    return out


def eef_quat_xyzw_to_wxyz(eef):
    """Convert 16D endpose quaternions from xyzw to wxyz order."""
    if isinstance(eef, torch.Tensor):
        out = eef.clone()
    else:
        out = np.array(eef, copy=True)
    out[..., 3:7] = eef[..., [6, 3, 4, 5]]
    out[..., 11:15] = eef[..., [14, 11, 12, 13]]
    return out




def _quat_conjugate_xyzw_torch(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for xyzw quaternions."""
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _quat_multiply_xyzw_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for xyzw quaternions."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def _canonicalize_quat_wxyz_torch(quat: torch.Tensor) -> torch.Tensor:
    """Fix quaternion sign ambiguity by enforcing w >= 0 in wxyz order."""
    sign = torch.where(quat[..., :1] < 0, -1.0, 1.0)
    return quat * sign


def _canonicalize_quat_xyzw_torch(quat: torch.Tensor) -> torch.Tensor:
    """Fix quaternion sign ambiguity by enforcing w >= 0 in xyzw order."""
    sign = torch.where(quat[..., 3:4] < 0, -1.0, 1.0)
    return quat * sign


def _quat_xyzw_to_rotmat_torch(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion (xyzw) to rotation matrix."""
    quat = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)


def _mat_to_rot6d_torch(mat: torch.Tensor) -> torch.Tensor:
    """Use Zhou et al. 6D convention: first two rotation-matrix columns."""
    return torch.cat([mat[..., :, 0], mat[..., :, 1]], dim=-1)


def _rot6d_to_mat_torch(rot6d: torch.Tensor) -> torch.Tensor:
    """Inverse of the column-major Zhou 6D rotation representation."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - a2_proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _rotmat_to_quat_xyzw_torch(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion in xyzw order (broadcast-safe, no boolean indexing)."""
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    trace = m00 + m11 + m22

    t0 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2.0
    x0, y0, z0, w0 = (m21 - m12) / t0, (m02 - m20) / t0, (m10 - m01) / t0, 0.25 * t0

    t1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 2.0
    x1, y1, z1, w1 = 0.25 * t1, (m01 + m10) / t1, (m02 + m20) / t1, (m21 - m12) / t1

    t2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-12)) * 2.0
    x2, y2, z2, w2 = (m01 + m10) / t2, 0.25 * t2, (m12 + m21) / t2, (m02 - m20) / t2

    t3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-12)) * 2.0
    x3, y3, z3, w3 = (m02 + m20) / t3, (m12 + m21) / t3, 0.25 * t3, (m10 - m01) / t3

    cond0 = trace > 0
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    cond2 = (~cond0) & (~cond1) & (m11 > m22)

    x = torch.where(cond0, x0, torch.where(cond1, x1, torch.where(cond2, x2, x3)))
    y = torch.where(cond0, y0, torch.where(cond1, y1, torch.where(cond2, y2, y3)))
    z = torch.where(cond0, z0, torch.where(cond1, z1, torch.where(cond2, z2, z3)))
    w = torch.where(cond0, w0, torch.where(cond1, w1, torch.where(cond2, w2, w3)))

    return torch.nn.functional.normalize(torch.stack([x, y, z, w], dim=-1), dim=-1)


def abs_eef_16d_to_relative_eef_20d_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Torch batched version of abs_eef_16d_to_relative_eef_20d.

    Args:
        state: [..., 16]
        action_chunk: [..., T, 16]

    Returns:
        relative_actions: [..., T, 20]
    """
    state_left_xyz = state[..., :3]
    state_left_quat = state[..., 3:7]
    state_left_gripper = state[..., 7]
    state_right_xyz = state[..., 8:11]
    state_right_quat = state[..., 11:15]
    state_right_gripper = state[..., 15]

    action_left_xyz = action_chunk[..., :3]
    action_left_quat = action_chunk[..., 3:7]
    action_left_gripper = action_chunk[..., 7]
    action_right_xyz = action_chunk[..., 8:11]
    action_right_quat = action_chunk[..., 11:15]
    action_right_gripper = action_chunk[..., 15]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    action_left_rot = _quat_xyzw_to_rotmat_torch(action_left_quat)
    action_right_rot = _quat_xyzw_to_rotmat_torch(action_right_quat)

    state_left_rot_t = state_left_rot.transpose(-1, -2).unsqueeze(-3)
    state_right_rot_t = state_right_rot.transpose(-1, -2).unsqueeze(-3)

    left_delta = action_left_xyz - state_left_xyz.unsqueeze(-2)
    right_delta = action_right_xyz - state_right_xyz.unsqueeze(-2)

    relative_left_xyz = torch.matmul(state_left_rot_t, left_delta.unsqueeze(-1)).squeeze(-1)
    relative_right_xyz = torch.matmul(state_right_rot_t, right_delta.unsqueeze(-1)).squeeze(-1)
    relative_left_rot = torch.matmul(state_left_rot_t, action_left_rot)
    relative_right_rot = torch.matmul(state_right_rot_t, action_right_rot)

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    action_left_gripper_norm = normalize_gripper(action_left_gripper, gripper_min, gripper_max)
    relative_left_gripper = action_left_gripper_norm - state_left_gripper_norm.unsqueeze(-1)

    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    action_right_gripper_norm = normalize_gripper(action_right_gripper, gripper_min, gripper_max)
    relative_right_gripper = action_right_gripper_norm - state_right_gripper_norm.unsqueeze(-1)

    relative_left_10d = torch.cat(
        [relative_left_xyz, _mat_to_rot6d_torch(relative_left_rot)],
        dim=-1,
    )
    relative_right_10d = torch.cat(
        [relative_right_xyz, _mat_to_rot6d_torch(relative_right_rot)],
        dim=-1,
    )

    return torch.cat([
        relative_left_10d,
        relative_left_gripper.unsqueeze(-1),
        relative_right_10d,
        relative_right_gripper.unsqueeze(-1),
    ], dim=-1)


def abs_eef_wxyz_16d_to_relative_eef_wxyz_16d_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Convert absolute 16D wxyz EEF actions to 16D relative EEF with wxyz quaternions.

    Args:
        state: [..., 16] absolute anchor state in wxyz quaternion layout
        action_chunk: [..., T, 16] absolute actions in wxyz quaternion layout

    Returns:
        relative_actions: [..., T, 16]
          [left_dxyz(3), left_dquat_wxyz(4), left_dgrip(1),
           right_dxyz(3), right_dquat_wxyz(4), right_dgrip(1)]
    """
    state_xyzw = eef_quat_wxyz_to_xyzw(state)
    action_chunk_xyzw = eef_quat_wxyz_to_xyzw(action_chunk)

    state_left_xyz = state_xyzw[..., :3]
    state_left_quat = torch.nn.functional.normalize(state_xyzw[..., 3:7], dim=-1)
    state_left_gripper = state_xyzw[..., 7]
    state_right_xyz = state_xyzw[..., 8:11]
    state_right_quat = torch.nn.functional.normalize(state_xyzw[..., 11:15], dim=-1)
    state_right_gripper = state_xyzw[..., 15]

    action_left_xyz = action_chunk_xyzw[..., :3]
    action_left_quat = torch.nn.functional.normalize(action_chunk_xyzw[..., 3:7], dim=-1)
    action_left_gripper = action_chunk_xyzw[..., 7]
    action_right_xyz = action_chunk_xyzw[..., 8:11]
    action_right_quat = torch.nn.functional.normalize(action_chunk_xyzw[..., 11:15], dim=-1)
    action_right_gripper = action_chunk_xyzw[..., 15]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    state_left_rot_t = state_left_rot.transpose(-1, -2).unsqueeze(-3)
    state_right_rot_t = state_right_rot.transpose(-1, -2).unsqueeze(-3)

    left_delta = action_left_xyz - state_left_xyz.unsqueeze(-2)
    right_delta = action_right_xyz - state_right_xyz.unsqueeze(-2)
    relative_left_xyz = torch.matmul(state_left_rot_t, left_delta.unsqueeze(-1)).squeeze(-1)
    relative_right_xyz = torch.matmul(state_right_rot_t, right_delta.unsqueeze(-1)).squeeze(-1)

    inv_state_left_quat = _quat_conjugate_xyzw_torch(state_left_quat).unsqueeze(-2)
    inv_state_right_quat = _quat_conjugate_xyzw_torch(state_right_quat).unsqueeze(-2)
    relative_left_quat_xyzw = _quat_multiply_xyzw_torch(inv_state_left_quat, action_left_quat)
    relative_right_quat_xyzw = _quat_multiply_xyzw_torch(inv_state_right_quat, action_right_quat)
    relative_left_quat_wxyz = _canonicalize_quat_wxyz_torch(
        relative_left_quat_xyzw[..., [3, 0, 1, 2]]
    )
    relative_right_quat_wxyz = _canonicalize_quat_wxyz_torch(
        relative_right_quat_xyzw[..., [3, 0, 1, 2]]
    )

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    action_left_gripper_norm = normalize_gripper(action_left_gripper, gripper_min, gripper_max)
    relative_left_gripper = action_left_gripper_norm - state_left_gripper_norm.unsqueeze(-1)

    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    action_right_gripper_norm = normalize_gripper(action_right_gripper, gripper_min, gripper_max)
    relative_right_gripper = action_right_gripper_norm - state_right_gripper_norm.unsqueeze(-1)

    return torch.cat([
        relative_left_xyz,
        relative_left_quat_wxyz,
        relative_left_gripper.unsqueeze(-1),
        relative_right_xyz,
        relative_right_quat_wxyz,
        relative_right_gripper.unsqueeze(-1),
    ], dim=-1)


def abs_eef_xyzw_16d_to_relative_eef_xyzw_16d_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Convert absolute 16D xyzw EEF actions to 16D relative EEF actions."""
    state_left_xyz = state[..., :3]
    state_left_quat = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
    state_left_gripper = state[..., 7]
    state_right_xyz = state[..., 8:11]
    state_right_quat = torch.nn.functional.normalize(state[..., 11:15], dim=-1)
    state_right_gripper = state[..., 15]

    action_left_xyz = action_chunk[..., :3]
    action_left_quat = torch.nn.functional.normalize(action_chunk[..., 3:7], dim=-1)
    action_left_gripper = action_chunk[..., 7]
    action_right_xyz = action_chunk[..., 8:11]
    action_right_quat = torch.nn.functional.normalize(action_chunk[..., 11:15], dim=-1)
    action_right_gripper = action_chunk[..., 15]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    state_left_rot_t = state_left_rot.transpose(-1, -2).unsqueeze(-3)
    state_right_rot_t = state_right_rot.transpose(-1, -2).unsqueeze(-3)

    left_delta = action_left_xyz - state_left_xyz.unsqueeze(-2)
    right_delta = action_right_xyz - state_right_xyz.unsqueeze(-2)
    relative_left_xyz = torch.matmul(state_left_rot_t, left_delta.unsqueeze(-1)).squeeze(-1)
    relative_right_xyz = torch.matmul(state_right_rot_t, right_delta.unsqueeze(-1)).squeeze(-1)

    inv_state_left_quat = _quat_conjugate_xyzw_torch(state_left_quat).unsqueeze(-2)
    inv_state_right_quat = _quat_conjugate_xyzw_torch(state_right_quat).unsqueeze(-2)
    relative_left_quat_xyzw = _canonicalize_quat_xyzw_torch(
        _quat_multiply_xyzw_torch(inv_state_left_quat, action_left_quat)
    )
    relative_right_quat_xyzw = _canonicalize_quat_xyzw_torch(
        _quat_multiply_xyzw_torch(inv_state_right_quat, action_right_quat)
    )

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    action_left_gripper_norm = normalize_gripper(action_left_gripper, gripper_min, gripper_max)
    relative_left_gripper = action_left_gripper_norm - state_left_gripper_norm.unsqueeze(-1)

    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    action_right_gripper_norm = normalize_gripper(action_right_gripper, gripper_min, gripper_max)
    relative_right_gripper = action_right_gripper_norm - state_right_gripper_norm.unsqueeze(-1)

    return torch.cat([
        relative_left_xyz,
        relative_left_quat_xyzw,
        relative_left_gripper.unsqueeze(-1),
        relative_right_xyz,
        relative_right_quat_xyzw,
        relative_right_gripper.unsqueeze(-1),
    ], dim=-1)


def relative_eef_20d_to_abs_eef_16d_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Recover absolute 16D EEF actions from 20D relative EEF actions."""
    state_left_xyz = state[..., :3]
    state_left_quat = state[..., 3:7]
    state_left_gripper = state[..., 7]
    state_right_xyz = state[..., 8:11]
    state_right_quat = state[..., 11:15]
    state_right_gripper = state[..., 15]

    rel_left_xyz = relative_actions[..., :3]
    rel_left_rot6d = relative_actions[..., 3:9]
    rel_left_gripper = relative_actions[..., 9]
    rel_right_xyz = relative_actions[..., 10:13]
    rel_right_rot6d = relative_actions[..., 13:19]
    rel_right_gripper = relative_actions[..., 19]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    rel_left_rot = _rot6d_to_mat_torch(rel_left_rot6d)
    rel_right_rot = _rot6d_to_mat_torch(rel_right_rot6d)

    abs_left_xyz = state_left_xyz.unsqueeze(-2) + torch.matmul(
        state_left_rot.unsqueeze(-3), rel_left_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_right_xyz = state_right_xyz.unsqueeze(-2) + torch.matmul(
        state_right_rot.unsqueeze(-3), rel_right_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_left_rot = torch.matmul(state_left_rot.unsqueeze(-3), rel_left_rot)
    abs_right_rot = torch.matmul(state_right_rot.unsqueeze(-3), rel_right_rot)
    abs_left_quat = _rotmat_to_quat_xyzw_torch(abs_left_rot)
    abs_right_quat = _rotmat_to_quat_xyzw_torch(abs_right_rot)

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    abs_left_gripper_norm = torch.clamp(
        state_left_gripper_norm.unsqueeze(-1) + rel_left_gripper, 0.0, 1.0
    )
    abs_right_gripper_norm = torch.clamp(
        state_right_gripper_norm.unsqueeze(-1) + rel_right_gripper, 0.0, 1.0
    )
    abs_left_gripper = denormalize_gripper(abs_left_gripper_norm, gripper_min, gripper_max)
    abs_right_gripper = denormalize_gripper(abs_right_gripper_norm, gripper_min, gripper_max)

    return torch.cat([
        abs_left_xyz,
        abs_left_quat,
        abs_left_gripper.unsqueeze(-1),
        abs_right_xyz,
        abs_right_quat,
        abs_right_gripper.unsqueeze(-1),
    ], dim=-1)


def relative_eef_xyzw_16d_to_abs_eef_xyzw_16d_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Recover absolute 16D xyzw EEF actions from 16D relative EEF actions."""
    state_left_xyz = state[..., :3]
    state_left_quat = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
    state_left_gripper = state[..., 7]
    state_right_xyz = state[..., 8:11]
    state_right_quat = torch.nn.functional.normalize(state[..., 11:15], dim=-1)
    state_right_gripper = state[..., 15]

    rel_left_xyz = relative_actions[..., :3]
    rel_left_quat = torch.nn.functional.normalize(relative_actions[..., 3:7], dim=-1)
    rel_left_gripper = relative_actions[..., 7]
    rel_right_xyz = relative_actions[..., 8:11]
    rel_right_quat = torch.nn.functional.normalize(relative_actions[..., 11:15], dim=-1)
    rel_right_gripper = relative_actions[..., 15]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    abs_left_xyz = state_left_xyz.unsqueeze(-2) + torch.matmul(
        state_left_rot.unsqueeze(-3), rel_left_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_right_xyz = state_right_xyz.unsqueeze(-2) + torch.matmul(
        state_right_rot.unsqueeze(-3), rel_right_xyz.unsqueeze(-1)
    ).squeeze(-1)

    abs_left_quat = _quat_multiply_xyzw_torch(state_left_quat.unsqueeze(-2), rel_left_quat)
    abs_right_quat = _quat_multiply_xyzw_torch(state_right_quat.unsqueeze(-2), rel_right_quat)

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    abs_left_gripper_norm = torch.clamp(
        state_left_gripper_norm.unsqueeze(-1) + rel_left_gripper, 0.0, 1.0
    )
    abs_right_gripper_norm = torch.clamp(
        state_right_gripper_norm.unsqueeze(-1) + rel_right_gripper, 0.0, 1.0
    )
    abs_left_gripper = denormalize_gripper(abs_left_gripper_norm, gripper_min, gripper_max)
    abs_right_gripper = denormalize_gripper(abs_right_gripper_norm, gripper_min, gripper_max)

    return torch.cat([
        abs_left_xyz,
        torch.nn.functional.normalize(abs_left_quat, dim=-1),
        abs_left_gripper.unsqueeze(-1),
        abs_right_xyz,
        torch.nn.functional.normalize(abs_right_quat, dim=-1),
        abs_right_gripper.unsqueeze(-1),
    ], dim=-1)


def relative_eef_wxyz_16d_to_abs_eef_16d_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Recover absolute 16D wxyz EEF actions from 16D relative EEF wxyz actions."""
    state_xyzw = eef_quat_wxyz_to_xyzw(state)
    rel_xyzw = eef_quat_wxyz_to_xyzw(relative_actions)

    state_left_xyz = state_xyzw[..., :3]
    state_left_quat = torch.nn.functional.normalize(state_xyzw[..., 3:7], dim=-1)
    state_left_gripper = state_xyzw[..., 7]
    state_right_xyz = state_xyzw[..., 8:11]
    state_right_quat = torch.nn.functional.normalize(state_xyzw[..., 11:15], dim=-1)
    state_right_gripper = state_xyzw[..., 15]

    rel_left_xyz = rel_xyzw[..., :3]
    rel_left_quat = torch.nn.functional.normalize(rel_xyzw[..., 3:7], dim=-1)
    rel_left_gripper = rel_xyzw[..., 7]
    rel_right_xyz = rel_xyzw[..., 8:11]
    rel_right_quat = torch.nn.functional.normalize(rel_xyzw[..., 11:15], dim=-1)
    rel_right_gripper = rel_xyzw[..., 15]

    state_left_rot = _quat_xyzw_to_rotmat_torch(state_left_quat)
    state_right_rot = _quat_xyzw_to_rotmat_torch(state_right_quat)
    abs_left_xyz = state_left_xyz.unsqueeze(-2) + torch.matmul(
        state_left_rot.unsqueeze(-3), rel_left_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_right_xyz = state_right_xyz.unsqueeze(-2) + torch.matmul(
        state_right_rot.unsqueeze(-3), rel_right_xyz.unsqueeze(-1)
    ).squeeze(-1)

    abs_left_quat = _quat_multiply_xyzw_torch(state_left_quat.unsqueeze(-2), rel_left_quat)
    abs_right_quat = _quat_multiply_xyzw_torch(state_right_quat.unsqueeze(-2), rel_right_quat)

    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    abs_left_gripper_norm = torch.clamp(
        state_left_gripper_norm.unsqueeze(-1) + rel_left_gripper, 0.0, 1.0
    )
    abs_right_gripper_norm = torch.clamp(
        state_right_gripper_norm.unsqueeze(-1) + rel_right_gripper, 0.0, 1.0
    )
    abs_left_gripper = denormalize_gripper(abs_left_gripper_norm, gripper_min, gripper_max)
    abs_right_gripper = denormalize_gripper(abs_right_gripper_norm, gripper_min, gripper_max)

    abs_xyzw = torch.cat([
        abs_left_xyz,
        torch.nn.functional.normalize(abs_left_quat, dim=-1),
        abs_left_gripper.unsqueeze(-1),
        abs_right_xyz,
        torch.nn.functional.normalize(abs_right_quat, dim=-1),
        abs_right_gripper.unsqueeze(-1),
    ], dim=-1)
    return eef_quat_xyzw_to_wxyz(abs_xyzw)


def _qpos_wrist_slices(hand_dim: int):
    """Slices for ``[L_qpos | L_wrist9 | R_qpos | R_wrist9]`` at a given hand DoF."""
    h = int(hand_dim)
    right = h + QPOS_WRIST_DIM
    return (
        slice(0, h),                                 # left qpos
        slice(h, h + 3),                             # left xyz
        slice(h + 3, h + QPOS_WRIST_DIM),            # left rot6d
        slice(right, right + h),                     # right qpos
        slice(right + h, right + h + 3),             # right xyz
        slice(right + h + 3, right + h + QPOS_WRIST_DIM),  # right rot6d
    )


def _infer_qpos_hand_dim(vector_dim: int) -> int:
    """Recover the per-hand DoF from a full ``2*(hand+9)`` vector width."""
    if vector_dim % 2 != 0 or vector_dim // 2 <= QPOS_WRIST_DIM:
        raise ValueError(f"Not a qpos+wrist9 vector width: {vector_dim}")
    return vector_dim // 2 - QPOS_WRIST_DIM


def abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    hand_dim: Optional[int] = None,
) -> torch.Tensor:
    """Convert absolute Tianji wrist actions to chunk-relative wrist actions.

    Layout:
      [L_qpos{N} | L_wrist_xyz3_rot6d6 | R_qpos{N} | R_wrist_xyz3_rot6d6]

    ``hand_dim`` (N) defaults to whatever the trailing vector width implies:
    20 for Wuji (58D), 22 for Sharpa (62D).

    Qpos dims are already normalized by dataset stats and are copied unchanged.
    Wrist xyz is in meters. Wrist rotation uses the row-major rot6d convention
    used by the existing EEF helpers in this file.
    """
    if hand_dim is None:
        hand_dim = _infer_qpos_hand_dim(int(state.shape[-1]))
    lq, lxyz, lrot, rq, rxyz, rrot = _qpos_wrist_slices(hand_dim)

    state_left_xyz = state[..., lxyz]
    state_left_rot = _rot6d_to_mat_torch(state[..., lrot])
    state_right_xyz = state[..., rxyz]
    state_right_rot = _rot6d_to_mat_torch(state[..., rrot])

    action_left_qpos = action_chunk[..., lq]
    action_left_xyz = action_chunk[..., lxyz]
    action_left_rot = _rot6d_to_mat_torch(action_chunk[..., lrot])
    action_right_qpos = action_chunk[..., rq]
    action_right_xyz = action_chunk[..., rxyz]
    action_right_rot = _rot6d_to_mat_torch(action_chunk[..., rrot])

    state_left_rot_t = state_left_rot.transpose(-1, -2).unsqueeze(-3)
    state_right_rot_t = state_right_rot.transpose(-1, -2).unsqueeze(-3)
    rel_left_xyz = torch.matmul(
        state_left_rot_t,
        (action_left_xyz - state_left_xyz.unsqueeze(-2)).unsqueeze(-1),
    ).squeeze(-1)
    rel_right_xyz = torch.matmul(
        state_right_rot_t,
        (action_right_xyz - state_right_xyz.unsqueeze(-2)).unsqueeze(-1),
    ).squeeze(-1)
    rel_left_rot = torch.matmul(state_left_rot_t, action_left_rot)
    rel_right_rot = torch.matmul(state_right_rot_t, action_right_rot)

    return torch.cat([
        action_left_qpos,
        rel_left_xyz,
        _mat_to_rot6d_torch(rel_left_rot),
        action_right_qpos,
        rel_right_xyz,
        _mat_to_rot6d_torch(rel_right_rot),
    ], dim=-1)


def abs_qpos20_wrist9_to_relqpos20_rel6d_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    hand_dim: Optional[int] = None,
) -> torch.Tensor:
    """Convert Tianji actions to hand-delta qpos + relative wrist.

    Qpos deltas are computed in the model-space qpos representation. For the
    Tianji loaders that means after qpos normalization, matching the absolute
    qpos values used by ``qpos{N}_rel6d``.
    """
    if hand_dim is None:
        hand_dim = _infer_qpos_hand_dim(int(state.shape[-1]))
    lq, _, _, rq, _, _ = _qpos_wrist_slices(hand_dim)
    rel = abs_qpos20_wrist9_to_rel_qpos20_wrist9_torch(state, action_chunk, hand_dim=hand_dim)
    out = rel.clone()
    out[..., lq] = action_chunk[..., lq] - state[..., lq].unsqueeze(-2)
    out[..., rq] = action_chunk[..., rq] - state[..., rq].unsqueeze(-2)
    return out


def rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    hand_dim: Optional[int] = None,
) -> torch.Tensor:
    """Recover absolute Tianji wrist actions from relative wrist actions."""
    if hand_dim is None:
        hand_dim = _infer_qpos_hand_dim(int(state.shape[-1]))
    lq, lxyz, lrot, rq, rxyz, rrot = _qpos_wrist_slices(hand_dim)

    state_left_xyz = state[..., lxyz]
    state_left_rot = _rot6d_to_mat_torch(state[..., lrot])
    state_right_xyz = state[..., rxyz]
    state_right_rot = _rot6d_to_mat_torch(state[..., rrot])

    rel_left_qpos = relative_actions[..., lq]
    rel_left_xyz = relative_actions[..., lxyz]
    rel_left_rot = _rot6d_to_mat_torch(relative_actions[..., lrot])
    rel_right_qpos = relative_actions[..., rq]
    rel_right_xyz = relative_actions[..., rxyz]
    rel_right_rot = _rot6d_to_mat_torch(relative_actions[..., rrot])

    abs_left_xyz = state_left_xyz.unsqueeze(-2) + torch.matmul(
        state_left_rot.unsqueeze(-3), rel_left_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_right_xyz = state_right_xyz.unsqueeze(-2) + torch.matmul(
        state_right_rot.unsqueeze(-3), rel_right_xyz.unsqueeze(-1)
    ).squeeze(-1)
    abs_left_rot = torch.matmul(state_left_rot.unsqueeze(-3), rel_left_rot)
    abs_right_rot = torch.matmul(state_right_rot.unsqueeze(-3), rel_right_rot)

    return torch.cat([
        rel_left_qpos,
        abs_left_xyz,
        _mat_to_rot6d_torch(abs_left_rot),
        rel_right_qpos,
        abs_right_xyz,
        _mat_to_rot6d_torch(abs_right_rot),
    ], dim=-1)


def relqpos20_rel6d_to_abs_qpos20_wrist9_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    hand_dim: Optional[int] = None,
) -> torch.Tensor:
    """Recover absolute actions from hand-delta qpos + relative wrist."""
    if hand_dim is None:
        hand_dim = _infer_qpos_hand_dim(int(state.shape[-1]))
    lq, _, _, rq, _, _ = _qpos_wrist_slices(hand_dim)
    abs_actions = rel_qpos20_wrist9_to_abs_qpos20_wrist9_torch(
        state, relative_actions, hand_dim=hand_dim
    )
    out = abs_actions.clone()
    out[..., lq] = state[..., lq].unsqueeze(-2) + relative_actions[..., lq]
    out[..., rq] = state[..., rq].unsqueeze(-2) + relative_actions[..., rq]
    return out


def abs_eef_16d_to_abs_eef_20d_torch(abs_eef: torch.Tensor) -> torch.Tensor:
    """Convert absolute EEF from 16D quaternion format to 20D rot6d format.

    Input layout:
      [left_xyz(3), left_quat_xyzw(4), left_gripper(1),
       right_xyz(3), right_quat_xyzw(4), right_gripper(1)]

    Output layout:
      [left_xyz(3), left_rot6d(6), left_gripper(1),
       right_xyz(3), right_rot6d(6), right_gripper(1)]
    """
    left_xyz = abs_eef[..., :3]
    left_quat = abs_eef[..., 3:7]
    left_grip = abs_eef[..., 7:8]
    right_xyz = abs_eef[..., 8:11]
    right_quat = abs_eef[..., 11:15]
    right_grip = abs_eef[..., 15:16]

    left_rot6d = _mat_to_rot6d_torch(_quat_xyzw_to_rotmat_torch(left_quat))
    right_rot6d = _mat_to_rot6d_torch(_quat_xyzw_to_rotmat_torch(right_quat))

    return torch.cat([
        left_xyz,
        left_rot6d,
        left_grip,
        right_xyz,
        right_rot6d,
        right_grip,
    ], dim=-1)


def abs_eef_20d_to_abs_eef_16d_torch(abs_eef: torch.Tensor) -> torch.Tensor:
    """Convert absolute EEF from 20D rot6d format to 16D quaternion format."""
    left_xyz = abs_eef[..., :3]
    left_rot6d = abs_eef[..., 3:9]
    left_grip = abs_eef[..., 9:10]
    right_xyz = abs_eef[..., 10:13]
    right_rot6d = abs_eef[..., 13:19]
    right_grip = abs_eef[..., 19:20]

    left_quat = _rotmat_to_quat_xyzw_torch(_rot6d_to_mat_torch(left_rot6d))
    right_quat = _rotmat_to_quat_xyzw_torch(_rot6d_to_mat_torch(right_rot6d))

    return torch.cat([
        left_xyz,
        left_quat,
        left_grip,
        right_xyz,
        right_quat,
        right_grip,
    ], dim=-1)


def abs_eef_20d_to_relative_eef_20d_torch(
    state: torch.Tensor,
    action_chunk: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Convert absolute 20D rot6d actions to 20D relative EEF using a 20D state anchor."""
    state_16d = abs_eef_20d_to_abs_eef_16d_torch(state)
    action_chunk_16d = abs_eef_20d_to_abs_eef_16d_torch(action_chunk)
    return abs_eef_16d_to_relative_eef_20d_torch(
        state_16d, action_chunk_16d, gripper_min, gripper_max
    )


def relative_eef_20d_to_abs_eef_20d_torch(
    state: torch.Tensor,
    relative_actions: torch.Tensor,
    gripper_min: float,
    gripper_max: float,
) -> torch.Tensor:
    """Recover absolute 20D rot6d EEF actions from 20D relative EEF actions."""
    state_16d = abs_eef_20d_to_abs_eef_16d_torch(state)
    abs_16d = relative_eef_20d_to_abs_eef_16d_torch(
        state_16d, relative_actions, gripper_min, gripper_max
    )
    return abs_eef_16d_to_abs_eef_20d_torch(abs_16d)


def abs_eef_14d_to_relative_eef_20d(state, action_chunk, gripper_min, gripper_max):
    """
    Convert 14D absolute end-effector actions to 20D relative end-effector representation

    Args:
        state: (14,) - current state [left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
        action_chunk: (T, 14) - action sequence
        gripper_min: float - minimum gripper value for normalization
        gripper_max: float - maximum gripper value for normalization

    Returns:
        relative_actions: (T, 20) - actions relative to state
            [left_xyz(3), left_rot6d(6), left_gripper(1),
             right_xyz(3), right_rot6d(6), right_gripper(1)]
    """
    # 1. Separate left and right arms
    # state
    state_left_pose = state[:6]  # xyz + rpy (radians)
    state_left_gripper = state[6]
    state_right_pose = state[7:13]  # xyz + rpy (radians)
    state_right_gripper = state[13]

    # action chunk
    T = action_chunk.shape[0]
    action_left_poses = action_chunk[:, :6]  # (T, 6) - xyz + rpy
    action_left_grippers = action_chunk[:, 6]  # (T,)
    action_right_poses = action_chunk[:, 7:13]  # (T, 6) - xyz + rpy
    action_right_grippers = action_chunk[:, 13]  # (T,)

    # 2. Convert to transformation matrices
    state_left_mat = pose_to_mat(state_left_pose)  # (4, 4)
    state_right_mat = pose_to_mat(state_right_pose)  # (4, 4)

    action_left_mats = pose_to_mat(action_left_poses)  # (T, 4, 4)
    action_right_mats = pose_to_mat(action_right_poses)  # (T, 4, 4)

    # 3. Compute relative poses (each action relative to state)
    # Use 'relative' mode: relative_mat = inv(base_mat) @ pose_mat
    relative_left_mats = convert_pose_mat_rep(
        action_left_mats,
        state_left_mat,
        pose_rep='relative',
        backward=False
    )  # (T, 4, 4)

    relative_right_mats = convert_pose_mat_rep(
        action_right_mats,
        state_right_mat,
        pose_rep='relative',
        backward=False
    )  # (T, 4, 4)

    # 4. Convert to 10D representation (xyz + 6d rotation)
    relative_left_10d = mat_to_pose10d(relative_left_mats)  # (T, 9)
    relative_right_10d = mat_to_pose10d(relative_right_mats)  # (T, 9)

    # 5. Compute relative gripper (normalized difference)
    # Normalize grippers to [0, 1] first
    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    action_left_grippers_norm = normalize_gripper(action_left_grippers, gripper_min, gripper_max)
    relative_left_grippers = action_left_grippers_norm - state_left_gripper_norm  # (T,) range: [-1, 1]

    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    action_right_grippers_norm = normalize_gripper(action_right_grippers, gripper_min, gripper_max)
    relative_right_grippers = action_right_grippers_norm - state_right_gripper_norm  # (T,) range: [-1, 1]

    # 6. Combine into 20D (all relative)
    relative_actions = np.concatenate([
        relative_left_10d,  # (T, 9) - relative xyz + 6d rotation
        relative_left_grippers[:, None],  # (T, 1) - relative gripper
        relative_right_10d,  # (T, 9) - relative xyz + 6d rotation
        relative_right_grippers[:, None]  # (T, 1) - relative gripper
    ], axis=-1)  # (T, 20)

    return relative_actions


def relative_eef_20d_to_abs_eef_14d(state, relative_actions, gripper_min, gripper_max):
    """
    Convert 20D relative end-effector representation back to 14D absolute actions

    Args:
        state: (14,) - current state [left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
        relative_actions: (T, 20) - relative actions
            [left_xyz(3), left_rot6d(6), left_gripper(1),
             right_xyz(3), right_rot6d(6), right_gripper(1)]
        gripper_min: float - minimum gripper value for denormalization
        gripper_max: float - maximum gripper value for denormalization

    Returns:
        action_chunk: (T, 14) - absolute action sequence
    """
    # 1. Separate left and right arms
    state_left_pose = state[:6]  # xyz + rpy (radians)
    state_left_gripper = state[6]
    state_right_pose = state[7:13]  # xyz + rpy (radians)
    state_right_gripper = state[13]

    # 2. Separate relative actions
    T = relative_actions.shape[0]
    relative_left_10d = relative_actions[:, :9]  # (T, 9)
    relative_left_grippers = relative_actions[:, 9]  # (T,)
    relative_right_10d = relative_actions[:, 10:19]  # (T, 9)
    relative_right_grippers = relative_actions[:, 19]  # (T,)

    # 3. Convert state to transformation matrices
    state_left_mat = pose_to_mat(state_left_pose)
    state_right_mat = pose_to_mat(state_right_pose)

    relative_left_mats = pose10d_to_mat(relative_left_10d)
    relative_right_mats = pose10d_to_mat(relative_right_10d)

    # 4. Inverse transform using 'relative' mode: abs_mat = base_mat @ relative_mat
    abs_left_mats = convert_pose_mat_rep(
        relative_left_mats,
        state_left_mat,
        pose_rep='relative',
        backward=True
    )

    abs_right_mats = convert_pose_mat_rep(
        relative_right_mats,
        state_right_mat,
        pose_rep='relative',
        backward=True
    )

    # 5. Convert back to 6D pose (xyz + rpy)
    abs_left_poses = mat_to_pose(abs_left_mats)  # (T, 6)
    abs_right_poses = mat_to_pose(abs_right_mats)  # (T, 6)

    # 6. Recover absolute gripper from relative
    # Normalize state grippers
    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)

    # Add relative to state (in normalized space)
    action_left_grippers_norm = state_left_gripper_norm + relative_left_grippers  # (T,)
    action_right_grippers_norm = state_right_gripper_norm + relative_right_grippers  # (T,)

    # Denormalize back to original range
    action_left_grippers = denormalize_gripper(action_left_grippers_norm, gripper_min, gripper_max)
    action_right_grippers = denormalize_gripper(action_right_grippers_norm, gripper_min, gripper_max)

    # 7. Combine into 14D
    action_chunk = np.zeros((T, 14), dtype=abs_left_poses.dtype)
    action_chunk[:, :6] = abs_left_poses  # left xyz + rpy
    action_chunk[:, 6] = action_left_grippers  # left gripper
    action_chunk[:, 7:13] = abs_right_poses  # right xyz + rpy
    action_chunk[:, 13] = action_right_grippers  # right gripper

    return action_chunk


def abs_eef_16d_to_relative_eef_20d(state, action_chunk, gripper_min, gripper_max):
    """
    Convert 16D absolute end-effector actions (quaternion format) to 20D relative end-effector representation

    Args:
        state: (16,) - current state [left_xyz(3), left_quaternion(4), left_gripper(1),
                                  right_xyz(3), right_quaternion(4), right_gripper(1)]
        action_chunk: (T, 16) - action sequence, same format as state
        gripper_min: float - minimum gripper value for normalization
        gripper_max: float - maximum gripper value for normalization

    Returns:
        relative_actions: (T, 20) - actions relative to state
            [left_xyz(3), left_rot6d(6), left_gripper(1),
             right_xyz(3), right_rot6d(6), right_gripper(1)]
    """
    # 1. Separate left and right arms
    # state: [left_xyz(3), left_quat(4), left_gripper(1), right_xyz(3), right_quat(4), right_gripper(1)]
    state_left_xyz = state[:3]
    state_left_quat = state[3:7]  # quaternion (x, y, z, w) format
    state_left_gripper = state[7]
    state_right_xyz = state[8:11]
    state_right_quat = state[11:15]  # quaternion (x, y, z, w) format
    state_right_gripper = state[15]

    # action chunk: (T, 16)
    T = action_chunk.shape[0]
    action_left_xyz = action_chunk[:, :3]  # (T, 3)
    action_left_quat = action_chunk[:, 3:7]  # (T, 4) - quaternion
    action_left_grippers = action_chunk[:, 7]  # (T,)
    action_right_xyz = action_chunk[:, 8:11]  # (T, 3)
    action_right_quat = action_chunk[:, 11:15]  # (T, 4) - quaternion
    action_right_grippers = action_chunk[:, 15]  # (T,)

    # 2. Convert to 7D pose (xyz + quaternion) and then to transformation matrices
    # pose7d format: [xyz(3), quat(4)] where quat is (x, y, z, w)
    state_left_7d = np.concatenate([state_left_xyz, state_left_quat])  # (7,)
    state_right_7d = np.concatenate([state_right_xyz, state_right_quat])  # (7,)

    action_left_7d = np.concatenate([
        action_left_xyz,  # (T, 3)
        action_left_quat  # (T, 4)
    ], axis=-1)  # (T, 7)
    action_right_7d = np.concatenate([
        action_right_xyz,  # (T, 3)
        action_right_quat  # (T, 4)
    ], axis=-1)  # (T, 7)

    # Convert to transformation matrices using pose7d_to_mat
    state_left_mat = pose7d_to_mat(state_left_7d)  # (4, 4)
    state_right_mat = pose7d_to_mat(state_right_7d)  # (4, 4)

    action_left_mats = pose7d_to_mat(action_left_7d)  # (T, 4, 4)
    action_right_mats = pose7d_to_mat(action_right_7d)  # (T, 4, 4)

    # 3. Compute relative poses (each action relative to state)
    # Use 'relative' mode: relative_mat = inv(base_mat) @ pose_mat
    relative_left_mats = convert_pose_mat_rep(
        action_left_mats,
        state_left_mat,
        pose_rep='relative',
        backward=False
    )  # (T, 4, 4)

    relative_right_mats = convert_pose_mat_rep(
        action_right_mats,
        state_right_mat,
        pose_rep='relative',
        backward=False
    )  # (T, 4, 4)

    # 4. Convert to 10D representation (xyz + 6d rotation)
    relative_left_10d = mat_to_pose10d(relative_left_mats)  # (T, 9)
    relative_right_10d = mat_to_pose10d(relative_right_mats)  # (T, 9)

    # 5. Compute relative gripper (normalized difference)
    # Normalize grippers to [0, 1] first
    state_left_gripper_norm = normalize_gripper(state_left_gripper, gripper_min, gripper_max)
    action_left_grippers_norm = normalize_gripper(action_left_grippers, gripper_min, gripper_max)
    relative_left_grippers = action_left_grippers_norm - state_left_gripper_norm  # (T,) range: [-1, 1]

    state_right_gripper_norm = normalize_gripper(state_right_gripper, gripper_min, gripper_max)
    action_right_grippers_norm = normalize_gripper(action_right_grippers, gripper_min, gripper_max)
    relative_right_grippers = action_right_grippers_norm - state_right_gripper_norm  # (T,) range: [-1, 1]

    # 6. Combine into 20D (all relative)
    relative_actions = np.concatenate([
        relative_left_10d,  # (T, 9) - relative xyz + 6d rotation
        relative_left_grippers[:, None],  # (T, 1) - relative gripper
        relative_right_10d,  # (T, 9) - relative xyz + 6d rotation
        relative_right_grippers[:, None]  # (T, 1) - relative gripper
    ], axis=-1)  # (T, 20)

    return relative_actions


def relative_eef_20d_to_abs_eef_16d(state, relative_actions, gripper_min, gripper_max):
    """NumPy wrapper around the torch implementation."""
    state_t = torch.as_tensor(state, dtype=torch.float32)
    rel_t = torch.as_tensor(relative_actions, dtype=torch.float32)
    out = relative_eef_20d_to_abs_eef_16d_torch(
        state_t,
        rel_t,
        gripper_min,
        gripper_max,
    )
    return out.cpu().numpy()
