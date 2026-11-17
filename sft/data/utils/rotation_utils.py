# Pure-torch rotation utilities for EEF action interpolation.
#
# Quaternion convention: [x, y, z, w] (scalar-last).

import torch
import torch.nn.functional as F  # noqa: N812

# ======================================================================
# Quaternion ↔ Rotation matrix
# ======================================================================

def rpy_to_quaternion(rpy: torch.Tensor) -> torch.Tensor:
    """Roll-pitch-yaw [roll, pitch, yaw] -> quaternion [x, y, z, w].

    Uses the common convention:
      R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    roll, pitch, yaw = rpy.unbind(-1)

    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw

    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return F.normalize(torch.stack([x, y, z, w], dim=-1), dim=-1)


def eef_rpy_to_quat(eef: torch.Tensor) -> torch.Tensor:
    """Convert 14D dual-arm EEF [xyz, rpy, grip]x2 -> 16D [xyz, quat, grip]x2."""
    if eef.shape[-1] != 14:
        raise ValueError(f"Expected 14D RPY EEF input, got {eef.shape[-1]}")

    out = torch.zeros(*eef.shape[:-1], 16, dtype=eef.dtype, device=eef.device)
    for arm_idx in range(2):
        in_off = arm_idx * 7
        out_off = arm_idx * 8
        out[..., out_off:out_off + 3] = eef[..., in_off:in_off + 3]
        out[..., out_off + 3:out_off + 7] = rpy_to_quaternion(
            eef[..., in_off + 3:in_off + 6]
        )
        out[..., out_off + 7:out_off + 8] = eef[..., in_off + 6:in_off + 7]
    return out


def quaternion_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [x, y, z, w] -> roll-pitch-yaw [roll, pitch, yaw]."""
    q = F.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(min=-1.0, max=1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def eef_quat_to_rpy(eef: torch.Tensor) -> torch.Tensor:
    """Convert 16D dual-arm EEF [xyz, quat, grip]x2 -> 14D [xyz, rpy, grip]x2."""
    if eef.shape[-1] != 16:
        raise ValueError(f"Expected 16D quaternion EEF input, got {eef.shape[-1]}")

    out = torch.zeros(*eef.shape[:-1], 14, dtype=eef.dtype, device=eef.device)
    for arm_idx in range(2):
        in_off = arm_idx * 8
        out_off = arm_idx * 7
        out[..., out_off:out_off + 3] = eef[..., in_off:in_off + 3]
        out[..., out_off + 3:out_off + 6] = quaternion_to_rpy(
            eef[..., in_off + 3:in_off + 7]
        )
        out[..., out_off + 6:out_off + 7] = eef[..., in_off + 7:in_off + 8]
    return out


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [x, y, z, w] → 3×3 rotation matrix.  Input: [..., 4]."""
    q = F.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)

    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → quaternion [x, y, z, w].  Input: [..., 3, 3].

    Uses Shepperd's method (pick the largest diagonal element to avoid
    division-by-zero).
    """
    batch_shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)

    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    m01, m02, m10 = m[:, 0, 1], m[:, 0, 2], m[:, 1, 0]
    m12, m20, m21 = m[:, 1, 2], m[:, 2, 0], m[:, 2, 1]

    # 4 * q_i^2  for [w, x, y, z]
    diags = torch.stack([
        1 + m00 + m11 + m22,   # 4w²
        1 + m00 - m11 - m22,   # 4x²
        1 - m00 + m11 - m22,   # 4y²
        1 - m00 - m11 + m22,   # 4z²
    ], dim=-1)                 # [N, 4]

    best = diags.argmax(dim=-1)  # [N]

    # Compute in [w, x, y, z] order internally
    q = torch.zeros_like(diags)

    for case in range(4):
        mask = best == case
        if not mask.any():
            continue
        s = torch.sqrt(torch.clamp(diags[mask, case], min=1e-10))  # 2|q_i|
        inv_s = 0.5 / s

        if case == 0:    # w largest
            q[mask, 0] = s * 0.5
            q[mask, 1] = (m21[mask] - m12[mask]) * inv_s
            q[mask, 2] = (m02[mask] - m20[mask]) * inv_s
            q[mask, 3] = (m10[mask] - m01[mask]) * inv_s
        elif case == 1:  # x largest
            q[mask, 0] = (m21[mask] - m12[mask]) * inv_s
            q[mask, 1] = s * 0.5
            q[mask, 2] = (m01[mask] + m10[mask]) * inv_s
            q[mask, 3] = (m02[mask] + m20[mask]) * inv_s
        elif case == 2:  # y largest
            q[mask, 0] = (m02[mask] - m20[mask]) * inv_s
            q[mask, 1] = (m01[mask] + m10[mask]) * inv_s
            q[mask, 2] = s * 0.5
            q[mask, 3] = (m12[mask] + m21[mask]) * inv_s
        else:            # z largest
            q[mask, 0] = (m10[mask] - m01[mask]) * inv_s
            q[mask, 1] = (m02[mask] + m20[mask]) * inv_s
            q[mask, 2] = (m12[mask] + m21[mask]) * inv_s
            q[mask, 3] = s * 0.5

    # [w, x, y, z] → [x, y, z, w]
    q = torch.cat([q[:, 1:4], q[:, 0:1]], dim=-1)
    return F.normalize(q, dim=-1).reshape(*batch_shape, 4)


# ======================================================================
# 6D rotation (Zhou et al. 2019) ↔ Rotation matrix
# ======================================================================

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """6D rotation → 3×3 matrix (Gram-Schmidt).  Input: [..., 6].

    6D = [col0(3), col1(3)] of the rotation matrix.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]  columns


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix → 6D rotation [col0, col1].  Input: [..., 3, 3]."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)  # [..., 6]


# ======================================================================
# Quaternion Slerp
# ======================================================================

def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor,
                     t: float) -> torch.Tensor:
    """Spherical linear interpolation.  q0, q1: [..., 4], t: scalar [0, 1].

    Quaternion convention: [x, y, z, w].

    Handles:
    - Shortest-path: flips q1 when dot < 0.
    - Small-angle fallback: uses normalised lerp when sin(θ) ≈ 0.
    """
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # [..., 1]

    # Ensure shortest path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Small-angle fallback to nlerp
    small = sin_theta.abs() < 1e-6
    s0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    s1 = torch.where(small, t,       torch.sin(t * theta) / sin_theta)

    return F.normalize(s0 * q0 + s1 * q1, dim=-1)


# ======================================================================
# High-level: upsample EEF trajectory
# ======================================================================

def upsample_eef(eef: torch.Tensor, ratio: int) -> torch.Tensor:
    """Upsample end-effector trajectory with Slerp for rotations.

    Args:
        eef: [T, D] where D = 16 (quat) or 20 (rot6d).
             Per-arm layout:
               16D: [xyz(3), quat_xyzw(4), grip(1)] × 2 arms
               20D: [xyz(3), rot6d(6), grip(1)] × 2 arms
        ratio: upsample factor (1 = no interpolation, 4 = insert 3 frames).

    Returns:
        [new_T, 20] — always in rot6d format.
        new_T = (T-1)*ratio + 1  when ratio > 1, else T.
    """
    T, D = eef.shape
    if D == 16:
        arm_size, rot_size, rot_is_quat = 8, 4, True
    elif D == 20:
        arm_size, rot_size, rot_is_quat = 10, 6, False
    else:
        raise ValueError(f"Expected 16 or 20 dims for EEF, got {D}")

    new_T = (T - 1) * ratio + 1 if ratio > 1 else T
    result = torch.zeros(new_T, 20, dtype=eef.dtype)

    for arm_idx in range(2):
        in_off  = arm_idx * arm_size
        out_off = arm_idx * 10

        xyz     = eef[:, in_off:in_off + 3]                          # [T, 3]
        grip    = eef[:, in_off + arm_size - 1:in_off + arm_size]    # [T, 1]
        rot_raw = eef[:, in_off + 3:in_off + 3 + rot_size]          # [T, 4|6]

        # Get rot6d for each frame (avoid redundant conversion if already rot6d)
        if rot_is_quat:
            quats = F.normalize(rot_raw, dim=-1)       # [T, 4]
            rot6d_orig = matrix_to_rot6d(quaternion_to_matrix(quats))  # [T, 6]
        else:
            rot6d_orig = rot_raw                       # [T, 6] — use directly
            quats = matrix_to_quaternion(rot6d_to_matrix(rot_raw))  # for Slerp

        if ratio <= 1:
            # No interpolation — just format-convert to 20D
            result[:, out_off:out_off + 3]     = xyz
            result[:, out_off + 3:out_off + 9] = rot6d_orig
            result[:, out_off + 9:out_off + 10] = grip
            continue

        # --- xyz & grip: linear interpolation (reuse F.interpolate) ---
        xyz_up = F.interpolate(
            xyz.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True,
        ).squeeze(0).T                                               # [new_T, 3]

        grip_up = F.interpolate(
            grip.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True,
        ).squeeze(0).T                                               # [new_T, 1]

        # --- rotation: batched Slerp ---
        q0 = quats[:-1]    # [T-1, 4]
        q1 = quats[1:]     # [T-1, 4]

        rot6d_segments = [rot6d_orig[:-1]]  # j=0: use original frames directly
        for j in range(1, ratio):
            t = j / ratio
            q_t = quaternion_slerp(q0, q1, t)                       # [T-1, 4]
            r6d_t = matrix_to_rot6d(quaternion_to_matrix(q_t))       # [T-1, 6]
            rot6d_segments.append(r6d_t)

        # Interleave: [T-1, ratio, 6] → [(T-1)*ratio, 6]
        rot6d_body = torch.stack(rot6d_segments, dim=1).reshape(-1, 6)

        # Last frame
        rot6d_all = torch.cat([rot6d_body, rot6d_orig[-1:]], dim=0)  # [new_T, 6]

        result[:, out_off:out_off + 3]      = xyz_up
        result[:, out_off + 3:out_off + 9]  = rot6d_all
        result[:, out_off + 9:out_off + 10] = grip_up

    return result


def upsample_abs_eef_16d(eef: torch.Tensor, ratio: int) -> torch.Tensor:
    """Upsample absolute 16D EEF trajectory while keeping quaternion output.

    Args:
        eef: [T, 16], per arm layout [xyz(3), quat_xyzw(4), grip(1)] x 2 arms.
        ratio: upsample factor (1 = no interpolation).

    Returns:
        [new_T, 16] in the same absolute quaternion layout.
    """
    T, D = eef.shape
    if D != 16:
        raise ValueError(f"Expected 16D absolute EEF input, got {D}")
    if ratio <= 1 or T <= 1:
        return eef

    new_T = (T - 1) * ratio + 1
    result = torch.zeros(new_T, 16, dtype=eef.dtype, device=eef.device)

    for arm_idx in range(2):
        off = arm_idx * 8
        xyz = eef[:, off:off + 3]
        quat = F.normalize(eef[:, off + 3:off + 7], dim=-1)
        grip = eef[:, off + 7:off + 8]

        xyz_up = F.interpolate(
            xyz.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True,
        ).squeeze(0).T
        grip_up = F.interpolate(
            grip.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True,
        ).squeeze(0).T

        q0 = quat[:-1]
        q1 = quat[1:]
        quat_segments = [q0]
        for j in range(1, ratio):
            quat_segments.append(quaternion_slerp(q0, q1, j / ratio))
        quat_body = torch.stack(quat_segments, dim=1).reshape(-1, 4)
        quat_all = torch.cat([quat_body, quat[-1:]], dim=0)

        result[:, off:off + 3] = xyz_up
        result[:, off + 3:off + 7] = quat_all
        result[:, off + 7:off + 8] = grip_up

    return result
