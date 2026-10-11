"""Pure-numpy Franka Panda forward kinematics to panda_hand frame.

Parameters hardcoded from:
  the Franka Panda URDF used in SkillWeaver's Isaac Lab simulation (franka_gripper.urdf)

URDF convention for each joint:
  T(q) = Translate(xyz) @ Rpy(rpy) @ Rz(q)
All Panda revolute joints rotate about their local Z axis.
The hand joint is fixed: T_hand = Translate(xyz) @ Rpy(rpy).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint parameters from URDF <origin> tags
# ---------------------------------------------------------------------------

_JOINT_XYZ = np.array(
    [
        [0.0, 0.0, 0.333],      # joint 1
        [0.0, 0.0, 0.0],        # joint 2
        [0.0, -0.316, 0.0],     # joint 3
        [0.0825, 0.0, 0.0],     # joint 4
        [-0.0825, 0.384, 0.0],  # joint 5
        [0.0, 0.0, 0.0],        # joint 6
        [0.088, 0.0, 0.0],      # joint 7
    ],
    dtype=np.float64,
)

_JOINT_RPY = np.array(
    [
        [0.0, 0.0, 0.0],         # joint 1
        [-np.pi / 2, 0.0, 0.0],  # joint 2
        [np.pi / 2, 0.0, 0.0],   # joint 3
        [np.pi / 2, 0.0, 0.0],   # joint 4
        [-np.pi / 2, 0.0, 0.0],  # joint 5
        [np.pi / 2, 0.0, 0.0],   # joint 6
        [np.pi / 2, 0.0, 0.0],   # joint 7
    ],
    dtype=np.float64,
)

# panda_hand_joint: fixed, parent=panda_link7
_HAND_XYZ = np.array([0.0, 0.0, 0.107], dtype=np.float64)
_HAND_RPY = np.array([0.0, 0.0, -np.pi / 4], dtype=np.float64)

# Joint limits (rad) — used by callers for violation checking / clamping
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


# ---------------------------------------------------------------------------
# Helper: build constant homogeneous transforms
# ---------------------------------------------------------------------------


def _rpy_to_rot3(rpy: np.ndarray) -> np.ndarray:
    """rpy=[roll, pitch, yaw] → 3×3 rotation (URDF: Rz @ Ry @ Rx)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Precompute constant part of each joint transform: Translate(xyz) @ Rpy(rpy).
# The revolute part Rz(q) is applied at runtime.
_JOINT_CONST: np.ndarray = np.stack(
    [_make_4x4(_rpy_to_rot3(_JOINT_RPY[i]), _JOINT_XYZ[i]) for i in range(7)]
)  # (7, 4, 4)

_T_HAND: np.ndarray = _make_4x4(_rpy_to_rot3(_HAND_RPY), _HAND_XYZ)  # (4, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def panda_fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched Franka Panda FK to panda_hand frame.

    Args:
        q: (B, 7) joint angles in radians (panda_joint1..7).
           Values outside joint limits are accepted; callers should clamp
           beforehand if a valid EE pose is required.

    Returns:
        pos:     (B, 3) panda_hand position in panda_link0 frame (metres).
        rot_mat: (B, 3, 3) panda_hand rotation matrix in panda_link0 frame.
    """
    q = np.asarray(q, dtype=np.float64)
    B = q.shape[0]

    # Build batched Rz(q_j) for all joints: shape (B, 7, 4, 4)
    cq = np.cos(q)  # (B, 7)
    sq = np.sin(q)  # (B, 7)
    Rz_q = np.zeros((B, 7, 4, 4))
    Rz_q[:, :, 0, 0] = cq
    Rz_q[:, :, 0, 1] = -sq
    Rz_q[:, :, 1, 0] = sq
    Rz_q[:, :, 1, 1] = cq
    Rz_q[:, :, 2, 2] = 1.0
    Rz_q[:, :, 3, 3] = 1.0

    # T_joints[b, j] = _JOINT_CONST[j] @ Rz_q[b, j]
    # _JOINT_CONST shape (7,4,4) broadcast-matmul with (B,7,4,4) → (B,7,4,4)
    T_joints = np.matmul(_JOINT_CONST[None], Rz_q)  # (B, 7, 4, 4)

    # Accumulate: T = T_1 @ T_2 @ ... @ T_7 @ T_hand
    T: np.ndarray = np.eye(4)  # broadcast seed; becomes (B,4,4) after first iter
    for j in range(7):
        T = T @ T_joints[:, j]  # (B, 4, 4)

    T = T @ _T_HAND  # (B, 4, 4)

    pos = T[:, :3, 3]       # (B, 3)
    rot_mat = T[:, :3, :3]  # (B, 3, 3)
    return pos, rot_mat
