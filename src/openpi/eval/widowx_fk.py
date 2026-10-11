"""Pure-numpy WidowX 250s forward kinematics to gripper_link frame.

Parameters hardcoded from:
  the WidowX 250s URDF used in SkillWeaver's Isaac Lab simulation (wx250s.urdf)

URDF convention for each joint:
  T(q) = Translate(xyz) @ Rpy(rpy) @ R_axis(q)

Two structural differences from panda_fk, both load-bearing:

  1. The joints do NOT all rotate about local Z. wx250s uses Z for the waist,
     Y for shoulder/elbow/wrist_angle, and X for forearm_roll/wrist_rotate, so
     the revolute part is a Rodrigues rotation about an arbitrary axis rather
     than a hardcoded Rz.
  2. There is no fixed hand transform. The recorded end-effector frame is
     `gripper_link`, i.e. the child of the last revolute joint. The URDF does
     carry further fixed links (ee_arm_link +43mm, grip_site +93.6mm), but those
     are NOT what the trajectories store -- see the verification below.

Verified against recorded data: FK(state/arm_joint_pos) reproduces
state/ee/pos to 0.00 mm over sampled frames of carrot_on_plate and
spoon_on_towel. The other candidate links are off by 43.0 mm (ee_arm_link) and
93.6 mm (grip_site / ee_gripper_link), so the choice is unambiguous.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint parameters from URDF <origin> and <axis> tags, in chain order:
#   waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
# ---------------------------------------------------------------------------

_JOINT_XYZ = np.array(
    [
        [0.0, 0.0, 0.072],      # waist         base_link      -> shoulder_link
        [0.0, 0.0, 0.03865],    # shoulder      shoulder_link  -> upper_arm_link
        [0.04975, 0.0, 0.25],   # elbow         upper_arm_link -> upper_forearm_link
        [0.175, 0.0, 0.0],      # forearm_roll  upper_forearm  -> lower_forearm_link
        [0.075, 0.0, 0.0],      # wrist_angle   lower_forearm  -> wrist_link
        [0.065, 0.0, 0.0],      # wrist_rotate  wrist_link     -> gripper_link
    ],
    dtype=np.float64,
)

# Every <origin> in this chain has rpy="0 0 0"; kept explicit so a URDF revision
# that introduces a non-zero rpy has an obvious place to land.
_JOINT_RPY = np.zeros((6, 3), dtype=np.float64)

_JOINT_AXIS = np.array(
    [
        [0.0, 0.0, 1.0],  # waist
        [0.0, 1.0, 0.0],  # shoulder
        [0.0, 1.0, 0.0],  # elbow
        [1.0, 0.0, 0.0],  # forearm_roll
        [0.0, 1.0, 0.0],  # wrist_angle
        [1.0, 0.0, 0.0],  # wrist_rotate
    ],
    dtype=np.float64,
)

# Joint limits (rad) from the URDF <limit> tags — used by callers for violation
# checking / clamping. Same role as panda_fk.JOINT_LOWER/UPPER but 6-wide.
JOINT_LOWER = np.array(
    [-3.141592653589793, -1.8849555921538759, -2.1467549799530254,
     -3.141582653589793, -1.7453292519943295, -3.141582653589793]
)
JOINT_UPPER = np.array(
    [3.141592653589793, 1.9896753472735358, 1.6057029118347832,
     3.141582653589793, 2.1467549799530254, 3.141582653589793]
)

N_JOINTS: int = 6


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


# Constant part of each joint transform: Translate(xyz) @ Rpy(rpy).
# The revolute part R_axis(q) is applied at runtime.
_JOINT_CONST: np.ndarray = np.stack(
    [_make_4x4(_rpy_to_rot3(_JOINT_RPY[i]), _JOINT_XYZ[i]) for i in range(N_JOINTS)]
)  # (6, 4, 4)

# Skew-symmetric generator K for each joint axis, so the runtime rotation is the
# Rodrigues form  R = I + sin(q) K + (1 - cos(q)) K².
_AXIS_K: np.ndarray = np.stack(
    [
        np.array(
            [
                [0.0, -a[2], a[1]],
                [a[2], 0.0, -a[0]],
                [-a[1], a[0], 0.0],
            ]
        )
        for a in (_JOINT_AXIS / np.linalg.norm(_JOINT_AXIS, axis=1, keepdims=True))
    ]
)  # (6, 3, 3)
_AXIS_K2: np.ndarray = np.matmul(_AXIS_K, _AXIS_K)  # (6, 3, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def widowx_fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched WidowX 250s FK to gripper_link frame.

    Args:
        q: (B, 6) joint angles in radians, in URDF chain order
           (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate).
           Values outside joint limits are accepted; callers should clamp
           beforehand if a valid EE pose is required.

    Returns:
        pos:     (B, 3) gripper_link position in base_link frame (metres).
        rot_mat: (B, 3, 3) gripper_link rotation matrix in base_link frame.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != N_JOINTS:
        raise ValueError(f"widowx_fk expects (B, {N_JOINTS}) joint angles, got {q.shape}")
    B = q.shape[0]

    # Rodrigues per joint: (B, 6, 3, 3)
    sq = np.sin(q)[..., None, None]  # (B, 6, 1, 1)
    cq = np.cos(q)[..., None, None]  # (B, 6, 1, 1)
    R_q3 = np.eye(3)[None, None] + sq * _AXIS_K[None] + (1.0 - cq) * _AXIS_K2[None]

    R_q = np.zeros((B, N_JOINTS, 4, 4))
    R_q[:, :, :3, :3] = R_q3
    R_q[:, :, 3, 3] = 1.0

    # T_joints[b, j] = _JOINT_CONST[j] @ R_q[b, j]
    T_joints = np.matmul(_JOINT_CONST[None], R_q)  # (B, 6, 4, 4)

    # Accumulate: T = T_waist @ ... @ T_wrist_rotate. No fixed hand transform:
    # gripper_link is the child of the last revolute joint.
    T: np.ndarray = np.eye(4)  # broadcast seed; becomes (B,4,4) after first iter
    for j in range(N_JOINTS):
        T = T @ T_joints[:, j]  # (B, 4, 4)

    pos = T[:, :3, 3]       # (B, 3)
    rot_mat = T[:, :3, :3]  # (B, 3, 3)
    return pos, rot_mat
