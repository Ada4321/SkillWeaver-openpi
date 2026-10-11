"""Embodiment dispatch for the FK used by the EEF metrics.

`openpi.eval.action_metrics` turns predicted joint actions into end-effector
poses so it can report a positional/rotational error in metres and radians. That
step needs the *right* arm: feeding 6-DoF WidowX joint angles to the Franka model
silently produces poses ~0.7 m away from the truth, which is exactly what
happened before this module existed (the 7-wide reshape happened to accept
"6 joints + gripper" without complaint).

Which arm a shard was recorded with is a property of the data, so it travels in
the shard's ``dataset_info.json`` under ``robot``. A missing key means "panda":
every suite predating the simpler/WidowX port is Franka.
"""

from collections.abc import Callable
import dataclasses

import numpy as np

from openpi.eval import panda_fk as _panda
from openpi.eval import widowx_fk as _widowx


@dataclasses.dataclass(frozen=True)
class RobotKinematics:
    """Everything the EEF metrics need to know about one embodiment."""

    name: str
    # (B, n_joints) joint angles -> (pos (B,3), rot_mat (B,3,3)) in robot-base frame.
    fk: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    n_joints: int

    def __post_init__(self) -> None:
        for field, arr in (("joint_lower", self.joint_lower), ("joint_upper", self.joint_upper)):
            if arr.shape != (self.n_joints,):
                raise ValueError(
                    f"{self.name}.{field} has shape {arr.shape}, expected ({self.n_joints},)"
                )


_REGISTRY: dict[str, RobotKinematics] = {
    "panda": RobotKinematics(
        name="panda",
        fk=_panda.panda_fk,
        joint_lower=_panda.JOINT_LOWER,
        joint_upper=_panda.JOINT_UPPER,
        n_joints=7,
    ),
    "widowx": RobotKinematics(
        name="widowx",
        fk=_widowx.widowx_fk,
        joint_lower=_widowx.JOINT_LOWER,
        joint_upper=_widowx.JOINT_UPPER,
        n_joints=_widowx.N_JOINTS,
    ),
}

DEFAULT_ROBOT: str = "panda"


def get_kinematics(robot: str | None) -> RobotKinematics:
    """Look up an embodiment; None falls back to the pre-WidowX default."""
    key = (robot or DEFAULT_ROBOT).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown robot {robot!r}. Known: {sorted(_REGISTRY)}. "
            "Add a RobotKinematics entry in openpi/eval/robot_fk.py, and make sure "
            "the shard's dataset_info.json records the matching `robot` key."
        )
    return _REGISTRY[key]


def check_arm_dim(robot: str | None, arm_dim: int, *, arm_repr: str) -> None:
    """Fail loudly when the arm width and the embodiment disagree.

    This is the guard that was missing before: a 7-wide action vector holding
    "6 WidowX joints + 1 gripper" reshaped cleanly into the Franka model's 7
    joints, so the mismatch produced plausible-looking but meaningless numbers
    instead of an error.
    """
    if arm_repr != "joint":
        return  # 'ee' repr is pos(3)+axis_angle(3); no joint model involved.
    kin = get_kinematics(robot)
    if arm_dim != kin.n_joints:
        raise ValueError(
            f"robot={kin.name!r} has {kin.n_joints} joints but arm_dim={arm_dim}. "
            "Either the shard's `robot` key is wrong, or arm_dim was derived with the "
            "wrong gripper width (arm_dim = action_dim - gripper_dim)."
        )
