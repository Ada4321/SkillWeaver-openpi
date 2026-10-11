"""Replay LIBERO GT demonstrations from HDF5 files and save NPZ trajectories.

Schema matches vla-data-collection (IsaacLab) NPZ output so that the same
convert_npz_to_rlds.py pipeline can consume both datasets without modification.

See REPLAY_DESIGN.md for full design rationale and field-level notes.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Quaternion / rotation-matrix helpers (self-contained, adapted from
# replay_libero.py — keep in sync if that file changes)
# ---------------------------------------------------------------------------

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(qq))
    if n <= 1e-8:
        return _IDENTITY_QUAT_WXYZ.copy()
    return (qq / n).astype(np.float32)


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm([x, y, z, w]))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize_wxyz(np.array([w, x, y, z], dtype=np.float32))


def _quat2axisangle_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to axis-angle (same function as IsaacLab's)."""
    x, y, z, w = float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0])
    w = max(-1.0, min(1.0, w))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.acos(w)
    return np.array([x, y, z], dtype=np.float64) * angle / den


def _axisangle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to quaternion wxyz (4,)."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = aa / angle
    h = angle / 2.0
    return np.array([math.cos(h), *(math.sin(h) * axis)], dtype=np.float32)


# ---------------------------------------------------------------------------
# LIBERO env helpers
# ---------------------------------------------------------------------------

def _decode_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _list_model_body_names(model) -> list[str]:
    if hasattr(model, "body_names"):
        return [_decode_name(x) for x in list(model.body_names)]
    names: list[str] = []
    if hasattr(model, "nbody") and hasattr(model, "body_id2name"):
        for bid in range(int(model.nbody)):
            nm = model.body_id2name(int(bid))
            if nm is not None:
                names.append(_decode_name(nm))
    return names


def _get_libero_env(
    bddl_file: str,
    resolution: int = 256,
    controller: str | None = None,
):
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if controller is not None:
        env_args["controller"] = controller
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def _override_joint_controller(
    env,
    joint_kp: np.ndarray | None = None,
    damping_ratio: float | None = None,
    policy_freq: int | None = None,
) -> None:
    """Override JOINT_POSITION controller params. Must be called after env.reset()."""
    ctrl = env.env.robots[0].controller
    if joint_kp is not None:
        ctrl.kp = np.asarray(joint_kp, dtype=float)
    dr = damping_ratio if damping_ratio is not None else 1.0
    ctrl.kd = 2.0 * np.sqrt(ctrl.kp) * dr
    if policy_freq is not None:
        ctrl.control_freq = policy_freq


def _get_robot_base_pose(env) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_world (3,), xmat_world (3,3)) of the robot base body."""
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    if sim is None:
        raise RuntimeError("Cannot access env.sim")
    model = sim.model
    body_names = _list_model_body_names(model)
    lower_to_name = {x.lower(): x for x in body_names}

    # Try root_body from robot model
    robots = getattr(env_core, "robots", None)
    root_body = None
    if isinstance(robots, (list, tuple)) and robots:
        robot_model = getattr(robots[0], "robot_model", None)
        if robot_model is not None:
            rb = getattr(robot_model, "root_body", None)
            if rb is not None:
                root_body = _decode_name(rb)
    if isinstance(root_body, str) and root_body:
        hit = lower_to_name.get(root_body.lower())
        if hit is not None:
            root_body = hit
        try:
            _ = model.body_name2id(root_body)
        except Exception:
            root_body = None

    if root_body is None:
        for name in ["robot0_base", "robot0_base_link", "robot0_link0", "panda_link0", "base_link"]:
            hit = lower_to_name.get(name.lower())
            if hit is not None:
                root_body = hit
                break
    if root_body is None:
        for nm in body_names:
            if "robot0" in nm.lower() and ("base" in nm.lower() or "link0" in nm.lower()):
                root_body = nm
                break
    if root_body is None:
        raise RuntimeError(f"Cannot find robot base body. bodies={body_names[:20]}")

    body_id = int(sim.model.body_name2id(root_body))
    p_bw = np.asarray(sim.data.body_xpos[body_id], dtype=np.float32).reshape(3)
    r_bw = np.asarray(sim.data.body_xmat[body_id], dtype=np.float32).reshape(3, 3)
    return p_bw, r_bw


def _get_eef_body_name(env) -> str:
    """Return MuJoCo body name for panda_hand (flange, matches IsaacLab hand_link_idx)."""
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    if sim is None:
        raise RuntimeError("Cannot access env.sim")
    body_names = _list_model_body_names(sim.model)
    lower_to_name = {x.lower(): x for x in body_names}

    # Try robot_model.eef_name first
    robots = getattr(env_core, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        robot_model = getattr(robots[0], "robot_model", None)
        if robot_model is not None:
            eef_name = getattr(robot_model, "eef_name", None)
            if isinstance(eef_name, str) and eef_name:
                hit = lower_to_name.get(eef_name.lower())
                if hit is not None:
                    return hit
                try:
                    _ = sim.model.body_name2id(eef_name)
                    return eef_name
                except Exception:
                    pass

    for name in ["robot0_right_hand", "robot0_panda_hand", "panda_hand", "right_hand"]:
        hit = lower_to_name.get(name.lower())
        if hit is not None:
            return hit

    for nm in body_names:
        if "hand" in nm.lower() and "finger" not in nm.lower():
            return nm

    raise RuntimeError(f"Cannot find eef body name. bodies={body_names[:20]}")


def _obs_to_base_ee_state(
    env,
    eef_body_name: str,
    base_pos_world: np.ndarray,
    base_xmat_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Read panda_hand body pose from MuJoCo (NOT obs["robot0_eef_pos"] which is grip_site),
    transform to robot base frame, return {eef_pos_base, eef_quat_wxyz_base, gripper_pos}.
    axis_angle is NOT returned here — caller must call _quat2axisangle_wxyz(quat).
    """
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    if sim is None:
        raise RuntimeError("Cannot access env.sim")

    body_id = int(sim.model.body_name2id(eef_body_name))
    p_w = np.asarray(sim.data.body_xpos[body_id], dtype=np.float32).reshape(3)
    # MuJoCo body_xquat: [w, x, y, z] → convert to xyzw for _quat_xyzw_to_rotmat
    q_wxyz = np.asarray(sim.data.body_xquat[body_id], dtype=np.float32).reshape(4)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)

    robots = getattr(env_core, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        ref_idx = getattr(robots[0], "_ref_gripper_joint_pos_indexes", None)
        if ref_idx is not None:
            gripper_pos = np.asarray(sim.data.qpos[ref_idx], dtype=np.float32).reshape(2)
        else:
            gripper_pos = np.asarray(sim.data.qpos[-2:], dtype=np.float32).reshape(2)
    else:
        gripper_pos = np.asarray(sim.data.qpos[-2:], dtype=np.float32).reshape(2)

    r_bw = np.asarray(base_xmat_world, dtype=np.float32).reshape(3, 3)
    p_bw = np.asarray(base_pos_world, dtype=np.float32).reshape(3)
    p_b = (r_bw.T @ (p_w - p_bw)).astype(np.float32)

    r_we = _quat_xyzw_to_rotmat(q_xyzw)
    r_be = r_bw.T @ r_we
    q_be_wxyz = _rotmat_to_quat_wxyz(r_be).astype(np.float32)

    return {
        "eef_pos_base": p_b,
        "eef_quat_wxyz_base": q_be_wxyz,
        "gripper_pos": gripper_pos,
    }


def _get_gripper_finger1_target(env) -> float:
    """
    Return finger1 position target (m, range [0, 0.04]) from sim.data.ctrl.
    Read AFTER env.step() — the gripper integrator has already updated ctrl.
    Only finger1 is used because finger2 has opposite sign convention ([-0.04, 0]).
    """
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    if sim is None:
        raise RuntimeError("Cannot access env.sim")
    robots = getattr(env_core, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        actuator_idxs = getattr(robots[0], "_ref_joint_gripper_actuator_indexes", None)
        if actuator_idxs is not None and len(actuator_idxs) > 0:
            return float(sim.data.ctrl[actuator_idxs[0]])
    # Fallback: find gripper_finger_joint1 actuator by name
    try:
        act_id = sim.model.actuator_name2id("gripper_finger_joint1")
        return float(sim.data.ctrl[act_id])
    except Exception:
        pass
    raise RuntimeError("Cannot find gripper finger1 actuator index")


def _collect_arm_joints(env) -> np.ndarray:
    """Return arm joint positions (7,) float32 from obs or sim.data.qpos."""
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    robots = getattr(env_core, "robots", None)
    if isinstance(robots, (list, tuple)) and robots:
        ref_idx = getattr(robots[0], "_ref_joint_pos_indexes", None)
        if ref_idx is not None:
            return np.asarray(sim.data.qpos[ref_idx], dtype=np.float32).reshape(-1)
    raise RuntimeError("Cannot find arm joint position indexes")


def _fk_ee_base_pose(
    env,
    arm_target: np.ndarray,
    eef_body_name: str,
    base_pos_world: np.ndarray,
    base_xmat_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK: temporarily set arm qpos to arm_target, forward(), read EE pose in base frame.

    Returns (pos (3,), quat_wxyz (4,), axis_angle (3,)) — equivalent to Isaac's _fk_ee_pose_from_joint.
    """
    env_core = getattr(env, "env", env)
    sim = getattr(env_core, "sim", None)
    if sim is None:
        raise RuntimeError("Cannot access env.sim")
    robots = getattr(env_core, "robots", None)
    if not (isinstance(robots, (list, tuple)) and robots):
        raise RuntimeError("Cannot access env robots")
    ref_idx = getattr(robots[0], "_ref_joint_pos_indexes", None)
    if ref_idx is None:
        raise RuntimeError("Cannot find arm joint position indexes")

    state_backup = sim.get_state()
    try:
        sim.data.qpos[ref_idx] = arm_target
        sim.forward()
        body_id = int(sim.model.body_name2id(eef_body_name))
        p_w = np.asarray(sim.data.body_xpos[body_id], dtype=np.float32).reshape(3)
        q_wxyz = np.asarray(sim.data.body_xquat[body_id], dtype=np.float32).reshape(4)
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    finally:
        sim.set_state(state_backup)
        sim.forward()

    r_bw = np.asarray(base_xmat_world, dtype=np.float32).reshape(3, 3)
    p_bw = np.asarray(base_pos_world, dtype=np.float32).reshape(3)
    p_b = (r_bw.T @ (p_w - p_bw)).astype(np.float32)
    r_we = _quat_xyzw_to_rotmat(q_xyzw)
    r_be = r_bw.T @ r_we
    q_be_wxyz = _rotmat_to_quat_wxyz(r_be).astype(np.float32)
    aa_be = _quat2axisangle_wxyz(q_be_wxyz).astype(np.float32)
    return p_b, q_be_wxyz, aa_be


def _resize_rgb(img: np.ndarray, resolution: int) -> np.ndarray:
    """Resize HxWx3 uint8 image to resolution×resolution."""
    if img.shape[0] == resolution and img.shape[1] == resolution:
        return img
    return cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)


def _flip_and_resize(img: np.ndarray, resolution: int) -> np.ndarray:
    """Apply vertical flip (OpenGL origin) then resize."""
    flipped = np.ascontiguousarray(img[::-1, :, :3])
    return _resize_rgb(flipped, resolution)


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def _replay_demo(
    hdf5_path: Path,
    demo_key: str,
    output_dir: Path,
    bddl_file: str,
    cmd_limit: np.ndarray,
    resolution: int,
    language: str,
    warmup_steps: int = 0,
    replay_mode: str = "ee_delta",
    track_error: bool = False,
    joint_kp: np.ndarray | None = None,
    joint_damping_ratio: float | None = None,
    joint_policy_freq: int | None = None,
) -> dict:
    """Replay one demo, save NPZ, return report dict."""
    report: dict = {"demo": demo_key, "hdf5": str(hdf5_path)}

    with h5py.File(hdf5_path, "r") as f:
        demo = f[f"data/{demo_key}"]
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float32)  # (T, 7)
        if replay_mode == "joint_pos":
            joint_targets = np.asarray(demo["obs/joint_states"], dtype=np.float32)  # (T, 7)
    T = actions.shape[0]

    env = _get_libero_env(
        bddl_file,
        resolution=resolution,
        controller="JOINT_POSITION" if replay_mode == "joint_pos" else None,
    )
    try:
        env.reset()
        if replay_mode == "joint_pos" and any(
            x is not None for x in (joint_kp, joint_damping_ratio, joint_policy_freq)
        ):
            _override_joint_controller(env, joint_kp, joint_damping_ratio, joint_policy_freq)
        obs = env.set_init_state(init_state)

        # Warmup steps (default 0 for GT demos)
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        # Cache per-episode quantities
        base_pos, base_xmat = _get_robot_base_pose(env)
        eef_body = _get_eef_body_name(env)

        # Camera calibration (fixed for the whole episode).
        try:
            from robosuite.utils import camera_utils as _CU  # type: ignore
            _sim = env.env.sim
            _ixt = _CU.get_camera_intrinsic_matrix(_sim, "agentview", resolution, resolution)   # (3,3)
            _ext_w2c = np.linalg.inv(_CU.get_camera_extrinsic_matrix(_sim, "agentview"))        # (4,4) w2c ROS
        except Exception:
            _ixt = np.eye(3, dtype=np.float32)
            _ext_w2c = np.eye(4, dtype=np.float32)

        # Trajectory buffers (list of per-step dicts, stacked at end)
        steps: list[dict] = []
        success_step: int = -1  # -1 = never succeeded

        for t in range(T):
            step: dict = {}

            # ---- pre-step sensor ----
            # Live env obs keys: "agentview_image", "robot0_eye_in_hand_image"
            # (HDF5 stores as "agentview_rgb" / "eye_in_hand_rgb" — different names)
            _front = obs.get("agentview_image") if obs.get("agentview_image") is not None else obs.get("agentview_rgb")
            _wrist = (obs.get("robot0_eye_in_hand_image") if obs.get("robot0_eye_in_hand_image") is not None
                      else obs.get("eye_in_hand_image") if obs.get("eye_in_hand_image") is not None
                      else obs.get("eye_in_hand_rgb"))
            if _front is None:
                raise KeyError(f"No front camera key found. obs keys: {list(obs.keys())}")
            if _wrist is None:
                raise KeyError(f"No wrist camera key found. obs keys: {list(obs.keys())}")
            front_rgb = np.asarray(_front)
            wrist_rgb = np.asarray(_wrist)
            step["sensor/rgbs/front"] = _flip_and_resize(front_rgb, resolution)
            step["sensor/rgbs/wrist"] = _flip_and_resize(wrist_rgb, resolution)

            # ---- pre-step state ----
            ee = _obs_to_base_ee_state(env, eef_body, base_pos, base_xmat)
            step["state/ee/pos"] = ee["eef_pos_base"]
            step["state/ee/quat"] = ee["eef_quat_wxyz_base"]
            step["state/ee/axis_angle"] = _quat2axisangle_wxyz(ee["eef_quat_wxyz_base"]).astype(np.float32)
            step["state/arm_joint_pos"] = _collect_arm_joints(env)
            step["state/gripper_pos"] = ee["gripper_pos"]

            # ---- action cmd ----
            if replay_mode == "joint_pos":
                arm_target = joint_targets[min(t + 1, T - 1)]
                gripper_cmd = float(actions[t, 6])
                step_action = np.concatenate([arm_target, [gripper_cmd]]).tolist()
                step["action/arm_joint_targets"] = arm_target
                tgt_pos, tgt_quat, tgt_aa = _fk_ee_base_pose(env, arm_target, eef_body, base_pos, base_xmat)
                step["action/ee_targets/pos"] = tgt_pos
                step["action/ee_targets/quat"] = tgt_quat
                step["action/ee_targets/axis_angle"] = tgt_aa
            else:
                delta_pos = (actions[t, 0:3] * cmd_limit[0:3]).astype(np.float32)
                delta_aa = (actions[t, 3:6] * cmd_limit[3:6]).astype(np.float32)
                step["action/ee_cmd/delta_pos"] = delta_pos
                step["action/ee_cmd/delta_axis_angle"] = delta_aa
                step["action/ee_cmd/delta_quat"] = _axisangle_to_quat_wxyz(delta_aa)
                step_action = actions[t].tolist()

            # ---- physics step ----
            obs, _reward, _done, _info = env.step(step_action)
            if env.check_success():
                success_step = t

            # ---- gripper target (read AFTER step, finger1 only, [0,0.04] m) ----
            step["action/gripper_targets"] = np.array(
                [_get_gripper_finger1_target(env)], dtype=np.float32
            )

            # ---- post-step actuals ----
            ee_post = _obs_to_base_ee_state(env, eef_body, base_pos, base_xmat)
            step["action/ee_actual/pos"] = ee_post["eef_pos_base"]
            step["action/ee_actual/quat"] = ee_post["eef_quat_wxyz_base"]
            step["action/ee_actual/axis_angle"] = _quat2axisangle_wxyz(ee_post["eef_quat_wxyz_base"]).astype(np.float32)
            step["action/arm_joint_actual"] = _collect_arm_joints(env)
            step["action/gripper_actual"] = ee_post["gripper_pos"]

            steps.append(step)

        # ---- tracking error (joint_pos mode only) ----
        if track_error and replay_mode == "joint_pos":
            jt = np.stack([s["action/arm_joint_targets"] for s in steps], axis=0)   # (T, 7)
            ja = np.stack([s["action/arm_joint_actual"]  for s in steps], axis=0)   # (T, 7)
            ep = np.stack([s["action/ee_targets/pos"]    for s in steps], axis=0)   # (T, 3)
            ea = np.stack([s["action/ee_actual/pos"]     for s in steps], axis=0)   # (T, 3)
            joint_err = np.abs(ja - jt)          # (T, 7)
            ee_err    = np.linalg.norm(ea - ep, axis=1)  # (T,)
            print(f"  [track_error] {hdf5_path.name}/{demo_key}  T={T}")
            print(f"    joint MAE per-joint: {joint_err.mean(axis=0).round(4).tolist()}")
            print(f"    joint MAE overall:   {joint_err.mean():.4f} rad  max: {joint_err.max():.4f} rad")
            print(f"    EE pos L2:  mean={ee_err.mean():.4f} m  max={ee_err.max():.4f} m  final={ee_err[-1]:.4f} m", flush=True)

        # ---- stack over time ----
        npz_data: dict[str, np.ndarray] = {}
        for key in steps[0].keys():
            npz_data[key] = np.stack([s[key] for s in steps], axis=0)

        # ---- meta ----
        npz_data["meta/node_end_indices_exclusive"] = np.array([T], dtype=np.int32)
        npz_data["meta/node_subgoal"] = np.array([language], dtype=object)
        npz_data["meta/language"] = np.array(language, dtype=object)

        # ---- camera calibration (constant for whole episode) ----
        npz_data["sensor/ixts/front"]    = np.stack([_ixt]     * T, axis=0).astype(np.float32)  # (T,3,3)
        npz_data["sensor/exts/front"]    = np.stack([_ext_w2c] * T, axis=0).astype(np.float32)  # (T,4,4)
        T_base = np.eye(4, dtype=np.float32)
        T_base[:3, :3] = base_xmat
        T_base[:3, 3]  = base_pos
        npz_data["meta/robot_base_pose_w"] = T_base.flatten()  # (16,)

        # ---- save ----
        demo_idx = int(demo_key.replace("demo_", ""))
        suc_tag = "success" if success_step >= 0 else "fail"
        out_traj_dir = output_dir / f"demo_{demo_idx}" / "traj"
        out_traj_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_traj_dir / f"demo_{demo_idx}_traj0_{suc_tag}.npz"
        np.savez_compressed(str(out_path), **npz_data)

        report["status"] = "ok"
        report["T"] = T
        report["success"] = success_step >= 0
        report["success_step"] = success_step
        report["path"] = str(out_path)

    except Exception:
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            env.close()
        except Exception:
            pass

    return report


# ---------------------------------------------------------------------------
# Per-HDF5 file
# ---------------------------------------------------------------------------

def _replay_hdf5_file(args_tuple: tuple) -> list[dict]:
    hdf5_path, output_dir, cmd_limit, resolution, max_demos, warmup_steps, replay_mode, track_error, joint_kp, joint_damping_ratio, joint_policy_freq = args_tuple

    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir) / hdf5_path.stem
    reports = []

    try:
        from libero.libero import get_libero_path  # type: ignore
        bddl_root = get_libero_path("bddl_files")

        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            # Parse bddl path — stored as relative path in attrs
            bddl_rel = _decode_name(data.attrs.get("bddl_file_name", ""))
            if Path(bddl_rel).is_absolute():
                bddl_file = bddl_rel
            else:
                # HDF5 stores e.g. "libero/libero/bddl_files/libero_object/xxx.bddl"
                # get_libero_path("bddl_files") already ends at "bddl_files/",
                # so strip everything up to and including "bddl_files/" to avoid duplication.
                marker = "bddl_files/"
                idx = bddl_rel.find(marker)
                suffix = bddl_rel[idx + len(marker):] if idx >= 0 else bddl_rel
                bddl_file = str(Path(bddl_root) / suffix)
            # Language instruction from problem_info (JSON) or env_name fallback
            import json as _json
            problem_info_raw = data.attrs.get("problem_info", "")
            if isinstance(problem_info_raw, bytes):
                problem_info_raw = problem_info_raw.decode("utf-8")
            language = hdf5_path.stem  # fallback
            if problem_info_raw:
                try:
                    pi = _json.loads(problem_info_raw)
                    language = str(pi.get("language_instruction", hdf5_path.stem))
                except Exception:
                    language = str(problem_info_raw)

            demo_keys = sorted(
                [k for k in data.keys() if k.startswith("demo_")],
                key=lambda x: int(x.replace("demo_", "")),
            )

        if max_demos > 0:
            demo_keys = demo_keys[:max_demos]

        for demo_key in demo_keys:
            report = _replay_demo(
                hdf5_path=hdf5_path,
                demo_key=demo_key,
                output_dir=output_dir,
                bddl_file=bddl_file,
                cmd_limit=cmd_limit,
                resolution=resolution,
                language=language,
                warmup_steps=warmup_steps,
                replay_mode=replay_mode,
                track_error=track_error,
                joint_kp=joint_kp,
                joint_damping_ratio=joint_damping_ratio,
                joint_policy_freq=joint_policy_freq,
            )
            reports.append(report)
            status = report.get("status", "?")
            T = report.get("T", "?")
            suc = report.get("success")
            suc_str = ("SUCCESS@%d" % report["success_step"]) if suc else ("FAIL" if suc is False else "?")
            print(f"  [{status}] {hdf5_path.name}/{demo_key}  T={T}  {suc_str}", flush=True)

    except Exception:
        reports.append({
            "hdf5": str(hdf5_path),
            "status": "error",
            "traceback": traceback.format_exc(),
        })

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cmd_limit(raw: str) -> np.ndarray:
    vals = np.array([float(x.strip()) for x in raw.split(",")], dtype=np.float32)
    if vals.shape[0] == 1:
        vals = np.repeat(vals, 6)
    if vals.shape[0] != 6:
        raise ValueError(f"cmd-limit must have 1 or 6 values, got {vals.shape[0]}")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay LIBERO HDF5 demos → NPZ (IsaacLab schema)")
    parser.add_argument("--hdf5-root", required=True, help="Root directory with *.hdf5 files")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution (default 256)")
    parser.add_argument(
        "--cmd-limit",
        default="0.05,0.05,0.05,0.5,0.5,0.5",
        help="6 comma-separated cmd_limit values for arm (pos×3, rot×3). Default: 0.05,0.05,0.05,0.5,0.5,0.5",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--max-demos", type=int, default=-1, help="Max demos per HDF5 file (-1 = all)")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Dummy steps before replay (default 0)")
    parser.add_argument("--track-error", action="store_true", default=False,
                        help="Print joint MAE and EE pos L2 error per demo (joint_pos mode only, no NPZ output)")
    parser.add_argument("--joint-kp", default=None,
                        help="Override JOINT_POSITION kp: 7 comma-separated values, e.g. '800,800,800,400,400,300,240' (joint_pos mode only)")
    parser.add_argument("--joint-damping-ratio", type=float, default=None,
                        help="Override JOINT_POSITION damping_ratio (default 1.0 = critical damping, joint_pos mode only)")
    parser.add_argument("--joint-policy-freq", type=int, default=None,
                        help="Override JOINT_POSITION policy_freq in Hz (joint_pos mode only)")
    parser.add_argument(
        "--replay-mode",
        choices=["ee_delta", "joint_pos"],
        default="ee_delta",
        help="Replay mode: ee_delta (default, OSC delta EE) or joint_pos (JOINT_POSITION controller)",
    )
    args = parser.parse_args()

    hdf5_root = Path(args.hdf5_root)
    hdf5_files = sorted(hdf5_root.glob("*.hdf5")) + sorted(hdf5_root.glob("*.h5"))
    if not hdf5_files:
        raise RuntimeError(f"No HDF5 files found under {hdf5_root}")

    cmd_limit = _parse_cmd_limit(args.cmd_limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_kp: np.ndarray | None = None
    if args.joint_kp is not None:
        joint_kp = np.array([float(v) for v in args.joint_kp.split(",")], dtype=float)
        if len(joint_kp) != 7:
            raise ValueError(f"--joint-kp expects 7 values, got {len(joint_kp)}")

    print(f"Found {len(hdf5_files)} HDF5 files, workers={args.num_workers}")
    print(f"cmd_limit={cmd_limit.tolist()}, resolution={args.resolution}, replay_mode={args.replay_mode}")
    if joint_kp is not None:
        print(f"joint_kp override={joint_kp.tolist()}")
    if args.joint_damping_ratio is not None:
        print(f"joint_damping_ratio override={args.joint_damping_ratio}")
    if args.joint_policy_freq is not None:
        print(f"joint_policy_freq override={args.joint_policy_freq}")

    task_args = [
        (str(p), str(output_dir), cmd_limit, args.resolution, args.max_demos, args.warmup_steps, args.replay_mode, args.track_error, joint_kp, args.joint_damping_ratio, args.joint_policy_freq)
        for p in hdf5_files
    ]

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **_kw):  # type: ignore
            return it

    all_reports: list[dict] = []
    if args.num_workers <= 1:
        for ta in _tqdm(task_args, desc="HDF5 files", unit="file"):
            all_reports.extend(_replay_hdf5_file(ta))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for reports in _tqdm(
                pool.imap_unordered(_replay_hdf5_file, task_args),
                total=len(task_args),
                desc="HDF5 files",
                unit="file",
            ):
                all_reports.extend(reports)

    ok = sum(1 for r in all_reports if r.get("status") == "ok")
    err = sum(1 for r in all_reports if r.get("status") == "error")
    suc = sum(1 for r in all_reports if r.get("success") is True)
    print(f"\nDone. ok={ok}, error={err}, success={suc}/{ok} ({100*suc/ok:.1f}%)" if ok else f"\nDone. ok={ok}, error={err}")
    for r in all_reports:
        if r.get("status") == "error":
            print(f"  ERROR {r.get('hdf5')} / {r.get('demo', '?')}")
            print("  " + r.get("traceback", "").replace("\n", "\n  "))


if __name__ == "__main__":
    main()
