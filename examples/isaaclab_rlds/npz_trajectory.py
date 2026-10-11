"""Trajectory (NPZ) helpers shared by the RLDS converters.

Reads the per-trajectory NPZ files written by SkillWeaver's data collection, builds state / action arrays
according to DataCfg, and resolves each trajectory's task entry in the scene description JSON.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation
from tqdm import tqdm


_TRAJ_STEM_RE = re.compile(r"^(?P<tag>.+?)_traj\d+(?:_restore2step)?$")

_TASK_TAG_RE = re.compile(r"^(scene_\d+)\+(?:(tasks\d*)\+)?([A-Za-z0-9_]+)\+task(\d+)$")

@dataclass(frozen=True)
class TrajRecord:
    npz_path: pathlib.Path
    scene_name: str | None
    task_key_from_tag: str | None
    task_type: str | None
    task_idx: int | None

@dataclass(frozen=True)
class ChunkRange:
    frame_start: int
    frame_end: int
    node_start: int
    node_end: int

class MissingAugmentDataError(RuntimeError):
    """Raised when convert cannot find required augment-generated instruction data."""

@dataclass(frozen=True)
class DataCfg:
    """Controls which NPZ keys are used to build state and action vectors."""

    # --- state ---
    state_repr: str = "joint"           # "ee" | "joint"
    state_gripper_1d: bool = False   # True: pool 2D gripper state to 1D mean

    # --- action arm ---
    arm_repr: str = "joint"             # "ee" | "joint"
    arm_action_type: str = "actual"  # "cmd" | "target" | "actual"

    # --- action gripper ---
    gripper_action_type: str = "actual"  # "cmd" | "target" | "actual"
    gripper_1d: bool = False         # True: pool gripper to 1D mean; False: keep 2D

    # --- residual ---
    make_target_residual: bool = False  # subtract current state from action targets to get deltas

    def validate(self) -> None:
        if self.state_repr not in ("ee", "joint"):
            raise ValueError(f"state_repr must be 'ee' or 'joint', got '{self.state_repr}'")
        if self.arm_repr not in ("ee", "joint"):
            raise ValueError(f"arm_repr must be 'ee' or 'joint', got '{self.arm_repr}'")
        if self.arm_action_type not in ("cmd", "target", "actual"):
            raise ValueError(f"arm_action_type must be 'cmd'|'target'|'actual', got '{self.arm_action_type}'")
        if self.gripper_action_type not in ("cmd", "target", "actual"):
            raise ValueError(f"gripper_action_type must be 'cmd'|'target'|'actual', got '{self.gripper_action_type}'")
        if self.arm_repr == "joint" and self.arm_action_type == "cmd":
            raise ValueError("arm_repr='joint' + arm_action_type='cmd' is invalid: no joint delta cmd is stored.")
        if self.make_target_residual and self.arm_action_type == "cmd":
            raise ValueError("make_target_residual=True with arm_action_type='cmd' is invalid: cmd is already a delta.")

    def _arm_dim_from_npz(self, npz_dict: dict[str, Any], repr_: str, action_type: str) -> int:
        if repr_ == "joint":
            key = "action/arm_joint_actual" if action_type == "actual" else "action/arm_joint_targets"
            arr = npz_dict.get(key)
            if arr is None:
                # fallback: try state key for state_repr=joint
                arr = npz_dict.get("state/arm_joint_pos")
            if arr is None:
                raise KeyError("Cannot determine arm_dim for repr=joint")
            return int(np.asarray(arr).shape[-1])
        return 6  # ee: pos(3) + axis_angle(3)

    def state_dim(self, npz_dict: dict[str, Any]) -> int:
        arm = self._arm_dim_from_npz(npz_dict, self.state_repr, "actual")
        gripper_dim = 1 if self.state_gripper_1d else 2
        return arm + gripper_dim

    def action_dim(self, npz_dict: dict[str, Any]) -> int:
        arm = self._arm_dim_from_npz(npz_dict, self.arm_repr, self.arm_action_type)
        gripper_dim = 1 if self.gripper_1d or self.gripper_action_type == "cmd" else 2
        return arm + gripper_dim

def _build_state_array(traj: dict[str, Any], cfg: DataCfg) -> np.ndarray:
    """Build (T, state_dim) float32 array from traj according to DataCfg."""
    if cfg.state_repr == "ee":
        pos = traj.get("state/ee/pos")
        aa = traj.get("state/ee/axis_angle")
        if pos is None or aa is None:
            raise KeyError("Missing 'state/ee/pos' or 'state/ee/axis_angle'.")
        arm = np.concatenate(
            [np.asarray(pos, dtype=np.float32), np.asarray(aa, dtype=np.float32)], axis=-1
        )  # (T, 6)
    else:  # joint
        arr = traj.get("state/arm_joint_pos")
        if arr is None:
            raise KeyError("Missing 'state/arm_joint_pos'.")
        arm = np.asarray(arr, dtype=np.float32)  # (T, arm_dim)

    grip = traj.get("state/gripper_pos")
    if grip is None:
        raise KeyError("Missing 'state/gripper_pos'.")
    gripper = np.asarray(grip, dtype=np.float32)
    if gripper.ndim == 1:
        gripper = gripper.reshape(-1, 1)
    if gripper.shape[-1] > 1:
        if cfg.state_gripper_1d:
            gripper = gripper[:, 0:1]  # finger1 only; finger2 has opposite sign convention
        else:
            gripper = gripper * np.array([1.0, -1.0], dtype=np.float32)  # negate finger2

    state = np.concatenate([arm, gripper], axis=-1).astype(np.float32)
    if state.ndim == 3 and state.shape[1] == 1:
        state = np.squeeze(state, axis=1)
    elif state.ndim > 2:
        state = state.reshape(state.shape[0], -1)
    return state

def _build_action_array(traj: dict[str, Any], cfg: DataCfg) -> np.ndarray:
    """Build (T, action_dim) float32 array from traj according to DataCfg."""
    # --- arm ---
    if cfg.arm_repr == "ee":
        if cfg.arm_action_type == "cmd":
            pos = traj.get("action/ee_cmd/delta_pos")
            aa = traj.get("action/ee_cmd/delta_axis_angle")
            if pos is None or aa is None:
                raise KeyError("Missing 'action/ee_cmd/delta_pos' or 'action/ee_cmd/delta_axis_angle'. "
                               "Was the trajectory collected with planner_execute_mode1_replan=True?")
        elif cfg.arm_action_type == "target":
            pos = traj.get("action/ee_targets/pos")
            aa = traj.get("action/ee_targets/axis_angle")
            if pos is None or aa is None:
                raise KeyError("Missing 'action/ee_targets/pos' or 'action/ee_targets/axis_angle'.")
        else:  # actual
            pos = traj.get("action/ee_actual/pos")
            aa = traj.get("action/ee_actual/axis_angle")
            if pos is None or aa is None:
                raise KeyError("Missing 'action/ee_actual/pos' or 'action/ee_actual/axis_angle'.")
        arm = np.concatenate(
            [np.asarray(pos, dtype=np.float32), np.asarray(aa, dtype=np.float32)], axis=-1
        )  # (T, 6)
    else:  # joint
        key = "action/arm_joint_actual" if cfg.arm_action_type == "actual" else "action/arm_joint_targets"
        arr = traj.get(key)
        if arr is None:
            raise KeyError(f"Missing '{key}'.")
        arm = np.asarray(arr, dtype=np.float32)  # (T, arm_dim)

    # --- gripper ---
    if cfg.gripper_action_type == "cmd":
        g = traj.get("action/gripper_cmd_delta")
        if g is None:
            raise KeyError("Missing 'action/gripper_cmd_delta'. "
                           "Was the trajectory collected with planner_execute_mode1_replan=True?")
        gripper = np.asarray(g, dtype=np.float32).reshape(arm.shape[0], 1)
    elif cfg.gripper_action_type == "target":
        g = traj.get("action/gripper_targets")
        if g is None:
            raise KeyError("Missing 'action/gripper_targets'.")
        gripper = np.asarray(g, dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        if cfg.gripper_1d and gripper.shape[-1] > 1:
            gripper = gripper.mean(axis=-1, keepdims=True)
    else:  # actual
        g = traj.get("action/gripper_actual")
        if g is None:
            raise KeyError("Missing 'action/gripper_actual'.")
        gripper = np.asarray(g, dtype=np.float32)
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        if gripper.shape[-1] > 1:
            if cfg.gripper_1d:
                gripper = gripper[:, 0:1]  # finger1 only; finger2 has opposite sign convention
            else:
                gripper = gripper * np.array([1.0, -1.0], dtype=np.float32)  # negate finger2

    actions = np.concatenate([arm, gripper], axis=-1).astype(np.float32)

    # Squeeze spurious batch dims
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = np.squeeze(actions, axis=1)
    elif actions.ndim > 2:
        actions = actions.reshape(actions.shape[0], -1)
    return actions

def _apply_target_residual(
    actions: np.ndarray,
    traj: dict[str, Any],
    cfg: DataCfg,
) -> np.ndarray:
    """Subtract current state from action arm targets to produce deltas.

    - joint arm:  delta_joint = joint_target[t] - joint_state[t]  (simple subtraction)
    - ee pos:     delta_pos   = ee_target_pos[t] - ee_state_pos[t]
    - ee rot:     delta_aa    = rotvec(q_rel) where q_rel = q_tgt * q_state^{-1}  (world-frame)
    - gripper:    unchanged

    Only the arm columns (first 6 for ee, first arm_dim for joint) are modified.
    """
    actions = actions.copy()

    if cfg.arm_repr == "joint":
        state_joint = np.asarray(traj["state/arm_joint_pos"], dtype=np.float32)  # (T, arm_dim)
        arm_dim = state_joint.shape[-1]
        actions[:, :arm_dim] = actions[:, :arm_dim] - state_joint
    else:  # ee: pos(3) + axis_angle(3) = 6 columns
        state_pos = np.asarray(traj["state/ee/pos"], dtype=np.float32)        # (T, 3)
        state_aa  = np.asarray(traj["state/ee/axis_angle"], dtype=np.float32) # (T, 3)
        tgt_aa    = actions[:, 3:6].copy()                                     # (T, 3)

        # position: simple subtraction
        actions[:, :3] = actions[:, :3] - state_pos

        # rotation: q_rel = q_tgt * q_state^{-1}, expressed as rotation vector (axis * angle)
        q_state = _Rotation.from_rotvec(state_aa)
        q_tgt   = _Rotation.from_rotvec(tgt_aa)
        q_rel   = q_tgt * q_state.inv()
        actions[:, 3:6] = q_rel.as_rotvec().astype(np.float32)

    return actions

def _load_traj(npz_path: pathlib.Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}

def _get_base_views(npz_dict: dict[str, Any]) -> list[str]:
    """List all non-wrist rgb views."""
    return sorted([k for k in npz_dict.keys() if k.startswith("sensor/rgbs/") and "wrist" not in k])

def _parse_task_identity_from_npz(npz_path: pathlib.Path) -> tuple[str, str | None, str, int] | None:
    stem_match = _TRAJ_STEM_RE.match(npz_path.stem)
    if stem_match is None:
        return None
    task_tag = stem_match.group("tag")
    tag_match = _TASK_TAG_RE.match(task_tag)
    if tag_match is None:
        return None
    scene_name, task_key, task_type, task_idx_text = tag_match.groups()
    return scene_name, task_key, task_type, int(task_idx_text)

def gather_traj_records(data_root: pathlib.Path) -> list[TrajRecord]:
    """Traverse data_root for traj/*.npz and infer task identity."""
    traj_paths = sorted(data_root.rglob("traj/*.npz"))
    if not traj_paths:
        raise FileNotFoundError(f"No traj npz found under {data_root}")

    records: list[TrajRecord] = []

    for npz_path in traj_paths:
        parsed = _parse_task_identity_from_npz(npz_path)
        if parsed is None:
            scene_name = None
            task_key_from_tag = None
            task_type = None
            task_idx = None
        else:
            scene_name, task_key_from_tag, task_type, task_idx = parsed

        records.append(
            TrajRecord(
                npz_path=npz_path,
                scene_name=scene_name,
                task_key_from_tag=task_key_from_tag,
                task_type=task_type,
                task_idx=task_idx,
            )
        )
    return records

def _load_scene_desc(scene_desc_root: pathlib.Path, scene_name: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if scene_name in cache:
        return cache[scene_name]
    scene_json = scene_desc_root / scene_name / f"{scene_name}.json"
    with open(scene_json, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid scene json: {scene_json}")
    # Task file paths ("tasks", "tasks1", ...) may be given relative to the scene json.
    for key, value in list(data.items()):
        if key.startswith("tasks") and isinstance(value, str) and value and not pathlib.Path(value).is_absolute():
            data[key] = str(scene_json.parent / value)
    cache[scene_name] = data
    return data

def _load_tasks_for_key(
    scene_desc: dict[str, Any],
    scene_name: str,
    task_key: str,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (scene_name, task_key)
    if cache_key in cache:
        return cache[cache_key]
    tasks_path = scene_desc.get(task_key)
    if not isinstance(tasks_path, str) or not tasks_path:
        raise KeyError(f"scene={scene_name} missing task_key={task_key}")
    with open(tasks_path, "r", encoding="utf-8") as fp:
        tasks = json.load(fp)
    if not isinstance(tasks, dict):
        raise ValueError(f"Invalid tasks json: {tasks_path}")
    cache[cache_key] = tasks
    return tasks

def _resolve_task_entry(
    record: TrajRecord,
    scene_desc_root: pathlib.Path,
    default_task_key: str,
    scene_cache: dict[str, dict[str, Any]],
    tasks_cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[Any | None, str | None]:
    if (
        record.scene_name is None
        or record.task_type is None
        or record.task_idx is None
    ):
        return None, None

    effective_task_key = record.task_key_from_tag or default_task_key
    scene_desc = _load_scene_desc(scene_desc_root, record.scene_name, scene_cache)
    tasks = _load_tasks_for_key(scene_desc, record.scene_name, effective_task_key, tasks_cache)
    task_list = tasks.get(record.task_type)
    if not isinstance(task_list, list):
        raise KeyError(
            f"scene={record.scene_name} task_key={effective_task_key} missing task_type={record.task_type}"
        )
    if not (0 <= int(record.task_idx) < len(task_list)):
        raise IndexError(
            f"scene={record.scene_name} task_key={effective_task_key} task_idx={record.task_idx} out of range"
        )
    return task_list[int(record.task_idx)], effective_task_key

def _extract_seq_split_specs(task_entry: Any) -> list[list[int]]:
    if not isinstance(task_entry, list) or len(task_entry) < 4 or task_entry[0] != "seq":
        return []
    raw_specs = task_entry[3]
    if not isinstance(raw_specs, list):
        return []

    specs: list[list[int]] = []
    for one in raw_specs:
        if not isinstance(one, list):
            continue
        try:
            specs.append([int(x) for x in one])
        except Exception:
            continue
    return specs

def _load_instruction_pool_rows(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out

    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                tqdm.write(f"[warn] skip malformed instruction pool line {line_no}: {path}")
                continue
            if not isinstance(row, dict):
                continue
            npz_raw = row.get("npz_path")
            if not isinstance(npz_raw, str) or not npz_raw.strip():
                continue
            npz_file = pathlib.Path(npz_raw.strip())
            if not npz_file.is_absolute():
                # Relative npz paths are relative to the manifest file.
                npz_file = path.parent / npz_file
            try:
                key = str(npz_file.resolve())
            except Exception:
                key = npz_raw.strip()
            out[key] = row
    return out
