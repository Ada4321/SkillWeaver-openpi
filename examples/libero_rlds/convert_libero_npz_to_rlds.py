"""
Convert LIBERO NPZ trajectories (from replay_libero_hdf5_to_npz.py) to
IsaacLab RLDS TFRecord format, reusing the same schema as isaaclab_rlds/.

Only *_success.npz files are processed.  Each episode has exactly one
segment (full episode) with a single language instruction from meta/language.

Usage
-----
python examples/libero_rlds/convert_libero_npz_to_rlds.py \\
    --data_root data/libero_npz/libero_object \\
    --output_dir data/libero_rlds/libero_object \\
    [--state_repr joint] [--state_gripper_1d] \\
    [--arm_repr joint] [--arm_action_type actual] \\
    [--gripper_action_type actual] [--gripper_1d] \\
    [--image_size 224] [--pad_hold_steps 0] \\
    [--num_workers 8] [--num_shards 128]

Output layout
-------------
<output_dir>/
  dataset_info.json
  libero_rlds-train.tfrecord-00000-of-00128
  libero_rlds-train.tfrecord-00001-of-00128
  ...

Supported DataCfg options vs IsaacLab
--------------------------------------
  --arm_action_type   : cmd, actual   (target not available: LIBERO OSC has no joint targets)
  --gripper_action_type: actual, target  (cmd not available: binary mismatch)
  --gripper_1d on/off  : both supported
    actual  => (T,2) both on and off work directly
    target  => (T,1) source; off tiles to (T,2) to match IsaacLab format

Recommended config (state_dim=8, action_dim=8, matches IsaacLab joint config):
  --state_repr joint --state_gripper_1d
  --arm_repr joint --arm_action_type actual
  --gripper_action_type actual --gripper_1d
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io as _io
import multiprocessing as _mp
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add isaaclab_rlds (shared schema and trajectory helpers) to path.
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).parent
_RLDS_DIR = str(_THIS_DIR.parent / "isaaclab_rlds")
if _RLDS_DIR not in sys.path:
    sys.path.insert(0, _RLDS_DIR)

from npz_trajectory import (  # noqa: E402
    DataCfg,
)
from dataset_builder import (  # noqa: E402
    DEFAULT_MAX_SEGMENTS,
    MAX_BASE_VIEWS,
    DatasetInfo,
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_and_encode_jpeg(frame: np.ndarray, size: int, quality: int = 90) -> bytes:
    from PIL import Image

    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Hold-action builder (identical to isaaclab_rlds version)
# ---------------------------------------------------------------------------

def _build_hold_action(
    action: np.ndarray,   # (T, action_dim)
    frame_end: int,
    action_dim: int,
    gripper_dim: int,
    arm_is_delta: bool,
) -> np.ndarray:
    hold = np.zeros(action_dim, dtype=np.float32)
    if not arm_is_delta:
        hold[:] = action[frame_end - 1]
    # Gripper always takes the real last value.
    hold[-gripper_dim:] = action[frame_end - 1, -gripper_dim:]
    return hold


# ---------------------------------------------------------------------------
# Gripper binary intent builder (LIBERO-specific: finger1 only)
# ---------------------------------------------------------------------------

def _build_gripper_binary(
    traj: dict[str, Any],
    *,
    gripper_dim: int,
    eps: float = 1e-4,
    init: float = -1.0,
) -> np.ndarray:
    """Per-step ±1 gripper intent recovered from action/gripper_targets vs state/gripper_pos.

    LIBERO-specific: state/gripper_pos has a mirror sign convention
    (state[:, 0] = +x, state[:, 1] = -x), and action/gripper_targets only stores
    finger1 (shape (T, 1)). Use finger1 from both to bypass the sign mirror;
    averaging would give state.mean ≈ 0 and break the comparison.

    Sign convention: -1 = open intent (target > state), +1 = close intent.
    Dead-zone steps (|delta| ≤ eps) carry forward last non-zero sign.
    """
    tgt = traj.get("action/gripper_targets")
    sta = traj.get("state/gripper_pos")
    if tgt is None:
        raise KeyError("Missing 'action/gripper_targets' (required to build gripper_binary).")
    if sta is None:
        raise KeyError("Missing 'state/gripper_pos' (required to build gripper_binary).")
    tgt = np.asarray(tgt, dtype=np.float32)
    sta = np.asarray(sta, dtype=np.float32)
    if tgt.ndim == 1:
        tgt = tgt.reshape(-1, 1)
    if sta.ndim == 1:
        sta = sta.reshape(-1, 1)
    delta = tgt[:, 0] - sta[:, 0]   # finger1 only

    s = np.zeros_like(delta)
    s[delta >  eps] = -1.0   # open intent
    s[delta < -eps] = +1.0   # close intent
    out = np.empty_like(s); last = float(init)
    for i, v in enumerate(s):
        if v != 0.0:
            last = float(v)
        out[i] = last
    out = out.reshape(-1, 1).astype(np.float32)          # (T, 1)
    if gripper_dim > 1:
        out = np.tile(out, (1, gripper_dim))             # (T, gripper_dim)
    return out


# ---------------------------------------------------------------------------
# LIBERO validation (stricter than base DataCfg.validate())
# ---------------------------------------------------------------------------

def _validate_libero_cfg(cfg: DataCfg) -> None:
    cfg.validate()
    if cfg.arm_action_type == "target":
        raise ValueError(
            "arm_action_type='target' is not supported for LIBERO "
            "(OSC control has no arm joint targets)."
        )
    if cfg.gripper_action_type == "cmd":
        raise ValueError(
            "gripper_action_type='cmd' is not supported for LIBERO "
            "(binary {-1,+1} command does not align with IsaacLab)."
        )


# ---------------------------------------------------------------------------
# TFRecord serialisation helpers (no TF computation graph needed)
# ---------------------------------------------------------------------------

def _bytes_feature(value: bytes):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values: list[bytes]):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value: int):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _make_example(
    *,
    exterior_jpegs: list[list[bytes]],    # [slot_idx][frame_idx]
    wrist_jpegs: list[bytes],             # [frame_idx]
    state: np.ndarray,                    # (T, state_dim)
    action: np.ndarray,                   # (T, action_dim)
    action_gripper_binary: np.ndarray,    # (T, gripper_dim) float32 ±1
    exterior_view_count: int,
    segment_count: int,
    segment_starts: np.ndarray,           # (max_segments,) int32
    segment_ends: np.ndarray,             # (max_segments,) int32
    segment_hold_actions: np.ndarray,     # (max_segments, action_dim) float32
    segment_hold_action_gripper_binary: np.ndarray,  # (max_segments, gripper_dim) float32 ±1
    segment_instruction_pools: list[str], # len == max_segments, each \n-joined
    file_path: str,
    recording_folderpath: str,
    camera_ixts: list[np.ndarray],        # [slot] (9,)  float32 — K matrix flattened
    camera_ixt_hws: list[np.ndarray],     # [slot] (2,)  int32  — (H, W) at collection res
    camera_exts: list[np.ndarray],        # [slot] (16,) float32 — w2c 4×4 flattened
    robot_base_pose_w: np.ndarray,        # (16,) float32 — robot base→world 4×4
) -> bytes:
    """Build and serialise a tf.train.Example for one LIBERO episode."""
    import tensorflow as tf

    feature: dict = {}

    for i, frames in enumerate(exterior_jpegs):
        feature[f"steps/observation/exterior_image_{i}"] = _bytes_list_feature(frames)
    feature["steps/observation/wrist_image"] = _bytes_list_feature(wrist_jpegs)

    feature["steps/observation/state"] = _bytes_feature(
        state.astype(np.float32).tobytes()
    )
    feature["steps/action"] = _bytes_feature(
        action.astype(np.float32).tobytes()
    )
    feature["steps/action_gripper_binary"] = _bytes_feature(
        action_gripper_binary.astype(np.float32).tobytes()
    )

    feature["episode_metadata/exterior_view_count"] = _int64_feature(exterior_view_count)
    feature["episode_metadata/segment_count"] = _int64_feature(segment_count)

    feature["episode_metadata/segment_starts"] = _bytes_feature(
        segment_starts.astype(np.int32).tobytes()
    )
    feature["episode_metadata/segment_ends"] = _bytes_feature(
        segment_ends.astype(np.int32).tobytes()
    )
    feature["episode_metadata/segment_hold_actions"] = _bytes_feature(
        segment_hold_actions.astype(np.float32).tobytes()
    )
    feature["episode_metadata/segment_hold_action_gripper_binary"] = _bytes_feature(
        segment_hold_action_gripper_binary.astype(np.float32).tobytes()
    )

    feature["episode_metadata/segment_instruction_pools"] = _bytes_list_feature(
        [s.encode("utf-8") for s in segment_instruction_pools]
    )

    feature["episode_metadata/file_path"] = _bytes_feature(file_path.encode("utf-8"))
    feature["episode_metadata/recording_folderpath"] = _bytes_feature(
        recording_folderpath.encode("utf-8")
    )

    # Camera calibration — per-slot fields matching isaaclab_rlds schema.
    for i, (ixt, ixt_hw, ext) in enumerate(zip(camera_ixts, camera_ixt_hws, camera_exts)):
        feature[f"episode_metadata/camera_ixt_{i}"]    = _bytes_feature(ixt.astype(np.float32).tobytes())
        feature[f"episode_metadata/camera_ixt_hw_{i}"] = _bytes_feature(ixt_hw.astype(np.int32).tobytes())
        feature[f"episode_metadata/camera_ext_{i}"]    = _bytes_feature(ext.astype(np.float32).tobytes())
    feature["episode_metadata/robot_base_pose_w"] = _bytes_feature(robot_base_pose_w.astype(np.float32).tobytes())

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ---------------------------------------------------------------------------
# Per-NPZ worker function (runs in subprocess via spawn)
# ---------------------------------------------------------------------------

def _process_one_libero_npz(work_item: tuple) -> tuple[bytes | None, str | None, int]:
    """Process one LIBERO *_success.npz and return (serialised_bytes, error_msg).

    All heavy imports are done inside this function so spawned subprocesses
    start clean without a GPU context.
    """
    (
        npz_path_str,
        data_cfg_kwargs,
        image_size,
        max_segments,
        pad_hold_steps,
    ) = work_item

    import pathlib as _pl

    # Re-add paths so imports work in spawned subprocesses.
    _this = _pl.Path(__file__).parent
    _d = str(_this.parent / "isaaclab_rlds")
    if _d not in sys.path:
        sys.path.insert(0, _d)

    from npz_trajectory import (
        DataCfg as _DataCfg,
        _build_action_array as _baa,
        _build_state_array as _bsa,
    )
    from dataset_builder import MAX_BASE_VIEWS as _MAX_BASE_VIEWS

    npz_path = _pl.Path(npz_path_str)
    data_cfg = _DataCfg(**data_cfg_kwargs)

    # Load NPZ.
    try:
        with np.load(npz_path, allow_pickle=True) as _f:
            traj = {k: _f[k] for k in _f.files}
    except Exception as exc:
        return None, f"load_error({npz_path.name}): {exc}", 0

    try:
        # ---- Images -------------------------------------------------------
        front_raw = traj.get("sensor/rgbs/front")
        wrist_raw = traj.get("sensor/rgbs/wrist")
        if front_raw is None:
            raise KeyError("Missing 'sensor/rgbs/front'.")
        if wrist_raw is None:
            raise KeyError("Missing 'sensor/rgbs/wrist'.")

        front_frames = np.asarray(front_raw)   # (T, H, W, 3)
        wrist_frames = np.asarray(wrist_raw)   # (T, H, W, 3)
        T = front_frames.shape[0]

        # Slot 0 = front view; slots 1-3 = empty padding bytes.
        exterior_jpegs: list[list[bytes]] = []
        exterior_jpegs.append(
            [_resize_and_encode_jpeg(front_frames[t], image_size) for t in range(T)]
        )
        for _ in range(_MAX_BASE_VIEWS - 1):
            exterior_jpegs.append([b""] * T)

        wrist_jpegs = [_resize_and_encode_jpeg(wrist_frames[t], image_size) for t in range(T)]

        # ---- Language instruction ------------------------------------------
        raw_lang = traj.get("meta/language")
        if raw_lang is None:
            raise KeyError("Missing 'meta/language'.")
        language = str(np.asarray(raw_lang).flat[0])

        # ---- State / Action -----------------------------------------------
        state = _bsa(traj, data_cfg)   # (T, state_dim)
        action = _baa(traj, data_cfg)  # (T, action_dim_raw)

        # gripper_action_type='target' produces (T, arm+1) because the LIBERO
        # source only stores finger1.  When gripper_1d=False we tile to (T, arm+2)
        # to match IsaacLab's per-finger format.
        if data_cfg.gripper_action_type == "target" and not data_cfg.gripper_1d:
            g = action[:, -1:]  # (T, 1)
            action = np.concatenate([action[:, :-1], g, g], axis=-1)

        action_dim = action.shape[1]
        gripper_dim = 1 if (data_cfg.gripper_1d or data_cfg.gripper_action_type == "cmd") else 2
        arm_is_delta = (data_cfg.arm_action_type == "cmd") or data_cfg.make_target_residual

        # Per-step ±1 gripper intent (independent of gripper_action_type).
        gripper_binary = _build_gripper_binary(traj, gripper_dim=gripper_dim)  # (T, gripper_dim)

        # ---- Optional hold-pad frames at episode end ----------------------
        if pad_hold_steps > 0:
            hold = _build_hold_action(action, T, action_dim, gripper_dim, arm_is_delta)
            hold_state = state[-1:].repeat(pad_hold_steps, axis=0)
            hold_action = np.tile(hold, (pad_hold_steps, 1))
            hold_gripper_binary = np.tile(gripper_binary[-1:], (pad_hold_steps, 1))
            state = np.concatenate([state, hold_state], axis=0)
            action = np.concatenate([action, hold_action], axis=0)
            gripper_binary = np.concatenate([gripper_binary, hold_gripper_binary], axis=0)
            # Repeat last real frame for front; keep empty bytes for padding slots.
            exterior_jpegs[0] = exterior_jpegs[0] + [exterior_jpegs[0][-1]] * pad_hold_steps
            for slot in range(1, _MAX_BASE_VIEWS):
                exterior_jpegs[slot] = exterior_jpegs[slot] + [b""] * pad_hold_steps
            wrist_jpegs = wrist_jpegs + [wrist_jpegs[-1]] * pad_hold_steps
            T = state.shape[0]

        # ---- Segments: single full-episode segment ------------------------
        seg_starts = np.zeros(max_segments, dtype=np.int32)
        seg_ends = np.zeros(max_segments, dtype=np.int32)
        seg_hold_actions = np.zeros((max_segments, action_dim), dtype=np.float32)
        seg_hold_gripper_binary = np.zeros((max_segments, gripper_dim), dtype=np.float32)
        seg_pools: list[str] = [""] * max_segments

        seg_starts[0] = 0
        seg_ends[0] = T
        seg_hold_actions[0] = _build_hold_action(action, T, action_dim, gripper_dim, arm_is_delta)
        seg_hold_gripper_binary[0] = gripper_binary[T - 1]
        seg_pools[0] = language
        # Slots 1..max_segments-1 remain zeros / empty strings.

        # ---- Camera calibration -------------------------------------------
        # Slot 0 = front camera; slots 1-3 = identity padding.
        if "sensor/ixts/front" not in traj:
            raise KeyError("Missing 'sensor/ixts/front' in NPZ.")
        if "sensor/exts/front" not in traj:
            raise KeyError("Missing 'sensor/exts/front' in NPZ.")
        slot0_ixt = np.asarray(traj["sensor/ixts/front"][0]).flatten().astype(np.float32)
        slot0_ixt_hw = np.array(front_frames[0].shape[:2], dtype=np.int32)
        slot0_ext = np.asarray(traj["sensor/exts/front"][0]).flatten().astype(np.float32)

        pad_ixt = np.eye(3, dtype=np.float32).flatten()
        pad_ixt_hw = np.zeros(2, dtype=np.int32)
        pad_ext = np.eye(4, dtype=np.float32).flatten()
        camera_ixts    = [slot0_ixt]    + [pad_ixt    for _ in range(_MAX_BASE_VIEWS - 1)]
        camera_ixt_hws = [slot0_ixt_hw] + [pad_ixt_hw for _ in range(_MAX_BASE_VIEWS - 1)]
        camera_exts    = [slot0_ext]    + [pad_ext    for _ in range(_MAX_BASE_VIEWS - 1)]

        if "meta/robot_base_pose_w" not in traj:
            raise KeyError("Missing 'meta/robot_base_pose_w' in NPZ.")
        robot_base_pose_w = np.asarray(traj["meta/robot_base_pose_w"]).flatten().astype(np.float32)

        # ---- Serialise ----------------------------------------------------
        serialised = _make_example(
            exterior_jpegs=exterior_jpegs,
            wrist_jpegs=wrist_jpegs,
            state=state,
            action=action,
            action_gripper_binary=gripper_binary,
            exterior_view_count=1,
            segment_count=1,
            segment_starts=seg_starts,
            segment_ends=seg_ends,
            segment_hold_actions=seg_hold_actions,
            segment_hold_action_gripper_binary=seg_hold_gripper_binary,
            segment_instruction_pools=seg_pools,
            file_path=str(npz_path),
            recording_folderpath=str(npz_path.parent),
            camera_ixts=camera_ixts,
            camera_ixt_hws=camera_ixt_hws,
            camera_exts=camera_exts,
            robot_base_pose_w=robot_base_pose_w,
        )
        return serialised, None, int(T)

    except Exception:
        return None, f"process_error({npz_path.name}):\n{traceback.format_exc()}", 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LIBERO *_success.npz trajectories to RLDS TFRecords."
    )
    p.add_argument("--data_root", required=True,
                   help="Root directory; scans recursively for **/traj/*_success.npz.")
    p.add_argument("--output_dir", required=True,
                   help="Destination directory for TFRecord shards.")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--pad_hold_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_shards", type=int, default=128)
    p.add_argument(
        "--max_segments", type=int, default=DEFAULT_MAX_SEGMENTS,
        help=(
            f"Segment slot count written into each TFRecord example "
            f"(default {DEFAULT_MAX_SEGMENTS}; must match IsaacLab data when merging)."
        ),
    )
    # DataCfg flags.
    p.add_argument("--state_repr", default="joint", choices=["ee", "joint"])
    p.add_argument("--state_gripper_1d", action="store_true")
    p.add_argument("--arm_repr", default="joint", choices=["ee", "joint"])
    p.add_argument(
        "--arm_action_type", default="actual", choices=["cmd", "actual"],
        help="'target' is not supported for LIBERO (OSC has no arm joint targets).",
    )
    p.add_argument(
        "--gripper_action_type", default="actual", choices=["actual", "target"],
        help=(
            "'actual' keeps both fingers (T,2) or pools to (T,1) with --gripper_1d. "
            "'target' uses finger1 only; off tiles to (T,2), on keeps (T,1). "
            "'cmd' not supported."
        ),
    )
    p.add_argument("--gripper_1d", action="store_true",
                   help="Pool 2D per-finger gripper to 1D mean.")
    p.add_argument("--make_target_residual", action="store_true")
    # Demo selection.
    p.add_argument(
        "--demos_per_task", type=int, default=0,
        help=(
            "Maximum demos to convert per task (0 = all). "
            "Demos are sampled uniformly by demo_idx with --split_seed. "
            "Run twice with different --output_dir to create separate subsets."
        ),
    )
    p.add_argument("--split_seed", type=int, default=42,
                   help="Random seed for demo sampling when --demos_per_task > 0.")
    # Task filtering (by task-dir name = npz.parent.parent.parent.name).
    p.add_argument(
        "--include_tasks", nargs="*", default=None,
        help=(
            "If set, keep only tasks whose dir name contains ANY of these substrings. "
            "Applied before --exclude_tasks and before --demos_per_task."
        ),
    )
    p.add_argument(
        "--exclude_tasks", nargs="*", default=None,
        help="Drop tasks whose dir name contains ANY of these substrings.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg_kwargs = dict(
        state_repr=args.state_repr,
        state_gripper_1d=args.state_gripper_1d,
        arm_repr=args.arm_repr,
        arm_action_type=args.arm_action_type,
        gripper_action_type=args.gripper_action_type,
        gripper_1d=args.gripper_1d,
        make_target_residual=args.make_target_residual,
    )
    data_cfg = DataCfg(**data_cfg_kwargs)
    _validate_libero_cfg(data_cfg)

    # ------------------------------------------------------------------
    # Discover NPZ files (*_success.npz only, under **/traj/).
    # ------------------------------------------------------------------
    print(f"Scanning {data_root} for **/traj/*_success.npz …")
    npz_files = sorted(data_root.glob("**/traj/*_success.npz"))
    if not npz_files:
        raise RuntimeError(f"No *_success.npz files found under {data_root}.")
    print(f"Found {len(npz_files)} success trajectories.")

    # ------------------------------------------------------------------
    # Optional: filter by task-dir name (PnP-only selection, etc.).
    # task_name = npz.parent.parent.parent.name
    # ------------------------------------------------------------------
    if args.include_tasks or args.exclude_tasks:
        inc = args.include_tasks or []
        exc = args.exclude_tasks or []
        kept = []
        dropped_tasks: set[str] = set()
        for p in npz_files:
            task_name = p.parent.parent.parent.name
            if inc and not any(s in task_name for s in inc):
                dropped_tasks.add(task_name)
                continue
            if exc and any(s in task_name for s in exc):
                dropped_tasks.add(task_name)
                continue
            kept.append(p)
        if not kept:
            raise RuntimeError(
                f"Task filter removed all trajectories (include={inc}, exclude={exc})."
            )
        print(
            f"Task filter (include={inc}, exclude={exc}): kept {len(kept)} / "
            f"{len(npz_files)} trajectories; dropped {len(dropped_tasks)} task(s)."
        )
        npz_files = kept

    # ------------------------------------------------------------------
    # Optional: limit demos per task.
    # Path structure: <data_root>/<task_name>/demo_<idx>/traj/<name>.npz
    # task_name = npz.parent.parent.parent.name
    # demo_idx  = int from npz.parent.parent.name  ("demo_N")
    # ------------------------------------------------------------------
    if args.demos_per_task > 0:
        from collections import defaultdict
        rng = np.random.RandomState(args.split_seed)
        groups: dict[str, list[pathlib.Path]] = defaultdict(list)
        for p in npz_files:
            task_name = p.parent.parent.parent.name
            groups[task_name].append(p)

        selected: list[pathlib.Path] = []
        for task_name, paths in sorted(groups.items()):
            n_pick = min(args.demos_per_task, len(paths))
            if n_pick < args.demos_per_task:
                print(
                    f"[warn] Task '{task_name}': only {len(paths)} demos available, "
                    f"picking all {n_pick} (requested {args.demos_per_task})."
                )
            chosen = rng.choice(len(paths), n_pick, replace=False)
            selected.extend(paths[i] for i in sorted(chosen))

        print(
            f"demos_per_task={args.demos_per_task} seed={args.split_seed}: "
            f"selected {len(selected)} / {len(npz_files)} trajectories "
            f"across {len(groups)} tasks."
        )
        npz_files = sorted(selected)

    # ------------------------------------------------------------------
    # Infer state_dim / action_dim from first readable NPZ.
    # ------------------------------------------------------------------
    state_dim = action_dim = None
    for npz_path in npz_files:
        try:
            with np.load(npz_path, allow_pickle=True) as _f:
                probe = {k: _f[k] for k in _f.files}
            state_dim = data_cfg.state_dim(probe)
            # data_cfg.action_dim already returns the intended final dim (e.g. arm+2
            # when gripper_1d=False), so it matches after the gripper-tile step.
            action_dim = data_cfg.action_dim(probe)
            break
        except Exception:
            continue
    if state_dim is None or action_dim is None:
        raise RuntimeError("Could not determine state_dim/action_dim from any NPZ.")
    gripper_dim = 1 if (data_cfg.gripper_1d or data_cfg.gripper_action_type == "cmd") else 2
    print(
        f"state_dim={state_dim}  action_dim={action_dim}  gripper_dim={gripper_dim}  "
        f"max_segments={args.max_segments}"
    )

    # ------------------------------------------------------------------
    # Build work items for subprocess workers.
    # ------------------------------------------------------------------
    work_items = [
        (str(p), data_cfg_kwargs, args.image_size, args.max_segments, args.pad_hold_steps)
        for p in npz_files
    ]

    # ------------------------------------------------------------------
    # Open TFRecord shard writers.
    # ------------------------------------------------------------------
    import tensorflow as tf

    # Never create more shards than episodes (otherwise --include_tasks /
    # --exclude_tasks / --demos_per_task leave trailing empty shard files).
    if args.num_shards > len(npz_files):
        print(
            f"[shards] clamping num_shards {args.num_shards} -> {len(npz_files)} "
            f"(== episodes) to avoid empty shards."
        )
        args.num_shards = max(1, len(npz_files))

    shard_tmpl = str(
        output_dir / f"libero_rlds-train.tfrecord-{{:05d}}-of-{args.num_shards:05d}"
    )
    writers = [tf.io.TFRecordWriter(shard_tmpl.format(i)) for i in range(args.num_shards)]

    # ------------------------------------------------------------------
    # Launch workers and write results round-robin.
    # ------------------------------------------------------------------
    total_episodes = 0
    total_steps = 0
    errors = 0
    shard_cursor = 0

    ctx = _mp.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.num_workers), mp_context=ctx
    ) as executor:
        futures = {
            executor.submit(_process_one_libero_npz, wi): wi[0]
            for wi in work_items
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Converting",
        ):
            npz_path_str = futures[future]
            try:
                serialised, err, n_steps = future.result()
            except Exception as exc:
                print(f"[error] {pathlib.Path(npz_path_str).name}: {exc}")
                errors += 1
                continue

            if serialised is None:
                print(f"[warn] {err}")
                errors += 1
                continue

            writers[shard_cursor % args.num_shards].write(serialised)
            shard_cursor += 1
            total_episodes += 1
            total_steps += n_steps

    for w in writers:
        w.close()

    # ------------------------------------------------------------------
    # Write dataset_info.json.
    # ------------------------------------------------------------------
    info = DatasetInfo(
        state_dim=state_dim,
        action_dim=action_dim,
        max_segments=args.max_segments,
        max_base_views=MAX_BASE_VIEWS,  # schema always has 4 slots; exterior_view_count=1
        total_episodes=total_episodes,
        total_steps=total_steps,
        gripper_dim=gripper_dim,
    )
    info.save(output_dir)

    print(
        f"\nDone.  episodes={total_episodes}  errors={errors}  "
        f"shards={args.num_shards}  output={output_dir}"
    )


if __name__ == "__main__":
    main()
