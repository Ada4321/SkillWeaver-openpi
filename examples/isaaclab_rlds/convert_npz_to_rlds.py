"""
Convert IsaacLab NPZ trajectories to IsaacLab RLDS TFRecord dataset.

Each NPZ file becomes one tf.train.Example (= one episode) written into one
of --num_shards TFRecord files.  The schema is defined in dataset_builder.py.

Usage
-----
python examples/isaaclab_rlds/convert_npz_to_rlds.py \
    --data_root /data/isaaclab_trajs \
    --output_dir /data/rlds_out \
    --scene_desc_root /data/scenes \
    [--instruction_manifest_path /data/instruction_pool_manifest.jsonl] \
    [--state_repr joint] [--state_gripper_1d] \
    [--arm_repr joint] [--arm_action_type actual] \
    [--gripper_action_type actual] [--gripper_1d] \
    [--num_workers 16] [--num_shards 512]

Output layout
-------------
<output_dir>/
  dataset_info.json
  isaaclab_rlds-train.tfrecord-00000-of-00512
  isaaclab_rlds-train.tfrecord-00001-of-00512
  ...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io as _io
import json
import multiprocessing as _mp
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
from tqdm import tqdm

# Make the sibling modules (npz_trajectory, dataset_builder) importable when running as __main__ from any CWD.
_RLDS_DIR = str(pathlib.Path(__file__).parent)
if _RLDS_DIR not in sys.path:
    sys.path.insert(0, _RLDS_DIR)

from npz_trajectory import (  # noqa: E402
    DataCfg,
    MissingAugmentDataError,
    TrajRecord,
    _extract_seq_split_specs,
    _load_instruction_pool_rows,
    _resolve_task_entry,
    gather_traj_records,
)

from dataset_builder import (  # noqa: E402  (same dir on sys.path via __file__)
    MAX_BASE_VIEWS,
    DatasetInfo,
    max_segments_from_max_nodes,
)

# ---------------------------------------------------------------------------
# Image helpers (PIL only — no TF in module scope)
# ---------------------------------------------------------------------------

def _resize_and_encode_jpeg(frame: np.ndarray, size: int, quality: int = 90) -> bytes:
    """Letterbox HxWxC uint8 numpy array into size×size and JPEG-encode.

    Aspect ratio is preserved: the frame is scaled until its long edge matches
    `size`, then centred on a black square. A square input is therefore just a
    plain resize (bit-identical to the previous `img.resize((size, size))`), so
    only non-square suites see any change — e.g. simpler's 640x480 front camera
    becomes 224x168 content with 28 black rows top and bottom instead of being
    stretched 33% vertically.

    This is the same `resize_with_pad` the training-time `ResizeImages` transform
    and the eval-time policy client use, so all three stages agree.
    """
    from openpi_client import image_tools
    from PIL import Image

    padded = image_tools.resize_with_pad(
        frame.astype(np.uint8)[None], size, size, method=Image.LANCZOS
    )[0]
    buf = _io.BytesIO()
    Image.fromarray(padded).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _build_chunk_ranges(
    traj: dict[str, Any],
    split_specs: list[list[int]],
) -> list[tuple[int, int, int, int]]:
    """Return list of (frame_start, frame_end, node_start, node_end) tuples."""
    node_ends_raw = traj.get("meta/node_end_indices_exclusive")
    if node_ends_raw is None:
        return []
    node_ends = np.asarray(node_ends_raw, dtype=np.int64).reshape(-1)
    n_nodes = int(node_ends.shape[0])
    if n_nodes <= 0:
        return []
    node_starts_list = [0] + [int(x) for x in node_ends[:-1].tolist()]

    chunks: list[tuple[int, int, int, int]] = []
    for split in split_specs:
        valid_cuts = sorted({int(x) for x in split if 0 <= int(x) < (n_nodes - 1)})
        group_start_node = 0
        for cut in valid_cuts + [n_nodes - 1]:
            group_end_node = int(cut)
            fs = int(node_starts_list[group_start_node])
            fe = int(node_ends[group_end_node])
            if fe > fs:
                chunks.append((fs, fe, group_start_node, group_end_node))
            group_start_node = group_end_node + 1
    return chunks


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
    exterior_jpegs: list[list[bytes]],   # [slot_idx][frame_idx]
    wrist_jpegs: list[bytes],            # [frame_idx]
    state: np.ndarray,                   # (T, state_dim)
    action: np.ndarray,                  # (T, action_dim)
    action_gripper_binary: np.ndarray,   # (T, gripper_dim) float32 ±1
    exterior_view_count: int,
    segment_count: int,
    segment_starts: np.ndarray,          # (max_segments,) int32
    segment_ends: np.ndarray,            # (max_segments,) int32
    segment_hold_actions: np.ndarray,    # (max_segments, action_dim) float32
    segment_hold_action_gripper_binary: np.ndarray,  # (max_segments, gripper_dim) float32 ±1
    segment_instruction_pools: list[str],           # len == max_segments, each \n-joined
    segment_instruction_pools_attr_aug: list[str],  # len == max_segments, each \n-joined
    file_path: str,
    recording_folderpath: str,
    camera_ixts: list[np.ndarray],       # MAX_BASE_VIEWS × (9,)  float32 — K matrix flattened
    camera_ixt_hws: list[np.ndarray],   # MAX_BASE_VIEWS × (2,)  int32  — (H, W) at collection res
    camera_exts: list[np.ndarray],      # MAX_BASE_VIEWS × (16,) float32 — w2c 4×4 flattened
    robot_base_pose_w: np.ndarray,      # (16,) float32 — panda_link0→world 4×4
) -> bytes:
    """Build and serialise a tf.train.Example for one episode."""
    import tensorflow as tf

    feature: dict = {}

    # Per-step images.
    for i, frames in enumerate(exterior_jpegs):
        feature[f"steps/observation/exterior_image_{i}"] = _bytes_list_feature(frames)
    feature["steps/observation/wrist_image"] = _bytes_list_feature(wrist_jpegs)

    # Per-step float arrays as raw bytes.
    feature["steps/observation/state"] = _bytes_feature(
        state.astype(np.float32).tobytes()
    )
    feature["steps/action"] = _bytes_feature(
        action.astype(np.float32).tobytes()
    )
    feature["steps/action_gripper_binary"] = _bytes_feature(
        action_gripper_binary.astype(np.float32).tobytes()
    )

    # Episode metadata scalars.
    feature["episode_metadata/exterior_view_count"] = _int64_feature(exterior_view_count)
    feature["episode_metadata/segment_count"] = _int64_feature(segment_count)

    # Episode metadata arrays.
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

    # Per-segment instruction pools (each pool = \n-joined candidates).
    feature["episode_metadata/segment_instruction_pools"] = _bytes_list_feature(
        [s.encode("utf-8") for s in segment_instruction_pools]
    )
    feature["episode_metadata/segment_instruction_pools_attr_aug"] = _bytes_list_feature(
        [s.encode("utf-8") for s in segment_instruction_pools_attr_aug]
    )

    # String scalars.
    feature["episode_metadata/file_path"] = _bytes_feature(file_path.encode("utf-8"))
    feature["episode_metadata/recording_folderpath"] = _bytes_feature(
        recording_folderpath.encode("utf-8")
    )

    # Camera calibration — per exterior camera slot.
    for i, (ixt, hw, ext) in enumerate(zip(camera_ixts, camera_ixt_hws, camera_exts)):
        feature[f"episode_metadata/camera_ixt_{i}"]    = _bytes_feature(ixt.astype(np.float32).tobytes())
        feature[f"episode_metadata/camera_ixt_hw_{i}"] = _bytes_feature(hw.astype(np.int32).tobytes())
        feature[f"episode_metadata/camera_ext_{i}"]    = _bytes_feature(ext.astype(np.float32).tobytes())
    feature["episode_metadata/robot_base_pose_w"] = _bytes_feature(robot_base_pose_w.astype(np.float32).tobytes())

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ---------------------------------------------------------------------------
# Gripper binary intent builder
# ---------------------------------------------------------------------------

def _build_gripper_binary(
    traj: dict[str, Any],
    *,
    gripper_dim: int,
    eps: float = 1e-4,
    init: float = -1.0,
) -> np.ndarray:
    """Per-step ±1 gripper intent recovered from action/gripper_targets vs state/gripper_pos.

    Mode 1 (RL skill, grasp_hold): target − state ≡ act_EMA[6]·0.01 ≡ gripper_cmd_delta.
    Mode 0 (cuRobo motion planning): target − state = (frozen move-target) − q_t,
                                     sign matches grasping intent at move() call time.
    Both modes: sign correctly encodes open(-1) / close(+1) intent.
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
    delta = tgt.mean(axis=-1) - sta.mean(axis=-1)        # (T,)

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
# Hold-action builder
# ---------------------------------------------------------------------------

def _build_hold_action(
    action: np.ndarray,  # (T, action_dim)
    frame_end: int,
    action_dim: int,
    gripper_dim: int,
    arm_is_delta: bool,
) -> np.ndarray:
    """Build the hold action for a segment ending at frame_end (exclusive)."""
    hold = np.zeros(action_dim, dtype=np.float32)
    if not arm_is_delta:
        hold[:] = action[frame_end - 1]
    # Gripper always takes the real last value.
    hold[-gripper_dim:] = action[frame_end - 1, -gripper_dim:]
    return hold


def _expect_instruction_group(
    raw: Any,
    *,
    field_name: str,
    npz_path: pathlib.Path,
) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        raise MissingAugmentDataError(
            f"Invalid type for {field_name} (expected dict with other/attr_aug) "
            f"npz={npz_path}: {type(raw).__name__}"
        )

    out: dict[str, list[str]] = {}
    for group_name in ("other", "attr_aug"):
        vals = raw.get(group_name, [])
        if vals is None:
            vals = []
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            raise MissingAugmentDataError(
                f"Invalid type for {field_name}.{group_name} (expected list[str]) "
                f"npz={npz_path}: {type(vals).__name__}"
            )
        group: list[str] = []
        for idx, one in enumerate(vals):
            if not isinstance(one, str):
                raise MissingAugmentDataError(
                    f"Invalid type for {field_name}.{group_name}[{idx}] "
                    f"(expected str) npz={npz_path}: {type(one).__name__}"
                )
            if one:
                group.append(one)
        out[group_name] = group

    if not out["other"] and not out["attr_aug"]:
        raise MissingAugmentDataError(f"Empty instruction group {field_name} for npz={npz_path}")
    return out


def _expect_subgoal_instruction_group_pool(
    raw: Any,
    *,
    npz_path: pathlib.Path,
) -> list[dict[str, list[str]]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MissingAugmentDataError(
            f"Invalid type for subgoal_instructions (expected list[dict]) "
            f"npz={npz_path}: {type(raw).__name__}"
        )
    return [
        _expect_instruction_group(
            one,
            field_name=f"subgoal_instructions[{idx}]",
            npz_path=npz_path,
        )
        for idx, one in enumerate(raw)
    ]


def _expect_chunk_candidate_group_pool(
    raw: Any,
    *,
    npz_path: pathlib.Path,
) -> dict[str, dict[str, list[str]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MissingAugmentDataError(
            f"Invalid type for chunk_candidates (expected dict[str, dict]) "
            f"npz={npz_path}: {type(raw).__name__}"
        )

    out: dict[str, dict[str, list[str]]] = {}
    for key, vals in raw.items():
        if not isinstance(key, str):
            raise MissingAugmentDataError(
                f"Invalid type for chunk_candidates key (expected str) "
                f"npz={npz_path}: {type(key).__name__}"
            )
        out[key] = _expect_instruction_group(
            vals,
            field_name=f"chunk_candidates[{key}]",
            npz_path=npz_path,
        )
    return out


def _join_instruction_group(group: dict[str, list[str]], group_name: str) -> str:
    return "\n".join(group.get(group_name, []))


def _build_segment_pools_for_episode(
    pool_row: dict,
    n_nodes: int,
    chunks: list[tuple[int, int, int, int]],
    max_segments: int,
    npz_path: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Return (seg_pools, seg_pools_attr_aug), each list of length max_segments.

    Slot 0 is the full-episode pool; slots 1..K mirror the chunks list (truncated
    if it exceeds max_segments-1). Padding slots are empty strings.
    """
    overall_instructions = _expect_instruction_group(
        pool_row.get("overall_instructions"),
        field_name="overall_instructions",
        npz_path=npz_path,
    )
    subgoal_instructions = _expect_subgoal_instruction_group_pool(
        pool_row.get("subgoal_instructions"),
        npz_path=npz_path,
    )
    chunk_candidates = _expect_chunk_candidate_group_pool(
        pool_row.get("chunk_candidates"),
        npz_path=npz_path,
    )

    seg_pools: list[str] = [""] * max_segments
    seg_pools_attr_aug: list[str] = [""] * max_segments

    def _set_segment_pool(slot: int, group: dict[str, list[str]]) -> None:
        seg_pools[slot] = _join_instruction_group(group, "other")
        seg_pools_attr_aug[slot] = _join_instruction_group(group, "attr_aug")

    full_span_key = f"0-{n_nodes - 1}"
    if n_nodes > 1 and full_span_key in chunk_candidates:
        full_group = chunk_candidates[full_span_key]
    else:
        full_group = overall_instructions
    _set_segment_pool(0, full_group)

    for i, (_fs, _fe, ns, ne) in enumerate(chunks):
        slot = i + 1
        if slot >= max_segments:
            break
        if ns == ne:
            if ns < len(subgoal_instructions):
                _set_segment_pool(slot, subgoal_instructions[ns])
            else:
                _set_segment_pool(slot, full_group)
        else:
            key = f"{ns}-{ne}"
            if key in chunk_candidates:
                _set_segment_pool(slot, chunk_candidates[key])
            else:
                _set_segment_pool(slot, full_group)

    return seg_pools, seg_pools_attr_aug


# ---------------------------------------------------------------------------
# Per-NPZ worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_one_npz(work_item: tuple) -> tuple[bytes | None, str | None, int]:
    """Process one NPZ file and return (serialised_example_bytes, error_msg, n_steps).

    All heavy imports are done inside this function so the spawned subprocess
    starts clean without a TF GPU context.
    """
    (
        npz_path_str,
        pool_row_json,
        split_specs_json,
        data_cfg_kwargs,
        image_size,
        max_base_views,
        max_segments,
        pad_hold_steps,
        no_wrist,
    ) = work_item

    # Re-add path so imports work in spawned processes.
    import pathlib as _pl
    _rdir = str(_pl.Path(__file__).parent)
    if _rdir not in sys.path:
        sys.path.insert(0, _rdir)

    from npz_trajectory import (
        DataCfg as _DataCfg,
        _apply_target_residual as _art,
        _build_action_array as _baa,
        _build_state_array as _bsa,
        _get_base_views as _gbv,
        _load_traj as _lt,
    )
    from dataset_builder import MAX_BASE_VIEWS as _MAX_BASE_VIEWS

    npz_path = _pl.Path(npz_path_str)
    pool_row: dict = json.loads(pool_row_json)
    split_specs: list[list[int]] = json.loads(split_specs_json)
    data_cfg = _DataCfg(**data_cfg_kwargs)

    try:
        traj = _lt(npz_path)
    except Exception as exc:
        return None, f"load_error({npz_path.name}): {exc}", 0

    try:
        # ---- Images -------------------------------------------------------
        base_view_keys = _gbv(traj)
        wrist_key = next(
            (k for k in traj if k.startswith("sensor/rgbs/") and "wrist" in k), None
        )
        if not base_view_keys:
            raise KeyError("Missing base camera images")
        if wrist_key is None and not no_wrist:
            raise KeyError(
                "Missing wrist camera images. Pass --no_wrist for embodiments that have "
                "no wrist camera (e.g. the simpler WidowX suite); the wrist field is then "
                "filled with a blank placeholder and dataset_info.json records has_wrist=false."
            )

        n_views = min(len(base_view_keys), max_base_views)

        # Raw images: shape (T, H, W, C) per key.
        raw_base = [np.asarray(traj[k]) for k in base_view_keys[:n_views]]
        T = raw_base[0].shape[0]

        # Encode all frames.
        exterior_jpegs: list[list[bytes]] = []
        for slot in range(_MAX_BASE_VIEWS):
            if slot < n_views:
                frames = raw_base[slot]
                exterior_jpegs.append(
                    [_resize_and_encode_jpeg(frames[t], image_size) for t in range(T)]
                )
            else:
                # Padding slot: empty bytes for every frame.
                exterior_jpegs.append([b""] * T)

        if wrist_key is None:
            # No wrist camera: one blank JPEG, reused for every frame. It must be a
            # decodable image (IsaacLabRldsDataset.decode_images runs tf.io.decode_image
            # on this field unconditionally) and the same size as the base views so the
            # batched shapes stay uniform. Training ignores it via use_wrist_camera=False.
            blank = _resize_and_encode_jpeg(
                np.zeros((image_size, image_size, 3), dtype=np.uint8), image_size
            )
            wrist_jpegs = [blank] * T
        else:
            raw_wrist = np.asarray(traj[wrist_key])
            wrist_jpegs = [_resize_and_encode_jpeg(raw_wrist[t], image_size) for t in range(T)]

        # ---- State / Action -----------------------------------------------
        state = _bsa(traj, data_cfg)   # (T, state_dim)
        action = _baa(traj, data_cfg)  # (T, action_dim)
        if data_cfg.make_target_residual:
            action = _art(action, traj, data_cfg)

        action_dim = action.shape[1]
        gripper_dim = 1 if (data_cfg.gripper_1d or data_cfg.gripper_action_type == "cmd") else 2
        arm_is_delta = (data_cfg.arm_action_type == "cmd") or data_cfg.make_target_residual

        # Per-step ±1 gripper intent (independent of gripper_action_type).
        gripper_binary = _build_gripper_binary(traj, gripper_dim=gripper_dim)  # (T, gripper_dim)

        # Optional hold-pad frames at episode end.
        if pad_hold_steps > 0:
            hold = _build_hold_action(action, T, action_dim, gripper_dim, arm_is_delta)
            hold_state = state[-1:].repeat(pad_hold_steps, axis=0)
            hold_action = np.tile(hold, (pad_hold_steps, 1))
            hold_gripper_binary = np.tile(gripper_binary[-1:], (pad_hold_steps, 1))
            state = np.concatenate([state, hold_state], axis=0)
            action = np.concatenate([action, hold_action], axis=0)
            gripper_binary = np.concatenate([gripper_binary, hold_gripper_binary], axis=0)
            for slot in range(_MAX_BASE_VIEWS):
                exterior_jpegs[slot] = exterior_jpegs[slot] + [exterior_jpegs[slot][-1]] * pad_hold_steps
            wrist_jpegs = wrist_jpegs + [wrist_jpegs[-1]] * pad_hold_steps
            T = state.shape[0]

        # ---- Segments -----------------------------------------------------
        chunks = _build_chunk_ranges(traj, split_specs)
        n_nodes_raw = traj.get("meta/node_end_indices_exclusive")
        n_nodes = int(np.asarray(n_nodes_raw).reshape(-1).shape[0]) if n_nodes_raw is not None else 1

        seg_pools, seg_pools_attr_aug = _build_segment_pools_for_episode(
            pool_row=pool_row,
            n_nodes=n_nodes,
            chunks=chunks,
            max_segments=max_segments,
            npz_path=npz_path,
        )

        seg_starts = np.zeros(max_segments, dtype=np.int32)
        seg_ends = np.zeros(max_segments, dtype=np.int32)
        seg_hold_actions = np.zeros((max_segments, action_dim), dtype=np.float32)
        seg_hold_gripper_binary = np.zeros((max_segments, gripper_dim), dtype=np.float32)

        # Segment 0: full episode.
        seg_starts[0] = 0
        seg_ends[0] = T
        seg_hold_actions[0] = _build_hold_action(action, T, action_dim, gripper_dim, arm_is_delta)
        seg_hold_gripper_binary[0] = gripper_binary[T - 1]

        # Segments 1..K: from split_specs.
        valid_chunks = 0
        for i, (fs, fe, _ns, _ne) in enumerate(chunks):
            slot = i + 1
            if slot >= max_segments:
                break
            seg_starts[slot] = fs
            seg_ends[slot] = fe
            seg_hold_actions[slot] = _build_hold_action(action, fe, action_dim, gripper_dim, arm_is_delta)
            seg_hold_gripper_binary[slot] = gripper_binary[fe - 1]
            valid_chunks += 1

        segment_count = 1 + valid_chunks

        # ---- Camera calibration (IsaacLab: robot base = world origin) ----
        cam_names = [k[len("sensor/rgbs/"):] for k in base_view_keys[:n_views]]
        camera_ixts:    list[np.ndarray] = []
        camera_ixt_hws: list[np.ndarray] = []
        camera_exts:    list[np.ndarray] = []
        for slot in range(_MAX_BASE_VIEWS):
            if slot < n_views:
                name = cam_names[slot]
                _ixt = traj.get(f"sensor/ixts/{name}")
                _ext = traj.get(f"sensor/exts/{name}")
                ixt = np.asarray(_ixt[0]).flatten().astype(np.float32) if _ixt is not None else np.eye(3, dtype=np.float32).flatten()
                ext = np.asarray(_ext[0]).flatten().astype(np.float32) if _ext is not None else np.eye(4, dtype=np.float32).flatten()
                hw  = np.array(raw_base[slot][0].shape[:2], dtype=np.int32)
            else:
                ixt = np.eye(3, dtype=np.float32).flatten()
                ext = np.eye(4, dtype=np.float32).flatten()
                hw  = np.zeros(2, dtype=np.int32)
            camera_ixts.append(ixt)
            camera_ixt_hws.append(hw)
            camera_exts.append(ext)
        robot_base_pose_w = np.eye(4, dtype=np.float32).flatten()  # panda_link0 == world origin in IsaacLab

        # ---- Serialise ----------------------------------------------------
        serialised = _make_example(
            exterior_jpegs=exterior_jpegs,
            wrist_jpegs=wrist_jpegs,
            state=state,
            action=action,
            action_gripper_binary=gripper_binary,
            exterior_view_count=n_views,
            segment_count=segment_count,
            segment_starts=seg_starts,
            segment_ends=seg_ends,
            segment_hold_actions=seg_hold_actions,
            segment_hold_action_gripper_binary=seg_hold_gripper_binary,
            segment_instruction_pools=seg_pools,
            segment_instruction_pools_attr_aug=seg_pools_attr_aug,
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
    p = argparse.ArgumentParser(description="Convert IsaacLab NPZ trajectories to RLDS TFRecords.")
    p.add_argument("--data_root", required=True, help="Root directory containing traj/*.npz files.")
    p.add_argument("--output_dir", required=True, help="Destination directory for train TFRecord shards.")
    p.add_argument(
        "--scene_desc_root", required=True,
        help="Root directory of the scene description JSON files (scene_XXXX.json).",
    )
    p.add_argument(
        "--instruction_manifest_path",
        default=None,
        help="Path to instruction_pool_manifest.jsonl (default: <data_root>/instruction_pool_manifest.jsonl).",
    )
    p.add_argument("--max_base_views", type=int, default=MAX_BASE_VIEWS)
    p.add_argument(
        "--max_nodes", type=int, default=10,
        help="Max task nodes; determines max_segments = 1 + max_nodes*(max_nodes+1)//2 (rounded up).",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument(
        "--no_wrist", action="store_true",
        help="Convert NPZs that have no wrist camera (e.g. the simpler WidowX suite). "
             "steps/observation/wrist_image is filled with a blank JPEG and "
             "dataset_info.json records has_wrist=false, which makes training configs "
             "with use_wrist_camera=True fail fast instead of consuming the placeholder.",
    )
    p.add_argument(
        "--robot", default=None, choices=["panda", "widowx"],
        help="Embodiment these NPZs were recorded with. Written to dataset_info.json and "
             "used at eval time to pick the forward-kinematics model and joint limits for "
             "the EEF metrics (openpi.eval.robot_fk). Omit for the Franka suites: a missing "
             "key reads back as 'panda', so existing shards keep working unchanged.",
    )
    p.add_argument("--pad_hold_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    # Train/val split.
    p.add_argument(
        "--val_ratio", type=float, default=0.0,
        help="Fraction of trajectories held out as val (0 = no split, all go to output_dir).",
    )
    p.add_argument(
        "--val_output_dir", default=None,
        help="Destination for val TFRecord shards. Defaults to <output_dir>_val when val_ratio > 0.",
    )
    p.add_argument("--split_seed", type=int, default=42,
                   help="Random seed for deterministic train/val split.")
    # DataCfg flags.
    p.add_argument("--state_repr", default="joint", choices=["ee", "joint"])
    p.add_argument("--state_gripper_1d", action="store_true")
    p.add_argument("--arm_repr", default="joint", choices=["ee", "joint"])
    p.add_argument("--arm_action_type", default="actual", choices=["cmd", "target", "actual"])
    p.add_argument("--gripper_action_type", default="actual", choices=["cmd", "target", "actual"])
    p.add_argument("--gripper_1d", action="store_true")
    p.add_argument("--make_target_residual", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    scene_desc_root = pathlib.Path(args.scene_desc_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_segments = max_segments_from_max_nodes(args.max_nodes)
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
    data_cfg.validate()

    # ------------------------------------------------------------------
    # Discover NPZ files.
    # ------------------------------------------------------------------
    print(f"Scanning {data_root} for trajectory NPZ files …")
    records: list[TrajRecord] = gather_traj_records(data_root)
    print(f"Found {len(records)} NPZ files.")

    # ------------------------------------------------------------------
    # Load instruction pool manifest.
    # ------------------------------------------------------------------
    manifest_path = (
        pathlib.Path(args.instruction_manifest_path)
        if args.instruction_manifest_path
        else data_root / "instruction_pool_manifest.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Instruction pool manifest not found: {manifest_path}\n"
            "Pass --instruction_manifest_path (one JSON row per trajectory with its instruction pools)."
        )
    pool_rows = _load_instruction_pool_rows(manifest_path)
    print(f"Loaded {len(pool_rows)} instruction pool rows.")

    # ------------------------------------------------------------------
    # Determine state_dim / action_dim from first readable NPZ.
    # ------------------------------------------------------------------
    state_dim = action_dim = None
    for rec in records:
        try:
            with np.load(rec.npz_path, allow_pickle=True) as d:
                probe = {k: d[k] for k in d.files}
            state_dim = data_cfg.state_dim(probe)
            action_dim = data_cfg.action_dim(probe)
            break
        except Exception:
            continue
    if state_dim is None or action_dim is None:
        raise RuntimeError("Could not determine state_dim/action_dim from any NPZ.")
    gripper_dim = 1 if (data_cfg.gripper_1d or data_cfg.gripper_action_type == "cmd") else 2
    print(f"state_dim={state_dim}, action_dim={action_dim}, gripper_dim={gripper_dim}, max_segments={max_segments}")

    # ------------------------------------------------------------------
    # Precompute split_specs per NPZ (scene JSON loads are cheap).
    # ------------------------------------------------------------------
    scene_cache: dict = {}
    tasks_cache: dict = {}

    def _get_split_specs(rec: TrajRecord) -> list[list[int]]:
        task_entry, _ = _resolve_task_entry(
            rec, scene_desc_root, "tasks", scene_cache, tasks_cache
        )
        return _extract_seq_split_specs(task_entry) if task_entry is not None else []

    # ------------------------------------------------------------------
    # Build work items for workers.
    # ------------------------------------------------------------------
    all_work_items: list[tuple] = []
    skipped_no_pool = 0
    for rec in records:
        key = str(rec.npz_path.resolve())
        pool_row = pool_rows.get(key) or pool_rows.get(str(rec.npz_path))
        if pool_row is None:
            skipped_no_pool += 1
            continue
        split_specs = _get_split_specs(rec)
        all_work_items.append((
            str(rec.npz_path),
            json.dumps(pool_row),
            json.dumps(split_specs),
            data_cfg_kwargs,
            args.image_size,
            args.max_base_views,
            max_segments,
            args.pad_hold_steps,
            args.no_wrist,
        ))
    if skipped_no_pool:
        print(f"[warn] Skipped {skipped_no_pool} NPZ files with no instruction pool row.")

    # ------------------------------------------------------------------
    # Train / val split.
    # ------------------------------------------------------------------
    do_val = args.val_ratio > 0.0
    if do_val:
        n_val = max(1, int(len(all_work_items) * args.val_ratio))
        n_val = min(n_val, len(all_work_items) - 1)
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(all_work_items)).tolist()
        val_indices = set(perm[:n_val])
        train_items = [wi for i, wi in enumerate(all_work_items) if i not in val_indices]
        val_items   = [wi for i, wi in enumerate(all_work_items) if i in val_indices]

        val_output_dir = pathlib.Path(
            args.val_output_dir if args.val_output_dir else str(output_dir) + "_val"
        )
        val_output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Split: {len(train_items)} train / {len(val_items)} val  "
            f"(seed={args.split_seed}, val_output={val_output_dir})"
        )
    else:
        train_items = all_work_items
        val_items = []
        val_output_dir = None

    print(f"Processing {len(train_items)} train + {len(val_items)} val NPZ files …")

    # Auto-size TFRecord shards: ~8 episodes per shard, floor at num_workers (train) / 1 (val).
    def _auto_num_shards(n: int, floor: int) -> int:
        if n <= 0:
            return 0
        return max(floor, (n + 7) // 8)

    num_shards = _auto_num_shards(len(train_items), args.num_workers)
    val_num_shards = _auto_num_shards(len(val_items), 1)
    print(f"Auto-sized shards: train={num_shards} val={val_num_shards}")

    # ------------------------------------------------------------------
    # Open TFRecord writers (train + optional val).
    # ------------------------------------------------------------------
    import tensorflow as tf

    shard_template = str(output_dir / f"isaaclab_rlds-train.tfrecord-{{:05d}}-of-{num_shards:05d}")
    train_writers = [tf.io.TFRecordWriter(shard_template.format(i)) for i in range(num_shards)]

    val_writers: list = []
    if do_val:
        val_tmpl = str(val_output_dir / f"isaaclab_rlds-train.tfrecord-{{:05d}}-of-{val_num_shards:05d}")
        val_writers = [tf.io.TFRecordWriter(val_tmpl.format(i)) for i in range(val_num_shards)]

    # ------------------------------------------------------------------
    # Launch workers and write results round-robin.
    # ------------------------------------------------------------------
    train_episodes = val_episodes = 0
    train_total_steps = val_total_steps = 0
    errors = 0
    train_cursor = val_cursor = 0

    # Tag each future with (npz_path_str, is_val) so we can route after
    # as_completed() returns them out of order.
    ctx = _mp.get_context("spawn")
    num_workers = max(1, args.num_workers)

    combined = [(wi, False) for wi in train_items] + [(wi, True) for wi in val_items]

    # Bounded-submit: keep at most `in_flight_cap` futures alive at any time.
    # Without this, submitting all ~300 tasks up-front makes the parent hold
    # every completed episode's serialised protobuf (multi-MB JPEG payload)
    # until the executor exits — OOMs the main process on large datasets even
    # when num_workers == 1.
    in_flight_cap = max(2, num_workers * 2)
    pending_iter = iter(combined)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures: dict[concurrent.futures.Future, tuple[str, bool]] = {}

        def _submit_next() -> bool:
            try:
                wi, is_val = next(pending_iter)
            except StopIteration:
                return False
            fut = executor.submit(_process_one_npz, wi)
            futures[fut] = (wi[0], is_val)
            return True

        for _ in range(in_flight_cap):
            if not _submit_next():
                break

        with tqdm(total=len(combined), desc="Converting") as pbar:
            while futures:
                done, _not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    npz_path_str, is_val = futures.pop(future)
                    try:
                        serialised, err, n_steps = future.result()
                    except Exception as exc:
                        print(f"[error] {pathlib.Path(npz_path_str).name}: {exc}")
                        errors += 1
                        pbar.update(1)
                        _submit_next()
                        del future
                        continue

                    if serialised is None:
                        print(f"[warn] {err}")
                        errors += 1
                        pbar.update(1)
                        _submit_next()
                        del future
                        continue

                    if is_val:
                        val_writers[val_cursor % val_num_shards].write(serialised)
                        val_cursor += 1
                        val_episodes += 1
                        val_total_steps += n_steps
                    else:
                        train_writers[train_cursor % num_shards].write(serialised)
                        train_cursor += 1
                        train_episodes += 1
                        train_total_steps += n_steps

                    pbar.update(1)
                    del serialised
                    _submit_next()
                    del future

    for w in train_writers:
        w.close()
    for w in val_writers:
        w.close()

    # ------------------------------------------------------------------
    # Write dataset_info.json (one per output dir).
    # ------------------------------------------------------------------
    def _save_info(ep_count: int, step_count: int, out_dir: pathlib.Path, n_shards: int) -> None:
        DatasetInfo(
            state_dim=state_dim,
            action_dim=action_dim,
            max_segments=max_segments,
            max_base_views=args.max_base_views,
            total_episodes=ep_count,
            total_steps=step_count,
            gripper_dim=gripper_dim,
            # None keeps the key out of the file, so plain converts write the same
            # 7-key dataset_info.json they always have.
            has_wrist=False if args.no_wrist else None,
            robot=args.robot,
        ).save(out_dir)

    _save_info(train_episodes, train_total_steps, output_dir, num_shards)
    if do_val:
        _save_info(val_episodes, val_total_steps, val_output_dir, val_num_shards)

    print(
        f"\nDone.  train={train_episodes}  val={val_episodes}  errors={errors}\n"
        f"  train → {output_dir}\n"
        + (f"  val   → {val_output_dir}\n" if do_val else "")
    )


if __name__ == "__main__":
    main()
