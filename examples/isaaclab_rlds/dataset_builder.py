"""
IsaacLab RLDS dataset schema constants and feature spec.

This module provides:
  - Constants (MAX_BASE_VIEWS, max_segments_from_max_nodes)
  - make_feature_spec(): returns a tf.io.parse_single_example feature dict
    used by both convert_npz_to_rlds.py (for writing reference) and
    IsaacLabRldsDataset (for reading).
  - DatasetInfo: lightweight dataclass mirroring the dataset_info.json written
    by convert_npz_to_rlds.py.

TFRecord schema (one Example per episode)
==========================================

Per-step sequences (stored as T-element bytes_list):
  steps/observation/exterior_image_{0..3}   tf.string  JPEG bytes per frame
  steps/observation/wrist_image              tf.string  JPEG bytes per frame

Per-step float arrays (serialised as raw float32 bytes, shape inferred at load):
  steps/observation/state                    tf.string  → (T, state_dim)
  steps/action                               tf.string  → (T, action_dim)
  steps/action_gripper_binary                tf.string  → (T, gripper_dim) float32, ±1
                                                          (-1 = open intent, +1 = close intent)

Episode metadata scalars:
  episode_metadata/exterior_view_count       tf.int64
  episode_metadata/segment_count            tf.int64

Episode metadata arrays (serialised as raw bytes):
  episode_metadata/segment_starts           tf.string  → (max_segments,) int32
  episode_metadata/segment_ends             tf.string  → (max_segments,) int32
  episode_metadata/segment_hold_actions     tf.string  → (max_segments, action_dim) float32
  episode_metadata/segment_hold_action_gripper_binary
                                            tf.string  → (max_segments, gripper_dim) float32, ±1

Per-segment instruction pools (max_segments strings, each \\n-joined candidate pool):
  episode_metadata/segment_instruction_pools           tf.string  (VarLen, dense len=max_segments)
  episode_metadata/segment_instruction_pools_attr_aug  tf.string  (VarLen, dense len=max_segments)

String scalars:
  episode_metadata/file_path                tf.string
  episode_metadata/recording_folderpath     tf.string

Camera calibration, one entry per exterior camera slot i (raw bytes):
  episode_metadata/camera_ixt_{i}           tf.string  → (9,)  float32 intrinsics
  episode_metadata/camera_ixt_hw_{i}        tf.string  → (2,)  int32 image height/width
  episode_metadata/camera_ext_{i}           tf.string  → (16,) float32 world-to-camera
  episode_metadata/robot_base_pose_w        tf.string  → (16,) float32 robot base pose in world
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

# Maximum number of base (non-wrist) camera slots per episode.
MAX_BASE_VIEWS: int = 4


def max_segments_from_max_nodes(max_nodes: int) -> int:
    """Number of segment slots = 1 (full episode) + all pairwise node spans."""
    raw = 1 + max_nodes * (max_nodes + 1) // 2
    # Round up to next multiple of 4 for alignment.
    return ((raw + 3) // 4) * 4


DEFAULT_MAX_NODES: int = 10
DEFAULT_MAX_SEGMENTS: int = max_segments_from_max_nodes(DEFAULT_MAX_NODES)

@dataclasses.dataclass
class DatasetInfo:
    """Mirrors the dataset_info.json written by convert_npz_to_rlds.py."""

    state_dim: int
    action_dim: int
    max_segments: int
    max_base_views: int
    total_episodes: int
    total_steps: int
    gripper_dim: int = 1   # width of action_gripper_binary trailing axis
    # --- set only by `--no_wrist` converts; absent means "shard has a real wrist" ---
    # False => steps/observation/wrist_image holds a blank placeholder, not a camera.
    has_wrist: bool | None = None
    # Embodiment whose forward kinematics turn these joint actions into EE poses
    # (openpi.eval.robot_fk). Absent means "panda", which every suite predating the
    # simpler/WidowX port is. Must match the arm the NPZs were recorded with: the
    # FK model, its joint limits, and the arm dim all follow from it.
    robot: str | None = None
    # --- optional; absent on shards that never recorded them ---
    max_nodes: int | None = None          # node count that max_segments was derived from
    arm_is_delta: bool | None = None      # recorded so hold actions never have to be guessed

    @classmethod
    def load(cls, data_dir: str | pathlib.Path) -> "DatasetInfo":
        p = pathlib.Path(data_dir) / "dataset_info.json"
        with open(p) as f:
            d = json.load(f)
        return cls(
            state_dim=int(d["state_dim"]),
            action_dim=int(d["action_dim"]),
            max_segments=int(d["max_segments"]),
            max_base_views=int(d.get("max_base_views", MAX_BASE_VIEWS)),
            total_episodes=int(d.get("total_episodes", 0)),
            total_steps=int(d.get("total_steps", 0)),
            gripper_dim=int(d.get("gripper_dim", 1)),
            has_wrist=None if d.get("has_wrist") is None else bool(d["has_wrist"]),
            robot=None if d.get("robot") is None else str(d["robot"]),
            max_nodes=None if d.get("max_nodes") is None else int(d["max_nodes"]),
            arm_is_delta=None if d.get("arm_is_delta") is None else bool(d["arm_is_delta"]),
        )

    def save(self, data_dir: str | pathlib.Path) -> None:
        p = pathlib.Path(data_dir) / "dataset_info.json"
        # Drop never-set optional keys so a plain convert keeps writing the
        # exact same 7-key file it always has.
        payload = {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)


def make_feature_spec(state_dim: int, action_dim: int, max_segments: int) -> dict:
    """Return a feature spec dict for tf.io.parse_single_example.

    Parameters are needed because some arrays are stored as raw bytes and must
    be reshaped after decoding (requires knowing the trailing dimension).

    Args:
        state_dim: Observation state vector width.
        action_dim: Action vector width.
        max_segments: Number of segment slots (fixed at convert time).

    Returns:
        Dict mapping feature key → tf.io feature spec.
    """
    import tensorflow as tf

    spec: dict = {}

    # Per-step images: T JPEG-encoded byte strings per episode.
    for i in range(MAX_BASE_VIEWS):
        spec[f"steps/observation/exterior_image_{i}"] = tf.io.VarLenFeature(tf.string)
    spec["steps/observation/wrist_image"] = tf.io.VarLenFeature(tf.string)

    # Per-step float arrays serialised as raw float32 bytes.
    spec["steps/observation/state"] = tf.io.FixedLenFeature([], tf.string)
    spec["steps/action"] = tf.io.FixedLenFeature([], tf.string)
    # Per-step ±1 binary gripper intent, tiled across gripper_dim.
    # default_value=b"" lets shards written before this field was added still parse.
    spec["steps/action_gripper_binary"] = tf.io.FixedLenFeature([], tf.string, default_value=b"")

    # Episode metadata scalars.
    spec["episode_metadata/exterior_view_count"] = tf.io.FixedLenFeature([], tf.int64)
    spec["episode_metadata/segment_count"] = tf.io.FixedLenFeature([], tf.int64)

    # Episode metadata arrays serialised as raw bytes.
    spec["episode_metadata/segment_starts"] = tf.io.FixedLenFeature([], tf.string)
    spec["episode_metadata/segment_ends"] = tf.io.FixedLenFeature([], tf.string)
    spec["episode_metadata/segment_hold_actions"] = tf.io.FixedLenFeature([], tf.string)
    spec["episode_metadata/segment_hold_action_gripper_binary"] = tf.io.FixedLenFeature([], tf.string, default_value=b"")

    # Per-segment instruction pools: max_segments strings, each \n-joined.
    spec["episode_metadata/segment_instruction_pools"] = tf.io.VarLenFeature(tf.string)
    spec["episode_metadata/segment_instruction_pools_attr_aug"] = tf.io.VarLenFeature(tf.string)

    # String scalars.
    spec["episode_metadata/file_path"] = tf.io.FixedLenFeature([], tf.string)
    spec["episode_metadata/recording_folderpath"] = tf.io.FixedLenFeature([], tf.string)

    # Camera calibration — one entry per exterior camera slot (i=0..MAX_BASE_VIEWS-1).
    for i in range(MAX_BASE_VIEWS):
        spec[f"episode_metadata/camera_ixt_{i}"]    = tf.io.FixedLenFeature([], tf.string)  # (9,)  float32
        spec[f"episode_metadata/camera_ixt_hw_{i}"] = tf.io.FixedLenFeature([], tf.string)  # (2,)  int32
        spec[f"episode_metadata/camera_ext_{i}"]    = tf.io.FixedLenFeature([], tf.string)  # (16,) float32 w2c
    spec["episode_metadata/robot_base_pose_w"] = tf.io.FixedLenFeature([], tf.string)  # (16,) float32

    return spec
