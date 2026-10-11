"""
RLDS-based data loader for IsaacLab trajectories.

Mirrors the structure of DroidRldsDataset but:
  - Loads from multiple TFRecord directories (multi-batch support).
  - Implements dynamic chunking, multi-view sampling, and augmented
    instruction sampling entirely inside the tf.data pipeline.
  - Does NOT use TFDS or dlimp — loads TFRecords directly with
    tf.data.TFRecordDataset and a hand-written parse function.

Schema expected on disk: see examples/isaaclab_rlds/dataset_builder.py.
"""

from __future__ import annotations

import glob
import json
import logging
import pathlib
from typing import Sequence


class IsaacLabRldsDataset:
    """TF dataset loading IsaacLab RLDS TFRecord shards.

    Args:
        data_dirs: List of directories each containing:
            - ``dataset_info.json`` (written by convert_npz_to_rlds.py)
            - ``isaaclab_rlds-train.tfrecord-*`` shard files
        batch_size: Number of steps per batch.
        shuffle: Whether to shuffle the step buffer.
        action_chunk_size: Number of future actions to return per step.
        shuffle_buffer_size: Shuffle buffer for individual steps.
        num_parallel_reads: tf.data parallelism for TFRecordDataset.
        num_parallel_calls: tf.data parallelism for map operations.
        per_step_segment_aug: If true, sample a valid segment per step. If false,
            always use segment 0 (the full episode / overall instruction).
        include_attr_aug_instruction_pool: If true, merge attr-augmented instruction
            pools with the default segment pool when the shard provides non-empty
            attr candidates.
        replace_gripper_with_binary: If true, replace the trailing ``gripper_dim``
            columns of every action (and per-segment hold action) with the ±1
            binary intent stored in ``steps/action_gripper_binary`` /
            ``episode_metadata/segment_hold_action_gripper_binary``.  Requires
            shards that contain ``steps/action_gripper_binary``.  Norm stats
            must be (re)computed with this flag set to match training distribution.
    """

    def __init__(
        self,
        data_dirs: Sequence[str],
        batch_size: int,
        *,
        extra_dirs_full: Sequence[str] = (),
        shuffle: bool = True,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE
        per_step_segment_aug: bool = False,
        include_attr_aug_instruction_pool: bool = False,
        replace_gripper_with_binary: bool = False,
        filter_interior_holds: bool = False,
        hold_vel_threshold: float = 3e-3,
        hold_cmd_threshold: float = 5e-3,
        hold_gripper_threshold: float = 1e-3,
        hold_keep_per_run: int = 1,
        hold_filter_head_pause: bool = True,
    ):
        # Defer all TF imports so that this module can be imported without
        # a TF installation (e.g., during pure-PyTorch training).
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        autotune = tf.data.AUTOTUNE
        num_parallel_reads = num_parallel_reads if num_parallel_reads != -1 else autotune
        num_parallel_calls = num_parallel_calls if num_parallel_calls != -1 else autotune

        # ------------------------------------------------------------------
        # Read dataset_info.json from the first data dir to get dims.
        # ------------------------------------------------------------------
        if not data_dirs:
            raise ValueError("data_dirs must not be empty.")

        info = _load_dataset_info(data_dirs[0])
        state_dim: int = info["state_dim"]
        action_dim: int = info["action_dim"]
        max_segments: int = info["max_segments"]
        gripper_dim: int = int(info.get("gripper_dim", 1))
        total_steps: int = max(
            1, sum(_load_dataset_info(d).get("total_steps", 0) for d in [*data_dirs, *extra_dirs_full])
        )

        if replace_gripper_with_binary:
            if gripper_dim <= 0 or gripper_dim >= action_dim:
                raise ValueError(
                    f"replace_gripper_with_binary requires 0 < gripper_dim < action_dim, "
                    f"got gripper_dim={gripper_dim} action_dim={action_dim}."
                )
            logging.info(
                "IsaacLabRldsDataset: replace_gripper_with_binary=True (gripper_dim=%d). "
                "Ensure norm stats were computed with this flag set.",
                gripper_dim,
            )

        logging.info(
            "IsaacLabRldsDataset: state_dim=%d action_dim=%d max_segments=%d gripper_dim=%d",
            state_dim, action_dim, max_segments, gripper_dim,
        )

        # ------------------------------------------------------------------
        # Collect shard files from all data dirs (training dirs + extra dirs).
        # ------------------------------------------------------------------
        all_shards: list[str] = []
        for d in [*data_dirs, *extra_dirs_full]:
            shards = sorted(glob.glob(str(pathlib.Path(d) / "*.tfrecord-*")))
            if not shards:
                logging.warning("No TFRecord shards found in %s", d)
            all_shards.extend(shards)
        if not all_shards:
            raise FileNotFoundError(
                f"No TFRecord shards found in any of: data_dirs={list(data_dirs)} "
                f"extra_dirs_full={list(extra_dirs_full)}"
            )
        logging.info(
            "IsaacLabRldsDataset: found %d shards across %d dirs.",
            len(all_shards), len(data_dirs) + len(extra_dirs_full),
        )

        # ------------------------------------------------------------------
        # Build the feature spec for parsing.
        # ------------------------------------------------------------------
        import sys
        _rlds_dir = str(pathlib.Path(__file__).parent.parent.parent.parent / "examples" / "isaaclab_rlds")
        if _rlds_dir not in sys.path:
            sys.path.insert(0, _rlds_dir)
        from dataset_builder import MAX_BASE_VIEWS, make_feature_spec  # type: ignore[import]

        feature_spec = make_feature_spec(state_dim, action_dim, max_segments)

        # ------------------------------------------------------------------
        # TFRecordDataset → parse → restructure → chunk → flatten → decode
        # ------------------------------------------------------------------
        dataset = tf.data.TFRecordDataset(
            all_shards,
            num_parallel_reads=num_parallel_reads,
        )
        dataset = dataset.repeat()

        # ---- Parse -------------------------------------------------------
        def parse_episode(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_spec)

            # Per-step float arrays.
            state = tf.reshape(
                tf.io.decode_raw(parsed["steps/observation/state"], tf.float32),
                [-1, state_dim],
            )  # (T, state_dim)
            action = tf.reshape(
                tf.io.decode_raw(parsed["steps/action"], tf.float32),
                [-1, action_dim],
            )  # (T, action_dim)

            if replace_gripper_with_binary:
                gripper_binary = tf.reshape(
                    tf.io.decode_raw(parsed["steps/action_gripper_binary"], tf.float32),
                    [-1, gripper_dim],
                )  # (T, gripper_dim) ±1
                action = tf.concat([action[:, :-gripper_dim], gripper_binary], axis=-1)

            # Per-step images: VarLen → dense.
            ext_imgs = [
                tf.sparse.to_dense(
                    parsed[f"steps/observation/exterior_image_{i}"], default_value=b""
                )
                for i in range(MAX_BASE_VIEWS)
            ]  # each (T,) string
            wrist_img = tf.sparse.to_dense(
                parsed["steps/observation/wrist_image"], default_value=b""
            )  # (T,) string

            # Episode metadata scalars.
            view_count = tf.cast(parsed["episode_metadata/exterior_view_count"], tf.int32)
            seg_count = tf.cast(parsed["episode_metadata/segment_count"], tf.int32)

            # Episode metadata arrays.
            seg_starts = tf.reshape(
                tf.io.decode_raw(parsed["episode_metadata/segment_starts"], tf.int32),
                [max_segments],
            )
            seg_ends = tf.reshape(
                tf.io.decode_raw(parsed["episode_metadata/segment_ends"], tf.int32),
                [max_segments],
            )
            seg_hold_actions = tf.reshape(
                tf.io.decode_raw(parsed["episode_metadata/segment_hold_actions"], tf.float32),
                [max_segments, action_dim],
            )
            if replace_gripper_with_binary:
                seg_hold_gripper_binary = tf.reshape(
                    tf.io.decode_raw(
                        parsed["episode_metadata/segment_hold_action_gripper_binary"], tf.float32
                    ),
                    [max_segments, gripper_dim],
                )  # (max_segments, gripper_dim) ±1
                seg_hold_actions = tf.concat(
                    [seg_hold_actions[:, :-gripper_dim], seg_hold_gripper_binary], axis=-1
                )
            seg_pools = tf.sparse.to_dense(
                parsed["episode_metadata/segment_instruction_pools"], default_value=b""
            )  # (max_segments,) string
            seg_pools_attr_aug = tf.sparse.to_dense(
                parsed["episode_metadata/segment_instruction_pools_attr_aug"], default_value=b""
            )  # (max_segments,) string
            seg_pools_attr_aug = tf.pad(
                seg_pools_attr_aug[:max_segments],
                [[0, tf.maximum(0, max_segments - tf.shape(seg_pools_attr_aug)[0])]],
                constant_values=b"",
            )

            # Camera calibration fields — per exterior camera slot.
            camera_ixts    = [tf.io.decode_raw(parsed[f"episode_metadata/camera_ixt_{i}"],    tf.float32) for i in range(MAX_BASE_VIEWS)]  # each (9,)
            camera_ixt_hws = [tf.io.decode_raw(parsed[f"episode_metadata/camera_ixt_hw_{i}"], tf.int32)   for i in range(MAX_BASE_VIEWS)]  # each (2,)
            camera_exts    = [tf.io.decode_raw(parsed[f"episode_metadata/camera_ext_{i}"],    tf.float32) for i in range(MAX_BASE_VIEWS)]  # each (16,)
            robot_base_pose_w = tf.io.decode_raw(
                parsed["episode_metadata/robot_base_pose_w"], tf.float32
            )  # (16,)

            ep: dict = {
                "steps/action": action,
                "steps/observation/state": state,
                "steps/observation/wrist_image": wrist_img,
                "episode_metadata/exterior_view_count": view_count,
                "episode_metadata/segment_count": seg_count,
                "episode_metadata/segment_starts": seg_starts,
                "episode_metadata/segment_ends": seg_ends,
                "episode_metadata/segment_hold_actions": seg_hold_actions,
                "episode_metadata/segment_instruction_pools": seg_pools,
                "episode_metadata/segment_instruction_pools_attr_aug": seg_pools_attr_aug,
                "camera_ixts":    tf.stack(camera_ixts),     # (MAX_BASE_VIEWS, 9)
                "camera_ixt_hws": tf.stack(camera_ixt_hws),  # (MAX_BASE_VIEWS, 2)
                "camera_exts":    tf.stack(camera_exts),      # (MAX_BASE_VIEWS, 16)
                "robot_base_pose_w": robot_base_pose_w,
            }
            for i, img in enumerate(ext_imgs):
                ep[f"steps/observation/exterior_image_{i}"] = img

            return ep

        dataset = dataset.map(parse_episode, num_parallel_calls=num_parallel_calls)

        # ---- Episode-level hold-step filter (BEFORE any chunking) ------------
        # The filter operates on the raw parsed episode: it gathers EVERY
        # per-step array (state, action, wrist, all exterior views) by the same
        # keep mask, and remaps segment_starts / segment_ends from original
        # indices to new (compressed) indices via tf.searchsorted on the kept
        # original indices. Empty segments end up with new_start == new_end and
        # are naturally excluded by `_per_step_aug`'s `start <= t < end` check
        # (no extra plumbing needed). Per-episode constants (camera params,
        # robot_base_pose_w, seg_hold_actions, instruction pools) are left
        # untouched. After this step, every downstream stage (restructure,
        # per_step_aug, chunk_actions, flat_map) operates on a clean compressed
        # trajectory; chunks built from it never contain dropped-hold actions.
        # Episodes that shrink below action_chunk_size + hold_keep_boundary are
        # dropped — they can't produce any usable chunk.
        if filter_interior_holds:
            logging.info(
                "IsaacLabRldsDataset: filter_interior_holds=True at parse stage "
                "(vel_thr=%.4f, cmd_thr=%.4f, gripper_thr=%.4f, "
                "keep_per_run=%d, filter_head_pause=%s, gripper_dim=%d). "
                "Per-step arrays gathered, segment metadata remapped. "
                "Recompute norm_stats after toggling this flag.",
                hold_vel_threshold, hold_cmd_threshold, hold_gripper_threshold,
                hold_keep_per_run, hold_filter_head_pause, gripper_dim,
            )
            min_post_filter_len = action_chunk_size

            def filter_episode(ep):
                state  = ep["steps/observation/state"]   # (T, state_dim)
                action = ep["steps/action"]              # (T, action_dim)
                T = tf.shape(state)[0]

                # --- per-step is_hold (arm vel + |action-state| + gripper) ---
                if gripper_dim > 0:
                    s_arm = state[:, :-gripper_dim]
                    a_arm = action[:, :-gripper_dim]
                    s_grip = state[:, -gripper_dim:]
                else:
                    s_arm = state; a_arm = action; s_grip = None

                arm_vel = tf.reduce_max(
                    tf.abs(s_arm[1:] - s_arm[:-1]), axis=-1,
                )
                arm_vel = tf.concat([arm_vel, arm_vel[-1:]], axis=0)
                cmd_dist = tf.reduce_max(tf.abs(a_arm - s_arm), axis=-1)
                if s_grip is not None:
                    grip_change = tf.reduce_max(
                        tf.abs(s_grip[1:] - s_grip[:-1]), axis=-1,
                    )
                    grip_change = tf.concat([grip_change, grip_change[-1:]], axis=0)
                else:
                    grip_change = tf.zeros([T], dtype=tf.float32)

                is_hold = (
                    (arm_vel < hold_vel_threshold)
                    & (cmd_dist < hold_cmd_threshold)
                    & (grip_change < hold_gripper_threshold)
                )
                is_move = tf.logical_not(is_hold)

                # For each hold frame, compute 1-indexed position within its
                # current hold run (motion frames reset to 0). Frame is kept
                # iff its position <= hold_keep_per_run. This means each hold
                # run contributes at most hold_keep_per_run frames (the FIRST
                # N — the ones immediately following motion end) to the output.
                # Initializer=0 = virtual motion at index -1, so a leading
                # all-hold prefix gets positions 1, 2, ... and contributes its
                # first N frames unless filter_head_pause drops them.
                is_hold_int = tf.cast(is_hold, tf.int32)
                pos_in_run = tf.scan(
                    lambda prev, x: tf.where(tf.cast(x, tf.bool), prev + 1, 0),
                    is_hold_int,
                    initializer=tf.constant(0, dtype=tf.int32),
                )                                                  # (T,)
                # Motion frames have pos_in_run == 0, so they are always kept.
                keep = pos_in_run <= hold_keep_per_run

                # head pause: drop leading all-hold prefix outright (overrides
                # the per-run keep budget for the leading prefix only).
                if hold_filter_head_pause:
                    any_move = tf.reduce_any(is_move)
                    first_move = tf.argmax(tf.cast(is_move, tf.int32), output_type=tf.int32)
                    idx = tf.range(T)
                    after_first_move = idx >= first_move
                    after_first_move = tf.where(
                        any_move,
                        after_first_move,
                        tf.zeros([T], dtype=tf.bool),
                    )
                    keep = keep & after_first_move

                keep_idx = tf.where(keep)[:, 0]                  # int64 (T_new,)

                # --- gather every per-step array ----------------------------
                new_ep = {}
                for k, v in ep.items():
                    if k.startswith("steps/"):
                        new_ep[k] = tf.gather(v, keep_idx)
                    else:
                        new_ep[k] = v

                # --- remap segment_starts / segment_ends --------------------
                # tf.searchsorted(keep_idx_sorted, x, "left") returns the new
                # index of the smallest kept original step >= x. For an empty
                # segment (no kept step in [start, end)), start and end both
                # map to the same new index → segment effectively excluded.
                keep_idx_i32 = tf.cast(keep_idx, tf.int32)
                seg_starts = ep["episode_metadata/segment_starts"]
                seg_ends   = ep["episode_metadata/segment_ends"]
                new_seg_starts = tf.cast(
                    tf.searchsorted(keep_idx_i32, seg_starts, side="left"),
                    tf.int32,
                )
                new_seg_ends = tf.cast(
                    tf.searchsorted(keep_idx_i32, seg_ends, side="left"),
                    tf.int32,
                )
                new_ep["episode_metadata/segment_starts"] = new_seg_starts
                new_ep["episode_metadata/segment_ends"]   = new_seg_ends
                # segment_count, segment_hold_actions, instruction pools,
                # camera params, robot_base_pose_w — all unchanged.
                return new_ep

            dataset = dataset.map(filter_episode, num_parallel_calls=num_parallel_calls)
            dataset = dataset.filter(
                lambda ep: tf.shape(ep["steps/action"])[0] >= min_post_filter_len
            )

        # ---- Restructure (view sampling + full-episode pass-through) --------
        def restructure(ep):
            seg_count        = ep["episode_metadata/segment_count"]
            seg_starts       = ep["episode_metadata/segment_starts"]
            seg_ends         = ep["episode_metadata/segment_ends"]
            seg_hold_actions = ep["episode_metadata/segment_hold_actions"]
            seg_pools        = ep["episode_metadata/segment_instruction_pools"]
            seg_pools_attr   = ep["episode_metadata/segment_instruction_pools_attr_aug"]
            view_count       = ep["episode_metadata/exterior_view_count"]

            T = tf.shape(ep["steps/action"])[0]

            # Per-step view sampling: pick one camera index per frame, then gather.
            # Done here (before flat_map) so the shuffle buffer holds (T,) strings,
            # not (T, MAX_BASE_VIEWS) strings.
            all_views = tf.stack(
                [ep[f"steps/observation/exterior_image_{i}"] for i in range(MAX_BASE_VIEWS)],
                axis=1,
            )  # (T, MAX_BASE_VIEWS) string — temporary
            view_indices = tf.random.uniform([T], 0, view_count, dtype=tf.int32)
            exterior_img = tf.gather(all_views, view_indices, batch_dims=1)  # (T,) string

            # Per-step camera params: gather slot matching the selected view index.
            camera_ixt    = tf.gather(ep["camera_ixts"],    view_indices)  # (T, 9)
            camera_ixt_hw = tf.gather(ep["camera_ixt_hws"], view_indices)  # (T, 2)
            camera_ext    = tf.gather(ep["camera_exts"],    view_indices)  # (T, 16)

            out = {
                "actions":                  ep["steps/action"],                    # (T, action_dim)
                "observation/image":        exterior_img,                          # (T,) string
                "observation/wrist_image":  ep["steps/observation/wrist_image"],   # (T,) string
                "observation/state":        ep["steps/observation/state"],          # (T, state_dim)
                "camera_ixt":          camera_ixt,                                 # (T, 9)
                "camera_ixt_hw":       camera_ixt_hw,                              # (T, 2)
                "camera_ext":          camera_ext,                                 # (T, 16)
                "robot_base_pose_w":   tf.tile(ep["robot_base_pose_w"][None], [T, 1]),  # (T, 16)
            }

            if per_step_segment_aug:
                # Pass segment metadata as-is; _per_step_aug() will consume and drop them.
                out["seg_starts"]       = seg_starts        # (max_segments,)
                out["seg_ends"]         = seg_ends          # (max_segments,)
                out["seg_hold_actions"] = seg_hold_actions  # (max_segments, action_dim)
                out["seg_pools"]        = seg_pools         # (max_segments,)
                out["seg_pools_attr"]   = seg_pools_attr    # (max_segments,)
                out["seg_count"]        = seg_count         # scalar int32
            else:
                # No segment augmentation: always train on segment 0 (full episode / overall instruction).
                seg_idx     = tf.constant(0, dtype=tf.int32)
                seg_start   = seg_starts[seg_idx]
                seg_end     = seg_ends[seg_idx]
                hold_action = seg_hold_actions[seg_idx]
                instr_str   = seg_pools[seg_idx]
                attr_str    = seg_pools_attr[seg_idx]
                if include_attr_aug_instruction_pool:
                    base_nonempty = tf.strings.length(instr_str) > 0
                    attr_nonempty = tf.strings.length(attr_str) > 0
                    instr_str = tf.cond(
                        attr_nonempty,
                        lambda: tf.cond(
                            base_nonempty,
                            lambda: tf.strings.join([instr_str, attr_str], separator="\n"),
                            lambda: attr_str,
                        ),
                        lambda: instr_str,
                    )
                instruction = tf.random.shuffle(tf.strings.split(instr_str, sep="\n"))[0]
                out["hold_action"] = hold_action                                   # (action_dim,)
                out["prompt"]      = tf.tile(tf.expand_dims(instruction, 0), [T])  # (T,)

            return out

        dataset = dataset.map(restructure, num_parallel_calls=num_parallel_calls)

        # ---- Per-step segment augmentation (only when enabled) --------------
        if per_step_segment_aug:
            def _per_step_aug(traj):
                T          = tf.shape(traj["actions"])[0]
                seg_starts = traj.pop("seg_starts")        # (max_segments,)
                seg_ends   = traj.pop("seg_ends")          # (max_segments,)
                seg_holds  = traj.pop("seg_hold_actions")  # (max_segments, action_dim)
                seg_pools  = traj.pop("seg_pools")         # (max_segments,)
                seg_pools_attr = traj.pop("seg_pools_attr") # (max_segments,)
                seg_count  = traj.pop("seg_count")         # scalar int32

                episode_ts = tf.cast(tf.range(T), tf.int32)  # (T,)

                # Valid mask: (T, max_segments) — True where segment j covers frame t.
                valid = tf.logical_and(
                    seg_starts[None] <= episode_ts[:, None],
                    episode_ts[:, None] < seg_ends[None],
                )
                idx_range = tf.range(tf.shape(seg_starts)[0], dtype=tf.int32)
                valid = tf.logical_and(valid, idx_range[None] < seg_count)

                # Uniform sample per step via gumbel-max trick → chosen: (T,)
                logits = tf.where(valid, 0.0, -1e9)
                noise  = -tf.math.log(-tf.math.log(
                    tf.random.uniform(tf.shape(logits), minval=1e-10, maxval=1.0)
                ))
                chosen = tf.argmax(logits + noise, axis=1, output_type=tf.int32)  # (T,)

                traj["hold_actions"]   = tf.gather(seg_holds, chosen)  # (T, action_dim)
                traj["seg_end_chosen"] = tf.gather(seg_ends,  chosen)  # (T,) exclusive end

                chosen_pools = tf.gather(seg_pools, chosen)  # (T,) strings
                if include_attr_aug_instruction_pool:
                    chosen_attr_pools = tf.gather(seg_pools_attr, chosen)  # (T,) strings
                    base_nonempty = tf.strings.length(chosen_pools) > 0
                    attr_nonempty = tf.strings.length(chosen_attr_pools) > 0
                    joined_pools = tf.strings.join([chosen_pools, chosen_attr_pools], separator="\n")
                    merged_pools = tf.where(base_nonempty, joined_pools, chosen_attr_pools)
                    chosen_pools = tf.where(attr_nonempty, merged_pools, chosen_pools)
                def _sample_instr(pool_str):
                    instrs = tf.strings.split(pool_str, sep="\n")
                    return tf.random.shuffle(instrs)[0]
                traj["prompt"] = tf.map_fn(
                    _sample_instr, chosen_pools, fn_output_signature=tf.string
                )  # (T,)

                return traj

            dataset = dataset.map(_per_step_aug, num_parallel_calls=num_parallel_calls)

        # ---- Chunk actions -----------------------------------------------
        def chunk_actions(traj):
            """Build (T, action_chunk_size, action_dim) action windows."""
            traj_len = tf.shape(traj["actions"])[0]

            indices = (
                tf.range(action_chunk_size)[None] + tf.range(traj_len)[:, None]
            )  # (T, action_chunk_size); indices[t, k] = t + k

            capped = tf.minimum(indices, traj_len - 1)
            real   = tf.gather(traj["actions"], capped)  # (T, action_chunk_size, action_dim)

            if per_step_segment_aug:
                hold_actions   = traj.pop("hold_actions")    # (T, action_dim)
                seg_end_chosen = traj.pop("seg_end_chosen")  # (T,) exclusive end
                # Pad when chunk crosses the chosen segment's exclusive boundary.
                beyond = indices >= seg_end_chosen[:, None]          # (T, action_chunk_size)
                hold   = hold_actions[:, None, :]                    # (T, 1, action_dim)
            else:
                hold_action = traj.pop("hold_action")  # (action_dim,)
                beyond      = indices >= traj_len      # (T, action_chunk_size)
                hold        = tf.broadcast_to(hold_action[None, None], tf.shape(real))

            traj["actions"] = tf.where(beyond[:, :, None], hold, real)
            return traj

        dataset = dataset.map(chunk_actions, num_parallel_calls=num_parallel_calls)

        # ---- Flatten trajectory → individual steps -----------------------
        dataset = dataset.flat_map(
            lambda traj: tf.data.Dataset.from_tensor_slices(traj)
        )

        # ---- Shuffle before decode: buffer holds JPEG strings (~8 KB/step)
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)

        # ---- Decode JPEG images ------------------------------------------
        def decode_images(step):
            step["observation/image"] = tf.io.decode_image(
                step["observation/image"], expand_animations=False, dtype=tf.uint8
            )
            step["observation/wrist_image"] = tf.io.decode_image(
                step["observation/wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return step

        dataset = dataset.map(decode_images, num_parallel_calls=num_parallel_calls)

        # ---- Batch, memory budget ----------------------------------------
        dataset = dataset.batch(batch_size)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._total_steps = total_steps

    # ------------------------------------------------------------------
    # Protocol conformance
    # ------------------------------------------------------------------

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self) -> int:
        # Returns an approximate total step count across all data dirs.
        # Used by compute_norm_stats.py to estimate num_batches.
        return max(self._total_steps, self.batch_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset_info(data_dir: str) -> dict:
    p = pathlib.Path(data_dir) / "dataset_info.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"dataset_info.json not found in {data_dir}. "
            "Run convert_npz_to_rlds.py first."
        )
    with open(p) as f:
        return json.load(f)
