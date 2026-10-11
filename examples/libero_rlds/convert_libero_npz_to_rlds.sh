#!/usr/bin/env bash
# Convert replayed LIBERO demonstrations (*_success.npz from replay_libero_hdf5_to_npz.sh) to RLDS TFRecord shards
# in the same format as the SkillWeaver trajectories (the libero_demos/ directory of SkillWeaver-RLDS).
#
# Usage (from the repository root):
#   bash examples/libero_rlds/convert_libero_npz_to_rlds.sh <npz_dir> <output_dir>
#   e.g. bash examples/libero_rlds/convert_libero_npz_to_rlds.sh data/libero_npz/libero_object data/libero_rlds/libero_object
set -euo pipefail
DATA_ROOT="$1"
OUTPUT_DIR="$2"
NUM_WORKERS="${NUM_WORKERS:-8}"

# About 8 episodes per shard, at least NUM_WORKERS shards.
N=$(find "$DATA_ROOT" -name "*_success.npz" | wc -l)
NUM_SHARDS=$(( (N + 7) / 8 ))
NUM_SHARDS=$(( NUM_SHARDS < NUM_WORKERS ? NUM_WORKERS : NUM_SHARDS ))

uv run --group rlds examples/libero_rlds/convert_libero_npz_to_rlds.py \
    --data_root           "$DATA_ROOT" \
    --output_dir          "$OUTPUT_DIR" \
    --num_workers         "$NUM_WORKERS" \
    --num_shards          "$NUM_SHARDS" \
    --image_size          224 \
    --pad_hold_steps      0 \
    --max_segments        56 \
    --state_repr          joint \
    --state_gripper_1d \
    --arm_repr            joint \
    --arm_action_type     actual \
    --gripper_action_type target \
    --gripper_1d \
    --demos_per_task      0 \
    --split_seed          42
