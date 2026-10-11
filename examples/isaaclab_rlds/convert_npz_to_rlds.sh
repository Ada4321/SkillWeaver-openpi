#!/usr/bin/env bash
# Convert SkillWeaver trajectories (NPZ) to RLDS TFRecord shards, with the settings used for the released
# SkillWeaver-RLDS data: joint-position state and actions with a 1-D gripper, 224x224 JPEG images.
#
# Usage (from the repository root):
#   bash examples/isaaclab_rlds/convert_npz_to_rlds.sh <trajectories_dir> <scenes_dir> <instruction_manifest.jsonl> <output_dir>
#
# Example on the samples of the SkillWeaver-RLDS dataset:
#   bash examples/isaaclab_rlds/convert_npz_to_rlds.sh \
#       data/SkillWeaver-RLDS/samples/libero_object_task0/trajectories \
#       data/SkillWeaver-RLDS/samples/libero_object_task0/scenes \
#       data/SkillWeaver-RLDS/samples/libero_object_task0/instruction_pool_manifest.jsonl \
#       data/sample_rlds
#
# VAL_RATIO > 0 additionally writes a held-out split to <output_dir>_val (the released data used 0.01).
set -euo pipefail
DATA_ROOT="$1"
SCENE_DESC_ROOT="$2"
INSTRUCTION_MANIFEST="$3"
OUTPUT_DIR="$4"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.0}"
SPLIT_SEED="${SPLIT_SEED:-42}"

uv run --group rlds examples/isaaclab_rlds/convert_npz_to_rlds.py \
    --data_root                 "$DATA_ROOT" \
    --output_dir                "$OUTPUT_DIR" \
    --scene_desc_root           "$SCENE_DESC_ROOT" \
    --instruction_manifest_path "$INSTRUCTION_MANIFEST" \
    --num_workers               "$NUM_WORKERS" \
    --image_size                224 \
    --pad_hold_steps            0 \
    --max_nodes                 10 \
    --state_repr                joint \
    --state_gripper_1d \
    --arm_repr                  joint \
    --arm_action_type           actual \
    --gripper_action_type       target \
    --gripper_1d \
    --val_ratio                 "$VAL_RATIO" \
    --split_seed                "$SPLIT_SEED"
