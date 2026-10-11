#!/usr/bin/env bash
# Replay the official LIBERO demonstrations (HDF5) in simulation and save NPZ trajectories in the same format as
# SkillWeaver's data collection, using a joint-position controller. Only successful replays are converted later
# (*_success.npz). Requires LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO) in the Python environment.
#
# Usage: bash examples/libero_rlds/replay_libero_hdf5_to_npz.sh <libero_hdf5_dir> <output_dir> [extra args]
#   e.g. bash examples/libero_rlds/replay_libero_hdf5_to_npz.sh LIBERO/datasets/libero_object data/libero_npz/libero_object
set -euo pipefail
HDF5_ROOT="$1"
OUTPUT_DIR="$2"
shift 2

python examples/libero_rlds/replay_libero_hdf5_to_npz.py \
  --hdf5-root           "$HDF5_ROOT" \
  --output-dir          "$OUTPUT_DIR" \
  --resolution          256 \
  --cmd-limit           "0.05,0.05,0.05,0.5,0.5,0.5" \
  --num-workers         1 \
  --max-demos           -1 \
  --replay-mode         joint_pos \
  --track-error \
  --joint-kp            "800,800,800,600,600,600,600" \
  --joint-damping-ratio 0.8 \
  --joint-policy-freq   10 \
  "$@"
