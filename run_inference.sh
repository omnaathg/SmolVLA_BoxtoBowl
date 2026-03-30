#!/bin/zsh
# Run inference with trained SmolVLA model on SO-101 robot
# The robot will autonomously execute the learned task using the trained policy.
# Usage: ./run_inference.sh

export HF_TOKEN=$HF_TOKEN

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
# Trained model on HuggingFace
POLICY_REPO_ID="RajatDandekar/smolvla_box_to_bowl"

# Robot ports (update if changed — run lerobot-find-port to check)
FOLLOWER_PORT="/dev/tty.wchusbserial5AE60830811"

# Optional: leader arm for manual control between episodes
LEADER_PORT="/dev/tty.wchusbserial5A7C1167331"

# Dataset to save inference recordings (optional — for review/debugging)
DATASET_REPO_ID="RajatDandekar/eval_so101_box_to_bowl"
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/activate.sh"

echo "=== Running inference with trained SmolVLA policy ==="
echo "Policy: ${POLICY_REPO_ID}"
echo "Follower: ${FOLLOWER_PORT}"
echo ""

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=my_follower \
  --robot.cameras='{"webcam": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "arm_cam": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=my_leader \
  --policy.path="${POLICY_REPO_ID}" \
  --policy.device=mps \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="Pick up the box and place it in the bowl which has the same color as that of the ocean" \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=1 \
  --dataset.rename_map='{"observation.images.webcam": "observation.images.camera1", "observation.images.arm_cam": "observation.images.camera2"}' \
  --display_data=true
