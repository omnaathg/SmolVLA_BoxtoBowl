#!/bin/bash
# Run inference with trained SmolVLA model on SO-101 robot
# Usage: bash ./run_inference.sh [red|green|blue]
#   red   -> "Pick up the box and place it in the red bowl"
#   green -> "Pick up the box and place it in the green bowl"
#   blue  -> "Pick up the box and place it in the blue bowl"

export HF_TOKEN=$HF_TOKEN

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
POLICY_REPO_ID="omnaathg/smolvla_so101_policy_box_to_bowl_2026-05-22"
FOLLOWER_PORT="COM8"
LEADER_PORT="COM7"
DATASET_REPO_ID="omnaathg/rollout_so101_box_to_bowl"
# ──────────────────────────────────────────────────────────────────────────────

COLOR="${1:-red}"
case "$COLOR" in
  red)   TASK="Pick up the box and place it in the red bowl" ;;
  green) TASK="Pick up the box and place it in the green bowl" ;;
  blue)  TASK="Pick up the box and place it in the blue bowl" ;;
  *)     echo "Usage: bash ./run_inference.sh [red|green|blue]"; exit 1 ;;
esac

source "$(dirname "$0")/activate.sh"

echo "=== Running SmolVLA inference ==="
echo "Policy:  ${POLICY_REPO_ID}"
echo "Task:    ${TASK}"
echo ""

lerobot-rollout \
  --strategy.type=sentry \
  --strategy.upload_every_n_episodes=5 \
  --policy.path="${POLICY_REPO_ID}" \
  --policy.device=cuda \
  --inference.type=rtc \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=ojas_follower_arm \
  --robot.cameras='{"tripod_cam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "gripper_cam": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=ojas_leader_arm \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --task="${TASK}" \
  --duration=60 \
  --rename_map='{"observation.images.tripod_cam": "observation.images.camera1", "observation.images.gripper_cam": "observation.images.camera2"}' \
  --display_data=true
