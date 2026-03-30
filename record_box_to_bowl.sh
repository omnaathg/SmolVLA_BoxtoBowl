#!/bin/zsh
# =============================================================================
# Record SO-101 episodes: Pick up box and place in colored bowl
#
# This is a LANGUAGE-CONDITIONED task with 3 bowl targets (red, green, blue).
# Record in 3 batches — each batch uses a different task instruction.
#
# Usage:
#   ./record_box_to_bowl.sh red      # Record 15 episodes for red bowl
#   ./record_box_to_bowl.sh green    # Record 15 episodes for green bowl (appends)
#   ./record_box_to_bowl.sh blue     # Record 15 episodes for blue bowl (appends)
#
# Recommended order: Record red first (creates dataset), then green, then blue.
# Total: ~45 episodes across 3 bowl colors.
# =============================================================================

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_box_to_bowl"
EPISODES_PER_COLOR=15
# ─────────────────────────────────────────────────────────────────────────────

COLOR="${1}"

if [[ -z "$COLOR" ]]; then
  echo "Usage: ./record_box_to_bowl.sh <red|green|blue>"
  echo ""
  echo "Record episodes for each bowl color:"
  echo "  1. ./record_box_to_bowl.sh red     (creates dataset, records 15 episodes)"
  echo "  2. ./record_box_to_bowl.sh green   (resumes dataset, records 15 more)"
  echo "  3. ./record_box_to_bowl.sh blue    (resumes dataset, records 15 more)"
  exit 1
fi

# Set task description based on bowl color
case "$COLOR" in
  red)
    TASK="Pick up the box and place it in the red bowl"
    RESUME_FLAG=""
    ;;
  green)
    TASK="Pick up the box and place it in the green bowl"
    RESUME_FLAG="--resume=true"
    ;;
  blue)
    TASK="Pick up the box and place it in the blue bowl"
    RESUME_FLAG="--resume=true"
    ;;
  *)
    echo "Error: Unknown color '$COLOR'. Use red, green, or blue."
    exit 1
    ;;
esac

echo "============================================"
echo "  Bowl color:  $COLOR"
echo "  Task:        $TASK"
echo "  Episodes:    $EPISODES_PER_COLOR"
echo "  Dataset:     $DATASET_REPO_ID"
echo "  Resume:      ${RESUME_FLAG:-no (creating new dataset)}"
echo "============================================"
echo ""

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.wchusbserial5AE60830811 \
  --robot.id=my_follower \
  --robot.cameras='{"webcam": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "arm_cam": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.wchusbserial5A7C1167331 \
  --teleop.id=my_leader \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes="${EPISODES_PER_COLOR}" \
  --display_data=true \
  ${RESUME_FLAG}

echo ""
echo "=== Done recording $COLOR bowl episodes! ==="
