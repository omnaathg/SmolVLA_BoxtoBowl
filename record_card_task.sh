#!/bin/zsh
# =============================================================================
# Record SO-101 episodes: Card Memory Task
#
# Layout: 3 face-up playing cards are shown to the robot for ~5 seconds.
# Cards are then flipped face-down by the operator.
# The robot must pick the correct card based on a voice instruction.
#
# Usage:
#   ./record_card_task.sh "ace of hearts"              # creates dataset (first run)
#   ./record_card_task.sh "queen of spades" resume     # adds episodes to existing dataset
#   ./record_card_task.sh "ten of clubs"    resume
#
# Each episode is 12 seconds. Default: 10 episodes per card.
# Recommend collecting ~10-15 episodes per unique card layout configuration.
# =============================================================================

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET_REPO_ID="omnaathg/so101_card_memory"
EPISODES_PER_RUN=10
EPISODE_TIME_S=12
RESET_TIME_S=15
# ──────────────────────────────────────────────────────────────────────────────

CARD="${1}"
RESUME_ARG="${2}"

if [[ -z "$CARD" ]]; then
    echo "Usage: ./record_card_task.sh <card-name> [resume]"
    echo ""
    echo "Examples:"
    echo "  ./record_card_task.sh \"ace of hearts\"         # first run (creates dataset)"
    echo "  ./record_card_task.sh \"queen of spades\" resume  # add to existing dataset"
    echo "  ./record_card_task.sh \"ten of clubs\"   resume"
    echo ""
    echo "Cards format: <rank> of <suit>"
    echo "  Ranks: ace two three four five six seven eight nine ten jack queen king"
    echo "  Suits: hearts diamonds clubs spades"
    exit 1
fi

TASK="Pick up the ${CARD}"

if [[ "${RESUME_ARG}" == "resume" ]]; then
    RESUME_FLAG="--resume=true"
else
    RESUME_FLAG=""
fi

echo "============================================"
echo "  Card:      $CARD"
echo "  Task:      $TASK"
echo "  Episodes:  $EPISODES_PER_RUN"
echo "  Time/ep:   ${EPISODE_TIME_S}s"
echo "  Dataset:   $DATASET_REPO_ID"
echo "  Resume:    ${RESUME_FLAG:-no (creating new dataset)}"
echo "============================================"
echo ""

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM8 \
    --robot.id=ojas_follower_arm \
    --robot.cameras='{"tripod_cam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "gripper_cam": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=COM7 \
    --teleop.id=ojas_leader_arm \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="$(dirname "$0")/data" \
    --dataset.single_task="${TASK}" \
    --dataset.fps=30 \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.num_episodes="${EPISODES_PER_RUN}" \
    --display_data=true \
    ${RESUME_FLAG}

echo ""
echo "=== Done recording: ${TASK} ==="
