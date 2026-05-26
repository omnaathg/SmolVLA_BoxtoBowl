#!/bin/zsh
# =============================================================================
# TRIAL recording — 1 episode, throwaway dataset.
# Use this to check camera angles, flip timing, and overall workflow
# BEFORE committing to the real dataset.
#
# Usage:
#   ./record_card_task_trial.sh "ace of hearts"
#
# After it finishes, check the HuggingFace dataset viewer:
#   https://huggingface.co/datasets/omnaathg/so101_card_memory_trial
#
# Things to verify in the viewer:
#   1. tripod_cam shows all 3 card faces clearly during first ~8s
#   2. Cards are hidden (covered) for the remainder
#   3. Arm reaches the target card cleanly
#   4. 20s feels like enough time — or adjust EPISODE_TIME_S below
# =============================================================================

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

# ─── TRIAL CONFIGURATION ─────────────────────────────────────────────────────
DATASET_REPO_ID="omnaathg/so101_card_memory_trial"
EPISODES=1
EPISODE_TIME_S=25
RESET_TIME_S=10
# ─────────────────────────────────────────────────────────────────────────────

CARD="${1:-ace of hearts}"
TASK="Pick up the ${CARD}"

echo "============================================"
echo "  TRIAL RECORDING (1 episode)"
echo "  Card:     $CARD"
echo "  Task:     $TASK"
echo "  Time/ep:  ${EPISODE_TIME_S}s"
echo "  Dataset:  $DATASET_REPO_ID  (trial — safe to delete)"
echo "============================================"
echo ""
echo "Workflow reminder:"
echo "   0– 5s : cards face-up, all 3 visible to tripod cam"
echo "   5–10s : slide cover over all 3 cards"
echo "  10–25s : teleop the pick (15s)"
echo ""

# Clear stale local trial data so lerobot-record can create a fresh dataset.
TRIAL_ROOT="$(dirname "$0")/data_trial"
if [[ -d "$TRIAL_ROOT" ]]; then
    echo "Removing stale trial data at $TRIAL_ROOT ..."
    rm -rf "$TRIAL_ROOT"
fi

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=COM8 \
    --robot.id=ojas_follower_arm \
    --robot.cameras='{"tripod_cam": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "gripper_cam": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=COM7 \
    --teleop.id=ojas_leader_arm \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="$(dirname "$0")/data_trial" \
    --dataset.single_task="${TASK}" \
    --dataset.fps=30 \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.num_episodes=${EPISODES} \
    --display_data=true

# Fallback push — in case lerobot-record crashed before uploading
echo ""
echo "Ensuring data is on HuggingFace Hub..."
python -c "
from huggingface_hub import HfApi
import os, sys
root = '${TRIAL_ROOT}'
if not os.path.exists(root):
    print('  No local data found — nothing to push.')
    sys.exit(0)
api = HfApi()
api.create_repo('${DATASET_REPO_ID}', repo_type='dataset', exist_ok=True)
api.upload_folder(folder_path=root, repo_id='${DATASET_REPO_ID}', repo_type='dataset')
print('  Upload complete.')
"

echo ""
echo "=== Trial done! Review at: ==="
echo "    https://huggingface.co/datasets/${DATASET_REPO_ID}"
echo ""
echo "What to check:"
echo "  tripod_cam : card faces visible and readable for first 8s?"
echo "  tripod_cam : cleanly hidden after the cover goes on?"
echo "  gripper_cam: arm trajectory looks correct?"
echo "  Timing     : was 20s enough, or do you need more?"
