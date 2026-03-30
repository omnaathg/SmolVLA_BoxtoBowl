#!/bin/zsh
# =============================================================================
# Continuous voice-controlled SmolVLA inference (60 minutes)
#
# The robot runs autonomously. Speak into your mic anytime to change the task.
# SmolVLA's VLM grounds language semantically:
#   "place it in the bowl colored like blood"    → red bowl
#   "move the cube to the sky-colored bowl"      → blue bowl
#   "put it in the bowl that looks like grass"   → green bowl
#
# Usage:
#   ./run_voice_inference.sh               # Voice mode (speak anytime)
#   ./run_voice_inference.sh --text        # Keyboard mode (type instructions)
# =============================================================================

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
POLICY_REPO_ID="RajatDandekar/smolvla_box_to_bowl"
DATASET_REPO_ID="RajatDandekar/so101_box_to_bowl"
WHISPER_MODEL="base"        # tiny | base | small | medium
LISTEN_DURATION=8           # seconds per voice capture window
DURATION=3600               # total runtime in seconds (3600 = 60 min)
INITIAL_TASK="Pick up the box and place it in the red bowl"
# ─────────────────────────────────────────────────────────────────────────────

# Check for --text flag
TEXT_FLAG=""
if [[ "$1" == "--text" ]]; then
    TEXT_FLAG="--text_mode"
    echo "=== Keyboard mode: type instructions anytime ==="
fi

python voice_inference.py \
    --policy_path="${POLICY_REPO_ID}" \
    --dataset_repo_id="${DATASET_REPO_ID}" \
    --whisper_model="${WHISPER_MODEL}" \
    --listen_duration="${LISTEN_DURATION}" \
    --duration="${DURATION}" \
    --initial_task="${INITIAL_TASK}" \
    ${TEXT_FLAG}
