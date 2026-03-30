#!/bin/zsh
# =============================================================================
# Intent-classified SmolVLA inference
#
# 1. Listen for voice command (or type with --text)
# 2. Classify into red/green/blue bowl task
# 3. Launch lerobot-record with the correct task description
#
# Usage:
#   ./run_intent_inference.sh               # Voice mode
#   ./run_intent_inference.sh --text        # Keyboard mode
# =============================================================================

export HF_TOKEN=$HF_TOKEN
# Sentence-transformers model cache (avoid permission issues on ~/.cache/huggingface/hub)
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-/tmp/st_cache}"

source "$(dirname "$0")/activate.sh"

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
POLICY_REPO_ID="RajatDandekar/smolvla_box_to_bowl"
DATASET_REPO_ID="RajatDandekar/so101_box_to_bowl"
FOLLOWER_PORT="/dev/tty.wchusbserial5AE60830811"
LEADER_PORT="/dev/tty.wchusbserial5A7C1167331"
WHISPER_MODEL="base"
LISTEN_DURATION=8
# ─────────────────────────────────────────────────────────────────────────────

TEXT_FLAG=""
if [[ "$1" == "--text" ]]; then
    TEXT_FLAG="--text_mode"
fi

python intent_inference.py \
    --policy_path="${POLICY_REPO_ID}" \
    --dataset_repo_id="${DATASET_REPO_ID}" \
    --follower_port="${FOLLOWER_PORT}" \
    --leader_port="${LEADER_PORT}" \
    --whisper_model="${WHISPER_MODEL}" \
    --listen_duration="${LISTEN_DURATION}" \
    ${TEXT_FLAG}
