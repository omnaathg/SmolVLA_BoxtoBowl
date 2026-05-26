#!/bin/bash
# =============================================================================
# ContextVLA Training on RunPod (CUDA GPU) — Card Memory Task
# Task: Watch 3 face-up cards, cards flip face-down, pick correct card from voice
#
# RECOMMENDED PODS (runpod.io):
#   - 1x A100 80GB → batch_size=32, ~4-5 hrs  (bfloat16, freeze_vision_encoder)
#   - 1x A40 48GB  → batch_size=16, ~7-8 hrs
#   - 1x 4090 24GB → batch_size=8,  ~12 hrs
#
# SETUP ON RUNPOD (run once after pod starts):
#   apt-get update && apt-get install -y ffmpeg
#   pip install "lerobot[smolvla]"
#   git clone https://github.com/omnaathg/SmolVLA_BoxtoBowl.git
#   cd SmolVLA_BoxtoBowl
#
# Then run:
#   HF_TOKEN=<your-token> bash train_contextvla_runpod.sh
# =============================================================================

export HF_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET_REPO_ID="omnaathg/so101_card_memory"
OUTPUT_REPO_ID="omnaathg/contextvla_card_memory"
OUTPUT_DIR="outputs/contextvla_card_memory"

# Tune BATCH_SIZE to your GPU VRAM (11-frame sequence is ~37% longer than 8-frame):
#   A100 80GB → 16 | A40 48GB → 8 | 4090 24GB → 4
BATCH_SIZE=16
STEPS=30000
# ──────────────────────────────────────────────────────────────────────────────

echo "=== Training ContextVLA on card-memory dataset ==="
echo "  Dataset:   $DATASET_REPO_ID"
echo "  Output:    $OUTPUT_REPO_ID"
echo "  Batch:     $BATCH_SIZE"
echo "  Steps:     $STEPS"
echo ""

python scripts/train_contextvla.py \
    --dataset_repo_id "$DATASET_REPO_ID" \
    --output_repo_id  "$OUTPUT_REPO_ID" \
    --output_dir      "$OUTPUT_DIR" \
    --batch_size      "$BATCH_SIZE" \
    --steps           "$STEPS" \
    --n_obs_steps     11 \
    --temporal_stride 30 \
    --fps             30 \
    --device          cuda

echo ""
echo "=== Done! Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
