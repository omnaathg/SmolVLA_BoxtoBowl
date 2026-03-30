#!/bin/bash
# =============================================================================
# SmolVLA Training on RunPod (CUDA GPU) — Box to Bowl (language-conditioned)
# Task: Pick up box → place in red/green/blue bowl based on instruction
#
# RECOMMENDED PODS (runpod.io):
#   - 1x A100 80GB  → batch_size=64, ~2 hrs
#   - 1x A40 48GB   → batch_size=32, ~3.5 hrs
#   - 1x 4090 24GB  → batch_size=16, ~5 hrs
#
# SETUP ON RUNPOD (run once after pod starts):
#   apt-get update && apt-get install -y ffmpeg
#   pip install "lerobot[smolvla]"
#   huggingface-cli login --token $HF_TOKEN
#
# Then run:
#   bash train_smolvla_box_bowl_runpod.sh
# =============================================================================

export HF_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_box_to_bowl"
OUTPUT_REPO_ID="RajatDandekar/smolvla_box_to_bowl"

# Tune BATCH_SIZE to your GPU VRAM:
#   A100 80GB → 64 | A40 48GB → 32 | 4090 24GB → 16
BATCH_SIZE=64
STEPS=30000
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Training SmolVLA on box-to-bowl (multi-task) dataset ==="
echo "  Dataset:  $DATASET_REPO_ID"
echo "  Output:   $OUTPUT_REPO_ID"
echo "  Batch:    $BATCH_SIZE"
echo "  Steps:    $STEPS"
echo ""

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${OUTPUT_REPO_ID}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size=${BATCH_SIZE} \
  --steps=${STEPS} \
  --num_workers=8 \
  --save_freq=5000 \
  --log_freq=50 \
  --output_dir=outputs/train/smolvla_box_to_bowl \
  --rename_map='{"observation.images.webcam": "observation.images.camera1", "observation.images.arm_cam": "observation.images.camera2"}'

echo ""
echo "=== Done! Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
