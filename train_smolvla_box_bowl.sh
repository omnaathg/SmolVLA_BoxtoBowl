#!/bin/zsh
# =============================================================================
# Train SmolVLA on box-to-bowl dataset (language-conditioned)
# Task: Pick up box → place in red/green/blue bowl based on instruction
#
# Local training on Mac M4 (MPS)
# Usage: ./train_smolvla_box_bowl.sh
# =============================================================================

export HF_TOKEN=$HF_TOKEN

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_box_to_bowl"
OUTPUT_REPO_ID="RajatDandekar/smolvla_box_to_bowl"
# ─────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/activate.sh"

echo "=== Training SmolVLA on box-to-bowl (multi-task) dataset ==="
echo "  Dataset:  $DATASET_REPO_ID"
echo "  Output:   $OUTPUT_REPO_ID"
echo ""

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${OUTPUT_REPO_ID}" \
  --policy.device=mps \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir=outputs/train/smolvla_box_to_bowl \
  --rename_map='{"observation.images.webcam": "observation.images.camera1", "observation.images.arm_cam": "observation.images.camera2"}'

echo ""
echo "=== Training complete! ==="
echo "=== Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
