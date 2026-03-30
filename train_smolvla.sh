#!/bin/zsh
# Train SmolVLA on the cup-to-bowl dataset
# Task: Pick up the white plastic cup and place it in the blue bowl
# Usage: ./train_smolvla.sh

export HF_TOKEN=$HF_TOKEN

# ─── CONFIGURE THIS ───────────────────────────────────────────────────────────
# Set to your HuggingFace dataset repo (e.g. RajatDandekar/so101_cup_bowl)
DATASET_REPO_ID="RajatDandekar/so101_pour_chocolates"

# Where to save the fine-tuned model on the Hub
OUTPUT_REPO_ID="RajatDandekar/smolvla_cup_bowl"

# Correct task description for language conditioning
TASK="Pick up the white plastic cup and place it in the blue bowl"
# ──────────────────────────────────────────────────────────────────────────────

source "$(dirname "$0")/activate.sh"

echo "=== Step 1: Fix task description in the dataset ==="
echo "Setting task to: '${TASK}'"
lerobot-edit-dataset \
  --repo_id="${DATASET_REPO_ID}" \
  --operation.type=modify_tasks \
  --operation.new_task="${TASK}" \
  --push_to_hub=true

echo ""
echo "=== Step 2: Train SmolVLA (fine-tune from smolvla_base) ==="
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${OUTPUT_REPO_ID}" \
  --policy.device=mps \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=100 \
  --output_dir=outputs/train/smolvla_cup_bowl

echo ""
echo "=== Training complete! Model saved to: outputs/train/smolvla_cup_bowl ==="
echo "=== Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
