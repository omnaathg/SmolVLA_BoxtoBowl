#!/bin/bash
# ============================================================
# SmolVLA Training Script for RunPod (CUDA GPU)
# Task: Pick up the white plastic cup and place it in the blue bowl
#
# RECOMMENDED PODS (runpod.io):
#   - 1x A100 80GB  → batch_size=64, ~1.5 hrs
#   - 1x A40 48GB   → batch_size=32, ~2.5 hrs
#   - 1x 4090 24GB  → batch_size=16, ~3.5 hrs
#
# SETUP ON RUNPOD (run once after pod starts):
#   apt-get update && apt-get install -y ffmpeg
#   pip install "lerobot[smolvla]"
#   huggingface-cli login --token $HF_TOKEN
#
# Then run:
#   bash train_smolvla_runpod.sh
# ============================================================

export HF_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ─── CONFIGURE PER GPU ────────────────────────────────────────
DATASET_REPO_ID="RajatDandekar/so101_pour_chocolates"
OUTPUT_REPO_ID="RajatDandekar/smolvla_cup_bowl"
TASK="Pick up the white plastic cup and place it in the blue bowl"

# Tune BATCH_SIZE to your GPU VRAM:
#   A100 80GB → 64 | A40 48GB → 32 | 4090 24GB → 16
BATCH_SIZE=64
STEPS=20000
# ──────────────────────────────────────────────────────────────

echo "=== Step 1: Fix task description in dataset ==="
lerobot-edit-dataset \
  --repo_id="${DATASET_REPO_ID}" \
  --operation.type=modify_tasks \
  --operation.new_task="${TASK}" \
  --push_to_hub=true

echo ""
echo "=== Step 2: Train SmolVLA ==="
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
  --output_dir=outputs/train/smolvla_cup_bowl

echo ""
echo "=== Done! Model pushed to Hub: ${OUTPUT_REPO_ID} ==="
