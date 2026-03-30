#!/bin/zsh
# RLT Inference: VLA + RL Actor with spacebar switching
# Press SPACE to toggle between VLA and RL Actor
# Press Ctrl+C to stop

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

echo "=== RLT Inference ==="
echo "Press SPACE to switch between VLA and RL Actor"
echo ""

python scripts/run_rlt_inference.py \
  --smolvla_path RajatDandekar/smolvla_cup_bowl \
  --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
  --actor_checkpoint checkpoints/rlt_stage2/final_checkpoint.pt \
  --task "Pick up the white cup and place in the blue bowl" \
  --duration 120 \
  --fps 30 \
  --device mps \
  --follower_port /dev/tty.wchusbserial5AE60830811 \
  --camera_names webcam arm_cam
