#!/bin/zsh
# RLT Stage 2: Online RL with human intervention on SO-101
# The robot attempts the task, you correct it by grabbing the leader arm when needed.
# Usage: ./run_rlt_stage2.sh

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

echo "=== RLT Stage 2: Online RL with Intervention ==="
echo "Grab the leader arm at any time to correct the robot!"
echo ""

python scripts/train_rlt_stage2.py \
  --smolvla_path RajatDandekar/smolvla_cup_bowl \
  --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
  --task "Pick up the white cup and place in the blue bowl" \
  --follower_port /dev/tty.wchusbserial5AE60830811 \
  --leader_port /dev/tty.wchusbserial5A7C1167331 \
  --camera_names webcam arm_cam \
  --max_episodes 200 \
  --warmup_episodes 5 \
  --device mps \
  --fps 30 \
  --steps_per_episode 3000 \
  --output_dir checkpoints/rlt_stage2 \
  --save_every 20
