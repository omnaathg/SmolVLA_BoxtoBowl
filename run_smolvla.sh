#!/bin/zsh
# Simple SmolVLA inference with fixed task.
# Type a new task in the terminal and press Enter to change it.
# Type "q" to quit.
#
# Usage:
#   ./run_smolvla.sh                                          # default task
#   ./run_smolvla.sh "Pick up the white cup and place in the blue bowl"  # custom task

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

TASK="${1:-Pick up the box and place it in the red bowl}"

echo "=== SmolVLA Inference ==="
echo "Task: ${TASK}"
echo "Type a new task + Enter to change. Type 'q' to quit."
echo ""

python voice_inference.py \
  --policy_path RajatDandekar/smolvla_box_to_bowl \
  --initial_task "${TASK}" \
  --text_mode \
  --duration 3600
