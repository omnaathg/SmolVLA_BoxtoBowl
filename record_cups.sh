#!/bin/zsh
# Record SO-101 episodes: Pick up paper cups and place them in the blue bowl
# Usage: ./record_cups.sh

export HF_TOKEN=$HF_TOKEN

source "$(dirname "$0")/activate.sh"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.wchusbserial5AE60830811 \
  --robot.id=my_follower \
  --robot.cameras='{"webcam": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "arm_cam": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.wchusbserial5A7C1167331 \
  --teleop.id=my_leader \
  --dataset.repo_id=RajatDandekar/so101_cups_to_bowl \
  --dataset.single_task="Pick up the paper cup and place it in the blue bowl" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.num_episodes=10 \
  --display_data=true
