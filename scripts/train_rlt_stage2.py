#!/usr/bin/env python3
"""Stage 2: Online RL Training on SO-101 with Human Intervention.

Trains a lightweight actor-critic using z_rl from the frozen RLT encoder.
The human provides two forms of feedback:

  1. **Sparse reward labels** — after each episode, human says success/fail
  2. **Intervention via teleop** — during an episode, human can grab the
     leader arm to override RL actions. These corrective demonstrations
     flow into the replay buffer as high-quality transitions, dramatically
     improving sample efficiency.

The intervention mechanism is key: without it, the agent only learns from
sparse binary signals. With it, the critic sees dense corrective demos
that show *what the robot should have done* at each failure state.

Usage:
    python scripts/train_rlt_stage2.py \
        --smolvla_path lerobot/smolvla_base \
        --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
        --task "pick up the block and place it in the bin" \
        --follower_port /dev/tty.usbmodem58760431541 \
        --leader_port /dev/tty.usbmodem58760431551 \
        --max_episodes 200 \
        --device mps

Hardware: M4 Pro Mac connected to SO-101 (follower + leader arms)
"""

import argparse
import json
import logging
import random
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """Single transition in the replay buffer."""
    z_rl: np.ndarray           # (rlt_hidden_dim,)
    state: np.ndarray          # (state_dim,)
    actions: np.ndarray        # (n_action_steps_rl, action_dim)
    ref_actions: np.ndarray    # (n_action_steps_rl, action_dim)
    reward: float
    next_z_rl: np.ndarray
    next_state: np.ndarray
    next_ref_actions: np.ndarray
    done: float
    is_intervention: bool = False  # True if this action came from human teleop


class ReplayBuffer:
    """Replay buffer with intervention-aware sampling."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return {
            "z_rl": torch.FloatTensor(np.stack([t.z_rl for t in batch])),
            "state": torch.FloatTensor(np.stack([t.state for t in batch])),
            "actions": torch.FloatTensor(np.stack([t.actions for t in batch])),
            "ref_actions": torch.FloatTensor(np.stack([t.ref_actions for t in batch])),
            "rewards": torch.FloatTensor([[t.reward] for t in batch]),
            "next_z_rl": torch.FloatTensor(np.stack([t.next_z_rl for t in batch])),
            "next_state": torch.FloatTensor(np.stack([t.next_state for t in batch])),
            "next_ref_actions": torch.FloatTensor(np.stack([t.next_ref_actions for t in batch])),
            "dones": torch.FloatTensor([[t.done] for t in batch]),
            "is_intervention": torch.FloatTensor([[float(t.is_intervention)] for t in batch]),
        }

    @property
    def intervention_count(self) -> int:
        return sum(1 for t in self.buffer if t.is_intervention)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Intervention detection
# ─────────────────────────────────────────────────────────────────────────────

class InterventionDetector:
    """Detects when a human is actively moving the leader arm.

    Monitors position deltas between consecutive leader readings. If any
    joint moves more than `threshold_degrees` between frames, the human
    is intervening. Uses a small temporal window to avoid single-frame noise.
    """

    # All joints including gripper
    MONITORED_JOINTS = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(
        self,
        threshold_degrees: float = 2.0,
        confirmation_frames: int = 2,
        release_frames: int = 150,
    ):
        """
        Args:
            threshold_degrees: Min joint delta (degrees) to trigger intervention
            confirmation_frames: Consecutive frames above threshold to confirm
            release_frames: Consecutive frames below threshold to release
        """
        self.threshold = threshold_degrees
        self.confirmation_frames = confirmation_frames
        self.release_frames = release_frames

        self._prev_positions: dict[str, float] | None = None
        self._active_count = 0      # Consecutive frames with motion detected
        self._inactive_count = 0    # Consecutive frames without motion
        self._is_intervening = False

    def update(self, leader_positions: dict[str, float]) -> bool:
        """Update with new leader reading. Returns True if human is intervening.

        Args:
            leader_positions: Dict from leader.get_action(), e.g.
                {"shoulder_pan.pos": 45.0, "shoulder_lift.pos": 90.0, ...}
        """
        if self._prev_positions is None:
            self._prev_positions = dict(leader_positions)
            return False

        # Check if any monitored joint moved significantly
        motion_detected = False
        for joint in self.MONITORED_JOINTS:
            key = f"{joint}.pos"
            if key in leader_positions and key in self._prev_positions:
                delta = abs(leader_positions[key] - self._prev_positions[key])
                if delta > self.threshold:
                    motion_detected = True
                    break

        self._prev_positions = dict(leader_positions)

        # State machine: require sustained motion/stillness to transition
        if motion_detected:
            self._active_count += 1
            self._inactive_count = 0
            if self._active_count >= self.confirmation_frames:
                if not self._is_intervening:
                    logger.info("  >> INTERVENTION DETECTED — switching to human teleop")
                self._is_intervening = True
        else:
            self._inactive_count += 1
            self._active_count = 0
            if self._inactive_count >= self.release_frames:
                if self._is_intervening:
                    logger.info("  >> Intervention ended — returning to RL agent")
                self._is_intervening = False

        return self._is_intervening

    def reset(self):
        """Reset state between episodes."""
        self._prev_positions = None
        self._active_count = 0
        self._inactive_count = 0
        self._is_intervening = False


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback
# ─────────────────────────────────────────────────────────────────────────────

def get_human_reward(had_intervention: bool) -> tuple[float, bool]:
    """Prompt human for episode outcome.

    If the human intervened during the episode, the episode is automatically
    considered a success (the human corrected it). Otherwise, ask.

    Returns:
        (reward, should_continue): reward is 1.0 (success) or 0.0 (fail),
        should_continue is False if user wants to quit.
    """
    if had_intervention:
        logger.info("  Episode had human intervention → auto-labeled SUCCESS (reward=1)")
        return 1.0, True

    while True:
        response = input("\n>>> Was this episode successful? [y/n/q(uit)]: ").strip().lower()
        if response in ("y", "yes", "1"):
            return 1.0, True
        elif response in ("n", "no", "0"):
            return 0.0, True
        elif response in ("q", "quit"):
            return 0.0, False
        print("  Please enter 'y', 'n', or 'q'")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RLT Stage 2: Online RL on SO-101 with Intervention")
    # Model paths
    parser.add_argument("--smolvla_path", type=str, required=True,
                        help="Path to pretrained SmolVLA checkpoint")
    parser.add_argument("--rlt_checkpoint", type=str, required=True,
                        help="Path to Stage 1 RLT encoder-decoder checkpoint")
    parser.add_argument("--task", type=str, default="pick up the block",
                        help="Task description for the VLA")
    # Robot connection
    parser.add_argument("--follower_port", type=str, required=True,
                        help="SO-101 follower arm serial port")
    parser.add_argument("--leader_port", type=str, required=True,
                        help="SO-101 leader arm serial port")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["front"],
                        help="Camera names matching training dataset")
    parser.add_argument("--fps", type=float, default=30.0)
    # Training
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--steps_per_episode", type=int, default=600,
                        help="Max steps per episode (at 30fps: 600 = 20 seconds)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage2")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--utd_ratio", type=int, default=5)
    parser.add_argument("--critic_per_actor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--bc_weight", type=float, default=0.1)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--warmup_episodes", type=int, default=5,
                        help="Episodes of pure VLA rollout before RL updates")
    parser.add_argument("--buffer_capacity", type=int, default=100_000)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Intervention
    parser.add_argument("--intervention_threshold", type=float, default=3.0,
                        help="Joint motion threshold (degrees) for intervention detection")
    parser.add_argument("--intervention_reward_bonus", type=float, default=0.5,
                        help="Per-step reward bonus for intervention transitions")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SHUTDOWN
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Load frozen SmolVLA (same pipeline as run_inference.sh / lerobot-record) ──
    logger.info(f"Loading SmolVLA from {args.smolvla_path}...")
    from copy import copy
    from contextlib import nullcontext

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector, resize_with_pad
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.control_utils import predict_action
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    # Camera rename map (webcam/arm_cam → camera1/camera2 as expected by the policy)
    rename_map = {
        "observation.images.webcam": "observation.images.camera1",
        "observation.images.arm_cam": "observation.images.camera2",
    }

    from lerobot.configs.types import FeatureType, PolicyFeature
    smolvla_config = SmolVLAConfig(load_vlm_weights=True)
    # Set features from the pretrained config (from_pretrained doesn't populate these)
    smolvla_config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    smolvla_config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    smolvla_policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path,
        config=smolvla_config,
    )
    smolvla_policy.to(device)
    smolvla_policy.eval()
    for p in smolvla_policy.parameters():
        p.requires_grad = False
    vla_model = smolvla_policy.model

    # Create a SEPARATE CPU copy of the VLA for z_rl embedding extraction
    # (MPS has bfloat16 matmul issues, so embeddings must run on CPU)
    logger.info("Creating CPU copy of SmolVLA for embedding extraction...")
    smolvla_cpu = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path,
        config=smolvla_config,
    )
    smolvla_cpu.to(torch.device("cpu"), dtype=torch.float32)
    smolvla_cpu.eval()
    vla_model_cpu = smolvla_cpu.model
    logger.info("CPU copy ready.")

    # Pre-cache tokenizer for z_rl computation (avoid repeated HF downloads)
    from transformers import AutoTokenizer
    _cached_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Load preprocessor/postprocessor from the pretrained checkpoint
    # (these contain the normalization stats needed to unnormalize actions)
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    smolvla_preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.smolvla_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    smolvla_postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=args.smolvla_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # ── Load RLT encoder (frozen from Stage 1) ─────────────────────────
    logger.info(f"Loading RLT encoder from {args.rlt_checkpoint}...")
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import (
        RLActorMLP,
        RLCriticMLP,
        RLTokenEncoder,
    )

    rlt_config = SmolVLARLTConfig(
        mode="online_rl",
        discount=args.discount,
        bc_weight=args.bc_weight,
        target_tau=args.target_tau,
    )

    from lerobot.configs.types import NormalizationMode
    torch.serialization.add_safe_globals([NormalizationMode])
    ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=True)
    encoder = RLTokenEncoder(rlt_config).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("RLT encoder loaded and frozen.")

    # ── Create actor + critic ───────────────────────────────────────────
    actor = RLActorMLP(rlt_config).to(device)
    critic = RLCriticMLP(rlt_config).to(device)
    critic_target = RLCriticMLP(rlt_config).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    logger.info(f"Actor: {actor_params:,} params, Critic: {critic_params:,} params (x2 twin)")

    # ── Replay buffer + intervention detector ──────────────────────────
    buffer = ReplayBuffer(args.buffer_capacity)
    intervention_detector = InterventionDetector(
        threshold_degrees=args.intervention_threshold,
    )

    # ── Connect to SO-101 ──────────────────────────────────────────────
    logger.info("Connecting to SO-101 arms...")
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    # Build camera configs for the follower
    camera_configs = {}
    for i, cam_name in enumerate(args.camera_names):
        camera_configs[cam_name] = OpenCVCameraConfig(
            index_or_path=i,
            fps=int(args.fps),
            width=640,
            height=480,
        )

    try:
        follower = SOFollower(SOFollowerRobotConfig(
            port=args.follower_port,
            id="my_follower",
            cameras=camera_configs,
        ))
        leader = SOLeader(SOLeaderTeleopConfig(
            port=args.leader_port,
            id="my_leader",
        ))
        follower.connect()
        leader.connect()
        logger.info("Both arms connected.")
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        logger.error("Check ports with: ls /dev/tty.usbmodem*")
        return

    # ── Joint name mapping ─────────────────────────────────────────────
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def leader_action_to_tensor(leader_action: dict) -> torch.Tensor:
        """Convert leader joint positions dict → (action_dim,) tensor."""
        values = [leader_action.get(f"{j}.pos", 0.0) for j in JOINT_NAMES]
        return torch.FloatTensor(values[:rlt_config.action_dim])

    def obs_to_state_tensor(obs: dict) -> torch.Tensor:
        """Convert robot observation dict → (state_dim,) tensor."""
        values = [obs.get(f"{j}.pos", 0.0) for j in JOINT_NAMES]
        return torch.FloatTensor(values[:rlt_config.state_dim])

    def tensor_to_robot_action(action_tensor: torch.Tensor) -> dict:
        """Convert action tensor → robot action dict. Handles any shape."""
        action_np = action_tensor.detach().cpu().numpy().flatten()
        return {f"{JOINT_NAMES[i]}.pos": float(action_np[i])
                for i in range(min(len(JOINT_NAMES), len(action_np)))}

    # Preprocessor handles tokenization — reset before use
    smolvla_policy.reset()
    smolvla_preprocessor.reset()
    smolvla_postprocessor.reset()

    # ── Rerun visualization ──────────────────────────────────────────────
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    init_rerun(session_name="rlt_stage2")
    logger.info("Rerun viewer launched for live camera feeds.")

    # ── Keyboard listener (right arrow = next episode, left arrow = quit) ──
    from lerobot.utils.control_utils import init_keyboard_listener
    keyboard_listener, events = init_keyboard_listener()

    # ── Training loop ───────────────────────────────────────────────────
    logger.info(f"\nStarting online RL for {args.max_episodes} episodes...")
    logger.info(f"Intervention threshold: {args.intervention_threshold} degrees")
    logger.info(f"Warmup episodes (pure VLA): {args.warmup_episodes}")
    logger.info("Controls: RIGHT arrow = next episode, LEFT arrow = quit")
    logger.info(f"Grab the leader arm at any time to intervene!\n")

    episode_rewards = []
    episode_interventions = []
    total_updates = 0
    target_dt = 1.0 / args.fps

    for episode in range(1, args.max_episodes + 1):
        if SHUTDOWN:
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}/{args.max_episodes}")
        logger.info(f"{'='*60}")

        intervention_detector.reset()
        episode_had_intervention = False
        intervention_steps = 0

        # Reset policy and processors each episode (clears action queue)
        smolvla_policy.reset()
        smolvla_preprocessor.reset()
        smolvla_postprocessor.reset()

        # Save raw observations for post-episode z_rl batch computation
        episode_obs_history = []  # list of (obs_snapshot, state, is_intervention)

        for step in range(args.steps_per_episode):
            if SHUTDOWN:
                break
            # Esc → stop training entirely
            if events.get("stop_recording", False):
                logger.info("  >> Escape pressed — ending training")
                SHUTDOWN = True
                break
            # Right or Left arrow → end this episode early
            if events.get("exit_early", False):
                events["exit_early"] = False
                events["rerecord_episode"] = False
                logger.info("  >> Arrow key pressed — ending episode early")
                break
            step_start = time.time()

            # ── 1. Get observation from robot ───────────────────────────
            obs = follower.get_observation()
            state_tensor = obs_to_state_tensor(obs).to(device)

            # Log camera feeds to rerun
            log_rerun_data(observation=obs)

            # ── 2. Read leader arm (for intervention detection) ─────────
            leader_action = leader.get_action()
            is_intervening = intervention_detector.update(leader_action)

            if is_intervening:
                episode_had_intervention = True
                intervention_steps += 1

            # ── 3. Build observation frame exactly like lerobot-record ──
            observation_frame = {}
            for cam_name in args.camera_names:
                if cam_name in obs:
                    observation_frame[f"observation.images.{cam_name}"] = obs[cam_name]

            state_names = [f"{j}.pos" for j in JOINT_NAMES]
            observation_frame["observation.state"] = np.array(
                [obs.get(name, 0.0) for name in state_names], dtype=np.float32
            )

            # ── 4. Decide action: RL actor vs human intervention ────────
            if is_intervening:
                # HUMAN OVERRIDE: blend from current follower position to leader
                follower_pos = follower.get_observation()
                leader_target = {k: v for k, v in leader_action.items() if k.endswith(".pos")}
                blend_alpha = min(intervention_steps / 10.0, 1.0)
                blended_action = {}
                for key in leader_target:
                    current = follower_pos.get(key, leader_target[key])
                    target = leader_target[key]
                    blended_action[key] = current + blend_alpha * (target - current)
                follower.send_action(blended_action)
                smolvla_policy.reset()
                action_source = "HUMAN"
            else:
                # Use predict_action — the EXACT same function lerobot-record uses
                action_values = predict_action(
                    observation=observation_frame,
                    policy=smolvla_policy,
                    device=device,
                    preprocessor=smolvla_preprocessor,
                    postprocessor=smolvla_postprocessor,
                    use_amp=False,
                    task=args.task,
                    robot_type="so101_follower",
                )
                robot_action = tensor_to_robot_action(action_values)
                follower.send_action(robot_action)
                action_source = "VLA" if episode <= args.warmup_episodes else "ACTOR"

            # ── 5. Save snapshot for post-episode z_rl computation ──────
            # Only save every N steps to limit memory (subsample)
            if step % 50 == 0:
                obs_snapshot = {
                    "images": {cam: obs[cam].copy() for cam in args.camera_names if cam in obs and isinstance(obs[cam], np.ndarray)},
                    "state": observation_frame["observation.state"].copy(),
                    "is_intervention": is_intervening,
                }
                episode_obs_history.append(obs_snapshot)

            # Logging
            if step % 10 == 0:
                logger.info(
                    f"  Step {step}/{args.steps_per_episode} | "
                    f"Source: {action_source} | "
                    f"Interventions so far: {intervention_steps}"
                )

            # Maintain target FPS
            elapsed = time.time() - step_start
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ── End of episode ──────────────────────────────────────────────
        if SHUTDOWN:
            break

        # Get human reward label
        if episode <= args.warmup_episodes:
            # Warmup episodes are just baseline — auto-label, no prompt
            logger.info("  Warmup episode complete — auto-labeled (no reward)")
            reward, should_continue = 0.0, True
        else:
            reward, should_continue = get_human_reward(episode_had_intervention)

        episode_rewards.append(reward)
        episode_interventions.append(intervention_steps)

        # ── Post-episode: batch compute z_rl and build transitions ─────
        episode_transitions = []
        if episode > args.warmup_episodes and len(episode_obs_history) >= 2:
            logger.info(f"  Computing z_rl for {len(episode_obs_history)} snapshots on CPU...")
            cpu = torch.device("cpu")

            # Pre-tokenize task once
            task_text = args.task if args.task.endswith("\n") else args.task + "\n"
            _tok = _cached_tokenizer(
                task_text, return_tensors="pt", padding="max_length",
                max_length=rlt_config.tokenizer_max_length, truncation=True,
            )
            cpu_lang = _tok["input_ids"]
            cpu_lang_mask = _tok["attention_mask"].bool()

            z_rl_list = []
            ref_actions_list = []
            state_list = []

            with torch.inference_mode():
                for snap in episode_obs_history:
                    # Build CPU inputs
                    cpu_images = []
                    cpu_masks = []
                    for cam_name in args.camera_names:
                        if cam_name in snap["images"]:
                            img = torch.from_numpy(snap["images"][cam_name]).float()
                            img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
                            img = resize_with_pad(img, *smolvla_policy.config.resize_imgs_with_padding, pad_value=0)
                            img = img * 2.0 - 1.0
                            cpu_images.append(img)
                            cpu_masks.append(torch.ones(1, dtype=torch.bool))

                    cpu_state = torch.from_numpy(snap["state"]).float().unsqueeze(0)
                    cpu_state_padded = pad_vector(cpu_state, smolvla_policy.config.max_state_dim)

                    # Run VLA on CPU
                    ref_actions_emb = vla_model_cpu.sample_actions(
                        cpu_images, cpu_masks, cpu_lang, cpu_lang_mask, cpu_state_padded,
                    )
                    dummy_actions = pad_vector(ref_actions_emb, rlt_config.max_action_dim)
                    vlm_emb, expert_emb = vla_model_cpu.extract_embeddings(
                        cpu_images, cpu_masks, cpu_lang, cpu_lang_mask, cpu_state_padded, dummy_actions,
                    )
                    z_rl = encoder(vlm_emb.to(device).float(), expert_emb.to(device).float())

                    ref_actions_rl = ref_actions_emb[:, :, :rlt_config.action_dim]
                    stride = rlt_config.action_stride
                    ref_sub = ref_actions_rl[:, ::stride, :][:, :rlt_config.n_action_steps_rl, :]

                    z_rl_list.append(z_rl.cpu().numpy().squeeze())
                    ref_actions_list.append(ref_sub.cpu().numpy().squeeze())
                    # Pad state to rlt_config.state_dim (7) if robot has fewer joints (6)
                    raw_state = snap["state"]
                    if len(raw_state) < rlt_config.state_dim:
                        raw_state = np.pad(raw_state, (0, rlt_config.state_dim - len(raw_state)))
                    state_list.append(raw_state[:rlt_config.state_dim])

            logger.info(f"  z_rl computation done. Building transitions...")

            # Build transitions from consecutive snapshots
            for i in range(len(z_rl_list) - 1):
                is_terminal = (i == len(z_rl_list) - 2)
                snap = episode_obs_history[i]

                if is_terminal:
                    step_reward = reward
                elif snap["is_intervention"]:
                    step_reward = args.intervention_reward_bonus
                else:
                    step_reward = 0.0

                transition = Transition(
                    z_rl=z_rl_list[i],
                    state=state_list[i],
                    actions=ref_actions_list[i],  # Use ref actions as executed actions
                    ref_actions=ref_actions_list[i],
                    reward=step_reward,
                    next_z_rl=z_rl_list[i + 1],
                    next_state=state_list[i + 1],
                    next_ref_actions=ref_actions_list[i + 1],
                    done=1.0 if is_terminal else 0.0,
                    is_intervention=snap["is_intervention"],
                )
                episode_transitions.append(transition)
                buffer.add(transition)

            logger.info(f"  Added {len(episode_transitions)} transitions to buffer (total: {len(buffer)})")

        # ── RL Updates ──────────────────────────────────────────────────
        if episode > args.warmup_episodes and len(buffer) >= args.batch_size:
            n_updates = args.utd_ratio * len(episode_transitions)
            if n_updates > 0:
                logger.info(f"  Running {n_updates} RL updates...")

                for update_idx in range(n_updates):
                    batch = buffer.sample(args.batch_size)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # ── Critic update ───────────────────────────────────
                    with torch.no_grad():
                        next_action_mean, _ = actor(
                            batch["next_z_rl"], batch["next_state"],
                            batch["next_ref_actions"]
                        )
                        target_q = critic_target.q_min(
                            batch["next_z_rl"], batch["next_state"], next_action_mean
                        )
                        td_target = (
                            batch["rewards"]
                            + args.discount * (1.0 - batch["dones"]) * target_q
                        )

                    q1, q2 = critic(batch["z_rl"], batch["state"], batch["actions"])
                    critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    # ── Actor update (every critic_per_actor steps) ─────
                    if update_idx % args.critic_per_actor == 0:
                        action_mean, _ = actor(
                            batch["z_rl"], batch["state"],
                            batch["ref_actions"], training=True
                        )
                        q_value = critic.q_min(
                            batch["z_rl"].detach(), batch["state"], action_mean
                        )
                        q_loss = -q_value.mean()

                        # BC regularization: For intervention transitions,
                        # regularize toward the human's action (not just ref).
                        # For non-intervention, regularize toward VLA ref.
                        bc_target = batch["ref_actions"].clone()
                        interv_mask = batch["is_intervention"].bool().squeeze(-1)
                        if interv_mask.any():
                            # For intervention transitions, the stored "actions"
                            # ARE the human's corrective actions — use those
                            bc_target[interv_mask] = batch["actions"][interv_mask]

                        bc_loss = F.mse_loss(action_mean, bc_target)
                        actor_loss = q_loss + args.bc_weight * bc_loss

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                    # ── EMA target update ───────────────────────────────
                    tau = args.target_tau
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

                    total_updates += 1

                logger.info(
                    f"  Updates done. Critic loss: {critic_loss.item():.4f}, "
                    f"Q mean: {q1.mean().item():.4f}"
                )

        # ── Logging ─────────────────────────────────────────────────────
        recent_rewards = episode_rewards[-20:]
        success_rate = sum(recent_rewards) / len(recent_rewards)
        recent_interventions = episode_interventions[-20:]
        avg_interventions = sum(recent_interventions) / len(recent_interventions)

        logger.info(
            f"  Episode {episode} | Reward: {reward} | "
            f"Interventions: {intervention_steps} steps | "
            f"Success (last 20): {success_rate:.0%} | "
            f"Avg interventions (last 20): {avg_interventions:.1f} | "
            f"Buffer: {len(buffer)} ({buffer.intervention_count} from human) | "
            f"Updates: {total_updates}"
        )

        # ── Save checkpoint ─────────────────────────────────────────────
        if episode % args.save_every == 0:
            ckpt = {
                "episode": episode,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "critic_target_state_dict": critic_target.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "episode_rewards": episode_rewards,
                "episode_interventions": episode_interventions,
                "total_updates": total_updates,
            }
            ckpt_path = output_dir / f"checkpoint_ep{episode}.pt"
            torch.save(ckpt, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")

        if not should_continue:
            logger.info("User requested quit.")
            break

    # ── Cleanup ─────────────────────────────────────────────────────────
    logger.info("\nDisconnecting from robot...")
    try:
        follower.disconnect()
        leader.disconnect()
    except Exception:
        pass

    # Final save
    final_ckpt = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "critic_target_state_dict": critic_target.state_dict(),
        "episode_rewards": episode_rewards,
        "episode_interventions": episode_interventions,
        "total_updates": total_updates,
    }
    torch.save(final_ckpt, output_dir / "final_checkpoint.pt")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "episode_rewards": episode_rewards,
            "episode_interventions": episode_interventions,
            "total_updates": total_updates,
        }, f, indent=2)

    total_episodes = len(episode_rewards)
    total_interventions = sum(episode_interventions)
    logger.info(f"\nTraining complete!")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Total updates: {total_updates}")
    logger.info(f"  Total intervention steps: {total_interventions}")
    logger.info(f"  Final success rate (last 20): "
                f"{sum(episode_rewards[-20:]) / max(1, len(episode_rewards[-20:])):.0%}")
    logger.info(f"  Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
