#!/usr/bin/env python3
"""RLT Inference on SO-101.

Matches the paper's strategy (Algorithm 1, line 9):
  - VLA runs the easy parts (approach, positioning)
  - Press SPACE to switch to RL actor for the critical phase
  - Press SPACE again to switch back to VLA
  - Press Ctrl+C to stop

Usage:
    python scripts/run_rlt_inference.py \
        --smolvla_path RajatDandekar/smolvla_cup_bowl \
        --rlt_checkpoint checkpoints/rlt_stage1/best_checkpoint.pt \
        --actor_checkpoint checkpoints/rlt_stage2/final_checkpoint.pt \
        --task "Pick up the white cup and place in the blue bowl" \
        --duration 120 \
        --device mps
"""

import argparse
import logging
import signal
import time
import threading

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHUTDOWN = False


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received.")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Inference on SO-101")
    parser.add_argument("--smolvla_path", type=str, required=True)
    parser.add_argument("--rlt_checkpoint", type=str, required=True)
    parser.add_argument("--actor_checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="Pick up the white cup and place in the blue bowl")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--follower_port", type=str, default="/dev/tty.wchusbserial5AE60830811")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["webcam", "arm_cam"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    # ── Load SmolVLA ───────────────────────────────────────────────────
    logger.info(f"Loading SmolVLA from {args.smolvla_path}...")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector, resize_with_pad
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    from copy import copy

    import lerobot.policies.smolvla.processor_smolvla  # noqa: registers processor

    smolvla_config = SmolVLAConfig(load_vlm_weights=True)
    smolvla_config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    smolvla_config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    smolvla_policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path, config=smolvla_config,
    )
    smolvla_policy.to(device).eval()

    # Preprocessor / postprocessor
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    rename_map = {
        "observation.images.webcam": "observation.images.camera1",
        "observation.images.arm_cam": "observation.images.camera2",
    }
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.smolvla_path, f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": args.device}, "rename_observations_processor": {"rename_map": rename_map}},
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.smolvla_path, f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        overrides={}, to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )

    # ── Load RLT encoder + actor ───────────────────────────────────────
    logger.info("Loading RLT encoder + actor...")
    from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
    from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP, RLTokenEncoder
    from lerobot.configs.types import NormalizationMode
    torch.serialization.add_safe_globals([NormalizationMode])

    rlt_config = SmolVLARLTConfig(mode="inference")
    rlt_ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=True)
    encoder = RLTokenEncoder(rlt_config).to(device)
    encoder.load_state_dict(rlt_ckpt["encoder_state_dict"])
    encoder.eval()

    actor = RLActorMLP(rlt_config).to(device)
    actor_ckpt = torch.load(args.actor_checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(actor_ckpt["actor_state_dict"])
    actor.eval()
    logger.info("RLT actor loaded.")

    # CPU copy of VLA for z_rl (avoids MPS bfloat16 crash)
    logger.info("Creating CPU VLA copy for z_rl...")
    smolvla_cpu = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.smolvla_path, config=smolvla_config,
    )
    smolvla_cpu.to(torch.device("cpu"), dtype=torch.float32).eval()
    vla_model_cpu = smolvla_cpu.model

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    logger.info("CPU VLA copy ready.")

    # ── Connect to robot ───────────────────────────────────────────────
    logger.info("Connecting to SO-101...")
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    camera_configs = {
        cam: OpenCVCameraConfig(index_or_path=i, fps=int(args.fps), width=640, height=480)
        for i, cam in enumerate(args.camera_names)
    }
    follower = SOFollower(SOFollowerRobotConfig(
        port=args.follower_port, id="my_follower", cameras=camera_configs,
    ))
    follower.connect()
    logger.info("Robot connected.")

    # ── Rerun + keyboard ───────────────────────────────────────────────
    init_rerun(session_name="rlt_inference")

    # Thread-safe switch state
    use_actor = False
    switch_lock = threading.Lock()

    def keyboard_listener():
        """Listen for spacebar to toggle VLA ↔ Actor."""
        nonlocal use_actor
        try:
            from pynput import keyboard
            def on_press(key):
                nonlocal use_actor
                if key == keyboard.Key.space:
                    with switch_lock:
                        use_actor = not use_actor
                    mode = "RL ACTOR" if use_actor else "VLA"
                    print(f"\n>>> SWITCHED TO: {mode}\n")
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
        except Exception:
            logger.warning("Keyboard listener unavailable. Use terminal input instead.")

    keyboard_listener()

    # ── Helper ─────────────────────────────────────────────────────────
    def tensor_to_robot_action(action_tensor):
        action_np = action_tensor.detach().cpu().numpy().flatten()
        return {f"{JOINT_NAMES[i]}.pos": float(action_np[i])
                for i in range(min(len(JOINT_NAMES), len(action_np)))}

    # ── Inference loop ─────────────────────────────────────────────────
    smolvla_policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    target_dt = 1.0 / args.fps
    start_time = time.time()
    frame_count = 0
    n_action_steps = smolvla_policy.config.n_action_steps

    # Actor state: z_rl + ref_actions, recomputed each action chunk
    cached_z_rl = None
    cached_ref_actions_sub = None
    actor_action_queue = []  # Actor outputs C action steps
    actor_queue_idx = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"  RLT INFERENCE")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Duration: {args.duration}s at {args.fps} FPS")
    logger.info(f"  Press SPACE to toggle between VLA and RL Actor")
    logger.info(f"  Press Ctrl+C to stop")
    logger.info(f"{'='*60}\n")
    logger.info("Starting in VLA mode...")

    while not SHUTDOWN:
        loop_start = time.time()
        elapsed = loop_start - start_time
        if elapsed >= args.duration:
            logger.info(f"Duration limit ({args.duration}s) reached.")
            break

        # ── 1. Get observation ─────────────────────────────────────────
        obs = follower.get_observation()
        log_rerun_data(observation=obs)

        observation_frame = {}
        for cam_name in args.camera_names:
            if cam_name in obs:
                observation_frame[f"observation.images.{cam_name}"] = obs[cam_name]
        state_names = [f"{j}.pos" for j in JOINT_NAMES]
        observation_frame["observation.state"] = np.array(
            [obs.get(name, 0.0) for name in state_names], dtype=np.float32
        )

        # ── 2. Check mode ─────────────────────────────────────────────
        with switch_lock:
            currently_using_actor = use_actor

        if not currently_using_actor:
            # ── VLA MODE: exact same as lerobot-record ─────────────────
            action_values = predict_action(
                observation=observation_frame,
                policy=smolvla_policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
                task=args.task,
                robot_type="so101_follower",
            )
            robot_action = tensor_to_robot_action(action_values)
            follower.send_action(robot_action)
            # Reset actor queue when switching back
            actor_action_queue = []
            actor_queue_idx = 0
            mode_str = "VLA"

        else:
            # ── RL ACTOR MODE: actor takes full control ────────────────
            # Compute z_rl + ref_actions at each action chunk boundary
            if len(actor_action_queue) == 0 or actor_queue_idx >= len(actor_action_queue):
                cpu = torch.device("cpu")
                with torch.inference_mode():
                    # Build CPU inputs
                    cpu_images = []
                    cpu_masks = []
                    for cam_name in args.camera_names:
                        if cam_name in obs and isinstance(obs[cam_name], np.ndarray):
                            img = torch.from_numpy(obs[cam_name]).float()
                            img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
                            img = resize_with_pad(img, *smolvla_policy.config.resize_imgs_with_padding, pad_value=0)
                            img = img * 2.0 - 1.0
                            cpu_images.append(img)
                            cpu_masks.append(torch.ones(1, dtype=torch.bool))

                    cpu_state = torch.tensor(
                        [obs.get(f"{j}.pos", 0.0) for j in JOINT_NAMES], dtype=torch.float32
                    ).unsqueeze(0)
                    cpu_state_padded = pad_vector(cpu_state, smolvla_policy.config.max_state_dim)

                    task_text = args.task if args.task.endswith("\n") else args.task + "\n"
                    _tok = _tokenizer(task_text, return_tensors="pt", padding="max_length",
                                      max_length=rlt_config.tokenizer_max_length, truncation=True)

                    # VLA reference actions on CPU
                    ref_actions_emb = vla_model_cpu.sample_actions(
                        cpu_images, cpu_masks, _tok["input_ids"], _tok["attention_mask"].bool(), cpu_state_padded,
                    )
                    ref_actions_rl = ref_actions_emb[:, :, :rlt_config.action_dim]
                    stride = rlt_config.action_stride
                    ref_sub = ref_actions_rl[:, ::stride, :][:, :rlt_config.n_action_steps_rl, :].to(device).float()

                    # z_rl
                    dummy_actions = pad_vector(ref_actions_emb, rlt_config.max_action_dim)
                    vlm_emb, expert_emb = vla_model_cpu.extract_embeddings(
                        cpu_images, cpu_masks, _tok["input_ids"], _tok["attention_mask"].bool(),
                        cpu_state_padded, dummy_actions,
                    )
                    z_rl = encoder(vlm_emb.to(device).float(), expert_emb.to(device).float())

                    # Actor: full action chunk (replaces VLA, not blended)
                    raw_state = np.array([obs.get(f"{j}.pos", 0.0) for j in JOINT_NAMES], dtype=np.float32)
                    if len(raw_state) < rlt_config.state_dim:
                        raw_state = np.pad(raw_state, (0, rlt_config.state_dim - len(raw_state)))
                    state_rl = torch.from_numpy(raw_state[:rlt_config.state_dim]).float().unsqueeze(0).to(device)

                    actor_output, _ = actor(z_rl, state_rl, ref_sub)
                    # actor_output: (1, n_action_steps_rl, action_dim)
                    # Unnormalize using postprocessor (each step)
                    actor_action_queue = []
                    for step_idx in range(actor_output.shape[1]):
                        act_step = actor_output[0, step_idx, :6]  # First 6 dims for our robot
                        act_unnorm = postprocessor(act_step)
                        actor_action_queue.append(act_unnorm)
                    actor_queue_idx = 0

                    # Also reset VLA queue so it re-plans when switching back
                    smolvla_policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

            # Pop next action from actor queue
            if actor_queue_idx < len(actor_action_queue):
                actor_action = actor_action_queue[actor_queue_idx]
                # Safety clamp
                actor_action = actor_action.clamp(-300.0, 300.0)
                robot_action = tensor_to_robot_action(actor_action)
                follower.send_action(robot_action)
                actor_queue_idx += 1
            mode_str = "ACTOR"

        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count} | {elapsed:.1f}s | Mode: {mode_str}")

        loop_dt = time.time() - loop_start
        sleep_time = max(0, target_dt - loop_dt)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Cleanup ────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    actual_fps = frame_count / max(total_time, 1e-6)
    logger.info(f"Done. {frame_count} frames in {total_time:.1f}s ({actual_fps:.1f} FPS)")
    follower.disconnect()


if __name__ == "__main__":
    main()
