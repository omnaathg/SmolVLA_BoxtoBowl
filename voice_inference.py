#!/usr/bin/env python3
"""
Continuous voice-controlled SmolVLA inference for SO-101.

The robot runs autonomously for up to 60 minutes. A background thread
continuously listens for voice commands. When you speak a new instruction,
the task updates live and the robot's behavior changes on the very next
control step — no stopping, no episode boundaries.

SmolVLA's VLM backbone grounds your instruction semantically:
  "put the cube in the bowl colored like blood"  → red bowl
  "move it to the sky-colored bowl"              → blue bowl
  "place it in the bowl that looks like grass"   → green bowl

Usage:
    python voice_inference.py                          # voice mode, 60 min
    python voice_inference.py --duration 3600          # 60 min (default)
    python voice_inference.py --text_mode              # keyboard instead of mic
    python voice_inference.py --whisper_model small    # more accurate STT
    python voice_inference.py --initial_task "Pick up the box and place it in the red bowl"

Dependencies (install once):
    python -m pip install openai-whisper sounddevice
"""

import argparse
import logging
import threading
import time

import numpy as np
import torch


# ─── Shared state between control loop and voice thread ─────────────────────
class TaskState:
    """Thread-safe container for the current language instruction."""

    def __init__(self, initial_task: str = ""):
        self._task = initial_task
        self._lock = threading.Lock()
        self._change_count = 0

    @property
    def task(self) -> str:
        with self._lock:
            return self._task

    @task.setter
    def task(self, new_task: str):
        with self._lock:
            if new_task and new_task != self._task:
                self._task = new_task
                self._change_count += 1
                print(f"\n  >> TASK UPDATED ({self._change_count}): \"{new_task}\"")

    @property
    def change_count(self) -> int:
        with self._lock:
            return self._change_count


# ─── Background voice listener ──────────────────────────────────────────────
def find_microphone():
    """Find the best available microphone (prefers Kreo Owl webcam mic)."""
    import sounddevice as sd
    mic_device = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            if "kreo" in d["name"].lower() or "owl" in d["name"].lower():
                mic_device = i
            elif mic_device is None and "benq" not in d["name"].lower():
                mic_device = i
    if mic_device is None:
        mic_device = sd.default.device[0]
    name = sd.query_devices(mic_device)["name"]
    print(f"  Microphone: [{mic_device}] {name}")
    return mic_device


def voice_listener_thread(
    task_state: TaskState,
    whisper_model,
    listen_duration: float,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    stop_event: threading.Event = None,
):
    """Continuously listen for voice commands and update the task."""
    import sounddevice as sd

    mic_device = find_microphone()
    print("  Voice listener active. Speak anytime to change the task.")

    while not stop_event.is_set():
        try:
            # Record audio
            audio = sd.rec(
                int(listen_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=mic_device,
            )
            sd.wait()

            if stop_event.is_set():
                break

            audio = audio.flatten().astype(np.float32)

            # Skip if audio is mostly silence (no one spoke)
            rms = np.sqrt(np.mean(audio**2))
            if rms < silence_threshold:
                continue

            # Transcribe — pass numpy array directly (no ffmpeg needed)
            result = whisper_model.transcribe(audio, language="en", fp16=False)

            text = result["text"].strip()

            # Filter out Whisper hallucinations on silence/noise
            if not text or len(text) < 5:
                continue
            hallucination_phrases = [
                "thank you", "thanks for watching", "subscribe",
                "you", "bye", "okay", "the end", "...",
            ]
            if text.lower().strip(".!? ") in hallucination_phrases:
                continue

            # Update the task
            task_state.task = text

        except Exception as e:
            print(f"  [Voice thread error: {e}]")
            time.sleep(1)


# ─── Background keyboard listener (text mode) ───────────────────────────────
def keyboard_listener_thread(
    task_state: TaskState,
    stop_event: threading.Event,
):
    """Read instructions from stdin in a loop."""
    import sys
    import select

    print("  Keyboard mode active. Type a new instruction and press Enter anytime.")
    print("  Type 'quit' to stop.\n")

    while not stop_event.is_set():
        try:
            if select.select([sys.stdin], [], [], 1.0)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() in ("quit", "exit", "stop", "q"):
                    stop_event.set()
                    break
                if line:
                    task_state.task = line
        except Exception:
            break


def main():
    parser = argparse.ArgumentParser(description="Continuous voice-controlled SmolVLA inference")
    parser.add_argument("--policy_path", type=str, default="RajatDandekar/smolvla_box_to_bowl")
    parser.add_argument("--dataset_repo_id", type=str, default="RajatDandekar/so101_box_to_bowl")
    parser.add_argument("--follower_port", type=str, default="/dev/tty.wchusbserial5AE60830811")
    parser.add_argument("--leader_port", type=str, default="/dev/tty.wchusbserial5A7C1167331")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3600, help="Total run time in seconds (default: 3600 = 60 min)")
    parser.add_argument("--whisper_model", type=str, default="base", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--listen_duration", type=float, default=8.0, help="Seconds per voice capture window")
    parser.add_argument("--silence_threshold", type=float, default=0.01, help="RMS threshold below which audio is treated as silence")
    parser.add_argument("--text_mode", action="store_true", help="Type instructions instead of speaking")
    parser.add_argument("--initial_task", type=str, default="Pick up the box and place it in the red bowl",
                        help="Starting task instruction")
    args = parser.parse_args()

    # ── Imports from lerobot ──
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.utils import init_logging

    init_logging()

    # ── Camera rename map (dataset uses webcam/arm_cam, policy expects camera1/camera2) ──
    rename_map = {
        "observation.images.webcam": "observation.images.camera1",
        "observation.images.arm_cam": "observation.images.camera2",
    }

    # ── Load Whisper ──
    if not args.text_mode:
        import whisper
        print(f"Loading Whisper '{args.whisper_model}' model...")
        whisper_model = whisper.load_model(args.whisper_model)
        print("Whisper ready.")
    else:
        whisper_model = None

    # ── Load dataset metadata (for feature shapes and normalization stats) ──
    print(f"Loading dataset metadata from {args.dataset_repo_id}...")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    # ── Load policy config from pretrained model ──
    print(f"Loading policy from {args.policy_path}...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device

    # ── Create policy with proper dataset metadata ──
    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)

    # ── Create pre/post processors ──
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    device = get_safe_torch_device(args.device)

    # ── Build robot config with proper LeRobot types ──
    print("Connecting to robot...")
    robot_cfg = SOFollowerRobotConfig(
        port=args.follower_port,
        id="my_follower",
        cameras={
            "webcam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            "arm_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        },
    )
    robot = make_robot_from_config(robot_cfg)

    teleop_cfg = SOLeaderTeleopConfig(
        port=args.leader_port,
        id="my_leader",
    )
    teleop = make_teleoperator_from_config(teleop_cfg)

    # ── Connect hardware ──
    robot.connect()
    teleop.connect()

    # ── Shared task state ──
    task_state = TaskState(initial_task=args.initial_task)
    stop_event = threading.Event()

    # ── Start voice/keyboard listener in background ──
    if args.text_mode:
        listener = threading.Thread(
            target=keyboard_listener_thread,
            args=(task_state, stop_event),
            daemon=True,
        )
    else:
        listener = threading.Thread(
            target=voice_listener_thread,
            args=(task_state, whisper_model, args.listen_duration, 16000, args.silence_threshold, stop_event),
            daemon=True,
        )
    listener.start()

    # ── Continuous control loop ──
    print("\n" + "=" * 60)
    print("  CONTINUOUS VOICE-CONTROLLED SmolVLA INFERENCE")
    print(f"  Duration: {args.duration/60:.0f} minutes")
    print(f"  Initial task: \"{args.initial_task}\"")
    print(f"  Input mode: {'keyboard' if args.text_mode else 'voice (speak anytime)'}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    step = 0

    try:
        while (time.perf_counter() - start_t) < args.duration and not stop_event.is_set():
            loop_start = time.perf_counter()

            # Read current task (may change at any time from voice thread)
            current_task = task_state.task

            if not current_task:
                time.sleep(0.1)
                continue

            # Get observation from robot
            obs = robot.get_observation()

            # Build observation frame matching dataset feature keys
            observation_frame = {}
            for key, feat_info in ds_meta.features.items():
                if key.startswith("observation.images."):
                    cam_name = key.replace("observation.images.", "")
                    if cam_name in obs and isinstance(obs[cam_name], np.ndarray):
                        observation_frame[key] = obs[cam_name]
                elif key == "observation.state":
                    # Build state vector from individual motor readings
                    state_keys = sorted(
                        k for k in obs
                        if not isinstance(obs[k], np.ndarray) or obs[k].ndim != 3
                    )
                    if state_keys:
                        state_values = np.concatenate(
                            [np.atleast_1d(obs[k]).astype(np.float32) for k in state_keys]
                        )
                        observation_frame[key] = state_values

            # Predict action using predict_action (same path as lerobot-record)
            action = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_cfg.use_amp if hasattr(policy_cfg, "use_amp") else False,
                task=current_task,
                robot_type=robot.robot_type,
            )

            # Convert action tensor to robot command and send
            robot_action = make_robot_action(action, ds_meta.features)
            robot.send_action(robot_action)

            step += 1

            # Periodic status (every 5 seconds)
            if step % (args.fps * 5) == 0:
                elapsed = time.perf_counter() - start_t
                task_display = current_task[:50] + "..." if len(current_task) > 50 else current_task
                print(
                    f"  [{elapsed/60:.1f}m / {args.duration/60:.0f}m] "
                    f"Step {step} | Task changes: {task_state.change_count} | "
                    f"Current: \"{task_display}\""
                )

            # Maintain target FPS
            dt = time.perf_counter() - loop_start
            sleep_time = 1.0 / args.fps - dt
            if sleep_time > 0:
                precise_sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        stop_event.set()
        elapsed = time.perf_counter() - start_t
        print(f"\nRan for {elapsed/60:.1f} minutes, {step} steps, {task_state.change_count} task changes.")
        print("Disconnecting...")
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
