#!/usr/bin/env python3
"""
ContextVLA card-memory inference for SO-101.

The robot has seen 3 face-up playing cards, which are then flipped face-down.
Speak the name of the card to pick up (e.g. "ace of hearts").
ContextVLA uses a 15-second visual memory window to identify the target.

Usage:
    python run_contextvla_card_inference.py
    python run_contextvla_card_inference.py --initial_task "ace of hearts"
    python run_contextvla_card_inference.py --text_mode
    python run_contextvla_card_inference.py --policy_path omnaathg/contextvla_card_memory

Dependencies (install once):
    python -m pip install openai-whisper sounddevice
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contextvla import ContextVLAConfig, ContextVLAPolicy  # noqa: F401 (registers config subclass)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Card parsing ──────────────────────────────────────────────────────────────

RANKS = {
    "ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "jack", "queen", "king",
    "a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k",
}
SUITS = {"hearts", "diamonds", "clubs", "spades"}


def parse_card_command(text: str) -> str | None:
    """Return a canonical task string if text names a valid card, else None."""
    lower = text.lower()
    words = lower.split()
    suit = next((s for s in SUITS if s in lower), None)
    rank = next((r for r in RANKS if r in words), None)
    if rank and suit:
        return f"Pick up the {rank} of {suit}"
    return None


# ─── Shared task state ─────────────────────────────────────────────────────────

class TaskState:
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
                print(f"\n  >> CARD TARGET ({self._change_count}): \"{new_task}\"")

    @property
    def change_count(self) -> int:
        with self._lock:
            return self._change_count


# ─── Voice listener ────────────────────────────────────────────────────────────

def find_microphone():
    import sounddevice as sd
    mic = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            if "kreo" in d["name"].lower() or "owl" in d["name"].lower():
                mic = i
            elif mic is None and "benq" not in d["name"].lower():
                mic = i
    if mic is None:
        mic = sd.default.device[0]
    print(f"  Microphone: [{mic}] {sd.query_devices(mic)['name']}")
    return mic


def voice_listener_thread(
    task_state: TaskState,
    whisper_model,
    listen_duration: float,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    stop_event: threading.Event = None,
):
    import sounddevice as sd

    mic = find_microphone()
    print("  Voice listener active. Speak the card name (e.g. 'ace of hearts').")

    while not stop_event.is_set():
        try:
            audio = sd.rec(
                int(listen_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=mic,
            )
            sd.wait()
            if stop_event.is_set():
                break

            audio = audio.flatten()
            if np.sqrt(np.mean(audio ** 2)) < silence_threshold:
                continue

            result = whisper_model.transcribe(audio, language="en", fp16=False)
            text = result["text"].strip()
            if not text or len(text) < 3:
                continue

            card_cmd = parse_card_command(text)
            if card_cmd:
                task_state.task = card_cmd
            else:
                print(f"  [Voice: \"{text}\" — not a valid card, ignoring]")

        except Exception as e:
            print(f"  [Voice error: {e}]")
            time.sleep(1)


def keyboard_listener_thread(task_state: TaskState, stop_event: threading.Event):
    import select

    print("  Keyboard mode. Type a card name (e.g. 'ace of hearts') and press Enter.")
    print("  Type 'quit' to stop.\n")

    while not stop_event.is_set():
        try:
            if select.select([sys.stdin], [], [], 1.0)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() in ("quit", "exit", "stop", "q"):
                    stop_event.set()
                    break
                if line:
                    cmd = parse_card_command(line)
                    task_state.task = cmd if cmd else f"Pick up the {line}"
        except Exception:
            break


# ─── Language tokenization ─────────────────────────────────────────────────────

def tokenize_task(
    task_str: str, processor, max_length: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    text = task_str if task_str.endswith("\n") else task_str + "\n"
    enc = processor.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return (
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device).bool(),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ContextVLA card-memory inference")
    parser.add_argument("--policy_path",      default="omnaathg/contextvla_card_memory")
    parser.add_argument("--dataset_repo_id",  default="omnaathg/so101_card_memory",
                        help="Dataset used during training (needed for normalization stats)")
    parser.add_argument("--follower_port",    default="COM8")
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--fps",              type=int,   default=30)
    parser.add_argument("--duration",         type=float, default=60.0,
                        help="Episode duration in seconds")
    parser.add_argument("--whisper_model",    default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--listen_duration",  type=float, default=6.0)
    parser.add_argument("--silence_threshold",type=float, default=0.01)
    parser.add_argument("--text_mode",        action="store_true",
                        help="Type instructions instead of speaking")
    parser.add_argument("--initial_task",     default="",
                        help="Card target set at start (e.g. 'ace of hearts')")
    parser.add_argument("--observe_s",        type=float, default=15.0,
                        help="Seconds to observe before sending actions (let buffer fill with face-up frames)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ── Policy ──
    logger.info(f"Loading ContextVLAPolicy from {args.policy_path}...")
    policy = ContextVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()
    policy.reset()
    logger.info(f"Policy ready. n_obs_steps={policy.config.n_obs_steps}, stride={policy.config.temporal_stride}")

    # ── Tokenizer ──
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)

    # ── Dataset metadata (for make_robot_action motor ordering) ──
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.utils import make_robot_action
    logger.info(f"Loading dataset metadata from {args.dataset_repo_id}...")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    # ── Whisper ──
    if not args.text_mode:
        import whisper
        logger.info(f"Loading Whisper '{args.whisper_model}'...")
        whisper_model = whisper.load_model(args.whisper_model)
        logger.info("Whisper ready.")
    else:
        whisper_model = None

    # ── Robot ──
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.utils.robot_utils import precise_sleep

    robot_cfg = SOFollowerRobotConfig(
        port=args.follower_port,
        id="ojas_follower_arm",
        cameras={
            "tripod_cam": OpenCVCameraConfig(index_or_path=2, width=640,  height=480, fps=30),
            "gripper_cam": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=30),
        },
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    logger.info("Robot connected.")

    # ── Task state + background listener ──
    initial = args.initial_task
    if initial and not initial.lower().startswith("pick"):
        initial = f"Pick up the {initial}"
    task_state = TaskState(initial_task=initial)
    stop_event  = threading.Event()

    if args.text_mode:
        listener = threading.Thread(
            target=keyboard_listener_thread,
            args=(task_state, stop_event),
            daemon=True,
        )
    else:
        listener = threading.Thread(
            target=voice_listener_thread,
            args=(task_state, whisper_model, args.listen_duration, 16000,
                  args.silence_threshold, stop_event),
            daemon=True,
        )
    listener.start()

    print("\n" + "=" * 60)
    print("  CONTEXTVLA CARD-MEMORY INFERENCE")
    print(f"  Policy:   {args.policy_path}")
    print(f"  Device:   {device}")
    print(f"  Duration: {args.duration}s  (+{args.observe_s}s observe phase)")
    print(f"  Mode:     {'keyboard' if args.text_mode else 'voice'}")
    if args.initial_task:
        print(f"  Card:     {args.initial_task}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # ── Ready prompt — wait until user is set up ──────────────────────────────
    print("  Arrange all 3 cards FACE-UP in front of the robot.")
    print("  Press Enter when ready to start the countdown...")
    input()

    # ── Observe phase: structured countdown matching training timing ──────────
    show_s  = 7.0                          # cards face-up  (match actual recorded 0–7s)
    hide_s  = args.observe_s - show_s      # cards covered  (match actual recorded 7–15s)

    def _beep():
        try:
            import winsound
            winsound.Beep(880, 120)
        except Exception:
            pass

    current_task = task_state.task or args.initial_task

    def _observe_batch(obs):
        if not current_task:
            return
        lang_ids, lang_masks = tokenize_task(
            current_task, processor, policy.config.tokenizer_max_length, device
        )
        state_vals = np.concatenate([
            np.atleast_1d(obs[k]).astype(np.float32)
            for k in sorted(obs)
            if not (isinstance(obs[k], np.ndarray) and obs[k].ndim == 3)
        ])
        state_dim = policy.config.input_features[OBS_STATE].shape[0]
        batch = {
            "observation.images.tripod_cam":  torch.from_numpy(obs["tripod_cam"]).permute(2,0,1).float().div_(255.0).unsqueeze(0).to(device),
            "observation.images.gripper_cam": torch.from_numpy(obs["gripper_cam"]).permute(2,0,1).float().div_(255.0).unsqueeze(0).to(device),
            OBS_STATE:                        torch.from_numpy(state_vals[:state_dim]).unsqueeze(0).to(device),
            OBS_LANGUAGE_TOKENS:              lang_ids,
            OBS_LANGUAGE_ATTENTION_MASK:      lang_masks,
        }
        with torch.no_grad():
            policy.select_action(batch)  # fill temporal buffer, discard action

    if args.observe_s > 0:
        # Phase 1: show cards (0–5s)
        print(f"\n  *** CARDS FACE-UP — keep them visible! ***")
        _beep()
        phase_end = time.perf_counter() + show_s
        last_announce = 99
        while time.perf_counter() < phase_end and not stop_event.is_set():
            t0 = time.perf_counter()
            remaining = phase_end - t0
            tick = int(remaining) + 1
            if tick != last_announce and tick <= 5:
                print(f"\r  Cards visible: {tick}s ...   ", end="", flush=True)
                last_announce = tick
            _observe_batch(robot.get_observation())
            dt = time.perf_counter() - t0
            if (s := 1.0 / args.fps - dt) > 0:
                time.sleep(s)

        # Phase 2: hide cards (5–10s)
        print(f"\n\n  *** COVER THE CARDS NOW ***")
        _beep(); _beep()
        phase_end = time.perf_counter() + hide_s
        last_announce = 99
        while time.perf_counter() < phase_end and not stop_event.is_set():
            t0 = time.perf_counter()
            remaining = phase_end - t0
            tick = int(remaining) + 1
            if tick != last_announce and tick <= 5:
                print(f"\r  Cards hidden:  {tick}s ...   ", end="", flush=True)
                last_announce = tick
            _observe_batch(robot.get_observation())
            dt = time.perf_counter() - t0
            if (s := 1.0 / args.fps - dt) > 0:
                time.sleep(s)

        print(f"\n\n  *** ROBOT PICKING NOW — stand back! ***\n")
        _beep(); _beep(); _beep()

    start_t = time.perf_counter()
    step = 0

    try:
        while (time.perf_counter() - start_t) < args.duration and not stop_event.is_set():
            loop_start = time.perf_counter()

            current_task = task_state.task
            if not current_task:
                print("  Waiting for card command (voice or keyboard)...")
                time.sleep(0.5)
                continue

            # ── Observation ──
            obs = robot.get_observation()

            def img_tensor(arr: np.ndarray) -> torch.Tensor:
                # (H, W, C) uint8 → (1, C, H, W) float32 [0, 1]
                return (
                    torch.from_numpy(arr).permute(2, 0, 1).float()
                    .div_(255.0)
                    .unsqueeze(0)
                    .to(device)
                )

            # State: concatenate all non-image scalar readings in sorted order
            state_vals = np.concatenate([
                np.atleast_1d(obs[k]).astype(np.float32)
                for k in sorted(obs)
                if not (isinstance(obs[k], np.ndarray) and obs[k].ndim == 3)
            ])
            state_dim = policy.config.input_features[OBS_STATE].shape[0]
            state_tensor = torch.from_numpy(state_vals[:state_dim]).unsqueeze(0).to(device)

            # Language tokens for current task
            lang_ids, lang_masks = tokenize_task(
                current_task, processor, policy.config.tokenizer_max_length, device
            )

            # Batch — images are single frames (B=1, C, H, W);
            # ContextVLAPolicy.select_action accumulates them into (B, T, C, H, W)
            batch = {
                "observation.images.tripod_cam":  img_tensor(obs["tripod_cam"]),
                "observation.images.gripper_cam": img_tensor(obs["gripper_cam"]),
                OBS_STATE:                        state_tensor,
                OBS_LANGUAGE_TOKENS:              lang_ids,
                OBS_LANGUAGE_ATTENTION_MASK:      lang_masks,
            }

            # ── Predict action ──
            with torch.no_grad():
                action = policy.select_action(batch)   # (1, action_dim)

            robot_action = make_robot_action(action, ds_meta.features)
            robot.send_action(robot_action)

            step += 1

            if step % (args.fps * 5) == 0:
                elapsed = time.perf_counter() - start_t
                print(
                    f"  [{elapsed:.0f}s/{args.duration:.0f}s] "
                    f"step={step} | task=\"{current_task}\""
                )

            dt = time.perf_counter() - loop_start
            if (sleep_t := 1.0 / args.fps - dt) > 0:
                precise_sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        stop_event.set()
        elapsed = time.perf_counter() - start_t
        print(f"\nRan {step} steps in {elapsed:.1f}s  ({task_state.change_count} card changes).")
        if robot.is_connected:
            robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
