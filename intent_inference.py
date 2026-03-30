#!/usr/bin/env python3
"""
Intent-classified SmolVLA inference with multi-episode support and video recording.

Loads the policy, robot, and classifier ONCE, then loops:
  1. Start recording (captures the voice instruction too)
  2. Listen for voice command (or text input)
  3. Classify → canonical task string (green / blue bowl)
  4. Run inference episode for --episode_time_s seconds
  5. Stop recording, prompt for next instruction

Press Ctrl+C during an episode to end it early and get the next prompt.
Press Ctrl+C at the instruction prompt to quit.

Usage:
    python intent_inference.py --text_mode        # keyboard mode
    python intent_inference.py                    # voice mode
    python intent_inference.py --no_record        # skip recording

Dependencies (install once):
    python -m pip install openai-whisper sounddevice sentence-transformers
"""

import argparse
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util


# ─── Task descriptions (must match training) ────────────────────────────────
TASK_TEMPLATE = "Pick up the box and place it in the {color} bowl"

CANONICAL_TASKS = {
    "green": TASK_TEMPLATE.format(color="green"),
    "blue": TASK_TEMPLATE.format(color="blue"),
}

# Reference phrases for semantic matching
COLOR_REFERENCES = {
    "green": [
        "green", "green bowl",
        "the color of grass", "the color of leaves",
        "lime", "emerald", "forest",
    ],
    "blue": [
        "blue", "blue bowl",
        "the color of the sky", "the color of water",
        "the color of the ocean", "azure", "cobalt", "navy",
    ],
}


class IntentClassifier:
    """Maps free-form instructions to canonical task strings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading intent classifier ({model_name})...")
        cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.colors = list(COLOR_REFERENCES.keys())

        self._ref_embeddings = {}
        for color, phrases in COLOR_REFERENCES.items():
            self._ref_embeddings[color] = self.model.encode(
                phrases, convert_to_tensor=True, normalize_embeddings=True
            )
        print(f"  Ready. Colors: {self.colors}")

    def classify(self, instruction: str) -> tuple[str, str, float]:
        """Returns: (color, canonical_task, confidence)"""
        instr_emb = self.model.encode(
            instruction, convert_to_tensor=True, normalize_embeddings=True
        )

        best_color = None
        best_score = -1.0

        for color in self.colors:
            sims = util.cos_sim(instr_emb, self._ref_embeddings[color])
            score = sims.max().item()
            if score > best_score:
                best_score = score
                best_color = color

        canonical = CANONICAL_TASKS[best_color]
        return best_color, canonical, best_score


# ─── Video + Audio Recorder ─────────────────────────────────────────────────
class Recorder:
    """Records video frames (pushed from control loop) + audio (background stream) into an MP4."""

    def __init__(self, output_dir: str = "recordings", audio_device: int = None, audio_rate: int = 48000):
        import sounddevice as sd
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.audio_rate = audio_rate
        self.audio_device = audio_device

        # Find Kreo mic if not specified
        if self.audio_device is None:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and ("kreo" in d["name"].lower() or "owl" in d["name"].lower()):
                    self.audio_device = i
                    break
        if self.audio_device is not None:
            print(f"  Recorder audio: [{self.audio_device}] {sd.query_devices(self.audio_device)['name']}")
        else:
            print("  Recorder: no Kreo mic found, recording video only")

        self._video_writer = None
        self._audio_chunks = []
        self._audio_stream = None
        self._recording = False
        self._video_path = None
        self._frame_count = 0
        self._start_time = None
        self._episode_files = []  # Collect final MP4 paths for concatenation

    def start(self, tag: str = ""):
        """Start a new recording (video only — audio starts later via start_audio)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = tag.replace(" ", "_")[:30] if tag else "episode"
        basename = f"{ts}_{safe_tag}"
        self._video_path = os.path.join(self.output_dir, f"{basename}.mp4")
        self._audio_path = os.path.join(self.output_dir, f"{basename}_audio.wav")
        self._final_path = os.path.join(self.output_dir, f"{basename}_final.mp4")
        self._video_writer = None  # Created on first frame (need size)
        self._audio_chunks = []
        self._frame_count = 0
        self._recording = True
        self._start_time = time.perf_counter()

        print(f"  Recording started: {self._video_path}")

    def start_audio(self):
        """Start the audio capture stream (call after Whisper is done with the mic)."""
        import sounddevice as sd
        if self.audio_device is not None and self._audio_stream is None:
            self._audio_stream = sd.InputStream(
                device=self.audio_device,
                samplerate=self.audio_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(self.audio_rate * 0.1),
            )
            self._audio_stream.start()

    def inject_audio(self, audio_float32: np.ndarray, sample_rate: int):
        """Inject externally recorded audio (e.g. from Whisper's sd.rec) into the recording.
        Resamples to match recorder's audio_rate if needed."""
        if not self._recording:
            return
        if sample_rate != self.audio_rate:
            # Simple resample by linear interpolation
            ratio = self.audio_rate / sample_rate
            n_out = int(len(audio_float32) * ratio)
            indices = np.linspace(0, len(audio_float32) - 1, n_out)
            audio_float32 = np.interp(indices, np.arange(len(audio_float32)), audio_float32)
        self._audio_chunks.append(audio_float32.reshape(-1, 1).astype(np.float32))

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._audio_chunks.append(indata.copy())

    def push_frame(self, frame: np.ndarray):
        """Push a video frame (BGR numpy array)."""
        if not self._recording:
            return
        if self._video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(self._video_path, fourcc, 25.0, (w, h))
        self._video_writer.write(frame)
        self._frame_count += 1

    def stop(self):
        """Stop recording and mux audio+video."""
        if not self._recording:
            return None
        self._recording = False
        elapsed = time.perf_counter() - self._start_time

        # Stop audio
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

        # Close video
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        # Save audio as WAV
        has_audio = len(self._audio_chunks) > 0 and self.audio_device is not None
        if has_audio:
            import wave
            audio_data = np.concatenate(self._audio_chunks)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(self._audio_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.audio_rate)
                wf.writeframes(audio_int16.tobytes())

        actual_fps = self._frame_count / elapsed if elapsed > 0 else 0
        print(f"  Recording stopped: {self._frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} fps)")
        print(f"  Video: {self._video_path}")
        if has_audio:
            print(f"  Audio: {self._audio_path}")
            self._try_mux()

        return self._video_path

    def _try_mux(self):
        """Combine video + audio into a single MP4 using PyAV."""
        try:
            import av
            import wave

            # Read audio
            with wave.open(self._audio_path, "r") as wf:
                audio_rate = wf.getframerate()
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

            # Open video input
            video_in = av.open(self._video_path)
            video_stream_in = video_in.streams.video[0]

            # Create output
            output = av.open(self._final_path, "w")
            video_stream_out = output.add_stream("h264", rate=video_stream_in.average_rate)
            video_stream_out.width = video_stream_in.width
            video_stream_out.height = video_stream_in.height
            video_stream_out.pix_fmt = "yuv420p"

            audio_stream_out = output.add_stream("aac", rate=audio_rate)
            audio_stream_out.layout = "mono"

            # Write video
            for frame in video_in.decode(video=0):
                for packet in video_stream_out.encode(frame):
                    output.mux(packet)
            for packet in video_stream_out.encode():
                output.mux(packet)

            # Write audio (s16 format for broad player compatibility)
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                audio_frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
                audio_frame.sample_rate = audio_rate
                for packet in audio_stream_out.encode(audio_frame):
                    output.mux(packet)
            for packet in audio_stream_out.encode():
                output.mux(packet)

            output.close()
            video_in.close()

            # Clean up intermediate files
            os.remove(self._video_path)
            os.remove(self._audio_path)
            self._episode_files.append(self._final_path)
            print(f"  Final: {self._final_path}")
        except Exception as e:
            print(f"  (mux failed: {e} — keeping separate video + audio files)")

    def combine_all(self):
        """Concatenate all episode videos into one final video."""
        if len(self._episode_files) < 1:
            return None
        if len(self._episode_files) == 1:
            combined = os.path.join(self.output_dir, f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            import shutil
            shutil.copy2(self._episode_files[0], combined)
            print(f"\n  Combined video (1 episode): {combined}")
            return combined

        try:
            import av

            combined = os.path.join(self.output_dir, f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

            # Decode all episodes to raw frames first
            all_video_frames = []
            all_audio_samples = []

            for ep_file in self._episode_files:
                print(f"    Reading: {os.path.basename(ep_file)}")
                inp = av.open(ep_file)
                for frame in inp.decode(video=0):
                    all_video_frames.append(frame.to_ndarray(format="bgr24"))
                inp.close()

                inp = av.open(ep_file)
                if inp.streams.audio:
                    for frame in inp.decode(audio=0):
                        arr = frame.to_ndarray().flatten()
                        if arr.dtype != np.float32:
                            arr = arr.astype(np.float32) / 32767.0
                        all_audio_samples.append(arr)
                inp.close()

            # Re-encode into one output
            out = av.open(combined, "w")
            v_out = out.add_stream("h264", rate=25)
            v_out.width = 640
            v_out.height = 480
            v_out.pix_fmt = "yuv420p"
            a_out = out.add_stream("aac", rate=self.audio_rate)
            a_out.layout = "mono"

            for bgr in all_video_frames:
                frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
                for p in v_out.encode(frame):
                    out.mux(p)
            for p in v_out.encode():
                out.mux(p)

            if all_audio_samples:
                audio_all = np.concatenate(all_audio_samples)
                audio_int16 = (np.clip(audio_all, -1.0, 1.0) * 32767).astype(np.int16)
                chunk_size = 1024
                for i in range(0, len(audio_int16), chunk_size):
                    chunk = audio_int16[i:i + chunk_size]
                    audio_frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
                    audio_frame.sample_rate = self.audio_rate
                    for p in a_out.encode(audio_frame):
                        out.mux(p)
                for p in a_out.encode():
                    out.mux(p)

            out.close()

            print(f"\n  Combined video ({len(self._episode_files)} episodes): {combined}")
            return combined
        except Exception as e:
            print(f"\n  (combine failed: {e})")
            return None


# ─── Microphone / Voice ─────────────────────────────────────────────────────
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


def listen_for_command(whisper_model, mic_device, listen_duration: float = 8.0, sample_rate: int = 16000):
    """Record audio and transcribe with Whisper. Returns (text, raw_audio, sample_rate)."""
    import sounddevice as sd

    hallucination_phrases = [
        "thank you", "thanks for watching", "subscribe",
        "you", "bye", "okay", "the end", "...",
    ]

    while True:
        print(f"\n  Listening for {listen_duration}s... Speak your instruction now.")
        audio = sd.rec(
            int(listen_duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=mic_device,
        )
        sd.wait()
        audio = audio.flatten().astype(np.float32)

        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            print("  (silence — try again)")
            continue

        result = whisper_model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()

        if not text or len(text) < 5:
            print("  (too short — try again)")
            continue
        if text.lower().strip(".!? ") in hallucination_phrases:
            print(f"  (filtered hallucination: \"{text}\" — try again)")
            continue

        print(f"  Heard: \"{text}\"")
        return text, audio, sample_rate


def get_instruction(args, whisper_model, mic_device):
    """Get instruction from voice or keyboard. Returns (text, raw_audio, sample_rate) or (None, None, None)."""
    if args.text_mode:
        try:
            line = input("\n  Enter instruction (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None, None, None
        if line.lower() in ("quit", "exit", "stop", "q"):
            return None, None, None
        return (line, None, None) if line else (None, None, None)
    else:
        return listen_for_command(whisper_model, mic_device, args.listen_duration)


def main():
    parser = argparse.ArgumentParser(description="Intent-classified SmolVLA inference (multi-episode)")
    parser.add_argument("--text_mode", action="store_true", help="Type instructions instead of speaking")
    parser.add_argument("--whisper_model", type=str, default="base", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--listen_duration", type=float, default=8.0)
    parser.add_argument("--classifier_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--policy_path", type=str, default="RajatDandekar/smolvla_box_to_bowl")
    parser.add_argument("--dataset_repo_id", type=str, default="RajatDandekar/so101_box_to_bowl")
    parser.add_argument("--follower_port", type=str, default="/dev/tty.wchusbserial5AE60830811")
    parser.add_argument("--leader_port", type=str, default="/dev/tty.wchusbserial5A7C1167331")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=float, default=60, help="Seconds per episode (default: 60)")
    parser.add_argument("--no_record", action="store_true", help="Disable video+audio recording")
    parser.add_argument("--record_dir", type=str, default="recordings", help="Directory to save recordings")
    args = parser.parse_args()

    # ── Imports from lerobot ──
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import OBS_STR
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

    rename_map = {
        "observation.images.webcam": "observation.images.camera1",
        "observation.images.arm_cam": "observation.images.camera2",
    }

    # ── Load everything ONCE ─────────────────────────────────────────────────
    # 1. Intent classifier
    classifier = IntentClassifier(model_name=args.classifier_model)

    # 2. Whisper (if voice mode)
    whisper_model = None
    mic_device = None
    if not args.text_mode:
        import whisper
        print(f"Loading Whisper '{args.whisper_model}'...")
        whisper_model = whisper.load_model(args.whisper_model)
        mic_device = find_microphone()
        print("Whisper ready.")

    # 3. Recorder
    recorder = None
    if not args.no_record:
        recorder = Recorder(output_dir=args.record_dir)

    # 4. Dataset metadata
    print(f"Loading dataset metadata from {args.dataset_repo_id}...")
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    # 5. Policy
    print(f"Loading policy from {args.policy_path}...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device

    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)

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

    # 6. Robot
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

    robot.connect()
    teleop.connect()

    # ── Ready ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SmolVLA MULTI-EPISODE INFERENCE")
    print(f"  Episode duration: {args.episode_time_s}s")
    print(f"  Input mode: {'keyboard' if args.text_mode else 'voice'}")
    print(f"  Recording: {'OFF' if args.no_record else 'ON → ' + args.record_dir + '/'}")
    print("  Ctrl+C during episode → end early, get next prompt")
    print("  Ctrl+C at prompt → quit")
    print("=" * 60)

    episode = 0

    try:
        while True:
            # ── Start recording BEFORE instruction (captures voice command) ──
            if recorder:
                recorder.start(tag=f"episode_{episode + 1}")

            # ── Capture video frames during voice listening ──
            # We run a background thread to keep pushing camera frames
            # while Whisper records audio, so the video shows the scene
            listening_done = threading.Event()

            def capture_frames_while_listening():
                while not listening_done.is_set():
                    try:
                        obs = robot.get_observation()
                        if recorder and "arm_cam" in obs and isinstance(obs["arm_cam"], np.ndarray):
                            frame_bgr = cv2.cvtColor(obs["arm_cam"], cv2.COLOR_RGB2BGR)
                            recorder.push_frame(frame_bgr)
                        time.sleep(1.0 / args.fps)
                    except Exception:
                        break

            if recorder:
                frame_thread = threading.Thread(target=capture_frames_while_listening, daemon=True)
                frame_thread.start()

            # ── Get instruction (Whisper uses the mic, recorder captures video only) ──
            raw_instruction, instruction_audio, instruction_sr = get_instruction(args, whisper_model, mic_device)
            listening_done.set()
            if recorder:
                # Inject the voice instruction audio into the recording
                if instruction_audio is not None:
                    recorder.inject_audio(instruction_audio, instruction_sr)
                # Now start audio stream for the episode
                recorder.start_audio()
            if raw_instruction is None:
                if recorder:
                    recorder.stop()
                break

            # ── Classify ──
            color, canonical_task, confidence = classifier.classify(raw_instruction)

            print(f"\n  {'─' * 50}")
            print(f"  Input:      \"{raw_instruction}\"")
            print(f"  Classified: {color} (confidence: {confidence:.3f})")
            print(f"  Task:       \"{canonical_task}\"")
            print(f"  {'─' * 50}")

            # ── Update recording tag now that we know the color ──
            if recorder:
                recorder._final_path = os.path.join(
                    recorder.output_dir,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{color}_final.mp4",
                )

            # ── Run episode ──
            episode += 1
            print(f"\n  ▶ Episode {episode}: \"{canonical_task}\"")
            print(f"    Running for {args.episode_time_s}s (Ctrl+C to end early)...\n")

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            start_t = time.perf_counter()
            step = 0

            try:
                while (time.perf_counter() - start_t) < args.episode_time_s:
                    loop_start = time.perf_counter()

                    obs = robot.get_observation()

                    # Record from Kreo camera (arm_cam = index 1)
                    if recorder and "arm_cam" in obs and isinstance(obs["arm_cam"], np.ndarray):
                        frame_bgr = cv2.cvtColor(obs["arm_cam"], cv2.COLOR_RGB2BGR)
                        recorder.push_frame(frame_bgr)

                    observation_frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)

                    action = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy_cfg.use_amp if hasattr(policy_cfg, "use_amp") else False,
                        task=canonical_task,
                        robot_type=robot.robot_type,
                    )

                    robot_action = make_robot_action(action, ds_meta.features)
                    robot.send_action(robot_action)

                    step += 1

                    if step % (args.fps * 5) == 0:
                        elapsed = time.perf_counter() - start_t
                        remaining = args.episode_time_s - elapsed
                        print(f"    Step {step} | {remaining:.0f}s remaining")

                    dt = time.perf_counter() - loop_start
                    sleep_time = 1.0 / args.fps - dt
                    if sleep_time > 0:
                        precise_sleep(sleep_time)

            except KeyboardInterrupt:
                pass  # End episode early, continue to next prompt

            if recorder:
                recorder.stop()

            elapsed = time.perf_counter() - start_t
            print(f"\n  ■ Episode {episode} done ({elapsed:.1f}s, {step} steps)")
            print(f"    Ready for next instruction.\n")

    except (KeyboardInterrupt, EOFError):
        pass

    # ── Cleanup ──
    if recorder:
        recorder.stop()  # In case recording was still active
        recorder.combine_all()
    print(f"\nCompleted {episode} episodes. Disconnecting...")
    if robot.is_connected:
        robot.disconnect()
    if teleop.is_connected:
        teleop.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
