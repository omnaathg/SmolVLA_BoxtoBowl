#!/usr/bin/env python3
"""Quick test: record 5 seconds from mic → transcribe with Whisper."""

import numpy as np
import sounddevice as sd
import whisper

DURATION = 8  # seconds
SAMPLE_RATE = 16000

# ── Find the right microphone ──
print("Available input devices:")
mic_device = None
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']}")
        # Prefer Kreo Owl (webcam mic), fall back to any non-monitor device
        if "kreo" in d["name"].lower() or "owl" in d["name"].lower():
            mic_device = i
        elif mic_device is None and "benq" not in d["name"].lower():
            mic_device = i

if mic_device is None:
    mic_device = sd.default.device[0]

device_name = sd.query_devices(mic_device)["name"]
print(f"\nUsing: [{mic_device}] {device_name}")

print("\nLoading Whisper 'base' model...")
model = whisper.load_model("base")
print("Ready.\n")

for i in range(3):
    input(f"[Test {i+1}/3] Press Enter, then speak for {DURATION} seconds...")
    print(f"  🎤 Listening for {DURATION}s...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=mic_device,
    )
    sd.wait()

    audio_np = audio.flatten().astype(np.float32)
    rms = np.sqrt(np.mean(audio_np**2))
    print(f"  Audio RMS: {rms:.4f} {'(good)' if rms > 0.01 else '(very quiet — speak louder or move closer)'}")

    result = model.transcribe(audio_np, language="en", fp16=False)
    print(f"  📝 Transcription: \"{result['text'].strip()}\"\n")

print("Whisper test complete!")
