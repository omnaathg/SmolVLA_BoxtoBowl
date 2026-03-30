#!/usr/bin/env python3
"""Test intervention detection on the leader arm.

No VLA, no cameras — just connects to the leader arm and prints
whether it thinks you're intervening or not.

Usage:
    python scripts/test_intervention.py --leader_port /dev/tty.wchusbserial5A7C1167331
"""

import argparse
import signal
import time

SHUTDOWN = False

def signal_handler(sig, frame):
    global SHUTDOWN
    SHUTDOWN = True

signal.signal(signal.SIGINT, signal_handler)


class InterventionDetector:
    MONITORED_JOINTS = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(self, threshold_degrees=2.0, confirmation_frames=2, release_frames=150):
        self.threshold = threshold_degrees
        self.confirmation_frames = confirmation_frames
        self.release_frames = release_frames
        self._prev_positions = None
        self._active_count = 0
        self._inactive_count = 0
        self._is_intervening = False

    def update(self, leader_positions):
        if self._prev_positions is None:
            self._prev_positions = dict(leader_positions)
            return False, 0.0, "", 0

        motion_detected = False
        max_delta = 0.0
        max_joint = ""
        for joint in self.MONITORED_JOINTS:
            key = f"{joint}.pos"
            if key in leader_positions and key in self._prev_positions:
                delta = abs(leader_positions[key] - self._prev_positions[key])
                if delta > max_delta:
                    max_delta = delta
                    max_joint = joint
                if delta > self.threshold:
                    motion_detected = True

        self._prev_positions = dict(leader_positions)

        if motion_detected:
            self._active_count += 1
            self._inactive_count = 0
            if self._active_count >= self.confirmation_frames:
                self._is_intervening = True
        else:
            self._inactive_count += 1
            self._active_count = 0
            if self._inactive_count >= self.release_frames:
                self._is_intervening = False

        return self._is_intervening, max_delta, max_joint, self._inactive_count

    @property
    def inactive_count(self):
        return self._inactive_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader_port", type=str, default="/dev/tty.wchusbserial5A7C1167331")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--release_frames", type=int, default=150)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    print(f"Connecting to leader arm at {args.leader_port}...")
    leader = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id="my_leader"))
    leader.connect()
    print("Connected!\n")

    detector = InterventionDetector(
        threshold_degrees=args.threshold,
        release_frames=args.release_frames,
    )

    print(f"Threshold: {args.threshold} degrees")
    print(f"Release after: {args.release_frames} frames ({args.release_frames / args.fps:.1f}s)")
    print(f"Move the leader arm to test intervention detection.")
    print(f"Press Ctrl+C to quit.\n")

    target_dt = 1.0 / args.fps
    prev_state = None

    while not SHUTDOWN:
        loop_start = time.time()

        action = leader.get_action()
        is_intervening, max_delta, max_joint, idle_frames = detector.update(action)

        state = "HUMAN" if is_intervening else "AGENT"
        if state != prev_state:
            print(f"\n>>> SWITCHED TO: {state}")
            prev_state = state

        idle_sec = idle_frames / args.fps
        bar = "█" * min(int(max_delta), 50)
        print(
            f"  {state:5s} | max_delta: {max_delta:6.2f}° ({max_joint:15s}) | "
            f"idle: {idle_sec:5.1f}s / {args.release_frames / args.fps:.1f}s | {bar}",
            end="\r",
        )

        elapsed = time.time() - loop_start
        sleep_time = max(0, target_dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n\nDisconnecting...")
    leader.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
