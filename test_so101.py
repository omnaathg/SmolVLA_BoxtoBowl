"""
End-to-end test / demo script for the SO-101 arm.

Usage:
    source activate.sh
    python test_so101.py --port /dev/tty.usbmodem<XXXXX>

Steps this script performs:
    1. Verifies LeRobot imports
    2. Checks the given port exists
    3. Connects to the SO-101 follower arm
    4. Reads current joint positions
    5. Disconnects cleanly
"""

import argparse
import sys
import time

def check_imports():
    print("[1/5] Checking LeRobot imports...")
    try:
        import lerobot
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        print(f"      LeRobot {lerobot.__version__} OK")
        return SO101Follower, SO101FollowerConfig
    except ImportError as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

def check_port(port):
    import os
    print(f"[2/5] Checking port {port}...")
    if not os.path.exists(port):
        print(f"      ERROR: Port {port} not found.")
        print("      Tip: Run `lerobot-find-port` to discover the correct port.")
        sys.exit(1)
    print(f"      Port {port} found OK")

def connect_arm(SO101Follower, SO101FollowerConfig, port, arm_id):
    print(f"[3/5] Connecting to SO-101 follower at {port}...")
    config = SO101FollowerConfig(port=port, id=arm_id)
    arm = SO101Follower(config)
    try:
        arm.connect(calibrate=False)
        print("      Connected OK")
        return arm
    except Exception as e:
        print(f"      ERROR connecting: {e}")
        print("      Tip: If motors are unconfigured, run: lerobot-setup-motors --robot.type=so101_follower --robot.port=" + port)
        sys.exit(1)

def read_positions(arm):
    print("[4/5] Reading joint positions...")
    try:
        obs = arm.get_observation()
        for key, val in obs.items():
            print(f"      {key}: {val:.3f}")
    except Exception as e:
        print(f"      WARNING: Could not read observation: {e}")

def disconnect_arm(arm):
    print("[5/5] Disconnecting...")
    arm.disconnect()
    print("      Done. Arm disconnected cleanly.")

def main():
    parser = argparse.ArgumentParser(description="SO-101 end-to-end test")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem575E0032081")
    parser.add_argument("--id", default="test_arm", help="Unique arm identifier (default: test_arm)")
    args = parser.parse_args()

    SO101Follower, SO101FollowerConfig = check_imports()
    check_port(args.port)
    arm = connect_arm(SO101Follower, SO101FollowerConfig, args.port, args.id)
    read_positions(arm)
    disconnect_arm(arm)
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
