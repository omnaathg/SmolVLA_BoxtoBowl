"""Snap one frame from each camera index and save as camera_X.jpg. No GUI needed."""
import cv2
import numpy as np
from pathlib import Path

INDICES = list(range(7))
out_dir = Path("camera_snapshots")
out_dir.mkdir(exist_ok=True)

print("Probing camera indices...")
for i in INDICES:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        continue

    # Read a few frames to let auto-exposure settle
    for _ in range(5):
        ret, frame = cap.read()

    if ret and frame is not None:
        path = out_dir / f"camera_{i}.jpg"
        cv2.putText(frame, f"Index {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(path), frame)
        h, w = frame.shape[:2]
        print(f"  index {i}: {w}x{h}  -> saved {path}")
    else:
        print(f"  index {i}: opened but no frame")

    cap.release()

print(f"\nDone. Open the '{out_dir}' folder and match each image to your physical camera.")
