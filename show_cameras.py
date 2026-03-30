"""Live dual camera feed viewer. Press Q to quit."""
import cv2
import numpy as np

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Showing live feeds. Press Q to quit.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0:
        frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame0, "Camera 0 unavailable", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if not ret1:
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame1, "Camera 1 unavailable", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame0, "Webcam (Kreo Owl)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame1, "Arm Camera (USB2.0_CAM1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    combined = np.hstack([frame0, frame1])
    cv2.imshow("SO-101 Live Feeds — Press Q to quit", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
