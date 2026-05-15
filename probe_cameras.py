"""Save one frame from each camera index so you can identify the wrist cam.

Run from the squint env:
    python probe_cameras.py

Then look at cam_0.jpg, cam_1.jpg, ... and pick the one showing the wrist view.
macOS will prompt for camera permission the first time — accept it.
"""
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"index {i}: not opened")
        cap.release()
        continue
    ok, frame = cap.read()
    if ok:
        path = f"cam_{i}.jpg"
        cv2.imwrite(path, frame)
        print(f"index {i}: {frame.shape} -> {path}")
    else:
        print(f"index {i}: opened but no frame")
    cap.release()
