"""Save one frame from a camera index so you can preview it in VSCode.

    python peek_camera.py                # default index 1
    python peek_camera.py --index 0      # MacBook built-in
    python peek_camera.py --index 2      # iPhone Continuity
"""
import argparse
import time
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--out", type=str, default="camera_snapshot.jpg")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame = None
    for _ in range(90):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.033)
    cap.release()

    if frame is None:
        raise SystemExit(f"Camera index {args.index} opened but never delivered a frame.")

    cv2.imwrite(args.out, frame)
    h, w = frame.shape[:2]
    print(f"Saved {args.out} ({w}x{h}) from camera index {args.index}")


if __name__ == "__main__":
    main()
