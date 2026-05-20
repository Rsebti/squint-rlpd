"""Live wrist-camera view (Linux / V4L2).

Run alongside `examples/visualize_sim.py` to eyeball how the real wrist
camera compares against the sim's domain-randomized renders.

Usage:
    python examples/camera_live.py                  # default /dev/video0
    python examples/camera_live.py --camera 1       # use /dev/video1
    python examples/camera_live.py --raw            # show full 1920x1080 frame
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer_linux import Cv2Camera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0,
                   help="V4L2 device index (default: 0). See `v4l2-ctl --list-devices`.")
    p.add_argument("--width", type=int, default=1920, help="capture width")
    p.add_argument("--height", type=int, default=1080, help="capture height")
    p.add_argument("--fps", type=int, default=30, help="capture FPS")
    p.add_argument("--display_width", type=int, default=640,
                   help="display width (frame is area-downsampled). 640x360 = sim sensor size.")
    p.add_argument("--raw", action="store_true",
                   help="show the native capture resolution instead of downsampling")
    args = p.parse_args()

    cam = Cv2Camera(index=args.camera, width=args.width, height=args.height, fps=args.fps)
    print(f"Opened /dev/video{args.camera} at {args.width}x{args.height}@{args.fps} fps")
    print("Press q or Esc to quit.")

    display_size = None
    if not args.raw:
        ratio = args.height / args.width
        display_size = (args.display_width, int(round(args.display_width * ratio)))

    ema_alpha = 0.2
    ema_fps = None
    prev_t = None
    prev_count = cam.frame_count

    try:
        while True:
            t0 = time.perf_counter()
            rgb = cam.async_read()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if display_size is not None:
                bgr = cv2.resize(bgr, display_size, interpolation=cv2.INTER_AREA)

            cur_count = cam.frame_count
            if prev_t is not None:
                dt = t0 - prev_t
                if dt > 0:
                    inst_fps = (cur_count - prev_count) / dt
                    ema_fps = inst_fps if ema_fps is None else (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps
            prev_t, prev_count = t0, cur_count

            if ema_fps is not None:
                cv2.putText(bgr, f"{ema_fps:5.1f} fps", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("wrist camera (live)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
