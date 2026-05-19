#!/usr/bin/env python3
"""Intrinsic calibration for the SO101 deploy camera.

Captures chessboard images interactively, then runs cv2.calibrateCamera
and prints fx, fy, cx, cy, horizontal/vertical/diagonal FOV, distortion,
and the mean reprojection error. Saves results to camera_intrinsics.json.

Controls (preview window must be focused):
    SPACE   capture frame (only when the full board is detected)
    c       run calibration with the captured frames
    q       quit without calibrating

Notes:
    - Move the *camera*, not the board. ~15+ views from varied angles and
      distances. Include strong tilts (board not face-on) — flat-on shots
      alone under-constrain focal length.
    - Keep the board static; a screen checkerboard works if there's no glare.
    - Verify the board geometry: --cols and --rows count INNER corners,
      which is (squares_per_side - 1). Detection fails silently if wrong.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=28,
                   help="inner corners on the long side (= squares - 1). Default 28 (= 29-square side).")
    p.add_argument("--rows", type=int, default=20,
                   help="inner corners on the short side (= squares - 1). Default 20 (= 21-square side).")
    p.add_argument("--square-mm", type=float, default=150.0 / 23,
                   help="square side length in mm. Default = 150 mm / 23 squares ≈ 6.522 mm.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1920,
                   help="capture width in px. Must match deploy resolution. Default 1920 (matches deploy_utils/robot_config.py).")
    p.add_argument("--height", type=int, default=1080,
                   help="capture height in px. Default 1080 (matches deploy_utils/robot_config.py).")
    p.add_argument("--min-frames", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("camera_intrinsics.json"))
    p.add_argument("--images-dir", type=Path, default=None,
                   help="if set, skip live capture and run calibration on all images in this dir (jpg/png).")
    p.add_argument("--save-frames-dir", type=Path, default=None,
                   help="if set, also save each captured live frame here as PNG.")
    args = p.parse_args()

    pattern = (args.cols, args.rows)
    sq = float(args.square_mm)
    print(f"Pattern: {pattern[0]}x{pattern[1]} inner corners | square = {sq:.4f} mm")

    obj_template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj_template[:, :2] = np.indices(pattern).T.reshape(-1, 2) * sq

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    if args.images_dir is not None:
        # ── Saved-frames mode ──────────────────────────────────────────
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        paths = sorted(p for ext in exts for p in args.images_dir.glob(ext))
        if not paths:
            sys.exit(f"no images found in {args.images_dir}")
        print(f"loading {len(paths)} images from {args.images_dir}")
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  skip (unreadable): {path.name}"); continue
            if img_size is None:
                img_size = (frame.shape[1], frame.shape[0])
            elif (frame.shape[1], frame.shape[0]) != img_size:
                print(f"  skip (resolution mismatch {frame.shape[1]}x{frame.shape[0]} "
                      f"vs {img_size[0]}x{img_size[1]}): {path.name}")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
            if not found:
                print(f"  skip (no board): {path.name}"); continue
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix)
            img_points.append(refined)
            obj_points.append(obj_template.copy())
            print(f"  used {path.name}")
        if not img_points:
            sys.exit("no usable frames (no board detected in any)")
    else:
        # ── Live-capture mode ──────────────────────────────────────────
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.device, backend)
        if not cap.isOpened():
            sys.exit(f"failed to open camera at index {args.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        for _ in range(5):
            cap.read()
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (args.width, args.height):
            print(f"WARNING: requested {args.width}x{args.height}, camera gave {actual_w}x{actual_h}")

        if args.save_frames_dir is not None:
            args.save_frames_dir.mkdir(parents=True, exist_ok=True)

        win = "calibration  (SPACE=grab, c=calibrate, q=quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_grab = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); break
            if img_size is None:
                img_size = (frame.shape[1], frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern, corners, True)
            status = "BOARD" if found else "no board"
            col = (0, 255, 0) if found else (0, 0, 255)
            msg = f"captures {len(img_points)} (min {args.min_frames}) | {status}"
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                sys.exit("quit without calibrating")
            if key == ord(" ") and found:
                now = time.time()
                if now - last_grab < 0.3:
                    continue
                last_grab = now
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix)
                img_points.append(refined)
                obj_points.append(obj_template.copy())
                if args.save_frames_dir is not None:
                    fname = args.save_frames_dir / f"frame_{len(img_points):03d}.png"
                    cv2.imwrite(str(fname), frame)
                print(f"captured frame #{len(img_points)}")
            if key == ord("c"):
                if len(img_points) < args.min_frames:
                    print(f"need >= {args.min_frames} captures, have {len(img_points)}")
                    continue
                break

        cap.release()
        cv2.destroyAllWindows()
        if not img_points:
            sys.exit("no captures, exiting")
    assert img_size is not None

    print(f"\nRunning calibration on {len(img_points)} frames...")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    W, H = img_size
    fov_h = float(np.degrees(2 * np.arctan(W / (2 * fx))))
    fov_v = float(np.degrees(2 * np.arctan(H / (2 * fy))))
    fov_d = float(np.degrees(2 * np.arctan(np.hypot(W, H) / (2 * np.hypot(fx, fy)))))

    print(f"\n  reprojection error : {rms:.4f} px   (good < 0.5; great < 0.3)")
    print(f"  resolution         : {W} x {H}")
    print(f"  fx, fy             : {fx:.2f}, {fy:.2f}")
    print(f"  cx, cy             : {cx:.2f}, {cy:.2f}   (image centre = {W/2:.1f}, {H/2:.1f})")
    print(f"  FOV horizontal     : {fov_h:.2f} deg")
    print(f"  FOV vertical       : {fov_v:.2f} deg")
    print(f"  FOV diagonal       : {fov_d:.2f} deg")
    print(f"  distortion         : {dist.ravel().tolist()}")

    out = {
        "image_size": [W, H],
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "fov_horizontal_deg": fov_h,
        "fov_vertical_deg": fov_v,
        "fov_diagonal_deg": fov_d,
        "reprojection_error_px": float(rms),
        "pattern_inner_corners": list(pattern),
        "square_mm": sq,
        "n_frames": len(img_points),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
