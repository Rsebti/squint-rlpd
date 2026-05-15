#!/usr/bin/env python3
"""Scan cv2 camera indices 0..N and report which produce frames.

Use this whenever camera latency measurement fails — macOS AVFoundation
reassigns indices when devices are added/removed/woken, and the wrist USB
cam can land at any index. Also saves one still per working index to
debug_artifacts/camera_scan_<idx>.png so you can visually identify which
camera is which.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "debug_artifacts"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning cv2.VideoCapture indices 0..{n-1}:")
    print(f"{'idx':>4} {'opened':>6} {'res':>10} {'fps':>5} "
          f"{'frames/1s':>10} {'snapshot':>40}")
    print("-" * 80)
    for idx in range(n):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"{idx:>4} {'no':>6}")
            continue
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_ok = 0
        last_frame = None
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            ok, f = cap.read()
            if ok:
                n_ok += 1
                last_frame = f
        snap = "—"
        if last_frame is not None:
            p = ARTIFACT_DIR / f"camera_scan_{idx}.png"
            cv2.imwrite(str(p), last_frame)
            snap = p.name
        print(f"{idx:>4} {'yes':>6} {f'{w}x{h}':>10} {fps:>5.0f} "
              f"{n_ok:>10} {snap:>40}")
        cap.release()
        time.sleep(0.2)

    print()
    print("Pick the index of the wrist USB cam (compare snapshots in "
          "debug_artifacts/) and pass it as --camera_index N to probe_all.py.")


if __name__ == "__main__":
    sys.exit(main())
