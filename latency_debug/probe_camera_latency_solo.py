#!/usr/bin/env python3
"""Camera-latency probes that do NOT need the robot.

Two modes:

  timing  : open the camera, stream frames at max rate for N seconds, report
            effective fps and inter-frame jitter. Cannot give absolute latency
            but reveals USB buffering and useful cadence stats.

  flash   : flash the laptop screen between black and white at known times,
            point the camera at the screen, find each flash in the captured
            video, report (screen_change_time -> first_white_frame_time).
            Gives an absolute upper bound on capture latency (display refresh
            ~8-16 ms is bundled in but the residual is mostly camera).

Usage:
    python probe_camera_latency_solo.py timing --duration 10
    python probe_camera_latency_solo.py flash  --n_flashes 8

For mode `flash` the camera must point at the laptop screen during the run.
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "debug_artifacts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["timing", "flash"])
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps_req", type=int, default=30)
    p.add_argument("--duration", type=float, default=6.0,
                   help="timing mode: seconds to stream")
    p.add_argument("--n_flashes", type=int, default=6,
                   help="flash mode: number of black<->white transitions")
    p.add_argument("--warmup_s", type=float, default=2.5)
    return p.parse_args()


def open_camera(idx, w, h, fps_req):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps_req)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ─── mode: timing ────────────────────────────────────────────────────────────
def mode_timing(args, cap):
    print(f"Warm-up {args.warmup_s}s (drain initial buffer + AE/WB)...")
    t_end_warm = time.perf_counter() + args.warmup_s
    while time.perf_counter() < t_end_warm:
        cap.read()

    print(f"Streaming for {args.duration:.1f}s...")
    ts = []
    t_end = time.perf_counter() + args.duration
    while time.perf_counter() < t_end:
        ok, _ = cap.read()
        if ok:
            ts.append(time.perf_counter())
    ts = np.asarray(ts)
    if len(ts) < 5:
        print("Too few frames received.")
        return 1
    dt = np.diff(ts) * 1000.0   # ms

    print()
    print("=" * 66)
    print(f"frames captured     : {len(ts)}")
    print(f"effective fps       : {1000.0/dt.mean():.2f}")
    print(f"frame period stats  : "
          f"mean={dt.mean():.2f} ms  median={np.median(dt):.2f} ms  "
          f"p95={np.percentile(dt, 95):.2f} ms  "
          f"max={dt.max():.2f} ms")
    print()
    print("Interpretation: cv2.read() returns when a frame is ready. The")
    print("median dt approximates the camera period (~1/fps).")
    print()
    print("Latency floor inference:")
    print(f"  - capture period ~{np.median(dt):.1f} ms means the freshest")
    print(f"    available frame is on average {np.median(dt)/2:.1f} ms old.")
    print(f"  - on top of that, USB driver + cv2 decode add another ~5-30 ms")
    print(f"    on Mac AVFoundation backends (rule of thumb).")
    print(f"  - so the deploy pipeline sees images ~{np.median(dt)/2+15:.0f}-"
          f"{np.median(dt)/2+30:.0f} ms in the past, before any model inference.")
    print(f"  - run `flash` mode for an empirical absolute measurement.")
    print("=" * 66)
    return 0


# ─── mode: flash ─────────────────────────────────────────────────────────────
def mode_flash(args, cap):
    """Show a fullscreen black<->white toggle on the laptop display. Capture
    frames continuously in a thread; record each frame's mean brightness with
    a perf_counter timestamp. For each scheduled flash time t_flash, find the
    first post-t_flash frame whose mean brightness crosses the midpoint —
    delta = first_bright_frame_t - t_flash."""

    print("Flash mode: point the USB camera at the LAPTOP SCREEN now.")
    print("A fullscreen black/white pane will toggle a few times.")
    print(f"Warm-up {args.warmup_s}s...")
    for _ in range(int(args.warmup_s * args.fps_req)):
        cap.read()
    time.sleep(0.3)

    # Streaming thread — record (t, mean_brightness)
    records = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ok, f = cap.read()
            t = time.perf_counter()
            if not ok or f is None:
                continue
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            records.append((t, float(gray.mean())))

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    # Get baseline brightness
    time.sleep(0.5)
    baseline = float(np.median([b for _, b in records[-15:]]))

    # Build the flash window. Use a separate cv2.imshow with WND_PROP_FULLSCREEN
    # so we can be confident the display change is near-instant from the OS.
    cv2.namedWindow("flash", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    black = np.zeros((args.height, args.width), dtype=np.uint8)
    white = np.full((args.height, args.width), 255, dtype=np.uint8)

    cv2.imshow("flash", black)
    cv2.waitKey(500)

    flash_times = []
    cycle_ms = 700
    for k in range(args.n_flashes):
        # show black then immediately schedule a white flip
        cv2.imshow("flash", black)
        cv2.waitKey(1)
        time.sleep(cycle_ms / 1000.0)
        t_on = time.perf_counter()
        cv2.imshow("flash", white)
        cv2.waitKey(1)
        flash_times.append(t_on)
        time.sleep(cycle_ms / 1000.0)

    cv2.imshow("flash", black)
    cv2.waitKey(300)
    cv2.destroyWindow("flash")
    cv2.waitKey(1)

    stop.set()
    th.join(timeout=1.0)

    if not records:
        print("No frames captured.")
        return 1

    ts = np.array([r[0] for r in records])
    br = np.array([r[1] for r in records])

    # Threshold: midpoint between baseline (dark) and observed peaks. Use the
    # 95th percentile of post-flash brightness as the "bright" anchor.
    post_first = ts >= flash_times[0]
    bright_anchor = float(np.percentile(br[post_first], 95))
    thr = (baseline + bright_anchor) / 2.0
    print(f"baseline brightness = {baseline:.2f}, "
          f"bright anchor = {bright_anchor:.2f}, "
          f"threshold = {thr:.2f}")

    latencies = []
    for t_on in flash_times:
        mask = (ts >= t_on) & (br > thr)
        if not mask.any():
            latencies.append(None)
            continue
        idx = np.argmax(mask)
        latencies.append((ts[idx] - t_on) * 1000.0)

    csv_path = ARTIFACT_DIR / f"camera_flash_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s", "brightness"])
        t0 = records[0][0]
        for t, b in records:
            w.writerow([f"{t-t0:.6f}", f"{b:.3f}"])
    print(f"Wrote {csv_path}")

    print()
    print("=" * 66)
    valid = [l for l in latencies if l is not None]
    if not valid:
        print("No flashes detected — check that the camera is pointed at the screen.")
        return 1
    arr = np.array(valid)
    print(f"flashes detected     : {len(valid)} / {len(flash_times)}")
    print(f"latency per flash    : "
          + ", ".join(f"{l:.1f}" for l in valid) + " ms")
    print(f"summary (ms)         : mean={arr.mean():.1f}  "
          f"median={np.median(arr):.1f}  std={arr.std():.1f}")
    print()
    fps = 1.0 / np.median(np.diff(ts))
    print(f"camera frame period  : {1000/fps:.1f} ms (= {fps:.1f} fps)")
    print(f"quantisation noise   : ±{1000/(2*fps):.1f} ms (half-frame)")
    print()
    print("Includes display-refresh latency (~8-16 ms on 60-120 Hz Mac LCDs).")
    print("Subtract that for the camera-only contribution; either way this is")
    print("the operationally relevant value for the policy's visual lag.")
    print("=" * 66)
    return 0


def main():
    args = parse_args()
    print(f"Opening camera index {args.camera_index} "
          f"({args.width}x{args.height} @ {args.fps_req} fps)...")
    cap = open_camera(args.camera_index, args.width, args.height, args.fps_req)
    try:
        if args.mode == "timing":
            return mode_timing(args, cap)
        else:
            return mode_flash(args, cap)
    finally:
        cap.release()


if __name__ == "__main__":
    sys.exit(main())
