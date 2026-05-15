#!/usr/bin/env python3
"""Measure USB-camera latency on the same hardware setup deploy.py / infer.py
use. Strategy: command a fast arm step, simultaneously stream frames from the
webcam at full rate AND poll the servo's Present_Position at ~200 Hz. Compare
the timestamps of:

    t_cmd          : when send_action() returned
    t_servo_motion : first servo readback where shoulder_lift moved >= 0.5 deg
    t_visual_motion: first camera frame whose pixel diff vs the pre-step
                     baseline exceeds a noise-driven threshold

Latencies:
    cmd -> servo  ~= 60 ms  (re-verifies the earlier step-response probe)
    cmd -> visual = full pipeline (servo + camera sensor + USB + cv2.read)
    visual - servo readback ~= camera-only sim-to-real lag

Critical implementation detail: a daemon thread MUST drain frames from the
camera continuously, otherwise frames pile up in the OS / USB driver queue
and the measured latency is dominated by buffering, not by capture.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.motors.motors_bus import MotorNormMode

from probe_step_response import ROBOT_PORT, build_robot, read_all_deg, send_targets_deg
from probe_all_joints import rest_servo_deg, smooth_move_to


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "debug_artifacts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera_index", type=int, default=0,
                   help="cv2.VideoCapture index (deploy.py default = 0)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps_req", type=int, default=30,
                   help="requested camera fps; logged effective rate may differ")
    p.add_argument("--step_rad", type=float, default=0.2,
                   help="arm step magnitude in rad (0.2 rad = ~11.5 deg, "
                        "large enough for an unambiguous visual diff)")
    p.add_argument("--joint", default="shoulder_lift",
                   help="joint to step (visual motion is dominated by upper arm)")
    p.add_argument("--baseline_s", type=float, default=2.0,
                   help="static-scene recording before the step")
    p.add_argument("--hold_s", type=float, default=2.0,
                   help="post-step recording")
    p.add_argument("--warmup_s", type=float, default=2.5,
                   help="post-camera-open settle (auto-exposure / WB convergence)")
    return p.parse_args()


def open_camera(idx, width, height, fps_req):
    # AVFOUNDATION on Mac, V4L2 on Linux. Both honour cv2.CAP_ANY.
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_req)
    # Try to keep the driver buffer shallow so cv2.read returns the freshest
    # frame and we don't measure stale-buffer latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraStreamer:
    """Continuously calls cap.read() in a thread and stores
    (timestamp, mean_abs_diff_vs_baseline) per frame so we can find the first
    frame that significantly differs from the static scene. Keeping the diff
    instead of every frame is essential — full 640x480 uint8 frames at 30 Hz
    would fill memory fast."""

    def __init__(self, cap):
        self.cap = cap
        self.records = []            # list[(t_perf, mean_abs_diff)]
        self._baseline = None        # float32 (H, W, 3)
        self._frame_lock = threading.Lock()
        self._collect_for_baseline = False
        self._baseline_frames = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def set_baseline_from_now(self, n_frames=8, timeout_s=3.0):
        """Block until the next n_frames are collected, then set their mean
        as the reference frame for all subsequent diff calculations."""
        with self._frame_lock:
            self._collect_for_baseline = True
            self._baseline_frames = []
        t_deadline = time.perf_counter() + timeout_s
        while True:
            with self._frame_lock:
                if len(self._baseline_frames) >= n_frames:
                    self._baseline = np.mean(
                        np.stack(self._baseline_frames), axis=0).astype(np.float32)
                    self._collect_for_baseline = False
                    return
            if time.perf_counter() > t_deadline:
                raise RuntimeError(
                    "Camera produced no frames during baseline window. "
                    "Check that the camera is opened and streaming.")
            time.sleep(0.01)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=1.0)

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok or frame is None:
                continue
            with self._frame_lock:
                if self._collect_for_baseline:
                    self._baseline_frames.append(frame.astype(np.float32))
                    continue
                if self._baseline is None:
                    continue
                # Mean absolute pixel diff vs reference frame.
                diff = float(
                    np.abs(frame.astype(np.float32) - self._baseline).mean())
            self.records.append((t, diff))


def servo_logger(robot, joint, stop_event, log, rate_hz=200.0):
    period = 1.0 / rate_hz
    while not stop_event.is_set():
        try:
            deg = robot.bus.sync_read("Present_Position")
            log.append((time.perf_counter(), deg[joint]))
        except Exception:
            pass
        time.sleep(period)


def first_above(records, threshold, t_after):
    """Return the first timestamp in `records` with value > threshold that
    occurs after `t_after`. None if no such record."""
    for t, v in records:
        if t >= t_after and v > threshold:
            return t, v
    return None, None


def main():
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    print(f"Opening camera index {args.camera_index} "
          f"({args.width}x{args.height} @ {args.fps_req} fps)...")
    cap = open_camera(args.camera_index, args.width, args.height, args.fps_req)

    print(f"Connecting to robot on {ROBOT_PORT}...")
    robot = build_robot(ROBOT_PORT)
    robot.connect()

    streamer = CameraStreamer(cap)
    streamer.start()

    servo_log = []
    servo_stop = threading.Event()

    try:
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        rest_pose = rest_servo_deg()
        print("Moving to deployment start pose...")
        smooth_move_to(robot, rest_pose)
        time.sleep(0.6)

        print(f"Camera warm-up ({args.warmup_s}s) so AE/WB converge...")
        time.sleep(args.warmup_s)

        print("Collecting reference frame (static scene)...")
        streamer.set_baseline_from_now(n_frames=10)

        print(f"Baseline observation: {args.baseline_s}s of static frames + servo")
        servo_thread = threading.Thread(
            target=servo_logger,
            args=(robot, args.joint, servo_stop, servo_log),
            daemon=True)
        servo_thread.start()
        time.sleep(args.baseline_s)

        initial = read_all_deg(robot)
        step_deg = math.degrees(args.step_rad)
        target_pose = dict(initial)
        target_pose[args.joint] = initial[args.joint] + step_deg
        print(f"Step: {args.joint} {initial[args.joint]:+.2f} -> "
              f"{target_pose[args.joint]:+.2f} deg")
        send_targets_deg(robot, target_pose)
        t_cmd = time.perf_counter()

        time.sleep(args.hold_s)

    finally:
        servo_stop.set()
        streamer.stop()
        try:
            send_targets_deg(robot, rest_servo_deg())
            time.sleep(0.4)
        except Exception:
            pass
        robot.disconnect()
        cap.release()
        print("Disconnected.")

    # Effective frame rate
    if len(streamer.records) < 2:
        print("Not enough frames captured — camera connection issue?")
        return 1
    cam_ts = np.array([t for t, _ in streamer.records])
    cam_diffs = np.array([d for _, d in streamer.records])
    eff_fps = 1.0 / np.mean(np.diff(cam_ts))
    print(f"Camera effective rate: {eff_fps:.1f} fps "
          f"({len(streamer.records)} frames over "
          f"{cam_ts[-1] - cam_ts[0]:.1f} s)")

    if not servo_log:
        print("No servo samples — bus read failed?")
        return 1
    s_ts  = np.array([t for t, _ in servo_log])
    s_pos = np.array([p for _, p in servo_log])

    # Noise floors before the step
    pre_cam_mask  = cam_ts < t_cmd
    pre_serv_mask = s_ts   < t_cmd
    if pre_cam_mask.sum() < 5 or pre_serv_mask.sum() < 5:
        print("Not enough pre-step samples for noise floor estimation.")
        return 1
    cam_pre = cam_diffs[pre_cam_mask]
    cam_noise_std = float(cam_pre.std())
    cam_thr = max(cam_pre.mean() + 5 * cam_noise_std, 1.0)

    s_pre = s_pos[pre_serv_mask]
    s_baseline = float(s_pre.mean())
    s_thr = 0.5  # deg

    t_servo_motion, s_at = first_above(
        list(zip(s_ts.tolist(), np.abs(s_pos - s_baseline).tolist())),
        s_thr, t_cmd)
    t_visual_motion, c_at = first_above(
        list(zip(cam_ts.tolist(), cam_diffs.tolist())),
        cam_thr, t_cmd)

    # CSV dump
    csv_path = ARTIFACT_DIR / f"camera_latency_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s", "source", "value"])
        t0 = min(cam_ts[0], s_ts[0])
        for t, d in zip(cam_ts.tolist(), cam_diffs.tolist()):
            w.writerow([f"{t - t0:.6f}", "cam_diff", f"{d:.4f}"])
        for t, p in zip(s_ts.tolist(), s_pos.tolist()):
            w.writerow([f"{t - t0:.6f}", "servo_deg", f"{p:.4f}"])
        w.writerow([f"{t_cmd - t0:.6f}", "cmd", "step"])
    print(f"Wrote {csv_path}")

    print()
    print("=" * 72)
    print(f"Camera noise floor:  mean_diff={cam_pre.mean():.2f}, "
          f"std={cam_noise_std:.2f}  (threshold={cam_thr:.2f})")
    print(f"Servo noise floor:   pre-step deg std="
          f"{float(s_pre.std()):.3f}  (threshold=0.5 deg)")
    print()
    if t_servo_motion is None:
        print("Servo never moved past threshold; aborting latency report.")
        return 1
    if t_visual_motion is None:
        print("Camera never saw motion past threshold; aborting latency report.")
        return 1

    cmd_to_servo  = (t_servo_motion  - t_cmd) * 1000.0
    cmd_to_visual = (t_visual_motion - t_cmd) * 1000.0
    cam_only      = (t_visual_motion - t_servo_motion) * 1000.0
    frame_dt_ms   = 1000.0 / eff_fps

    print(f"cmd -> servo readback motion   = {cmd_to_servo:7.1f} ms"
          f"  (re-verifies servo step probe)")
    print(f"cmd -> first visual motion     = {cmd_to_visual:7.1f} ms"
          f"  (full deploy pipeline)")
    print(f"camera lag vs servo readback   = {cam_only:7.1f} ms"
          f"  ±{frame_dt_ms:.1f} ms frame quantisation")
    print()
    print("Interpretation:")
    print(f"  the policy sees an image of the world ~{cam_only:.0f} ms older")
    print(f"  than the servo readback for the same frame. Combined with the")
    print(f"  {cmd_to_servo:.0f} ms cmd->servo delay, the visual feedback loop is")
    print(f"  {cmd_to_visual:.0f} ms behind the policy's most recent command.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
