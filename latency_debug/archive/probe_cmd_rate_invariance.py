#!/usr/bin/env python3
"""Empirical check: is the measured 60 ms actuator latency influenced by
how often the host re-issues the same Goal_Position?

Probes shoulder_lift twice from the deployment start pose:
  (A) one-shot:  send the step command exactly once
  (B) streamed:  send the step command repeatedly at 30 Hz during the hold

If L and tau differ by more than measurement noise, the delay is host-side
(bus / scheduler) — otherwise it's intrinsic to the servo and unchanged
when we bump the sim control rate.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Make root-level probe_*.py importable when this lives in debug_scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.motors.motors_bus import MotorNormMode

from probe_step_response import (
    ROBOT_PORT, build_robot, read_all_deg, send_targets_deg,
    record_at_rate, fit_fopdt,
)
from probe_all_joints import (
    rest_servo_deg, smooth_move_to, choose_step_sign,
)


def probe_with_streamed_cmd(robot, joint, *, step_rad, cmd_hz,
                            baseline_s=1.0, hold_s=2.5,
                            sample_rate_hz=100.0):
    """Like probe_one_joint but, during the step segment, re-sends the
    step Goal_Position at cmd_hz. cmd_hz <= 0 means one-shot.
    """
    current_all = read_all_deg(robot)
    cur = current_all[joint]
    step_deg = math.degrees(abs(step_rad)) * choose_step_sign(
        joint, cur, math.degrees(abs(step_rad)))
    target_step_pose = dict(current_all)
    target_step_pose[joint] = cur + step_deg

    send_targets_deg(robot, current_all)
    time.sleep(0.25)

    rows = []
    record_at_rate(robot, joint, baseline_s, sample_rate_hz, rows, "baseline")

    t_step = time.perf_counter()
    send_targets_deg(robot, target_step_pose)
    if cmd_hz <= 0:
        record_at_rate(robot, joint, hold_s, sample_rate_hz, rows, "step")
    else:
        # Interleave commanding at cmd_hz with sampling at sample_rate_hz.
        cmd_period = 1.0 / cmd_hz
        next_cmd = time.perf_counter() + cmd_period
        sample_period = 1.0 / sample_rate_hz
        next_sample = time.perf_counter()
        t_end = time.perf_counter() + hold_s
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break
            wait_until = min(next_cmd, next_sample)
            if now < wait_until:
                time.sleep(max(0.0, wait_until - now))
            now = time.perf_counter()
            if now >= next_cmd:
                send_targets_deg(robot, target_step_pose)
                next_cmd += cmd_period
            if now >= next_sample:
                deg = read_all_deg(robot)
                rows.append((time.perf_counter(), "step", deg[joint]))
                next_sample += sample_period

    # Return to baseline so the next trial starts clean.
    send_targets_deg(robot, current_all)
    record_at_rate(robot, joint, 1.5, sample_rate_hz, rows, "return")

    times = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows])
    fit = fit_fopdt(times, y, t_step)
    fit["step_deg"] = step_deg
    return fit


def main():
    print("Connecting...")
    robot = build_robot(ROBOT_PORT)
    robot.connect()
    try:
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        rest_pose = rest_servo_deg()
        print("Moving to deployment start pose...")
        smooth_move_to(robot, rest_pose)
        time.sleep(0.8)

        results = []
        for label, cmd_hz in [("one-shot", 0), ("30 Hz stream", 30.0),
                              ("100 Hz stream", 100.0)]:
            smooth_move_to(robot, rest_pose)
            time.sleep(0.5)
            fit = probe_with_streamed_cmd(robot, "shoulder_lift",
                                          step_rad=0.1, cmd_hz=cmd_hz)
            results.append((label, fit))
            print(f"  {label:<14}  L={fit['L']*1000:6.1f} ms  "
                  f"tau={fit['tau']*1000:6.1f} ms  R^2={fit['r2']:.3f}")
    finally:
        try:
            smooth_move_to(robot, rest_servo_deg())
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")

    print()
    print("=" * 60)
    print("Result: if L and tau cluster within ~5 ms / ~10 ms across all")
    print("three trials, the 60 ms is intrinsic to the servo and is")
    print("invariant to host command rate.")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
