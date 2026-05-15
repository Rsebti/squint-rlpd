#!/usr/bin/env python3
"""Step-response probe for all five SO101 arm joints, started from the
deployment "start" keyframe (REST_QPOS in infer.py).

Reuses helpers from probe_step_response.py. Prints one compact table at
the end with per-joint L, tau, R^2, and the matching sim-side constants.
"""
from __future__ import annotations

import csv as _csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.motors.motors_bus import MotorNormMode

from probe_step_response import (
    ROBOT_PORT,
    ARTIFACT_DIR,
    DT_CTRL_DEFAULT,
    build_robot,
    read_all_deg,
    send_targets_deg,
    record_at_rate,
    fit_fopdt,
)


# Probe "rest" pose. Biased away from the deployment "start" keyframe
# (shoulder_lift=0, elbow_flex=0) so the arm clears the table during step
# tests — this matters because each joint is stepped ±0.1 to ±0.2 rad
# from rest, and at the bare start pose the wrist can be uncomfortably
# close to the workspace surface.
#
# URDF keyframes for reference:
#   - "rest"  (so101.py:160) : shoulder_lift=-90° elbow_flex=+90°  (fully tucked)
#   - "start" (so101.py:55)  : shoulder_lift=0    elbow_flex=0     (fully extended)
# Negative shoulder_lift = up; positive elbow_flex = elbow bent toward chest.
REST_QPOS_RAD = {
    "shoulder_pan":  0.0,
    "shoulder_lift": math.radians(-45.0),
    "elbow_flex":    math.radians( 30.0),
    "wrist_flex":    math.pi / 2,
    "wrist_roll":   -math.pi / 2,
    "gripper":       math.radians(60.0),
}

# Joint hard limits in degrees, from JOINT_LOWER/UPPER in infer.py.
HARD_LIMITS_DEG = {
    "shoulder_pan":  (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex":    ( -97.0,  97.0),
    "wrist_flex":    ( -95.0,  95.0),
    "wrist_roll":    (-157.2, 162.8),
}

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex",   "wrist_roll")

# Gripper sim<->servo mapping from infer.py:82-86.
GRIPPER_SIM_MIN,   GRIPPER_SIM_MAX   = -10.0, 120.0
GRIPPER_SERVO_MIN, GRIPPER_SERVO_MAX = -60.13, 66.73


def gripper_sim_deg_to_servo_deg(sim_deg: float) -> float:
    return ((sim_deg - GRIPPER_SIM_MIN)
            / (GRIPPER_SIM_MAX - GRIPPER_SIM_MIN)
            * (GRIPPER_SERVO_MAX - GRIPPER_SERVO_MIN)
            + GRIPPER_SERVO_MIN)


def rest_servo_deg() -> dict:
    out = {j: math.degrees(REST_QPOS_RAD[j]) for j in ARM_JOINTS}
    out["gripper"] = gripper_sim_deg_to_servo_deg(
        math.degrees(REST_QPOS_RAD["gripper"]))
    return out


def smooth_move_to(robot, target_deg: dict, *,
                   rate_hz: float = 30.0,
                   max_deg_per_step: float = 1.5,
                   timeout_s: float = 20.0):
    """Rate-limited ramp to target_deg. Mirrors infer.py's RealRobotAgent.reset."""
    period = 1.0 / rate_hz
    t_end = time.perf_counter() + timeout_s
    cur = read_all_deg(robot)
    target_pose = dict(cur)
    while time.perf_counter() < t_end:
        any_moved = False
        for k in cur:
            if k not in target_deg:
                continue
            delta = target_deg[k] - target_pose[k]
            clipped = max(-max_deg_per_step, min(max_deg_per_step, delta))
            if abs(clipped) > 1e-4:
                any_moved = True
            target_pose[k] += clipped
        send_targets_deg(robot, target_pose)
        if not any_moved:
            return
        time.sleep(period)
    print("[move] timed out reaching target pose.", file=sys.stderr)


def choose_step_sign(joint: str, current_deg: float, step_mag_deg: float,
                     margin_deg: float = 5.0) -> float:
    lo, hi = HARD_LIMITS_DEG[joint]
    up_room   = hi - current_deg - margin_deg
    down_room = current_deg - lo - margin_deg
    if step_mag_deg <= up_room:
        return +1.0
    if step_mag_deg <= down_room:
        return -1.0
    raise RuntimeError(
        f"{joint} has no safe step direction at {current_deg:.2f} deg "
        f"(hard limits {lo}..{hi}, magnitude {step_mag_deg:.2f}).")


def probe_one_joint(robot, joint: str, *,
                    step_rad: float = 0.1,
                    baseline_s: float = 1.0,
                    hold_s: float = 2.5,
                    rate_hz: float = 100.0,
                    rest_pose: dict | None = None) -> dict:
    if rest_pose is not None:
        smooth_move_to(robot, rest_pose)
        time.sleep(0.4)

    current_all = read_all_deg(robot)
    cur = current_all[joint]
    step_mag_deg = math.degrees(abs(step_rad))
    sign = choose_step_sign(joint, cur, step_mag_deg)
    signed_step_deg = sign * step_mag_deg
    target_value = cur + signed_step_deg

    # Engage torque at current pose so the step is the only motion.
    send_targets_deg(robot, current_all)
    time.sleep(0.25)

    rows = []
    record_at_rate(robot, joint, baseline_s, rate_hz, rows, "baseline")

    t_step = time.perf_counter()
    step_pose = dict(current_all)
    step_pose[joint] = target_value
    send_targets_deg(robot, step_pose)
    record_at_rate(robot, joint, hold_s, rate_hz, rows, "step")

    t_return = time.perf_counter()
    send_targets_deg(robot, current_all)
    record_at_rate(robot, joint, hold_s, rate_hz, rows, "return")

    # Persist CSV
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    csv_path = ARTIFACT_DIR / f"step_response_{joint}_{tag}.csv"
    t_first = rows[0][0]
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t_s", "label", f"{joint}_deg"])
        for (t, lab, d) in rows:
            w.writerow([f"{t-t_first:.6f}", lab, f"{d:.6f}"])

    # IMPORTANT: fit ONLY on baseline + step rows, NOT the return segment.
    # FOPDT computes y_inf as the mean of the last `settle_window` seconds
    # of the post-t_step region; if the return rows are included, that
    # window lands back at baseline -> delta ~ 0 -> fit blows up (tau pinned
    # at lower bound, R^2 negative). Slice up to the first "return" label.
    fit_rows = [r for r in rows if r[1] != "return"]
    times = np.array([r[0] for r in fit_rows])
    y_arr = np.array([r[2] for r in fit_rows])
    fit = fit_fopdt(times, y_arr, t_step)
    fit.update(joint=joint, csv=str(csv_path),
               step_deg=signed_step_deg,
               t_step=t_step, t_return=t_return)
    return fit


def main():
    print("Connecting to robot...")
    robot = build_robot(ROBOT_PORT)
    robot.connect()
    fits = []
    try:
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        rest_pose = rest_servo_deg()
        print(f"Moving to deployment start pose: {rest_pose}")
        smooth_move_to(robot, rest_pose)
        time.sleep(0.8)

        for j in ARM_JOINTS:
            try:
                print(f"-- probing {j} --")
                fit = probe_one_joint(robot, j, rest_pose=rest_pose)
                fits.append(fit)
                print(f"   L={fit['L']*1000:.1f} ms  "
                      f"tau={fit['tau']*1000:.1f} ms  "
                      f"R^2={fit['r2']:.3f}")
            except Exception as e:
                print(f"[{j}] probe failed: {e}", file=sys.stderr)
                fits.append({"joint": j, "error": str(e)})
    finally:
        try:
            smooth_move_to(robot, rest_servo_deg())
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")

    dt_ctrl = DT_CTRL_DEFAULT
    print()
    print("=" * 78)
    print(f"{'joint':<14} {'step°':>7} {'L ms':>7} {'tau ms':>8} "
          f"{'R^2':>6} {'delay_steps':>12} {'lag_alpha':>10}")
    print("-" * 78)
    Ls, taus = [], []
    for fit in fits:
        if "error" in fit:
            print(f"{fit['joint']:<14}  ERROR: {fit['error']}")
            continue
        L_ms   = fit["L"] * 1000.0
        tau_ms = fit["tau"] * 1000.0
        n      = int(round(fit["L"] / dt_ctrl))
        alpha  = dt_ctrl / (dt_ctrl + max(fit["tau"], 1e-6))
        print(f"{fit['joint']:<14} {fit['step_deg']:>7.2f} "
              f"{L_ms:>7.1f} {tau_ms:>8.1f} {fit['r2']:>6.3f} "
              f"{n:>12d} {alpha:>10.3f}")
        Ls.append(fit["L"]); taus.append(fit["tau"])

    if Ls:
        L_mean   = float(np.mean(Ls))
        tau_mean = float(np.mean(taus))
        print("-" * 78)
        print(f"mean over {len(Ls)} joints:  "
              f"L={L_mean*1000:.1f} ms   tau={tau_mean*1000:.1f} ms")
        print(f"  -> ACTION_DELAY_STEPS_DEFAULT = "
              f"{int(round(L_mean/dt_ctrl))}")
        print(f"  -> LAG_ALPHA_DEFAULT          = "
              f"{dt_ctrl/(dt_ctrl+tau_mean):.3f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
