#!/usr/bin/env python3
"""End-to-end sim-to-real characterisation suite.

Runs every empirical test we need to centre the sim plant on the real arm.
Designed for a one-shot launch once the robot + camera are reconnected:

    python probe_all.py

Stages:
  [1/3] PID registers       — read on-board P / I / D from each Feetech servo
  [2/3] Servo step response — 0.1 rad step on each arm joint at 100 Hz sample
                              rate; fits FOPDT delay L and time constant tau
  [3/3] Camera latency      — step shoulder_lift by 0.2 rad, log Present_Position
                              at 200 Hz AND camera frames at native fps; align
                              motion onsets to extract camera-only lag

Auto-detects:
  - robot serial port: first /dev/cu.usbmodem* (Mac) or /dev/tty[USB|ACM]* (Linux)
  - camera index:      tries --camera_index (default 0) then falls back to 1, 2

Outputs:
  - per-stage CSVs in debug_artifacts/
  - one consolidated summary at the end listing the numbers to paste into
    envs/robot/so101.py + (the not-yet-implemented) observation delay buffer.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# Helpers from the individual probes
from probe_step_response import (
    find_robot_port,
    ARTIFACT_DIR,
    DT_CTRL_DEFAULT,
    build_robot,
    read_all_deg,
    send_targets_deg,
    record_at_rate,
    fit_fopdt,
)
from probe_all_joints import (
    ARM_JOINTS,
    rest_servo_deg,
    smooth_move_to,
    choose_step_sign,
    probe_one_joint,
)
from probe_camera_latency import (
    CameraStreamer,
    open_camera,
    servo_logger,
    first_above,
)

from lerobot.motors.motors_bus import MotorNormMode
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None,
                   help="robot serial port; auto-detected if omitted")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--camera_width", type=int, default=640)
    p.add_argument("--camera_height", type=int, default=480)
    p.add_argument("--camera_fps", type=int, default=30)
    p.add_argument("--skip_pid", action="store_true",
                   help="skip stage 1 (PID register read)")
    p.add_argument("--skip_servo", action="store_true",
                   help="skip stage 2 (servo step response, ~70 s)")
    p.add_argument("--skip_camera", action="store_true",
                   help="skip stage 3 (camera latency test, ~15 s)")
    p.add_argument("--dt_ctrl", type=float, default=DT_CTRL_DEFAULT)
    return p.parse_args()


# ─── Stage 1: PID registers ─────────────────────────────────────────────────
def stage_pid(robot):
    p = robot.bus.sync_read("P_Coefficient")
    i = robot.bus.sync_read("I_Coefficient")
    d = robot.bus.sync_read("D_Coefficient")
    uniform = (len(set(p.values())) == 1 and len(set(i.values())) == 1
               and len(set(d.values())) == 1)
    print(f"  motors      P    I    D")
    for m in p:
        print(f"  {m:<12}{p[m]:>4} {i[m]:>4} {d[m]:>4}")
    return {"P": dict(p), "I": dict(i), "D": dict(d), "uniform": uniform}


# ─── Stage 2: servo step response (all joints) ──────────────────────────────
def stage_servo(robot, dt_ctrl):
    rest_pose = rest_servo_deg()
    fits = []
    for j in ARM_JOINTS:
        print(f"  -- {j}")
        try:
            fit = probe_one_joint(robot, j, rest_pose=rest_pose)
            print(f"     L={fit['L']*1000:.1f} ms  "
                  f"tau={fit['tau']*1000:.1f} ms  R^2={fit['r2']:.3f}")
            fits.append(fit)
        except Exception as e:
            print(f"     FAILED: {e}")
            fits.append({"joint": j, "error": str(e)})
    Ls   = [f["L"]   for f in fits if "L"   in f]
    taus = [f["tau"] for f in fits if "tau" in f]
    if not Ls:
        return {"fits": fits, "L_mean": None, "tau_mean": None,
                "delay_steps": None, "lag_alpha": None}
    L_mean   = float(np.mean(Ls))
    tau_mean = float(np.mean(taus))
    return {
        "fits": fits,
        "L_mean": L_mean,
        "tau_mean": tau_mean,
        "delay_steps": int(round(L_mean / dt_ctrl)),
        "lag_alpha":   dt_ctrl / (dt_ctrl + max(tau_mean, 1e-6)),
    }


# ─── Stage 3: camera-vs-servo motion-correlation ────────────────────────────
def stage_camera(robot, cap, joint="shoulder_lift", step_rad=0.2,
                 baseline_s=1.5, hold_s=2.0, warmup_s=2.5):
    """Step the arm hard enough for an unambiguous visual diff and align
    motion onsets in the servo readback (~200 Hz) and the camera diff stream
    (~native fps)."""
    # Prime the camera pipeline directly (no thread) so the first frame is
    # ready by the time the streamer starts. On macOS the very first
    # cap.read() after open can take 1-2 s and would otherwise eat into the
    # warmup window.
    print(f"  camera priming + warm-up {warmup_s}s")
    n_ok, n_fail = 0, 0
    t_end_prime = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end_prime:
        ok, _ = cap.read()
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            time.sleep(0.01)
    print(f"  priming result: {n_ok} ok frames, {n_fail} failed reads")
    if n_ok == 0:
        raise RuntimeError(
            "cap.read() returned no frames during priming. The cv2 device "
            "opened, but it's not delivering video. Likely causes: (1) the "
            "USB webcam is unplugged so VideoCapture(0) fell back to the "
            "built-in FaceTime cam, which then refused (privacy permission "
            "or lid closed); (2) wrong --camera_index. Run "
            "`scripts/list_cameras.py` (or use any cam app) to find the "
            "right index.")

    streamer = CameraStreamer(cap)
    streamer.start()
    # Brief moment for the streamer thread to settle into its read loop.
    time.sleep(0.3)

    print("  collecting reference frame")
    streamer.set_baseline_from_now(n_frames=10, timeout_s=5.0)

    servo_log = []
    servo_stop = threading.Event()
    servo_thread = threading.Thread(
        target=servo_logger,
        args=(robot, joint, servo_stop, servo_log, 200.0),
        daemon=True,
    )
    servo_thread.start()

    print(f"  baseline {baseline_s}s")
    time.sleep(baseline_s)

    initial = read_all_deg(robot)
    target = dict(initial)
    target[joint] = initial[joint] + math.degrees(step_rad)
    print(f"  step: {joint} {initial[joint]:+.2f} -> {target[joint]:+.2f} deg")
    send_targets_deg(robot, target)
    t_cmd = time.perf_counter()
    time.sleep(hold_s)

    servo_stop.set()
    streamer.stop()
    servo_thread.join(timeout=1.0)

    # Restore
    send_targets_deg(robot, initial)
    time.sleep(0.4)

    if not streamer.records or len(streamer.records) < 10:
        return {"error": "no camera frames"}
    if not servo_log:
        return {"error": "no servo samples"}

    cam_ts  = np.array([t for t, _ in streamer.records])
    cam_dv  = np.array([d for _, d in streamer.records])
    s_ts    = np.array([t for t, _ in servo_log])
    s_pos   = np.array([p for _, p in servo_log])

    eff_fps = 1.0 / np.mean(np.diff(cam_ts)) if len(cam_ts) > 1 else float("nan")

    pre_cam = cam_dv[cam_ts < t_cmd]
    if len(pre_cam) < 5:
        return {"error": "no pre-step camera samples"}
    cam_thr = max(pre_cam.mean() + 5 * pre_cam.std(), 1.0)

    pre_servo = s_pos[s_ts < t_cmd]
    if len(pre_servo) < 5:
        return {"error": "no pre-step servo samples"}
    s_baseline = float(pre_servo.mean())
    s_dev = np.abs(s_pos - s_baseline)

    t_servo, _  = first_above(list(zip(s_ts.tolist(), s_dev.tolist())),
                              0.5, t_cmd)
    t_visual, _ = first_above(list(zip(cam_ts.tolist(), cam_dv.tolist())),
                              cam_thr, t_cmd)

    if t_servo is None or t_visual is None:
        return {"error": "motion not detected in one of the channels",
                "t_servo": t_servo, "t_visual": t_visual}

    return {
        "t_cmd": t_cmd,
        "t_servo": t_servo,
        "t_visual": t_visual,
        "cmd_to_servo_ms":  (t_servo  - t_cmd) * 1000.0,
        "cmd_to_visual_ms": (t_visual - t_cmd) * 1000.0,
        "camera_lag_ms":    (t_visual - t_servo) * 1000.0,
        "effective_fps":    eff_fps,
        "frame_period_ms":  1000.0 / eff_fps if eff_fps > 0 else float("nan"),
    }


# ─── consolidated calibration file ──────────────────────────────────────────
def write_calibration_md(results: dict, args, path: Path) -> None:
    """Overwrite `latency_calibration.md` with the current run's numbers.

    Robust to partial results: every section reports what it has and prints
    "(skipped)" or "(failed)" otherwise. probe_all.py is the source of truth
    for this file — it always overwrites it after a run.
    """
    dt_ctrl_ms = args.dt_ctrl * 1000.0
    ctrl_hz    = 1.0 / args.dt_ctrl

    L = []
    L.append("# SO101 latency calibration\n")
    L.append(f"**Robot:** SO101 follower (Feetech STS3215 servos × 6)  ")
    L.append(f"**Calibration ID:** `so101_follower_arm`  ")
    L.append(f"**Control rate:** {ctrl_hz:.0f} Hz (`dt_ctrl = {dt_ctrl_ms:.1f} ms`)  ")
    L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    L.append("**Source:** `probe_all.py`\n")
    L.append("---\n")

    # ── PID ─────────────────────────────────────────────────────────────────
    L.append("## 1. On-board PID coefficients\n")
    pid = results.get("pid")
    if pid and "error" not in pid:
        L.append("| motor | P | I | D |")
        L.append("|---|---:|---:|---:|")
        for m in pid["P"]:
            L.append(f"| {m} | {pid['P'][m]} | {pid['I'][m]} | {pid['D'][m]} |")
        if pid.get("uniform"):
            any_m = next(iter(pid["P"]))
            L.append(f"\nAll six servos share the same firmware preset: "
                     f"**P={pid['P'][any_m]}, I={pid['I'][any_m]}, "
                     f"D={pid['D'][any_m]}**. PD only (I=0). The sim "
                     "controller (PD + delay + lag) is the correct structural "
                     "model. uint8 firmware coefficients are *not* "
                     "transferable to PhysX gains — only the step-response "
                     "behaviour is.\n")
        else:
            L.append("\n⚠ Per-motor PID values differ — see table.\n")
    elif pid:
        L.append(f"(failed: {pid.get('error')})\n")
    else:
        L.append("(skipped)\n")

    # ── Servo step response ─────────────────────────────────────────────────
    L.append("---\n\n## 2. Servo step response per arm joint\n")
    s = results.get("servo")
    if s and s.get("L_mean") is not None:
        L.append("FOPDT fit `y(t) = y₀ + Δ·(1 − exp(−(t − L)/τ))` "
                 "(scipy.curve_fit refinement of graphical estimate).\n")
        L.append("| joint | step° | L (ms) | τ (ms) | R² |")
        L.append("|---|---:|---:|---:|---:|")
        for f in s["fits"]:
            if "error" in f:
                L.append(f"| {f['joint']} | — | — | — | — (failed: {f['error']}) |")
            else:
                L.append(f"| {f['joint']} | {f['step_deg']:+.2f} | "
                         f"{f['L']*1000:.1f} | {f['tau']*1000:.1f} | "
                         f"{f['r2']:.3f} |")
        L.append(f"| **mean** | | **{s['L_mean']*1000:.1f}** | "
                 f"**{s['tau_mean']*1000:.1f}** | — |")
        L.append(f"\nAt `dt_ctrl = {dt_ctrl_ms:.1f} ms`:")
        L.append(f"- `ACTION_DELAY_STEPS_DEFAULT = {s['delay_steps']}` "
                 f"(modeled {s['delay_steps']*dt_ctrl_ms:.1f} ms vs measured "
                 f"{s['L_mean']*1000:.1f} ms)")
        L.append(f"- `LAG_ALPHA_DEFAULT = {s['lag_alpha']:.3f}` "
                 f"(τ_target = {s['tau_mean']*1000:.1f} ms)\n")
    elif s:
        L.append(f"(failed: {s.get('error')})\n")
    else:
        L.append("(skipped)\n")

    # ── Camera ──────────────────────────────────────────────────────────────
    L.append("---\n\n## 3. Camera latency (motion-correlation)\n")
    c = results.get("camera")
    if c and "error" not in c:
        obs_steps = max(1, int(round(c["camera_lag_ms"] / dt_ctrl_ms)))
        L.append("Single shoulder_lift 0.2 rad step, simultaneous "
                 "servo readback (200 Hz) + camera frame stream (native fps), "
                 "motion onsets aligned.\n")
        L.append("| metric | value |")
        L.append("|---|---:|")
        L.append(f"| effective fps | {c['effective_fps']:.2f} |")
        L.append(f"| frame period (ms) | {c['frame_period_ms']:.2f} |")
        L.append(f"| cmd → servo motion (ms) | {c['cmd_to_servo_ms']:.1f} |")
        L.append(f"| cmd → first visual motion (ms) | {c['cmd_to_visual_ms']:.1f} |")
        L.append(f"| **camera-only lag (ms)** | "
                 f"**{c['camera_lag_ms']:.1f}** "
                 f"(± {c['frame_period_ms']/2:.1f} ms frame quantisation) |")
        L.append(f"\nAt `dt_ctrl = {dt_ctrl_ms:.1f} ms`: "
                 f"`OBS_DELAY_STEPS` suggestion = **{obs_steps}** "
                 "(to plug into the observation-delay buffer once "
                 "implemented).\n")
    elif c:
        L.append(f"(failed: {c.get('error')})\n")
    else:
        L.append("(skipped)\n")

    # ── Where the constants live in the codebase ────────────────────────────
    L.append("---\n\n## 4. Constants currently committed in source\n")
    L.append("| constant | value | location |")
    L.append("|---|---:|---|")
    L.append("| `ACTION_DELAY_STEPS_DEFAULT` | 2 | "
             "`envs/robot/so101.py:33` |")
    L.append("| `LAG_ALPHA_DEFAULT` | 0.378 | "
             "`envs/robot/so101.py:34` |")
    L.append("| arm delta cap (rad/step) | ±0.0333 | "
             "`envs/robot/so101.py:212-213` |")
    L.append("| gripper delta cap (rad/step) | ±0.0667 | "
             "`envs/robot/so101.py:212-213` |")
    L.append("| `sim_freq / control_freq` | 300 / 30 | "
             "`envs/base_random_env.py:149` |\n")
    L.append("Controller-DR ranges live in "
             "`envs/base_random_env.py:62-76` (arm_*_range, "
             "action_delay_steps_range, lag_alpha_range, "
             "gripper_*_range).\n")

    # ── Reproduction commands ───────────────────────────────────────────────
    L.append("---\n\n## 5. Reproducing this calibration\n")
    L.append("```bash")
    L.append("# Full sweep (robot + camera connected, ~90 s):")
    L.append("python latency_debug/probe_all.py")
    L.append("")
    L.append("# Per-stage isolation:")
    L.append("python latency_debug/probe_pid_registers.py     # PID registers, ~1 s")
    L.append("python latency_debug/probe_all_joints.py        # step response, ~70 s")
    L.append("python latency_debug/probe_camera_latency.py    # camera latency, ~15 s")
    L.append("")
    L.append("# Camera diagnostics (no robot needed):")
    L.append("python latency_debug/probe_camera_scan.py 6                            # scan cv2 indices")
    L.append("python latency_debug/probe_camera_latency_solo.py timing --duration 8  # cadence stats")
    L.append("python latency_debug/probe_camera_latency_solo.py flash --n_flashes 6  # screen-flash")
    L.append("")
    L.append("# Archived one-off verifications (in latency_debug/archive/):")
    L.append("python latency_debug/archive/probe_cmd_rate_invariance.py")
    L.append("python latency_debug/archive/refit_step_response.py")
    L.append("```\n")

    L.append("---\n")
    L.append("*This file is regenerated by `latency_debug/probe_all.py` on every run.*\n")

    path.write_text("\n".join(L))
    print(f"Wrote {path}")


# ─── glue ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    port = args.port or find_robot_port()
    if port is None:
        print("ERROR: no robot serial port found on /dev/cu.usbmodem* or "
              "/dev/tty[ACM|USB]*. Is the arm plugged in and powered?")
        return 1
    print(f"Robot port: {port}")

    # NB: the camera is intentionally NOT opened here. An idle USB webcam
    # tends to go dormant on Mac after ~30 s of nobody reading from it,
    # and our servo stage takes ~70 s. The camera is opened inside
    # stage_camera (just before it's needed) and released straight after.

    print(f"Connecting to robot on {port}...")
    robot = build_robot(port)
    robot.connect()
    cap = None  # filled in just before stage 3

    results = {}
    try:
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        rest_pose = rest_servo_deg()
        print("Moving to deployment start pose...")
        smooth_move_to(robot, rest_pose)
        time.sleep(0.6)

        if not args.skip_pid:
            print("\n[1/3] PID register read")
            try:
                results["pid"] = stage_pid(robot)
            except Exception as e:
                print(f"  FAILED: {e}")
                results["pid"] = {"error": str(e)}

        if not args.skip_servo:
            print("\n[2/3] Servo step response (5 joints, ~70 s)")
            try:
                results["servo"] = stage_servo(robot, args.dt_ctrl)
            except Exception:
                traceback.print_exc()
                results["servo"] = {"error": "exception"}

        if not args.skip_camera:
            print("\n[3/3] Camera-vs-servo motion-correlation (~15 s)")
            try:
                # Re-rest before camera test so the arm starts from the same
                # pose servo stage left it (it should already, but be safe).
                smooth_move_to(robot, rest_pose)
                time.sleep(0.4)
                # Open the camera HERE (deferred from main()) so it doesn't
                # idle through the ~70 s servo stage and get dropped by
                # macOS USB power management.
                try:
                    cap = open_camera(args.camera_index, args.camera_width,
                                      args.camera_height, args.camera_fps)
                    print(f"  Camera index {args.camera_index} opened.")
                except Exception as e:
                    print(f"  WARNING: camera open failed: {e}")
                    raise
                results["camera"] = stage_camera(robot, cap)
            except Exception:
                traceback.print_exc()
                results["camera"] = {"error": "exception"}

    finally:
        try:
            smooth_move_to(robot, rest_servo_deg())
            time.sleep(0.3)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        print("Robot disconnected, camera released.")

    # ── Unified summary ─────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("CALIBRATION SUMMARY")
    print("=" * 78)

    pid = results.get("pid")
    if pid and "error" not in pid and pid.get("uniform"):
        any_m = next(iter(pid["P"]))
        print(f"On-board PID (all six servos): "
              f"P={pid['P'][any_m]}, I={pid['I'][any_m]}, D={pid['D'][any_m]}  "
              "(uint8 firmware preset)")
    elif pid and "error" not in pid:
        print(f"On-board PID: NOT uniform — per-motor values "
              "(see debug_artifacts logs or re-run probe_pid_registers.py)")
    elif pid:
        print(f"On-board PID: read failed ({pid.get('error')})")
    else:
        print("On-board PID: skipped")

    servo = results.get("servo")
    if servo and servo.get("L_mean") is not None:
        print(f"Servo step-response (mean over 5 joints):")
        print(f"  L  = {servo['L_mean']*1000:5.1f} ms")
        print(f"  tau= {servo['tau_mean']*1000:5.1f} ms")
        print(f"  -> at dt_ctrl={args.dt_ctrl*1000:.1f} ms ("
              f"{1/args.dt_ctrl:.0f} Hz):")
        print(f"       ACTION_DELAY_STEPS_DEFAULT = {servo['delay_steps']}")
        print(f"       LAG_ALPHA_DEFAULT          = {servo['lag_alpha']:.3f}")
        print(f"     (paste into envs/robot/so101.py lines 33-34)")
    elif servo:
        print(f"Servo: skipped/failed ({servo.get('error', 'no fits')})")
    else:
        print("Servo: skipped")

    cam = results.get("camera")
    if cam and "error" not in cam:
        frame_dt = cam["frame_period_ms"]
        print(f"Camera (single shoulder_lift step):")
        print(f"  cmd -> servo motion        = {cam['cmd_to_servo_ms']:5.1f} ms")
        print(f"  cmd -> first visual motion = {cam['cmd_to_visual_ms']:5.1f} ms")
        print(f"  camera-only lag            = {cam['camera_lag_ms']:5.1f} ms  "
              f"(±{frame_dt/2:.1f} ms frame quantisation)")
        # Suggested observation-delay in sim:
        obs_steps = max(1, int(round(cam["camera_lag_ms"] / (args.dt_ctrl * 1000))))
        print(f"  -> at dt_ctrl={args.dt_ctrl*1000:.1f} ms:")
        print(f"       OBS_DELAY_STEPS suggestion = {obs_steps}  "
              "(add to the observation-delay buffer once implemented)")
    elif cam:
        print(f"Camera: failed ({cam.get('error')})")
    else:
        print("Camera: skipped")

    print("=" * 78)

    # Overwrite the consolidated calibration file with the current run.
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        write_calibration_md(results, args,
                             ARTIFACT_DIR / "latency_calibration.md")
    except Exception as e:
        print(f"WARNING: could not write latency_calibration.md: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
