#!/usr/bin/env python3
"""Real-arm step-response probe for the SO101 follower.

Commands a small joint step (default: shoulder_lift, +0.1 rad), records
Present_Position at ~100 Hz, and fits a first-order-plus-dead-time (FOPDT)
model
    y(t) = y0 + Delta * (1 - exp(-(t - L) / tau))    for t >= L
to extract the actuator delay L and time constant tau. From those, prints
the matching values for the sim controller in envs/robot/so101.py:

    ACTION_DELAY_STEPS_DEFAULT = round(L / dt_ctrl)
    LAG_ALPHA_DEFAULT          = dt_ctrl / (dt_ctrl + tau)

where dt_ctrl is the training control period (default 0.1 s = 10 Hz).

Run with the policy disabled and the workspace clear. The script torques
the arm at its current pose, steps a single joint, records ~3 s of
response, and returns to the starting pose.

Usage:
    python probe_step_response.py
    python probe_step_response.py --joint elbow_flex --step_rad 0.05 --rate_hz 100
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode


# Defaults mirror infer.py so the probe sees the same arm/calibration.
# These probes now live under latency_debug/, but the calibration JSON
# (so101_follower_arm.json) and the shared debug_artifacts/ directory both
# sit at the project root, so resolve one level up.
REPO_ROOT       = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = REPO_ROOT
CALIBRATION_ID  = "so101_follower_arm"
ARTIFACT_DIR    = REPO_ROOT / "debug_artifacts"


def find_robot_port(default: str | None = None) -> str | None:
    """Auto-detect the SO101 follower's serial port.

    Mac shows the Feetech CH343 bridge as `/dev/cu.usbmodem*`; Linux uses
    `/dev/ttyACM*` (CDC) or `/dev/ttyUSB*` (FTDI). If multiple devices match,
    returns the first one alphabetically. Returns `default` if nothing found
    so callers can still run without the robot plugged in at import time.
    """
    import glob
    import platform
    if platform.system() == "Darwin":
        candidates = sorted(glob.glob("/dev/cu.usbmodem*"))
    else:
        candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else default


# Resolve at import; tests should call find_robot_port() again at runtime
# in case the user unplugged/replugged the arm between import and use.
ROBOT_PORT = find_robot_port(default="/dev/cu.usbmodem5B141129871")

# Soft per-joint guard: refuse a step that would push us past this many
# degrees from zero. Joint hard limits are ~100 deg for shoulder_lift,
# elbow_flex, wrist_flex; this gives generous headroom.
SOFT_LIMIT_DEG  = 80.0
ARM_JOINTS      = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll")

DT_CTRL_DEFAULT = 1.0 / 30.0   # 30 Hz training control rate (matches infer.py CONTROL_HZ)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joint", default="shoulder_lift", choices=ARM_JOINTS,
                   help="arm joint to probe (gripper excluded)")
    p.add_argument("--step_rad", type=float, default=0.1,
                   help="step magnitude in radians (use negative to step the other way)")
    p.add_argument("--baseline_s", type=float, default=1.5,
                   help="seconds of pre-step recording for baseline / noise floor")
    p.add_argument("--hold_s", type=float, default=3.0,
                   help="seconds to record after the step before returning")
    p.add_argument("--rate_hz", type=float, default=100.0,
                   help="best-effort sampling rate; serial bus may cap this")
    p.add_argument("--dt_ctrl", type=float, default=DT_CTRL_DEFAULT,
                   help="training control period in seconds; 0.1 = 10 Hz")
    p.add_argument("--port", default=ROBOT_PORT)
    p.add_argument("--no_return", action="store_true",
                   help="skip the return-to-baseline recording")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--out_tag", default=None,
                   help="suffix for the artefact filenames (default: timestamp)")
    return p.parse_args()


def build_robot(port):
    cfg = SO101FollowerConfig(
        port=port,
        use_degrees=True,
        cameras={},                       # cameras irrelevant for this probe
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    return make_robot_from_config(cfg)


def read_all_deg(robot):
    """sync_read returns {motor_name: degrees}."""
    return robot.bus.sync_read("Present_Position")


def send_targets_deg(robot, targets_deg):
    cmd = {f"{k}.pos": float(v) for k, v in targets_deg.items()}
    robot.send_action(cmd)


def record_at_rate(robot, joint, duration_s, rate_hz, rows, label):
    """Tight loop reading Present_Position at best-effort rate_hz.

    Records (perf_counter timestamp, label, joint_deg). The serial bus may
    not sustain 100 Hz for sync_read of 6 motors; actual timestamps are
    used for the fit so loop jitter is harmless.
    """
    period = 1.0 / rate_hz
    t_end = time.perf_counter() + duration_s
    next_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now >= t_end:
            return
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        try:
            deg = read_all_deg(robot)
        except Exception as e:
            print(f"[probe] read failed at t={now:.3f}: {e}", file=sys.stderr)
            time.sleep(0.005)
            continue
        rows.append((time.perf_counter(), label, deg[joint]))
        next_t += period


def fit_fopdt(t, y, t_step, baseline_window=0.3, settle_window=0.4):
    """Fit y(t) = y0 + Delta * (1 - exp(-(t-L)/tau)) using graphical
    estimates as initialisation, refined with scipy.optimize.curve_fit when
    available. Returns a dict of parameters and goodness-of-fit.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rel = t - t_step
    pre = rel < 0
    post = rel >= 0
    if pre.sum() < 5 or post.sum() < 20:
        raise RuntimeError("Too few samples around the step to fit.")

    pre_mask = pre & (rel > -baseline_window)
    y0 = float(y[pre_mask].mean()) if pre_mask.any() else float(y[pre].mean())

    # Settled value = mean of last `settle_window` seconds of recorded data.
    post_t_max = rel[post].max()
    settle_mask = post & (rel > post_t_max - settle_window)
    y_inf = float(y[settle_mask].mean())
    delta = y_inf - y0
    if abs(delta) < 0.2:        # < 0.2 deg of actual movement
        raise RuntimeError(
            f"Joint barely moved ({delta:+.3f} deg). Either the command was "
            "rejected, the joint is at a hard stop, or torque is disabled.")

    # Normalised rise [0, 1]
    r = (y - y0) / delta

    # Graphical delay: first post-step sample with r > 0.05
    post_idx = np.where(post)[0]
    above5 = (r[post] > 0.05)
    if not above5.any():
        raise RuntimeError("Joint never rose 5% toward the commanded target.")
    L_graph = float(rel[post_idx[np.argmax(above5)]])

    # Graphical tau: time after delay when r >= 0.632, minus delay
    cross632 = (r >= 0.632) & post
    tau_graph = (float(rel[cross632].min()) - L_graph) if cross632.any() else 0.15

    L, tau, method, r2 = L_graph, tau_graph, "graphical", float("nan")

    try:
        from scipy.optimize import curve_fit

        def model_norm(tt, L_, tau_):
            tt = np.asarray(tt)
            return np.where(
                tt < L_, 0.0,
                1.0 - np.exp(-np.maximum(tt - L_, 1e-9) / max(tau_, 1e-6)),
            )

        tt_post = rel[post]
        upper_L = max(0.5, float(tt_post.max()))
        upper_tau = max(1.0, float(tt_post.max()))
        bounds = ([0.0, 1e-3], [upper_L, upper_tau])
        p0 = [max(L_graph, 1e-3), max(tau_graph, 1e-3)]
        popt, _ = curve_fit(model_norm, tt_post, r[post], p0=p0,
                            bounds=bounds, maxfev=5000)
        L_fit, tau_fit = float(popt[0]), float(popt[1])
        y_pred = y0 + delta * model_norm(tt_post, L_fit, tau_fit)
        ss_res = float(((y[post] - y_pred) ** 2).sum())
        ss_tot = float(((y[post] - y[post].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        L, tau, method = L_fit, tau_fit, "scipy.curve_fit"
    except Exception as e:
        print(f"[fit] scipy refinement skipped: {e}")

    return dict(y0=y0, y_inf=y_inf, delta=delta, L=L, tau=tau,
                method=method, r2=r2, L_graph=L_graph, tau_graph=tau_graph)


def make_plot(rows, t_step, t_return, joint, fit, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable, skipping: {e}")
        return

    t_arr = np.array([r[0] for r in rows])
    y_arr = np.array([r[2] for r in rows])
    t0 = t_arr[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_arr - t0, y_arr, lw=1.0, label="measured Present_Position")
    ax.axvline(t_step - t0, color="r", lw=0.8, ls="--", label="step commanded")
    if t_return is not None:
        ax.axvline(t_return - t0, color="b", lw=0.8, ls="--", label="return commanded")

    post = t_arr >= t_step
    tt = t_arr[post] - t_step
    L, tau = fit["L"], fit["tau"]
    yhat = fit["y0"] + fit["delta"] * np.where(
        tt < L, 0.0, 1.0 - np.exp(-np.maximum(tt - L, 1e-9) / max(tau, 1e-6))
    )
    ax.plot(t_arr[post] - t0, yhat, color="orange", lw=1.2,
            label=f"FOPDT fit: L={L*1000:.0f} ms, tau={tau*1000:.0f} ms "
                  f"(R^2={fit['r2']:.3f})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{joint} (deg)")
    ax.set_title(f"SO101 step response: {joint}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"Wrote {png_path}")


def main():
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    csv_path = ARTIFACT_DIR / f"step_response_{args.joint}_{tag}.csv"
    png_path = ARTIFACT_DIR / f"step_response_{args.joint}_{tag}.png"

    step_rad = float(args.step_rad)
    step_deg = math.degrees(step_rad)

    print(f"Connecting to {args.port}...")
    robot = build_robot(args.port)
    robot.connect()
    try:
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

        initial = read_all_deg(robot)
        cur = initial[args.joint]
        target_value = cur + step_deg
        print("Initial positions (deg):",
              {k: round(v, 2) for k, v in initial.items()})

        if abs(target_value) > SOFT_LIMIT_DEG:
            raise RuntimeError(
                f"Aborting: target {target_value:+.2f} deg on {args.joint} "
                f"exceeds soft limit ±{SOFT_LIMIT_DEG:.0f} deg. "
                "Move the arm closer to zero, flip --step_rad sign, or "
                "raise SOFT_LIMIT_DEG.")

        # Engage torque at the current pose so the step is the only motion.
        send_targets_deg(robot, initial)
        time.sleep(0.5)

        target_step = dict(initial)
        target_step[args.joint] = target_value
        target_back = dict(initial)
        rows = []   # (perf_counter, label, deg)

        print(f"Baseline: {args.baseline_s:.1f}s at ~{args.rate_hz:.0f} Hz")
        record_at_rate(robot, args.joint, args.baseline_s, args.rate_hz,
                       rows, "baseline")

        print(f"Step: {args.joint} {cur:+.2f} -> {target_value:+.2f} deg "
              f"({step_deg:+.2f} deg / {step_rad:+.3f} rad)")
        t_step = time.perf_counter()
        send_targets_deg(robot, target_step)
        record_at_rate(robot, args.joint, args.hold_s, args.rate_hz,
                       rows, "step")

        t_return = None
        if not args.no_return:
            print(f"Return: {args.joint} {target_value:+.2f} -> {cur:+.2f} deg")
            t_return = time.perf_counter()
            send_targets_deg(robot, target_back)
            record_at_rate(robot, args.joint, args.hold_s, args.rate_hz,
                           rows, "return")
    finally:
        try:
            send_targets_deg(robot, initial)
            time.sleep(0.3)
        except Exception:
            pass
        robot.disconnect()
        print("Disconnected.")

    if not rows:
        raise RuntimeError("No samples recorded.")

    # Save CSV (time normalised to first sample for readability)
    t_first = rows[0][0]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "label", f"{args.joint}_deg"])
        for (t, lab, d) in rows:
            w.writerow([f"{t - t_first:.6f}", lab, f"{d:.6f}"])
    print(f"Wrote {csv_path}")

    # Effective sample rate report
    times = np.array([r[0] for r in rows])
    dt = np.diff(times)
    eff_hz = 1.0 / dt.mean() if len(dt) else float("nan")
    print(f"Effective sample rate: {eff_hz:.1f} Hz "
          f"(median dt {np.median(dt)*1000:.2f} ms, "
          f"p95 dt {np.percentile(dt, 95)*1000:.2f} ms)")

    # Fit the step segment
    y_arr = np.array([r[2] for r in rows])
    fit = fit_fopdt(times, y_arr, t_step)

    L, tau = fit["L"], fit["tau"]
    dt_ctrl = args.dt_ctrl
    delay_steps = int(round(L / dt_ctrl))
    lag_alpha = dt_ctrl / (dt_ctrl + max(tau, 1e-6))
    tracking_err = (fit["y_inf"] - fit["y0"]) - step_deg

    print()
    print("=" * 72)
    print(f"Fit ({fit['method']}, R^2={fit['r2']:.3f})")
    print(f"  baseline y0          {fit['y0']:8.3f} deg")
    print(f"  settled y_inf        {fit['y_inf']:8.3f} deg")
    print(f"  commanded delta      {step_deg:+8.3f} deg")
    print(f"  realised delta       {fit['delta']:+8.3f} deg  "
          f"(tracking error {tracking_err:+.3f} deg)")
    print(f"  pure delay L         {L*1000:8.1f} ms     "
          f"(graphical: {fit['L_graph']*1000:.1f} ms)")
    print(f"  time constant tau    {tau*1000:8.1f} ms     "
          f"(graphical: {fit['tau_graph']*1000:.1f} ms)")
    print()
    print(f"For dt_ctrl = {dt_ctrl*1000:.0f} ms  ({1.0/dt_ctrl:.0f} Hz training control):")
    print(f"  ACTION_DELAY_STEPS_DEFAULT = {delay_steps}")
    print(f"  LAG_ALPHA_DEFAULT          = {lag_alpha:.3f}")
    print()
    print("Plug these into envs/robot/so101.py (top of file).")
    print("=" * 72)

    if not args.no_plot:
        make_plot(rows, t_step, t_return, args.joint, fit, png_path)


if __name__ == "__main__":
    sys.exit(main())
