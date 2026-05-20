"""Calibrate the table plane in FK-z, as a function of reach distance.

The grasp gate/nudge assume FK z=0 is the table everywhere, but the real arm's
geometry diverges from the URDF with reach, so the FK z that actually corresponds
to "touching the table" drifts with horizontal distance from the base. This script
records that: with the gripper held CLOSED and arm torque OFF, you slide the closed
fingertip along the table from near the base ("feet") out to full extension
("end"). It logs (r, z) where r = horizontal distance of the TCP from the base and
z = FK TCP height, then fits z_table(r) = a*r + b and saves it to table_z_calib.json.

infer_linux.py loads that fit and makes the gate/nudge relative to z_table(r).

    python examples/table_z_calib.py                 # default /dev/ttyACM0
    python examples/table_z_calib.py --port /dev/ttyUSB0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux
from infer_linux import create_real_robot
from so101_fk import tcp_pos
from lerobot.motors.motors_bus import MotorNormMode

G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73
CLOSE_SIM = -10.0
CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "table_z_calib.json")


def srv_to_sim(s):
    return (s - G_SRV_MIN) / (G_SRV_MAX - G_SRV_MIN) * (G_SIM_MAX - G_SIM_MIN) + G_SIM_MIN


def sim_to_srv(s):
    return (s - G_SIM_MIN) / (G_SIM_MAX - G_SIM_MIN) * (G_SRV_MAX - G_SRV_MIN) + G_SRV_MIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=str, default=infer_linux.ROBOT_PORT,
                   help=f"serial device (default {infer_linux.ROBOT_PORT})")
    args = p.parse_args()
    infer_linux.ROBOT_PORT = args.port

    robot = create_real_robot()
    robot.connect()
    bus = robot.bus
    bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    present = bus.sync_read("Present_Position")
    keys = list(present.keys())                       # bus order == FK joint order
    arm_keys = [k for k in keys if k != "gripper"]

    def qpos_rad():
        d = bus.sync_read("Present_Position")
        deg = [srv_to_sim(d[k]) if k == "gripper" else d[k] for k in keys]
        return np.deg2rad(np.array(deg, dtype=np.float64))

    # Close the gripper (torque on) so the contact point is a stable single tip,
    # then free the arm so it can be slid by hand.
    cmd = {f"{k}.pos": float(present[k]) for k in keys}
    cmd["gripper.pos"] = float(sim_to_srv(CLOSE_SIM))
    robot.send_action(cmd)
    time.sleep(1.5)
    bus.disable_torque(arm_keys)

    print("\nArm is now FREE (gripper held closed).")
    print("Keep the closed fingertip touching the table and slide it slowly from")
    print("near the base out to full extension (and back). Press Ctrl+C when done.\n")

    rs, zs = [], []
    try:
        while True:
            tcp = tcp_pos(qpos_rad())
            r = float(np.hypot(tcp[0], tcp[1]))
            rs.append(r)
            zs.append(float(tcp[2]))
            print(f"  samples={len(rs):4d}  r={r*100:6.2f}cm  z={tcp[2]*100:6.2f}cm   ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    rs, zs = np.array(rs), np.array(zs)
    print(f"\n\nCollected {len(rs)} samples, r ∈ [{rs.min()*100:.1f}, {rs.max()*100:.1f}] cm")
    if len(rs) < 10:
        print("Too few samples — not saving. Re-run and slide for longer.")
        bus.enable_torque(arm_keys)
        time.sleep(0.5)
        robot.disconnect()
        return

    # Density/outlier-robust fit: bin by r, take the MEDIAN z per bin, then fit a
    # line over the bins. This way each ~1 cm of reach counts once regardless of
    # how long you lingered there, and brief lift-offs (outliers) are rejected.
    BIN_W = 0.01
    edges = np.arange(rs.min(), rs.max() + BIN_W, BIN_W)
    idx = np.digitize(rs, edges)
    bin_r, bin_z = [], []
    for bi in np.unique(idx):
        m = idx == bi
        bin_r.append(float(rs[m].mean()))
        bin_z.append(float(np.median(zs[m])))
    bin_r, bin_z = np.array(bin_r), np.array(bin_z)
    if len(bin_r) < 2:
        print("Samples span <2 cm of reach — sweep a wider range. Not saving.")
        bus.enable_torque(arm_keys)
        time.sleep(0.5)
        robot.disconnect()
        return

    a, b = np.polyfit(bin_r, bin_z, 1)
    rmse = float(np.sqrt(np.mean((np.polyval([a, b], bin_r) - bin_z) ** 2)))
    # Persist raw samples too, so the fit can be redone/inspected without re-sweeping.
    np.savez(os.path.splitext(CALIB_PATH)[0] + "_raw.npz", r=rs, z=zs)
    out = {"a": float(a), "b": float(b), "r_min": float(rs.min()),
           "r_max": float(rs.max()), "n": int(len(rs)), "n_bins": int(len(bin_r)),
           "rmse_m": rmse, "bin_w_m": BIN_W}
    with open(CALIB_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"fit (median over {len(bin_r)} bins of {BIN_W*100:.0f} cm): "
          f"z_table[m] = {a:.4f}·r + {b:.4f}")
    print(f"  slope {a*100:.2f} cm per 1 m reach, intercept {b*100:.2f} cm, RMSE {rmse*100:.2f} cm")
    print(f"  → saved {CALIB_PATH}  (+ raw samples in *_raw.npz)")

    bus.enable_torque(arm_keys)
    time.sleep(0.5)
    robot.disconnect()


if __name__ == "__main__":
    main()
