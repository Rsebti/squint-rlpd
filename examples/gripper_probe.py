"""Measure the gripper's closed angle (sim degrees) empty vs. on the cube, to
calibrate GRASP_EMPTY_BELOW_DEG in infer_linux.py.

The grasp detector treats "measured gripper angle after a full close > threshold"
as "stalled on an object = grasped". This probe reads the actual stall angle so
the threshold can be set from data instead of guessed.

Only the gripper joint moves; the arm is commanded to hold its current pose.
Run with the follower connected:

    python examples/gripper_probe.py                 # default /dev/ttyACM0
    python examples/gripper_probe.py --port /dev/ttyUSB0
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux
from infer_linux import create_real_robot
from lerobot.motors.motors_bus import MotorNormMode

# Gripper sim<->servo mapping (must match RealRobotAgent in infer_linux.py).
G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73
OPEN_SIM, CLOSE_SIM = 120.0, -10.0          # sim deg: full open / full close
SETTLE_S = 3.0


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

    # Snapshot the arm so we can command it to hold while only the gripper moves.
    present = bus.sync_read("Present_Position")
    arm_hold = {k: float(present[k]) for k in present}

    def gripper_sim():
        return srv_to_sim(bus.sync_read("Present_Position")["gripper"])

    def command_gripper(sim_deg):
        cmd = {f"{k}.pos": arm_hold[k] for k in arm_hold}
        cmd["gripper.pos"] = float(sim_to_srv(sim_deg))
        robot.send_action(cmd)

    def close_and_read(label):
        command_gripper(OPEN_SIM)
        time.sleep(1.5)
        command_gripper(CLOSE_SIM)
        t0 = time.time()
        while time.time() - t0 < SETTLE_S:
            print(f"  {label} closing… gripper = {gripper_sim():6.2f}° (sim)   ", end="\r")
            time.sleep(0.1)
        ang = gripper_sim()
        print(f"\n{label}: stalled at {ang:.2f}° (sim)")
        return ang

    try:
        input("\nRemove anything from the gripper, then [Enter] to measure EMPTY close ")
        empty = close_and_read("EMPTY")
        input("\nPlace the CUBE between the fingers (hold it), then [Enter] to measure CUBE close ")
        cube = close_and_read("CUBE")

        print("\n──────── results ────────")
        print(f"  empty close: {empty:6.2f}°")
        print(f"  cube  close: {cube:6.2f}°")
        if cube > empty:
            thr = (empty + cube) / 2.0
            print(f"  → set --grasp_empty_below_deg {thr:.1f}  "
                  f"(midpoint; grasped if measured > {thr:.1f}°)")
        else:
            print("  ⚠ cube angle is not above empty — position-based detection won't "
                  "separate them; consider the load/current method.")
    finally:
        command_gripper(OPEN_SIM)
        time.sleep(1.0)
        robot.disconnect()


if __name__ == "__main__":
    main()
