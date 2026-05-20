"""Teach the bowl xy in the robot base frame (the frame the place IK uses).

The place phase drives the TCP to an ABSOLUTE (x, y) in the base_link frame —
the same frame FK/tcp_pos lives in, NOT whatever frame you measured the bowl in.
The reliable way to get the right numbers: physically hold the closed gripper
over the bowl centre and read the base-frame xy off FK.

Arm torque is disabled so you can move it by hand; the gripper is held closed.
Move the gripper so its tip is centred over the bowl, then read the printed xy
(or Ctrl+C to freeze the last value) and pass it to pick_and_place:

    python -m final_utils.teach_bowl_xy
    ...  ->  use --bowl_xy 0.21 -0.06
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux as il
from infer_linux import create_real_robot
from so101_fk import tcp_pos
from lerobot.motors.motors_bus import MotorNormMode

G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73
CLOSE_SIM = -10.0


def sim_to_srv(s):
    return (s - G_SIM_MIN) / (G_SIM_MAX - G_SIM_MIN) * (G_SRV_MAX - G_SRV_MIN) + G_SRV_MIN


def srv_to_sim(s):
    return (s - G_SRV_MIN) / (G_SRV_MAX - G_SRV_MIN) * (G_SIM_MAX - G_SIM_MIN) + G_SIM_MIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=str, default=il.ROBOT_PORT)
    args = p.parse_args()
    il.ROBOT_PORT = args.port

    robot = create_real_robot()
    robot.connect()
    bus = robot.bus
    bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    present = bus.sync_read("Present_Position")
    keys = list(present.keys())                          # bus order == FK joint order
    arm_keys = [k for k in keys if k != "gripper"]

    # Close gripper (so the contact point is a single tip), then free the arm.
    cmd = {f"{k}.pos": float(present[k]) for k in keys}
    cmd["gripper.pos"] = float(sim_to_srv(CLOSE_SIM))
    robot.send_action(cmd)
    time.sleep(1.0)
    bus.disable_torque(arm_keys)

    print("\nArm FREE. Move the gripper tip directly over the BOWL CENTRE.")
    print("Read base-frame xy below; Ctrl+C to freeze the last value.\n")
    last = None
    try:
        while True:
            d = bus.sync_read("Present_Position")
            deg = [srv_to_sim(d[k]) if k == "gripper" else d[k] for k in keys]
            tcp = tcp_pos(np.deg2rad(np.array(deg, dtype=np.float64)))
            last = tcp
            print(f"  base xy = ({tcp[0]:+.3f}, {tcp[1]:+.3f})  z={tcp[2]*100:+5.1f}cm   ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    if last is not None:
        print(f"\n\nBowl centre (base frame): x={last[0]:.3f}  y={last[1]:.3f}")
        print(f"  → run:  python -m final_utils.pick_place --goal_color <C> --bowl_xy {last[0]:.3f} {last[1]:.3f}")

    bus.enable_torque(arm_keys)
    time.sleep(0.5)
    robot.disconnect()


if __name__ == "__main__":
    main()
