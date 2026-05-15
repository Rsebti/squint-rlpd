"""Read the gripper's servo-degree position at its two extremes.

Use the two numbers it prints to update _g_servo_min / _g_servo_max in infer.py
(line ~78). They tell the sim->servo gripper remap what "closed" and "open" mean
on this physical arm.

Run:
    python probe_gripper.py
"""
import time
from pathlib import Path

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

from infer import ROBOT_PORT, CAMERA_INDEX, CALIBRATION_ID, CALIBRATION_DIR


def read_gripper_deg(robot):
    return robot.bus.sync_read("Present_Position")["gripper"]


def main():
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX, fps=30, width=640, height=480,
        )},
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    robot = make_robot_from_config(config)
    robot.connect()
    robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    # Disable torque on the gripper so it can be moved by hand.
    try:
        robot.bus.disable_torque(motor_names=["gripper"])
    except Exception:
        try:
            robot.bus.disable_torque("gripper")
        except Exception:
            print("(could not disable torque programmatically — hold gripper relaxed if needed)")

    print("\nGripper probe — torque should be off so you can move it by hand.\n")
    try:
        input("[1/2] Close the gripper fully by hand, then press Enter… ")
        closed_vals = [read_gripper_deg(robot) for _ in range(10)]
        time.sleep(0.05)
        closed = sum(closed_vals) / len(closed_vals)
        print(f"   closed servo deg = {closed:+.3f}")

        input("\n[2/2] Open the gripper fully by hand, then press Enter… ")
        open_vals = [read_gripper_deg(robot) for _ in range(10)]
        opened = sum(open_vals) / len(open_vals)
        print(f"   open   servo deg = {opened:+.3f}")

        print("\n────────────────────────────────────────────────────────────────")
        print("Update infer.py (around line 78) with these two values:")
        print(f"    self._g_servo_min, self._g_servo_max = {closed:.2f}, {opened:.2f}")
        print("────────────────────────────────────────────────────────────────")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
