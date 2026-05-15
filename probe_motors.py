"""Read each actuator's servo-degree position at its two physical extremes.

Use this to sanity-check that the calibrated motor ranges match what the
policy expects (and to update the gripper's _g_servo_min/_g_servo_max in
infer.py, which is the only joint with a sim<->servo remap).

Run:
    python probe_motors.py
"""
import time

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

from infer import ROBOT_PORT, CAMERA_INDEX, CALIBRATION_ID, CALIBRATION_DIR

# Calibration-JSON motor name -> friendly label used in infer.py / JOINT_NAMES.
MOTORS = [
    ("shoulder_pan", "pan"),
    ("shoulder_lift", "lift"),
    ("elbow_flex", "elbow"),
    ("wrist_flex", "wrist_flex"),
    ("wrist_roll", "wrist_roll"),
    ("gripper", "gripper"),
]


def read_deg(robot, motor):
    return robot.bus.sync_read("Present_Position")[motor]


def avg_read(robot, motor, n=10, dt=0.01):
    vals = []
    for _ in range(n):
        vals.append(read_deg(robot, motor))
        time.sleep(dt)
    return sum(vals) / len(vals)


def disable_torque_all(robot):
    """Best-effort: try a couple of API shapes seen across lerobot versions."""
    try:
        robot.bus.disable_torque()
        return
    except Exception:
        pass
    for motor, _ in MOTORS:
        try:
            robot.bus.disable_torque(motor)
        except Exception:
            try:
                robot.bus.disable_torque(motor_names=[motor])
            except Exception:
                pass


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
    disable_torque_all(robot)

    print("\nMotor probe — torque should be off so you can move each joint by hand.")
    print("For each motor: move it to one extreme, Enter; move to the other extreme, Enter.\n")

    results = {}
    try:
        for motor, label in MOTORS:
            print(f"── {label}  (calib name: {motor}) ──")
            input(f"  [1/2] Move {label} to one extreme, then press Enter… ")
            v1 = avg_read(robot, motor)
            input(f"  [2/2] Move {label} to the other extreme, then press Enter… ")
            v2 = avg_read(robot, motor)
            lo, hi = min(v1, v2), max(v1, v2)
            results[label] = (lo, hi)
            print(f"    {label}: min = {lo:+.3f}°   max = {hi:+.3f}°   (span {hi - lo:.2f}°)\n")
    finally:
        robot.disconnect()

    print("────────────────────────────────────────────────────────────────")
    print(" Summary (servo degrees, post-calibration):")
    for label, (lo, hi) in results.items():
        print(f"   {label:11s}  [{lo:+7.2f}, {hi:+7.2f}]")
    if "gripper" in results:
        lo, hi = results["gripper"]
        print("\n Suggested infer.py line ~84:")
        print(f"   self._g_servo_min, self._g_servo_max = {lo:.2f}, {hi:.2f}")
    print("────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
