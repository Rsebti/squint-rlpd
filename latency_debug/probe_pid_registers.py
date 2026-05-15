#!/usr/bin/env python3
"""Read the on-board PID coefficients of each SO101 Feetech STS3215 servo.

LeRobot's Feetech control table exposes them as one-byte registers at:
    P_Coefficient (addr 21)
    D_Coefficient (addr 22)
    I_Coefficient (addr 23)

These are the gains the servo's MCU uses in its ~1 kHz position-control
loop. Reading them confirms (a) which firmware preset is loaded and
(b) whether all six servos share gains.
"""
from __future__ import annotations

import sys

from probe_step_response import ROBOT_PORT, build_robot


# Other registers worth showing for context (read-only is fine).
EXTRA_REGS = ["Torque_Enable", "Lock", "Goal_Position", "Present_Position"]


def main():
    print(f"Connecting to {ROBOT_PORT}...")
    robot = build_robot(ROBOT_PORT)
    robot.connect()
    try:
        p = robot.bus.sync_read("P_Coefficient")
        i = robot.bus.sync_read("I_Coefficient")
        d = robot.bus.sync_read("D_Coefficient")
        extras = {name: robot.bus.sync_read(name) for name in EXTRA_REGS}
    finally:
        robot.disconnect()

    motors = list(p.keys())
    print()
    print(f"{'motor':<15} {'P':>4} {'I':>4} {'D':>4}   "
          f"{'torque':>7} {'lock':>5} {'goal':>6} {'present':>8}")
    print("-" * 72)
    for m in motors:
        print(f"{m:<15} {p[m]:>4} {i[m]:>4} {d[m]:>4}   "
              f"{extras['Torque_Enable'][m]:>7} "
              f"{extras['Lock'][m]:>5} "
              f"{extras['Goal_Position'][m]:>6} "
              f"{extras['Present_Position'][m]:>8.2f}")

    # Diagnostic: are gains uniform across motors?
    if len(set(p.values())) == 1 and len(set(i.values())) == 1 \
            and len(set(d.values())) == 1:
        print()
        print("All six servos share the same P/I/D firmware preset: "
              f"P={list(p.values())[0]}, I={list(i.values())[0]}, "
              f"D={list(d.values())[0]} (uint8, scale 0..255).")


if __name__ == "__main__":
    sys.exit(main())
