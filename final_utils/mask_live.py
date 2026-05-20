"""Live RAW | MASKED viewer for the deploy mask (table + colour-selective).

Shows the wrist camera and the exact masked frame the policy is fed, side by
side, in real time — table background greyed to the table mean, and saturated
non-goal objects (bowl, other cubes) greyed too, keeping the table + gripper +
the goal-colour cube. Uses the same mask_background_to_table() as the deploy and
auto-loads hue_calib.json.

    python -m final_utils.mask_live --goal_color 0          # default /dev/video0
    python -m final_utils.mask_live --goal_color 2 --camera_index 1

Keys (in the window):
    0–5  switch goal colour       c  toggle harsh keep-only mask
    t    toggle table mask        [ ]  shrink / grow border cover
    q / Esc  quit
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux as il
from infer_linux import Cv2Camera

COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]

G_SIM_MIN, G_SIM_MAX = -10.0, 120.0
G_SRV_MIN, G_SRV_MAX = -60.13, 66.73


def _srv_to_sim(s):
    return (s - G_SRV_MIN) / (G_SRV_MAX - G_SRV_MIN) * (G_SIM_MAX - G_SIM_MIN) + G_SIM_MIN


def _connect_robot(port):
    """Connect to the SO101 and free the arm so it can be hand-moved while the
    bowl silhouette tracks the live camera pose. Returns (robot, bus, keys)."""
    from infer_linux import create_real_robot
    from lerobot.motors.motors_bus import MotorNormMode
    il.ROBOT_PORT = port
    robot = create_real_robot()
    robot.connect()
    bus = robot.bus
    bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
    keys = list(bus.sync_read("Present_Position").keys())     # bus order == FK joint order
    bus.disable_torque([k for k in keys if k != "gripper"])
    return robot, bus, keys


def _read_qpos(bus, keys):
    d = bus.sync_read("Present_Position")
    deg = [_srv_to_sim(d[k]) if k == "gripper" else d[k] for k in keys]
    return np.deg2rad(np.array(deg, dtype=np.float64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--camera_index", type=int, default=il.CAMERA_INDEX)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--display_width", type=int, default=480, help="per-pane display width")
    p.add_argument("--bowl_xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="bowl centre in robot base frame (m); enables the geometric bowl mask "
                        "(connects the robot to track the camera pose, arm goes limp to hand-move)")
    p.add_argument("--robot_port", type=str, default=il.ROBOT_PORT)
    args = p.parse_args()

    il.TABLE_MASK_ENABLED = True
    il.COLOR_DISTRACTOR_MASK = True
    il.load_hue_calib()
    goal = args.goal_color

    robot = bus = keys = None
    bowl_on = args.bowl_xy is not None
    if bowl_on:
        import bowl_mask as bm
        robot, bus, keys = _connect_robot(args.robot_port)
        print(f"Robot connected, arm FREE — hand-move it. Bowl xy = {tuple(args.bowl_xy)} (base frame).")

    cam = Cv2Camera(index=args.camera_index, width=args.width, height=args.height, fps=args.fps)
    print(f"Opened /dev/video{args.camera_index} at {args.width}x{args.height}@{args.fps}. Press q/Esc to quit.")

    dw = args.display_width
    ema_fps, prev_t = None, None

    def label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    try:
        while True:
            t0 = time.perf_counter()
            rgb = np.asarray(cam.async_read())

            det_w = il.TABLE_DETECT_W
            det_h = int(round(det_w * rgb.shape[0] / rgb.shape[1]))
            det = cv2.resize(rgb, (det_w, det_h), interpolation=cv2.INTER_AREA)
            last_det = det
            masked = il.mask_background_to_table(det.copy(), goal) if il.TABLE_MASK_ENABLED else det.copy()

            bmask = None
            if bowl_on:
                try:
                    q = _read_qpos(bus, keys)
                    bmask = bm.bowl_mask(q, args.bowl_xy, (det_h, det_w))
                    if bmask.any():                                  # grey the bowl from neighbours
                        masked = cv2.inpaint(masked, bmask, 4, cv2.INPAINT_TELEA)
                except Exception as e:                               # don't kill the viewer on a read hiccup
                    print(f"[bowl] {e}")
            last_masked = masked

            dh = int(round(dw * rgb.shape[0] / rgb.shape[1]))
            raw_disp = cv2.cvtColor(cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            msk_disp = cv2.cvtColor(cv2.resize(masked, (dw, dh), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            if bmask is not None and bmask.any():                    # outline the projected bowl on RAW
                cnt, _ = cv2.findContours(cv2.resize(bmask, (dw, dh), interpolation=cv2.INTER_NEAREST),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(raw_disp, cnt, -1, (0, 0, 255), 2)

            if prev_t is not None:
                inst = 1.0 / max(1e-6, t0 - prev_t)
                ema_fps = inst if ema_fps is None else 0.8 * ema_fps + 0.2 * inst
            prev_t = t0

            label(raw_disp, f"RAW  {ema_fps:4.1f} fps" if ema_fps else "RAW")
            label(msk_disp, f"MASKED goal={COLOR_NAMES[goal]} "
                            f"{'harsh:ON' if il.COLOR_DISTRACTOR_MASK else 'harsh:OFF'} "
                            f"{'table:ON' if il.TABLE_MASK_ENABLED else 'table:OFF'} "
                            f"{'AGG:ON' if il.MASK_AGGRESSIVE else 'agg:off'} "
                            f"{'bowl:ON' if bowl_on else ''} "
                            f"grow={il.MASK_GROW_PX}")
            cv2.imshow("deploy mask  (raw | masked)", np.hstack([raw_disp, msk_disp]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("0") <= key <= ord("5"):
                goal = key - ord("0")
            elif key == ord("c"):
                il.COLOR_DISTRACTOR_MASK = not il.COLOR_DISTRACTOR_MASK
            elif key == ord("a"):
                il.MASK_AGGRESSIVE = not il.MASK_AGGRESSIVE
            elif key == ord("b") and robot is not None:
                bowl_on = not bowl_on
            elif key == ord("t"):
                il.TABLE_MASK_ENABLED = not il.TABLE_MASK_ENABLED
            elif key == ord("["):
                il.MASK_GROW_PX = max(0, il.MASK_GROW_PX - 1)
            elif key == ord("]"):
                il.MASK_GROW_PX = il.MASK_GROW_PX + 1
            elif key == ord("s"):
                outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_artifacts")
                os.makedirs(outdir, exist_ok=True)
                cv2.imwrite(os.path.join(outdir, "fail_raw.png"), cv2.cvtColor(last_det, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(outdir, "fail_masked.png"), cv2.cvtColor(last_masked, cv2.COLOR_RGB2BGR))
                print(f"saved debug_artifacts/fail_raw.png + fail_masked.png  (goal={COLOR_NAMES[goal]})")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if robot is not None:
            try:
                bus.enable_torque([k for k in keys if k != "gripper"])
                time.sleep(0.3)
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
