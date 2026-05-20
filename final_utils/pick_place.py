"""Canonical, reusable SO101 pick-and-place.

One call picks the cube of the requested colour and drops it into a bowl:

    from final_utils import pick_and_place
    ok = pick_and_place(goal_color=0, bowl_xy=(0.25, 0.10))   # True if placed

Pipeline (single episode):
    PICK  — vision RL policy + the FK-gated hardcoded grasp (frozen; solved
            2026-05-20). approach → gate (descent-stall / height above the
            reach-calibrated table) → nudge back+down → close → verify by
            gripper stall angle → hold → lift. Retreat+retry on a miss.
    PLACE — IK the (closed) gripper to the bowl centre at a fixed height, wait,
            then open to drop the cube. Success only once the cube is released.

The low-level infra (camera, robot driver, CNN, FK/IK, the grasp constants and
the grasp state machine's building blocks) is imported from infer_linux.py and
so101_fk.py so there is a single source of truth; this module only adds the
orchestration + the place phase.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux as il
from infer_linux import (
    create_real_robot, RealRobotAgent, CNNEncoder, Actor,
    derive_arch_from_ckpt, preprocess_image, build_state, back_nudge_joint_target,
    init_viz, log_step, REST_QPOS, DELTA_CAP, JOINT_LOWER, JOINT_UPPER, CONTROL_HZ,
)
from so101_fk import tcp_pos, nudge_arm_joints

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_place_policy.pt")


def _load_policy(checkpoint, device):
    """Load encoder+actor and push the architecture into infer_linux's globals
    so preprocess_image / the model classes build to the checkpoint's widths."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = derive_arch_from_ckpt(ckpt)
    n_state = arch["n_state"]
    if n_state not in (18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state} (expected 18 or 21)")
    il.IMAGE_H, il.IMAGE_W = arch["image_h"], arch["image_w"]
    il.CNN_FLATTEN_DIM = arch["cnn_flatten_dim"]
    il.RGB_PROJ_DIM = arch["rgb_proj_dim"]
    encoder = CNNEncoder(layers=arch["layers"]).to(device).eval()
    actor = Actor(n_state=n_state).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded policy (step {ckpt.get('global_step', '?')}): input={il.IMAGE_H}×{il.IMAGE_W}, "
          f"n_state={n_state}")
    return encoder, actor, n_state == 21


def _load_table_calib():
    """Mirror infer_linux's table-plane calibration load."""
    if il.TABLE_Z_CALIB_PATH.exists():
        import json
        c = json.loads(il.TABLE_Z_CALIB_PATH.read_text())
        il.TABLE_Z_A, il.TABLE_Z_B = float(c["a"]), float(c["b"])
        print(f"Table-z calib: z_table(r) = {il.TABLE_Z_A:.4f}·r + {il.TABLE_Z_B:.4f}")
    else:
        print("No table_z_calib.json — flat z=0 table assumption.")


def pick_and_place(
    goal_color,
    bowl_xy,
    action_scale=0.45,
    episode_steps=5000,
    checkpoint=DEFAULT_CHECKPOINT,
    place_z=0.10,
    place_open_wait_s=0.5,
    place_speed=0.30,
    place_xy_tol=0.015,
    robot_port=None,
    camera_index=None,
    viz=True,
    table_mask=True,
    distractor_mask=True,
):
    """Pick the cube of `goal_color` and drop it into the bowl at `bowl_xy`.

    goal_color: 0 red 1 blue 2 green 3 yellow 4 purple 5 orange.
    bowl_xy:    (x, y) of the bowl centre in the robot base frame (metres).
    place_z:    fixed TCP height above the calibrated table when dropping (m).
    place_open_wait_s: hold over the bowl this long before opening (drop).
    place_speed: Cartesian speed (m/s) the gripper travels to the bowl.

    Returns True iff the cube was grasped, carried to the bowl, and released.
    """
    bowl_xy = np.asarray(bowl_xy, dtype=np.float64).flatten()
    if robot_port is not None:
        il.ROBOT_PORT = robot_port
    if camera_index is not None:
        il.CAMERA_INDEX = camera_index
    il.TABLE_MASK_ENABLED = table_mask
    il.COLOR_DISTRACTOR_MASK = distractor_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_table_calib()
    il.load_hue_calib()                                            # measured cube hues, if calibrated
    encoder, actor, use_bowl_xyz = _load_policy(checkpoint, device)
    viz_on = init_viz() if viz else False

    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)

    # Place target: bowl centre at a fixed height above the calibrated table.
    bowl_r = float(np.hypot(bowl_xy[0], bowl_xy[1]))
    place_target = np.array([bowl_xy[0], bowl_xy[1],
                             il.TABLE_Z_A * bowl_r + il.TABLE_Z_B + place_z])
    policy_bowl_xyz = [float(bowl_xy[0]), float(bowl_xy[1]), 0.0]   # state z=0 for the policy
    max_place_step = place_speed / CONTROL_HZ                       # Cartesian step per tick

    grasp_close_rad = float(np.deg2rad(il.GRASP_CLOSE_DEG))
    grasp_open_rad = float(np.deg2rad(120.0))                       # full open = drop
    steps = lambda s: max(1, int(round(s * CONTROL_HZ)))

    success = False
    try:
        print(f"\n── pick goal_color={goal_color} → place at bowl xy={bowl_xy} (z={place_z*100:.0f} cm) ──")
        agent.reset(REST_QPOS)
        target_qpos = agent.get_qpos().cpu().numpy().flatten()

        phase, ctr = "approach", 0
        retries = 0
        retreat_target = None                                      # IK back-off target on a miss
        min_above, stall_ctr = float("inf"), 0
        result = None                                              # "success" | "failed"

        for step in range(episode_steps):
            t0 = time.perf_counter()

            qpos = agent.get_qpos().cpu().numpy().flatten()
            agent.capture_sensor_data()
            rgb = agent.get_sensor_data()["base_camera"]["rgb"]

            il.set_mask_aggressive(qpos)             # aggressive mask when tip is close to the table
            obs_rgb = preprocess_image(rgb, goal_color).to(device)
            obs_state = build_state(qpos, target_qpos, goal_color,
                                    bowl_xyz=policy_bowl_xyz if use_bowl_xyz else None).to(device)
            with torch.no_grad():
                raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()
            action = np.clip(raw_action * action_scale, -1.0, 1.0)

            tcp_xyz = tcp_pos(qpos)
            tcp_r = float(np.hypot(tcp_xyz[0], tcp_xyz[1]))
            z_table = il.TABLE_Z_A * tcp_r + il.TABLE_Z_B
            tcp_above = float(tcp_xyz[2]) - z_table
            gate_z_eff = il.GRASP_GATE_Z + il.GRASP_GATE_Z_SLOPE * tcp_r

            # ── PICK: FK-gated hardcoded grasp (mirrors infer_linux) ──────────
            if phase == "approach":
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                if tcp_above < min_above - il.GRASP_STALL_EPS:
                    min_above, stall_ctr = tcp_above, 0
                else:
                    stall_ctr += 1
                stalled = (il.GRASP_GATE_STALL and min_above <= il.GRASP_ENGAGE_Z
                           and stall_ctr >= steps(il.GRASP_STALL_S))
                if tcp_above <= gate_z_eff or stalled:
                    phase, ctr = "wait", 0
                    why = "stalled" if (stalled and tcp_above > gate_z_eff) else "height"
                    print(f"  [grasp] gate ({why}): {tcp_above*100:.1f} cm above table @ r={tcp_r*100:.0f} cm")
            elif phase == "wait":
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                ctr += 1
                if ctr >= steps(il.GRASP_WAIT_S):
                    target_qpos, info = back_nudge_joint_target(
                        qpos, target_qpos, il.GRASP_NUDGE_M, z_table + il.GRASP_NUDGE_Z)
                    phase, ctr = "nudge", 0
                    print(f"  [grasp] nudge back {info}")
            elif phase == "nudge":
                ctr += 1
                if ctr >= steps(il.GRASP_NUDGE_SETTLE_S):
                    phase, ctr = "close", 0
                    print("  [grasp] closing")
            elif phase == "close":
                target_qpos[5] = grasp_close_rad
                ctr += 1
                if ctr >= steps(il.GRASP_CLOSE_S):
                    grip_deg = float(np.rad2deg(qpos[5]))
                    if grip_deg > il.GRASP_EMPTY_BELOW_DEG:
                        phase, ctr = "hold", 0
                        print(f"  [grasp] GRASPED ({grip_deg:.1f}°) → hold {il.GRASP_HOLD_S:.1f}s")
                    elif retries < il.GRASP_MAX_RETRIES:
                        retries += 1
                        cur = tcp_pos(qpos)
                        rad = float(np.hypot(cur[0], cur[1]))
                        back = (-cur[:2] / rad * il.GRASP_RETREAT_BACK_M) if rad > 1e-6 else np.zeros(2)
                        retreat_target = cur + np.array([back[0], back[1], il.GRASP_RETREAT_UP_M])
                        phase, ctr = "retreat", 0
                        print(f"  [grasp] empty ({grip_deg:.1f}°); back off "
                              f"+{il.GRASP_RETREAT_UP_M*100:.0f}cm up/{il.GRASP_RETREAT_BACK_M*100:.0f}cm back, "
                              f"retry {retries}/{il.GRASP_MAX_RETRIES}")
                    else:
                        result = "failed"
                        print(f"  [grasp] FAILED after {il.GRASP_MAX_RETRIES} retries")
            elif phase == "retreat":
                # IK up + back toward the base (gripper open) to view the cube, then rerun policy.
                vec = retreat_target - tcp_pos(qpos)
                if float(np.linalg.norm(vec)) <= 0.01:
                    min_above, stall_ctr = float("inf"), 0
                    phase = "approach"
                    print("  [grasp] backed off → rerunning policy")
                else:
                    step_vec = vec * min(1.0, (il.GRASP_RETREAT_SPEED / CONTROL_HZ) / float(np.linalg.norm(vec)))
                    dq = nudge_arm_joints(qpos, step_vec)
                    target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                    target_qpos[5] = grasp_open_rad
            elif phase == "hold":
                target_qpos[5] = grasp_close_rad
                ctr += 1
                if ctr >= steps(il.GRASP_HOLD_S):
                    phase, ctr = "lift", 0
                    print(f"  [grasp] lifting {il.GRASP_LIFT_M*100:.0f} cm")
            elif phase == "lift":
                dz = il.GRASP_LIFT_M / steps(il.GRASP_LIFT_S)
                dq = nudge_arm_joints(qpos, np.array([0.0, 0.0, dz]))
                target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                target_qpos[5] = grasp_close_rad
                ctr += 1
                if ctr >= steps(il.GRASP_LIFT_S):
                    phase, ctr = "to_bowl", 0
                    print(f"  [place] carrying to bowl xy={bowl_xy} @ z={place_z*100:.0f} cm")
            # ── PLACE: IK to the bowl, wait, then open ────────────────────────
            elif phase == "to_bowl":
                vec = place_target - tcp_xyz
                dist = float(np.linalg.norm(vec))
                if dist <= place_xy_tol:
                    phase, ctr = "drop_wait", 0
                    print(f"  [place] over bowl → wait {place_open_wait_s:.1f}s")
                else:
                    step_vec = vec * min(1.0, max_place_step / dist)   # cap to fast speed
                    dq = nudge_arm_joints(qpos, step_vec)
                    target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                    target_qpos[5] = grasp_close_rad
            elif phase == "drop_wait":
                target_qpos[5] = grasp_close_rad                   # hold over the bowl
                ctr += 1
                if ctr >= steps(place_open_wait_s):
                    phase, ctr = "release", 0
                    print("  [place] opening (drop)")
            elif phase == "release":
                target_qpos[5] = grasp_open_rad                    # open → cube drops
                ctr += 1
                if ctr >= steps(0.7):                              # let it open + fall
                    result = "success"

            agent.set_target_qpos(torch.from_numpy(target_qpos.copy()))

            if viz_on:
                log_step(step=step,
                         raw_rgb=rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0]),
                         policy_rgb=obs_rgb[0].cpu().numpy(),
                         qpos=qpos, target_qpos=target_qpos, action_raw=raw_action)

            if step % 30 == 0:
                print(f"  step {step:4d}  tcp=({tcp_xyz[0]:+.3f},{tcp_xyz[1]:+.3f})  "
                      f"r={tcp_r*100:4.0f}cm  above_table={tcp_above*100:5.1f}cm  phase={phase}")

            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            if result is not None:
                break

        success = result == "success"
        print(f"\n{'PLACED — success' if success else 'did NOT finish (no place)'} after {step + 1} steps.")
    finally:
        for c in agent.cameras.values():
            try:
                c.close()
            except Exception:
                pass
        agent.reset(REST_QPOS)
        robot.disconnect()

    return success


def main():
    p = argparse.ArgumentParser(description="SO101 pick-and-place: pick the colour cube, drop it in the bowl.")
    p.add_argument("--goal_color", type=int, required=True, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--bowl_xy", type=float, nargs=2, required=True, metavar=("X", "Y"),
                   help="bowl centre xy in the robot base frame (m)")
    p.add_argument("--action_scale", type=float, default=0.45)
    p.add_argument("--episode_steps", type=int, default=5000)
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--place_z", type=float, default=0.10, help="drop height above the table (m)")
    p.add_argument("--place_open_wait_s", type=float, default=0.5, help="hold over bowl before opening (s)")
    p.add_argument("--place_speed", type=float, default=0.30, help="travel speed to the bowl (m/s)")
    p.add_argument("--robot_port", type=str, default=None)
    p.add_argument("--camera_index", type=int, default=None)
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--table_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="grey the background behind the table (default ON)")
    p.add_argument("--distractor_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="grey the non-goal cubes (default ON)")
    args = p.parse_args()

    ok = pick_and_place(
        goal_color=args.goal_color, bowl_xy=tuple(args.bowl_xy),
        action_scale=args.action_scale, episode_steps=args.episode_steps,
        checkpoint=args.checkpoint, place_z=args.place_z,
        place_open_wait_s=args.place_open_wait_s, place_speed=args.place_speed,
        robot_port=args.robot_port, camera_index=args.camera_index, viz=args.viz,
        table_mask=args.table_mask, distractor_mask=args.distractor_mask,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
