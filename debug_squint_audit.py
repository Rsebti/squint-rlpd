"""ManiSkill3 / SAPIEN side audit for the Squint SO-101 Place env.

Mirrors the output of the Isaac side script. Run on a Linux machine where
ManiSkill3 works, then send stdout to the Isaac side for side-by-side
comparison.

Audits, in order:

  1. Sim config (sim_freq, control_freq, control_mode, action space)
  2. Robot world pose + orientation
  3. Joint qpos / qvel / limits at home keyframe
  4. Body world poses (pos + quat) for every robot link
  5. Local axes of every body in world (exact rotations)
  6. Cube + bin world poses
  7. Wrist camera world pose + computed optical-axis ground projection
  8. Actuator controller gains
  9. Joint-axis directions in world (probed by commanding +0.1 on each
     joint individually and observing displacement)
 10. State vector composition the policy actually sees (after
     FlattenRGBDObservationWrapper)
 11. Controller target_qpos probe (verifies action -> target scaling)
 12. RGB image stats (raw 128x128 and downsampled 16x16) + .npy dump
 13. Optional: policy inference on the initial observation -- canonical
     comparison point. Pass --checkpoint=runs/<name>/ckpt.pt to enable.

Launch
------
    cd squint
    python debug_squint_audit.py                          # audit only
    python debug_squint_audit.py --checkpoint=runs/placecube_b8ada9_run1/ckpt.pt --apply_overlay
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs  # noqa: F401  -- registers SO101*-v1


# ---------------------------------------------------------------------------
# Quaternion helpers (Hamilton scalar-first, w, x, y, z)
# ---------------------------------------------------------------------------

def _quat_rotate(q, v):
    qw, qx, qy, qz = q
    vx, vy, vz = v
    rx = (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qw * qz) * vy + 2 * (qx * qz + qw * qy) * vz
    ry = 2 * (qx * qy + qw * qz) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy + 2 * (qy * qz - qw * qx) * vz
    rz = 2 * (qx * qz - qw * qy) * vx + 2 * (qy * qz + qw * qx) * vy + (1 - 2 * (qx * qx + qy * qy)) * vz
    return (rx, ry, rz)


def _fmt_vec(v, prec=4):
    return f"({v[0]:+.{prec}f}, {v[1]:+.{prec}f}, {v[2]:+.{prec}f})"


def _fmt_quat(q, prec=4):
    return f"(w={q[0]:+.{prec}f}, x={q[1]:+.{prec}f}, y={q[2]:+.{prec}f}, z={q[3]:+.{prec}f})"


def _to_tuple(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return tuple(float(v) for v in np.asarray(x).reshape(-1))


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


HR = "=" * 90
SUB = "-" * 90


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full ManiSkill scene audit at SO-101 home pose")
    parser.add_argument("--env_id", type=str, default="SO101PlaceCube-v1")
    parser.add_argument("--control_mode", type=str, default="pd_joint_target_delta_pos")
    parser.add_argument("--obs_mode", type=str, default="rgb+segmentation",
                        help="Use rgb+segmentation to match training; 'rgb+state' for richer state debugging.")
    parser.add_argument("--settle_steps", type=int, default=5)
    parser.add_argument("--apply_overlay", action="store_true", default=False,
                        help="If set, applies the #B8ADA9 greenscreen overlay (matches the b8ada9 checkpoint training).")
    parser.add_argument("--probe_joint_axes", action="store_true", default=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a ckpt.pt. If set, loads the policy and runs inference on the initial obs.")
    parser.add_argument("--image_dump_dir", type=str, default="/tmp",
                        help="Where to drop the .npy / .png image dumps for cross-machine comparison.")
    args = parser.parse_args()

    # Same env_kwargs for the SAPIEN audit, the state probe, and the policy rollout.
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="sensors",
        max_episode_steps=200,
        domain_randomization=False,
        reward_mode="none",
        control_mode=args.control_mode,
        sensor_configs=dict(width=128, height=128),
    )
    if args.apply_overlay:
        env_kwargs["domain_randomization_config"] = {"apply_overlay": True}
    else:
        env_kwargs["domain_randomization_config"] = {"apply_overlay": False}

    env = gym.make(args.env_id, **env_kwargs)
    obs, _ = env.reset(seed=0)
    base_env = env.unwrapped
    agent = base_env.agent
    robot = agent.robot

    # Action shape
    n_act = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 6
    if isinstance(n_act, tuple):
        n_act = n_act[-1]
    zero_action = np.zeros((1, n_act), dtype=np.float32)

    print(f"[audit] Settling physics for {args.settle_steps} steps...")
    for _ in range(args.settle_steps):
        env.step(zero_action)

    print(f"\n{HR}\n  FULL MANISKILL AUDIT  env={args.env_id}  apply_overlay={args.apply_overlay}\n{HR}")

    # ====================== 1. SIM CONFIG ======================
    print(f"\n{SUB}\n  1. SIM CONFIG\n{SUB}")
    sim_cfg = base_env.sim_config
    print(f"  sim_freq       = {sim_cfg.sim_freq}")
    print(f"  control_freq   = {sim_cfg.control_freq}")
    print(f"  control_mode   = {args.control_mode}")
    print(f"  obs_mode       = {args.obs_mode}")
    try:
        gravity = getattr(base_env.scene, "gravity", (0, 0, -9.81))
        print(f"  gravity        = {_to_tuple(gravity)}")
    except Exception:
        pass
    print(f"  action_space   = {env.action_space}")
    try:
        print(f"    high = {env.action_space.high.tolist()}")
        print(f"    low  = {env.action_space.low.tolist()}")
    except Exception:
        pass

    # ====================== 2. ROBOT ======================
    print(f"\n{SUB}\n  2. ROBOT\n{SUB}")

    joint_names = [j.name for j in robot.active_joints]
    n_joints = len(joint_names)
    print(f"\n[joint count] {n_joints}")
    print(f"[joint names] {joint_names}")

    qpos = _to_tuple(robot.get_qpos())
    qvel = _to_tuple(robot.get_qvel())
    qlim = _np(robot.get_qlimits())
    qlim = qlim.reshape(-1, 2) if qlim.shape[-1] == 2 else qlim[0]
    print(f"\n[joints -- qpos / qvel / hard limits]")
    for i, jn in enumerate(joint_names):
        lo, hi = float(qlim[i, 0]), float(qlim[i, 1])
        print(f"  [{i}] {jn:20s}  qpos={qpos[i]:+9.4f} rad ({math.degrees(qpos[i]):+8.2f} deg)  "
              f"qvel={qvel[i]:+9.4f}  hard=[{lo:+6.3f}, {hi:+6.3f}]")

    # ---- Controller info ----
    print(f"\n[controller info]")
    try:
        ctrl = agent.controller
        print(f"  active controller = {type(ctrl).__name__}")
        if hasattr(ctrl, "config"):
            cfg = ctrl.config
            for attr in ("stiffness", "damping", "force_limit", "lower", "upper",
                         "use_delta", "use_target", "normalize_action"):
                if hasattr(cfg, attr):
                    val = getattr(cfg, attr)
                    print(f"    {attr:18s} = {val}")
    except Exception as e:
        print(f"  (failed to introspect controller: {e})")

    # ---- Robot world pose ----
    print(f"\n[robot world pose]")
    try:
        rp = _to_tuple(robot.pose.p)
        rq = _to_tuple(robot.pose.q)
        print(f"  pos_w = {_fmt_vec(rp)}")
        print(f"  quat_w = {_fmt_quat(rq)}   (Hamilton w,x,y,z)")
    except Exception as e:
        print(f"  (failed: {e})")

    # ---- Bodies (links) ----
    print(f"\n[bodies / links -- world pose + local axes in world]")
    for link_name, link in robot.links_map.items():
        try:
            p = _to_tuple(link.pose.p)
            q = _to_tuple(link.pose.q)
        except Exception:
            continue
        xax = _quat_rotate(q, (1, 0, 0))
        yax = _quat_rotate(q, (0, 1, 0))
        zax = _quat_rotate(q, (0, 0, 1))
        print(f"  {link_name:30s}")
        print(f"        pos_w  = {_fmt_vec(p)}")
        print(f"        quat_w = {_fmt_quat(q)}")
        print(f"        local +X -> world {_fmt_vec(xax, prec=3)}    "
              f"local +Y -> world {_fmt_vec(yax, prec=3)}    "
              f"local +Z -> world {_fmt_vec(zax, prec=3)}")

    # ====================== 3. CUBE + BIN ======================
    print(f"\n{SUB}\n  3. CUBE + BIN\n{SUB}")
    for label, attr in [("cube/item", "item"), ("bin", "bin")]:
        if hasattr(base_env, attr):
            actor = getattr(base_env, attr)
            try:
                p = _to_tuple(actor.pose.p)
                q = _to_tuple(actor.pose.q)
                print(f"  {label:18s}  pos_w = {_fmt_vec(p)}    quat_w = {_fmt_quat(q)}")
            except Exception as e:
                print(f"  {label:18s}  (failed: {e})")
    if hasattr(base_env, "item_half_sizes"):
        try:
            print(f"  item_half_sizes      = {_to_tuple(base_env.item_half_sizes)}")
        except Exception:
            pass
    if hasattr(base_env, "bin_dimensions"):
        try:
            print(f"  bin_dimensions       = {_to_tuple(base_env.bin_dimensions)}")
        except Exception:
            pass
    for attr in ("bin_thickness", "spawn_box_pos", "spawn_box_half_size"):
        if hasattr(base_env, attr):
            try:
                print(f"  {attr:20s} = {getattr(base_env, attr)}")
            except Exception:
                pass

    # ====================== 4. CAMERA ======================
    print(f"\n{SUB}\n  4. WRIST CAMERA\n{SUB}")
    try:
        cam_configs = base_env._sensor_configs
        for name, cfg in cam_configs.items():
            print(f"\n  [{name}]")
            for attr in ("width", "height", "fov", "near", "far"):
                if hasattr(cfg, attr):
                    print(f"    {attr:10s} = {getattr(cfg, attr)}")
    except Exception as e:
        print(f"  (sensor_configs failed: {e})")

    if hasattr(base_env, "wrist_camera_mount"):
        try:
            mp = base_env.wrist_camera_mount.pose
            p = _to_tuple(mp.p)
            q = _to_tuple(mp.q)
            print(f"\n  [wrist_camera_mount -- world pose]")
            print(f"    pos_w  = {_fmt_vec(p)}")
            print(f"    quat_w = {_fmt_quat(q)}")
            for sign, lbl in [((1, 0, 0), "+X"), ((-1, 0, 0), "-X"),
                              ((0, 1, 0), "+Y"), ((0, -1, 0), "-Y"),
                              ((0, 0, 1), "+Z"), ((0, 0, -1), "-Z")]:
                ax = _quat_rotate(q, sign)
                print(f"    local {lbl} -> world {_fmt_vec(ax, prec=3)}")
            # Ground projection for plausible optical-axis conventions
            print(f"    optical-axis ground projection (z=0):")
            cam_z = p[2]
            for name, opt in [("local -Z (OpenGL/SAPIEN render cam)", _quat_rotate(q, (0, 0, -1))),
                              ("local +X (some SAPIEN camera frames)", _quat_rotate(q, (1, 0, 0)))]:
                if opt[2] < -1e-6:
                    t = (0.0 - cam_z) / opt[2]
                    hit = (p[0] + t * opt[0], p[1] + t * opt[1], 0.0)
                    print(f"      {name}: fwd={_fmt_vec(opt, prec=3)}  hit={_fmt_vec(hit, prec=3)}")
                else:
                    print(f"      {name}: fwd={_fmt_vec(opt, prec=3)}  (does not point down)")
        except Exception as e:
            print(f"  (failed: {e})")

    # ====================== 5. KEY DISTANCES ======================
    print(f"\n{SUB}\n  5. KEY DISTANCES\n{SUB}")
    try:
        gp = _to_tuple(robot.links_map["gripper_link"].pose.p)
        for label, attr in [("item", "item"), ("bin", "bin")]:
            if hasattr(base_env, attr):
                tp = _to_tuple(getattr(base_env, attr).pose.p)
                dxyz = (tp[0] - gp[0], tp[1] - gp[1], tp[2] - gp[2])
                d = math.sqrt(sum(c * c for c in dxyz))
                print(f"  gripper_link -> {label:5s} distance = {d:.4f} m  (dxyz = {_fmt_vec(dxyz, prec=3)})")
    except Exception as e:
        print(f"  (failed: {e})")

    # ====================== 6. JOINT AXIS PROBE ======================
    if args.probe_joint_axes:
        print(f"\n{SUB}\n  6. JOINT AXIS PROBE\n"
              f"  +0.1 normalized action on ONE joint at a time x 10 control-steps.\n"
              f"  Reveals which world-direction each joint moves the end-effector.\n{SUB}")
        try:
            tip_name = None
            for cand in ("jaw", "moving_jaw_so101_v1_link", "gripper_link", "finger1_tip"):
                if cand in robot.links_map:
                    tip_name = cand
                    break
            if tip_name is None:
                print("  No tip link found in robot.links_map")
            else:
                print(f"  Tip link = {tip_name}")
                env.reset(seed=0)
                for i, jn in enumerate(joint_names):
                    env.reset(seed=0)
                    act = np.zeros((1, n_act), dtype=np.float32)
                    act[0, i] = 0.1
                    p0 = _to_tuple(robot.links_map[tip_name].pose.p)
                    for _ in range(10):
                        env.step(act)
                    p1 = _to_tuple(robot.links_map[tip_name].pose.p)
                    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                    mag = math.sqrt(sum(c * c for c in d))
                    print(f"  joint[{i}] {jn:20s}  action=+0.1 x 10  delta_tip = {_fmt_vec(d, prec=4)}  |delta|={mag:.4f}")
        except Exception as e:
            print(f"  (probe failed: {e})")

    # ====================== 7. CONTROLLER TARGET PROBE ======================
    # Verifies the action -> delta_qpos scaling that the controller actually applies.
    # This is one of the top two cross-sim mismatches.
    print(f"\n{SUB}\n  7. CONTROLLER TARGET PROBE (action -> target_qpos delta)\n{SUB}")
    try:
        env.reset(seed=0)
        ctrl = agent.controller
        # Snapshot of initial target. SO-101's pd_joint_target_delta_pos initializes
        # target to current qpos on reset, per ManiSkill convention.
        def _target_qpos():
            for cand in ("_target_qpos", "target_qpos"):
                if hasattr(ctrl, cand):
                    v = getattr(ctrl, cand)
                    if v is not None:
                        return _to_tuple(v)
            st = ctrl.get_state()
            return _to_tuple(st) if hasattr(st, "__len__") else None

        t_init = _target_qpos()
        q_init = _to_tuple(robot.get_qpos())
        print(f"  qpos        @ reset = {q_init}")
        print(f"  target_qpos @ reset = {t_init}")
        if t_init and q_init:
            diff = tuple(t - q for t, q in zip(t_init, q_init))
            print(f"  target - qpos       = {diff}   (should be ~0 right after reset)")
        # Probe one joint at a time with +0.1 normalized, single control step
        print(f"\n  Per-joint single-step probe (+0.1 normalized, 1 control step):")
        print(f"    expected scaling: arm joints +-0.1 rad/step, gripper +-0.2 rad/step")
        for i, jn in enumerate(joint_names):
            env.reset(seed=0)
            t0 = _target_qpos()
            act = np.zeros((1, n_act), dtype=np.float32)
            act[0, i] = 0.1
            env.step(act)
            t1 = _target_qpos()
            if t0 is None or t1 is None:
                continue
            delta = tuple(t1[k] - t0[k] for k in range(len(t0)))
            print(f"    joint[{i}] {jn:20s}  delta_target = {[f'{d:+.4f}' for d in delta]}")
    except Exception as e:
        print(f"  (probe failed: {e})")

    # ====================== 8. STATE VECTOR AUDIT ======================
    # What FlattenRGBDObservationWrapper produces for obs['state'].
    print(f"\n{SUB}\n  8. STATE VECTOR (post FlattenRGBDObservationWrapper)\n{SUB}")
    try:
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        env.close()
        env2 = gym.make(args.env_id, **env_kwargs)
        env2 = FlattenRGBDObservationWrapper(env2, rgb=True, depth=False, state=True)
        obs2, _ = env2.reset(seed=0)
        for _ in range(args.settle_steps):
            env2.step(zero_action)
        # Re-fetch after settling
        # (FlattenRGBDObservationWrapper returns the wrapped obs in dict form)
        for k, v in obs2.items():
            if hasattr(v, "shape"):
                print(f"  obs['{k}'].shape = {tuple(v.shape)}, dtype = {v.dtype}")
        state_vec = _np(obs2["state"][0]).flatten() if obs2["state"].ndim > 1 else _np(obs2["state"]).flatten()
        print(f"\n  obs['state'] length = {len(state_vec)}")
        print(f"  values (full):  {[f'{x:+.4f}' for x in state_vec]}")
        if len(state_vec) >= 12:
            print(f"\n  Expected layout: [noisy_qpos(6), controller_target_qpos(6)]")
            print(f"    state[0:6]   (noisy_qpos)    = {[f'{x:+.4f}' for x in state_vec[0:6]]}")
            print(f"    state[6:12]  (target_qpos)   = {[f'{x:+.4f}' for x in state_vec[6:12]]}")
            qpos_now = _to_tuple(env2.unwrapped.agent.robot.get_qpos())
            print(f"    actual qpos (no noise)       = {[f'{x:+.4f}' for x in qpos_now]}")
        # ====================== 9. RGB IMAGE ======================
        print(f"\n{SUB}\n  9. RGB IMAGE STATS / DUMP\n{SUB}")
        rgb_raw = _np(obs2["rgb"][0])  # (128, 128, 3) uint8 (or (128,128,3*N))
        print(f"  raw rgb shape = {rgb_raw.shape}, dtype = {rgb_raw.dtype}")
        print(f"  raw rgb min/mean/max = {rgb_raw.min()} / {rgb_raw.mean():.2f} / {rgb_raw.max()}")
        print(f"  raw rgb mean per channel = R:{rgb_raw[..., 0].mean():.2f}  "
              f"G:{rgb_raw[..., 1].mean():.2f}  B:{rgb_raw[..., 2].mean():.2f}")
        # Save raw + downsampled
        os.makedirs(args.image_dump_dir, exist_ok=True)
        np.save(os.path.join(args.image_dump_dir, "squint_rgb_128.npy"), rgb_raw)
        # Downsample to 16x16 with mode='area' (matches DeployAgent)
        import torch.nn.functional as F
        rgb_t = torch.from_numpy(rgb_raw).permute(2, 0, 1).unsqueeze(0).float()
        rgb_16 = F.interpolate(rgb_t, size=(16, 16), mode='area').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        print(f"  downsampled (16x16, mode='area') min/mean/max = {rgb_16.min()} / {rgb_16.mean():.2f} / {rgb_16.max()}")
        print(f"  downsampled mean per channel = R:{rgb_16[..., 0].mean():.2f}  "
              f"G:{rgb_16[..., 1].mean():.2f}  B:{rgb_16[..., 2].mean():.2f}")
        np.save(os.path.join(args.image_dump_dir, "squint_rgb_16.npy"), rgb_16)
        try:
            import cv2
            cv2.imwrite(os.path.join(args.image_dump_dir, "squint_rgb_128.png"),
                        cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.image_dump_dir, "squint_rgb_16.png"),
                        cv2.cvtColor(rgb_16, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
        print(f"  dumped to {args.image_dump_dir}/squint_rgb_{{128,16}}.{{npy,png}}")

        # ====================== 10. POLICY INFERENCE ======================
        if args.checkpoint:
            print(f"\n{SUB}\n 10. POLICY INFERENCE on initial obs\n{SUB}")
            from train_squint import CNNEncoder, Actor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            print(f"  checkpoint = {args.checkpoint}")
            print(f"  global_step = {ckpt.get('global_step')}")

            n_obs = (16, 16, 3)
            n_state = state_vec.shape[0]
            n_a = env2.unwrapped.single_action_space.shape[0]
            encoder = CNNEncoder(n_obs=n_obs, device=device).to(device)
            actor = Actor(env2, n_obs=1024, n_state=n_state, n_act=n_a, device=device).to(device)
            encoder.load_state_dict(ckpt['encoder'])
            actor.load_state_dict(ckpt['actor'])
            encoder.eval(); actor.eval()
            with torch.no_grad():
                rgb_in = torch.from_numpy(rgb_16).unsqueeze(0).to(device)  # (1, 16, 16, 3) uint8
                state_in = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(device)
                feats = encoder(rgb_in)
                mean = actor.forward(rgb_in if False else feats, state_in)
                action = torch.tanh(mean) * actor.action_scale + actor.action_bias
            print(f"  encoder.repr_dim = {encoder.repr_dim}")
            print(f"  actor.action_scale = {actor.action_scale.cpu().tolist()}")
            print(f"  actor.action_bias  = {actor.action_bias.cpu().tolist()}")
            print(f"  state input (len={state_vec.shape[0]}) = {[f'{x:+.4f}' for x in state_vec]}")
            print(f"  rgb input mean RGB = R:{rgb_16[..., 0].mean():.2f}  "
                  f"G:{rgb_16[..., 1].mean():.2f}  B:{rgb_16[..., 2].mean():.2f}")
            print(f"  feats[:8] = {feats[0, :8].cpu().tolist()}")
            print(f"  ACTION (deterministic, on initial obs) = "
                  f"{[f'{a:+.6f}' for a in action[0].cpu().tolist()]}")
            print(f"\n  >>> The Isaac side should reproduce this exact action vector when fed")
            print(f"  >>> the same checkpoint, same 16x16 rgb (saved to squint_rgb_16.npy),")
            print(f"  >>> and same 12-d state. Differences localize where the mismatch is.")
        env2.close()
    except Exception as e:
        print(f"  (failed: {e})")
        import traceback; traceback.print_exc()

    print(f"\n{HR}\n  END AUDIT\n{HR}\n")


if __name__ == "__main__":
    main()
