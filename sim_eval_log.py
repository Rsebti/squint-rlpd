"""Headless: run 1 sim episode and log everything needed to compare with real-arm trace.

Saves to /tmp/sim_eval.npz with:
    qpos          (T, 6)   clean joint positions (sim radians)
    target_qpos   (T, 6)   controller running target (sim radians)
    action        (T, 6)   policy's tanh action (clipped to [-1,1])
    cube_xyz_init (3,)     starting cube position in world frame
    cube_xyz_final(3,)     final cube position in world frame
    bin_xyz       (3,)     bowl position
    goal_color    int      goal color index
    wrist_rgb     (T,H,W,3) policy-input camera at 128px (matches infer's intermediate)

Usage:
    python sim_eval_log.py --checkpoint runs/placecube_bowl_eval1_run2/ckpt.pt --seed 0
"""
import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import cv2

SIM_CONTROL_HZ = 10  # must match training's control_freq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import CNNEncoder, Actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101PlaceCube-v1")
    ap.add_argument("--n_distractors", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--no_dr", action="store_true", default=True)
    ap.add_argument("--out", default="/tmp/sim_eval.npz")
    ap.add_argument("--video", default="/tmp/sim_eval.mp4", help="path to write the third-person mp4 (set empty to skip)")
    ap.add_argument("--render_size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--spawn_box_size", type=float, default=0.20,
                    help="full side length of the cube spawn box in metres (eval default: 0.20 = 20cm; training uses 0.25)")
    args = ap.parse_args()

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=not args.no_dr,
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=640, height=480),
        human_render_camera_configs=dict(
            shader_pack="default", width=args.render_size, height=args.render_size
        ),
        n_distractors=args.n_distractors,
        use_real_bowl=True,
        spawn_box_half_size=args.spawn_box_size / 2,
    )
    env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(32, 42, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval(); actor.eval()
    print(f"loaded {args.checkpoint} @ step {ckpt.get('global_step')}")

    base = env.unwrapped
    obs, _ = env.reset(seed=args.seed)

    cube_xyz_init = base.item.pose.p.detach().cpu().numpy().flatten()
    bin_xyz = base.bin.pose.p.detach().cpu().numpy().flatten()
    goal_color = int(base.goal_color_idx[0].item()) if hasattr(base, "goal_color_idx") else -1
    color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
    gname = color_names[goal_color] if 0 <= goal_color < 6 else "?"

    print(f"goal color: {goal_color} ({gname})")
    print(f"cube xyz (start): [{cube_xyz_init[0]:+.4f}, {cube_xyz_init[1]:+.4f}, {cube_xyz_init[2]:+.4f}]")
    print(f"bowl xyz:        [{bin_xyz[0]:+.4f}, {bin_xyz[1]:+.4f}, {bin_xyz[2]:+.4f}]")

    log_qpos, log_target, log_action, log_rgb, log_scene = [], [], [], [], []
    log_t_sim, log_t_wall = [], []
    succeeded = False
    t_wall0 = time.perf_counter()
    for step in range(args.max_steps):
        # Third-person view of the whole scene — for finding/replicating cube position.
        scene = env.render()
        if torch.is_tensor(scene):
            scene = scene.detach().cpu().numpy()
        scene = np.asarray(scene)
        if scene.ndim == 4:
            scene = scene[0]
        log_scene.append(scene.astype(np.uint8))

        t_sim = step / SIM_CONTROL_HZ          # nominal sim time advanced so far (s)
        t_wall = time.perf_counter() - t_wall0  # wall clock elapsed since episode start (s)
        log_t_sim.append(t_sim)
        log_t_wall.append(t_wall)
        # CLEAN qpos straight from the sim robot (no DR noise injection):
        q = base.agent.robot.get_qpos().detach().cpu().numpy().flatten()
        # controller running target:
        ctrl = base.agent.controller
        tgt = getattr(ctrl, "_target_qpos", None)
        if tgt is None:
            tgt = ctrl.get_state().get("target_qpos") if hasattr(ctrl, "get_state") else None
        tgt = tgt.detach().cpu().numpy().flatten() if torch.is_tensor(tgt) else np.full(6, np.nan)

        rgb_now = obs["rgb"]
        state_now = obs["state"]
        if not torch.is_tensor(rgb_now):
            rgb_now = torch.from_numpy(rgb_now)
        if not torch.is_tensor(state_now):
            state_now = torch.from_numpy(state_now)
        rgb_t = rgb_now.permute(0, 3, 1, 2).float()
        rgb_small = F.interpolate(rgb_t, size=(32, 42), mode="area").permute(0, 2, 3, 1).to(torch.uint8)

        with torch.no_grad():
            feats = encoder(rgb_small.to(device))
            mean = actor.forward(feats, state_now.float().to(device))
            action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)
        action_np = action.detach().cpu().numpy().astype(np.float32).flatten()

        log_qpos.append(q)
        log_target.append(tgt)
        log_action.append(action_np)
        log_rgb.append(rgb_now.detach().cpu().numpy()[0].astype(np.uint8))

        obs, rew, term, trunc, info = env.step(action.detach().cpu().numpy().astype(np.float32))
        if "success" in info:
            if float(torch.as_tensor(info["success"]).flatten()[0]) > 0.5:
                succeeded = True
        if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
            break

    cube_xyz_final = base.item.pose.p.detach().cpu().numpy().flatten()
    print(f"cube xyz (end):   [{cube_xyz_final[0]:+.4f}, {cube_xyz_final[1]:+.4f}, {cube_xyz_final[2]:+.4f}]")
    print(f"steps taken: {len(log_qpos)}, success={succeeded}")

    np.savez(
        args.out,
        qpos=np.stack(log_qpos),
        target_qpos=np.stack(log_target),
        action=np.stack(log_action),
        cube_xyz_init=cube_xyz_init,
        cube_xyz_final=cube_xyz_final,
        bin_xyz=bin_xyz,
        goal_color=np.array([goal_color]),
        wrist_rgb=np.stack(log_rgb),
        t_sim=np.asarray(log_t_sim, dtype=np.float32),
        t_wall=np.asarray(log_t_wall, dtype=np.float32),
        joint_names=np.array(["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]),
        succeeded=np.array([succeeded]),
    )
    print(f"saved {args.out}")
    total_sim = log_t_sim[-1] if log_t_sim else 0.0
    total_wall = log_t_wall[-1] if log_t_wall else 0.0
    speedup = total_sim / total_wall if total_wall > 0 else float("nan")
    print(f"sim time elapsed: {total_sim:.2f}s   wall time: {total_wall:.2f}s   sim/wall = {speedup:.2f}x")

    if args.video and len(log_scene) > 0:
        H, W = log_scene[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.video, fourcc, args.fps, (W, H))
        for i, f in enumerate(log_scene):
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            label = f"sim {log_t_sim[i]:5.2f}s   wall {log_t_wall[i]:5.2f}s   step {i}"
            cv2.rectangle(bgr, (0, 0), (W, 28), (0, 0, 0), -1)
            cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            vw.write(bgr)
        vw.release()
        print(f"saved {args.video}")
    env.close()


if __name__ == "__main__":
    main()
