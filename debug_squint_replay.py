"""Isaac-side handoff bundle for SO-101 Place.

Runs the priorities requested by the Isaac team (P1, P2, P4, P5, P6 — P3 is
skipped per instruction) and writes everything to a single folder so it can
be zipped and sent.

Usage:
    python debug_squint_replay.py \
        --checkpoint=runs/placecube_flattable_woodcube_run1/ckpt.pt \
        --out_dir=runs/placecube_flattable_woodcube_run1/isaacsim_handoff

Outputs in --out_dir:
    replay_trajs.json         (P1) full-episode action+state dumps
    action_sensitivity.txt    (P2) deterministic policy probes
    physics_params.txt        (P4) sim/cube/gripper physics snapshot
    settle_behavior.txt       (P5) qpos/qvel under zero action
    action_replay.txt         (P6) ground-truth qpos trajectories per joint
    run_console.log           full stdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs  # noqa: F401  - registers SO101*-v1
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import CNNEncoder, Actor


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tuple(x):
    a = _np(x).flatten()
    return tuple(float(v) for v in a)


class Tee:
    """Mirror stdout to a file."""
    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()
    def flush(self):
        self.stdout.flush()
        self.f.flush()
    def close(self):
        self.f.close()


def build_env(args, num_envs=1, seed=0):
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode="sensors",
        max_episode_steps=50,
        domain_randomization=False,
        reward_mode="none",
        control_mode=args.control_mode,
        sensor_configs=dict(width=128, height=128),
    )
    env = gym.make(args.env_id, num_envs=num_envs, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.reset(seed=seed)
    return env


def load_policy(ckpt_path, env, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(16, 16, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval(); actor.eval()
    return encoder, actor, ckpt, n_state, n_act


def _rgb_to_16(rgb_uint8_hw3):
    """Downsample (H,W,3) uint8 to (16,16,3) uint8 using mode='area' (matches DeployAgent)."""
    t = torch.from_numpy(rgb_uint8_hw3).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(16, 16), mode="area")
    return t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# P1 — SUCCESSFUL TRAJECTORY DUMP
# ---------------------------------------------------------------------------
def priority1(args, device, out_dir):
    print("\n=========== P1  SUCCESSFUL TRAJECTORY DUMP ===========")
    env = build_env(args, num_envs=1)
    encoder, actor, ckpt, n_state, n_act = load_policy(args.checkpoint, env, device)
    base = env.unwrapped
    print(f"  ckpt step = {ckpt.get('global_step')}, n_state = {n_state}, n_act = {n_act}")

    trajectories = []
    for ep in range(args.n_replay_episodes):
        obs, _ = env.reset(seed=ep)
        traj = {
            "seed": ep,
            "n_state": n_state,
            "n_act": n_act,
            "cube_init_xyz": _to_tuple(base.item.pose.p)[:3],
            "bin_init_xyz": _to_tuple(base.bin.pose.p)[:3],
            "cube_init_quat_wxyz": _to_tuple(base.item.pose.q)[:4],
            "bin_init_quat_wxyz": _to_tuple(base.bin.pose.q)[:4],
            "robot_init_qpos": _to_tuple(base.agent.robot.get_qpos()),
            "goal_color_idx": int(_np(base.goal_color_idx)[0]) if hasattr(base, "goal_color_idx") else None,
            "distractor_color_idx": (
                int(_np(base.distractor_color_idx)[0])
                if hasattr(base, "distractor_color_idx") and base.distractor is not None else None
            ),
            "distractor_init_xyz": (
                _to_tuple(base.distractor.pose.p)[:3] if base.distractor is not None else None
            ),
            "steps": [],
            "success": False,
            "success_step": None,
        }
        for step in range(50):
            rgb_now = _np(obs["rgb"])
            if rgb_now.ndim == 4:
                rgb_now = rgb_now[0]
            rgb16 = _rgb_to_16(rgb_now)
            state_now = _np(obs["state"]).flatten().astype(np.float32).tolist()
            with torch.no_grad():
                rgb_in = torch.from_numpy(rgb16).unsqueeze(0).to(device)
                state_in = torch.tensor(state_now, dtype=torch.float32, device=device).unsqueeze(0)
                feats = encoder(rgb_in)
                mean = actor.forward(feats, state_in)
                action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)[0].cpu().numpy()
            traj["steps"].append({
                "step": step,
                "state": state_now,
                "rgb16_mean_rgb": [float(rgb16[..., c].mean()) for c in range(3)],
                "action": [float(a) for a in action.tolist()],
                "feats_first8": feats[0, :8].cpu().tolist(),
            })
            obs, _, term, trunc, info = env.step(action.reshape(1, -1).astype(np.float32))
            if "success" in info:
                ok = bool(_np(info["success"]).flatten()[0])
                if ok and not traj["success"]:
                    traj["success"] = True
                    traj["success_step"] = step
            done = bool(_np(term).any()) or bool(_np(trunc).any())
            if done:
                break
        trajectories.append(traj)
        print(f"  ep {ep}: success={traj['success']}  success_step={traj['success_step']}  steps={len(traj['steps'])}")

    out_path = Path(out_dir) / "replay_trajs.json"
    with open(out_path, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"  wrote {out_path}")
    env.close()


# ---------------------------------------------------------------------------
# P2 — ACTION SENSITIVITY TO STATE
# ---------------------------------------------------------------------------
def priority2(args, device, out_dir):
    print("\n=========== P2  ACTION SENSITIVITY TO STATE ===========")
    env = build_env(args, num_envs=1)
    encoder, actor, ckpt, n_state, n_act = load_policy(args.checkpoint, env, device)
    print(f"  n_state = {n_state}  (12-d qpos|target_qpos + 6-d goal_color_one_hot)")

    # Base 12-d states (qpos | target_qpos). The 6 goal_color dims are appended.
    base_states_12 = [
        # Isaac home state observed (from previous handoff)
        [-0.018, +0.068, -0.050, +1.571, -1.574, +1.035,
         -0.018, -0.009, -0.053, +1.574, -1.573, +1.036],
        # Squint's actual home state (qpos and target_qpos identical)
        [+0.031, -0.006, -0.044, +1.582, -1.593, +1.019,
         +0.031, -0.006, -0.044, +1.582, -1.593, +1.019],
        # Pure home (no noise)
        [0.0, 0.0, 0.0, 1.5708, -1.5708, 1.0472,
         0.0, 0.0, 0.0, 1.5708, -1.5708, 1.0472],
    ]
    # Sweep over goal colors so Isaac can probe color sensitivity.
    color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
    grey_rgb16 = (np.ones((16, 16, 3), dtype=np.uint8) * 180)

    lines = []
    lines.append("# Deterministic policy probe — same RGB (grey 180), varied state.")
    lines.append(f"# n_state = {n_state}  (12-d qpos|target_qpos + 6-d goal_color_one_hot)")
    lines.append(f"# COLOR_PALETTE index -> name: {dict(enumerate(color_names))}")
    lines.append("")
    print("\n  ".join(lines))

    for goal_idx in range(6):
        one_hot = [0.0] * 6
        one_hot[goal_idx] = 1.0
        for i, base12 in enumerate(base_states_12):
            full_state = list(base12) + one_hot
            with torch.no_grad():
                rgb_in = torch.from_numpy(grey_rgb16).unsqueeze(0).to(device)
                state_in = torch.tensor(full_state, dtype=torch.float32, device=device).unsqueeze(0)
                feats = encoder(rgb_in)
                mean = actor.forward(feats, state_in)
                action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)[0].cpu().tolist()
            tag = f"goal={goal_idx}({color_names[goal_idx]}) state={i}"
            msg = f"  {tag:32s} qpos_first6={base12[:6]}  action={[f'{a:+.5f}' for a in action]}"
            print(msg)
            lines.append(msg)

    with open(Path(out_dir) / "action_sensitivity.txt", "w") as f:
        f.write("\n".join(lines))
    env.close()


# ---------------------------------------------------------------------------
# P4 — RUNTIME PHYSICS PARAMS
# ---------------------------------------------------------------------------
def priority4(args, device, out_dir):
    print("\n=========== P4  RUNTIME PHYSICS PARAMS ===========")
    env = build_env(args, num_envs=1)
    base = env.unwrapped
    robot = base.agent.robot

    lines = []
    def out(msg):
        print(msg); lines.append(msg)

    sim_cfg = base.sim_config
    out(f"  sim_freq         = {sim_cfg.sim_freq}")
    out(f"  control_freq     = {sim_cfg.control_freq}")
    try:
        gravity = getattr(base.scene, "gravity", None)
        out(f"  gravity          = {None if gravity is None else _to_tuple(gravity)}")
    except Exception:
        pass

    try:
        sc = sim_cfg.scene_config
        for attr in ("solver_position_iterations", "solver_velocity_iterations",
                     "contact_offset", "rest_offset", "bounce_threshold",
                     "friction_offset_threshold", "friction_correlation_distance",
                     "enable_pcm", "enable_tgs", "enable_ccd"):
            out(f"  scene.{attr:32s} = {getattr(sc, attr, '?')}")
    except Exception as e:
        out(f"  scene_config inspection failed: {e}")

    # Cube physical params (post-_load_scene)
    try:
        out(f"  cube friction    = {_to_tuple(base.item_frictions)}")
        out(f"  cube density     = {_to_tuple(base.item_densities)}")
        out(f"  cube half_sizes  = {_to_tuple(base.item_half_sizes)}")
        try:
            comp = base.item._objs[0].find_component_by_type(sapien.pysapien.physx.PhysxRigidDynamicComponent)
            if comp is None:
                comp = base.item._objs[0].find_component_by_type(sapien.pysapien.physx.PhysxRigidBaseComponent)
            out(f"  cube mass        = {getattr(comp, 'mass', '?')}")
        except Exception as e:
            out(f"  cube mass inspection failed: {e}")
    except Exception as e:
        out(f"  cube inspection failed: {e}")

    # Gripper joint drive
    try:
        gj = robot.joints_map["gripper"]._objs[0]
        out(f"  gripper stiffness = {getattr(gj, 'drive_stiffness', '?')}")
        out(f"  gripper damping   = {getattr(gj, 'drive_damping', '?')}")
        out(f"  gripper force_lim = {getattr(gj, 'drive_force_limit', '?')}")
    except Exception as e:
        out(f"  gripper drive inspection failed: {e}")

    # All active-joint drives (for sign-flip / gain comparison)
    try:
        for j in robot.active_joints:
            obj = j._objs[0]
            out(f"  joint[{j.name:18s}] stiffness={getattr(obj,'drive_stiffness','?')}  "
                f"damping={getattr(obj,'drive_damping','?')}  "
                f"force_lim={getattr(obj,'drive_force_limit','?')}")
    except Exception as e:
        out(f"  joint drive sweep failed: {e}")

    with open(Path(out_dir) / "physics_params.txt", "w") as f:
        f.write("\n".join(lines))
    env.close()


# ---------------------------------------------------------------------------
# P5 — SETTLE BEHAVIOR
# ---------------------------------------------------------------------------
def priority5(args, device, out_dir):
    print("\n=========== P5  SETTLE BEHAVIOR ===========")
    env = build_env(args, num_envs=1, seed=0)
    base = env.unwrapped
    robot = base.agent.robot
    n_act = env.unwrapped.single_action_space.shape[0]

    joint_names = [j.name for j in robot.active_joints]
    lines = [f"  joints: {joint_names}", "  step | qpos | qvel"]
    print(lines[0]); print(lines[1])

    for step in range(20):
        qpos_now = _to_tuple(robot.get_qpos())
        qvel_now = _to_tuple(robot.get_qvel())
        max_v = max(abs(v) for v in qvel_now)
        row = (
            f"  step {step:2d}  qpos=[{', '.join(f'{x:+.5f}' for x in qpos_now)}]  "
            f"qvel=[{', '.join(f'{v:+.5f}' for v in qvel_now)}]  max|qvel|={max_v:+.5f}"
        )
        print(row); lines.append(row)
        env.step(np.zeros((1, n_act), dtype=np.float32))

    with open(Path(out_dir) / "settle_behavior.txt", "w") as f:
        f.write("\n".join(lines))
    env.close()


# ---------------------------------------------------------------------------
# P6 — ACTION REPLAY GROUND TRUTH
# ---------------------------------------------------------------------------
def priority6(args, device, out_dir):
    print("\n=========== P6  ACTION REPLAY GROUND TRUTH ===========")
    env = build_env(args, num_envs=1, seed=0)
    base = env.unwrapped
    robot = base.agent.robot
    joint_names = [j.name for j in robot.active_joints]
    n_act = env.unwrapped.single_action_space.shape[0]

    lines = [f"  joints: {joint_names}", "  10 steps of +0.5 on each joint independently."]
    print(lines[0]); print(lines[1])

    for j_idx, j_name in enumerate(joint_names):
        env.reset(seed=0)
        act = np.zeros((1, n_act), dtype=np.float32); act[0, j_idx] = 0.5
        header = f"\n  joint[{j_idx}] {j_name}:"
        print(header); lines.append(header)
        for step in range(10):
            env.step(act)
            qp = _to_tuple(robot.get_qpos())
            row = f"    step {step}: qpos = [{', '.join(f'{x:+.4f}' for x in qp)}]"
            print(row); lines.append(row)

    with open(Path(out_dir) / "action_replay.txt", "w") as f:
        f.write("\n".join(lines))
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Isaac-side handoff: P1, P2, P4, P5, P6.")
    parser.add_argument("--env_id", type=str, default="SO101PlaceCube-v1")
    parser.add_argument("--control_mode", type=str, default="pd_joint_target_delta_pos")
    parser.add_argument("--obs_mode", type=str, default="rgb")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_replay_episodes", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / "run_console.log")
    sys.stdout = tee

    try:
        print(f"# isaacsim_handoff produced {datetime.now().isoformat()}")
        print(f"# checkpoint = {args.checkpoint}")
        print(f"# env_id = {args.env_id}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        priority1(args, device, out_dir)
        priority2(args, device, out_dir)
        priority4(args, device, out_dir)
        priority5(args, device, out_dir)
        priority6(args, device, out_dir)
        print("\nDONE.")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
