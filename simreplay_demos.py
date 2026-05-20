"""Sim-replay validation (cube spawned at demo's gripper end-position).

Workflow:
  1. Load offline bundle, isolate one episode.
  2. Extract the demo's qpos sequence (slice noisy_qpos from the 58-D state).
  3. Compute TCP positions via FK; pick a target xyz = TCP at the moment
     the gripper closes (state[5] > 0.5 rad, first occurrence).
  4. Reset the sim env. Override `env.unwrapped.item.pose` so the cube sits
     at (target_xy, table_z). Replay the demo's normalized actions.
  5. Report whether the gripper finished CLOSED on the cube. We only care
     about the grasp configuration — the project hands off to FK/IK for
     lift and place, so cube z-rise is not a success criterion here.

This is an ARTIFICIAL check — we move the cube to where the demo expected
it instead of validating real→sim alignment. If the demo's action stream
is grasp-shaped, this test passes. If it doesn't pass even with the cube
placed right, the demo bundle is incompatible with the sim controller.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import sapien
import torch
from tensordict import TensorDict
import gymnasium as gym

import envs  # noqa: F401  -- registers SO101PlaceCube-v1
from so101_fk import tcp_pos, finger_positions


def load_bundle(path: str) -> TensorDict:
    print(f"Loading bundle from {path}...")
    td = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(td, TensorDict):
        raise TypeError(f"Bundle is not a TensorDict (got {type(td).__name__})")
    print(f"  batch_size: {td.batch_size}")
    print(f"  state dim: {td['observations', 'state'].shape[-1]}")
    return td


def find_episodes(td: TensorDict) -> list[tuple[int, int]]:
    dones = td["dones"].to(torch.bool).numpy()
    end_idx = np.where(dones)[0]
    if len(end_idx) == 0:
        return [(0, len(dones))]
    start_idx = np.concatenate([[0], end_idx[:-1] + 1])
    return [(int(s), int(e) + 1) for s, e in zip(start_idx, end_idx)]


def compute_target_from_demo(qpos_traj: np.ndarray, mode: str = "most-closed",
                              grasp_threshold_rad: float = 0.5):
    """Cube target = midpoint between the two finger tips at a chosen frame.

    Modes:
      - 'first-closed': first frame where gripper > threshold (grasp moment)
      - 'most-closed':  frame with max gripper qpos (firmest grip)
      - 'last-closed':  last frame still > threshold (before release)
    """
    grip = qpos_traj[:, 5]
    if mode == "most-closed":
        t_target = int(np.argmax(grip))
    elif mode == "first-closed":
        idx = np.where(grip > grasp_threshold_rad)[0]
        t_target = int(idx[0]) if len(idx) else len(qpos_traj) - 1
    elif mode == "last-closed":
        idx = np.where(grip > grasp_threshold_rad)[0]
        t_target = int(idx[-1]) if len(idx) else len(qpos_traj) - 1
    else:
        raise ValueError(f"unknown target mode {mode!r}")
    q = qpos_traj[t_target]
    p1, p2 = finger_positions(q)
    midpoint = (p1 + p2) / 2.0
    return midpoint.astype(np.float32), p1, p2, t_target


def get_cube_pose(env):
    """Return cube xyz (numpy) for num_envs=1."""
    pose = env.unwrapped.item.pose
    p = pose.p  # torch tensor [N, 3] or sapien.Pose.p depending on backend
    if hasattr(p, "cpu"):
        return p[0].detach().cpu().numpy()
    return np.asarray(p)


def set_cube_pose(env, xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Teleport the cube to the given xyz + quat (wxyz)."""
    device = env.unwrapped.device
    new_p = torch.as_tensor(xyz[None], dtype=torch.float32, device=device)
    new_q = torch.as_tensor(quat_wxyz[None], dtype=torch.float32, device=device)
    from mani_skill.utils.structs.pose import Pose
    env.unwrapped.item.set_pose(Pose.create_from_pq(p=new_p, q=new_q))
    return get_cube_pose(env)


def banish_bowl(env, far_xyz=(2.0, 2.0, -1.0)) -> None:
    """The env always spawns a bowl (`self.bin`). Move it well outside the
    robot's reach and out of the wrist-camera FOV so it can't influence
    physics, rendering, or contact detection during sim-replay."""
    device = env.unwrapped.device
    p = torch.as_tensor([list(far_xyz)], dtype=torch.float32, device=device)
    q = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    from mani_skill.utils.structs.pose import Pose
    env.unwrapped.bin.set_pose(Pose.create_from_pq(p=p, q=q))


def replay(env, actions: np.ndarray, *, debug_every: int = 0, capture_frames: bool = False):
    """Step env through `actions`. Returns per-step traces, optionally sim
    wrist + 3rd-person frames."""
    cube_z, cube_xyz_traj, tcp_xyz_traj, gripper_qpos, tcp_to_cube = [], [], [], [], []
    sim_wrist, sim_3p = ([], []) if capture_frames else (None, None)
    for t, a in enumerate(actions):
        action_t = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)  # [1, 6]
        obs, _, _, _, _ = env.step(action_t)
        cube_xyz = get_cube_pose(env)
        cube_z.append(float(cube_xyz[2]))
        cube_xyz_traj.append(cube_xyz.copy())
        qpos_arm_grip = env.unwrapped.agent.robot.get_qpos()[0].detach().cpu().numpy()
        gq = float(qpos_arm_grip[5])
        gripper_qpos.append(gq)
        tcp = tcp_pos(qpos_arm_grip)
        tcp_xyz_traj.append(tcp.copy())
        tcp_to_cube.append(float(np.linalg.norm(tcp - cube_xyz)))
        if capture_frames:
            sim_wrist.append(
                obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()
            )
            # 3rd-person: env.render() gives the human render camera (an
            # overhead view that always sees the whole scene). Shadows are
            # disabled on the env so this shouldn't crash the buffer alloc.
            third = env.render()
            if hasattr(third, "detach"):
                third = third.detach().cpu().numpy()
            if third.ndim == 4:
                third = third[0]
            sim_3p.append(third.astype(np.uint8))
        if debug_every and t % debug_every == 0:
            print(f"    step {t:3d}: cube_z={cube_xyz[2]:.4f}  "
                  f"gripper={gq:+.3f}  |tcp-cube|={tcp_to_cube[-1]:.4f}")
    return {
        "cube_z": np.array(cube_z),
        "cube_xyz": np.array(cube_xyz_traj),
        "tcp_xyz": np.array(tcp_xyz_traj),
        "gripper": np.array(gripper_qpos),
        "tcp_to_cube": np.array(tcp_to_cube),
        "sim_wrist": sim_wrist,
        "sim_3p": sim_3p,
    }


def write_video(out_path: str, demo_rgb: np.ndarray, sim_wrist: list[np.ndarray],
                sim_3p: list[np.ndarray], fps: int = 30):
    """Write a 3-pane video: demo wrist | sim wrist | sim 3rd-person. All
    panes rescaled to a common height. Labels overlaid."""
    import cv2

    def _to_h(img: np.ndarray, target_h: int, interp=cv2.INTER_AREA) -> np.ndarray:
        new_w = int(img.shape[1] * target_h / img.shape[0])
        return cv2.resize(img, (new_w, target_h), interpolation=interp)

    H = max(sim_3p[0].shape[0], sim_wrist[0].shape[0], demo_rgb.shape[1])
    out_frames = []
    gutter = np.zeros((H, 4, 3), dtype=np.uint8)
    for t in range(len(sim_wrist)):
        d = _to_h(demo_rgb[t], H, interp=cv2.INTER_NEAREST)
        sw = _to_h(sim_wrist[t], H)
        s3 = _to_h(sim_3p[t], H)
        d = cv2.putText(d.copy(), f"demo wrist t={t}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        sw = cv2.putText(sw.copy(), "sim wrist", (8, 24),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        s3 = cv2.putText(s3.copy(), "sim 3rd-person", (8, 24),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        side = np.concatenate([d, gutter, sw, gutter, s3], axis=1)
        out_frames.append(side)

    import imageio.v2 as imageio
    imageio.mimsave(out_path, out_frames, fps=fps, codec="libx264", quality=8)
    print(f"  wrote {len(out_frames)} frames to {out_path} ({fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="offline_bundles/projet3_v1_80x144.pt")
    parser.add_argument("--episode", type=int, default=0,
                        help="Single episode index (ignored if --all-episodes is set)")
    parser.add_argument("--all-episodes", action="store_true",
                        help="Iterate over all episodes; per-episode pass/fail report")
    parser.add_argument("--env-id", default="SO101PlaceCube-v1")
    parser.add_argument("--debug-every", type=int, default=0)
    parser.add_argument("--video", type=str, default=None,
                        help="Write side-by-side mp4 (demo wrist | sim base_camera) to this path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS for the side-by-side mp4")
    parser.add_argument("--target-mode", choices=["most-closed", "first-closed", "last-closed"],
                        default="last-closed",
                        help="How to pick the demo frame whose FK midpoint becomes the cube spawn pose.")
    parser.add_argument("--annotations", default=None,
                        help="Optional: path to Hugo's episodes.json. If set, uses "
                             "`cube_pose_urdf_world` per episode instead of the FK heuristic.")
    args = parser.parse_args()

    td = load_bundle(args.bundle)
    episodes = find_episodes(td)
    print(f"  episodes: {len(episodes)}")
    if not args.all_episodes and args.episode >= len(episodes):
        sys.exit(f"--episode {args.episode} out of range")

    print(f"Creating env (num_envs=1, no DR, no shadows)...")
    env = gym.make(
        args.env_id, num_envs=1, obs_mode="rgb",
        render_mode="rgb_array",  # required for env.render() (3rd-person view)
        pick_only_reward=True,
        n_distractors=0,
        domain_randomization=False,
        domain_randomization_config={"shadows": False},
    )

    # Optionally load Hugo's per-episode annotations as the cube spawn source.
    anns = None
    if args.annotations is not None:
        import json
        with open(args.annotations) as f:
            jf = json.load(f)
        anns = {int(k): {"pose": ep["cube_pose_urdf_world"],
                          "grasp_frame": int(ep["grasp_frame_local"])}
                for k, ep in jf["episodes"].items()}
        print(f"  loaded {len(anns)} annotations from {args.annotations}")

    ep_indices = range(len(episodes)) if args.all_episodes else [args.episode]
    results = []
    for ep_idx in ep_indices:
        s, e = episodes[ep_idx]
        actions = td["actions"][s:e].numpy()
        state = td["observations", "state"][s:e].numpy()
        qpos_traj = state[:, 0:6]
        if anns is not None and ep_idx in anns:
            pose = anns[ep_idx]["pose"]
            target_xyz = np.array(pose[:3], dtype=np.float32)
            target_quat = np.array(pose[3:], dtype=np.float32)
            # Use Hugo's grasp_frame_local as the reference moment for the
            # verdict — that's the demo frame his XYZ corresponds to.
            t_target = min(anns[ep_idx]["grasp_frame"], len(qpos_traj) - 1)
            jaw1 = jaw2 = target_xyz  # placeholder, not used
        else:
            target_xyz, jaw1, jaw2, t_target = compute_target_from_demo(qpos_traj,
                                                                         mode=args.target_mode)
            target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Reset env + override sim qpos to demo's first frame
        env.reset(seed=0)
        demo_q0 = qpos_traj[0]
        init_target = torch.zeros_like(env.unwrapped.agent.robot.get_qpos())
        init_target[:, :6] = torch.as_tensor(demo_q0, dtype=init_target.dtype,
                                             device=init_target.device)
        env.unwrapped.agent.robot.set_qpos(init_target)
        ctrl = env.unwrapped.agent.controller
        sub_ctrls = ctrl.controllers if hasattr(ctrl, "controllers") else {"_": ctrl}
        for sub in sub_ctrls.values():
            if hasattr(sub, "_target_qpos") and sub._target_qpos is not None:
                sub._target_qpos = init_target[:, sub.active_joint_indices].clone()
                sub.set_drive_targets(sub._target_qpos)

        # Place cube at the FK midpoint between the two finger tips at the
        # last frame where the gripper is still closed (per user method).
        final = set_cube_pose(env, target_xyz, target_quat)
        # Banish the bowl — env always spawns it but it has no role in this
        # grasp-only validation.
        banish_bowl(env)

        capture = (args.video is not None and not args.all_episodes)
        traces = replay(env, actions, debug_every=args.debug_every,
                        capture_frames=capture)

        cz, gp, d = traces["cube_z"], traces["gripper"], traces["tcp_to_cube"]
        tg = min(t_target, len(d) - 1)  # demo's last-closed-gripper frame
        t_min = int(np.argmin(d))
        closed = gp[tg] > 0.4  # SO-101 sim gripper closes a tad less than the real one
        near = d[tg] < 0.04
        verdict = (
            "GRASP" if (closed and near) else
            "miss-far" if closed else
            "miss-open" if near else
            "miss-both"
        )
        results.append({
            "ep": ep_idx,
            "T": e - s,
            "t_target": t_target,
            "target_xyz": target_xyz.tolist(),
            "jaw_gap_mm": float(np.linalg.norm(jaw1 - jaw2) * 1000),
            "tcp_cube_at_target_cm": float(d[tg] * 100),
            "tcp_cube_min_cm": float(d.min() * 100),
            "t_min": t_min,
            "gripper_at_target": float(gp[tg]),
            "gripper_max": float(gp.max()),
            "verdict": verdict,
        })

        line = (
            f"ep {ep_idx:>2d} | T={e-s:>3d} t*={t_target:>3d} | "
            f"target=({target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}) | "
            f"|tcp-cube|@t*={d[tg]*100:5.1f}cm  min={d.min()*100:5.1f}cm@t={t_min:>3d} | "
            f"grip@t*={gp[tg]:+.2f} max={gp.max():+.2f} | {verdict}"
        )
        print(line)

        if capture:
            demo_rgb = td["observations", "rgb"][s:e].numpy()
            out_path = args.video
            print(f"  → writing video to {out_path}")
            write_video(out_path, demo_rgb, traces["sim_wrist"], traces["sim_3p"], fps=args.fps)

    # Summary table for --all-episodes
    if args.all_episodes:
        print()
        from collections import Counter
        c = Counter(r["verdict"] for r in results)
        print(f"== Summary ({len(results)} episodes) ==")
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            print(f"  {k:>10s}: {v}")
        avg_tg = float(np.mean([r["tcp_cube_at_target_cm"] for r in results]))
        avg_min = float(np.mean([r["tcp_cube_min_cm"] for r in results]))
        print(f"  mean |tcp-cube|: at last-closed t* = {avg_tg:.1f}cm  min over traj = {avg_min:.1f}cm")


if __name__ == "__main__":
    main()
