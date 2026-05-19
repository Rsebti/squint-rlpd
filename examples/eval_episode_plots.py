"""Run ONE eval episode per checkpoint and dump:

  * runs/eval_episodes/<name>/log.npz            (qpos, target_qpos, action, ...)
  * runs/eval_episodes/<name>/wrist.mp4          (the wrist-camera feed the
                                                  policy sees — 640x480, 10 fps)
  * runs/eval_episodes/<name>/actions.png        (per-joint command trace,
                                                  normalized AND in rad/step)
  * runs/eval_episodes/<name>/qpos.png           (per-joint qpos + target_qpos
                                                  in radians)

Action units (from envs/robot/so101.py + pd_joint_pos.py):
  Controller is `pd_joint_target_delta_pos` with `normalize_action=True`, so
  the action space presented to the policy is [-1, 1] per joint. Internally
  the controller scales each [-1, 1] back to the configured per-step delta:
      arm joints (5): +-0.05 rad / step
      gripper      : +-0.20 rad / step
  At control_freq=10 Hz that's +-0.5 rad/s arm, +-2.0 rad/s gripper.

Usage:
  python examples/eval_episode_plots.py
"""
import os
import sys

# Repo root on path before importing envs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.disable(level=logging.WARN)

import argparse
import time
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import envs  # noqa: F401  (registers SO101PlaceCube-v1)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import Actor, CNNEncoder

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
# Per-step delta caps (radians) the controller maps the +-1 action to.
# Matches envs/robot/so101.py pd_joint_delta_pos: arm +-0.05, gripper +-0.20.
RAD_PER_STEP_CAP = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.20], dtype=np.float32)
CONTROL_HZ = 10.0
# (sim_freq, (cam_lag_min, cam_lag_max)) — matches scripts/brev_run_ablation.sh.
RUN_CONFIGS = {
    "eval1_sim100_lat":   {"sim_freq": 100, "camera_lag": (1, 5)},
    "eval1_sim100_nolat": {"sim_freq": 100, "camera_lag": (0, 0)},
    "eval1_sim300_lat":   {"sim_freq": 300, "camera_lag": (3, 15)},
    "eval1_sim300_nolat": {"sim_freq": 300, "camera_lag": (0, 0)},
}


def _make_env(sim_freq: int, camera_lag: tuple, max_episode_steps: int):
    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=False,  # clean trace
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=640, height=360),  # 16:9, matches real camera
        human_render_camera_configs=dict(shader_pack="default", width=512, height=512),
        n_distractors=0,
        sim_freq=sim_freq,
        control_freq=int(CONTROL_HZ),
        domain_randomization_config={"camera_lag_substeps_range": camera_lag},
        max_episode_steps=max_episode_steps,
    )
    env = gym.make("SO101PlaceCube-v1", num_envs=1, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    return env


def _load_actor(env, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(36, 64, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval(); actor.eval()
    return encoder, actor, int(ckpt.get("global_step", -1))


def run_episode(ckpt_path: str, cfg: dict, out_dir: Path,
                seed: int = 0, max_steps: int = 100):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _make_env(cfg["sim_freq"], cfg["camera_lag"], max_episode_steps=max_steps)
    encoder, actor, step_at_save = _load_actor(env, ckpt_path, device)
    base = env.unwrapped

    obs, _ = env.reset(seed=seed)
    cube_xyz_init = base.item.pose.p.detach().cpu().numpy().flatten()
    bin_xyz = base.bin.pose.p.detach().cpu().numpy().flatten()
    goal_color = int(base.goal_color_idx[0].item()) if hasattr(base, "goal_color_idx") else -1
    color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
    gname = color_names[goal_color] if 0 <= goal_color < 6 else "?"
    print(f"  ckpt step={step_at_save} | goal={goal_color} ({gname}) | cube_init={cube_xyz_init.round(3)}")

    log_qpos, log_target, log_action, log_wrist = [], [], [], []
    t_wall0 = time.perf_counter()
    succeeded = False
    for step in range(max_steps):
        # --- log clean qpos + controller's running target ---
        q = base.agent.robot.get_qpos().detach().cpu().numpy().flatten()
        ctrl_arm = base.agent.controller.controllers["arm"]
        tgt = getattr(ctrl_arm, "_target_qpos", None)
        tgt = tgt.detach().cpu().numpy().flatten() if torch.is_tensor(tgt) else np.full(6, np.nan)

        # --- compute action from policy ---
        rgb_now = obs["rgb"]
        state_now = obs["state"]
        if not torch.is_tensor(rgb_now):
            rgb_now = torch.from_numpy(rgb_now)
        if not torch.is_tensor(state_now):
            state_now = torch.from_numpy(state_now)
        # F.interpolate `size` is (H, W) → output is H=36, W=64 (landscape).
        # The full 640×360 wrist frame is *area-resized* (not cropped) so the
        # policy sees the same field-of-view it was trained on. 64/36 = 16/9
        # EXACTLY, and 640→64 / 360→36 is an exact ÷10 (uniform 10×10 pool).
        rgb_t = rgb_now.permute(0, 3, 1, 2).float()
        rgb_small = F.interpolate(rgb_t, size=(36, 64), mode="area").permute(0, 2, 3, 1).to(torch.uint8)
        with torch.no_grad():
            feats = encoder(rgb_small.to(device))
            mean = actor.forward(feats, state_now.float().to(device))
            action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)
        action_np = action.detach().cpu().numpy().astype(np.float32).flatten()

        log_qpos.append(q)
        log_target.append(tgt)
        log_action.append(action_np)
        # store wrist frame at native sensor resolution
        log_wrist.append(rgb_now.detach().cpu().numpy()[0].astype(np.uint8))

        obs, _, term, trunc, info = env.step(action.detach().cpu().numpy().astype(np.float32))
        if "success" in info:
            if float(torch.as_tensor(info["success"]).flatten()[0]) > 0.5:
                succeeded = True
        if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
            break

    cube_xyz_final = base.item.pose.p.detach().cpu().numpy().flatten()
    n_steps = len(log_qpos)
    t_total = time.perf_counter() - t_wall0
    print(f"  steps={n_steps}  wall={t_total:.2f}s  success={succeeded}  cube_final={cube_xyz_final.round(3)}")

    qpos_arr = np.stack(log_qpos)
    target_arr = np.stack(log_target)
    action_arr = np.stack(log_action)
    t_sim = np.arange(n_steps, dtype=np.float32) / CONTROL_HZ
    wrist_arr = np.stack(log_wrist)

    np.savez(
        out_dir / "log.npz",
        qpos=qpos_arr, target_qpos=target_arr, action=action_arr,
        wrist_rgb=wrist_arr, t_sim=t_sim,
        cube_xyz_init=cube_xyz_init, cube_xyz_final=cube_xyz_final,
        bin_xyz=bin_xyz, goal_color=np.array([goal_color]),
        succeeded=np.array([succeeded]),
        joint_names=np.array(JOINT_NAMES),
        sim_freq=np.array([cfg["sim_freq"]]),
        camera_lag_min=np.array([cfg["camera_lag"][0]]),
        camera_lag_max=np.array([cfg["camera_lag"][1]]),
    )

    # ── wrist-cam mp4 (H.264 via imageio-ffmpeg for browser-compatible mp4) ─
    H, W = wrist_arr.shape[1], wrist_arr.shape[2]
    writer = imageio.get_writer(
        str(out_dir / "wrist.mp4"),
        fps=int(CONTROL_HZ), codec="libx264", quality=8,
        macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for i, f in enumerate(wrist_arr):
        rgb = f.copy()
        label = f"step {i:3d}  t={t_sim[i]:.2f}s"
        cv2.rectangle(rgb, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(rgb, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
        writer.append_data(rgb)
    writer.close()

    # ── actions plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 2, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"{out_dir.name}  |  policy action (orders sent) per joint\n"
        f"left: normalized [-1, 1]  ·  right: scaled to rad / 100ms step",
        fontsize=12,
    )
    for j in range(6):
        ax_norm = axes[j, 0]
        ax_rad = axes[j, 1]
        ax_norm.plot(t_sim, action_arr[:, j], color="tab:blue", lw=1.2)
        ax_norm.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax_norm.axhline(1.0, color="r", lw=0.5, ls="--", alpha=0.4)
        ax_norm.axhline(-1.0, color="r", lw=0.5, ls="--", alpha=0.4)
        ax_norm.set_ylabel(f"{JOINT_NAMES[j]}\nnorm [-1,1]")
        ax_norm.set_ylim(-1.15, 1.15)
        ax_norm.grid(alpha=0.3)

        ax_rad.plot(t_sim, action_arr[:, j] * RAD_PER_STEP_CAP[j],
                    color="tab:orange", lw=1.2)
        ax_rad.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax_rad.axhline(RAD_PER_STEP_CAP[j], color="r", lw=0.5, ls="--", alpha=0.4)
        ax_rad.axhline(-RAD_PER_STEP_CAP[j], color="r", lw=0.5, ls="--", alpha=0.4)
        ax_rad.set_ylabel("rad / step")
        cap = RAD_PER_STEP_CAP[j]
        ax_rad.set_ylim(-1.15 * cap, 1.15 * cap)
        ax_rad.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "actions.png", dpi=120)
    plt.close(fig)

    # ── qpos + target plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"{out_dir.name}  |  joint positions [rad]\n"
        f"qpos (solid) vs controller running target_qpos (dashed)",
        fontsize=12,
    )
    for j in range(6):
        ax = axes[j]
        ax.plot(t_sim, qpos_arr[:, j], color="tab:blue", lw=1.4, label="qpos")
        ax.plot(t_sim, target_arr[:, j], color="tab:orange", lw=1.0,
                ls="--", label="target_qpos")
        ax.set_ylabel(JOINT_NAMES[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "qpos.png", dpi=120)
    plt.close(fig)

    env.close()
    return {
        "name": out_dir.name,
        "steps": n_steps,
        "success": succeeded,
        "ckpt_step": step_at_save,
        "cube_init": cube_xyz_init,
        "cube_final": cube_xyz_final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=100,
                    help="per-episode step cap; 100 @ 10 Hz = 10 s (default)")
    ap.add_argument("--out_root", default="runs/eval_episodes")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of run names to eval (default: all 4)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names = args.runs or list(RUN_CONFIGS.keys())
    summary = []
    for name in names:
        cfg = RUN_CONFIGS[name]
        ckpt = Path("runs") / name / "ckpt_best.pt"
        if not ckpt.exists():
            print(f"[skip] {name}: ckpt missing at {ckpt}")
            continue
        print(f"=== {name}  (sim_freq={cfg['sim_freq']}, camera_lag={cfg['camera_lag']}) ===")
        summary.append(run_episode(str(ckpt), cfg, out_root / name,
                                   seed=args.seed, max_steps=args.max_steps))

    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['name']:24s}  ckpt_step={s['ckpt_step']:>9}  "
              f"steps={s['steps']:>3}  success={s['success']}")


if __name__ == "__main__":
    main()
