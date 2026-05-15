"""Top-down view of the cube + bowl spawn distribution.

Resets the PlaceCube env many times and scatters every sampled cube /
bowl XY position on a matplotlib plot. Overlays the configured spawn
box rectangle, the robot base, and a few reference circles so you can
eyeball whether the spawn region is the size you want.

Coordinates are in the robot base frame (robot at origin).

Usage:
    python examples/spawn_box_view.py [--num_resets 200] [--n_distractors 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gymnasium as gym
import torch

import envs  # noqa: F401  registers SO101*-v1
import mani_skill.envs  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_resets", type=int, default=200,
                    help="Number of resets (each batched across num_envs envs)")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--n_distractors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/spawn_box_view.png")
    args = ap.parse_args()

    env = gym.make(
        "SO101PlaceCube-v1",
        num_envs=args.num_envs,
        obs_mode="state",
        sim_backend="gpu",
        domain_randomization=False,  # we only care about spawn geometry
        n_distractors=args.n_distractors,
        use_real_bowl=True,
        control_mode="pd_joint_target_delta_pos",
    )
    base = env.unwrapped
    sbp = base.spawn_box_pos
    sbh = base.spawn_box_half_size
    print(f"spawn_box_pos       = {sbp}")
    print(f"spawn_box_half_size = {sbh}  (full side = {2*sbh*100:.1f} cm)")

    cube_xy = []
    bowl_xy = []
    distractor_xy = []
    for r in range(args.num_resets):
        env.reset(seed=args.seed + r)
        # Express in robot frame: bowl in robot frame is what the policy sees.
        robot_pose_inv = base.agent.robot.pose.inv()
        bowl_p = (robot_pose_inv * base.bin.pose).p
        cube_p = (robot_pose_inv * base.item.pose).p
        bowl_xy.append(bowl_p[:, :2].cpu().numpy())
        cube_xy.append(cube_p[:, :2].cpu().numpy())
        # Distractors live as separate actors; grab their XY if present.
        if hasattr(base, "distractors") and base.distractors:
            for d in base.distractors:
                d_p = (robot_pose_inv * d.pose).p
                distractor_xy.append(d_p[:, :2].cpu().numpy())

    cube_xy = np.concatenate(cube_xy, axis=0)
    bowl_xy = np.concatenate(bowl_xy, axis=0)
    if distractor_xy:
        distractor_xy = np.concatenate(distractor_xy, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Spawn box outline (in robot frame).
    cx, cy = float(sbp[0]), float(sbp[1])
    rect = patches.Rectangle(
        (cx - sbh, cy - sbh), 2 * sbh, 2 * sbh,
        linewidth=2.0, edgecolor="black", facecolor="none",
        label=f"spawn box ({2*sbh*100:.1f}x{2*sbh*100:.1f} cm)",
    )
    ax.add_patch(rect)
    # Spawn box centre marker.
    ax.plot([cx], [cy], "k+", markersize=12, markeredgewidth=2.0)

    # Scatter spawn samples.
    ax.scatter(cube_xy[:, 0], cube_xy[:, 1], s=8, alpha=0.45,
               color="tab:red", label=f"goal cube  (n={len(cube_xy)})")
    ax.scatter(bowl_xy[:, 0], bowl_xy[:, 1], s=12, alpha=0.45,
               color="tab:blue", label=f"bowl       (n={len(bowl_xy)})")
    if len(distractor_xy):
        ax.scatter(distractor_xy[:, 0], distractor_xy[:, 1], s=4, alpha=0.25,
                   color="tab:gray", label=f"distractors (n={len(distractor_xy)})")

    # Robot base.
    ax.plot([0.0], [0.0], "s", markersize=12,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=2.0,
            label="robot base")
    # Reference: robot's +x axis (which is forward for SO-101 at base_z_rot=0).
    ax.annotate("", xy=(0.10, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Reach reference rings (5–35 cm).
    for r_cm in (10, 20, 30, 40):
        circ = patches.Circle((0, 0), r_cm / 100.0,
                              linewidth=0.5, edgecolor="gray",
                              facecolor="none", linestyle="--", alpha=0.6)
        ax.add_patch(circ)
        ax.text(r_cm/100.0 / np.sqrt(2), -r_cm/100.0 / np.sqrt(2),
                f"{r_cm}cm", fontsize=7, color="gray", alpha=0.8)

    ax.set_aspect("equal")
    margin = max(sbh + 0.15, 0.5)
    ax.set_xlim(-0.1, cx + margin)
    ax.set_ylim(-margin, margin)
    ax.set_xlabel("x  (m, forward)")
    ax.set_ylabel("y  (m, sideways)")
    ax.set_title("Spawn distribution in robot base frame  (top-down)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
