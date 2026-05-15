"""Inspect cube palette colours under domain randomization.

Spawns 10 parallel PlaceCube envs with n_distractors=4 (env max; 5 cubes per
env). Across 10 envs that's 50 cube renders, spanning all 6 palette colours
(red, blue, green, yellow, purple, orange) multiple times each. Every env's
reset draws a fresh DR sample (cube colour jitter, lighting, image pipeline),
so the tiles let you spot-check whether any palette colour drifts beyond what
looks right.

Two cv2 windows pop up:
  - wrist: the policy's input camera (gets image-pipeline DR — gain, gamma,
           noise — applied; saturation/hue/colour-cast are off per current
           B/W-only config)
  - third-person: clean scene render, no image DR applied

Press R = resample (new reset, new DR samples), Q = quit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401  registers SO101*-v1
import mani_skill.envs  # noqa: F401


NUM_ENVS = 10
SENSOR_PX = 128
RENDER_PX = 256
TILE_COLS = 5  # 10 envs -> 2 rows x 5 cols


def make_env():
    env = gym.make(
        "SO101PlaceCube-v1",
        num_envs=NUM_ENVS,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=True,
        n_distractors=4,
        use_real_bowl=True,
        sensor_configs=dict(width=SENSOR_PX, height=SENSOR_PX),
        human_render_camera_configs=dict(
            shader_pack="default", width=RENDER_PX, height=RENDER_PX
        ),
        control_mode="pd_joint_target_delta_pos",
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    return env


def render_grid(env):
    obs, _ = env.reset()
    # Wrist camera (with image-pipeline DR applied).
    wrist = obs["rgb"]
    if not torch.is_tensor(wrist):
        wrist = torch.from_numpy(wrist)
    wrist = wrist[..., :3]  # in case multiple sensors are concatenated
    wrist_np = wrist.cpu().numpy().astype(np.uint8)
    # Third-person scene render (no image DR).
    scene = env.render()
    if torch.is_tensor(scene):
        scene = scene.cpu().numpy()
    scene = np.asarray(scene).astype(np.uint8)

    wrist_grid = tile_images(wrist_np, nrows=NUM_ENVS // TILE_COLS)
    scene_grid = tile_images(scene, nrows=NUM_ENVS // TILE_COLS)
    if torch.is_tensor(wrist_grid):
        wrist_grid = wrist_grid.cpu().numpy()
    if torch.is_tensor(scene_grid):
        scene_grid = scene_grid.cpu().numpy()
    return wrist_grid.astype(np.uint8), scene_grid.astype(np.uint8)


def main():
    env = make_env()
    print(f"Loaded SO101PlaceCube-v1 with num_envs={NUM_ENVS}, n_distractors=5.")
    print("Press R = resample DR, Q = quit.")

    while True:
        wrist_grid, scene_grid = render_grid(env)
        # Upscale wrist for visibility (128 -> ~256).
        wrist_disp = cv2.resize(
            wrist_grid,
            (wrist_grid.shape[1] * 2, wrist_grid.shape[0] * 2),
            interpolation=cv2.INTER_NEAREST,
        )
        wrist_bgr = cv2.cvtColor(wrist_disp, cv2.COLOR_RGB2BGR)
        scene_bgr = cv2.cvtColor(scene_grid, cv2.COLOR_RGB2BGR)

        cv2.imshow("wrist (image-DR)  R=resample  Q=quit", wrist_bgr)
        cv2.imshow("third-person (clean)  R=resample  Q=quit", scene_bgr)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                env.close()
                return
            if k == ord("r"):
                break  # resample by looping outer while


if __name__ == "__main__":
    main()
