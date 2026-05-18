# Visualize all SO-101 ManiSkill3 simulation tasks

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.visualization.misc import tile_images

import utils

# Add tasks
import envs
import mani_skill.envs


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Tasks to visualize
    'tasks': [
        'SO101PlaceCube-v1',
    ],

    # Environment settings
    'num_envs': 4,
    'seed': 1,
    'obs_mode': 'rgb',  # No segmentation — wrist camera RGB only.
    'render_mode': 'rgb_array',
    # True wrist-camera resolution (matches the real lerobot Cv2Camera setup
    # and the in-env sensor default in base_random_env.py: 640x480 landscape).
    'sensor_width': 640,
    'sensor_height': 480,
    # Third-person human render camera shares the same 4:3 landscape so the
    # side-by-side tile cells stay aspect-matched.
    'render_camera_width': 640,
    'render_camera_height': 480,
    'color_jitter': False,
    # None = no obs downsampling. Show the wrist obs at native sensor res so
    # the displayed quality matches the real camera's quality.
    'downsample_size': None,
    'control_mode': None,
    'domain_randomization': True,

    # Visualization settings
    'window_size': 512,
    'steps_per_task': 300,
    'reset_interval': 10,
}


# =============================================================================
# Environment Factory
# =============================================================================

def _fit_to_window(img: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Resize an image so its longer relative dimension hits the matching cap,
    preserving the source aspect ratio. Avoids the portrait↔landscape squash
    that cv2.resize to a fixed (w, h) tuple causes."""
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w, 1.0) if h and w else 1.0
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def make_env(task: str, config: dict = CONFIG):
    """Create a ManiSkill environment with the given configuration."""

    sensor_size = {'width': config['sensor_width'], 'height': config['sensor_height']}
    render_camera_size = {
        'width': config['render_camera_width'],
        'height': config['render_camera_height'],
    }

    env_kwargs = dict(
        obs_mode=config['obs_mode'],
        render_mode=config['render_mode'],
        sensor_configs=sensor_size,
        human_render_camera_configs=render_camera_size,
        num_envs=config['num_envs'],
        domain_randomization=config['domain_randomization'],
        # Zero the per-episode wrist-camera extrinsic jitter so what we see in
        # the visualizer reflects the exact configured mount, with no DR
        # offset shifting the framing each reset. Other DR (lighting, cube
        # colours, frictions, image pipeline) is unaffected.
        domain_randomization_config={
            'wrist_camera_pos_noise': (0.0, 0.0, 0.0),
            'wrist_camera_rot_noise': (0.0, 0.0, 0.0),
        },
        reconfiguration_freq=None,
    )

    if config['control_mode'] is not None:
        env_kwargs['control_mode'] = config['control_mode']

    env = gym.make(task, **env_kwargs)

    if "rgb" in config['obs_mode']:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if config['downsample_size'] is not None:
            env = utils.DownsampleObsWrapper(env, target_size=config['downsample_size'])
        if config['color_jitter']:
            env = utils.ColorJitterWrapper(env)

    env.reset(seed=config['seed'])
    return env


# =============================================================================
# Visualization
# =============================================================================

def visualize_tasks(config: dict = CONFIG):
    """Visualize all configured tasks with random actions."""

    tasks = config['tasks']
    window_size = config['window_size']
    steps_per_task = config['steps_per_task']
    reset_interval = config['reset_interval']

    for task in tasks:
        print(f"Instantiating: {task}")
        env = make_env(task, config)

        obs, info = env.reset()
        action_shape = env.action_space.shape
        num_envs = config['num_envs']
        video_nrows = int(np.sqrt(num_envs))

        # Print the dimensions the user actually sees so portrait vs landscape
        # is unambiguous (window resizing preserves aspect — see _fit_to_window).
        if isinstance(obs, dict) and 'rgb' in obs:
            o = obs['rgb']
            print(f"  obs rgb shape: {tuple(o.shape)}  (N, H, W, C)")
        print(f"  sensor: {config['sensor_width']}w x {config['sensor_height']}h"
              f"  ({'portrait' if config['sensor_height'] > config['sensor_width'] else 'landscape'})")

        print(f"Running: {task}")

        for step in range(steps_per_task):
            # Generate action: open gripper for first 20 steps, close after
            action = np.zeros(action_shape)
            if step < 20:
                action[..., -1] = 1
            else:
                action[..., -1] = -1

            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).any()

            # Get third-person render view (N, H, W, 3)
            render_rgb = env.render()

            # Get observation RGB (wrist camera view)
            if isinstance(obs, dict) and 'rgb' in obs:
                obs_rgb = obs['rgb']  # (N, H, W, C) where C may be 3 or 3*num_views

                # Handle multiple camera views - just take first view for simplicity
                if obs_rgb.shape[-1] != 3 and obs_rgb.shape[-1] % 3 == 0:
                    obs_rgb = obs_rgb[..., :3]  # Take first camera view

                # Resize obs to match render size (obs may be downsampled)
                render_h, render_w = render_rgb.shape[1], render_rgb.shape[2]
                if obs_rgb.shape[1] != render_h or obs_rgb.shape[2] != render_w:
                    obs_rgb = torch.nn.functional.interpolate(
                        obs_rgb.permute(0, 3, 1, 2).float(),  # (N, 3, H, W)
                        size=(render_h, render_w),
                        mode='nearest',
                    ).permute(0, 2, 3, 1).to(torch.uint8)  # (N, H, W, 3)

                # Interleave: concatenate obs and render for each env, then tile
                paired = torch.cat([obs_rgb, render_rgb], dim=2)
                rgb = tile_images(paired, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                # Scale to display while preserving aspect (so portrait stays
                # portrait and landscape stays landscape on screen).
                rgb = _fit_to_window(rgb, max_h=window_size * 2, max_w=window_size * 4)
            else:
                # State mode: only show render view
                rgb = tile_images(render_rgb, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                rgb = _fit_to_window(rgb, max_h=window_size, max_w=window_size * 2)

            # Display
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            print(f"Step: {step}/{steps_per_task}, done={done}", end="\r")
            cv2.imshow("Interleaved: Obs | Render per env", rgb)
            cv2.waitKey(30)

            # Reset on interval or done
            if (step % reset_interval == 0) or done:
                env.reset()

        env.close()
        cv2.destroyAllWindows()
        print(f"Finished: {task}                    ")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    visualize_tasks()