"""Save a single wrist-camera frame from the initial position (post-reset),
using the same env/camera setup as examples/visualize_sim.py (640x480 sensor).

Output: vis_snapshot.png in the project root.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings, logging
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401
import mani_skill.envs  # noqa: F401

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vis_snapshot.png")

env = gym.make(
    'SO101PlaceCube-v1',
    obs_mode='rgb',
    render_mode='rgb_array',
    sim_backend='gpu',
    sensor_configs={'width': 1920, 'height': 1080},
    human_render_camera_configs={'width': 1920, 'height': 1080},
    num_envs=1,
    domain_randomization=True,
    domain_randomization_config={
        'wrist_camera_pos_noise': (0.0, 0.0, 0.0),
        'wrist_camera_rot_noise': (0.0, 0.0, 0.0),
        'wrist_camera_fov_noise': 0.0,
        'initial_qpos_noise_scale': 0.0,
    },
    reconfiguration_freq=None,
)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

obs, _ = env.reset(seed=1)

wrist = obs['rgb']
if wrist.shape[-1] != 3 and wrist.shape[-1] % 3 == 0:
    wrist = wrist[..., :3]
if torch.is_tensor(wrist):
    wrist = wrist.cpu().numpy()
wrist = np.asarray(wrist)[0].astype(np.uint8)  # (H, W, 3)

cv2.imwrite(OUT, cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
print(f"wrote {OUT}  shape={wrist.shape}")
env.close()
