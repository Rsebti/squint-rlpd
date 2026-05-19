# Brightness sweep: render N variants of the env at different exposure values
# and save a single labeled grid PNG so the user can pick the best look.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401  — registers SO101* envs
import mani_skill.envs  # noqa: F401


TASK = "SO101PlaceCube-v1"
SEED = 1
SENSOR_W, SENSOR_H = 640, 360
RENDER_W, RENDER_H = 640, 360
SETTLE_STEPS = 25  # let robot/gripper settle before snapshot

# Pinning all lighting knobs to their range midpoints so only `exposure` varies.
PINNED = dict(
    room_brightness_range=(0.10, 0.10),
    directional_key_intensity_range=(0.65, 0.65),
    directional_fill_intensity_range=(0.15, 0.15),
    point_light_intensity_range=(0.15, 0.15),
    item_emission_range=(0.05, 0.05),
    # Camera/scene framing must be identical across variants
    wrist_camera_pos_noise=(0.0, 0.0, 0.0),
    wrist_camera_rot_noise=(0.0, 0.0, 0.0),
    wrist_camera_fov_noise=0.0,
    third_camera_pos_noise=(0.0, 0.0, 0.0),
    third_camera_target_noise=0.0,
    third_camera_rot_noise=0.0,
    third_camera_fov_noise=0.0,
    initial_qpos_noise_scale=0.0,
    # Photometric DR also off so we see the raw lighting result
    image_noise_sigma_range=(0.0, 0.0),
    image_channel_gain_range=(1.0, 1.0),
    image_gamma_range=(1.0, 1.0),
    image_hue_shift_deg=0.0,
    image_saturation_range=(1.0, 1.0),
    # Disable camera-lag substep caching (its CPU cache fights the GPU
    # image-pipeline tensors when DR is on).
    camera_lag_substeps_range=(0, 0),
    obs_delay_steps_range=(0, 0),
)


def make_env(exposure_value: float):
    dr = dict(PINNED)
    dr["exposure_range"] = (float(exposure_value), float(exposure_value))

    env = gym.make(
        TASK,
        obs_mode="rgb",
        render_mode="rgb_array",
        sensor_configs={"width": SENSOR_W, "height": SENSOR_H},
        human_render_camera_configs={"width": RENDER_W, "height": RENDER_H},
        num_envs=1,
        sim_backend="physx_cuda",  # force GPU sim so DR tensors land on cuda
        domain_randomization=True,
        domain_randomization_config=dr,
        reconfiguration_freq=None,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.reset(seed=SEED)
    return env


def capture(env):
    """Settle for a few steps with a closed gripper, then grab wrist + 3rd-person frames."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[..., -1] = -1.0  # close gripper
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _, _ = env.step(action)
    render_rgb = env.render()  # (1, H, W, 3) tensor
    wrist_rgb = obs["rgb"]     # (1, H, W, 3)
    if wrist_rgb.shape[-1] != 3 and wrist_rgb.shape[-1] % 3 == 0:
        wrist_rgb = wrist_rgb[..., :3]
    return (
        wrist_rgb[0].detach().cpu().numpy().astype(np.uint8),
        render_rgb[0].detach().cpu().numpy().astype(np.uint8),
    )


def label_panel(img: np.ndarray, text: str, *, scale: float = 0.7) -> np.ndarray:
    """Stack a black header band with white text above `img`."""
    band_h = 32
    band = np.zeros((band_h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def main():
    values = [0.30, 0.50, 0.70, 0.90, 1.00, 1.10, 1.30, 1.50, 1.80, 2.20]

    thumb_w, thumb_h = 320, 180  # 16:9
    wrist_thumbs, render_thumbs = [], []

    for i, v in enumerate(values):
        print(f"[{i+1}/{len(values)}] exposure={v:.2f}", flush=True)
        env = make_env(v)
        wrist, render = capture(env)
        env.close()

        wrist_small  = cv2.resize(wrist,  (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        render_small = cv2.resize(render, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # cv2 wants BGR for imwrite
        wrist_small  = cv2.cvtColor(wrist_small,  cv2.COLOR_RGB2BGR)
        render_small = cv2.cvtColor(render_small, cv2.COLOR_RGB2BGR)

        label = f"#{i+1}  exposure={v:.2f}"
        wrist_thumbs.append(label_panel(wrist_small,  f"{label}  (wrist)"))
        render_thumbs.append(label_panel(render_small, f"{label}  (3rd-person)"))

    # Two stacked grids, 5 columns × 2 rows each: wrist on top, render below.
    cols = 5
    def grid(panels):
        rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
        return np.vstack(rows)

    wrist_grid  = grid(wrist_thumbs)
    render_grid = grid(render_thumbs)
    separator = np.full((6, wrist_grid.shape[1], 3), 80, dtype=np.uint8)
    full = np.vstack([wrist_grid, separator, render_grid])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "brightness.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}")
    print(f"Grid: {full.shape[1]}w x {full.shape[0]}h  (top half = wrist obs, bottom half = 3rd-person render)")
    print(f"Values (left→right, top→bottom): {values}")


if __name__ == "__main__":
    main()
