# Table-albedo sweep, with exposure=1.80 and bowl_emission=0.80 locked.
# Currently SCENE_NEUTRAL_RGB = (0.78, 0.78, 0.78). Varying this gray value
# changes how the bowl pops against the table without altering the light rig.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging; logging.disable(level=logging.WARN)

import numpy as np
import cv2
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from sapien.render import RenderBodyComponent

import envs  # noqa: F401  — registers SO101* envs
import envs.place as place_mod
import mani_skill.envs  # noqa: F401
from envs.place import Place as PlaceEnv


TASK = "SO101PlaceCube-v1"
SEED = 1
SENSOR_W, SENSOR_H = 640, 360
RENDER_W, RENDER_H = 640, 360
SETTLE_STEPS = 25
EXPOSURE = 1.80
BOWL_EMISSION = 0.80   # locked from prior sweep

PINNED = dict(
    exposure_range=(EXPOSURE, EXPOSURE),
    room_brightness_range=(0.10, 0.10),
    directional_key_intensity_range=(0.65, 0.65),
    directional_fill_intensity_range=(0.15, 0.15),
    point_light_intensity_range=(0.15, 0.15),
    item_emission_range=(0.05, 0.05),
    wrist_camera_pos_noise=(0.0, 0.0, 0.0),
    wrist_camera_rot_noise=(0.0, 0.0, 0.0),
    wrist_camera_fov_noise=0.0,
    third_camera_pos_noise=(0.0, 0.0, 0.0),
    third_camera_target_noise=0.0,
    third_camera_rot_noise=0.0,
    third_camera_fov_noise=0.0,
    initial_qpos_noise_scale=0.0,
    image_noise_sigma_range=(0.0, 0.0),
    image_channel_gain_range=(1.0, 1.0),
    image_gamma_range=(1.0, 1.0),
    image_hue_shift_deg=0.0,
    image_saturation_range=(1.0, 1.0),
    camera_lag_substeps_range=(0, 0),
    obs_delay_steps_range=(0, 0),
)


# Patch bowl tint to use the locked emission value.
def _bowl_tint_locked(self, env_idx):
    if self.bin is None:
        return
    env_idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
    e = BOWL_EMISSION
    base = [1.0, 1.0, 1.0, 1.0]
    emission = [e, e, e, 1.0]
    for i in env_idx_list:
        obj = self.bin._objs[i]
        entity = getattr(obj, "entity", obj)
        comp = entity.find_component_by_type(RenderBodyComponent)
        if comp is None:
            continue
        for render_shape in comp.render_shapes:
            for part in render_shape.parts:
                part.material.set_base_color(base)
                part.material.set_emission(emission)

PlaceEnv._randomize_bowl_tint = _bowl_tint_locked


def make_env(table_albedo: float):
    # Module-global SCENE_NEUTRAL_RGB is read by _load_scene at env build time,
    # so reassign BEFORE gym.make.
    place_mod.SCENE_NEUTRAL_RGB = (table_albedo, table_albedo, table_albedo)

    env = gym.make(
        TASK,
        obs_mode="rgb",
        render_mode="rgb_array",
        sensor_configs={"width": SENSOR_W, "height": SENSOR_H},
        human_render_camera_configs={"width": RENDER_W, "height": RENDER_H},
        num_envs=1,
        sim_backend="physx_cuda",
        domain_randomization=True,
        domain_randomization_config=PINNED,
        reconfiguration_freq=None,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.reset(seed=SEED)
    return env


def capture(env):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[..., -1] = -1.0
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _, _ = env.step(action)
    render_rgb = env.render()
    wrist_rgb = obs["rgb"]
    if wrist_rgb.shape[-1] != 3 and wrist_rgb.shape[-1] % 3 == 0:
        wrist_rgb = wrist_rgb[..., :3]
    return (
        wrist_rgb[0].detach().cpu().numpy().astype(np.uint8),
        render_rgb[0].detach().cpu().numpy().astype(np.uint8),
    )


def label_panel(img, text, scale=0.7):
    band = np.zeros((32, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def main():
    values = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.98]

    tw, th = 320, 180
    wrist_thumbs, render_thumbs = [], []

    for i, v in enumerate(values):
        print(f"[{i+1}/{len(values)}] table_albedo={v:.2f}  "
              f"(exposure={EXPOSURE}, bowl_emission={BOWL_EMISSION})", flush=True)
        env = make_env(v)
        wrist, render = capture(env)
        env.close()

        wrist_bgr  = cv2.cvtColor(cv2.resize(wrist,  (tw, th), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_RGB2BGR)
        render_bgr = cv2.cvtColor(cv2.resize(render, (tw, th), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_RGB2BGR)

        label = f"#{i+1}  table_alb={v:.2f}"
        wrist_thumbs.append(label_panel(wrist_bgr,  f"{label} (wrist)"))
        render_thumbs.append(label_panel(render_bgr, f"{label} (3rd-p)"))

    cols = 5
    def grid(panels):
        rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
        return np.vstack(rows)

    full = np.vstack([
        grid(wrist_thumbs),
        np.full((6, cols * tw, 3), 80, dtype=np.uint8),
        grid(render_thumbs),
    ])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "table_albedo.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}")
    print(f"Values (dark->light): {values}")


if __name__ == "__main__":
    main()
