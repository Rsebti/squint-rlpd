# Bowl emission sweep at exposure=1.80 (locked from prior sweep).
# Bowl currently uses emission=0.4 -> looks darker than the table at high
# exposure because it is curved (Lambertian falloff + self-shadow) while the
# table is flat-on to the lights. Sweeping emission lifts the bowl back.

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

import envs  # noqa: F401
import mani_skill.envs  # noqa: F401
from envs.place import Place as PlaceItemEnv


TASK = "SO101PlaceCube-v1"
SEED = 1
SENSOR_W, SENSOR_H = 640, 360
RENDER_W, RENDER_H = 640, 360
SETTLE_STEPS = 25
EXPOSURE = 1.80   # locked from brightness sweep

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


# Monkey-patch _randomize_bowl_tint so we can dial emission per run.
_orig_bowl_tint = PlaceItemEnv._randomize_bowl_tint

def _bowl_tint_with_emission(self, env_idx):
    """Same as _orig but takes emission from `self._sweep_emission`."""
    if self.bin is None:
        return
    env_idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
    e = float(getattr(self, "_sweep_emission", 0.4))
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

PlaceItemEnv._randomize_bowl_tint = _bowl_tint_with_emission


def make_env(bowl_emission: float):
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
    # Stash the sweep value where the patched method can find it. unwrapped
    # peels the gym wrappers down to the PlaceItemEnv instance.
    env.unwrapped._sweep_emission = bowl_emission
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
    values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.8]

    tw, th = 320, 180
    wrist_thumbs, render_thumbs = [], []

    for i, v in enumerate(values):
        print(f"[{i+1}/{len(values)}] bowl_emission={v:.2f}  (exposure={EXPOSURE})", flush=True)
        env = make_env(v)
        wrist, render = capture(env)
        env.close()

        wrist_bgr  = cv2.cvtColor(cv2.resize(wrist,  (tw, th), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_RGB2BGR)
        render_bgr = cv2.cvtColor(cv2.resize(render, (tw, th), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_RGB2BGR)

        label = f"#{i+1}  bowl_em={v:.2f}"
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
    out_path = os.path.join(out_dir, "bowl_emission.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}")
    print(f"Values (low->high): {values}")


if __name__ == "__main__":
    main()
