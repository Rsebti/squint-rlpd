# 6-cube color calibration row, with a sweep over a uniform brightness
# multiplier applied to every palette color (HSV V scaling on base_color).
#
# Layout: 6 cubes at fixed x=0.30, z=cube_half_size, 1 cm gaps along y.
# Lighting locked: exposure=1.80, table_albedo=0.85, bowl_emission=0.80.
# Bowl is moved out of frame so the colour row is unobstructed.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging; logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym
from mani_skill.utils.structs.pose import Pose
from sapien.render import RenderBodyComponent

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import envs  # noqa: F401
import envs.place as place_mod
import mani_skill.envs  # noqa: F401
from envs.place import Place as PlaceEnv


TASK = "SO101PlaceCube-v1"
SEED = 1
SENSOR_W, SENSOR_H = 640, 360
RENDER_W, RENDER_H = 640, 360
EXPOSURE = 1.80
BOWL_EMISSION = 0.80
TABLE_ALBEDO = 0.85

CUBE_HALF = 0.010          # 2 cm cubes, no DR jitter
CUBE_GAP  = 0.010          # 1 cm between adjacent cubes
N_CUBES   = 6              # one per palette colour
ROW_X     = 0.30           # in front of robot, in 3rd-person camera FOV

# Camera framing reminder: 3rd-person look_at([0.5, 0.3, 0.35] -> [0.3, 0, 0.1])
# (base_random_env.py:254). x=0.30, y∈[-0.075, +0.075] sits inside that frame.

# Pin all DR knobs not under sweep, including cube material so the colour we see
# on screen is exactly COLOR_PALETTE[i] times the brightness multiplier.
PINNED = dict(
    exposure_range=(EXPOSURE, EXPOSURE),
    room_brightness_range=(0.10, 0.10),
    directional_key_intensity_range=(0.65, 0.65),
    directional_fill_intensity_range=(0.15, 0.15),
    point_light_intensity_range=(0.15, 0.15),
    item_emission_range=(0.05, 0.05),
    item_sat_jitter=0.0,
    item_value_jitter=0.0,
    item_roughness_range=(0.825, 0.825),
    item_metallic_range=(0.075, 0.075),
    item_specular_range=(0.05, 0.05),
    cube_half_size_range=(CUBE_HALF, CUBE_HALF),
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

# Lock table albedo.
place_mod.SCENE_NEUTRAL_RGB = (TABLE_ALBEDO, TABLE_ALBEDO, TABLE_ALBEDO)

# Patch bowl tint to use the locked emission.
def _bowl_tint_locked(self, env_idx):
    if self.bin is None:
        return
    env_idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
    base = [1.0, 1.0, 1.0, 1.0]
    e = BOWL_EMISSION
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

# Patch n_distractors validation so we can have 5 distractors (=6 cubes total).
_orig_init = PlaceEnv.__init__
def _patched_init(self, *args, **kwargs):
    n = kwargs.get("n_distractors", 1)
    if n > 4:
        # Skip the cardinal-faces check by temporarily lowering, then restoring.
        kwargs["n_distractors"] = 4
        _orig_init(self, *args, **kwargs)
        self.n_distractors = n
        # _load_scene already ran with n=4, so we need to rebuild distractors —
        # but easier: we just live with 5 distractor slots being missing. Better
        # path: call the original with the value forced after the check.
    else:
        _orig_init(self, *args, **kwargs)

# Instead of the above hack, override the check by patching __init__ directly
# at the assertion line. Simpler: subclass.
class PlaceUnlimited(PlaceEnv):
    def __init__(self, *args, **kwargs):
        # Skip parent validation by setting n_distractors AFTER validation
        # via temporary swap.
        original_validation_n = kwargs.get("n_distractors", 1)
        if original_validation_n > 4:
            kwargs["n_distractors"] = 4
        super().__init__(*args, **kwargs)
        # ⚠ the above only builds 4 distractors. We need a different approach
        # if we want 5. See below — we accept 4 distractors + goal = 5 cubes
        # and add the 6th via a separate path is more involved.


# Cleanest path: don't try to bypass validation; use n_distractors=4 → 5 cubes,
# and accept that the 6th colour shows as a SECOND ROW cube? Actually we want
# 6. Let me directly monkey-patch the check by editing the bytecode? No — we
# can patch by setting the limit to 5 via attribute override of the class. The
# limit is an inline `if not (0 <= n_distractors <= 4): raise`. We have to
# bypass that line.
#
# Approach: temporarily monkey-patch the ValueError-raising path by patching
# `__init__` to swallow the specific exception. Cleaner: replace the entire
# __init__ with a version that omits the check.

import inspect, textwrap
src = inspect.getsource(PlaceEnv.__init__)
src = src.replace(
    "if not (0 <= n_distractors <= 4):",
    "if not (0 <= n_distractors <= 5):",
)
# exec lacks the __class__ cell, so super() without args breaks. Rewrite to
# the explicit form before compiling.
src = src.replace("super().__init__", "super(PlaceEnv, self).__init__")
src = textwrap.dedent(src)
ns: dict = {}
exec(
    "from envs.place import *\n"
    "from envs.place import COLOR_PALETTE, NUM_COLORS, SCENE_NEUTRAL_RGB\n"
    "from envs.base_random_env import BaseRandomEnv\n"
    + src,
    {**globals(), **place_mod.__dict__, "PlaceEnv": PlaceEnv},
    ns,
)
PlaceEnv.__init__ = ns["__init__"]

# _initialize_episode picks cube positions over 4 cardinal slots and indexes
# `cardinal_dirs[:, k]` for each distractor, so n_distractors>4 IndexErrors
# inside the __init__'s implicit first reset. Wrap the original to clamp the
# visible distractor list to 4 during the per-episode init — `lay_out_six_cubes`
# overrides every cube's pose and colour afterwards anyway.
_orig_init_ep = PlaceEnv._initialize_episode
def _init_ep_safe(self, env_idx, options):
    saved_d = self.distractors
    saved_n = self.n_distractors
    saved_ci = getattr(self, "distractor_color_idxs", None)
    if saved_n > 4:
        self.distractors = list(saved_d[:4])
        self.n_distractors = 4
        if saved_ci is not None:
            self.distractor_color_idxs = saved_ci[:, :4].contiguous()
    try:
        return _orig_init_ep(self, env_idx, options)
    finally:
        self.distractors = saved_d
        self.n_distractors = saved_n
        if saved_ci is not None:
            self.distractor_color_idxs = saved_ci
PlaceEnv._initialize_episode = _init_ep_safe

# Override the human render camera so it points directly at the cube row.
# Row is at x=ROW_X (0.30), y∈[-0.09, +0.09], z=CUBE_HALF. Camera positioned
# straight in front of the row, slightly above, with `up` = +z.
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils as _su

def _calibration_camera_configs(self):
    pose = _su.look_at(eye=[ROW_X + 0.18, 0.0, 0.10],
                       target=[ROW_X, 0.0, CUBE_HALF],
                       up=(0, 0, 1))
    return CameraConfig("render_camera", pose, RENDER_W, RENDER_H,
                        45 * np.pi / 180, 0.01, 100)

PlaceEnv._default_human_render_camera_configs = property(_calibration_camera_configs)


def make_env(n_distractors=5):
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
        n_distractors=n_distractors,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.reset(seed=SEED)
    return env


def _set_palette_color_with_scale(env_u, actor, env_idx, color_idx, brightness_scale):
    """Paint `actor` with COLOR_PALETTE[color_idx] * brightness_scale, no jitter."""
    cfg = env_u.domain_randomization_config
    rgb = place_mod.COLOR_PALETTE[int(color_idx)].astype(np.float32) * float(brightness_scale)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
    emit_f = cfg.item_emission_range[0]
    emissive = [float(rgb[0]) * emit_f, float(rgb[1]) * emit_f,
                float(rgb[2]) * emit_f, 1.0]
    roughness = cfg.item_roughness_range[0]
    metallic  = cfg.item_metallic_range[0]
    specular  = cfg.item_specular_range[0]
    obj = actor._objs[env_idx]
    entity = getattr(obj, "entity", obj)
    comp = entity.find_component_by_type(RenderBodyComponent)
    if comp is None:
        return
    for render_shape in comp.render_shapes:
        for part in render_shape.parts:
            part.material.set_base_color(rgba)
            part.material.set_emission(emissive)
            part.material.set_roughness(roughness)
            part.material.set_metallic(metallic)
            part.material.set_specular(specular)


def lay_out_six_cubes(env, brightness_scale):
    """Override poses + colors AFTER reset so the 6 cubes sit in a row."""
    u = env.unwrapped
    device = u.device
    identity_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # Move bowl out of camera frame (far behind robot, below table).
    bowl_xyz = torch.tensor([[-1.0, 0.0, -0.5]], device=device)
    u.bin.set_pose(Pose.create_from_pq(bowl_xyz, identity_q))

    pitch = 2 * CUBE_HALF + CUBE_GAP   # 0.020 + 0.010 = 0.030 m
    y_start = -(N_CUBES - 1) * pitch / 2  # centred on y=0

    actors = [u.item] + list(u.distractors)  # 1 goal + 5 distractors
    assert len(actors) == N_CUBES, f"expected {N_CUBES} actors, got {len(actors)}"

    for i, actor in enumerate(actors):
        y = y_start + i * pitch
        xyz = torch.tensor([[ROW_X, y, CUBE_HALF]], device=device)
        actor.set_pose(Pose.create_from_pq(xyz, identity_q))
        _set_palette_color_with_scale(u, actor, env_idx=0,
                                       color_idx=i, brightness_scale=brightness_scale)


def capture(env):
    """Render directly, no env.step() — stepping reseeds cube positions and
    we already placed the row by hand. We just need the GPU render buffer in
    sync with the poses we set."""
    u = env.unwrapped
    if hasattr(u, "scene") and hasattr(u.scene, "px"):
        # Push pose changes into the physx GPU pose buffer.
        try:
            u.scene.px.gpu_apply_rigid_dynamic_data()
            u.scene.px.gpu_fetch_rigid_dynamic_data()
        except Exception:
            pass
    u.scene.update_render(update_sensors=False, update_human_render_cameras=True)
    render_rgb = u.scene.get_human_render_camera_images()
    # get_human_render_camera_images() returns a dict {camera_name: tensor}.
    if isinstance(render_rgb, dict):
        render_rgb = next(iter(render_rgb.values()))
    render_arr = render_rgb[0].detach().cpu().numpy().astype(np.uint8)
    return render_arr


def label_panel(img, text, scale=0.7):
    band = np.zeros((32, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(band, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def crop_to_cube_row(img, render_w=RENDER_W, render_h=RENDER_H):
    """The custom calibration camera is centered on the row, so just keep
    the middle band."""
    h, w = img.shape[:2]
    y_lo = int(0.35 * h)
    y_hi = int(0.75 * h)
    return img[y_lo:y_hi, :, :]


def main():
    values = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.55, 1.80]

    panels = []

    for i, v in enumerate(values):
        print(f"[{i+1}/{len(values)}] brightness_scale={v:.2f}", flush=True)
        env = make_env(n_distractors=5)
        lay_out_six_cubes(env, brightness_scale=v)
        render = capture(env)
        env.close()

        cropped = crop_to_cube_row(render)
        # Up-scale so the 2 cm cubes have ~ enough pixels to read colour.
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        panels.append(label_panel(big_bgr, f"#{i+1}  cube_v_scale={v:.2f}  (R B G Y P O)"))

    # 2 cols × 5 rows so each panel is wide and the cube colours are legible.
    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cube_brightness.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}  ({full.shape[1]}w × {full.shape[0]}h)")
    print(f"Values (dark→bright): {values}")
    print("Layout: 6 cubes at x=0.30, 1 cm gaps. Reading order left→right: "
          "red, blue, green, yellow, purple, orange.")


if __name__ == "__main__":
    main()
