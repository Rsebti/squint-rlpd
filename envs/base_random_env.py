"""Base environment classes with domain randomization support.

This module provides a clean hierarchy of environment classes:
- BaseRandomEnv: Common DR (gripper, lighting, robot color)
- ThirdCameraEnv: Third-person camera with every-step pose randomization
- WristCameraEnv: Wrist camera with gripper-following randomization

Usage:
    from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

    class MyTask(DefaultCameraEnv):
        ...
"""

# =============================================================================
# CHANGE THIS TO SWITCH CAMERA TYPE FOR ALL TASKS
# Options: "wrist" or "third"
# =============================================================================
CAMERA_TYPE = "wrist"
# =============================================================================
# This sets the following aliases (defined at bottom of file):
#   "wrist" -> DefaultCameraEnv = WristCameraEnv
#   "third" -> DefaultCameraEnv = ThirdCameraEnv
# DefaultRandomizationConfig = RandomizationConfig (unified config for both)
# =============================================================================

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs import Pose
from mani_skill.utils.visualization.misc import tile_images

from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult


@dataclass
class RandomizationConfig:
    # === Static settings (not affected by domain_randomization flag) ===
    initial_qpos_noise_scale: float = np.deg2rad(10)
    """Std of Gaussian initial-qpos noise (per joint, rad). 1σ = 10°: ~68%
    of samples within ±10°, tails extending further for robustness."""

    # === Common randomization settings (affected by domain_randomization flag) ===
    gripper_stiffness_range: Sequence[float] = (1200, 1800)
    """Per-episode gripper-joint stiffness DR. Centre 1500, ±20% spread.
    Narrower than the previous (500, 2000) — 500 was too soft to maintain
    grasp under cube-reaction force, 2000 drove the PD into deep saturation
    at every contact (cube-squirt failure mode)."""
    gripper_damping_range: Sequence[float] = (120, 180)
    """Per-episode gripper-joint damping DR. Centre 150, ±20% spread."""

    # === Arm-controller DR (matches real Feetech servo characteristics) ===
    # Centred on the 2026-05-15 step-response calibration (delay 60 ms /
    # tau 55 ms @ 30 Hz control -> delay_steps=2, lag_alpha=0.378). Ranges
    # bracket realistic per-arm / per-load variation.
    arm_stiffness_range: Sequence[float] = (900.0, 1300.0)
    """Per-episode arm-joint stiffness DR. Centre 1100, ±18% spread."""
    arm_damping_range: Sequence[float] = (80.0, 120.0)
    """Per-episode arm-joint damping DR. Centre 100, ±20% spread."""
    action_delay_steps_range: Sequence[int] = (2, 2)
    """Inclusive integer range for per-env actuator delay (control steps).
    Pinned to 2 (= 66.7 ms at 30 Hz) — closest discrete approximation of
    the measured 60 ms STS3215 dead time, well under the 70 ms hard cap.
    No delay-DR variation; the 30 Hz step granularity (33 ms) is too coarse
    to express the ~20 ms DR window without exceeding the cap."""
    lag_alpha_range: Sequence[float] = (1.0, 1.0)
    """Per-episode first-order-lag EMA mix. 1.0 = no lag (commanded target
    arrives instantly through the EMA filter). Kept off so the only response
    delay is the discrete action_delay_steps above — total delay stays
    bounded by the hard constraint without needing to mix lag in."""
    robot_color: Optional[Union[str, Sequence[float]]] = (0.03, 0.03, 0.03)
    """Robot color in RGB (0-1). Near-black (~6% albedo) — visibly black but
    with enough diffuse response that Lambertian shading + specular sheen
    reveal the arm geometry, matching real black ABS/PLA plastic which
    reflects ~5-10%. Pure (0,0,0) made the robot look emissive-black with
    no surface detail. Set to "random" for per-episode randomization."""
    randomize_lighting: bool = True
    """Whether to randomize scene lighting per episode."""
    # ══ Lighting DR — every "how bright is the env" knob lives in this block ══
    # Each episode the scene is lit by: a global ambient fill (the dominant
    # "room brightness"), a few directional lights (shading + shadows) and a
    # couple of point lights (local highlights). Every level is re-sampled per
    # episode, then all are scaled by one global exposure multiplier. The lights
    # are always WHITE — only their intensity is randomized, never their hue.
    room_brightness_range: Sequence[float] = (0.10, 0.30)
    """Per-episode ambient fill level — the global, uniform room brightness.
    Lowered so the key light dominates and each cube face shows a distinct
    Lambertian shade (matches real-world desk lighting where lit:shadow
    contrast is roughly 3–8×)."""
    exposure_range: Sequence[float] = (0.55, 1.45)
    """Per-episode global exposure multiplier applied on top of every light."""
    num_directional_lights: int = 3
    """Directional lights per sub-scene (light 0 is the brighter 'key' light)."""
    directional_key_intensity_range: Sequence[float] = (0.45, 0.85)
    """Key directional light intensity, before the exposure multiplier.
    Raised together with the lowered ambient to drive a clear per-face
    shading gradient (3–5× lit:shadow ratio)."""
    directional_fill_intensity_range: Sequence[float] = (0.05, 0.25)
    """Fill directional light intensity, before the exposure multiplier.
    Kept above zero so the shadow side never goes pure-ambient flat."""
    num_point_lights: int = 2
    """Point lights per sub-scene, at random positions above the workspace."""
    point_light_intensity_range: Sequence[float] = (0.0, 0.3)
    """Per-episode per-point-light intensity, before the exposure multiplier."""
    item_emission_range: Sequence[float] = (0.05, 0.35)
    """Per-episode emissive glow on the task cubes, as a fraction of their base
    color. A small self-lit component (domain-randomized) so the goal color
    stays readable even in the dark tail of the brightness randomization.
    0.0 = no glow (purely lit by scene lights)."""

    # === Third-person camera settings (only used by ThirdCameraEnv) ===
    third_camera_pos_noise: Sequence[float] = (0.025, 0.025, 0.025)
    """Max camera position noise from base position (x, y, z)."""
    third_camera_target_noise: float = 0.001
    """Noise scale for camera look-at target position."""
    third_camera_rot_noise: float = np.deg2rad(1)
    """Noise scale for camera view rotation."""
    third_camera_fov_noise: float = np.deg2rad(5)
    """Noise scale for camera FOV."""

    # === Wrist camera settings (only used by WristCameraEnv) ===
    # Centred to bracket realistic mount slop / hand-held re-fit error on the
    # SO101 wrist mount. Widened 2026-05 to cover the larger sim-to-real
    # extrinsic mismatch we observed at deploy.
    wrist_camera_pos_noise: Sequence[float] = (0.001, 0.001, 0.001)
    """Max position noise (x, y, z) in metres, sampled ONCE per episode and held constant. ±1 mm — tight, training-friendly range."""
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) in radians, sampled ONCE per episode and held constant. ±1° — tight, training-friendly range."""
    wrist_camera_fov_noise: float = np.deg2rad(3)
    """Per-episode FOV noise (radians) around the base 71°. ±3° spans common phone-cam / USB-cam intrinsic variation."""
    wrist_camera_roll_discrete: bool = False
    """If True, additionally jitter wrist-camera roll over the discrete set {0°, 90°, 180°, 270°} per episode. Use for a robustness-phase curriculum: trains the policy to handle a misoriented wrist camera. Continuous roll noise (wrist_camera_rot_noise[0]) is applied on top of the discrete choice."""

    # === Observation latency (camera lag) ===
    # Measured 2026-05-15: ~49.4 ms camera-only lag at 30 Hz control
    # (cmd -> first visual motion - cmd -> servo motion). Range (1, 2)
    # corresponds to ~33-66 ms in 33.3 ms slots, ±17 ms half-spread.
    obs_delay_steps_range: Sequence[int] = (1, 2)
    """Inclusive integer range for per-env observation (RGB) delay in control steps. Centre is 1 → ~33 ms at 30 Hz; bracketing the measured 49 ms ± frame quantisation."""
    max_obs_delay_steps: int = 3
    """Capacity of the per-sensor circular RGB buffer. Must be > the max of obs_delay_steps_range."""

    # === Image-pipeline domain randomization ===
    # Applied to every RGB sensor frame BEFORE the policy sees it, to bracket
    # the photometric gap between PhysX-rendered images and real USB-cam
    # output (white balance, gamma, sensor noise, hue/sat drift).
    image_noise_sigma_range: Sequence[float] = (0.0033, 0.0067)
    """Per-episode std of additive Gaussian noise on RGB in [0,1] scale. Resampled each step from the same per-env sigma."""
    image_channel_gain_range: Sequence[float] = (0.85, 1.15)
    """Per-episode scalar luminance gain applied equally to R, G, B. Models exposure/brightness drift between cameras (no color cast)."""
    image_gamma_range: Sequence[float] = (0.8, 1.2)
    """Per-episode gamma exponent applied to pixel values in [0, 1]. <1 lightens, >1 darkens."""
    image_jpeg_quality_range: Sequence[float] = (50, 95)
    """Per-episode JPEG quality (used by the image-pipeline DR wrapper when JPEG roundtripping is enabled). NB: actual JPEG roundtrip is not yet wired into _apply_image_pipeline_dr because it requires a CPU bounce; left here as a hook for a future wrapper."""
    image_jpeg_probability: float = 0.2
    """Probability per episode that JPEG roundtripping is applied. Bracket of common deploy-side stream compression. Currently informational (see image_jpeg_quality_range note)."""
    image_hue_shift_deg: float = 0.0
    """Half-range of per-episode hue shift in degrees (±). Disabled (0.0) — colour randomization is restricted to the B/W spectrum, only luminance varies."""
    image_saturation_range: Sequence[float] = (1.0, 1.0)
    """Per-episode saturation scale in HSV. Pinned to 1.0 — colour randomization is restricted to the B/W spectrum, scene saturation is preserved."""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class BaseRandomEnv(BaseEnv):
    """Base environment with domain randomization.

    Handles:
    - Gripper stiffness/damping randomization
    - Lighting randomization
    - Robot color randomization

    Subclasses (ThirdCameraEnv, WristCameraEnv) handle camera-specific logic.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        domain_randomization: bool = True,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization

        # Parse config
        self.domain_randomization_config = RandomizationConfig()
        if isinstance(domain_randomization_config, dict):
            merged_config = self.domain_randomization_config.dict()
            common.dict_merge(merged_config, domain_randomization_config)
            for key, value in merged_config.items():
                if hasattr(self.domain_randomization_config, key):
                    setattr(self.domain_randomization_config, key, value)
        elif isinstance(domain_randomization_config, RandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        super().__init__(*args, **kwargs)


    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=300, control_freq=30)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100)

    def _load_lighting(self, options: dict):
        """Build per-sub-scene lighting: ambient + several directional lights +
        point lights. Light directions/positions are created here; intensities,
        colors and the ambient level are (re)sampled per episode in
        _randomize_lighting so each of the parallel envs — and each episode —
        sees different illumination (heavy brightness/illumination DR for
        sim2real)."""
        cfg = self.domain_randomization_config
        randomize = self.domain_randomization and cfg.randomize_lighting

        # Light component handles, indexed by sub-scene, for per-episode updates.
        self._dir_lights: list[list] = []
        self._point_lights: list[list] = []

        for i, sub_scene in enumerate(self.scene.sub_scenes):
            rng = self._batched_episode_rng[i]
            dir_lights, point_lights = [], []

            for j in range(cfg.num_directional_lights):
                if randomize:
                    direction = rng.uniform(-1.0, 1.0, size=(3,))
                    direction[2] = -abs(direction[2]) - 0.2  # always shine downward-ish
                else:
                    direction = np.array([1.0, 1.0, -1.0]) if j == 0 else np.array([0.0, 0.0, -1.0])
                dir_lights.append(self._add_directional_light(sub_scene, direction, [0.5, 0.5, 0.5]))

            for j in range(cfg.num_point_lights if randomize else 0):
                pos = rng.uniform([-0.2, -0.4, 0.25], [0.6, 0.4, 0.75])
                point_lights.append(self._add_point_light(sub_scene, pos, [0.0, 0.0, 0.0]))

            self._dir_lights.append(dir_lights)
            self._point_lights.append(point_lights)

        # Apply the initial intensities / colors / ambient to every sub-scene.
        self._randomize_lighting(torch.arange(len(self.scene.sub_scenes)))

    @staticmethod
    def _add_directional_light(sub_scene, direction, color):
        """Add a directional light to a single sub-scene, return its component.

        Mirrors ManiSkillScene.add_directional_light but keeps the handle so the
        light can be re-randomized per episode."""
        entity = sapien.Entity()
        entity.name = "directional_light"
        light = sapien.render.RenderDirectionalLightComponent()
        entity.add_component(light)
        light.color = list(color)
        light.shadow = False
        light.pose = sapien.Pose([0, 0, 0], sapien.math.shortest_rotation([1, 0, 0], list(direction)))
        sub_scene.add_entity(entity)
        return light

    @staticmethod
    def _add_point_light(sub_scene, position, color):
        """Add a point light to a single sub-scene, return its component."""
        entity = sapien.Entity()
        entity.name = "point_light"
        light = sapien.render.RenderPointLightComponent()
        entity.add_component(light)
        light.color = list(color)
        light.shadow = False
        light.pose = sapien.Pose(list(position))
        sub_scene.add_entity(entity)
        return light

    def _randomize_lighting(self, env_idx: torch.Tensor):
        """Per-episode lighting randomization (white lights, intensity only):
        the global ambient room brightness, each directional light's intensity
        + direction, each point light's intensity + position — all scaled by
        one global per-episode exposure multiplier. Runs for the envs being
        reset so each episode sees fresh illumination."""
        if not hasattr(self, "_dir_lights"):
            return
        cfg = self.domain_randomization_config

        if not (self.domain_randomization and cfg.randomize_lighting):
            # Deterministic fallback (eval / DR off): fixed neutral lighting.
            for i in env_idx.tolist():
                if i >= len(self._dir_lights):
                    continue
                self.scene.sub_scenes[i].render_system.ambient_light = [0.45, 0.45, 0.45]
                for k, light in enumerate(self._dir_lights[i]):
                    g = 0.5 if k == 0 else 0.2
                    light.set_color([g, g, g])
            return

        for i in env_idx.tolist():
            if i >= len(self._dir_lights):
                continue
            rng = self._batched_episode_rng[i]
            sub_scene = self.scene.sub_scenes[i]

            # One global per-episode exposure multiplier scaling every light.
            exposure = rng.uniform(*cfg.exposure_range)

            # Ambient fill = the global, uniform room brightness (white).
            amb = float(np.clip(rng.uniform(*cfg.room_brightness_range) * exposure, 0.0, 1.0))
            sub_scene.render_system.ambient_light = [amb, amb, amb]

            # Directional lights: re-sample intensity + direction (white).
            for k, light in enumerate(self._dir_lights[i]):
                lo, hi = (cfg.directional_key_intensity_range if k == 0
                          else cfg.directional_fill_intensity_range)
                g = float(max(rng.uniform(lo, hi) * exposure, 0.0))
                light.set_color([g, g, g])
                direction = rng.uniform(-1.0, 1.0, size=(3,))
                direction[2] = -abs(direction[2]) - 0.2
                light.set_pose(sapien.Pose(
                    [0, 0, 0], sapien.math.shortest_rotation([1, 0, 0], direction.tolist())))

            # Point lights: re-sample intensity + position (white).
            for light in self._point_lights[i]:
                g = float(max(rng.uniform(*cfg.point_light_intensity_range) * exposure, 0.0))
                light.set_color([g, g, g])
                pos = rng.uniform([-0.2, -0.4, 0.25], [0.6, 0.4, 0.75])
                light.set_pose(sapien.Pose(pos.tolist()))

    def _load_camera_mount(self):
        """Create camera mount actors for pose randomization."""
        # Third-person camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # Wrist camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

    def _recolor_entities_to(self, entities, rgb):
        """Mutate every render-shape base_color on `entities` to ``rgb`` (RGB in [0,1]).

        Mirrors the pattern used by _randomize_robot_color but for non-articulated
        scene actors (table, ground, walls). Pass the result of e.g.
        ``self.table_scene.scene_objects`` to repaint the workspace to a neutral
        background color.
        """
        rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
        for obj in entities:
            sub_entities = []
            if hasattr(obj, "_objs"):
                # ManiSkill managed Actor: one underlying entity per sub-scene.
                for sub in obj._objs:
                    sub_entities.append(getattr(sub, "entity", sub))
            else:
                sub_entities.append(getattr(obj, "entity", obj))
            for entity in sub_entities:
                comp = entity.find_component_by_type(RenderBodyComponent)
                if comp is None:
                    continue
                for render_shape in comp.render_shapes:
                    for part in render_shape.parts:
                        # Replace flat base color AND clear the diffuse texture --
                        # PBR multiplies texture * base_color, so without clearing
                        # the texture (e.g. the table's wood) keeps showing.
                        part.material.set_base_color(rgba)
                        if hasattr(part.material, "set_base_color_texture"):
                            try:
                                part.material.set_base_color_texture(None)
                            except Exception:
                                pass
                        if hasattr(part.material, "set_diffuse_texture"):
                            try:
                                part.material.set_diffuse_texture(None)
                            except Exception:
                                pass

    def _randomize_robot_color(self):
        """Apply robot color randomization if configured."""
        if self.domain_randomization_config.robot_color is None:
            return

        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                render_body_component: RenderBodyComponent = obj.entity.find_component_by_type(
                    RenderBodyComponent
                )
                if render_body_component is None:
                    continue

                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if (
                            self.domain_randomization
                            and self.domain_randomization_config.robot_color == "random"
                        ):
                            color = self._batched_episode_rng[i].uniform(0.0, 1.0, size=(3,)).tolist()
                        else:
                            color = list(self.domain_randomization_config.robot_color)
                        part.material.set_base_color(color + [1])

    def _randomize_gripper_speed(self, env_idx: torch.Tensor):
        """Randomize gripper stiffness/damping per episode."""
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        # Initialize storage for privileged observations
        if not hasattr(self, "_gripper_stiffness"):
            default_stiffness = (stiff_lo + stiff_hi) / 2
            default_damping = (damp_lo + damp_hi) / 2
            self._gripper_stiffness = torch.full((self.num_envs,), default_stiffness, device=self.device)
            self._gripper_damping = torch.full((self.num_envs,), default_damping, device=self.device)

        if not self.domain_randomization:
            return
        if stiff_lo == stiff_hi and damp_lo == damp_hi:
            return

        stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
        dampings = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
        gripper_joint = self.agent.robot.joints_map["gripper"]

        for i, idx in enumerate(env_idx.tolist()):
            gripper_joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100.0)
            self._gripper_stiffness[idx] = stiffnesses[i]
            self._gripper_damping[idx] = dampings[i]

    def get_gripper_params(self) -> dict[str, torch.Tensor]:
        """Get normalized gripper parameters for privileged observations."""
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        stiff_range = stiff_hi - stiff_lo if stiff_hi != stiff_lo else 1.0
        damp_range = damp_hi - damp_lo if damp_hi != damp_lo else 1.0

        return {
            "gripper_stiffness": (self._gripper_stiffness - stiff_lo) / stiff_range,
            "gripper_damping": (self._gripper_damping - damp_lo) / damp_range,
        }

    # ── Arm controller DR ───────────────────────────────────────────────────
    # Mirrors _randomize_gripper_speed but for the five arm joints plus the
    # delay/lag parameters of the PDJointPosDelayLagController. Always called
    # at episode init: when domain_randomization is False it just lazily
    # allocates the per-env tensors with the centre (default) values so
    # downstream code can read them uniformly.
    _ARM_JOINT_NAMES = (
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex",   "wrist_roll",
    )

    def _get_arm_delay_lag_controller(self):
        """Locate the PDJointPosDelayLagController inside the agent's
        controller chain. Returns None if the current control mode does not
        use the delay/lag controller (e.g. pd_joint_pos, pd_joint_vel)."""
        from .robot.so101 import PDJointPosDelayLagController
        ctrl = getattr(self.agent, "controller", None)
        if ctrl is None:
            return None
        # mani_skill's CombinedController exposes .controllers as a dict.
        sub = getattr(ctrl, "controllers", None)
        if isinstance(sub, dict):
            for c in sub.values():
                if isinstance(c, PDJointPosDelayLagController):
                    return c
        if isinstance(ctrl, PDJointPosDelayLagController):
            return ctrl
        return None

    def _randomize_arm_controller(self, env_idx: torch.Tensor):
        """Per-episode randomization of arm-joint stiffness/damping and the
        delay/lag controller's per-env (delay_steps, lag_alpha)."""
        cfg = self.domain_randomization_config
        stiff_lo, stiff_hi = cfg.arm_stiffness_range
        damp_lo,  damp_hi  = cfg.arm_damping_range
        d_lo,     d_hi     = cfg.action_delay_steps_range
        a_lo,     a_hi     = cfg.lag_alpha_range

        # Lazy-allocate per-env storage with centre values (for both DR-off
        # and the privileged-obs path). Centres = midpoint of each range so
        # the normalised privileged obs is 0.5 when randomization is off.
        if not hasattr(self, "_arm_stiffness"):
            self._arm_stiffness = torch.full(
                (self.num_envs, len(self._ARM_JOINT_NAMES)),
                (stiff_lo + stiff_hi) / 2, device=self.device)
            self._arm_damping = torch.full(
                (self.num_envs, len(self._ARM_JOINT_NAMES)),
                (damp_lo + damp_hi) / 2, device=self.device)
            self._arm_action_delay = torch.full(
                (self.num_envs,), int(round((d_lo + d_hi) / 2)),
                dtype=torch.long, device=self.device)
            self._arm_lag_alpha = torch.full(
                (self.num_envs,), (a_lo + a_hi) / 2,
                dtype=torch.float32, device=self.device)

        controller = self._get_arm_delay_lag_controller()

        if not self.domain_randomization:
            return
        # Nothing to do if all four ranges collapse to a point.
        flat = (stiff_lo == stiff_hi and damp_lo == damp_hi
                and d_lo == d_hi and a_lo == a_hi)
        if flat:
            return

        # Sample per env in env_idx
        n = len(env_idx)
        stiffs = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
        damps  = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
        # Delay sampled as float in [d_lo, d_hi+1), then floored to int so
        # each integer in [d_lo, d_hi] is sampled with equal probability.
        delays_f = self._batched_episode_rng[env_idx].uniform(
            float(d_lo), float(d_hi) + 1.0 - 1e-6)
        delays = np.clip(np.floor(delays_f).astype(np.int64), d_lo, d_hi)
        alphas = self._batched_episode_rng[env_idx].uniform(a_lo, a_hi)

        # Write per-env stiffness/damping to each arm joint, mirroring the
        # gripper pattern. _objs[idx] gives the per-env handle for set_drive_properties.
        for j_name in self._ARM_JOINT_NAMES:
            joint = self.agent.robot.joints_map[j_name]
            for i, idx in enumerate(env_idx.tolist()):
                joint._objs[idx].set_drive_properties(
                    float(stiffs[i]), float(damps[i]), force_limit=3.0)

        idx_t = env_idx.to(self.device)
        j_arange = torch.arange(len(self._ARM_JOINT_NAMES), device=self.device)
        stiff_t = torch.as_tensor(stiffs, dtype=torch.float32, device=self.device)
        damp_t  = torch.as_tensor(damps,  dtype=torch.float32, device=self.device)
        self._arm_stiffness[idx_t.unsqueeze(-1), j_arange.unsqueeze(0)] = stiff_t.unsqueeze(-1)
        self._arm_damping[idx_t.unsqueeze(-1),  j_arange.unsqueeze(0)] = damp_t.unsqueeze(-1)
        self._arm_action_delay[idx_t] = torch.as_tensor(
            delays, dtype=torch.long, device=self.device)
        self._arm_lag_alpha[idx_t]    = torch.as_tensor(
            alphas, dtype=torch.float32, device=self.device)

        # Push the new (delay, alpha) into the controller's per-env state.
        if controller is not None:
            controller.set_per_env_dynamics(
                env_idx=idx_t,
                delay_steps=self._arm_action_delay[idx_t],
                lag_alpha=self._arm_lag_alpha[idx_t],
            )

    def get_arm_controller_params(self) -> dict[str, torch.Tensor]:
        """Normalised per-env arm-controller DR values for privileged obs.
        Returns empty dict before the first randomization call."""
        if not hasattr(self, "_arm_stiffness"):
            return {}
        cfg = self.domain_randomization_config
        stiff_lo, stiff_hi = cfg.arm_stiffness_range
        damp_lo,  damp_hi  = cfg.arm_damping_range
        d_lo,     d_hi     = cfg.action_delay_steps_range
        a_lo,     a_hi     = cfg.lag_alpha_range
        sr = stiff_hi - stiff_lo if stiff_hi != stiff_lo else 1.0
        dr = damp_hi  - damp_lo  if damp_hi  != damp_lo  else 1.0
        delay_r = float(d_hi - d_lo) if d_hi != d_lo else 1.0
        ar = a_hi - a_lo if a_hi != a_lo else 1.0
        return {
            "arm_stiffness":     (self._arm_stiffness - stiff_lo) / sr,
            "arm_damping":       (self._arm_damping  - damp_lo)  / dr,
            "arm_action_delay":  (self._arm_action_delay.float() - d_lo) / delay_r,
            "arm_lag_alpha":     (self._arm_lag_alpha - a_lo)    / ar,
        }

    # ── Camera latency (observation delay) DR ───────────────────────────────
    # Mirrors the actuator-side PDJointPosDelayLagController: each env carries
    # its own integer obs_delay_steps; rendered RGB frames are pushed into a
    # per-sensor circular buffer and the policy reads the slot that's
    # delay_steps behind the head. Centred on the 2026-05-15 camera-latency
    # measurement (~49 ms at 30 Hz).

    def _randomize_camera_latency(self, env_idx: torch.Tensor):
        """Sample per-env obs_delay_steps. Always called at episode init so
        downstream code can read self._obs_delay_per_env uniformly even when
        DR is off (then it holds the centre)."""
        cfg = self.domain_randomization_config
        d_lo, d_hi = cfg.obs_delay_steps_range
        max_d = int(cfg.max_obs_delay_steps)
        if not hasattr(self, "_obs_delay_per_env"):
            default = int(round((d_lo + d_hi) / 2))
            self._obs_delay_per_env = torch.full(
                (self.num_envs,), default,
                dtype=torch.long, device=self.device).clamp(0, max_d)
        if not self.domain_randomization or d_lo == d_hi:
            return
        # Uniform integer sample over [d_lo, d_hi] inclusive.
        delays_f = self._batched_episode_rng[env_idx].uniform(
            float(d_lo), float(d_hi) + 1.0 - 1e-6)
        delays = np.clip(np.floor(delays_f).astype(np.int64), d_lo, d_hi)
        self._obs_delay_per_env[env_idx.to(self.device)] = torch.as_tensor(
            delays, dtype=torch.long, device=self.device)

    def _apply_obs_delay(self, sensor_name: str, rgb: torch.Tensor) -> torch.Tensor:
        """Push the current frame into a per-sensor circular buffer and
        return the slot that's obs_delay_per_env behind the head.
        rgb shape: (num_envs, H, W, 3) uint8."""
        if not hasattr(self, "_obs_delay_per_env"):
            return rgb
        cfg = self.domain_randomization_config
        max_d = int(cfg.max_obs_delay_steps) + 1   # +1 for the head slot itself
        if not hasattr(self, "_obs_delay_buffers"):
            self._obs_delay_buffers = {}
            self._obs_delay_heads   = {}
        if sensor_name not in self._obs_delay_buffers:
            # Lazy alloc with the current frame replicated across all slots
            # so the first few reads don't return zeros.
            self._obs_delay_buffers[sensor_name] = rgb.unsqueeze(0).expand(
                max_d, *rgb.shape).clone()
            self._obs_delay_heads[sensor_name] = 0

        buf  = self._obs_delay_buffers[sensor_name]
        head = self._obs_delay_heads[sensor_name]
        buf[head] = rgb
        read_pos = (head - self._obs_delay_per_env) % max_d
        env_arange = torch.arange(rgb.shape[0], device=rgb.device)
        delayed = buf[read_pos, env_arange]
        self._obs_delay_heads[sensor_name] = (head + 1) % max_d
        return delayed

    # ── Image-pipeline DR ───────────────────────────────────────────────────
    # Per-episode photometric perturbations applied to every rendered RGB
    # frame. Brackets the sim/real gap from sensor noise, white balance,
    # gamma, hue, and saturation drift. JPEG roundtrip is parameterised in
    # the config but not yet wired in this method — it would need a CPU
    # bounce per frame and is better added in an obs-wrapper.

    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
        """Batched RGB->HSV in [0,1]. rgb shape: (..., 3). Returns (..., 3)."""
        r, g, b = rgb.unbind(-1)
        max_c, max_idx = rgb.max(dim=-1)
        min_c = rgb.min(dim=-1).values
        delta = max_c - min_c
        v = max_c
        s = torch.where(max_c > 0, delta / (max_c + 1e-10), torch.zeros_like(max_c))
        h_r = ((g - b) / (delta + 1e-10)) % 6.0
        h_g = ((b - r) / (delta + 1e-10)) + 2.0
        h_b = ((r - g) / (delta + 1e-10)) + 4.0
        h = torch.where(max_idx == 0, h_r,
            torch.where(max_idx == 1, h_g, h_b))
        h = torch.where(delta == 0, torch.zeros_like(h), h) / 6.0   # [0,1]
        return torch.stack([h, s, v], dim=-1)

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        """Batched HSV->RGB. hsv shape: (..., 3). Returns (..., 3) in [0,1]."""
        h, s, v = hsv.unbind(-1)
        i = (h * 6.0).floor()
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i.long() % 6
        r = torch.where(i == 0, v,
            torch.where(i == 1, q,
            torch.where(i == 2, p,
            torch.where(i == 3, p,
            torch.where(i == 4, t, v)))))
        g = torch.where(i == 0, t,
            torch.where(i == 1, v,
            torch.where(i == 2, v,
            torch.where(i == 3, q,
            torch.where(i == 4, p, p)))))
        b = torch.where(i == 0, p,
            torch.where(i == 1, p,
            torch.where(i == 2, t,
            torch.where(i == 3, v,
            torch.where(i == 4, v, q)))))
        return torch.stack([r, g, b], dim=-1)

    def _randomize_image_pipeline(self, env_idx: torch.Tensor):
        """Sample per-env image-pipeline params at episode init."""
        cfg = self.domain_randomization_config
        sigma_lo, sigma_hi = cfg.image_noise_sigma_range
        gain_lo,  gain_hi  = cfg.image_channel_gain_range
        gamma_lo, gamma_hi = cfg.image_gamma_range
        jq_lo,    jq_hi    = cfg.image_jpeg_quality_range
        sat_lo,   sat_hi   = cfg.image_saturation_range
        hue_half = float(cfg.image_hue_shift_deg)

        if not hasattr(self, "_image_noise_sigma"):
            self._image_noise_sigma   = torch.full(
                (self.num_envs,), (sigma_lo + sigma_hi) / 2,
                dtype=torch.float32, device=self.device)
            self._image_channel_gain  = torch.full(
                (self.num_envs, 3), (gain_lo + gain_hi) / 2,
                dtype=torch.float32, device=self.device)
            self._image_gamma         = torch.full(
                (self.num_envs,), (gamma_lo + gamma_hi) / 2,
                dtype=torch.float32, device=self.device)
            self._image_hue_shift     = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device)
            self._image_saturation    = torch.full(
                (self.num_envs,), (sat_lo + sat_hi) / 2,
                dtype=torch.float32, device=self.device)
            self._image_jpeg_enabled  = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            self._image_jpeg_quality  = torch.full(
                (self.num_envs,), (jq_lo + jq_hi) / 2,
                dtype=torch.float32, device=self.device)

        if not self.domain_randomization:
            return

        sig   = self._batched_episode_rng[env_idx].uniform(sigma_lo, sigma_hi)
        # B/W-only: one scalar luminance gain per env, repeated across R, G, B.
        gain_scalar = self._batched_episode_rng[env_idx].uniform(gain_lo, gain_hi)
        gain = np.tile(np.asarray(gain_scalar)[:, None], (1, 3))
        gam   = self._batched_episode_rng[env_idx].uniform(gamma_lo, gamma_hi)
        hue   = self._batched_episode_rng[env_idx].uniform(-hue_half, hue_half)
        sat   = self._batched_episode_rng[env_idx].uniform(sat_lo,   sat_hi)
        jpeg_roll = self._batched_episode_rng[env_idx].rand()
        jpeg_q    = self._batched_episode_rng[env_idx].uniform(jq_lo, jq_hi)
        jpeg_on   = jpeg_roll < float(cfg.image_jpeg_probability)

        idx_t = env_idx.to(self.device)
        def _t(v, dtype=torch.float32):
            return torch.as_tensor(v, dtype=dtype, device=self.device)
        self._image_noise_sigma[idx_t]  = _t(sig)
        self._image_channel_gain[idx_t] = _t(gain)
        self._image_gamma[idx_t]        = _t(gam)
        self._image_hue_shift[idx_t]    = _t(hue)
        self._image_saturation[idx_t]   = _t(sat)
        self._image_jpeg_enabled[idx_t] = _t(jpeg_on, dtype=torch.bool)
        self._image_jpeg_quality[idx_t] = _t(jpeg_q)

    def _apply_image_pipeline_dr(self, rgb: torch.Tensor) -> torch.Tensor:
        """Apply per-env channel gain, gamma, hue shift, saturation scale,
        and additive Gaussian noise to a (num_envs, H, W, 3) uint8 frame.
        Returns the same shape and dtype.

        JPEG quality randomization is intentionally NOT applied here: it
        requires a CPU bounce that would dominate step time at large
        batch sizes. The per-env _image_jpeg_enabled / _image_jpeg_quality
        tensors are still populated so an obs-wrapper can apply JPEG
        roundtripping at the train_squint.py level if desired."""
        if not hasattr(self, "_image_noise_sigma") or not self.domain_randomization:
            return rgb

        # uint8 -> float32 in [0, 1]; broadcast shape (num_envs, 1, 1, *)
        x = rgb.float() / 255.0

        gain = self._image_channel_gain.view(-1, 1, 1, 3)
        x = x * gain

        gamma = self._image_gamma.view(-1, 1, 1, 1)
        x = x.clamp(min=1e-6).pow(gamma)

        x = x.clamp(0.0, 1.0)
        hsv = self._rgb_to_hsv(x)
        hue_shift = (self._image_hue_shift / 360.0).view(-1, 1, 1)   # ([0,1] fraction)
        sat_scale = self._image_saturation.view(-1, 1, 1)
        hsv = torch.stack([
            (hsv[..., 0] + hue_shift) % 1.0,
            (hsv[..., 1] * sat_scale).clamp(0.0, 1.0),
            hsv[..., 2],
        ], dim=-1)
        x = self._hsv_to_rgb(hsv)

        sigma = self._image_noise_sigma.view(-1, 1, 1, 1)
        x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)

        return (x * 255.0).round().to(torch.uint8)

    def get_camera_dr_params(self) -> dict[str, torch.Tensor]:
        """Normalised per-env camera DR values for privileged observations."""
        if not hasattr(self, "_image_noise_sigma"):
            return {}
        cfg = self.domain_randomization_config
        def _norm(t, lo, hi):
            r = hi - lo if hi != lo else 1.0
            return (t - lo) / r
        return {
            "obs_delay":      _norm(self._obs_delay_per_env.float(),
                                    *cfg.obs_delay_steps_range),
            "image_noise":    _norm(self._image_noise_sigma,
                                    *cfg.image_noise_sigma_range),
            "image_gain":     _norm(self._image_channel_gain,
                                    *cfg.image_channel_gain_range),
            "image_gamma":    _norm(self._image_gamma,
                                    *cfg.image_gamma_range),
            "image_hue":      self._image_hue_shift / max(
                cfg.image_hue_shift_deg, 1e-6),   # [-1, 1]
            "image_sat":      _norm(self._image_saturation,
                                    *cfg.image_saturation_range),
        }

    # ── Discrete wrist-camera roll jitter (robustness curriculum) ──────────
    # Sampled per episode from {0, 1, 2, 3} → roll offset {0, π/2, π, 3π/2}.
    # Only consumed by WristCameraEnv._update_wrist_camera_pose when
    # config.wrist_camera_roll_discrete is True. Sampled unconditionally so
    # the tensor exists for the privileged-obs path.

    def _randomize_wrist_camera_roll(self, env_idx: torch.Tensor):
        cfg = self.domain_randomization_config
        if not hasattr(self, "_wrist_camera_roll_quadrant"):
            self._wrist_camera_roll_quadrant = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device)
        if not (self.domain_randomization and cfg.wrist_camera_roll_discrete):
            return
        quadrant = self._batched_episode_rng[env_idx].uniform(0.0, 4.0 - 1e-6)
        quadrant = np.floor(quadrant).astype(np.int64)
        self._wrist_camera_roll_quadrant[env_idx.to(self.device)] = torch.as_tensor(
            quadrant, dtype=torch.long, device=self.device)

    # ── Per-episode wrist-camera pos/rot offsets ───────────────────────────
    # Held constant across the episode (was per-step → ~30 Hz shake). Models
    # a static mount-offset / re-clipping error per deploy, not vibration.
    def _randomize_wrist_camera_offsets(self, env_idx: torch.Tensor):
        cfg = self.domain_randomization_config
        if not hasattr(self, "_wrist_camera_dr_offsets"):
            self._wrist_camera_dr_offsets = torch.zeros(
                self.num_envs, 6, dtype=torch.float32, device=self.device)
        if not self.domain_randomization:
            return
        pos_n = cfg.wrist_camera_pos_noise
        rot_n = cfg.wrist_camera_rot_noise
        rand = 2.0 * torch.rand(len(env_idx), 6, device=self.device) - 1.0
        scales = torch.tensor(
            [pos_n[0], pos_n[1], pos_n[2], rot_n[0], rot_n[1], rot_n[2]],
            dtype=torch.float32, device=self.device)
        self._wrist_camera_dr_offsets[env_idx.to(self.device)] = rand * scales

    # ── Obs hook: apply latency + image-pipeline DR to every RGB sensor ─────
    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        sensor_obs = super()._get_obs_sensor_data(apply_texture_transforms)
        for name, data in sensor_obs.items():
            if not isinstance(data, dict) or "rgb" not in data:
                continue
            rgb = data["rgb"]
            # Order matters: delay BEFORE image DR so a stale frame still
            # carries its own per-step noise (mirrors a real camera, where
            # sensor noise is fresh each frame even when the frame is late).
            rgb = self._apply_obs_delay(name, rgb)
            rgb = self._apply_image_pipeline_dr(rgb)
            data["rgb"] = rgb
        return sensor_obs

    def render_all(self):
        """Renders all human render cameras and sensors together, excluding segmentation."""

        images = []
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        render_images = self.scene.get_human_render_camera_images()
        sensor_images = self.get_sensor_images()

        # Render sensor first and then human renders
        for image in sensor_images.values():
            for key, img in image.items():
                # Skip segmentation images
                if "segmentation" not in key:
                    images.append(img)
        for image in render_images.values():
            images.append(image)

        return tile_images(images)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Base episode initialization. Subclasses should call super() first."""
        self._randomize_gripper_speed(env_idx)
        self._randomize_arm_controller(env_idx)
        self._randomize_camera_latency(env_idx)
        self._randomize_image_pipeline(env_idx)
        self._randomize_wrist_camera_roll(env_idx)
        self._randomize_wrist_camera_offsets(env_idx)
        self._randomize_lighting(env_idx)


class ThirdCameraEnv(BaseRandomEnv):
    """Environment with third-person camera and every-step pose randomization.

    Camera pose is randomized at every control step when domain_randomization=True.
    """

    # Default camera position and target
    DEFAULT_CAMERA_POS = [0.6, 0.3, 0.3]
    DEFAULT_CAMERA_TARGET = [0.3, 0, 0.05]
    DEFAULT_CAMERA_FOV = np.deg2rad(60)  # 60 degrees

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        **kwargs,
    ):
        self.base_camera_settings = dict(
            pos=self.DEFAULT_CAMERA_POS,
            target=self.DEFAULT_CAMERA_TARGET,
        )

        super().__init__(*args, domain_randomization_config=domain_randomization_config, **kwargs)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config

        # FOV randomization
        if self.domain_randomization and config.third_camera_fov_noise > 0:
            fov_noise = config.third_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
        else:
            fov_noise = 0

        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.DEFAULT_CAMERA_FOV + fov_noise,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            )
        ]

    def sample_camera_poses(self, n: int):
        """Sample randomized camera poses."""
        from mani_skill.utils.structs import Pose

        config = self.domain_randomization_config

        if not self.domain_randomization:
            # Return static pose
            static_pose = sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )
            # raw_pose may have shape [1, 1, 7] or [1, 7], squeeze to [7] then expand to [n, 7]
            pose_tensor = static_pose.raw_pose.squeeze()
            return Pose.create(pose_tensor.unsqueeze(0).expand(n, -1))

        # Convert to tensors if needed
        pos = common.to_tensor(self.base_camera_settings["pos"], device=self.device)
        target = common.to_tensor(self.base_camera_settings["target"], device=self.device)
        max_offset = common.to_tensor(config.third_camera_pos_noise, device=self.device)

        # Sample random eye positions
        eyes = randomization.camera.make_camera_rectangular_prism(
            n,
            scale=max_offset,
            center=pos,
            theta=0,
            device=self.device,
        )

        # Sample poses with noise
        poses = randomization.camera.noised_look_at(
            eyes,
            target=target,
            look_at_noise=config.third_camera_target_noise,
            view_axis_rot_noise=config.third_camera_rot_noise,
            device=self.device,
        )

        return poses

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized camera pose."""
        super()._initialize_episode(env_idx, options)
        self.camera_mount.set_pose(self.sample_camera_poses(n=len(env_idx)))

    def _before_control_step(self):
        """Randomize camera pose every step."""
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()



class WristCameraEnv(BaseRandomEnv):
    """Environment with wrist camera that follows gripper with randomization.

    Camera is mounted relative to gripper_link and follows gripper movement.
    Position and rotation offsets are randomized every step when domain_randomization=True.
    """

    # Base pose relative to gripper_link.
    WRIST_CAMERA_BASE_POS = (-0.0049, 0.0498, -0.0591)
    WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))  # radians (roll, pitch, yaw)
    WRIST_CAMERA_FOV = np.deg2rad(71)  # 71 degrees

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        **kwargs,
    ):
        super().__init__(*args, domain_randomization_config=domain_randomization_config, **kwargs)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config

        # FOV noise (randomized per-env at initialization)
        if self.domain_randomization and config.wrist_camera_fov_noise > 0:
            fov_noise = config.wrist_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
        else:
            fov_noise = 0

        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.WRIST_CAMERA_FOV + fov_noise,
                near=0.01,
                far=100,
                mount=self.wrist_camera_mount,
            )
        ]

    def _update_wrist_camera_pose(self):
        """Update wrist camera mount to follow gripper with random offsets."""
        config = self.domain_randomization_config
        gripper_pose = self.agent.robot.links_map["gripper_link"].pose

        base_x, base_y, base_z = self.WRIST_CAMERA_BASE_POS
        base_roll, base_pitch, base_yaw = self.WRIST_CAMERA_BASE_ROT_RAD

        if self.domain_randomization and hasattr(self, "_wrist_camera_dr_offsets"):
            # Per-episode offsets (sampled at reset, held constant for the
            # episode) — replaces the previous per-step resampling that
            # produced visible ~30 Hz camera shake.
            offsets = self._wrist_camera_dr_offsets
            dx = offsets[:, 0]
            dy = offsets[:, 1]
            dz = offsets[:, 2]
            d_roll = offsets[:, 3]
            d_pitch = offsets[:, 4]
            d_yaw = offsets[:, 5]
        else:
            dx = dy = dz = torch.zeros(self.num_envs, device=self.device)
            d_roll = d_pitch = d_yaw = torch.zeros(self.num_envs, device=self.device)

        # Optional discrete roll jitter over {0°, 90°, 180°, 270°} for a
        # robustness-phase curriculum. Sampled once per episode, applied on
        # top of the continuous per-step rotation noise.
        if (self.domain_randomization
                and config.wrist_camera_roll_discrete
                and hasattr(self, "_wrist_camera_roll_quadrant")):
            d_roll = d_roll + self._wrist_camera_roll_quadrant.float() * (np.pi / 2)

        # Final position and rotation
        px, py, pz = base_x + dx, base_y + dy, base_z + dz
        roll_rad, pitch_rad, yaw_rad = base_roll + d_roll, base_pitch + d_pitch, base_yaw + d_yaw

        # Convert euler to quaternion (batched)
        cj, sj = torch.cos(pitch_rad / 2), torch.sin(pitch_rad / 2)
        ck, sk = torch.cos(yaw_rad / 2), torch.sin(yaw_rad / 2)
        ci, si = torch.cos(roll_rad / 2), torch.sin(roll_rad / 2)

        q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk

        qw = q_py_w * ci - q_py_x * si
        qx = q_py_w * si + q_py_x * ci
        qy = q_py_y * ci + q_py_z * si
        qz = q_py_z * ci - q_py_y * si

        p = torch.stack([px, py, pz], dim=-1)
        q = torch.stack([qw, qx, qy, qz], dim=-1)

        local_offset = Pose.create_from_pq(p=p, q=q)
        self.wrist_camera_mount.set_pose(gripper_pose * local_offset)

    def reset(self, *args, **kwargs):
        """Reset and sync wrist camera for correct first frame."""
        obs, info = super().reset(*args, **kwargs)
        # Sync wrist camera pose once at reset for correct first frame
        # Parent reset ends with _gpu_apply_all, so we need fetch first
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()  # Complete the cycle
        return obs, info

    def _after_control_step(self):
        """Update wrist camera pose after physics step."""
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


# =============================================================================
# Default aliases based on CAMERA_TYPE setting at top of file
# =============================================================================
if CAMERA_TYPE == "wrist":
    DefaultCameraEnv = WristCameraEnv
elif CAMERA_TYPE == "third":
    DefaultCameraEnv = ThirdCameraEnv
else:
    raise ValueError(f"Unknown CAMERA_TYPE: {CAMERA_TYPE}. Use 'wrist' or 'third'")

DefaultRandomizationConfig = RandomizationConfig
