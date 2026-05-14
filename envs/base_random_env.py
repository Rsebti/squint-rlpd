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
    initial_qpos_noise_scale: float = 0.02
    """Noise scale for initial robot joint positions."""

    # === Common randomization settings (affected by domain_randomization flag) ===
    gripper_stiffness_range: Sequence[float] = (500, 2000)
    """Range for gripper joint stiffness randomization (per-episode)."""
    gripper_damping_range: Sequence[float] = (50, 200)
    """Range for gripper joint damping randomization (per-episode)."""
    robot_color: Optional[Union[str, Sequence[float]]] = (0.0, 0.0, 0.0)
    """Robot color in RGB (0-1). Set to "random" for per-episode randomization."""
    randomize_lighting: bool = True
    """Whether to randomize scene lighting per episode."""
    # ══ Lighting DR — every "how bright is the env" knob lives in this block ══
    # Each episode the scene is lit by: a global ambient fill (the dominant
    # "room brightness"), a few directional lights (shading + shadows) and a
    # couple of point lights (local highlights). Every level is re-sampled per
    # episode, then all are scaled by one global exposure multiplier. The lights
    # are always WHITE — only their intensity is randomized, never their hue.
    room_brightness_range: Sequence[float] = (0.25, 0.62)
    """Per-episode ambient fill level — the global, uniform room brightness."""
    exposure_range: Sequence[float] = (0.55, 1.45)
    """Per-episode global exposure multiplier applied on top of every light."""
    num_directional_lights: int = 3
    """Directional lights per sub-scene (light 0 is the brighter 'key' light)."""
    directional_key_intensity_range: Sequence[float] = (0.15, 0.45)
    """Key directional light intensity, before the exposure multiplier."""
    directional_fill_intensity_range: Sequence[float] = (0.0, 0.18)
    """Fill directional light intensity, before the exposure multiplier (can drop to ~0)."""
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
    wrist_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z) relative to gripper."""
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) in radians."""
    wrist_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV. Base FOV is 71 degrees."""

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
        return SimConfig(sim_freq=100, control_freq=10)

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
            gripper_joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100)
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

    # Base pose relative to gripper_link
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

        if self.domain_randomization:
            # Batch all random numbers into one call (6 values per env)
            rand_vals = 2 * torch.rand(self.num_envs, 6, device=self.device) - 1

            pos_offset = config.wrist_camera_pos_noise
            rot_noise = config.wrist_camera_rot_noise

            dx = pos_offset[0] * rand_vals[:, 0]
            dy = pos_offset[1] * rand_vals[:, 1]
            dz = pos_offset[2] * rand_vals[:, 2]
            d_roll = rot_noise[0] * rand_vals[:, 3]
            d_pitch = rot_noise[1] * rand_vals[:, 4]
            d_yaw = rot_noise[2] * rand_vals[:, 5]
        else:
            dx = dy = dz = torch.zeros(self.num_envs, device=self.device)
            d_roll = d_pitch = d_yaw = torch.zeros(self.num_envs, device=self.device)

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
