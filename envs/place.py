import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
from sapien.render import RenderBodyComponent
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import DefaultMaterialsConfig, GPUMemoryConfig, SimConfig
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101


# Goal-conditioned cube colors. Index in this palette is the goal_color_idx
# the policy is conditioned on (one-hot 6) and is also accepted via
# env.reset(options={"goal_color_idx": <int or 1-D tensor>}).
COLOR_PALETTE = np.array(
    [
        [187/255, 47/255,  27/255],  # 0 red
        [  6/255, 33/255, 111/255],  # 1 blue
        [ 24/255, 72/255,  30/255],  # 2 green
        [216/255, 195/255, 73/255],  # 3 yellow
        [ 80/255, 43/255,  82/255],  # 4 purple
        [216/255, 86/255,  54/255],  # 5 orange
    ],
    dtype=np.float32,
)
NUM_COLORS = len(COLOR_PALETTE)

# Neutral grey applied to the table, ground plane, and other scene actors.
SCENE_NEUTRAL_RGB = (164 / 255.0, 166 / 255.0, 170 / 255.0)


def _rgb_to_hsv_np(rgb):
    """Scalar RGB->HSV in [0,1] (rgb in [0,1]). Mirrors torch helper in
    base_random_env._rgb_to_hsv but for a single (R, G, B) triplet."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx
    d = mx - mn
    s = 0.0 if mx == 0 else d / mx
    if d == 0:
        h = 0.0
    elif mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h / 6.0) % 1.0, s, v


def _hsv_to_rgb_np(h, s, v):
    """Scalar HSV->RGB in [0,1]."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    return [
        (v, t, p, p, t, v)[i],
        (t, v, v, q, p, p)[i],
        (p, p, t, v, v, q)[i],
    ]


class FlatTableSceneBuilder(TableSceneBuilder):
    """Table with the decorative GLB visual replaced by a flat matte box.

    Geometry and pose match TableSceneBuilder so downstream code (initialize,
    robot placement, aabb) is unchanged. If ``frictions`` is provided
    (per-env array), each env gets its own table actor with its own PhysX
    material; otherwise a single shared kinematic table is built.
    """

    TABLE_HALF = (2.418 / 2, 1.209 / 2, 0.9196429 / 2)
    TABLE_CENTER_Z = 0.9196429 / 2
    INIT_POS = [-0.12, 0, -0.9196429]

    def __init__(self, env, frictions=None):
        super().__init__(env)
        # Per-env static/dynamic friction values. None = use SAPIEN default.
        self.frictions = frictions

    def _make_render_material(self):
        return sapien.render.RenderMaterial(
            base_color=[*SCENE_NEUTRAL_RGB, 1.0],
            roughness=0.85,
            metallic=0.0,
            specular=0.1,
        )

    def build(self):
        if self.frictions is None:
            # Single shared kinematic table with SAPIEN default friction.
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, self.TABLE_CENTER_Z]),
                half_size=self.TABLE_HALF,
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, self.TABLE_CENTER_Z]),
                half_size=self.TABLE_HALF,
                material=self._make_render_material(),
            )
            builder.initial_pose = sapien.Pose(
                p=self.INIT_POS, q=euler2quat(0, 0, np.pi / 2)
            )
            self.table = builder.build_kinematic(name="table-workspace")
        else:
            # Per-env kinematic table, each with its own PhysxMaterial so we
            # can randomize friction independently across parallel envs.
            tables = []
            for i in range(self.env.num_envs):
                f = float(self.frictions[i])
                phys_mat = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=f, dynamic_friction=f, restitution=0.0,
                )
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    pose=sapien.Pose(p=[0, 0, self.TABLE_CENTER_Z]),
                    half_size=self.TABLE_HALF,
                    material=phys_mat,
                )
                builder.add_box_visual(
                    pose=sapien.Pose(p=[0, 0, self.TABLE_CENTER_Z]),
                    half_size=self.TABLE_HALF,
                    material=self._make_render_material(),
                )
                builder.initial_pose = sapien.Pose(
                    p=self.INIT_POS, q=euler2quat(0, 0, np.pi / 2)
                )
                builder.set_scene_idxs([i])
                tables.append(builder.build_kinematic(name=f"table-workspace-{i}"))
            self.table = Actor.merge(tables, name="table-workspace")

        self.table_length = 2 * self.TABLE_HALF[0]
        self.table_width = 2 * self.TABLE_HALF[1]
        self.table_height = 2 * self.TABLE_HALF[2]
        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.scene_objects = [self.table, self.ground]


@dataclass
class PlaceRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Place task, extending wrist camera randomization."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (0.018 / 2, 0.022 / 2)
    # Can-specific randomization
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)
    # Bin randomization (half sizes)
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)

    # Split static / dynamic friction for the cubes. Real wood has a higher
    # static coefficient than dynamic, so resist slipping initially then
    # slide once moving. Bug-fix: the previous single `item_friction_range`
    # used the same value for both, missing this asymmetry — the cube was
    # slipping out of the fingers because dynamic == static was too low for
    # a stable grasp.
    # Split static / dynamic with real-wood asymmetry (static > dynamic) so the
    # cube resists initial slip in the fingers, then slides smoothly once moving.
    item_static_friction_range:  Sequence[float] = (1.2, 2.0)
    item_dynamic_friction_range: Sequence[float] = (0.5, 1.0)
    # Restitution for the cubes — disabled (fully inelastic, no bounce).
    item_restitution_range: Sequence[float] = (0.0, 0.0)
    # Mass range in kg. Sampled directly per env; the per-env density passed
    # to SAPIEN is then mass / volume, so the mass is hard-bounded regardless
    # of cube_half_size_range. Real measured cube weight ≈ 4.5 g, so the
    # range straddles it symmetrically.
    item_mass_range: Sequence[float] = (0.003, 0.006)
    # Friction + restitution for the bowl (was hardcoded 0.5 / 0.0).
    bowl_friction_range: Sequence[float] = (0.3, 1.0)
    bowl_restitution_range: Sequence[float] = (0.0, 0.0)  # disabled (fully inelastic)
    # Friction for the table top. Fixed at 0.5 (no randomization) so contacts
    # average a clean 0.5 against cubes and the bowl.
    table_friction_range: Sequence[float] = (0.3, 0.5)
    randomize_item_color: bool = False

    # Per-episode DR on the cube materials (goal + distractors). HSV-based so
    # the goal-conditioned policy still sees a recognisable hue — only
    # saturation and value (brightness) jitter; hue is locked.
    item_sat_jitter: float = 0.10
    """Half-range of per-episode multiplicative jitter on cube HSV saturation. ±10% by default."""
    item_value_jitter: float = 0.10
    """Half-range of per-episode multiplicative jitter on cube HSV value (brightness). ±10% by default."""
    item_roughness_range: Sequence[float] = (0.35, 0.7)
    """Per-episode cube material roughness (matte <-> slightly glossy)."""
    item_metallic_range: Sequence[float] = (0.0, 0.15)
    """Per-episode cube material metallic (kept low — painted wood is non-metallic)."""
    item_specular_range: Sequence[float] = (0.3, 0.7)
    """Per-episode cube material specular reflection strength."""

    # Per-episode DR on the bowl material. Bowl uses baked vertex colors;
    # we apply a per-episode HSV-shifted tint via base_color (PBR multiplies
    # base_color * vertex_color, so this acts as a global tint).
    bowl_hue_jitter_deg: float = 10.0
    """Half-range of per-episode bowl hue shift in degrees (±)."""
    bowl_sat_jitter: float = 0.10
    """Half-range of per-episode multiplicative jitter on bowl HSV saturation. ±10% by default."""
    bowl_value_jitter: float = 0.10
    """Half-range of per-episode multiplicative jitter on bowl HSV value. ±10% by default."""


class Place(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up an item (cube or can) and place it in a bin.

    **Randomizations:**
    - the item's xy position is randomized on top of a table
    - the item's z-axis rotation is randomized
    - the bin's xy position is randomized (non-overlapping with item)

    **Success Conditions:**
    - the item is in the bin xy range
    - the robot is not touching the item or the bin
    - the robot is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101]

    def __init__(
        self,
        *args,
        item_type="cube",
        n_distractors: int = 1,
        use_real_bowl: bool = True,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PlaceRandomizationConfig, dict
        ] = PlaceRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.25, 0],
        spawn_box_half_size=0.10,
        action_smooth_coef: float = 0.0,
        **kwargs,
    ):
        # CAPS-style action-rate penalty: -coef * ||a_t - a_{t-1}||^2 added to
        # the dense reward. Sized for 30 Hz control: coef=0.67 keeps the
        # per-second jitter cost the same as the previous 10 Hz / coef=2.0
        # tuning (penalty/sec is N_steps_per_sec * coef * <||delta||^2>).
        # _last_action is lazily initialised on first reward call when
        # num_envs/device are known.
        self.action_smooth_coef = float(action_smooth_coef)
        self._last_action = None
        self._just_reset_mask = None
        if not (0 <= n_distractors <= 4):
            # Distractors are placed face-to-face on the target's 4 cardinal
            # faces (one per face), so the geometry caps at 4.
            raise ValueError(
                f"n_distractors must be in [0, 4] (cubes share faces with the target, "
                f"which has 4 cardinal faces in xy). Got {n_distractors}."
            )
        self.item_type = item_type
        self.n_distractors = n_distractors
        self.use_real_bowl = use_real_bowl

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config
        self.domain_randomization_config = PlaceRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlaceRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PlaceRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        # Bowl mesh uses many CoACD convex hulls per env. At 2048 envs the
        # default PhysX GPU buffers overflow (contact + broad-phase pairs).
        # Bump 2x; the bowl is also re-decomposed with max_convex_hull=16
        # below so per-env pair count stays bounded.
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 20,
                max_rigid_patch_count=2 ** 19,
                found_lost_pairs_capacity=2 ** 26,
            ),
            # Default material for any actor that does not set its own:
            # 0.5 (was 0.3) so the table/ground/scene baseline matches the
            # bowl and the table-friction range upper bound.
            default_materials_config=DefaultMaterialsConfig(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        # Sample per-env table friction so each parallel env sees a different
        # surface (matches a slippery plastic table at the low end of the range).
        cfg = self.domain_randomization_config
        if self.domain_randomization:
            table_frictions = self._batched_episode_rng.uniform(
                low=cfg.table_friction_range[0],
                high=cfg.table_friction_range[1],
            )
        else:
            table_frictions = np.ones(self.num_envs) * (
                cfg.table_friction_range[0] + cfg.table_friction_range[1]
            ) / 2
        self.table_frictions = common.to_tensor(table_frictions, device=self.device)
        self.table_scene = FlatTableSceneBuilder(self, frictions=table_frictions)
        self.table_scene.build()
        # Repaint the table, ground, and any other table-scene actors to a
        # neutral grey. The cubes, bin, robot, and goal_site are built
        # afterwards and keep their own materials.
        self._recolor_entities_to(self.table_scene.scene_objects, SCENE_NEUTRAL_RGB)

        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # Default values
        # Placeholder colors for the build; the material is mutated per episode
        # in _initialize_episode based on the sampled goal_color_idx.
        colors = np.tile(COLOR_PALETTE[0], (self.num_envs, 1))  # default red
        cfg = self.domain_randomization_config
        static_frictions  = np.ones(self.num_envs) * (cfg.item_static_friction_range[0]  + cfg.item_static_friction_range[1])  / 2
        dynamic_frictions = np.ones(self.num_envs) * (cfg.item_dynamic_friction_range[0] + cfg.item_dynamic_friction_range[1]) / 2
        restitutions = np.ones(self.num_envs) * (cfg.item_restitution_range[0] + cfg.item_restitution_range[1]) / 2
        mass_mid = (cfg.item_mass_range[0] + cfg.item_mass_range[1]) / 2
        masses = np.ones(self.num_envs) * mass_mid

        if self.item_type == "cube":
            half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0],
                    high=cfg.cube_half_size_range[1],
                )
                # Cube color is now goal-conditioned: sampled from COLOR_PALETTE
                # in _initialize_episode and applied as a material mutation.
                static_frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_static_friction_range[0],
                    high=cfg.item_static_friction_range[1],
                )
                dynamic_frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_dynamic_friction_range[0],
                    high=cfg.item_dynamic_friction_range[1],
                )
                restitutions = self._batched_episode_rng.uniform(
                    low=cfg.item_restitution_range[0],
                    high=cfg.item_restitution_range[1],
                )
                masses = self._batched_episode_rng.uniform(
                    low=cfg.item_mass_range[0],
                    high=cfg.item_mass_range[1],
                )
            volumes = (2 * half_sizes) ** 3
            densities = masses / volumes
            self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
            self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)

        elif self.item_type == "can":
            colors = np.zeros((self.num_envs, 3))
            colors[:, :] = 0
            colors[:, 2] = 1 # blue
            half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_radius_range[1]
                    + self.domain_randomization_config.can_radius_range[0]
                )
                / 2
            )
            half_heights = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_height_range[1]
                    + self.domain_randomization_config.can_half_height_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_radius_range[0],
                    high=cfg.can_radius_range[1],
                )
                half_heights = self._batched_episode_rng.uniform(
                    low=cfg.can_half_height_range[0],
                    high=cfg.can_half_height_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                static_frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_static_friction_range[0],
                    high=cfg.item_static_friction_range[1],
                )
                dynamic_frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_dynamic_friction_range[0],
                    high=cfg.item_dynamic_friction_range[1],
                )
                restitutions = self._batched_episode_rng.uniform(
                    low=cfg.item_restitution_range[0],
                    high=cfg.item_restitution_range[1],
                )
                masses = self._batched_episode_rng.uniform(
                    low=cfg.item_mass_range[0],
                    high=cfg.item_mass_range[1],
                )
            volumes = np.pi * (half_radii ** 2) * (2 * half_heights)
            densities = masses / volumes
            self.item_half_radii = common.to_tensor(half_radii, device=self.device)
            self.item_half_heights = common.to_tensor(half_heights, device=self.device)
            self.item_half_sizes = self.item_half_heights
            self.item_dimensions = torch.stack([self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1)

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_static_frictions  = common.to_tensor(static_frictions,  device=self.device)
        self.item_dynamic_frictions = common.to_tensor(dynamic_frictions, device=self.device)
        self.item_restitutions = common.to_tensor(restitutions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)
        self.item_masses = common.to_tensor(masses, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=float(static_frictions[i]),
                dynamic_friction=float(dynamic_frictions[i]),
                restitution=float(restitutions[i]),
            )

            if self.item_type == "cube":
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(
                        base_color=colors[i], roughness=0.5, metallic=0.0, specular=0.5,
                    ),
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0, half_sizes[i]])  # Offset to avoid collision with bin at creation

            elif self.item_type == "can":
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=half_radii[i], half_length=half_heights[i], material=material, density=densities[i],
                    pose=cylinder_pose
                )
                builder.add_cylinder_visual(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=sapien.render.RenderMaterial(
                        base_color=colors[i], roughness=0.5, metallic=0.0, specular=0.5,
                    ),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0, half_heights[i]])  # Offset to avoid collision with bin at creation

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Build bins (per-env). Two modes:
        #   - parametric: 5-box rectangular tray, dimensions randomized via
        #     bin_half_size_*_range in PlaceRandomizationConfig.
        #   - use_real_bowl=True: one shared .obj mesh (from sam3d/) loaded
        #     per env. Convex-decomposed (CoACD) for dynamic collision. Mesh
        #     origin is bowl bottom-center, so bin pose Z = 0 puts the bowl
        #     floor flush with the table. No size randomization; half_sizes
        #     are taken from the mesh AABB so the success check still works.
        cfg = self.domain_randomization_config
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])

        if self.use_real_bowl:
            import os
            mesh_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "meshes", "bowl.obj"
            )
            # Read mesh AABB to set bin_half_sizes (used by success check).
            try:
                import open3d as _o3d
                _mesh = _o3d.io.read_triangle_mesh(mesh_path)
                _v = np.asarray(_mesh.vertices)
                _mn, _mx = _v.min(0), _v.max(0)
                self.bowl_half_x = float((_mx[0] - _mn[0]) / 2)
                self.bowl_half_y = float((_mx[1] - _mn[1]) / 2)
                self.bowl_half_z = float((_mx[2] - _mn[2]) / 2)
                self.bowl_z_floor = float(_mn[2])  # ~0 since mesh was re-origined
            except Exception as e:
                # Fallback hardcoded from earlier inspection
                self.bowl_half_x, self.bowl_half_y, self.bowl_half_z = 0.074, 0.0745, 0.0265
                self.bowl_z_floor = 0.0
            # The "thickness" semantics are kept so bin pose z = thickness/2
            # places the bowl floor on the table. The bowl wall height is
            # 2*bowl_half_z. The success criterion only uses bin_half_x/y.
            self.bin_thickness = 0.0  # bowl floor is at actor origin
            bin_half_sizes_x = np.ones(self.num_envs) * self.bowl_half_x
            bin_half_sizes_y = np.ones(self.num_envs) * self.bowl_half_y
            bin_half_sizes_z = np.ones(self.num_envs) * self.bowl_half_z
            # Reward target z: at the bowl rim. Releasing the cube right at
            # rim height lets it drop straight down into the bowl center
            # (short 5 cm fall, no rim-bounce).
            self.target_z_above_floor = 2 * self.bowl_half_z
        else:
            self.bin_thickness = 0.005
            # Parametric bin: reward target is at the bin floor (legacy).
            self.target_z_above_floor = 0.0
            # Default bin half sizes (mid-range)
            bin_half_sizes_x = np.ones(self.num_envs) * (cfg.bin_half_size_x_range[0] + cfg.bin_half_size_x_range[1]) / 2
            bin_half_sizes_y = np.ones(self.num_envs) * (cfg.bin_half_size_y_range[0] + cfg.bin_half_size_y_range[1]) / 2
            bin_half_sizes_z = np.ones(self.num_envs) * (cfg.bin_half_size_z_range[0] + cfg.bin_half_size_z_range[1]) / 2

            if self.domain_randomization:
                bin_half_sizes_x = self._batched_episode_rng.uniform(
                    low=cfg.bin_half_size_x_range[0], high=cfg.bin_half_size_x_range[1]
                )
                bin_half_sizes_y = self._batched_episode_rng.uniform(
                    low=cfg.bin_half_size_y_range[0], high=cfg.bin_half_size_y_range[1]
                )
                bin_half_sizes_z = self._batched_episode_rng.uniform(
                    low=cfg.bin_half_size_z_range[0], high=cfg.bin_half_size_z_range[1]
                )

        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack([self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1)

        # Per-env bowl friction + restitution (only consumed when use_real_bowl,
        # but kept symmetric so downstream code can read uniformly).
        if self.domain_randomization:
            bowl_frictions = self._batched_episode_rng.uniform(
                low=cfg.bowl_friction_range[0], high=cfg.bowl_friction_range[1],
            )
            bowl_restitutions = self._batched_episode_rng.uniform(
                low=cfg.bowl_restitution_range[0], high=cfg.bowl_restitution_range[1],
            )
        else:
            bowl_frictions = np.ones(self.num_envs) * (
                cfg.bowl_friction_range[0] + cfg.bowl_friction_range[1]) / 2
            bowl_restitutions = np.ones(self.num_envs) * (
                cfg.bowl_restitution_range[0] + cfg.bowl_restitution_range[1]) / 2
        self.bowl_frictions = common.to_tensor(bowl_frictions, device=self.device)
        self.bowl_restitutions = common.to_tensor(bowl_restitutions, device=self.device)

        bins = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            if self.use_real_bowl:
                # Dynamic actor with CoACD-decomposed collision and the real
                # mesh visual. Origin is at the bowl bottom-center.
                bowl_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=float(bowl_frictions[i]),
                    dynamic_friction=float(bowl_frictions[i]),
                    restitution=float(bowl_restitutions[i]),
                )
                builder.add_multiple_convex_collisions_from_file(
                    filename=mesh_path,
                    scale=(1.0, 1.0, 1.0),
                    material=bowl_material,
                    density=500.0,
                    decomposition="coacd",
                    decomposition_params=dict(threshold=0.3, max_convex_hull=8),
                )
                # Visual from .ply alongside .obj — carries baked vertex
                # colors from the original Gaussian splat's SH DC values.
                ply_path = os.path.splitext(mesh_path)[0] + ".ply"
                visual_path = ply_path if os.path.exists(ply_path) else mesh_path
                builder.add_visual_from_file(
                    filename=visual_path,
                    scale=(1.0, 1.0, 1.0),
                )
                z0 = 0.0  # floor on table
                initial_z = z0
            else:
                bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
                thickness = self.bin_thickness
                # Bin floor
                bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
                bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
                builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
                builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

                # Bin walls
                for j in [-1, 1]:
                    # Y walls
                    y = j * bin_center_half_size[1]
                    wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                    wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                    builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                    builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                    # X walls
                    x = j * bin_center_half_size[0]
                    wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                    wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                    builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                    builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                initial_z = bin_half_size[2]

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, initial_z])  # offset away from cube cluster
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)

        # Distractor cubes (cube tasks only, when n_distractors > 0): same
        # size/physics as the target. Each distractor gets a palette color sampled
        # per episode that is distinct from the goal and from every other
        # distractor. They spawn on a ring around the target in
        # _initialize_episode.
        self.distractors: list = []
        if self.item_type == "cube" and self.n_distractors > 0:
            for k in range(self.n_distractors):
                # Placeholder color (mutated per episode). Use palette index
                # k+1 so an un-randomized rollout shows distinguishable cubes.
                placeholder_color = COLOR_PALETTE[(k + 1) % NUM_COLORS]
                distractor_colors_k = np.tile(placeholder_color, (self.num_envs, 1))
                distractor_colors_k = np.concatenate(
                    [distractor_colors_k, np.ones((self.num_envs, 1))], axis=-1
                )

                per_env_actors = []
                for i in range(self.num_envs):
                    builder = self.scene.create_actor_builder()
                    material = sapien.pysapien.physx.PhysxMaterial(
                        static_friction=float(static_frictions[i]),
                        dynamic_friction=float(dynamic_frictions[i]),
                        restitution=0,
                    )
                    builder.add_box_collision(
                        half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                    )
                    builder.add_box_visual(
                        half_size=[half_sizes[i]] * 3,
                        material=sapien.render.RenderMaterial(
                            base_color=distractor_colors_k[i], roughness=0.5, metallic=0.0, specular=0.5,
                        ),
                    )
                    # Offset initial pose so distractors don't intersect each
                    # other, the target, or the bin at creation. Spread along x.
                    builder.initial_pose = sapien.Pose(
                        p=[0.25 + 0.04 * k, 0.05, half_sizes[i]]
                    )
                    builder.set_scene_idxs([i])
                    actor = builder.build(name=f"distractor_{k}-{i}")
                    per_env_actors.append(actor)
                    self.remove_from_state_dict_registry(actor)
                self.distractors.append(Actor.merge(per_env_actors, name=f"distractor_{k}"))

        self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        # Goal-color buffers (per env). _initialize_episode populates these.
        self.goal_color_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-env distractor color indices, shape (num_envs, n_distractors).
        self.distractor_color_idxs = torch.zeros(
            (self.num_envs, self.n_distractors), dtype=torch.long, device=self.device
        )

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Goal site
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _set_actor_palette_color(self, actor, env_idx, color_idxs):
        """Mutate the base_color of ``actor`` in-place, per env, to COLOR_PALETTE[idx]."""
        if actor is None:
            return
        env_idx_list = env_idx.tolist() if isinstance(env_idx, torch.Tensor) else list(env_idx)
        color_idxs_list = (
            color_idxs.tolist() if isinstance(color_idxs, torch.Tensor) else list(color_idxs)
        )
        cfg = self.domain_randomization_config
        dr = self.domain_randomization
        for k, i in enumerate(env_idx_list):
            obj = actor._objs[i]
            entity = getattr(obj, "entity", obj)  # Link wraps entity; merged Actor stores entity directly
            comp = entity.find_component_by_type(RenderBodyComponent)
            if comp is None:
                continue
            rng = self._batched_episode_rng[i]
            rgb = COLOR_PALETTE[int(color_idxs_list[k])].astype(np.float32)
            # HSV-correct per-episode jitter: lock hue (goal-color semantics)
            # and only perturb saturation and value. Keeps the goal color
            # recognisable to the goal-conditioned policy while still
            # bracketing real-camera color drift.
            if dr and (cfg.item_sat_jitter > 0 or cfg.item_value_jitter > 0):
                h, s, v = _rgb_to_hsv_np(rgb)
                s = float(np.clip(s * rng.uniform(
                    1.0 - cfg.item_sat_jitter, 1.0 + cfg.item_sat_jitter), 0.0, 1.0))
                v = float(np.clip(v * rng.uniform(
                    1.0 - cfg.item_value_jitter, 1.0 + cfg.item_value_jitter), 0.0, 1.0))
                rgb = np.array(_hsv_to_rgb_np(h, s, v), dtype=np.float32)
            rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
            # A bit of emissive glow (domain-randomized per episode) so the cube
            # color stays readable across the wide brightness DR range.
            lo, hi = cfg.item_emission_range
            emit_f = rng.uniform(lo, hi) if dr else lo
            emissive = [float(rgb[0]) * emit_f, float(rgb[1]) * emit_f, float(rgb[2]) * emit_f, 1.0]
            # Per-episode PBR render-param DR (roughness / metallic / specular).
            if dr:
                roughness = float(rng.uniform(*cfg.item_roughness_range))
                metallic = float(rng.uniform(*cfg.item_metallic_range))
                specular = float(rng.uniform(*cfg.item_specular_range))
            else:
                roughness, metallic, specular = 0.5, 0.0, 0.5
            for render_shape in comp.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(rgba)
                    part.material.set_emission(emissive)
                    part.material.set_roughness(roughness)
                    part.material.set_metallic(metallic)
                    part.material.set_specular(specular)

    def _randomize_bowl_tint(self, env_idx: torch.Tensor):
        """Per-episode HSV tint on the bowl's render material. Hue ±h°,
        saturation ±sj, value ±vj. PBR multiplies base_color * vertex_color,
        so this tints the baked .ply colors without erasing them.

        DR off: leaves base_color at (1,1,1) (no tint)."""
        if self.bin is None:
            return
        cfg = self.domain_randomization_config
        dr = self.domain_randomization
        env_idx_list = env_idx.tolist() if isinstance(env_idx, torch.Tensor) else list(env_idx)
        for i in env_idx_list:
            obj = self.bin._objs[i]
            entity = getattr(obj, "entity", obj)
            comp = entity.find_component_by_type(RenderBodyComponent)
            if comp is None:
                continue
            if dr:
                rng = self._batched_episode_rng[i]
                h_shift = rng.uniform(-cfg.bowl_hue_jitter_deg, cfg.bowl_hue_jitter_deg) / 360.0
                s_scale = rng.uniform(1.0 - cfg.bowl_sat_jitter, 1.0 + cfg.bowl_sat_jitter)
                v_scale = rng.uniform(1.0 - cfg.bowl_value_jitter, 1.0 + cfg.bowl_value_jitter)
                # Build a near-white tint: HSV(h_shift, |h_shift|*sat_intensity, v_scale).
                # The small saturation makes the tint visible as a hue cast
                # without overpowering the baked vertex colors.
                tint_sat = float(np.clip(abs(h_shift) * 6.0 * s_scale, 0.0, 0.25))
                r, g, b = _hsv_to_rgb_np((h_shift % 1.0), tint_sat, float(np.clip(v_scale, 0.0, 1.5)))
                rgba = [float(r), float(g), float(b), 1.0]
            else:
                rgba = [1.0, 1.0, 1.0, 1.0]
            for render_shape in comp.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(rgba)

    def _sample_goal_and_distractor_colors(self, env_idx: torch.Tensor, options: dict):
        """Sample goal_color_idx (honoring options) and n_distractors distractor
        colors that are all distinct from the goal and from each other.

        Returns:
            goal_idx: (b,) long tensor.
            distractor_idxs: (b, n_distractors) long tensor.
        """
        b = len(env_idx)
        # 1) goal color: from options if provided, else uniform over the palette.
        goal_override = options.get("goal_color_idx") if isinstance(options, dict) else None
        if goal_override is None:
            goal_idx = torch.randint(NUM_COLORS, (b,), device=self.device, dtype=torch.long)
        else:
            if isinstance(goal_override, (int, np.integer)):
                goal_idx = torch.full((b,), int(goal_override), device=self.device, dtype=torch.long)
            else:
                goal_idx = torch.as_tensor(goal_override, device=self.device, dtype=torch.long).reshape(-1)
                assert goal_idx.numel() == b, (
                    f"goal_color_idx must be an int or length-{b} sequence, got {tuple(goal_idx.shape)}"
                )
        # 2) distractor colors: per-env random permutation of the palette,
        # remove the goal, take the first n_distractors entries.
        if self.n_distractors == 0:
            distractor_idxs = torch.empty((b, 0), dtype=torch.long, device=self.device)
        else:
            keys = torch.rand(b, NUM_COLORS, device=self.device)
            perm = keys.argsort(dim=-1)  # (b, NUM_COLORS)
            mask = perm != goal_idx.unsqueeze(1)  # exactly one False per row
            non_goal = perm[mask].reshape(b, NUM_COLORS - 1)
            distractor_idxs = non_goal[:, : self.n_distractors]
        return goal_idx, distractor_idxs

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Mark the just-reset envs so the action-rate penalty pays zero
            # on the first step of a new episode (it'd otherwise charge the
            # magnitude of a_0 instead of a true rate ||a_t - a_{t-1}||).
            if self._just_reset_mask is not None:
                self._just_reset_mask[env_idx] = True

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # Sample positions for item and bin
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            # Item/bin radii (worst-case to be conservative).
            # When distractors are face-to-face on the target, the cluster
            # extends from the target center by (half_size + 2*half_size) =
            # 3*half_size into any occupied direction. Use that as the
            # effective item radius so the bin avoids the whole cluster.
            if self.item_type == "can":
                item_radius = self.item_half_radii.max().item() + 0.01
            else:
                hs = self.item_half_sizes.max().item()
                cluster_mult = 3.0 if self.n_distractors > 0 else 1.0
                item_radius = cluster_mult * hs + 0.01
            bin_radius = self.bin_radius.max().item() + 0.01

            # Cube spawn region (unchanged): the original spawn_box.
            cube_region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size]
            ]
            cube_sampler = randomization.UniformPlacementSampler(
                bounds=cube_region, batch_size=b, device=self.device
            )
            item_xy_offset = cube_sampler.sample(item_radius, 100)

            # Bowl spawn region: WIDER than the cube region on 3 sides — wider
            # in y on both sides AND farther in +x. Keep the near-robot edge
            # (−x, robot side) UNCHANGED so the bowl never gets closer to the
            # robot than before. ±5 cm extra past the cube box on the 3 free
            # sides gives the bowl more spatial variety without bringing it
            # into a kinematically hard-to-reach pose.
            BOWL_FAR_EXTRA   = 0.05   # +5 cm in +x (farther from robot)
            BOWL_WIDTH_EXTRA = 0.05   # ±5 cm wider in y on both sides
            bin_x_lo = -self.spawn_box_half_size                       # unchanged
            bin_x_hi =  self.spawn_box_half_size + BOWL_FAR_EXTRA
            bin_y_lo = -self.spawn_box_half_size - BOWL_WIDTH_EXTRA
            bin_y_hi =  self.spawn_box_half_size + BOWL_WIDTH_EXTRA
            bin_xy_offset = torch.zeros((b, 2), device=self.device)
            bin_xy_offset[:, 0] = torch.rand(b, device=self.device) * (bin_x_hi - bin_x_lo) + bin_x_lo
            bin_xy_offset[:, 1] = torch.rand(b, device=self.device) * (bin_y_hi - bin_y_lo) + bin_y_lo

            # Cube–bowl exclusion: cube center must be ≥ 10 cm from bowl
            # center (bowl rim is at 7.5 cm, +2.5 cm safety). Rejection loop
            # re-samples only the envs whose cube fell too close. After the
            # loop any remaining bad envs are pushed radially outward.
            BOWL_EXCLUSION = 0.10
            for _ in range(20):
                delta_xy = item_xy_offset - bin_xy_offset
                dist = torch.linalg.norm(delta_xy, dim=-1)
                bad = dist < BOWL_EXCLUSION
                if not bad.any():
                    break
                new_offset = cube_sampler.sample(item_radius, 100)
                item_xy_offset = torch.where(bad.unsqueeze(-1), new_offset, item_xy_offset)
            # Hard fix for any holdouts: push radially away from bowl to the
            # exclusion boundary (rare; shows up when the cube region is small
            # and the bowl sits near the cube region's centre).
            delta_xy = item_xy_offset - bin_xy_offset
            dist = torch.linalg.norm(delta_xy, dim=-1, keepdim=True)
            still_bad = (dist < BOWL_EXCLUSION).squeeze(-1)
            if still_bad.any():
                direction = torch.where(
                    dist > 1e-6, delta_xy / dist.clamp(min=1e-6),
                    torch.tensor([1.0, 0.0], device=self.device).expand_as(delta_xy),
                )
                pushed = bin_xy_offset + direction * BOWL_EXCLUSION
                item_xy_offset = torch.where(still_bad.unsqueeze(-1), pushed, item_xy_offset)

            # Cluster of (1 + n_distractors) cubes glued face-to-face. The
            # sampled position is the geometric center of the cluster, so the
            # goal cube can be ANY of the slots (center or any cardinal),
            # uniformly random — not always the middle one.
            cluster_xy = spawn_center[env_idx, :2] + item_xy_offset
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            n_d = len(self.distractors)

            if n_d > 0:
                # Pick which of the 4 cardinal directions are occupied (subset
                # of size n_d).
                face_keys = torch.rand(b, 4, device=self.device)
                face_perm = face_keys.argsort(dim=-1)  # (b, 4)
                cardinal_dirs = face_perm[:, :n_d]  # (b, n_d)

                # Total slots = center + occupied cardinals. Pick the goal slot
                # uniformly so the goal cube isn't always at the center.
                total_slots = n_d + 1
                goal_slot = torch.randint(total_slots, (b,), device=self.device, dtype=torch.long)

                # Per-slot xy offsets relative to cluster center.
                # slot 0 = center (offset 0); slot k+1 = 2*half_size in
                # direction theta_target + cardinal_dirs[:,k]*pi/2.
                theta_target = 2.0 * torch.atan2(qs[:, 3], qs[:, 0])  # (b,)
                distance = 2 * self.item_half_sizes[env_idx]  # (b,) face-to-face
                slot_offsets = torch.zeros(b, total_slots, 2, device=self.device)
                for k in range(n_d):
                    face_k = cardinal_dirs[:, k].float() * (math.pi / 2)
                    theta_face = theta_target + face_k
                    slot_offsets[:, k + 1, 0] = distance * torch.cos(theta_face)
                    slot_offsets[:, k + 1, 1] = distance * torch.sin(theta_face)

                arange_b = torch.arange(b, device=self.device)

                # Goal cube at chosen slot.
                item_xyz = torch.zeros((b, 3))
                item_xyz[:, :2] = cluster_xy + slot_offsets[arange_b, goal_slot]
                item_xyz[:, 2] = self.item_half_sizes[env_idx]
                self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

                # Distractors at the remaining slots, faces flush with the
                # cluster (orientation shared with the target).
                slots_all = torch.arange(total_slots, device=self.device).unsqueeze(0).expand(b, total_slots)
                mask = slots_all != goal_slot.unsqueeze(1)  # exactly one False per row
                non_goal_slots = slots_all[mask].reshape(b, n_d)
                for k, d_actor in enumerate(self.distractors):
                    slot_k = non_goal_slots[:, k]
                    d_offset = slot_offsets[arange_b, slot_k]
                    d_xyz = torch.zeros((b, 3))
                    d_xyz[:, :2] = cluster_xy + d_offset
                    d_xyz[:, 2] = self.item_half_sizes[env_idx]
                    d_actor.set_pose(Pose.create_from_pq(d_xyz, qs))
            else:
                # No distractors: goal alone at the sampled position.
                item_xyz = torch.zeros((b, 3))
                item_xyz[:, :2] = cluster_xy
                item_xyz[:, 2] = self.item_half_sizes[env_idx]
                self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            # Goal-conditioned colors: sample new indices, paint the cubes, and
            # cache the indices for the obs vector (one-hot goal color).
            if self.item_type == "cube":
                goal_idx, distractor_idxs = self._sample_goal_and_distractor_colors(env_idx, options)
                self.goal_color_idx[env_idx] = goal_idx
                self._set_actor_palette_color(self.item, env_idx, goal_idx)
                if n_d > 0:
                    self.distractor_color_idxs[env_idx] = distractor_idxs
                    for k, d_actor in enumerate(self.distractors):
                        self._set_actor_palette_color(d_actor, env_idx, distractor_idxs[:, k])

            # Set bin pose
            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2] = self.bin_thickness / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            # Per-episode bowl color tint (hue ±10°, sat ±10%, value ±10%).
            # PBR multiplies base_color × vertex_color, so this tints the
            # baked .ply colors without flattening them.
            if self.use_real_bowl:
                self._randomize_bowl_tint(env_idx)

            # Goal is above bin center (above-rim for bowl, at-floor for parametric)
            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = self.bin_thickness + self.item_half_sizes[env_idx] + self.target_z_above_floor
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        # Adding joint noise for better sim2real
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        # Goal-color conditioning (one-hot over COLOR_PALETTE). Available on all
        # cube tasks; reward / success tracking remain tied to self.item.
        if self.item_type == "cube":
            obs["goal_color"] = torch.nn.functional.one_hot(
                self.goal_color_idx, num_classes=NUM_COLORS
            ).to(qpos.dtype)
        # Bowl centre in the robot base frame (xyz). Appended last so it lands
        # at the end of the flattened state vector.
        obs["bowl_xyz_robot_frame"] = (self.agent.robot.pose.inv() * self.bin.pose).p
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp_pos,
                item_to_bin_pos=self.bin.pose.p - self.item.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bin_dimensions=self.bin_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        bin_pos[:, 2] = self.bin_thickness + self.item_half_sizes

        offset = item_pos - bin_pos
        inside_x = torch.abs(offset[:, 0]) < self.bin_half_sizes_x
        inside_y = torch.abs(offset[:, 1]) < self.bin_half_sizes_y
        is_item_above_bin = inside_x & inside_y

        item_lifted = self.item.pose.p[..., -1] >= (self.item_half_sizes + 1e-3)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bin = self.agent.is_touching(self.bin)
        robot_touching_item = self.agent.is_touching(self.item)

        success = is_item_above_bin & (~robot_touching_item) & is_robot_static & (~robot_touching_bin)

        return {
            "inside_x": inside_x,
            "inside_y": inside_y,
            "item_vel": item_vel,
            "item_lifted": item_lifted,
            "is_item_static": is_item_static,
            "success": success,
            "is_item_above_bin": is_item_above_bin,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bin": robot_touching_bin,
            "robot_touching_item": robot_touching_item,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Baseline 4398ce9 reward (hard reset). Components:
        #   reach: dense up to 2.0
        #   grasped: replace reach with 3 + place_reward
        #   above bin: 4 + place_reward + dropped + gripper_openness + static
        #   success: 9
        #   penalties: -3 bin contact, -1 cube not lifted
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        goal_xyz = bin_pos.clone()
        goal_xyz[..., 2] = self.bin_thickness + self.item_half_sizes + self.target_z_above_floor

        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + (self.bin_dimensions[:, 2:] * 2) + 0.03) - item_pos[..., 2:], dim=1
        )
        item_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        item_close_to_goal = (item_to_goal_dist_xy <= self.bin_radius)
        item_to_goal_dist_z = torch.where(item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        is_item_dropped = (~info["robot_touching_item"]).float()
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        reward[info["is_item_above_bin"]] = (4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward)[info["is_item_above_bin"]]

        reward[info["success"]] = 9

        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * (~info["item_lifted"]).float()

        # Action-rate penalty (CAPS-style). action_smooth_coef defaults to 0,
        # so this branch is a no-op unless explicitly enabled.
        if self.action_smooth_coef > 0 and torch.is_tensor(action):
            if self._last_action is None or self._last_action.shape != action.shape:
                self._last_action = torch.zeros_like(action)
                self._just_reset_mask = torch.ones(
                    action.shape[0], dtype=torch.bool, device=action.device
                )
            fresh = self._just_reset_mask
            if fresh.any():
                self._last_action[fresh] = action[fresh].detach()
                self._just_reset_mask[:] = False
            delta = action - self._last_action
            reward = reward - self.action_smooth_coef * (delta * delta).sum(dim=-1)
            self._last_action = action.detach().clone()

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101PlaceCube-v1", max_episode_steps=75)
class PlaceCube(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PlaceCan-v1", max_episode_steps=150)
class PlaceCan(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
