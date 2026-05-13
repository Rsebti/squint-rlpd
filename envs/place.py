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
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig

from .robot.so100 import SO100
from .robot.so101 import SO101


# Goal-conditioned cube colors. Index in this palette is the goal_color_idx
# the policy is conditioned on (one-hot 6) and is also accepted via
# env.reset(options={"goal_color_idx": <int or 1-D tensor>}).
COLOR_PALETTE = np.array(
    [
        [1.00, 0.00, 0.00],  # 0 red
        [0.00, 0.00, 1.00],  # 1 blue
        [0.00, 1.00, 0.00],  # 2 green
        [1.00, 1.00, 0.00],  # 3 yellow
        [0.60, 0.00, 0.80],  # 4 purple
        [1.00, 0.50, 0.00],  # 5 orange
    ],
    dtype=np.float32,
)
NUM_COLORS = len(COLOR_PALETTE)

# Background color applied to the table, ground plane, and other non-foreground
# scene actors. Matches the #B8ADA9 overlay PNG so SAPIEN-rendered frames blend
# with the greenscreen background that the policy was trained on.
SCENE_NEUTRAL_RGB = (0xB8 / 255.0, 0xAD / 255.0, 0xA9 / 255.0)


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

    # Friction for the cubes (painted wood — can be quite slippery).
    item_friction_range: Sequence[float] = (0.05, 0.6)
    item_density_range: Sequence[float] = (600, 1000)
    # Friction for the table top (plastic in the real setup → low friction).
    table_friction_range: Sequence[float] = (0.05, 0.4)
    randomize_item_color: bool = False


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
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PlaceRandomizationConfig, dict
        ] = PlaceRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.3 / 2,
        **kwargs,
    ):
        if not (0 <= n_distractors <= 4):
            # Distractors are placed face-to-face on the target's 4 cardinal
            # faces (one per face), so the geometry caps at 4.
            raise ValueError(
                f"n_distractors must be in [0, 4] (cubes share faces with the target, "
                f"which has 4 cardinal faces in xy). Got {n_distractors}."
            )
        self.item_type = item_type
        self.n_distractors = n_distractors

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
        # Repaint the table, ground, and any other table-scene actors to the
        # neutral background color so the SAPIEN-rendered scene matches the
        # greenscreen overlay even before the overlay is applied. The cubes,
        # bin, robot, and goal_site are built afterwards and keep their own
        # materials.
        self._recolor_entities_to(self.table_scene.scene_objects, SCENE_NEUTRAL_RGB)

        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # Default values
        # Placeholder colors for the build; the material is mutated per episode
        # in _initialize_episode based on the sampled goal_color_idx.
        colors = np.tile(COLOR_PALETTE[0], (self.num_envs, 1))  # default red
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

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
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
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
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_radii = common.to_tensor(half_radii, device=self.device)
            self.item_half_heights = common.to_tensor(half_heights, device=self.device)
            self.item_half_sizes = self.item_half_heights
            self.item_dimensions = torch.stack([self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1)

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )

            if self.item_type == "cube":
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
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
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                    pose=cylinder_pose
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0, half_heights[i]])  # Offset to avoid collision with bin at creation

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Build bins (per-env for domain randomization)
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness

        # Default bin half sizes (mid-range)
        cfg = self.domain_randomization_config
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

        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

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

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, bin_half_size[2]])  # Offset to avoid collision with item at creation
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
                    friction = frictions[i]
                    material = sapien.pysapien.physx.PhysxMaterial(
                        static_friction=friction, dynamic_friction=friction, restitution=0
                    )
                    builder.add_box_collision(
                        half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                    )
                    builder.add_box_visual(
                        half_size=[half_sizes[i]] * 3,
                        material=sapien.render.RenderMaterial(base_color=distractor_colors_k[i]),
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

        # Set up greenscreening - keep robot, item, bin, and distractor visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)
            for d in self.distractors:
                self.remove_object_from_greenscreen(d)

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
        for k, i in enumerate(env_idx_list):
            obj = actor._objs[i]
            entity = getattr(obj, "entity", obj)  # Link wraps entity; merged Actor stores entity directly
            comp = entity.find_component_by_type(RenderBodyComponent)
            if comp is None:
                continue
            rgb = COLOR_PALETTE[int(color_idxs_list[k])]
            rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
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

            # Use placement sampler for non-overlapping positions
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size]
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            # Item/bin radius (use max for conservative placement).
            # When distractors are face-to-face on the target, the cluster
            # extends from the target center by (half_size + 2*half_size) =
            # 3*half_size into any occupied direction. Use that as the effective
            # radius so the bin sampler avoids the whole cluster, not just the
            # target cube.
            if self.item_type == "can":
                item_radius = self.item_half_radii.max().item() + 0.01
            else:
                hs = self.item_half_sizes.max().item()
                cluster_mult = 3.0 if self.n_distractors > 0 else 1.0
                item_radius = cluster_mult * hs + 0.01
            bin_radius = self.bin_radius.max().item() + 0.01

            item_xy_offset = sampler.sample(item_radius, 100)
            bin_xy_offset = sampler.sample(bin_radius, 100, verbose=False)

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

            # Goal is above bin center
            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = self.bin_thickness + self.item_half_sizes[env_idx]
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

        # Contact checks
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
        # Reaching reward
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        # Complex place reward
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        goal_xyz = bin_pos.clone()
        goal_xyz[..., 2] = self.bin_thickness + self.item_half_sizes

        # Overall distance reward
        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        # XY and Z distance with far/close logic
        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        # Far: target is above bin (encourages lifting before placing)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + (self.bin_dimensions[:, 2:] * 2) + 0.03) - item_pos[..., 2:], dim=1
        )
        # Close: target is final position
        item_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        item_close_to_goal = (item_to_goal_dist_xy <= self.bin_radius)
        item_to_goal_dist_z = torch.where(item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        # Ungrasp reward (inverted from Reach's close gripper)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasped: 3 + place_reward
        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        # Above bin: 3 + place_reward + gripper_openness
        is_item_dropped = (~info["robot_touching_item"]).float()
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1) 
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        reward[info["is_item_above_bin"]] = (4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward)[info["is_item_above_bin"]]


        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * (~info["item_lifted"]).float()  # Encourage picking item fast


        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101PlaceCube-v1", max_episode_steps=75)
class PlaceCube(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PlaceCan-v1", max_episode_steps=50)
class PlaceCan(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
