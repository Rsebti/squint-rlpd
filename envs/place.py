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
        # Tuned 2026-05-19 by interactive sweep (examples/sweep_palette_*.py)
        # against the locked scene (exposure=1.80, table_albedo=0.85,
        # bowl_emission=0.80). Starts from the Friday 2026-05-15 commit
        # 2783d83 palette (hand-picked vibrant) and lifts saturation toward
        # 1.0 (sat_lift=0.85). Per-color V targets keep red/orange/purple
        # deep in the rendered frame while bringing originally-dark blue and
        # green up enough to read against the table. Orange hue is shifted
        # from Friday's H=12° to H=16° (red-orange / tomato) — final pick
        # after a separate fine-tune sweep to make it more yellow-red while
        # staying distinguishable from the red cube at H=7.5°.
        #                                       (R,   G,   B)    S     V
        [128/255,  18/255,   3/255],  # 0 red     (128, 18,   3)  0.98  0.50
        [  1/255,  37/255, 140/255],  # 1 blue    (  1, 37, 140)  0.99  0.55
        [  6/255, 115/255,  19/255],  # 2 green   (  6,115,  19)  0.95  0.45
        [217/255, 187/255,  11/255],  # 3 yellow  (217,187,  11)  0.95  0.85
        [ 85/255,   7/255,  89/255],  # 4 purple  ( 85,  7,  89)  0.92  0.35
        [166/255,  49/255,   6/255],  # 5 orange  (166, 49,   6)  0.96  0.65   hue=16°
    ],
    dtype=np.float32,
)
NUM_COLORS = len(COLOR_PALETTE)

# Neutral grey applied to the table, ground plane, and other scene actors.
SCENE_NEUTRAL_RGB = (200 / 255.0, 200 / 255.0, 200 / 255.0)


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
    robot_qpos_noise_std: float = np.deg2rad(2)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (0.018 / 2, 0.022 / 2)
    # Can-specific randomization
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)
    # Bin randomization (half sizes)
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)

    # Single friction range for the cubes — same value used for both static
    # and dynamic friction. All friction values are strictly below 1.
    item_friction_range: Sequence[float] = (0.2, 0.99)
    # Restitution for the cubes — disabled (fully inelastic, no bounce).
    item_restitution_range: Sequence[float] = (0.0, 0.0)
    # Mass range in kg. Sampled directly per env; the per-env density passed
    # to SAPIEN is then mass / volume, so the mass is hard-bounded regardless
    # of cube_half_size_range. Real measured cube weight ≈ 4.5 g, so the
    # range straddles it symmetrically.
    item_mass_range: Sequence[float] = (0.003, 0.006)
    # Friction + restitution for the bowl (was hardcoded 0.5 / 0.0).
    bowl_friction_range: Sequence[float] = (0.3, 0.99)
    bowl_restitution_range: Sequence[float] = (0.0, 0.0)  # disabled (fully inelastic)
    # Friction for the table top. Fixed at 0.5 (no randomization) so contacts
    # average a clean 0.5 against cubes and the bowl.
    table_friction_range: Sequence[float] = (0.3, 0.5)
    randomize_item_color: bool = False

    # Per-episode DR on the cube materials (goal + distractors). HSV-based so
    # the goal-conditioned policy still sees a recognisable hue — only
    # saturation and value (brightness) jitter; hue is locked.
    item_sat_jitter: float = 0.05
    """Half-range of per-episode multiplicative jitter on cube HSV saturation. ±5% (tightened from ±10% — bigger jitters knock S down to ~0.65 on bad rolls, which is the threshold where the cube starts looking pastel/white-mixed)."""
    item_value_jitter: float = 0.10
    """Half-range of per-episode multiplicative jitter on cube HSV value (brightness). ±10% by default."""
    item_roughness_range: Sequence[float] = (0.7, 0.95)  # matte (painted-wood look)
    """Per-episode cube material roughness (matte <-> slightly glossy)."""
    item_metallic_range: Sequence[float] = (0.0, 0.15)
    """Per-episode cube material metallic (kept low — painted wood is non-metallic)."""
    item_specular_range: Sequence[float] = (0.0, 0.1)  # was (0.1, 0.4) — specular is the dominant "looks white" source: it's a white reflection layer on top of the diffuse base, and with the matte roughness it spreads across the cube face washing the colour out. Capped near zero now.
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
        skip_bowl: bool = False,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PlaceRandomizationConfig, dict
        ] = PlaceRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.25, 0],
        spawn_box_half_size=0.10,
        spawn_arc_center: Sequence[float] = (0.10, 0.0),
        spawn_arc_radius: float = 0.30,
        spawn_y_clip: Optional[float] = 0.22,
        spawn_x_min: Optional[float] = 0.12,
        spawn_stratified: bool = True,
        visualize_spawn_area: bool = False,
        visualize_robot_frame: bool = False,
        action_smooth_coef: float = 0.0,
        strong_grasp_coef: float = 0.5,
        pick_only_reward: bool = False,
        pick_stable_speed_threshold: float = 0.01,
        pick_stable_duration_s: float = 1.0,
        pick_side_approach: bool = False,
        pick_side_approach_open_coef: float = 0.3,
        drop_penalty_coef: float = 0.0,
        split_only_reward: bool = False,
        split_target_gap: float = 0.03,
        split_sep_coef: float = 1.0,
        split_table_penalty_coef: float = 0.5,
        split_bowl_penalty_coef: float = 3.0,
        split_stable_speed_threshold: float = 0.02,
        split_stable_duration_s: float = 0.5,
        split_hover_after_separate: bool = False,
        split_hover_z: float = 0.05,
        split_hover_coef: float = 1.0,
        split_hover_tol: float = 0.015,
        split_color_hierarchy: bool = False,
        split_far_penalty_coef: float = 0.0,
        split_far_penalty_dist: float = 0.15,
        **kwargs,
    ):
        # CAPS-style action-rate penalty: -coef * ||a_t - a_{t-1}||^2 added to
        # the dense reward. Sized for 30 Hz control: coef=0.67 keeps the
        # per-second jitter cost the same as the previous 10 Hz / coef=2.0
        # tuning (penalty/sec is N_steps_per_sec * coef * <||delta||^2>).
        # _last_action is lazily initialised on first reward call when
        # num_envs/device are known.
        self.action_smooth_coef = float(action_smooth_coef)
        # Sim2real strong-grasp bonus: rewards target_qpos[gripper] near min
        # (fully-closed) while grasped, so the policy learns to clamp HARD at
        # grasp formation. Gated by is_item_grasped AND ~xy_close_to_bowl (off
        # once the cube is over the bowl, to allow release).
        self.strong_grasp_coef = float(strong_grasp_coef)
        # Pick-only reward mode: when True, the env trains the policy to grasp
        # the cube and hold it nearly stationary for `pick_stable_duration_s`
        # seconds. Success terminates the episode early. The full place reward
        # (z lift / xy-to-bowl / above-bin / release) is skipped entirely.
        self.pick_only_reward = bool(pick_only_reward)
        self.pick_stable_speed_threshold = float(pick_stable_speed_threshold)
        self.pick_stable_duration_s = float(pick_stable_duration_s)
        # Side-approach curriculum (pick-only mode only). Forces the policy to
        # land the FIXED finger on the cube first, keeping the moving finger
        # fully open during approach. Reduces the failure mode where the policy
        # arrives top-down with the moving finger pre-closed (works in sim, the
        # cube physically holds the finger; fails on real where the finger taps
        # the cube top). Once the fixed finger has touched, the reward switches
        # to the normal grasp ladder for the rest of the episode.
        self.pick_side_approach = bool(pick_side_approach)
        self.pick_side_approach_open_coef = float(pick_side_approach_open_coef)
        # Per-env sticky flag: True once fixed finger has touched the cube in
        # the current episode. Reset on _initialize_episode. Lazy init on the
        # first reward call (when device/num_envs are known).
        self._fixed_finger_touched = None
        # Drop-penalty (pick-only mode only): -coef applied on each
        # grasped→not-grasped transition. Pushes the policy to one-shot the
        # grasp instead of fumbling and re-grasping. Disabled by default
        # (coef=0); set to e.g. 3.0 to enable.
        self.drop_penalty_coef = float(drop_penalty_coef)
        # Per-env state for the drop-penalty: previous step's is_item_grasped.
        # Lazy init on first reward call; reset to False on _initialize_episode.
        self._prev_is_grasped = None
        # Per-env consecutive-stable counter (pick-only mode). Init on _load_scene.
        self._grasp_slow_counter = None
        # Split-only reward mode: train the policy to push the two cubes apart
        # (no grasping) until the surface gap between them reaches
        # split_target_gap, then hold both cubes still. Success terminates the
        # episode early. Requires n_distractors >= 1. Mutually exclusive with
        # pick_only_reward.
        self.split_only_reward = bool(split_only_reward)
        self.split_target_gap = float(split_target_gap)
        self.split_sep_coef = float(split_sep_coef)
        self.split_table_penalty_coef = float(split_table_penalty_coef)
        self.split_bowl_penalty_coef = float(split_bowl_penalty_coef)
        self.split_stable_speed_threshold = float(split_stable_speed_threshold)
        self.split_stable_duration_s = float(split_stable_duration_s)
        # Two-phase split: once ALL cubes are separated, switch to a hover
        # phase that drives the TCP (finger-tip midpoint) split_hover_z above
        # the GOAL cube. Disabled by default (pure separate).
        self.split_hover_after_separate = bool(split_hover_after_separate)
        self.split_hover_z = float(split_hover_z)
        self.split_hover_coef = float(split_hover_coef)
        self.split_hover_tol = float(split_hover_tol)
        # Color-ordered separation curriculum (split mode): instead of pushing
        # on the worst pairwise gap (all at once), isolate ONE cube at a time
        # in a fixed color-priority order (lowest color index first), advancing
        # only once the current target is ≥ split_target_gap from all others.
        self.split_color_hierarchy = bool(split_color_hierarchy)
        # Harsh anti-fling penalty: −coef per cube that ends up further than
        # split_far_penalty_dist (m) from the cluster spawn centre. Also gates
        # success (no cube may be flung). 0 = disabled.
        self.split_far_penalty_coef = float(split_far_penalty_coef)
        self.split_far_penalty_dist = float(split_far_penalty_dist)
        # Per-env consecutive-stable counter (split mode). Lazy init in evaluate.
        self._split_slow_counter = None
        # Per-env sticky "all cubes separated this episode" flag (split mode).
        # Lazy init in evaluate; reset per-episode in _initialize_episode.
        self._all_separated_sticky = None
        # Per-env current curriculum stage (split_color_hierarchy). Reset to 0
        # per-episode; advances as each color-ranked cube is isolated.
        self._split_stage = None
        # Per-env cluster spawn centre (xy), snapshotted at episode init; used
        # by the anti-fling penalty. Lazy init in _initialize_episode.
        self._split_cluster_xy = None
        self._last_action = None
        self._just_reset_mask = None
        # Per-env state for the "stay at lift xy" reward: cube xy snapshotted
        # at the moment of grasp+lift transition. Lazy init on first reward call.
        self._cube_xy_at_lift = None
        self._prev_lifted = None
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
        # skip_bowl: skip bowl actor entirely. Used to match offline demos
        # recorded without a bowl. Only valid with pick_only_reward (other
        # reward modes dereference self.bin.pose.p). Assertion below.
        self.skip_bowl = bool(skip_bowl)
        if self.skip_bowl:
            assert pick_only_reward, (
                "skip_bowl=True is only supported with pick_only_reward=True "
                "(other reward modes reference bin.pose.p which is None when "
                "skip_bowl=True). Either enable pick_only_reward or disable skip_bowl."
            )
            assert not split_only_reward, (
                "skip_bowl=True is incompatible with split_only_reward."
            )

        if self.split_only_reward:
            if self.pick_only_reward:
                raise ValueError(
                    "split_only_reward and pick_only_reward are mutually exclusive."
                )
            if not (self.item_type == "cube" and self.n_distractors >= 1):
                raise ValueError(
                    "split_only_reward requires a cube task with n_distractors >= 1 "
                    f"(two cubes to separate). Got item_type={self.item_type!r}, "
                    f"n_distractors={self.n_distractors}."
                )

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
        # Cube/bowl spawn region in robot frame: a half-disk of radius
        # `spawn_arc_radius` centred at `spawn_arc_center`, occupying the +x
        # hemisphere (the half "in front of" the robot). The flat side of the
        # half-disk lies along the y-axis through the centre; the curved edge
        # extends in +x. The cube and bowl xy are both sampled in this region.
        self.spawn_arc_center = (float(spawn_arc_center[0]), float(spawn_arc_center[1]))
        self.spawn_arc_radius = float(spawn_arc_radius)
        # Straight-line clips on the half-disk so cubes only spawn where the
        # wrist camera can see them at the start pose (the policy is vision-only
        # — a cube it can't see is never grasped). spawn_y_clip cuts the wide
        # near-sides (|robot-frame y| ≤ clip); spawn_x_min cuts the near/back
        # strip the camera barely sees (robot-frame x ≥ min). None disables.
        self.spawn_y_clip = None if spawn_y_clip is None else float(spawn_y_clip)
        self.spawn_x_min = None if spawn_x_min is None else float(spawn_x_min)
        # Latin-hypercube sampling across each batch of resetting envs in
        # (r², θ) space — gives uniform area coverage with no two envs in
        # the same r-band or θ-band per batch. With 2048 envs in a typical
        # PPO reset that's 2048 radial bands of ~0.15 mm and 2048 angular
        # bands of ~0.05° — effectively perfect even spread per batch.
        self.spawn_stratified = bool(spawn_stratified)
        self.visualize_spawn_area = bool(visualize_spawn_area)
        self.visualize_robot_frame = bool(visualize_robot_frame)

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
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
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
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
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
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
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
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_restitutions = common.to_tensor(restitutions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)
        self.item_masses = common.to_tensor(masses, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=float(frictions[i]),
                dynamic_friction=float(frictions[i]),
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
            # Success-check rectangle is tighter than the bowl AABB so a cube
            # landing on the rim/outer wall does NOT count as success. 10x10 cm
            # full → 5 cm half. self.bowl_half_* still hold the actual mesh
            # extents for rendering / physics — don't reuse them here.
            bin_half_sizes_x = np.ones(self.num_envs) * 0.05
            bin_half_sizes_y = np.ones(self.num_envs) * 0.05
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
            # skip_bowl mode: exit before creating any bin actor. bins stays
            # empty; self.bin is set to None below. Done as a break instead
            # of wrapping the for loop in an if/else so the original body
            # (and its indentation) stays untouched.
            if self.skip_bowl:
                break
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
                # Visual from .ply alongside .obj. Bowl color is forced
                # white by _randomize_bowl_tint (base_color + emission),
                # which runs each reset.
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

        if self.skip_bowl:
            # bins is empty (loop broke before any actor build). Sentinel
            # value picked up by _initialize_episode / _get_obs* / evaluate.
            self.bin = None
        else:
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
                        static_friction=float(frictions[i]),
                        dynamic_friction=float(frictions[i]),
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

        if self.visualize_spawn_area:
            self._add_spawn_area_marker()
        if self.visualize_robot_frame:
            self._add_robot_frame_marker()

    def _add_spawn_area_marker(self, n_arc_segments: int = 24):
        """Visual-only red outline of the cube/bowl spawn region (the half-disk
        clipped by the straight-line edge cuts spawn_y_clip / spawn_x_min).

        Anchored at `spawn_arc_center` (robot frame). Drawn as: the front arc
        (only the |y| ≤ spawn_y_clip portion, as chord segments), two straight
        side lines at y = ±spawn_y_clip, and the near edge (a straight line at
        x = spawn_x_min, or the half-disk diameter if no x clip). Self-lit so it
        stays visible under the low-ambient / dim-exposure lighting DR."""
        cx, cy = self.spawn_arc_center
        R = self.spawn_arc_radius
        yclip = self.spawn_y_clip
        xmin = self.spawn_x_min
        wall_t = 0.003           # 3 mm border thickness
        wall_hz = 0.0025         # 5 mm total height
        red_mat = sapien.render.RenderMaterial(
            base_color=[1.0, 0.05, 0.05, 1.0],
            emission=[1.0, 0.05, 0.05, 1.0],
        )
        builder = self.scene.create_actor_builder()

        def add_segment(p0, p1):
            """Add a thin box between two robot-frame points (absolute)."""
            x0, y0 = p0[0] - cx, p0[1] - cy        # → marker-anchor frame
            x1, y1 = p1[0] - cx, p1[1] - cy
            L = float(np.hypot(x1 - x0, y1 - y0))
            if L < 1e-4:
                return
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            yaw = float(np.arctan2(y1 - y0, x1 - x0))
            builder.add_box_visual(
                pose=sapien.Pose(p=[float(mx), float(my), wall_hz], q=euler2quat(0, 0, yaw)),
                half_size=[L / 2 + wall_t, wall_t, wall_hz],
                material=red_mat,
            )

        # Arc half-angle visible after the |y| ≤ yclip cut.
        if yclip is None or yclip >= R:
            th = np.pi / 2
        else:
            th = float(np.arcsin(min(1.0, yclip / R)))
        # Front arc (θ ∈ [-th, +th]) as chord segments.
        ts = np.linspace(-th, th, n_arc_segments + 1)
        for i in range(n_arc_segments):
            p0 = (cx + R * np.cos(ts[i]),   cy + R * np.sin(ts[i]))
            p1 = (cx + R * np.cos(ts[i + 1]), cy + R * np.sin(ts[i + 1]))
            add_segment(p0, p1)

        # y-extent of the arc endpoints (= yclip if clipped, else R).
        ny = yclip if (yclip is not None and yclip < R) else R
        # Near edge x: the x-clip line if it's forward of the centre, else the diameter at cx.
        near_x = xmin if (xmin is not None and xmin > cx) else cx
        # Arc endpoint x (where the arc meets y = ±ny).
        arc_x = cx + R * np.cos(th)

        # Two side lines at y = ±ny, from the near edge to the arc endpoints.
        add_segment((near_x, cy + ny), (arc_x, cy + ny))
        add_segment((near_x, cy - ny), (arc_x, cy - ny))
        # Near edge (straight line) from -ny to +ny.
        add_segment((near_x, cy - ny), (near_x, cy + ny))

        builder.initial_pose = sapien.Pose(p=[cx, cy, 0.0])
        self.spawn_area_marker = builder.build_kinematic(name="spawn_area_marker")

    def _add_robot_frame_marker(self, length: float = 0.20, radius: float = 0.008):
        """Visual-only RGB triad at the world origin (= robot base for SO101)
        showing the robot frame: red = +X, green = +Y, blue = +Z. SAPIEN
        cylinders are along +X in their local frame; the Y and Z arrows are
        rotated 90° around Z and -Y respectively. Self-lit so they stay
        visible under the dim-exposure tail of the lighting DR."""
        half_l = length / 2
        red = sapien.render.RenderMaterial(
            base_color=[1.0, 0.1, 0.1, 1.0], emission=[1.0, 0.1, 0.1, 1.0],
        )
        grn = sapien.render.RenderMaterial(
            base_color=[0.1, 1.0, 0.1, 1.0], emission=[0.1, 1.0, 0.1, 1.0],
        )
        blu = sapien.render.RenderMaterial(
            base_color=[0.2, 0.4, 1.0, 1.0], emission=[0.2, 0.4, 1.0, 1.0],
        )
        builder = self.scene.create_actor_builder()
        # +X (red): cylinder's local axis = +X, centred at (half_l, 0, 0).
        builder.add_cylinder_visual(
            radius=radius, half_length=half_l, material=red,
            pose=sapien.Pose(p=[half_l, 0, 0]),
        )
        # +Y (green): rotate +90° around Z to swing local +X → world +Y.
        builder.add_cylinder_visual(
            radius=radius, half_length=half_l, material=grn,
            pose=sapien.Pose(p=[0, half_l, 0], q=euler2quat(0, 0, np.pi / 2)),
        )
        # +Z (blue): rotate -90° around Y to swing local +X → world +Z.
        builder.add_cylinder_visual(
            radius=radius, half_length=half_l, material=blu,
            pose=sapien.Pose(p=[0, 0, half_l], q=euler2quat(0, -np.pi / 2, 0)),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.robot_frame_marker = builder.build_kinematic(name="robot_frame_marker")

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
        """Per-episode bowl tint. LOCKED to pure white — base_color is set to
        white and a full-strength emission is applied so the bowl appears
        bright white regardless of scene lighting. DR knobs ignored.

        SAPIEN ignores PLY vertex colors, so the only color source is the
        per-part material; setting base_color + emission here is sufficient."""
        if self.bin is None:
            return
        env_idx_list = env_idx.tolist() if isinstance(env_idx, torch.Tensor) else list(env_idx)
        base = [1.0, 1.0, 1.0, 1.0]
        # Emission strength: 0=lit only, 1=full self-luminous. ~0.5 keeps the
        # bowl bright white but lets scene shading still modulate it.
        e = 0.4
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
            # Clear the lift-xy lock + lifted-prev flag for the reset envs.
            if self._cube_xy_at_lift is not None:
                self._cube_xy_at_lift[env_idx] = 0
                self._prev_lifted[env_idx] = False
            # Clear the pick-only stable-grasp counter for the reset envs.
            if self._grasp_slow_counter is not None:
                self._grasp_slow_counter[env_idx] = 0
            # Clear the sticky "fixed finger has touched cube" flag.
            if self._fixed_finger_touched is not None:
                self._fixed_finger_touched[env_idx] = False
            # Clear the prev-grasped flag used by the drop penalty.
            if self._prev_is_grasped is not None:
                self._prev_is_grasped[env_idx] = False
            # Clear the split-mode stable-separation counter for reset envs.
            if self._split_slow_counter is not None:
                self._split_slow_counter[env_idx] = 0
            # Clear the sticky "all cubes separated" flag for reset envs.
            if self._all_separated_sticky is not None:
                self._all_separated_sticky[env_idx] = False
            # Reset the color-hierarchy curriculum stage for reset envs.
            if self._split_stage is not None:
                self._split_stage[env_idx] = 0
            # Lazily allocate the cluster-centre buffer (snapshotted below).
            if self.split_only_reward and self._split_cluster_xy is None:
                self._split_cluster_xy = torch.zeros(
                    self.num_envs, 2, device=self.device
                )

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            # Bin sampling center (unchanged): robot.pose + spawn_box_pos.
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

            # Cube spawn — sample directly in the half-disk region in robot
            # frame (radius `spawn_arc_radius` around `spawn_arc_center`, +x
            # hemisphere only), with Latin-hypercube stratification on
            # (r², θ) across this reset batch when self.spawn_stratified.
            arc_cx, arc_cy = self.spawn_arc_center
            arc_R = self.spawn_arc_radius
            # Index by env_idx (NOT [:, :2]) so this is sized [b, 2] for partial
            # resets — otherwise it's [num_envs, 2] and broadcasts wrong against
            # the [b, 2] offsets, crashing whenever a subset of envs resets.
            arc_center_world = self.agent.robot.pose.p[env_idx, :2] + torch.tensor(
                [arc_cx, arc_cy], device=self.device,
            )

            # Straight-line clips, expressed as OFFSETS from the arc centre
            # (absolute robot-frame y = arc_cy + off_y; x = arc_cx + off_x).
            y_clip_off = None if self.spawn_y_clip is None else self.spawn_y_clip - arc_cy
            x_min_off = None if self.spawn_x_min is None else self.spawn_x_min - arc_cx

            def _in_clip(off: torch.Tensor) -> torch.Tensor:
                ok = torch.ones(off.shape[0], dtype=torch.bool, device=self.device)
                if y_clip_off is not None:
                    ok &= off[:, 1].abs() <= y_clip_off
                if x_min_off is not None:
                    ok &= off[:, 0] >= x_min_off
                return ok

            def sample_half_disk(n: int, stratified: bool) -> torch.Tensor:
                """Sample n (x, y) offsets uniformly in the half-disk of radius
                arc_R centred at the origin, with θ ∈ [-π/2, +π/2] (+x side),
                then clip to the visible band (|y| ≤ spawn_y_clip, x ≥
                spawn_x_min) by rejection-resampling out-of-band points. When
                stratified, performs LHS in (r², θ) space for even coverage; the
                (rare) resampled points are uniform, so the LHS spread of the
                in-band majority is preserved."""
                def draw(m, strat):
                    if strat and m > 1:
                        perm_r = torch.randperm(m, device=self.device)
                        perm_t = torch.randperm(m, device=self.device)
                        u_r = (perm_r.float() + torch.rand(m, device=self.device)) / m
                        u_t = (perm_t.float() + torch.rand(m, device=self.device)) / m
                    else:
                        u_r = torch.rand(m, device=self.device)
                        u_t = torch.rand(m, device=self.device)
                    r = arc_R * torch.sqrt(u_r)              # r² uniform → uniform area
                    theta = -np.pi / 2 + u_t * np.pi
                    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)

                off = draw(n, stratified)
                if y_clip_off is None and x_min_off is None:
                    return off
                # Rejection-resample out-of-band points (uniform) until valid.
                for _ in range(50):
                    bad = ~_in_clip(off)
                    nb = int(bad.sum().item())
                    if nb == 0:
                        break
                    off[bad] = draw(nb, False)
                return off

            item_xy_offset = sample_half_disk(b, self.spawn_stratified)
            # Bowl: same half-disk. Sampled uniformly (no LHS — keeping the
            # cube's LHS spread is the priority; the bowl re-samples on
            # collision below).
            bin_xy_offset = sample_half_disk(b, stratified=False)

            # Absolute world positions (both item and bin anchored at
            # arc_center_world now).
            item_xy_world = arc_center_world + item_xy_offset
            bin_xy_world  = arc_center_world + bin_xy_offset

            # Cube–bowl exclusion: cube center must be ≥ 10 cm from bowl
            # center (bowl rim is at 7.5 cm, +2.5 cm safety). Re-sample the
            # BOWL on conflict so the cube's LHS-spread structure is preserved.
            # Skipped entirely in skip_bowl mode (no bowl to avoid).
            if not self.skip_bowl:
                BOWL_EXCLUSION = 0.10
                for _ in range(20):
                    delta_xy = item_xy_world - bin_xy_world
                    dist = torch.linalg.norm(delta_xy, dim=-1)
                    bad = dist < BOWL_EXCLUSION
                    if not bad.any():
                        break
                    n_bad = int(bad.sum().item())
                    bin_xy_offset[bad] = sample_half_disk(n_bad, stratified=False)
                    bin_xy_world = arc_center_world + bin_xy_offset
                # Hard fix for any holdouts: push the BOWL radially away from the
                # cube to the exclusion boundary (rare). The push direction is
                # tangent-only when the conflict is exactly axial; otherwise away
                # from the cube. May leave the bowl slightly outside the disk in
                # degenerate cases — acceptable for the rare fallback.
                delta_xy = bin_xy_world - item_xy_world
                dist = torch.linalg.norm(delta_xy, dim=-1, keepdim=True)
                still_bad = (dist < BOWL_EXCLUSION).squeeze(-1)
                if still_bad.any():
                    direction = torch.where(
                        dist > 1e-6, delta_xy / dist.clamp(min=1e-6),
                        torch.tensor([1.0, 0.0], device=self.device).expand_as(delta_xy),
                    )
                    bin_xy_world = torch.where(
                        still_bad.unsqueeze(-1),
                        item_xy_world + direction * BOWL_EXCLUSION,
                        bin_xy_world,
                    )

            # Cluster of (1 + n_distractors) cubes glued face-to-face. The
            # sampled position is the geometric center of the cluster, so the
            # goal cube can be ANY of the slots (center or any cardinal),
            # uniformly random — not always the middle one.
            cluster_xy = item_xy_world
            # Snapshot the cluster spawn centre for the split anti-fling penalty.
            if self.split_only_reward and self._split_cluster_xy is not None:
                self._split_cluster_xy[env_idx] = cluster_xy
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

            # Set bin pose. Skipped in skip_bowl mode (self.bin is None).
            if self.bin is not None:
                bin_xyz = torch.zeros((b, 3))
                bin_xyz[:, :2] = bin_xy_world
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
            else:
                # skip_bowl: park goal_site at origin (unused — pick_only_reward
                # ignores goal_site, and it's a kinematic hidden actor).
                goal_xyz = torch.zeros((b, 3), device=self.device)
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
        # at the end of the flattened state vector. In real-deploy the real
        # robot has no `.pose`; fall back to a fixed measured bowl position.
        # skip_bowl mode: zero-pad so the state-vector dim stays stable.
        if getattr(self, "bin", None) is not None and hasattr(self.agent.robot, "pose"):
            obs["bowl_xyz_robot_frame"] = (self.agent.robot.pose.inv() * self.bin.pose).p
        elif getattr(self, "skip_bowl", False):
            obs["bowl_xyz_robot_frame"] = torch.zeros(
                qpos.shape[0], 3, dtype=qpos.dtype, device=qpos.device
            )
        else:
            bowl_xyz = torch.tensor([0.20, 0.10, 0.00], dtype=qpos.dtype, device=qpos.device)
            obs["bowl_xyz_robot_frame"] = bowl_xyz.unsqueeze(0).expand(qpos.shape[0], 3)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            # skip_bowl: zero-pad the bin-related slices so the state-vector dim
            # stays stable across modes. RGB is the primary policy input; the
            # privileged state slices are zero-padded in deploy and demo loads
            # too (see RLPD.md table), so this matches.
            if self.bin is not None:
                bin_pose = self.bin.pose.raw_pose
                tcp_to_bin_pos = self.bin.pose.p - self.agent.tcp_pos
                item_to_bin_pos = self.bin.pose.p - self.item.pose.p
            else:
                B = self.item.pose.p.shape[0]
                bin_pose = torch.zeros(B, 7, device=self.device)
                tcp_to_bin_pos = torch.zeros(B, 3, device=self.device)
                item_to_bin_pos = torch.zeros(B, 3, device=self.device)
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bin_pose=bin_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bin_pos=tcp_to_bin_pos,
                item_to_bin_pos=item_to_bin_pos,
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
        # skip_bowl: cube can never be "above bin" since there's no bin. Force
        # the boolean to False; bin_pos / offset / inside_* are skipped.
        if self.bin is not None:
            bin_pos = self.bin.pose.p.clone()
            bin_pos[:, 2] = self.bin_thickness + self.item_half_sizes

            offset = item_pos - bin_pos
            inside_x = torch.abs(offset[:, 0]) < self.bin_half_sizes_x
            inside_y = torch.abs(offset[:, 1]) < self.bin_half_sizes_y
            # Cube must also be at least 4 cm above the table (z = 0) so a cube
            # flush on the table can't trigger above_bin / success.
            is_cube_above_table = item_pos[:, 2] > 0.04
            is_item_above_bin = inside_x & inside_y & is_cube_above_table
        else:
            is_item_above_bin = torch.zeros(item_pos.shape[0], dtype=torch.bool, device=self.device)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        # "Lifted" = cube is clearly off the table: cube_z > 2 cm AND no
        # cube-table contact force. The 2 cm clearance gives the policy time
        # to fully clamp the gripper before the hold-still penalty fires on
        # `item_lifted`. Cube-table contact is checked via pairwise forces
        # (proxy threshold 1e-2 N, mirroring is_touching in so101.py).
        item_table_forces = self.scene.get_pairwise_contact_forces(
            self.item, self.table_scene.table
        )
        item_touching_table = torch.linalg.norm(item_table_forces, axis=1) >= 1e-2
        item_lifted = (self.item.pose.p[..., -1] > 0.02) & (~item_touching_table)
        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        # skip_bowl: no bin to touch; force the boolean to False.
        if self.bin is not None:
            robot_touching_bin = self.agent.is_touching(self.bin)
        else:
            robot_touching_bin = torch.zeros(item_pos.shape[0], dtype=torch.bool, device=self.device)
        robot_touching_item = self.agent.is_touching(self.item)

        # ── Split mode: inter-cube separation over ALL pairs ────────────────
        # min_gap = smallest surface gap (center_dist − 2·half_size) over every
        # pair of cubes (goal + distractors). All cubes are "separated" when
        # min_gap ≥ split_target_gap. Only computed in split mode.
        cube_separation = torch.zeros_like(item_vel)
        min_gap = torch.zeros_like(item_vel)
        split_num_far = torch.zeros_like(item_vel)
        split_stage_f = torch.zeros_like(item_vel)
        split_target_pos = item_pos
        split_target_min_gap = min_gap
        if self.split_only_reward:
            cube_pos = torch.stack(
                [item_pos] + [d.pose.p for d in self.distractors], dim=1
            )  # (E, C, 3)
            cube_vel = torch.stack(
                [item_vel]
                + [torch.linalg.norm(d.linear_velocity, axis=-1) for d in self.distractors],
                dim=1,
            )  # (E, C)
            C = cube_pos.shape[1]
            pair_dist = torch.linalg.norm(
                cube_pos[:, :, None, :] - cube_pos[:, None, :, :], axis=-1
            )  # (E, C, C)
            eye = torch.eye(C, dtype=torch.bool, device=self.device)
            min_center_dist = pair_dist.masked_fill(eye, float("inf")).amin(dim=(1, 2))
            cube_separation = min_center_dist
            min_gap = min_center_dist - 2.0 * self.item_half_sizes
            cubes_slow_all = (cube_vel < self.split_stable_speed_threshold).all(dim=1)
            E = cube_pos.shape[0]
            arange_E = torch.arange(E, device=self.device)

            # Anti-fling count: cubes further than split_far_penalty_dist (xy)
            # from the cluster spawn centre. no_fling gates success.
            if self.split_far_penalty_coef > 0.0 and self._split_cluster_xy is not None:
                far_dist = torch.linalg.norm(
                    cube_pos[..., :2] - self._split_cluster_xy[:, None, :], axis=-1
                )  # (E, C)
                split_num_far = (far_dist > self.split_far_penalty_dist).sum(dim=1).float()
                no_fling = split_num_far == 0
            else:
                no_fling = torch.ones(E, dtype=torch.bool, device=self.device)

            # Color-ordered curriculum: isolate ONE cube at a time, lowest color
            # index first. split_target_min_gap = the current target's smallest
            # gap to the others; the stage advances once the target is isolated.
            if self.split_color_hierarchy:
                if self._split_stage is None or self._split_stage.shape[0] != E:
                    self._split_stage = torch.zeros(E, dtype=torch.long, device=self.device)
                color_idx = torch.stack(
                    [self.goal_color_idx]
                    + [self.distractor_color_idxs[:, k] for k in range(len(self.distractors))],
                    dim=1,
                ).long()  # (E, C)
                order = torch.argsort(color_idx, dim=1)  # cube idx by color rank
                stage_c = self._split_stage.clamp(max=C - 1)
                target_idx = order[arange_E, stage_c]
                split_target_pos = cube_pos[arange_E, target_idx]  # (E, 3)
                d_to_target = torch.linalg.norm(
                    cube_pos - split_target_pos[:, None, :], axis=-1
                )  # (E, C)
                d_to_target[arange_E, target_idx] = float("inf")  # exclude self
                split_target_min_gap = d_to_target.amin(dim=1) - 2.0 * self.item_half_sizes
                advance = (split_target_min_gap >= self.split_target_gap) & (self._split_stage < C)
                self._split_stage = self._split_stage + advance.long()
                split_stage_f = self._split_stage.float()

            # Sticky "all cubes have been separated this episode" flag — gates
            # the hover phase so the reward doesn't oscillate if a cube drifts.
            if self._all_separated_sticky is None or self._all_separated_sticky.shape[0] != E:
                self._all_separated_sticky = torch.zeros(E, dtype=torch.bool, device=self.device)
            all_separated = min_gap >= self.split_target_gap
            self._all_separated_sticky = self._all_separated_sticky | all_separated

            if self._split_slow_counter is None or self._split_slow_counter.shape[0] != E:
                self._split_slow_counter = torch.zeros(E, dtype=torch.int32, device=self.device)
            if self.split_hover_after_separate:
                # Phase 2: the TCP (finger-tip midpoint) must reach split_hover_z
                # above the GOAL cube while all cubes stay separated and still.
                hover_target = item_pos.clone()
                hover_target[:, 2] = hover_target[:, 2] + self.split_hover_z
                hover_dist = torch.linalg.norm(self.agent.tcp_pose.p - hover_target, axis=1)
                stable_split = all_separated & (hover_dist < self.split_hover_tol) & cubes_slow_all
            else:
                stable_split = all_separated & cubes_slow_all
            stable_split = stable_split & no_fling  # can't succeed with a flung cube
            self._split_slow_counter = torch.where(
                stable_split,
                self._split_slow_counter + 1,
                torch.zeros_like(self._split_slow_counter),
            )
            split_steps_required = max(
                1, int(round(self.split_stable_duration_s * self._control_freq))
            )
            success = self._split_slow_counter >= split_steps_required
        elif self.pick_only_reward:
            # Pick-only success: grasped AND cube nearly stationary for N
            # *consecutive* control steps. Any single bad step resets the
            # counter, mirroring the lift.py pick_only_reward design.
            if self._grasp_slow_counter is None or self._grasp_slow_counter.shape[0] != item_pos.shape[0]:
                self._grasp_slow_counter = torch.zeros(
                    item_pos.shape[0], dtype=torch.int32, device=self.device
                )
            is_item_slow = item_vel < self.pick_stable_speed_threshold
            stable_grasp = is_item_grasped & is_item_slow
            self._grasp_slow_counter = torch.where(
                stable_grasp,
                self._grasp_slow_counter + 1,
                torch.zeros_like(self._grasp_slow_counter),
            )
            stable_steps_required = max(
                1, int(round(self.pick_stable_duration_s * self._control_freq))
            )
            success = self._grasp_slow_counter >= stable_steps_required
        else:
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
            "cube_separation": cube_separation,
            "min_gap": min_gap,
            "split_num_far": split_num_far,
            "split_stage": split_stage_f,
            "split_target_pos": split_target_pos,
            "split_target_min_gap": split_target_min_gap,
        }

    @property
    def _pick_only_max_steps(self) -> int:
        """Cached lookup of the registered max_episode_steps for the terminal
        bonus in pick-only mode. ManiSkill envs set this via the @register_env
        decorator (e.g. PlaceCube=75). spec.max_episode_steps is often None,
        so we read directly from the registry."""
        if not hasattr(self, "_cached_pick_max_steps"):
            from mani_skill.utils.registration import REGISTERED_ENVS
            env_id = getattr(getattr(self, "spec", None), "id", None)
            spec_max = getattr(getattr(self, "spec", None), "max_episode_steps", None)
            if spec_max is not None:
                self._cached_pick_max_steps = int(spec_max)
            elif env_id and env_id in REGISTERED_ENVS:
                self._cached_pick_max_steps = int(REGISTERED_ENVS[env_id].max_episode_steps)
            else:
                self._cached_pick_max_steps = 100  # PlaceCube default
        return self._cached_pick_max_steps

    def _compute_dense_reward_pick_only(self, obs: Any, action: torch.Tensor, info: dict):
        """Pick-only reward: reach → grasp → close-hard → stay still.
        Episode auto-terminates on success (grasped + slow for N consecutive
        control steps); terminal bonus replaces the remaining-step reward so
        terminating early matches "continue at peak forever" total return.

        Side-approach curriculum (pick_side_approach=True): before the FIXED
        finger touches the cube, the reward is (reach + open_coef·gripper_open)
        — no grasp / strong-grasp incentive — forcing the policy to approach
        with the gripper open. Once the fixed finger has touched (sticky for
        the rest of the episode), the reward switches to the normal grasp
        ladder. open_coef defaults to 0.3 so pre-touch peak (~1.3) is below
        the post-touch grasped-and-clamped peak (1 + strong_grasp_coef = 1.5);
        otherwise the policy would learn to hover and never touch."""
        tcp_to_item_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.item.pose.p, axis=1
        )
        reach = 1 - torch.tanh(5 * tcp_to_item_dist)
        is_grasped = info["is_item_grasped"].float()

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        target_qpos = self.agent.controller.controllers["arm"]._target_qpos
        target_grip = target_qpos[:, -1]
        target_closure = torch.clamp(
            (gripper_max - target_grip) / (gripper_max - gripper_min),
            0.0, 1.0,
        )

        normal_reward = (
            (1 - is_grasped) * reach
            + is_grasped
            + self.strong_grasp_coef * target_closure * is_grasped
        )

        if self.pick_side_approach:
            N = is_grasped.shape[0]
            if (self._fixed_finger_touched is None
                    or self._fixed_finger_touched.shape[0] != N):
                self._fixed_finger_touched = torch.zeros(
                    N, dtype=torch.bool, device=self.device
                )
            # Fixed finger = SO101's gripper_link (jaw + tip). Check contact
            # forces against the cube; threshold matches is_touching's 1e-2 N.
            link_forces = self.scene.get_pairwise_contact_forces(
                self.agent.finger1_link, self.item
            )
            tip_forces = self.scene.get_pairwise_contact_forces(
                self.agent.finger1_tip, self.item
            )
            fixed_touch_now = (
                (torch.linalg.norm(link_forces, axis=1) >= 1e-2)
                | (torch.linalg.norm(tip_forces, axis=1) >= 1e-2)
            )
            # Sticky: once True in the episode, stays True until reset.
            self._fixed_finger_touched = self._fixed_finger_touched | fixed_touch_now

            gripper_open = torch.clamp(
                (self.agent.robot.get_qpos()[:, -1] - gripper_min)
                / (gripper_max - gripper_min),
                0.0, 1.0,
            )
            pre_touch_reward = reach + self.pick_side_approach_open_coef * gripper_open
            reward = torch.where(
                self._fixed_finger_touched, normal_reward, pre_touch_reward
            )
        else:
            reward = normal_reward

        reward = reward - 0.5 * info["robot_touching_table"].float()

        # Drop penalty: fires on every grasped → not-grasped transition
        # (i.e., a fumble). Pushes the policy to one-shot the grasp instead
        # of dropping and trying again. is_item_grasped here is the same
        # bool used by the rest of the reward, so the transition detection
        # is consistent with the grasp ladder above.
        if self.drop_penalty_coef > 0.0:
            is_grasped_bool = info["is_item_grasped"]
            N = is_grasped_bool.shape[0]
            if self._prev_is_grasped is None or self._prev_is_grasped.shape[0] != N:
                self._prev_is_grasped = torch.zeros(
                    N, dtype=torch.bool, device=self.device
                )
            drop_event = self._prev_is_grasped & (~is_grasped_bool)
            reward = reward - self.drop_penalty_coef * drop_event.float()
            self._prev_is_grasped = is_grasped_bool.clone()

        # Terminal bonus: (max_steps - elapsed) · per_step_peak. With
        # per_step_peak = 1 + strong_grasp_coef the success branch yields the
        # same total return as a hypothetical "stay at peak" continuation.
        per_step_peak = 1.0 + self.strong_grasp_coef
        max_steps = self._pick_only_max_steps
        remaining = (max_steps - self.elapsed_steps.float()).clamp(min=0)
        reward = torch.where(
            info["success"], per_step_peak * remaining, reward
        )
        return reward

    def _compute_dense_reward_split(self, obs: Any, action: torch.Tensor, info: dict):
        """Split reward: push ALL cubes apart until every pair's surface gap
        reaches split_target_gap. No grasping.

        Phase 1 (until all separated): reach the NEAREST cube +
            split_sep_coef·separation_progress (progress on the smallest gap).
        Phase 2 (split_hover_after_separate=True, sticky once all separated):
            hold the separated baseline (= phase-1 peak) and drive the TCP
            (finger-tip midpoint) to split_hover_z above the GOAL cube. Phase-2
            peak (1 + sep_coef + hover_coef) sits above phase-1 peak so the
            policy is pulled forward into the hover.

        Episode auto-terminates on success; the terminal bonus replaces the
        remaining-step reward so terminating early matches a "stay at peak
        forever" continuation, mirroring _compute_dense_reward_pick_only."""
        item_pos = self.item.pose.p
        tcp = self.agent.tcp_pose.p

        if self.split_color_hierarchy:
            # Curriculum phase 1: reach the CURRENT target cube and isolate it;
            # staged progress = (completed stages + current-target progress)/C,
            # so the signal is focused on one cube at a time, in color order.
            C = len(self.distractors) + 1
            reach = 1 - torch.tanh(
                5 * torch.linalg.norm(tcp - info["split_target_pos"], axis=1)
            )
            target_progress = torch.clamp(
                info["split_target_min_gap"] / self.split_target_gap, 0.0, 1.0
            )
            staged = torch.clamp((info["split_stage"] + target_progress) / C, 0.0, 1.0)
            phase1 = reach + self.split_sep_coef * staged
        else:
            # Reach the NEAREST cube; progress on the smallest pairwise gap.
            cube_pos = torch.stack(
                [item_pos] + [d.pose.p for d in self.distractors], dim=1
            )  # (E, C, 3)
            reach = 1 - torch.tanh(
                5 * torch.linalg.norm(cube_pos - tcp[:, None, :], axis=-1).amin(dim=1)
            )
            sep_progress = torch.clamp(info["min_gap"] / self.split_target_gap, 0.0, 1.0)
            phase1 = reach + self.split_sep_coef * sep_progress

        if self.split_hover_after_separate:
            hover_target = item_pos.clone()
            hover_target[:, 2] = hover_target[:, 2] + self.split_hover_z
            hover_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp - hover_target, axis=1))
            phase2 = (1.0 + self.split_sep_coef) + self.split_hover_coef * hover_reward
            reward = torch.where(self._all_separated_sticky, phase2, phase1)
            per_step_peak = 1.0 + self.split_sep_coef + self.split_hover_coef
        else:
            reward = phase1
            per_step_peak = 1.0 + self.split_sep_coef

        # User-requested contact penalties: table + bowl (bin).
        reward = reward - self.split_table_penalty_coef * info["robot_touching_table"].float()
        reward = reward - self.split_bowl_penalty_coef * info["robot_touching_bin"].float()
        # Harsh anti-fling penalty: −coef per cube flung past split_far_penalty_dist.
        reward = reward - self.split_far_penalty_coef * info["split_num_far"]

        # Terminal bonus, mirroring pick-only's accounting: per_step_peak ·
        # remaining steps so an early success equals continuing at peak.
        max_steps = self._pick_only_max_steps
        remaining = (max_steps - self.elapsed_steps.float()).clamp(min=0)
        reward = torch.where(info["success"], per_step_peak * remaining, reward)
        return reward

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        if self.split_only_reward:
            return self._compute_dense_reward_split(obs, action, info)
        if self.pick_only_reward:
            return self._compute_dense_reward_pick_only(obs, action, info)
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
        # Hardcoded carry / drop altitudes:
        #   travel_z = 10 cm : altitude the policy lifts to immediately on grasp
        #                      and translates xy toward the bowl at.
        #   rim_z    = 6 cm  : final z (over the bowl) — descend here before
        #                      releasing the cube.
        # Once xy-close to the bowl the z target flips from travel_z → rim_z,
        # so the cube descends 4 cm before the gripper-open phase fires.
        FIXED_RIM_Z = 0.06
        FIXED_TRAVEL_Z = 0.10
        goal_xyz[..., 2] = FIXED_RIM_Z

        # Place reward (only used by the grasped-and-lifted branch below).
        # Two components, both in [0,1], summed → [0, 2]:
        #   place_reward_z  : encourages cube z to first reach travel altitude
        #                     (10 cm). Once xy-close, the z target flips to
        #                     rim_z (6 cm) so the cube descends to drop.
        #   place_reward_xy : phase-dependent —
        #                       below travel & not close-to-bowl → reward
        #                         staying at the cube xy snapshotted at the
        #                         lift moment (no diagonal drift while rising).
        #                       above travel OR close-to-bowl → reward moving
        #                         cube xy toward the bowl center.
        rim_z = goal_xyz[..., 2]
        travel_z = torch.full_like(rim_z, FIXED_TRAVEL_Z)

        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        # Fixed 10 cm "above-the-bowl" radius. Used by both place_reward_xy
        # (gates the xy reward) and the strong-grasp shutoff (lets the policy
        # relax its hold once it's over the bowl).
        XY_CLOSE_RADIUS = 0.10
        item_close_to_goal = (item_to_goal_dist_xy <= XY_CLOSE_RADIUS)

        z_target = torch.where(item_close_to_goal, rim_z, travel_z)
        item_to_goal_dist_z = torch.abs(item_pos[..., 2] - z_target)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)

        # ── Lift-xy snapshot (per env) ─────────────────────────────────────
        # On the step where (grasped & lifted) first goes True, snapshot the
        # cube's xy. While the cube is still rising (below travel altitude and
        # not yet over the bowl) the xy reward pushes the cube to STAY at
        # this snapshotted xy — i.e., go straight up, no diagonal drift.
        # Once above travel altitude OR within XY_CLOSE_RADIUS, the xy reward
        # flips to pulling toward the bowl center.
        if self._cube_xy_at_lift is None or self._cube_xy_at_lift.shape[0] != item_pos.shape[0]:
            self._cube_xy_at_lift = torch.zeros(
                (item_pos.shape[0], 2), device=item_pos.device, dtype=item_pos.dtype
            )
            self._prev_lifted = torch.zeros(
                (item_pos.shape[0],), dtype=torch.bool, device=item_pos.device
            )
        currently_lifted = info["is_item_grasped"] & info["item_lifted"]
        just_lifted = currently_lifted & (~self._prev_lifted)
        if just_lifted.any():
            self._cube_xy_at_lift[just_lifted] = item_pos[just_lifted, :2]
        self._prev_lifted = currently_lifted.clone()

        above_travel = item_pos[..., 2] >= (travel_z - 0.02)  # within 2cm of travel altitude
        xy_toward_bowl_active = above_travel | item_close_to_goal

        xy_to_bowl_reward = 1 - torch.tanh(5.0 * item_to_goal_dist_xy)
        d_xy_drift = torch.linalg.norm(item_pos[..., :2] - self._cube_xy_at_lift, dim=1)
        stay_xy_reward = 1 - torch.tanh(5.0 * d_xy_drift)

        place_reward_xy = torch.where(
            xy_toward_bowl_active,
            xy_to_bowl_reward,
            stay_xy_reward,
        )

        place_reward = place_reward_z + place_reward_xy

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasp ladder:
        #   grasped (any z)        → flat +2.5  — bridges 2.0 reaching to 3+place
        #   grasped AND lifted     → +3 + place_reward  (the place phase)
        # Later branches (above_bin, success) override these.
        reward[info["is_item_grasped"]] = 2.5
        grasped_and_lifted = info["is_item_grasped"] & info["item_lifted"]
        reward[grasped_and_lifted] = (3 + place_reward)[grasped_and_lifted]

        # ── Sim2real strong-grasp shaping ───────────────────────────────────
        # On real Feetech servos the delta-target controller drifts the gripper
        # open if the policy keeps sending small nonzero deltas during the lift.
        # Reward driving target_qpos[gripper] toward fully-closed *while* the
        # cube is grasped and NOT yet over the bowl. Shut off once xy-close to
        # the bowl so the policy can relax its clamp and release into the bin.
        # SO101's controller wraps the arm PDJointPos under controllers['arm'].
        target_qpos = self.agent.controller.controllers["arm"]._target_qpos
        target_grip = target_qpos[:, -1]
        target_closure = torch.clamp(
            (gripper_max - target_grip) / (gripper_max - gripper_min),
            0.0, 1.0,
        )  # 1.0 = target at fully-closed
        strong_grasp_active = info["is_item_grasped"] & (~item_close_to_goal)
        reward = reward + self.strong_grasp_coef * target_closure * strong_grasp_active.float()

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


@register_env("SO101PlaceCube-v1", max_episode_steps=100)
class PlaceCube(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PlaceCan-v1", max_episode_steps=150)
class PlaceCan(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
