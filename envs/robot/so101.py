import copy
from dataclasses import dataclass

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from pathlib import Path


# Default centred values for the real-arm-matched controller, from a
# step-response probe on the physical SO101 (see probe_all_joints.py +
# debug_artifacts/step_response_*.csv, 2026-05-15 calibration):
#   measured per-joint mean: pure delay L = 60 ms, time constant tau = 55 ms
#   R^2 >= 0.99 across all 5 arm joints
# At dt_ctrl = 33.3 ms (30 Hz training control, matches infer.py CONTROL_HZ):
#   delay_steps = round(60 / 33.3) = 2     (over-models L by ~7 ms)
#   lag_alpha   = dt_ctrl / (dt_ctrl + tau) = 33.3 / 88.3 = 0.378
# Domain-randomize around these once the controller-level DR plumbing is in.
ACTION_DELAY_STEPS_DEFAULT = 2
LAG_ALPHA_DEFAULT = 1.0  # 1.0 disables first-order lag (commanded target arrives instantly)


class PDJointPosDelayLagController(PDJointPosController):
    """PD joint-position controller with per-env actuator delay + lag.

    Implements two real-arm effects the analytic PhysX drive misses:
      - actuator delay: each commanded target is buffered for the env's
        action_delay_steps control steps before reaching the drive.
      - first-order lag: realised drive target = EMA of the delayed target;
        alpha = dt_ctrl / (dt_ctrl + tau). alpha=1 -> no lag.

    Per-env support
    ---------------
    Both knobs are stored as length-num_envs tensors so each parallel env
    can run a different (delay, alpha) pair. The delay buffer is a
    circular tensor of shape (max_delay_steps, num_envs, n_joints) sized
    by config.max_delay_steps; each env reads from its own delay-offset
    slot via gather. Per-env values are set externally by the env's DR
    code via `set_per_env_dynamics(env_idx, delay_steps, lag_alpha)`.

    The policy still observes the cumulative intended target via the
    parent's use_target accumulator (when enabled), so it does not have
    to fight the lag in its own observation/action contract.
    """

    config: "PDJointPosDelayLagControllerConfig"

    # --- internal state -----------------------------------------------------
    _delay_buffer = None         # (max_delay_steps, num_envs, n_joints)
    _delay_head   = 0            # int; write position in circular buffer
    _delay_per_env = None        # (num_envs,) long; per-env delay
    _alpha_per_env = None        # (num_envs,) float; per-env lag alpha
    _filtered_target = None      # (num_envs, n_joints); EMA state

    def _ensure_state(self):
        """Lazy-allocate per-env state. Called from reset (which sees qpos)."""
        max_d = max(int(self.config.max_delay_steps), 1)
        cur = self.qpos
        n_envs, n_j = cur.shape
        dev = cur.device
        need_alloc = (
            self._delay_buffer is None
            or self._delay_buffer.shape != (max_d, n_envs, n_j)
            or self._delay_per_env is None
            or self._delay_per_env.shape != (n_envs,)
            or self._alpha_per_env is None
            or self._alpha_per_env.shape != (n_envs,)
            or self._filtered_target is None
            or self._filtered_target.shape != (n_envs, n_j)
        )
        if need_alloc:
            self._delay_buffer = cur.unsqueeze(0).expand(max_d, n_envs, n_j).clone()
            self._filtered_target = cur.clone()
            # Default per-env values from the scalar config (DR can overwrite).
            default_delay = min(int(self.config.action_delay_steps), max_d - 1)
            default_delay = max(default_delay, 0)
            self._delay_per_env = torch.full(
                (n_envs,), default_delay, dtype=torch.long, device=dev)
            self._alpha_per_env = torch.full(
                (n_envs,), float(self.config.lag_alpha),
                dtype=torch.float32, device=dev)
            self._delay_head = 0

    def reset(self):
        super().reset()
        self._ensure_state()
        # Per-env reset: flush the circular buffer + EMA state for envs
        # being reset, using their current qpos.
        mask = self.scene._reset_mask
        if mask is None:
            return
        cur = self.qpos
        # Fill every buffer slot for the reset envs with the current pose.
        self._delay_buffer[:, mask, :] = cur[mask].unsqueeze(0).expand_as(
            self._delay_buffer[:, mask, :]).clone()
        self._filtered_target[mask] = cur[mask].clone()

    @torch.no_grad()
    def set_per_env_dynamics(self, env_idx, delay_steps=None, lag_alpha=None):
        """Hook for the env's randomizer. Writes per-env values into the
        controller's state. Both args broadcast to (len(env_idx),).
        """
        self._ensure_state()
        max_d = self._delay_buffer.shape[0]
        if delay_steps is not None:
            d = torch.as_tensor(delay_steps, dtype=torch.long,
                                device=self._delay_per_env.device)
            self._delay_per_env[env_idx] = d.clamp(0, max_d - 1)
        if lag_alpha is not None:
            a = torch.as_tensor(lag_alpha, dtype=torch.float32,
                                device=self._alpha_per_env.device)
            self._alpha_per_env[env_idx] = a.clamp(1e-3, 1.0)

    def set_action(self, action):
        self._ensure_state()
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        # Update cumulative intended target the same way the parent does.
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape).clone()

        # Per-env circular buffer: write at head, read at (head - delay).
        max_d = self._delay_buffer.shape[0]
        self._delay_buffer[self._delay_head] = self._target_qpos
        read_pos = (self._delay_head - self._delay_per_env) % max_d   # (num_envs,)
        env_arange = torch.arange(self._target_qpos.shape[0],
                                  device=self._target_qpos.device)
        delayed = self._delay_buffer[read_pos, env_arange]            # (num_envs, n_j)
        self._delay_head = (self._delay_head + 1) % max_d

        # Per-env first-order lag (broadcast alpha over joints).
        alpha = self._alpha_per_env.unsqueeze(-1)                     # (num_envs, 1)
        self._filtered_target = (1.0 - alpha) * self._filtered_target + alpha * delayed

        if self.config.interpolate:
            self._step_size = (
                self._filtered_target - self._start_qpos
            ) / self._sim_steps
        else:
            self.set_drive_targets(self._filtered_target)

    def before_simulation_step(self):
        self._step += 1
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)


@dataclass
class PDJointPosDelayLagControllerConfig(PDJointPosControllerConfig):
    """PDJointPosControllerConfig + per-env delay & first-order lag knobs.

    `action_delay_steps` and `lag_alpha` are the *default* (centre) values
    used when no domain-randomization writes per-env overrides. The DR
    pathway (BaseRandomEnv._randomize_arm_controller) calls
    `controller.set_per_env_dynamics(env_idx, delay, alpha)` to sample new
    values per reset.

    `max_delay_steps` sizes the circular delay buffer. Set it to the upper
    bound of action_delay_steps_range so randomization can sample up to
    that depth at runtime without re-allocation.
    """

    action_delay_steps: int = ACTION_DELAY_STEPS_DEFAULT
    """Default control-rate FIFO depth (overridden per-env by DR)."""
    lag_alpha: float = LAG_ALPHA_DEFAULT
    """Default EMA mix per control step (overridden per-env by DR)."""
    max_delay_steps: int = 5
    """Capacity of the circular delay buffer (= upper bound of any per-env delay). 5 at 30 Hz = ~165 ms."""
    controller_cls = PDJointPosDelayLagController


@register_agent()
class SO101(BaseAgent):
    uid = "so101"

    # Use the urdf file from this repo
    urdf_path = str(
        Path(__file__).parent
        / "so101.urdf"
    )
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.5, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            gripper_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            moving_jaw_so101_v1_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            finger1_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            finger2_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0, -1.5708, 1.5708, 0.66, -np.pi, -10 * np.pi / 180] # closed gripper
            ),  # Fully open gripper
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
        start=Keyframe(
            qpos=np.array(
                [0, 0, 0, np.pi / 2, -np.pi / 2, 60 * np.pi / 180] # sligtly open gripper
            ),  # Cam up, fully open gripper
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
        zero=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
        extended=Keyframe(
            qpos=np.array(
                [0, -0.7854, 0.7854, 0, 0, 100 * np.pi / 180]
            ),  # Fully open gripper
            pose=sapien.Pose(q=list(euler2quat(0, 0, np.pi / 2))),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=[3.0, 3.0, 3.0, 3.0, 3.0, 100.0],  # gripper ~33x arm torque (initial Squint value)
            normalize_action=False,
        )

        # Arm caps at ±0.0333 rad/step (30 Hz = ~57°/s, baseline-equivalent
        # arm velocity). Gripper cap at ±0.10 rad/step (30 Hz = 3.0 rad/s =
        # ~172°/s, 3× faster than arm) so closure completes in ~0.6 s and
        # the fingers settle on the cube before the arm has swept past it.
        pd_joint_delta_pos = PDJointPosDelayLagControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.0333, -0.0333, -0.0333, -0.0333, -0.0333, -0.10],
            [ 0.0333,  0.0333,  0.0333,  0.0333,  0.0333,  0.10],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=[3.0, 3.0, 3.0, 3.0, 3.0, 100.0],  # gripper ~33x arm torque (initial Squint value)
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # PD joint velocity - Not supported on real SO101
        pd_joint_vel = PDJointVelControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=[-1.0, -1.0, -1.0, -1.0, -1.0, -5.0],
            upper=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
            damping=[1e2] * 6,
            force_limit=[3.0, 3.0, 3.0, 3.0, 3.0, 100.0],  # gripper ~33x arm torque (initial Squint value)
            friction=0,
            normalize_action=True
        )

        # Wrap each controller in a dict with balance_passive_force=False so
        # ManiSkill does NOT disable gravity on the robot links. The default
        # (balance_passive_force=True) is a workaround for PhysX's lack of
        # gravity compensation; with it on, every robot link gets
        # disable_gravity=True, which makes the sim a poor match for Isaac.
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=pd_joint_delta_pos, balance_passive_force=False),
            pd_joint_pos=dict(arm=pd_joint_pos, balance_passive_force=False),
            pd_joint_target_delta_pos=dict(arm=pd_joint_target_delta_pos, balance_passive_force=False),
            pd_joint_vel=dict(arm=pd_joint_vel, balance_passive_force=False),
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        self.finger1_tip = self.robot.links_map["finger1_tip"]
        self.finger2_tip = self.robot.links_map["finger2_tip"]

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_touching(self, object: Actor):
        """Check if the robot is touching an object """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object (more lenient parameters)"""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        """Check if the robot is static (improved for SO101)"""
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
