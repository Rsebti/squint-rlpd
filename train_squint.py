import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["EXCLUDE_TD_FROM_PYTREE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Using lock_\\(\\) in a compiled graph")

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

import math
import random
import time
import glob
from typing import Optional

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import tqdm
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

# Add tasks
import envs
import mani_skill.envs

import utils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Args:
    exp_name: Optional[str] = "baseline"
    """the name of this experiment"""
    agent_name: Optional[str] = "squint"
    """for logging and tracking"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project. None = use the default entity from `wandb login`."""
    wandb_project_name: str = "maniskill-so101"
    """the wandb's project name"""
    wandb_group: str = "SQUINT"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_qpos: bool = False
    """whether to save per-step joint positions during eval to `runs/{run_name}/qpos/` (one .txt per parallel env, sampled at the env control freq of 10 Hz)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder, and to wandb if wandb is set"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from (if set to "wandb" will attempt downloading from wandb)"""
    freeze_encoder_after_frac: float = 0.9
    """fraction of total_timesteps after which the CNN encoder is frozen (no further updates). 0.0 = never freeze."""

    # Environment specific arguments
    env_id: str = "SO101PlaceCube-v1"
    """the id of the environment"""
    env_domain_randomization: bool = True
    """adds domain randomization flag if env supports it"""
    n_distractors: int = 1
    """for cube tasks, number of distractor cubes to spawn alongside the target (0 = single block, 1 = goal + 1 distractor, up to 5 = full palette). Distractors get unique palette colors distinct from the goal. The 6-d goal-color one-hot is always passed to the policy regardless."""
    use_real_bowl: bool = True
    """If True (default), use the SAM-3D bowl mesh at envs/meshes/bowl.obj (CoACD-decomposed dynamic collider, ~15 cm diameter). Pass --no-use_real_bowl to fall back to the parametric rectangular bin. Use the mesh-rebuild script in scripts/mesh_bowl_from_ply.py to regenerate."""
    num_envs: int = 2048
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation. Default True: episodes end the step an env reports success, so the buffer doesn't fill with already-succeeded frames."""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 400_000
    """evaluation frequency in terms of global steps"""
    eval_max_episode_steps: int = 0
    """override max episode steps for evaluation only (0 = use env spec)"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    action_smooth_coef: float = 0.0
    """Coefficient on the per-step action-rate penalty -coef * ||a_t - a_{t-1}||^2 added to the PlaceCube dense reward. Disabled by default — the penalty creates a 'do nothing' attractor for from-scratch training. Enable (e.g. 0.05-0.2) only for fine-tuning a working policy if eval shows jitter."""
    pick_only_reward: bool = False
    """If True, switch the Place env to pick-only mode: reward = reach → grasp → close hard; success = grasped + cube nearly stationary for 1 s. Episode auto-terminates on success. The full pick-and-place reward (z lift / xy-to-bowl / above-bin / release) is skipped entirely."""
    pick_side_approach: bool = False
    """Pick-only side-approach curriculum. Until the FIXED gripper finger touches the cube, the reward is (reach + open_coef·gripper_openness) only — no grasp/strong-grasp incentive — forcing the policy to approach with the gripper fully open and land the fixed finger first. Once touched (sticky for the episode), the normal grasp ladder kicks in. Reduces the failure mode where the policy arrives top-down with the moving finger pre-closed (works in sim, fails in real)."""
    pick_side_approach_open_coef: float = 0.3
    """Coefficient on the gripper-openness reward during the pre-touch phase. Default 0.3 keeps the pre-touch peak (~1.3) below the post-touch grasped-and-clamped peak (1 + strong_grasp_coef = 1.5) so the policy is incentivised to leave the pre-touch phase by touching."""
    drop_penalty_coef: float = 0.0
    """Pick-only mode: penalty applied on every grasped→not-grasped transition (i.e., each drop). Default 0 = disabled; set e.g. 3.0 to penalise fumbles and push the policy to one-shot the grasp."""
    sim_freq: int = 100
    """Physics substep rate (Hz). Default 100 Hz = 10 ms/substep. Won the 2026-05-20 sim2real ablation vs 300 Hz."""
    control_freq: int = 10
    """Control rate (Hz). Episode time per step = 1/control_freq. 7 s episode @ 10 Hz = 70 steps."""
    camera_lag_substeps_min: int = 0
    """Min per-env camera-lag substeps (inclusive). Default 0 = no camera lag (won the 2026-05-20 ablation vs latency-on). At sim_freq=100, 1 substep = 10 ms."""
    camera_lag_substeps_max: int = 0
    """Max per-env camera-lag substeps (inclusive). Default 0 = no camera lag. At sim_freq=100, 5 substeps = 50 ms; at sim_freq=300, 15 substeps = 50 ms."""
    obs_mode: Optional[str] = "rgb"
    """the observation output mode of the environment"""
    render_mode: Optional[str] = "all"
    """the rendering mode of the environment, could be rgb or all"""
    render_height: int = 360
    """sim wrist-camera render height (before downsampling). 360 matches the
    real-camera 16:9 aspect (1920x1080 calibrated) at ¼ resolution."""
    render_width: int = 640
    """sim wrist-camera render width (before downsampling)"""
    image_height: int = 80
    """policy-input image height (after downsampling)"""
    image_width: int = 144
    """policy-input image width (after downsampling). 144/80 = 1.8 (≈ 16:9
    1.778, within 1.2%). Sim renders 16:9 → area-resize 360×640 → 80×144;
    real cam 1920×1080 → area-resize 1920×1080 → 80×144. Both pipelines apply
    the same tiny non-uniform downsample, so the sim→real distortion is
    matched. CNN flatten dim (H≥56 branch) = 64*6*14 = 5376."""
    apply_jitter: bool = True
    """applies color jitter to all input RGB observations (better for sim2real)"""

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    batch_size: int = 512
    """the batch size of sample from the replay memory"""
    num_updates: int = 256
    """num updates per parallel env step"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha for policy"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    bootstrap_at_done: str = "always"
    """bootstrap method when episode ends. Options: ['always', 'never', 'on_truncation']"""
    gamma: float = 0.9
    """the discount factor gamma per step. 0.9 at 10 Hz (baseline 4398ce9 value)."""
    tau: float = 0.01
    """target smoothing coefficient"""
    num_q: int = 2
    """number of Q-networks in the critic ensemble"""
    num_atoms: int = 101
    """number of atoms for distributional RL (C51)"""
    v_min: float = -20.0
    """minimum value for distributional RL support"""
    v_max: float = 20.0
    """maximum value for distributional RL support"""

    # Optimizations
    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""

    # to be filled in runtime
    num_total_iterations: int = 0
    """the number of parallel envs steps given global total timesteps"""


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args, eval_envs, get_action_fn, logger, eval_output_dir, max_episode_steps, global_step, pbar):
    torch.cuda.empty_cache()
    stime = time.perf_counter()
    eval_obs, _ = eval_envs.reset()
    eval_metrics = defaultdict(list)

    qpos_buffers = None
    initial_state = None
    if args.save_qpos:
        qpos_buffers = [[] for _ in range(args.num_eval_envs)]
        base_env = eval_envs.unwrapped
        initial_state = {
            "robot_qpos": base_env.agent.robot.get_qpos().detach().cpu().numpy(),
            "cube_pose": base_env.item.pose.raw_pose.detach().cpu().numpy(),
            "bin_pose": base_env.bin.pose.raw_pose.detach().cpu().numpy(),
        }

    for _ in range(max_episode_steps):
        with torch.no_grad():
            eval_action = get_action_fn(eval_obs['rgb'], eval_obs['state'])
            eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[f'eval/{k}'].append(v[mask])
            if qpos_buffers is not None:
                qpos = eval_envs.unwrapped.agent.robot.get_qpos().detach().cpu().numpy()
                for i in range(args.num_eval_envs):
                    qpos_buffers[i].append(qpos[i])

    if qpos_buffers is not None:
        qpos_dir = os.path.join(
            os.path.dirname(eval_output_dir),
            "test_qpos" if args.evaluate else "qpos",
        )
        init_dir = os.path.join(
            os.path.dirname(eval_output_dir),
            "test_initial_state" if args.evaluate else "initial_state",
        )
        os.makedirs(qpos_dir, exist_ok=True)
        os.makedirs(init_dir, exist_ok=True)
        for i, traj in enumerate(qpos_buffers):
            if not traj:
                continue
            np.savetxt(
                os.path.join(qpos_dir, f"episode_{i}.txt"),
                np.stack(traj, axis=0),
                fmt="%.6f",
            )
            with open(os.path.join(init_dir, f"episode_{i}.txt"), "w") as f:
                f.write("# robot_qpos: 6 joint positions (rad)\n")
                f.write("# cube_pose / bin_pose: x y z qw qx qy qz\n")
                f.write("robot_qpos " + " ".join(f"{v:.6f}" for v in initial_state["robot_qpos"][i]) + "\n")
                f.write("cube_pose " + " ".join(f"{v:.6f}" for v in initial_state["cube_pose"][i]) + "\n")
                f.write("bin_pose " + " ".join(f"{v:.6f}" for v in initial_state["bin_pose"][i]) + "\n")

    eval_d = {}
    for k, v in eval_metrics.items():
        eval_d[k] = torch.stack(v).float().mean()

    pbar.set_description(
        f"success_at_end: {eval_d['eval/success_at_end']:.2f}, "
        f"success_once: {eval_d['eval/success_once']:.2f}, "
        f"return: {eval_d['eval/return']:.2f}"
    )
    eval_time = time.perf_counter() - stime
    eval_d["time/eval_time"] = eval_time

    if args.track and args.capture_video:
        video_files = glob.glob(f"{eval_output_dir}/*.mp4")
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            eval_d["eval/video"] = wandb.Video(latest_video, format="mp4")

    logger.total_eval_time += eval_time
    logger.log(d=eval_d, step=global_step)
    return eval_d


# ─────────────────────────────────────────────────────────────────────────────
#  Network Modules
# ─────────────────────────────────────────────────────────────────────────────

def _orthogonal_via_cpu(weight, gain=1.0):
    # SAPIEN's CUDA init corrupts cuSOLVER state on Blackwell, segfaulting
    # nn.init.orthogonal_ on GPU tensors. Init on CPU and copy back.
    w_cpu = torch.empty_like(weight, device='cpu')
    nn.init.orthogonal_(w_cpu, gain)
    weight.copy_(w_cpu)


def weight_init(m):
    if isinstance(m, nn.Linear):
        _orthogonal_via_cpu(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        _orthogonal_via_cpu(m.weight.data, nn.init.calculate_gain('relu'))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    def __init__(self, n_obs, device=None):
        super().__init__()
        # n_obs is (H, W, C). Non-square inputs (e.g. 32x42 for the landscape
        # wrist camera) are supported; the height is used as the dispatch key
        # to pick a stride profile, then the resulting repr_dim is measured
        # via a dummy forward so the projection layer can size to fit.
        assert len(n_obs) == 3
        self.num_channels = n_obs[2]
        H, W = int(n_obs[0]), int(n_obs[1])
        self.image_size = H

        if H >= 56:
            # Atari/DQN stride profile (8/4, 4/2, 3/1). Designed for 84x84,
            # safe for any H >= 56 (yields >=3 in the spatial dim). The
            # trailing Linear absorbs the width-dependent flatten size via
            # the dummy-forward below. Examples:
            #   64x... → 6x... (the original H=64 branch);
            #   80x144 → 6x14 → 64*6*14 = 5376 flatten;
            #   84x150 → 7x15 → 64*7*15 = 6720 flatten.
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 8, stride=4, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif H in (32, 36):
            # 32×42 (old 4:3) → flatten 1792.  36×64 (new 16:9) → flatten 3840.
            # Same stride profile for both; the trailing Linear projection
            # absorbs the size difference.
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif H == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        else:
            raise ValueError(f"No CNN encoder supported for image height: {H}")

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)
        # Measure flatten size for the given (H, W).
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_channels, H, W, device=device)
            dummy = dummy.contiguous(memory_format=torch.channels_last)
            self.repr_dim = int(self.conv(dummy).shape[-1])

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.contiguous(memory_format=torch.channels_last)
        obs = obs / 255.0 - 0.5
        return self.conv(obs)


class Projection(nn.Module):
    def __init__(self, n_obs, n_state, device=None):
        super().__init__()
        # rgb_proj output: kept at 50 even though we moved from 16x16 → 32x42 →
        # 36x64 inputs. Empirically 50 trains better than 75 here: bigger
        # bottleneck → more saturating Tanh units, weaker gradients, and ~25k
        # extra downstream params (per SAC head ×4) that slow early
        # convergence. The pressure of a tight bottleneck also helps sim2real.
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, 50, device=device), nn.LayerNorm(50, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_state, n_act, device=None):
        super().__init__()
        hidden_dim = 256
        activ = nn.ReLU

        self.proj = Projection(n_obs, n_state, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        action_space = env.unwrapped.single_action_space
        self.register_buffer("action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, rgb, state, get_log_std=False):
        x = self.proj(rgb, state)
        x = self.fc(x)
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
            return mean, log_std
        return mean

    def get_eval_action(self, rgb, state):
        mean = self.forward(rgb, state)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, rgb, state):
        mean, log_std = self.forward(rgb, state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    """Distributional C51 Ensemble-Q-network critic with vmap optimizations."""
    def __init__(self, n_obs, n_state, n_act, num_atoms, v_min, v_max, num_q=2, device=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)

        self.proj = Projection(n_obs, n_state, device=device)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act

        # Build Q-networks, apply weight init, then stack into q_params
        q_nets = [self._build_q_network(q_input_dim, num_atoms, device=device) for _ in range(num_q)]
        for qn in q_nets:
            qn.apply(weight_init)

        # q_params: registered stacked parameter container (what optimizer + vmap both use)
        self.q_params = from_modules(*q_nets, as_module=True)

        # Meta-device template for vmap dispatch (hidden from parameters()/state_dict())
        object.__setattr__(self, '_q_meta', self._build_q_network(q_input_dim, num_atoms, device="meta"))

        # Store architecture string for __repr__
        object.__setattr__(self, '_q_repr', repr(q_nets[0]))

    def __repr__(self):
        """Pretty module printing"""
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  (proj): {self.proj}")
        for i in range(self.num_q):
            lines.append(f"  (q{i}): {self._q_repr}")
        lines.append(")")
        return "\n".join(lines)

    def _build_q_network(self, input_dim, num_atoms, device=None):
        """Build a single Q-network. Used for q_nets, meta template."""
        hidden_dim = 512
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms, device=device)
        )

    def _vmap_q(self, params, x):
        """Single Q-network forward through meta template. Dispatched by vmap."""
        with params.to_module(self._q_meta):
            return self._q_meta(x)

    def forward(self, rgb_features, state, actions):
        """Batched forward: [num_q, batch, num_atoms]. Full gradient flow through all params."""
        proj = self.proj(rgb_features, state)
        x = torch.cat([proj, actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self.q_params, x)

    def get_q_values(self, rgb_features, state, actions, detach_critic=False):
        """Expected Q-values: [num_q, batch].

        Args:
            detach_critic: If True, freezes critic weights (proj + Q-networks) while
                preserving gradients through actions. Used for actor policy gradient.
        """
        if detach_critic:
            with torch.no_grad():
                proj = self.proj(rgb_features, state) 
            x = torch.cat([proj, actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self.q_params.data, x)
        else:
            logits = self.forward(rgb_features, state, actions)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.q_support, dim=-1)

    def categorical(self, rgb_features, state, actions, rewards, bootstrap, discount):
        """C51 categorical projection: [num_q, batch, num_atoms].
        Called under no_grad for target computation."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower = torch.where(torch.logical_and(lower > 0, is_integer), lower - 1, lower)
        upper = torch.where(torch.logical_and(lower == 0, is_integer), upper + 1, upper)

        # Batched forward through all Q-networks via vmap
        logits = self.forward(rgb_features, state, actions)  # [num_q, batch, atoms]
        next_dists = F.softmax(logits, dim=-1)

        # Fused projection: reshape to [num_q*batch, atoms]
        total_batch = self.num_q * batch_size
        next_dists_flat = next_dists.reshape(-1, self.num_atoms)
        offset = torch.arange(total_batch, device=device).unsqueeze(1) * self.num_atoms

        lower_exp = lower.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        upper_exp = upper.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        b_exp = b.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)

        max_index = total_batch * self.num_atoms - 1
        lower_indices = torch.clamp((lower_exp + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper_exp + offset).view(-1), 0, max_index)

        proj_dist_flat = torch.zeros_like(next_dists_flat)
        proj_dist_flat.view(-1).index_add_(0, lower_indices, (next_dists_flat * (upper_exp.float() - b_exp)).view(-1))
        proj_dist_flat.view(-1).index_add_(0, upper_indices, (next_dists_flat * (b_exp - lower_exp.float())).view(-1))

        return proj_dist_flat.reshape(self.num_q, batch_size, self.num_atoms)


# ─────────────────────────────────────────────────────────────────────────────
#  Deployment Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DeployAgent(nn.Module):
    """Standalone deployment wrapper for deploy.py file. Handles downsampling and inference."""

    def __init__(self, sim_env, sample_obs, target_image_size=(36, 64), device=None):
        super().__init__()
        self.device = device
        # Accept int (square) or (H, W).
        if isinstance(target_image_size, int):
            self.target_h, self.target_w = target_image_size, target_image_size
        else:
            self.target_h, self.target_w = int(target_image_size[0]), int(target_image_size[1])
        self.target_image_size = (self.target_h, self.target_w)

        n_act = np.prod(sim_env.unwrapped.single_action_space.shape)
        n_obs_shape = sample_obs['rgb'].shape
        c = n_obs_shape[3] if len(n_obs_shape) == 4 else n_obs_shape[2]
        n_obs = (self.target_h, self.target_w, c)
        n_state = np.prod(sample_obs['state'].shape[1:]) if len(sample_obs['state'].shape) > 1 else sample_obs['state'].shape[0]

        self.encoder = CNNEncoder(n_obs, device)
        self.actor = Actor(sim_env, n_obs=self.encoder.repr_dim, n_state=n_state, n_act=n_act, device=self.device)

    def load_checkpoint(self, checkpoint, checkpoint_config=None, version=None):
        if checkpoint.lower() == "wandb":
            assert checkpoint_config is not None, "Need checkpoint_config to download from wandb"
            cc = checkpoint_config
            artifact_path = f"{cc['wandb_entity']}/{cc['wandb_project_name']}/model_{cc['agent_name']}_{cc['env_id']}_{cc['seed']}:{cc['version']}"
            print(artifact_path)
            local_path = Logger().download_checkpoint(artifact_path)
            local_path = f"{local_path}/ckpt.pt"
            ckpt = torch.load(local_path, map_location=self.device)
        else:
            ckpt = torch.load(checkpoint, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor.load_state_dict(ckpt['actor'])
        print(f"Loaded checkpoint from {checkpoint} at step {ckpt['global_step']}")

    def downsample_rgb(self, rgb):
        if rgb.shape[-3] == self.target_h and rgb.shape[-2] == self.target_w:
            return rgb
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(self.target_h, self.target_w), mode='area')
        rgb = rgb.permute(0, 2, 3, 1).to(torch.uint8)
        if squeeze:
            rgb = rgb.squeeze(0)
        return rgb

    def get_action(self, obs):
        rgb = self.downsample_rgb(obs['rgb'])
        with torch.no_grad():
            rgb = self.encoder(rgb)
            return self.actor.get_eval_action(rgb, obs['state'])

    def forward(self, obs):
        return self.get_action(obs)


# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_wandb=False):
        self.log_wandb = log_wandb
        self.start_time = time.perf_counter()
        self.total_eval_time = 0 # to subtract from total wall_time

    @property
    def wall_time(self):
        return time.perf_counter() - self.start_time - self.total_eval_time

    def log(self, d, step):
        if self.log_wandb:
            d["time/wall_time"] = self.wall_time
            wandb.log(d, step=step)

    def close(self):
        if self.log_wandb:
            wandb.finish()

    def upload_checkpoint(self, model_path: str, model_name="model_checkpoint"):
        if self.log_wandb:
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            artifact.wait()
            print(f"Uploaded checkpoint {model_name} to wandb")

    def download_checkpoint(self, artifact_path: str):
        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download()
        run = artifact.logged_by()
        local_time = datetime.fromisoformat(run.createdAt.replace('Z', '+00:00')).astimezone()
        print(f"Downloaded checkpoint at {artifact_dir} from experiment: {run.config['exp_name']} at: {local_time}")
        return artifact_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_total_iterations = int(args.total_timesteps // args.num_envs)
    assert args.num_updates > 0, "No updates will be made to the model with the current setup"

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    model_path = os.path.abspath(f"runs/{run_name}/ckpt.pt")
    best_model_path = os.path.abspath(f"runs/{run_name}/ckpt_best.pt")
    best_success_at_end = -1.0  # tracks the high-water mark for success_at_end
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ──────────────────────────────────────────────────
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu",
                      sensor_configs=dict(width=args.render_width, height=args.render_height))
    eval_env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu",
                           sensor_configs=dict(width=args.render_width, height=args.render_height),
                           human_render_camera_configs=dict(shader_pack="default", width=args.render_width, height=args.render_height))
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
        eval_env_kwargs["control_mode"] = args.control_mode
    if args.env_domain_randomization:
        env_kwargs["domain_randomization"] = True
        eval_env_kwargs["domain_randomization"] = True
    if "PlaceCube" in args.env_id:
        env_kwargs["n_distractors"] = args.n_distractors
        eval_env_kwargs["n_distractors"] = args.n_distractors
        env_kwargs["use_real_bowl"] = args.use_real_bowl
        eval_env_kwargs["use_real_bowl"] = args.use_real_bowl
        env_kwargs["action_smooth_coef"] = args.action_smooth_coef
        eval_env_kwargs["action_smooth_coef"] = args.action_smooth_coef
        env_kwargs["pick_only_reward"] = args.pick_only_reward
        eval_env_kwargs["pick_only_reward"] = args.pick_only_reward
        env_kwargs["pick_side_approach"] = args.pick_side_approach
        eval_env_kwargs["pick_side_approach"] = args.pick_side_approach
        env_kwargs["pick_side_approach_open_coef"] = args.pick_side_approach_open_coef
        eval_env_kwargs["pick_side_approach_open_coef"] = args.pick_side_approach_open_coef
        env_kwargs["drop_penalty_coef"] = args.drop_penalty_coef
        eval_env_kwargs["drop_penalty_coef"] = args.drop_penalty_coef
    # Physics + control rate (passes through to BaseRandomEnv → SimConfig).
    env_kwargs["sim_freq"] = args.sim_freq
    eval_env_kwargs["sim_freq"] = args.sim_freq
    env_kwargs["control_freq"] = args.control_freq
    eval_env_kwargs["control_freq"] = args.control_freq
    # Camera-lag DR override: set both to 0 to disable image latency entirely.
    lag_range = (args.camera_lag_substeps_min, args.camera_lag_substeps_max)
    env_kwargs.setdefault("domain_randomization_config", {})
    eval_env_kwargs.setdefault("domain_randomization_config", {})
    env_kwargs["domain_randomization_config"]["camera_lag_substeps_range"] = lag_range
    eval_env_kwargs["domain_randomization_config"]["camera_lag_substeps_range"] = lag_range

    _make_steps = args.eval_max_episode_steps if args.eval_max_episode_steps > 0 else None
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq,
                    **({"max_episode_steps": _make_steps} if _make_steps else {}),
                    **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq,
                         **({"max_episode_steps": _make_steps} if _make_steps else {}),
                         **eval_env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)

    if (args.render_height, args.render_width) != (args.image_height, args.image_width):
        envs = utils.DownsampleObsWrapper(envs, target_size=(args.image_height, args.image_width))
        eval_envs = utils.DownsampleObsWrapper(eval_envs, target_size=(args.image_height, args.image_width))
    if args.apply_jitter:
        envs = utils.ColorJitterWrapper(envs)
        eval_envs = utils.ColorJitterWrapper(eval_envs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_output_dir = None
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"runs/{run_name}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        # Overlay the current goal color on each rendered frame before the recorder tiles them.
        envs = utils.GoalColorOverlayWrapper(envs)
        eval_envs = utils.GoalColorOverlayWrapper(eval_envs)
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // max_episode_steps) % args.save_train_video_freq == 0
            envs = utils.ClockedRecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False,
                                 save_video_trigger=save_video_trigger, max_steps_per_video=max_episode_steps, video_fps=20)
        eval_envs = utils.ClockedRecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory,
                                  save_video=args.capture_video, trajectory_name="trajectory",
                                  max_steps_per_video=max_episode_steps, video_fps=20)

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)

    n_act = math.prod(envs.unwrapped.single_action_space.shape)
    n_channels = envs.unwrapped.single_observation_space['rgb'].shape[2]
    n_obs = (args.image_height, args.image_width, n_channels)
    n_state = math.prod(envs.unwrapped.single_observation_space['state'].shape)
    assert isinstance(envs.unwrapped.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ── Logger ─────────────────────────────────────────────────────────────
    if not args.evaluate:
        print("Running training")
        if args.track:
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,
                                     reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**eval_env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,
                                          reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.eval_partial_reset)
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=False,
                       config=config, name=run_name, save_code=True, group=args.wandb_group,
                       tags=[args.wandb_group, args.agent_name, args.env_id, f"seed={args.seed}"])
    else:
        print("Running evaluation")
    logger = Logger(log_wandb=(args.track and not args.evaluate))

    # ── Instantiate modules ────────────────────────────────────────────────

    encoder = CNNEncoder(n_obs=n_obs, device=device)
    actor = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device)
    critic = Critic(n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act,
                    num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                    num_q=args.num_q, device=device)

    # Entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.unwrapped.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    # Load checkpoint
    if args.checkpoint is not None:
        if args.checkpoint.lower() == "wandb":
            artifact_path = f"{args.wandb_entity}/{args.wandb_project_name}/model_{args.agent_name}_{args.env_id}_{args.seed}:latest"
            print(artifact_path)
            local_path = logger.download_checkpoint(artifact_path)
            local_path = f"{local_path}/ckpt.pt"
            ckpt = torch.load(local_path, map_location=device)
        else:
            ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                log_alpha.copy_(ckpt['log_alpha'])
                alpha.copy_(log_alpha.exp())
        print(f"Loaded checkpoint from {args.checkpoint} at step {ckpt['global_step']}")

    # ── Inference copies (weight-sharing via from_module) ──────────────────

    encoder_detach = CNNEncoder(n_obs=n_obs, device=device)
    encoder_eval = CNNEncoder(n_obs=n_obs, device=device).eval()
    from_module(encoder).data.to_module(encoder_detach)
    from_module(encoder).data.to_module(encoder_eval)

    actor_detach = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device)
    actor_eval = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).eval()
    from_module(actor).data.to_module(actor_detach)
    from_module(actor).data.to_module(actor_eval)

    # Target critic 
    critic_target = Critic(n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act,
                           num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                           num_q=args.num_q, device=device)
    critic_target.load_state_dict(critic.state_dict())
    critic_online_params = list(critic.parameters())
    critic_target_params = list(critic_target.parameters())

    # ── Inference functions ────────────────────────────────────────────────

    def get_rollout_action(rgb, state):
        rgb_feat = encoder_detach(rgb)
        action, _, _ = actor_detach.get_action(rgb_feat, state)
        return action

    def get_eval_action(rgb, state):
        rgb_feat = encoder_eval(rgb)
        return actor_eval.get_eval_action(rgb_feat, state)

    # ── Optimizers ─────────────────────────────────────────────────────────

    critic_optimizer = optim.Adam(list(critic.parameters()) + list(encoder.parameters()),
                             lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()),
                                 lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    freeze_encoder_step = int(args.total_timesteps * args.freeze_encoder_after_frac) if args.freeze_encoder_after_frac > 0 else 0
    critic_only_optimizer = None
    if freeze_encoder_step > 0:
        critic_only_optimizer = optim.Adam(list(critic.parameters()),
                                 lr=args.q_lr, capturable=args.cudagraphs and not args.compile)

    # ── Replay buffer ──────────────────────────────────────────────────────

    # TODO: Buffer stores current and next observations, should only store one
    buffer_mem = utils.calc_buffer_memory(
        rgb_dim=np.prod(n_obs), 
        state_dim=n_state, 
        action_dim=n_act,
        max_length=min(args.buffer_size, args.total_timesteps), 
        rgb_dtype=np.uint8,
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    # ── Print summary ──────────────────────────────────────────────────────

    print("-----------------------")
    print(args)
    print("-----------------------")
    print("Squint")
    print("-----------------------")
    for mod in [encoder, actor, critic]:
        print(mod)
    print(f"Task: {args.env_id}, Control mode: {envs.unwrapped._control_mode}")
    print(f"Observations: {n_obs}, State: {n_state}, Actions: {n_act}")
    print(f"Buffer memory required: {buffer_mem:.2f} GB")
    print(f"Device: {device}")
    print("-----------------------")

    # ── Update functions ───────────────────────────────────────────────────

    def update_main(data):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                next_obs = encoder(data["next_observations"]['rgb'])
                next_state = data["next_observations"]['state']
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs, next_state)

                bootstrap = (~data["dones"]).float()
                discount = args.gamma
                rewards = data["rewards"].flatten()

                entropy_bonus = alpha * next_state_log_pi.flatten()
                rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

                target_distributions = critic_target.categorical(
                    next_obs, next_state, next_state_actions,
                    rewards_with_entropy, bootstrap, discount
                )

            obs = encoder(data["observations"]['rgb'])
            state = data["observations"]['state']

            # Shape: [num_q, batch, num_atoms]
            q_outputs = critic(obs, state, data["actions"])
            q_log_probs = F.log_softmax(q_outputs, dim=-1)

            # Cross-entropy: sum over num_atoms, mean over batch → [num_q]
            q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)
            
            # Sum over Q-networks losses
            critic_loss = q_losses.sum()

            # Logging q-value metrics
            with torch.no_grad():
                q_probs = F.softmax(q_outputs, dim=-1)
                q_values = torch.sum(q_probs * critic.q_support, dim=-1)
                q_max = q_values.max()
                q_min = q_values.min()

        # Update critic and encoder
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if args.autotune:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs, state)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            # Update actor entropy
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            alpha.copy_(log_alpha.detach().exp())
        else:
            alpha_loss = torch.tensor(0.0, device=device)

        return TensorDict(critic_loss=critic_loss.detach(), q_max=q_max, q_min=q_min,
                          alpha=alpha.detach(), alpha_loss=alpha_loss.detach(), 
                          encoded_rgb=obs.detach())

    def update_actor(data, encoded_rgb):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            state = data["observations"]["state"]
            obs = encoded_rgb

            pi, log_pi, _ = actor.get_action(obs, state)
            q_values = critic.get_q_values(obs, state, pi, detach_critic=True)
            
            # Mean (No CDQ)
            critic_value = q_values.mean(dim=0) 

            actor_loss = (alpha * log_pi - critic_value).mean()

        # Update actor 
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return TensorDict(actor_loss=actor_loss.detach())

    # Frozen-encoder variant of update_main. Built only when freeze is enabled.
    # Differences vs update_main: encoder forward wrapped in no_grad (no encoder
    # gradients computed), and uses critic_only_optimizer (no encoder params).
    # The next_obs branch was already in no_grad; we only change the obs branch.
    def update_main_frozen(data):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                next_obs = encoder(data["next_observations"]['rgb'])
                next_state = data["next_observations"]['state']
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs, next_state)

                bootstrap = (~data["dones"]).float()
                discount = args.gamma
                rewards = data["rewards"].flatten()

                entropy_bonus = alpha * next_state_log_pi.flatten()
                rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

                target_distributions = critic_target.categorical(
                    next_obs, next_state, next_state_actions,
                    rewards_with_entropy, bootstrap, discount
                )

                obs = encoder(data["observations"]['rgb'])

            state = data["observations"]['state']

            q_outputs = critic(obs, state, data["actions"])
            q_log_probs = F.log_softmax(q_outputs, dim=-1)
            q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)
            critic_loss = q_losses.sum()

            with torch.no_grad():
                q_probs = F.softmax(q_outputs, dim=-1)
                q_values = torch.sum(q_probs * critic.q_support, dim=-1)
                q_max = q_values.max()
                q_min = q_values.min()

        critic_only_optimizer.zero_grad()
        critic_loss.backward()
        critic_only_optimizer.step()

        if args.autotune:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs, state)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            alpha.copy_(log_alpha.detach().exp())
        else:
            alpha_loss = torch.tensor(0.0, device=device)

        return TensorDict(critic_loss=critic_loss.detach(), q_max=q_max, q_min=q_min,
                          alpha=alpha.detach(), alpha_loss=alpha_loss.detach(),
                          encoded_rgb=obs.detach())

    # ── Compile & CudaGraphs ──────────────────────────────────────────────

    if args.compile:
        update_main = torch.compile(update_main)
        update_actor = torch.compile(update_actor)
        get_rollout_action = torch.compile(get_rollout_action)
        get_eval_action = torch.compile(get_eval_action)
        if freeze_encoder_step > 0:
            update_main_frozen = torch.compile(update_main_frozen)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main)
        update_actor = CudaGraphModule(update_actor)
        if freeze_encoder_step > 0:
            update_main_frozen = CudaGraphModule(update_main_frozen)

    # ── Training loop ──────────────────────────────────────────────────────

    obs, _ = envs.reset(seed=args.seed)
    eval_envs.reset(seed=args.seed)

    global_step = 0
    pbar = tqdm.tqdm(total=args.total_timesteps, desc="steps")
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    d = {}
    encoder_frozen = False

    for iteration in range(args.num_total_iterations + 2):  # +2 for final eval
        # Evaluate
        if args.eval_freq > 0 and ((global_step - args.num_envs) // args.eval_freq) < (global_step // args.eval_freq):
            eval_d = evaluate(args, eval_envs, get_eval_action, logger, eval_output_dir,
                              max_episode_steps, global_step, pbar)
            if args.evaluate:
                break
            if args.save_model:
                ckpt_payload = {
                    'encoder': encoder.state_dict(),
                    'actor': actor.state_dict(),
                    'critic': critic_target.state_dict(),
                    'log_alpha': log_alpha,
                    'global_step': global_step,
                }
                torch.save(ckpt_payload, model_path)
                # Per-eval history (one file per eval, never overwritten).
                step_path = model_path.replace(
                    "ckpt.pt", f"ckpt_step{global_step:09d}.pt")
                torch.save(ckpt_payload, step_path)
                # Best-by-success_at_end (overwrites itself on new highs).
                # SAC commonly collapses after a peak; this lets you recover the
                # peak policy without manually scanning per-eval snapshots.
                cur_success = float(eval_d.get('eval/success_at_end', -1.0))
                msg_extra = ""
                new_best = cur_success > best_success_at_end
                if new_best:
                    best_success_at_end = cur_success
                    torch.save(ckpt_payload, best_model_path)
                    msg_extra = f"  (NEW BEST success_at_end={cur_success:.3f})"
                print(f"Step {global_step}: ckpt saved to {model_path} and {step_path}{msg_extra}")
                # Crash-safety: push the latest ckpt to wandb every eval (not
                # just at training end). Same artifact name → wandb versions
                # it (v0, v1, ... :latest) without exploding storage. Best
                # ckpt is uploaded under a separate name when it changes.
                if args.track:
                    base = f"model_{args.agent_name}_{args.env_id}_{args.seed}"
                    logger.upload_checkpoint(model_path=model_path, model_name=base)
                    if new_best and os.path.exists(best_model_path):
                        logger.upload_checkpoint(model_path=best_model_path, model_name=f"{base}_best")

        # Collect
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            actions = get_rollout_action(obs['rgb'], obs['state'])

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = {'rgb': next_obs['rgb'].clone(), 'state': next_obs['state'].clone()}

        # Determine bootstrap behavior 
        if args.bootstrap_at_done == 'never':
            need_final_obs = terminations | truncations
            dones = terminations | truncations
        elif args.bootstrap_at_done == 'always':
            need_final_obs = terminations | truncations
            dones = torch.zeros_like(terminations, dtype=torch.bool)
        else: # 'on_truncation' - only stop bootstrap on true termination, bootstrap on truncation
            need_final_obs = truncations & (~terminations)
            dones = terminations

        if "final_info" in infos:
            real_next_obs['rgb'][need_final_obs] = infos["final_observation"]['rgb'][need_final_obs]
            real_next_obs['state'][need_final_obs] = infos["final_observation"]['state'][need_final_obs]

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            dones=dones,
            batch_size=rewards.shape[0],
            device=device,
        )
        rb.extend(transition)

        # Setting next as current obs
        obs = next_obs

        # Freeze encoder once we cross the configured fraction of training.
        if freeze_encoder_step > 0 and not encoder_frozen and global_step >= freeze_encoder_step:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()
            encoder_frozen = True
            print(f"Step {global_step}: encoder frozen (no further CNN updates).")

        # Training updates
        if global_step > args.learning_starts:
            for grad_step in range(args.num_updates):
                data = rb.sample(args.batch_size)

                # update critic and encoder and actor entropy
                out_main = update_main_frozen(data) if encoder_frozen else update_main(data)
                encoded_rgb = out_main.pop("encoded_rgb", None)

                # update actor (policy)
                if grad_step % args.policy_frequency == 0:
                    out_main.update(update_actor(data, encoded_rgb))

                # update target networks
                if grad_step % args.target_network_frequency == 0:
                    with torch.no_grad():
                        torch._foreach_lerp_(critic_target_params, critic_online_params, args.tau)

                d.update(out_main)

        # Log
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                d[f"train/{k}"] = v[done_mask].float().mean()
            # logging for terminal bar
            max_ep_ret = max(infos["final_info"]["episode"]["return"][done_mask])
            avg_returns.extend(infos["final_info"]["episode"]["return"][done_mask])
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            # Calculate wall_time metrics
            sps = global_step / logger.wall_time
            d["time/sps"] = sps
            pbar.set_description(f"{sps: 4.4f} sps, " + desc)
            logger.log(d=d, step=global_step)

        # Increment counters
        pbar.update(args.num_envs)
        global_step += args.num_envs

    # Upload final checkpoint(s) to wandb. Best ckpt goes under a separate
    # artifact name so deploy.py can target either the final or peak policy.
    if args.save_model:
        base = f"model_{args.agent_name}_{args.env_id}_{args.seed}"
        if os.path.exists(model_path):
            logger.upload_checkpoint(model_path=model_path, model_name=base)
        else:
            print(f"WARNING: Checkpoint file not found at {model_path}, skipping upload")
        if os.path.exists(best_model_path):
            logger.upload_checkpoint(model_path=best_model_path, model_name=f"{base}_best")
        else:
            print(f"WARNING: Best checkpoint not found at {best_model_path}, skipping upload")

    print("Finishing logger...")
    logger.close()
    print("Starting cleanup...")
    try:
        envs.close()
        eval_envs.close()
    except:
        pass
    print("Cleanup complete. Exiting.")