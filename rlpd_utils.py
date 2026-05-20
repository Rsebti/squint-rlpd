"""Offline buffer loader and synthetic-demo generator for RLPD on top of Squint.

The offline buffer that ``train_rlpd.py`` consumes is just a torchrl
``ReplayBuffer`` populated once at startup. This module is responsible for
turning whatever demo source the user has — a local ``.pt`` bundle, a HuggingFace
LeRobot dataset, or rollouts of an existing Squint checkpoint — into the
TensorDict layout the buffer expects.

Layout of a single transition (matches the online buffer in ``train_rlpd.py``):

    TensorDict(
        observations={"rgb": uint8 [B, H, W, 3], "state": float32 [B, n_state]},
        next_observations={"rgb": uint8 [B, H, W, 3], "state": float32 [B, n_state]},
        actions=float32 [B, n_act],
        rewards=float32 [B],
        dones=bool [B],
        batch_size=B,
    )

Public API
----------
``load_offline_transitions(path, obs_shape, state_dim, action_dim, reward_mode, device)``
    Entry point called from ``train_rlpd.py``. Auto-detects the source format.

``save_offline_bundle(path, td)``
    Writes a TensorDict bundle to disk (CPU tensors).

``collect_synthetic_offline_from_ckpt(...)``
    CLI-style helper: load a trained Squint checkpoint, roll it out in the same
    env used by ``train_rlpd``, keep only successful episodes, save a ``.pt``
    bundle ready to load. Useful while real teleop demos are not yet available.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import TensorDict


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_offline_transitions(
    path: str,
    obs_shape: Tuple[int, int, int],
    state_dim: int,
    action_dim: int,
    reward_mode: str = "sparse",
    device: Optional[torch.device] = None,
    lerobot_kwargs: Optional[Dict] = None,
) -> TensorDict:
    """Load offline transitions from ``path`` and return a flat TensorDict.

    The returned TensorDict has ``batch_size=[N]`` where N is the total number
    of transitions (summed across all episodes). The buffer in
    ``train_rlpd.py`` calls ``offline_rb.extend(td)`` on this object.

    ``path`` formats supported:
      * Local ``.pt`` file produced by :func:`save_offline_bundle`. The bundle
        is assumed to already be in the expected layout.
      * HF LeRobot dataset id (any string that is not a path on disk). Loading
        currently raises ``NotImplementedError`` with guidance — finish the
        ``_load_lerobot_dataset`` conversion once the team's real demo schema
        is fixed.

    ``reward_mode``:
      * ``sparse`` — relabel rewards as 0 everywhere except the last step of
        each successful episode, which gets +1. RLPD handles sparse rewards.
      * ``recompute`` — use whatever reward field is stored in the bundle
        (assumes the source recorded the env's dense reward). If the bundle
        already carries dense rewards this is the right choice.
    """
    if reward_mode not in ("sparse", "recompute"):
        raise ValueError(f"reward_mode must be 'sparse' or 'recompute', got {reward_mode!r}")

    if os.path.isfile(path):
        td = _load_pt_bundle(path)
    else:
        # Treat as an HF dataset id.
        td = _load_lerobot_dataset(
            path, obs_shape, state_dim, action_dim,
            **(lerobot_kwargs or {}),
        )

    # If the bundle was decoded with the full 58D layout but the env exposes
    # a smaller state (e.g. 21D = noisy_qpos + controller_target + goal_color
    # + bowl_xyz_robot_frame when privileged dims are hidden), slice the
    # leading dims to match. The 21D prefix is exactly the env's state vector.
    bundle_dim = td["observations", "state"].shape[-1]
    if bundle_dim > state_dim:
        td["observations", "state"] = td["observations", "state"][:, :state_dim].contiguous()
        td["next_observations", "state"] = td["next_observations", "state"][:, :state_dim].contiguous()
        print(f"Sliced offline bundle state {bundle_dim}D → {state_dim}D to match env.")

    _validate_transitions(td, obs_shape=obs_shape, state_dim=state_dim, action_dim=action_dim)

    if reward_mode == "sparse":
        td = _relabel_rewards_sparse(td)

    # Drop any extra fields not in the online-buffer schema so that
    # torch.cat([online_data, offline_data]) in train_rlpd.py works
    # (it requires identical key sets in strict mode).
    online_keys = {"observations", "next_observations", "actions", "rewards", "dones"}
    for k in list(td.keys()):
        if k not in online_keys:
            del td[k]

    if device is not None:
        td = td.to(device)
    return td


# ─────────────────────────────────────────────────────────────────────────────
# Bundle I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_offline_bundle(path: str, td: TensorDict) -> None:
    """Save ``td`` (a flat transitions TensorDict) to disk as a torch file.

    Keeps tensors on CPU so the file is portable across devices. The training
    script lifts them back to ``device`` when populating the buffer.
    """
    cpu_td = td.detach().to("cpu")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(cpu_td, path)
    print(f"Saved {cpu_td.batch_size[0]} offline transitions to {path}")


def _load_pt_bundle(path: str) -> TensorDict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, TensorDict):
        raise TypeError(
            f"{path} did not contain a TensorDict (got {type(obj).__name__}). "
            "Use rlpd_utils.save_offline_bundle to write bundles in the expected layout."
        )
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Reward relabelling
# ─────────────────────────────────────────────────────────────────────────────

def _relabel_rewards_sparse(td: TensorDict) -> TensorDict:
    """Replace ``td['rewards']`` with sparse +1-on-success labelling.

    Convention: a transition gets reward +1 iff ``dones[i] is True`` AND the
    optional ``success`` field is True (or absent, in which case every done is
    treated as success). All other transitions get 0.
    """
    dones = td["dones"].to(torch.bool)
    if "success" in td.keys():
        success = td["success"].to(torch.bool)
    else:
        success = dones  # treat every terminal step as a success
    rewards = torch.where(dones & success, torch.ones_like(success, dtype=torch.float32),
                          torch.zeros_like(success, dtype=torch.float32))
    td = td.clone(recurse=False)
    td.set("rewards", rewards)
    return td


# ─────────────────────────────────────────────────────────────────────────────
# Shape validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_transitions(
    td: TensorDict,
    obs_shape: Tuple[int, int, int],
    state_dim: int,
    action_dim: int,
) -> None:
    H, W, C = obs_shape
    n = td.batch_size[0]

    def _check(field: str, expected_shape: Sequence[int], dtype: Optional[torch.dtype] = None) -> None:
        if field not in td.keys(include_nested=True):
            raise KeyError(f"Offline transitions missing required field {field!r}")
        t = td.get(field)
        if isinstance(t, TensorDict):
            return  # nested check covered by recursive _check calls below
        actual = tuple(t.shape)
        if actual != tuple(expected_shape):
            raise ValueError(f"{field} has shape {actual}, expected {tuple(expected_shape)}")
        if dtype is not None and t.dtype != dtype:
            raise ValueError(f"{field} has dtype {t.dtype}, expected {dtype}")

    _check(("observations", "rgb"), (n, H, W, C), dtype=torch.uint8)
    _check(("observations", "state"), (n, state_dim), dtype=torch.float32)
    _check(("next_observations", "rgb"), (n, H, W, C), dtype=torch.uint8)
    _check(("next_observations", "state"), (n, state_dim), dtype=torch.float32)
    _check("actions", (n, action_dim), dtype=torch.float32)
    _check("rewards", (n,), dtype=torch.float32)
    _check("dones", (n,), dtype=torch.bool)


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot HF dataset → ManiSkill layout
# ─────────────────────────────────────────────────────────────────────────────

# Action range from envs/robot/so101.py PDJointPosDelayLagControllerConfig:
#   arm joints (5): ±0.05 rad per step
#   gripper:        ±0.20 rad per step
# These match the limits in the controller config and are baked into the
# normalize_action mapping (sim policy outputs [-1, 1], internally rescaled
# to the per-joint ranges below).
_SIM_ACTION_LOWER = np.array([-0.05, -0.05, -0.05, -0.05, -0.05, -0.20], dtype=np.float32)
_SIM_ACTION_UPPER = np.array([ 0.05,  0.05,  0.05,  0.05,  0.05,  0.20], dtype=np.float32)

# State vector layout (58D, baseline: pick_only_reward=True, n_distractors=0,
# use_real_bowl=True, domain_randomization=False). Slices given as [start, end).
_STATE_SLICES = {
    "noisy_qpos":           (0,  6),    # current joint positions (rad), gripper in [0, ~1.6] rad
    "controller_target":    (6,  12),   # PD controller target qpos (rad) — the running target
    "goal_color":           (12, 18),   # 6D one-hot over target color
    "bowl_xyz_robot_frame": (18, 21),   # bowl center in robot base frame (m)
    "qvel":                 (21, 27),   # joint velocities (rad/s)
    "is_item_grasped":      (27, 28),   # 0/1 grasp flag
    "item_pose":            (28, 35),   # [xyz, qx, qy, qz, qw] in world frame
    "bin_pose":             (35, 42),   # [xyz, qx, qy, qz, qw] in world frame
    "tcp_pose":             (42, 49),   # [xyz, qx, qy, qz, qw] in world frame
    "tcp_to_item_grip_pos": (49, 52),   # tcp_xyz - item_xyz (m)
    "tcp_to_bin_pos":       (52, 55),   # tcp_xyz - bin_xyz (m)
    "item_to_bin_pos":      (55, 58),   # item_xyz - bin_xyz (m)
}
_EXPECTED_STATE_DIM = 58


def _load_lerobot_dataset(
    repo_id: str,
    obs_shape: Tuple[int, int, int],
    state_dim: int,
    action_dim: int,
    gripper_in_degrees: bool = True,
    grasp_threshold_rad: float = 0.5,
    keep_episode_indices: Optional[Sequence[int]] = None,
    max_episodes: Optional[int] = None,
) -> TensorDict:
    """Convert ``Rsebti/projet3_demos_v1``-style LeRobot datasets into the
    ManiSkill offline-buffer layout.

    Expected source schema (see ``meta/info.json`` of the HF dataset):
      * ``action``: float32 [6] — absolute joint position TARGETS in degrees
        (``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
        gripper].pos``).
      * ``observation.state``: float32 [6] — current joint positions, same
        units / ordering as ``action``.
      * ``observation.images.wrist``: uint8 [480, 640, 3] — wrist camera RGB
        at 30 FPS.
      * ``episode_index``, ``frame_index``, ``timestamp``: episode bookkeeping.

    Conversions applied here:
      * **Image** — area-resize 480×640 → ``(obs_shape[0], obs_shape[1])``
        (default 80×144) using ``F.interpolate(mode='area')``, identical to
        Squint's ``DownsampleObsWrapper`` so the offline RGB distribution
        matches the online one.
      * **State (58D)** — qpos and controller-target qpos from the demo
        (deg → rad), qvel via finite difference, TCP pose via the pure-numpy
        FK in ``so101_fk.py``, and zeros for the cube/bowl-dependent slices
        (Squint's Sim2RealEnv keeps those filled by its shadow sim at deploy
        time; offline-RL doesn't have that signal, but the policy is RGB-led).
      * **Action** — ``(deg2rad(action) - deg2rad(state)) / step_limit``,
        clipped to [-1, 1], where ``step_limit`` matches the sim's
        ``pd_joint_target_delta_pos`` per-step caps (±0.05 rad arm,
        ±0.20 rad gripper).

    Args:
        repo_id: HuggingFace dataset id, e.g. ``"Rsebti/projet3_demos_v1"``.
        gripper_in_degrees: Whether the recorded gripper position is in
            degrees (True; matches LeRobot's standard so101 follower
            calibration) or already in radians (False). Watch the
            sanity-check print for outlier values.
        grasp_threshold_rad: Gripper qpos above which we set
            ``is_item_grasped=1``. Calibrate by inspecting the demo
            distribution — the printed stats include gripper min/median/max.
        keep_episode_indices / max_episodes: optional filters for fast
            iteration / debugging (e.g. just load 3 episodes).

    The returned TensorDict has ``batch_size=[N]`` where N is the total
    number of transitions across the selected episodes. ``rewards`` are
    all zeros — :func:`_relabel_rewards_sparse` (called downstream by
    :func:`load_offline_transitions` when ``reward_mode='sparse'``) places
    the +1 at the last frame of each episode.
    """
    # Heavy imports kept local so this module can be imported on CPU-only
    # boxes that don't have the LeRobot stack installed.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading a LeRobot HF dataset requires the lerobot package. "
            "Install it with `pip install lerobot[feetech]==0.4.3` (already "
            "pinned in environment.yaml). Alternatively, pass a local .pt "
            "offline bundle to --offline_path."
        ) from exc

    import torch.nn.functional as F
    from so101_fk import tcp_pos  # repo-local FK utility

    print(f"Loading LeRobot dataset {repo_id!r}...")
    ds = LeRobotDataset(repo_id)

    if state_dim != _EXPECTED_STATE_DIM:
        raise ValueError(
            f"Demo loader produces a {_EXPECTED_STATE_DIM}-D state vector but "
            f"the env reports state_dim={state_dim}. Either rebuild the env "
            f"with the baseline flags (pick_only_reward=True, n_distractors=0, "
            f"use_real_bowl=True, domain_randomization=False) or update "
            f"_STATE_SLICES in rlpd_utils.py to match the new layout."
        )
    if action_dim != 6:
        raise ValueError(f"SO-101 action is 6D, got action_dim={action_dim}.")

    target_h, target_w, target_c = obs_shape
    if target_c != 3:
        raise ValueError(f"Demo loader expects RGB (3 channels), got {target_c}.")

    # Episode index list. lerobot <0.4 exposed `episode_data_index` (a dict
    # of tensors). lerobot >=0.4.3 moved this to `meta.episodes` (an HF
    # Dataset with `dataset_from_index` / `dataset_to_index` columns). Same
    # semantic: from is inclusive, to is exclusive.
    if hasattr(ds, "episode_data_index") and ds.episode_data_index is not None:
        ep_from = ds.episode_data_index["from"].tolist()
        ep_to = ds.episode_data_index["to"].tolist()
    elif getattr(ds, "meta", None) is not None and getattr(ds.meta, "episodes", None) is not None:
        ep_from = list(ds.meta.episodes["dataset_from_index"])
        ep_to = list(ds.meta.episodes["dataset_to_index"])
    else:  # pragma: no cover
        raise RuntimeError(
            "LeRobotDataset exposes neither episode_data_index nor "
            "meta.episodes — unsupported lerobot version."
        )
    num_episodes = len(ep_from)

    if keep_episode_indices is not None:
        selected = [i for i in keep_episode_indices if 0 <= i < num_episodes]
    else:
        selected = list(range(num_episodes))
    if max_episodes is not None:
        selected = selected[:max_episodes]

    # Per-joint conversion: degrees -> rad for arm; gripper depends on
    # `gripper_in_degrees`. If False we assume the recorded value is already
    # in radians (i.e. the demo was logged on a calibration that emits rad).
    deg2rad = float(np.pi / 180.0)
    arm_scale = np.array([deg2rad] * 5, dtype=np.float32)
    gripper_scale = float(deg2rad if gripper_in_degrees else 1.0)

    sim_step_range = (_SIM_ACTION_UPPER - _SIM_ACTION_LOWER) / 2.0  # [6]
    sim_step_bias  = (_SIM_ACTION_UPPER + _SIM_ACTION_LOWER) / 2.0  # zero in our case
    assert np.allclose(sim_step_bias, 0.0), "step bias should be zero for symmetric ranges"

    # Bulk-allocated growable lists; concatenate at the end.
    obs_rgb_chunks, obs_state_chunks = [], []
    next_rgb_chunks, next_state_chunks = [], []
    action_chunks, done_chunks = [], []
    success_chunks = []
    total = 0

    fps = float(getattr(ds, "fps", 30) or 30)
    dt = 1.0 / fps

    # Cheap stat tracker for the calibration sanity print.
    gripper_qpos_vals: list = []
    pre_norm_action_vals: list = []

    for ep_idx in selected:
        i0, i1 = ep_from[ep_idx], ep_to[ep_idx]
        T = i1 - i0
        if T < 2:
            continue

        # Pull the whole episode in one go. Each frame is a dict.
        frames = [ds[k] for k in range(i0, i1)]

        # --- State (6D qpos in rad, including gripper) -----------------------
        qpos_deg = np.stack([f["observation.state"].cpu().numpy() for f in frames])
        act_deg = np.stack([f["action"].cpu().numpy() for f in frames])
        qpos_rad = np.zeros_like(qpos_deg)
        qpos_rad[:, :5] = qpos_deg[:, :5] * arm_scale
        qpos_rad[:, 5] = qpos_deg[:, 5] * gripper_scale
        act_rad = np.zeros_like(act_deg)
        act_rad[:, :5] = act_deg[:, :5] * arm_scale
        act_rad[:, 5] = act_deg[:, 5] * gripper_scale
        gripper_qpos_vals.extend(qpos_rad[:, 5].tolist())

        # --- qvel via finite difference --------------------------------------
        qvel = np.zeros_like(qpos_rad)
        qvel[1:] = (qpos_rad[1:] - qpos_rad[:-1]) / dt
        qvel[0] = qvel[1]  # repeat the first computed velocity for frame 0

        # --- TCP pose via FK (numpy) -----------------------------------------
        tcp_xyz = np.stack([tcp_pos(qpos_rad[t]) for t in range(T)]).astype(np.float32)
        tcp_pose = np.zeros((T, 7), dtype=np.float32)
        tcp_pose[:, :3] = tcp_xyz
        # Quaternion left at zero; the encoder is RGB-driven, the privileged
        # quat is one of the slices Squint also leaves uninformative at deploy.
        tcp_pose[:, 6] = 1.0  # canonical identity quaternion w-component

        # --- is_item_grasped from gripper closure ----------------------------
        is_grasped = (qpos_rad[:, 5] > grasp_threshold_rad).astype(np.float32)

        # --- Assemble the 58D state vector for each frame --------------------
        state = np.zeros((T, _EXPECTED_STATE_DIM), dtype=np.float32)
        state[:, slice(*_STATE_SLICES["noisy_qpos"])] = qpos_rad
        state[:, slice(*_STATE_SLICES["controller_target"])] = act_rad
        # Default goal color = first index (gives the policy something to
        # condition on; real Eval-2 will be retrained on a multi-goal dataset).
        state[:, _STATE_SLICES["goal_color"][0]] = 1.0
        state[:, slice(*_STATE_SLICES["qvel"])] = qvel
        state[:, slice(*_STATE_SLICES["is_item_grasped"])] = is_grasped[:, None]
        state[:, slice(*_STATE_SLICES["tcp_pose"])] = tcp_pose
        # bowl_xyz, item_pose, bin_pose, tcp_to_*, item_to_bin left at zero.

        # --- Action: delta in controller TARGET, not in current qpos -------
        # The sim controller is `pd_joint_target_delta_pos` with `use_target=True`
        # (envs/robot/so101.py): target[t+1] = target[t] + action[t] * step_range.
        # So the bundle's action[t] must be the diff between consecutive demo
        # targets, NOT (target - current_qpos). Earlier versions did the latter,
        # which clipped to 1.0 on slow joints (gripper) and made the sim target
        # diverge by an order of magnitude from the demo.
        # Exception: at frame 0 the sim controller initializes target = current
        # qpos, so action[0] still uses (act_rad[0] - qpos_rad[0]).
        T_ep = act_rad.shape[0]
        delta_rad = np.zeros_like(act_rad)
        delta_rad[0] = act_rad[0] - qpos_rad[0]
        if T_ep > 1:
            delta_rad[1:] = act_rad[1:] - act_rad[:-1]
        pre_norm_action_vals.extend(delta_rad.flatten().tolist())
        action_norm = np.clip(delta_rad / sim_step_range, -1.0, 1.0).astype(np.float32)

        # --- Image: decode + area-resize 480×640 → target_h×target_w --------
        rgb_uint8 = np.stack([f["observation.images.wrist"].cpu().numpy() for f in frames])
        # LeRobot returns CHW float in [0, 1] for video frames by default —
        # standardise to HWC uint8 to match the sim obs layout.
        if rgb_uint8.dtype != np.uint8:
            if rgb_uint8.ndim == 4 and rgb_uint8.shape[1] in (1, 3):  # NCHW float
                rgb_uint8 = np.transpose(rgb_uint8, (0, 2, 3, 1))
            rgb_uint8 = np.clip(rgb_uint8 * 255.0, 0, 255).astype(np.uint8)
        # Area-resize from native to target (NHWC -> NCHW for F.interpolate).
        rgb_t = torch.from_numpy(rgb_uint8).permute(0, 3, 1, 2).float()
        rgb_resized = F.interpolate(rgb_t, size=(target_h, target_w), mode="area").to(torch.uint8)
        rgb_hwc = rgb_resized.permute(0, 2, 3, 1).contiguous()  # [T, H, W, 3] uint8

        # --- Build (obs, next_obs) pairs ------------------------------------
        # For frame t we record:
        #   obs[t]      = (rgb[t], state[t])
        #   action[t]   = action_norm[t]
        #   next_obs[t] = (rgb[t+1], state[t+1]) for t < T-1
        #   done[T-1]   = True; everywhere else False.
        obs_rgb_chunks.append(rgb_hwc[:-1])
        next_rgb_chunks.append(rgb_hwc[1:])
        obs_state_chunks.append(torch.from_numpy(state[:-1]))
        next_state_chunks.append(torch.from_numpy(state[1:]))
        action_chunks.append(torch.from_numpy(action_norm[:-1]))
        d = torch.zeros(T - 1, dtype=torch.bool)
        d[-1] = True
        done_chunks.append(d)
        success_chunks.append(d.clone())  # we treat every demo as a success
        total += T - 1

    if total == 0:
        raise RuntimeError(f"No usable transitions extracted from {repo_id!r}.")

    td = TensorDict(
        observations=TensorDict(
            rgb=torch.cat(obs_rgb_chunks, dim=0),
            state=torch.cat(obs_state_chunks, dim=0).float(),
            batch_size=[total],
        ),
        next_observations=TensorDict(
            rgb=torch.cat(next_rgb_chunks, dim=0),
            state=torch.cat(next_state_chunks, dim=0).float(),
            batch_size=[total],
        ),
        actions=torch.cat(action_chunks, dim=0).float(),
        rewards=torch.zeros(total, dtype=torch.float32),
        dones=torch.cat(done_chunks, dim=0),
        success=torch.cat(success_chunks, dim=0),
        batch_size=[total],
    )

    # --- Sanity-check printout ---------------------------------------------
    g = np.array(gripper_qpos_vals)
    a = np.array(pre_norm_action_vals)
    print(
        f"LeRobot → ManiSkill loader stats:\n"
        f"  episodes used : {len(selected)} / {num_episodes}\n"
        f"  transitions   : {total}\n"
        f"  gripper qpos  : min={g.min():.3f} median={np.median(g):.3f} max={g.max():.3f} rad "
        f"(grasp threshold = {grasp_threshold_rad} rad → fraction grasped = "
        f"{td['observations', 'state'][:, 27].mean().item():.2%})\n"
        f"  pre-norm delta: min={a.min():.4f} max={a.max():.4f} rad "
        f"({(np.abs(a) > 0.05).mean():.2%} of arm-joint deltas would clip at the "
        f"arm cap; >5% suggests an FPS / unit mismatch — verify the printed "
        f"gripper qpos range looks like radians, not degrees, before training)."
    )
    return td


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-demo collector (Squint checkpoint → offline bundle)
# ─────────────────────────────────────────────────────────────────────────────

def collect_synthetic_offline_from_ckpt(
    ckpt_path: str,
    env_id: str,
    out_path: str,
    num_episodes: int = 50,
    keep_only_successful: bool = True,
    num_envs: int = 16,
    device: str = "cuda",
    image_height: int = 80,
    image_width: int = 144,
    render_height: int = 360,
    render_width: int = 640,
    pick_only_reward: bool = True,
    n_distractors: int = 0,
    use_real_bowl: bool = True,
) -> None:
    """Roll out a trained Squint policy in sim and save the resulting
    transitions as an offline bundle for RLPD.

    This is the fallback while real teleop demos are not yet available. The
    function is intentionally small and self-contained: it imports the
    env/CNN/Actor classes from ``train_rlpd`` so the offline obs layout is
    guaranteed to match the online layout (same downsample, same state vector).

    Args:
        ckpt_path: path to a ``.pt`` checkpoint saved by ``train_squint.py``
            (must contain ``encoder``, ``actor``, ``critic`` state dicts).
        env_id: same ``--env_id`` the RLPD run will use.
        out_path: where to write the offline bundle (``.pt``).
        num_episodes: total successful (or total, if keep_only_successful=False)
            episodes to collect. Episodes are collected in batches of
            ``num_envs`` parallel envs.
        keep_only_successful: filter via ``success_at_end``.

    The other kwargs mirror ``train_rlpd.Args`` defaults so the recorded obs
    are pixel-perfect identical to what the online buffer will see.
    """
    # Imports kept inside the function so this module is importable on CPU-only
    # machines (e.g. the laptop the user dev's RLPD on).
    import gymnasium as gym
    from mani_skill.utils.wrappers.flatten import (
        FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper,
    )
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    import envs  # noqa: F401 — registers SO101*-v1 ids
    import mani_skill.envs  # noqa: F401
    import utils as squint_utils
    from train_rlpd import CNNEncoder, Actor

    device_t = torch.device(device)
    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        sensor_configs=dict(height=render_height, width=render_width),
        human_render_camera_configs=dict(height=render_height, width=render_width),
        viewer_camera_configs=dict(height=render_height, width=render_width),
        domain_randomization=False,  # cleaner demo distribution for the offline buffer
        n_distractors=n_distractors,
        use_real_bowl=use_real_bowl,
        pick_only_reward=pick_only_reward,
    )
    base_env = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=None, **env_kwargs)
    env = FlattenRGBDObservationWrapper(base_env, rgb=True, depth=False, state=True)
    if (render_height, render_width) != (image_height, image_width):
        env = squint_utils.DownsampleObsWrapper(env, target_size=(image_height, image_width))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, record_metrics=True)

    n_act = int(np.prod(env.unwrapped.single_action_space.shape))
    n_channels = env.unwrapped.single_observation_space["rgb"].shape[2]
    n_obs = (image_height, image_width, n_channels)
    n_state = int(np.prod(env.unwrapped.single_observation_space["state"].shape))

    encoder = CNNEncoder(n_obs=n_obs, device=device_t)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device_t).eval()
    ckpt = torch.load(ckpt_path, map_location=device_t, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()

    max_episode_steps = env.unwrapped.gym_env.spec.max_episode_steps if hasattr(
        env.unwrapped, "gym_env") else 80

    transitions: Dict[str, list] = {
        "obs_rgb": [], "obs_state": [], "next_rgb": [], "next_state": [],
        "actions": [], "rewards": [], "dones": [], "success": [],
    }

    obs, _ = env.reset()
    episodes_collected = 0
    while episodes_collected < num_episodes:
        # One pass over max_episode_steps fills num_envs rollouts.
        # Buffer per-env trajectories until we hit done, then commit or discard.
        per_env_buf = [[] for _ in range(num_envs)]
        for _ in range(max_episode_steps):
            with torch.no_grad():
                rgb_feat = encoder(obs["rgb"])
                action = actor.get_eval_action(rgb_feat, obs["state"])
            next_obs, rewards, terms, truncs, infos = env.step(action)
            done = terms | truncs
            for i in range(num_envs):
                per_env_buf[i].append((
                    obs["rgb"][i].cpu().clone(),
                    obs["state"][i].cpu().clone(),
                    next_obs["rgb"][i].cpu().clone(),
                    next_obs["state"][i].cpu().clone(),
                    action[i].cpu().clone().float(),
                    float(rewards[i].item()),
                    bool(done[i].item()),
                    bool(terms[i].item()),  # success proxy: hit a true termination, not a truncation
                ))
            obs = next_obs

        for buf in per_env_buf:
            if not buf:
                continue
            ep_success = buf[-1][7]  # last termination flag
            if keep_only_successful and not ep_success:
                continue
            for (or_, os_, nr, ns, a, r, d, _) in buf:
                transitions["obs_rgb"].append(or_)
                transitions["obs_state"].append(os_)
                transitions["next_rgb"].append(nr)
                transitions["next_state"].append(ns)
                transitions["actions"].append(a)
                transitions["rewards"].append(r)
                transitions["dones"].append(d)
                transitions["success"].append(ep_success)
            episodes_collected += 1
            if episodes_collected >= num_episodes:
                break
        obs, _ = env.reset()
        print(f"Collected {episodes_collected}/{num_episodes} episodes...")

    n = len(transitions["actions"])
    td = TensorDict(
        observations=TensorDict(
            rgb=torch.stack(transitions["obs_rgb"]),
            state=torch.stack(transitions["obs_state"]).float(),
            batch_size=[n],
        ),
        next_observations=TensorDict(
            rgb=torch.stack(transitions["next_rgb"]),
            state=torch.stack(transitions["next_state"]).float(),
            batch_size=[n],
        ),
        actions=torch.stack(transitions["actions"]).float(),
        rewards=torch.tensor(transitions["rewards"], dtype=torch.float32),
        dones=torch.tensor(transitions["dones"], dtype=torch.bool),
        success=torch.tensor(transitions["success"], dtype=torch.bool),
        batch_size=[n],
    )
    save_offline_bundle(out_path, td)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build an offline RLPD bundle.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- Sub-command 1: synthetic rollouts from a trained Squint ckpt ------
    p_synth = sub.add_parser(
        "from_ckpt",
        help="Roll out a Squint checkpoint in sim and bundle the successful episodes.",
    )
    p_synth.add_argument("--ckpt", required=True, help="Path to a Squint .pt checkpoint.")
    p_synth.add_argument("--env_id", required=True, help="Same env_id you'll train RLPD on.")
    p_synth.add_argument("--out", required=True, help="Where to write the offline .pt bundle.")
    p_synth.add_argument("--num_episodes", type=int, default=50)
    p_synth.add_argument("--num_envs", type=int, default=16)
    p_synth.add_argument("--pick_only_reward", action="store_true", default=True)
    p_synth.add_argument("--no-pick_only_reward", dest="pick_only_reward", action="store_false")
    p_synth.add_argument("--n_distractors", type=int, default=0)
    p_synth.add_argument("--no-success_filter", dest="keep_only_successful",
                         action="store_false", default=True)

    # --- Sub-command 2: convert a HF LeRobot dataset -> offline .pt --------
    p_hf = sub.add_parser(
        "from_lerobot",
        help="Decode a HuggingFace LeRobot dataset and bundle it as an offline .pt.",
    )
    p_hf.add_argument("--repo_id", required=True,
                      help="HF dataset id, e.g. 'Rsebti/projet3_demos_v1'.")
    p_hf.add_argument("--out", required=True, help="Where to write the offline .pt bundle.")
    p_hf.add_argument("--image_height", type=int, default=80)
    p_hf.add_argument("--image_width", type=int, default=144)
    p_hf.add_argument("--gripper_in_degrees", action="store_true", default=True,
                      help="Set if the gripper position is logged in degrees (default).")
    p_hf.add_argument("--no-gripper_in_degrees", dest="gripper_in_degrees",
                      action="store_false")
    p_hf.add_argument("--grasp_threshold_rad", type=float, default=0.5,
                      help="Gripper qpos threshold above which is_item_grasped=1.")
    p_hf.add_argument("--max_episodes", type=int, default=None,
                      help="If set, only convert the first N episodes (fast iteration).")
    p_hf.add_argument("--reward_mode", default="sparse", choices=("sparse", "recompute"))

    args = p.parse_args()

    if args.cmd == "from_ckpt":
        collect_synthetic_offline_from_ckpt(
            ckpt_path=args.ckpt,
            env_id=args.env_id,
            out_path=args.out,
            num_episodes=args.num_episodes,
            num_envs=args.num_envs,
            pick_only_reward=args.pick_only_reward,
            n_distractors=args.n_distractors,
            keep_only_successful=args.keep_only_successful,
        )
    elif args.cmd == "from_lerobot":
        # We bypass load_offline_transitions's device move so the bundle is
        # written as CPU tensors (portable between machines).
        td = _load_lerobot_dataset(
            args.repo_id,
            obs_shape=(args.image_height, args.image_width, 3),
            state_dim=_EXPECTED_STATE_DIM,
            action_dim=6,
            gripper_in_degrees=args.gripper_in_degrees,
            grasp_threshold_rad=args.grasp_threshold_rad,
            max_episodes=args.max_episodes,
        )
        _validate_transitions(td, obs_shape=(args.image_height, args.image_width, 3),
                              state_dim=_EXPECTED_STATE_DIM, action_dim=6)
        if args.reward_mode == "sparse":
            td = _relabel_rewards_sparse(td)
        save_offline_bundle(args.out, td)
