#!/usr/bin/env python3
"""Verify a v2-schema demo file is loader-compatible with train_rlpd.py.

Ported verbatim from fedecomi04/squint:tom-separating-cubes commit c6f9808
(Tommaso's validator). Standalone check the demo-collection author can run
before handing over the h5.

Usage:
    python scripts/check_rlpd_demos_v2.py <path-to-demos.h5>

Exit code 0 + "ALL CHECKS PASSED" → safe to point train_rlpd.py at this file
via --offline_path.

Note: the validator expects control_mode == 'pd_joint_target_delta_pos' in
the attrs but our collector deliberately writes 'pd_joint_pos' (absolute
targets). rlpd_utils._load_h5_v2 handles the conversion at load time, so a
single attr-check failure on control_mode is expected and harmless. All
other checks should pass.
"""
import os
import sys

import h5py
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401 — registers SO101PlaceCube-v1
import mani_skill.envs  # noqa: F401
import utils

from mani_skill.utils.wrappers.flatten import (
    FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def build_training_env(num_envs: int = 1):
    """Mirror train_rlpd.py's env stack so shape probes return canonical values."""
    e = gym.make(
        "SO101PlaceCube-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        render_mode="all",
        sim_backend="gpu",
        sensor_configs=dict(width=640, height=360),
        domain_randomization=True,
        domain_randomization_config={
            "shadows": True,
            "num_directional_lights": 3,
            "camera_lag_substeps_range": (0, 0),
        },
        n_distractors=0,
        use_real_bowl=True,
        sim_freq=100,
        control_freq=10,
    )
    e = FlattenRGBDObservationWrapper(e, rgb=True, depth=False, state=True)
    e = utils.DownsampleObsWrapper(e, target_size=(80, 144))
    e = utils.ColorJitterWrapper(e)
    if isinstance(e.action_space, gym.spaces.Dict):
        e = FlattenActionSpaceWrapper(e)
    e = ManiSkillVectorEnv(e, num_envs=num_envs,
                           ignore_terminations=False, record_metrics=True)
    return e


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/check_rlpd_demos_v2.py <h5>")
        sys.exit(2)
    h5_path = sys.argv[1]
    if not os.path.exists(h5_path):
        print(f"FATAL: file not found: {h5_path}")
        sys.exit(2)

    print(f"Verifying: {h5_path}")
    print("Building training-stack env...")
    env = build_training_env(num_envs=1)
    obs, _ = env.reset(seed=0)

    env_state_dim = int(np.prod(env.single_observation_space["state"].shape))
    env_rgb_shape = tuple(env.single_observation_space["rgb"].shape)
    env_act_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    env_act_high = np.asarray(env.single_action_space.high, dtype=np.float32)

    print(f"  env state_dim : {env_state_dim}")
    print(f"  env rgb shape : {env_rgb_shape}")
    print(f"  env act low   : {env_act_low}")
    print(f"  env act high  : {env_act_high}")

    failed = []

    with h5py.File(h5_path, "r") as f:
        print("\n=== file attrs ===")
        wanted = ["format_version", "env_id", "control_mode", "n_distractors",
                  "rgb_h", "rgb_w", "state_dim", "action_dim",
                  "arm_delta_max", "grip_delta_max", "num_demos"]
        for k in wanted:
            if k in f.attrs:
                print(f"  {k:18s}: {f.attrs[k]}")
            else:
                failed.append(f"missing attr: {k}")
                print(f"  {k:18s}: MISSING")

        def check(cond, msg):
            if not cond:
                failed.append(msg)
                print(f"  FAIL: {msg}")
            else:
                print(f"  OK  : {msg}")

        print("\n=== schema checks ===")
        check(str(f.attrs.get("format_version", "")) == "2.0",
              "format_version == '2.0'")
        cm = str(f.attrs.get("control_mode", ""))
        check(cm in ("pd_joint_target_delta_pos", "pd_joint_pos"),
              f"control_mode in {{pd_joint_target_delta_pos, pd_joint_pos}} (got {cm!r})")
        check(int(f.attrs.get("state_dim", -1)) == env_state_dim,
              f"state_dim == env ({env_state_dim})")
        check(int(f.attrs.get("rgb_h", -1)) == env_rgb_shape[0],
              f"rgb_h == env ({env_rgb_shape[0]})")
        check(int(f.attrs.get("rgb_w", -1)) == env_rgb_shape[1],
              f"rgb_w == env ({env_rgb_shape[1]})")

        arm = float(f.attrs.get("arm_delta_max", -1))
        grip = float(f.attrs.get("grip_delta_max", -1))
        expected_low = np.array([-arm] * 5 + [-grip], dtype=np.float32)
        expected_high = np.array([arm] * 5 + [grip], dtype=np.float32)
        check(np.allclose(env_act_low, expected_low, atol=1e-5),
              f"action low matches env (arm=±{arm}, grip=±{grip})")
        check(np.allclose(env_act_high, expected_high, atol=1e-5),
              f"action high matches env (arm=±{arm}, grip=±{grip})")

        print("\n=== per-demo spot checks ===")
        demo_keys = sorted(k for k in f.keys() if k.startswith("demo_"))
        check(len(demo_keys) >= 1, "at least one demo group present")

        n_success = 0
        all_rmin, all_rmax = 1e9, -1e9
        for name in demo_keys:
            g = f[name]
            success = bool(g.attrs.get("success", False))
            if success:
                n_success += 1
            r = g["rewards"][:]
            all_rmin = min(all_rmin, float(r.min()))
            all_rmax = max(all_rmax, float(r.max()))

        v_min = float(f.attrs.get("reward_v_min", -20.0))
        v_max = float(f.attrs.get("reward_v_max", 20.0))
        check(all_rmin >= v_min and all_rmax <= v_max,
              f"all rewards in [{v_min}, {v_max}] (saw [{all_rmin:.2f}, {all_rmax:.2f}])")
        check(n_success == len(demo_keys),
              f"all demos marked success ({n_success}/{len(demo_keys)})")

        d0 = f[demo_keys[0]]
        check(d0["obs/rgb"].dtype == np.uint8, "obs/rgb dtype == uint8")
        check(d0["obs/state"].dtype == np.float32, "obs/state dtype == float32")
        check(d0["actions"].dtype == np.float32, "actions dtype == float32")
        check(d0["obs/rgb"].shape[1:] == env_rgb_shape,
              f"obs/rgb per-step shape == {env_rgb_shape}")
        check(d0["obs/state"].shape[-1] == env_state_dim,
              f"obs/state per-step dim == {env_state_dim}")
        check(d0["actions"].shape[-1] == 6, "actions per-step dim == 6")

        T = d0["actions"].shape[0]
        check(d0["rewards"].shape[0] == T,
              f"rewards length == actions length ({T})")
        check(d0["obs/rgb"].shape[0] == T,
              f"obs/rgb length == actions length ({T})")

    env.close()

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED with {len(failed)} issue(s):")
        for m in failed:
            print(f"  - {m}")
        print("\nFix the demo collection before proceeding to training.")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — demo file is loader-compatible.")
        sys.exit(0)


if __name__ == "__main__":
    main()
