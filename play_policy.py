"""Live interactive viewer for a trained Squint policy in sim.

Opens an OpenCV window, runs N episodes one at a time. Between episodes,
press SPACE/ENTER to start the next, Q to quit.

Usage:
    python play_policy.py \
        --checkpoint=runs/placecube_slippery_slow_3M_run1/ckpt.pt \
        --n_distractors=3 \
        --n_episodes=10
"""
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envs  # noqa: F401  -- registers SO101*-v1
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import CNNEncoder, Actor


def _to_np(x):
    """Convert tensor (incl. CUDA) / list-of-tensors / numpy to numpy."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_np_img(x):
    x = _to_np(x)
    if x.ndim == 4:
        x = x[0]
    return x.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101PlaceCube-v1")
    ap.add_argument("--n_distractors", type=int, default=3)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--render_size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_dr", action="store_true",
                    help="Turn off domain randomization for cleaner replays.")
    ap.add_argument("--use_real_bowl", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Use envs/meshes/bowl.obj (default). Pass --no-use_real_bowl to use the parametric bin.")
    args = ap.parse_args()

    env_kwargs = dict(
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=not args.no_dr,
        control_mode="pd_joint_target_delta_pos",
        sensor_configs=dict(width=128, height=128),
        human_render_camera_configs=dict(
            shader_pack="default", width=args.render_size, height=args.render_size
        ),
        n_distractors=args.n_distractors,
        use_real_bowl=args.use_real_bowl,
    )
    env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    obs_space = env.unwrapped.single_observation_space
    n_state = obs_space["state"].shape[0]
    n_act = env.unwrapped.single_action_space.shape[0]
    encoder = CNNEncoder(n_obs=(16, 16, 3), device=device).to(device)
    actor = Actor(env, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval(); actor.eval()
    print(f"loaded {args.checkpoint} @ step {ckpt.get('global_step')}")
    print(f"n_state={n_state}, n_act={n_act}, n_distractors={args.n_distractors}")
    print("Controls: SPACE/ENTER = next episode, Q = quit.")

    delay_ms = max(1, int(1000 / args.fps))
    window = "Squint policy (sim)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.render_size, args.render_size)

    base = env.unwrapped
    max_steps = 75

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        succeeded = False
        goal_color = int(base.goal_color_idx[0].item()) if hasattr(base, "goal_color_idx") else -1
        color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
        gname = color_names[goal_color] if 0 <= goal_color < 6 else "?"

        for step in range(max_steps):
            frame = _to_np_img(env.render())
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"ep {ep+1}/{args.n_episodes}  step {step+1}/{max_steps}  goal={gname}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window, frame_bgr)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows(); env.close(); return

            rgb_now = obs["rgb"]
            state_now = obs["state"]
            if not torch.is_tensor(rgb_now):
                rgb_now = torch.from_numpy(rgb_now)
            if not torch.is_tensor(state_now):
                state_now = torch.from_numpy(state_now)
            rgb_t = rgb_now.permute(0, 3, 1, 2).float()
            rgb16 = F.interpolate(rgb_t, size=(16, 16), mode='area').permute(0, 2, 3, 1).to(torch.uint8)
            with torch.no_grad():
                feats = encoder(rgb16.to(device))
                mean = actor.forward(feats, state_now.float().to(device))
                action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            obs, rew, term, trunc, info = env.step(action_np)
            ep_return += float(_to_np(rew).flatten()[0])
            if "success" in info:
                s = float(_to_np(info["success"]).flatten()[0])
                if s > 0.5:
                    succeeded = True
            done = bool(_to_np(term).any()) or bool(_to_np(trunc).any())
            if done:
                break

        # End-of-episode banner
        last_frame = cv2.cvtColor(_to_np_img(env.render()), cv2.COLOR_RGB2BGR)
        color = (0, 200, 0) if succeeded else (0, 0, 255)
        msg = f"ep {ep+1}: {'SUCCESS' if succeeded else 'fail'}  return={ep_return:.1f}  SPACE=next  Q=quit"
        cv2.putText(last_frame, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow(window, last_frame)
        print(f"  ep {ep+1}: {'SUCCESS' if succeeded else 'fail'}  return={ep_return:.2f}  goal={gname}")
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows(); env.close(); return
            if k in (ord(' '), 13):
                break

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
