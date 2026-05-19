"""
Standalone policy inference for the SO101 PlaceCube checkpoint.

Everything is in this one file: the policy network, the obs/action contract,
and the robot driver. The only repo file you need is this script + the
checkpoint. Dependencies: torch, numpy, opencv-python, lerobot[feetech].

Usage:
    python infer.py --checkpoint ckpt.pt --goal_color 0

Goal colors: 0 red  1 blue  2 green  3 yellow  4 purple  5 orange
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode

import threading


class Cv2Camera:
    """Drop-in replacement for lerobot's OpenCVCamera on macOS.

    lerobot's wrapper uses cv2.CAP_ANY (backend code 0) which on macOS
    flakily times out in async_read despite the read thread being alive.
    This class uses CAP_AVFOUNDATION explicitly and runs a tiny background
    reader so async_read() just hands back the latest frame."""

    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: int = 30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cv2Camera({index}) failed to open via AVFoundation.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # macOS AVFoundation often needs a warm-up of a few hundred ms + several
        # discarded frames before the sensor delivers a valid one. Retry up to
        # ~3 s of polling at 30 Hz.
        frame = None
        for _ in range(90):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.033)
        if frame is None:
            self.cap.release()
            raise RuntimeError(f"Cv2Camera({index}) opened but no frame after 3 s of polling.")
        self._latest = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest = rgb

    def async_read(self):
        with self._lock:
            return self._latest.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.cap.release()

try:
    import rerun as rr
except ImportError:
    rr = None

JOINT_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

# �═══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE for your robot                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
ROBOT_PORT = "/dev/cu.usbmodem5B141129871"               # Mac: /dev/cu.usbmodemXXXX  (run: ls /dev/cu.*)
CAMERA_INDEX = 0
CALIBRATION_ID = "so101_follower_arm"     # filename (no extension) of your calibration .json
CALIBRATION_DIR = Path(__file__).parent   # folder that holds the calibration .json

# ── Contract constants (must match the training env) ───────────────────────
IMAGE_H = 36             # CNN input height (landscape, matches sim wrist cam 16:9)
IMAGE_W = 64             # CNN input width  (64/36 = 16/9 EXACTLY; matches the
                         # 2026-05-19 real-camera calibration at ÷30 of 1920×1080)
N_COLORS = 6             # goal-color one-hot length
CONTROL_HZ = 30          # sim sim_freq=300 / control_freq=30 (matches training)

# Per-joint delta caps (rad/step): arm ±0.0333 (= 1.0 rad/s = 57 deg/s),
# gripper ±0.10 (= 3.0 rad/s = 172 deg/s, 3x arm). Matches the sim's
# pd_joint_delta_pos config so the policy's action distribution maps to
# the same per-joint velocity envelope at deploy.
DELTA_CAP = np.array([0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.10], dtype=np.float32)
# Joint limits from so101.urdf, order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])
# SO101 "start" keyframe — robot rest pose. Must match envs/robot/so101.py keyframes["start"].
# Degrees: [-2.242, -80.791, 36.747, 86.901, -82.154, -14.686].
REST_QPOS = np.deg2rad(
    np.array([-2.242, -80.791, 36.747, 86.901, -82.154, -14.686], dtype=np.float32)
)


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Robot driver — wraps a LeRobot SO101 follower                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def create_real_robot():
    # Camera handled by our own Cv2Camera (see RealRobotAgent.__init__);
    # lerobot's wrapper is bypassed because its CAP_ANY backend is flaky on macOS.
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        use_degrees=True,
        cameras={},
        id=CALIBRATION_ID,
        calibration_dir=CALIBRATION_DIR,
    )
    return make_robot_from_config(config)


class RealRobotAgent:
    """Minimal driver. Handles the unit conversions the policy contract needs:
    joint positions sim-radians <-> servo-degrees, and the gripper's separate
    sim range (-10°..120°) <-> servo range (-62.5°..64.62°)."""

    def __init__(self, robot):
        self.real_robot = robot
        self._cached_qpos = None
        self._motor_keys = None
        # gripper mapping (measured): sim -10°..120°  <->  servo -62.5°..64.62°
        self._g_sim_min, self._g_sim_max = -10.0, 120.0
        self._g_servo_min, self._g_servo_max = -60.13, 66.73
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES
        # Bypass lerobot's flaky macOS camera wrapper.
        self.cameras = {"base_camera": Cv2Camera(index=CAMERA_INDEX, width=1280, height=720, fps=30)}

    def get_qpos(self):
        """Measured joint angles in sim radians, shape (1, 6)."""
        if self._cached_qpos is not None:
            return self._cached_qpos.clone()
        deg = self.real_robot.bus.sync_read("Present_Position")
        servo = deg["gripper"]                                          # gripper: servo deg -> sim deg
        deg["gripper"] = (servo - self._g_servo_min) / self._g_servo_range * self._g_sim_range + self._g_sim_min
        if self._motor_keys is None:
            self._motor_keys = list(deg.keys())
        flat = np.array([deg[k] for k in self._motor_keys], dtype=np.float32)
        self._cached_qpos = torch.deg2rad(torch.from_numpy(flat)).unsqueeze(0)
        return self._cached_qpos.clone()

    def set_target_qpos(self, qpos):
        """Send a joint-angle target (sim radians) to the servos."""
        self._cached_qpos = None
        deg = torch.rad2deg(torch.as_tensor(qpos, dtype=torch.float32).flatten())
        cmd = {f"{self._motor_keys[i]}.pos": float(deg[i]) for i in range(len(deg))}
        sim_deg = cmd["gripper.pos"]                                    # gripper: sim deg -> servo deg
        cmd["gripper.pos"] = (sim_deg - self._g_sim_min) / self._g_sim_range * self._g_servo_range + self._g_servo_min
        self.real_robot.send_action(cmd)

    def reset(self, qpos, freq=30, max_rad_per_step=0.025):
        """Move smoothly to qpos by ramping the target a little each tick."""
        qpos = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        target = self.get_qpos().flatten()
        for _ in range(int(20 * freq)):
            delta = (qpos - target).clamp(-max_rad_per_step, max_rad_per_step)
            if torch.linalg.norm(delta) <= 1e-4:
                break
            target = target + delta
            self.set_target_qpos(target)
            time.sleep(1.0 / freq)

    def capture_sensor_data(self):
        self._sensor_data = {}
        for name, cam in self.cameras.items():
            frame = np.asarray(cam.async_read())                        # (H, W, 3) uint8 RGB
            self._sensor_data[name] = {"rgb": torch.from_numpy(frame).unsqueeze(0)}

    def get_sensor_data(self):
        return self._sensor_data


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Policy network — architecture must match the checkpoint exactly          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# For 32×42 input the CNN flatten is 64 * 4 * 7 = 1792 (see padding math in
# train_squint.py CNNEncoder, height==32 branch).
CNN_FLATTEN_DIM = 1792
RGB_PROJ_DIM = 75


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, rgb_uint8):           # (B, IMAGE_H, IMAGE_W, 3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)                 # (B, CNN_FLATTEN_DIM)


class Projection(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(CNN_FLATTEN_DIM, RGB_PROJ_DIM), nn.LayerNorm(RGB_PROJ_DIM), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state=18, n_act=6):
        super().__init__()
        self.proj = Projection(n_state)
        self.fc = nn.Sequential(
            nn.Linear(RGB_PROJ_DIM + 256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, n_act)
        self.fc_logstd = nn.Linear(256, n_act)
        self.register_buffer("action_scale", torch.ones(n_act))
        self.register_buffer("action_bias", torch.zeros(n_act))

    def forward(self, rgb_feat, state):     # deterministic eval action ∈ [-1, 1]
        x = self.fc(self.proj(rgb_feat, state))
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias


# ── Observation / action helpers ────────────────────────────────────────────
def preprocess_image(rgb):
    """Real camera frame (1,H,W,3) uint8 → (1, IMAGE_H, IMAGE_W, 3) uint8 tensor.

    The real wrist camera was calibrated 2026-05-19 at 1920×1080 (16:9). The
    sim renders 640×360 (¼ res, same 16:9). Both are area-downsampled to
    (IMAGE_H, IMAGE_W) = (36, 64), an exact 16:9 grid. No center-crop —
    the 16:9 aspect is preserved end-to-end with uniform pooling.
    """
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def init_viz():
    """Spawn a Rerun viewer window for live camera + joint plots."""
    if rr is None:
        raise RuntimeError("rerun not installed in this env (pip install rerun-sdk)")
    rr.init("squint_infer", spawn=True)


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw):
    """Push one timestep to the Rerun viewer."""
    if rr is None:
        return
    rr.set_time("step", sequence=step)
    rr.log("camera/raw", rr.Image(raw_rgb))
    rr.log("camera/policy_input_16x16", rr.Image(policy_rgb))
    for i, name in enumerate(JOINT_NAMES):
        rr.log(f"joints/qpos_measured/{name}", rr.Scalars([float(qpos[i])]))
        rr.log(f"joints/qpos_target/{name}", rr.Scalars([float(target_qpos[i])]))
        rr.log(f"action_raw/{name}", rr.Scalars([float(action_raw[i])]))


def build_state(qpos, target_qpos, goal_color, bowl_xyz=None):
    """State vector for the policy.

    Default (18-d): [measured_qpos(6), controller_target_qpos(6), goal_onehot(6)].
    If `bowl_xyz` is given (3-d), it is appended → 21-d. Used for checkpoints
    trained with the bowl position as an extra observation.
    """
    onehot = np.zeros(N_COLORS, dtype=np.float32)
    onehot[goal_color] = 1.0
    parts = [qpos, target_qpos, onehot]
    if bowl_xyz is not None:
        parts.append(np.asarray(bowl_xyz, dtype=np.float32))
    vec = np.concatenate(parts).astype(np.float32)
    return torch.from_numpy(vec).unsqueeze(0)


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to ckpt.pt")
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.15, help="safety multiplier on policy action (lower = slower)")
    p.add_argument("--episode_steps", type=int, default=300, help="control steps per episode (300 @ 30 Hz = 10s, matches PlaceCube sim spec)")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="open a Rerun viewer with live camera + joint plots (--no-viz to disable)")
    p.add_argument("--n_episodes", type=int, default=0, help="if >0, run this many episodes back-to-back without waiting for Enter")
    p.add_argument("--log_dir", type=str, default=None, help="if set, dump per-step npz logs there (one file per episode)")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"),
                   help="bowl position fed to the policy when the checkpoint expects a 21-d state (default: 0.25 0.10 0.00)")
    args = p.parse_args()

    if args.viz:
        init_viz()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy (only encoder + actor are needed; critic/log_alpha are training-only).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Detect the state-vector width the checkpoint was trained with.
    n_state_ckpt = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]
    use_bowl_xyz = n_state_ckpt == 21
    if n_state_ckpt not in (18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state_ckpt} (expected 18 or 21)")
    encoder = CNNEncoder().to(device).eval()
    actor = Actor(n_state=n_state_ckpt).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded checkpoint (trained to step {ckpt.get('global_step', '?')}), n_state={n_state_ckpt}"
          + (f" → feeding bowl_xyz={args.bowl_xyz}" if use_bowl_xyz else ""))

    # Connect robot, then build the driver (it touches robot.bus on init).
    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    def episodes():
        if args.n_episodes > 0:
            for i in range(args.n_episodes):
                yield i
        else:
            i = 0
            while True:
                input(f"\n[Enter] start episode (goal color {args.goal_color}), Ctrl+C to quit ")
                yield i
                i += 1

    try:
        for ep in episodes():
            if args.n_episodes > 0:
                print(f"\n── Episode {ep + 1}/{args.n_episodes} (goal color {args.goal_color}) ──")
            agent.reset(REST_QPOS)                       # smooth move to rest pose
            target_qpos = agent.get_qpos().cpu().numpy().flatten()

            log_qpos, log_target, log_action_raw, log_policy_rgb = [], [], [], []

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]

                obs_rgb = preprocess_image(rgb).to(device)
                obs_state = build_state(
                    qpos, target_qpos, args.goal_color,
                    bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
                ).to(device)

                with torch.no_grad():
                    raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()

                action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                agent.set_target_qpos(torch.from_numpy(target_qpos))

                if args.viz:
                    log_step(
                        step=step,
                        raw_rgb=rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0]),
                        policy_rgb=obs_rgb[0].cpu().numpy(),
                        qpos=qpos,
                        target_qpos=target_qpos,
                        action_raw=raw_action,
                    )
                if log_dir:
                    log_qpos.append(qpos.copy())
                    log_target.append(target_qpos.copy())
                    log_action_raw.append(raw_action.copy())
                    log_policy_rgb.append(obs_rgb[0].cpu().numpy().copy())

                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            if log_dir:
                np.savez(
                    log_dir / f"ep{ep:03d}.npz",
                    qpos=np.stack(log_qpos),
                    target_qpos=np.stack(log_target),
                    action_raw=np.stack(log_action_raw),
                    policy_rgb=np.stack(log_policy_rgb),
                    joint_names=np.array(JOINT_NAMES),
                )
                print(f"  → saved {log_dir / f'ep{ep:03d}.npz'}")
            print(f"Episode done ({args.episode_steps} steps).")
    except KeyboardInterrupt:
        print("\nQuitting.")
    finally:
        agent.reset(REST_QPOS)
        robot.disconnect()


if __name__ == "__main__":
    main()
