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
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import MotorNormMode

# �═══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE for your robot                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
ROBOT_PORT = "/dev/ttyACM0"               # Mac: /dev/cu.usbmodemXXXX  (run: ls /dev/cu.*)
CAMERA_INDEX = 0                          # webcam index: try 0, 1, or 2
CALIBRATION_ID = "so101_follower_arm"     # filename (no extension) of your calibration .json
CALIBRATION_DIR = Path(__file__).parent   # folder that holds the calibration .json

# ── Contract constants (must match the training env) ───────────────────────
IMAGE_SIZE = 16          # CNN input H=W
SIM_CAM_SIZE = 128       # sim wrist-camera resolution (intermediate resize)
N_COLORS = 6             # goal-color one-hot length
CONTROL_HZ = 10          # sim_freq=100 / control_freq=10

# Per-joint delta caps (rad/step): arm joints ±0.1, gripper ±0.2.
DELTA_CAP = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
# Joint limits from so101.urdf, order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])
# SO101 "start" keyframe — robot rest pose.
REST_QPOS = np.array([0, 0, 0, np.pi / 2, -np.pi / 2, np.deg2rad(60)], dtype=np.float32)


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Robot driver — wraps a LeRobot SO101 follower                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def create_real_robot():
    config = SO101FollowerConfig(
        port=ROBOT_PORT,
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX, fps=30, width=640, height=480,
        )},
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
        self._g_servo_min, self._g_servo_max = -62.5, 64.62
        self._g_sim_range = self._g_sim_max - self._g_sim_min
        self._g_servo_range = self._g_servo_max - self._g_servo_min
        robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

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
        for name, cam in self.real_robot.cameras.items():
            frame = np.asarray(cam.async_read())                        # (H, W, 3) uint8 RGB
            self._sensor_data[name] = {"rgb": torch.from_numpy(frame).unsqueeze(0)}

    def get_sensor_data(self):
        return self._sensor_data


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Policy network — architecture must match the checkpoint exactly          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, rgb_uint8):           # (B, 16, 16, 3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = x / 255.0 - 0.5
        return self.conv(x)                 # (B, 1024)


class Projection(nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.rgb_proj = nn.Sequential(nn.Linear(1024, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, n_state=18, n_act=6):
        super().__init__()
        self.proj = Projection(n_state)
        self.fc = nn.Sequential(
            nn.Linear(306, 256), nn.LayerNorm(256), nn.ReLU(),
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
    """Real camera frame (1,H,W,3) uint8 → (1,16,16,3) uint8 tensor.

    Center-crop to square, resize to the 128px sim resolution, area-downsample
    to the 16px CNN input — same two-step path used during training.
    """
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    h, w = img.shape[:2]
    c = min(h, w)
    img = img[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    img = cv2.resize(img, (SIM_CAM_SIZE, SIM_CAM_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).unsqueeze(0).to(torch.uint8)


def build_state(qpos, target_qpos, goal_color):
    """18-d state vector = [measured_qpos(6), controller_target_qpos(6), goal_onehot(6)]."""
    onehot = np.zeros(N_COLORS, dtype=np.float32)
    onehot[goal_color] = 1.0
    vec = np.concatenate([qpos, target_qpos, onehot]).astype(np.float32)
    return torch.from_numpy(vec).unsqueeze(0)


# �═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to ckpt.pt")
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.15, help="safety multiplier on policy action (lower = slower)")
    p.add_argument("--episode_steps", type=int, default=150, help="control steps per episode")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy (only encoder + actor are needed; critic/log_alpha are training-only).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = CNNEncoder().to(device).eval()
    actor = Actor().to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"Loaded checkpoint (trained to step {ckpt.get('global_step', '?')})")

    # Connect robot, then build the driver (it touches robot.bus on init).
    robot = create_real_robot()
    robot.connect()
    agent = RealRobotAgent(robot)

    try:
        while True:
            input(f"\n[Enter] start episode (goal color {args.goal_color}), Ctrl+C to quit ")
            agent.reset(REST_QPOS)                       # smooth move to rest pose
            # use_target controller: target starts at the measured rest qpos
            target_qpos = agent.get_qpos().cpu().numpy().flatten()

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]

                obs_rgb = preprocess_image(rgb).to(device)
                obs_state = build_state(qpos, target_qpos, args.goal_color).to(device)

                with torch.no_grad():
                    action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()

                # action ∈ [-1,1] → safety-scaled → per-joint delta → accumulate onto target.
                action = np.clip(action * args.action_scale, -1.0, 1.0)
                target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                agent.set_target_qpos(torch.from_numpy(target_qpos))

                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

            print(f"Episode done ({args.episode_steps} steps).")
    except KeyboardInterrupt:
        print("\nQuitting.")
    finally:
        agent.reset(REST_QPOS)
        robot.disconnect()


if __name__ == "__main__":
    main()
