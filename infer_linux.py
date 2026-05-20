"""
Standalone policy inference for the SO101 PlaceCube checkpoint — Linux (Ubuntu).

Linux port of infer.py. Differences from the macOS original:
    - Cv2Camera uses cv2.CAP_V4L2 (the Linux-native Video4Linux2 backend)
      instead of cv2.CAP_AVFOUNDATION.
    - ROBOT_PORT default points to /dev/ttyACM0 (typical Feetech USB-serial
      enumeration on Linux). Use `ls /dev/tty{ACM,USB}*` to find yours.
    - V4L2 generally hands you a valid frame on the first read, so the
      AVFoundation-style multi-second warm-up loop is shortened. The retry
      stays in place to absorb slow USB negotiation on first plug-in.

Everything else (policy network, obs/action contract, calibration mapping)
is byte-for-byte identical to infer.py so checkpoints load unchanged.

Usage:
    python infer_linux.py --checkpoint ckpt.pt --goal_color 0

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
    """Drop-in V4L2 camera reader for Linux.

    Uses cv2.CAP_V4L2 explicitly (rather than CAP_ANY, which on some
    distros falls back to GStreamer/FFmpeg with subtle latency quirks)
    and runs a tiny background reader so async_read() just hands back
    the most recent frame."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080, fps: int = 30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cv2Camera({index}) failed to open via V4L2.")
        # Request MJPG so the kernel hands us pre-compressed frames; for
        # 1920x1080 @ 30 fps most UVC webcams cannot deliver raw YUYV at
        # that rate over USB-2 bandwidth.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # V4L2 normally delivers a valid frame on the first read once the
        # device is open, but slow USB negotiation can drop the first few
        # frames. Poll up to ~1 s at 30 Hz before failing.
        frame = None
        for _ in range(30):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.033)
        if frame is None:
            self.cap.release()
            raise RuntimeError(f"Cv2Camera({index}) opened but no frame after 1 s of polling.")
        self._latest = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_count = 1
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
                    self.frame_count += 1

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

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE for your robot                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
ROBOT_PORT = "/dev/ttyACM0"               # Linux: /dev/ttyACM0 or /dev/ttyUSB0
                                          # (run: ls /dev/tty{ACM,USB}* to find yours)
CAMERA_INDEX = 0                          # /dev/video0; use `v4l2-ctl --list-devices` to check
CALIBRATION_ID = "so101_follower_arm"     # filename (no extension) of your calibration .json
CALIBRATION_DIR = Path(__file__).parent   # folder that holds the calibration .json

# ── Contract constants (must match the training env) ───────────────────────
# IMAGE_H / IMAGE_W / CNN_FLATTEN_DIM / RGB_PROJ_DIM are DEFAULTS only — they
# get overwritten in main() from the checkpoint's actor weights so older runs
# (e.g. 16×16 or 32×32 inputs) load without code edits. See derive_arch_from_ckpt().
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
# Gripper at 120° = URDF upper limit, fully open (updated 2026-05-19 in commit b08096a).
REST_QPOS = np.deg2rad(
    np.array([-2.242, -80.791, 36.747, 86.901, -82.154, 120.0], dtype=np.float32)
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Robot driver — wraps a LeRobot SO101 follower                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def create_real_robot():
    # Camera handled by our own Cv2Camera (see RealRobotAgent.__init__);
    # lerobot's wrapper is bypassed so we can pin the V4L2 backend explicitly
    # and run a dedicated background reader.
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
        # Bypass lerobot's camera wrapper so we can pin the V4L2 backend.
        self.cameras = {"base_camera": Cv2Camera(index=CAMERA_INDEX, width=1920, height=1080, fps=30)}

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Policy network — architecture must match the checkpoint exactly          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Defaults for the 36×64 (16:9) calibration: CNN flatten = 64 * 5 * 12 = 3840
# from Conv(4,s=2) -> Conv(4,s=2) -> Conv(3,s=1). Both CNN_FLATTEN_DIM and
# RGB_PROJ_DIM are overwritten in main() based on the checkpoint's actor
# weights, so this file works for older runs (16×16, 32×32, etc.) too.
CNN_FLATTEN_DIM = 3840
RGB_PROJ_DIM = 50


def detect_encoder_arch(encoder_sd):
    """Inspect an encoder state_dict and return a list of conv-layer dicts
    [{'in_c', 'out_c', 'kernel', 'stride'}, ...] matching the training-script
    dispatch in train_squint.py: kernel sizes come from the weight shapes,
    strides are inferred from kernel size + layer count (see CNNEncoder there).
    """
    # Keys look like 'conv.0.weight', 'conv.2.weight', ... — extract & sort numerically.
    conv_keys = sorted(
        (k for k in encoder_sd if k.endswith(".weight") and k.startswith("conv.")),
        key=lambda k: int(k.split(".")[1]),
    )
    layers = []
    for k in conv_keys:
        out_c, in_c, kH, kW = encoder_sd[k].shape
        layers.append({"in_c": in_c, "out_c": out_c, "kernel": (int(kH), int(kW))})
    if len(layers) == 3 and layers[0]["kernel"] == (8, 8):
        strides = [4, 2, 1]        # H=64 family
    elif len(layers) == 3 and layers[0]["kernel"] == (4, 4):
        strides = [2, 2, 1]        # H=32 / H=36 family
    elif len(layers) == 2 and layers[0]["kernel"] == (4, 4):
        strides = [2, 1]           # H=16 family (Conv4 s=2 → Conv4 s=1)
    else:
        raise RuntimeError(
            f"Unknown encoder architecture: {len(layers)} conv layers with "
            f"kernels {[l['kernel'] for l in layers]}"
        )
    for layer, stride in zip(layers, strides):
        layer["stride"] = stride
    return layers


def _forward_spatial(layers, in_h, in_w):
    h, w = in_h, in_w
    for layer in layers:
        k = layer["kernel"][0]
        s = layer["stride"]
        h = (h - k) // s + 1
        w = (w - k) // s + 1
    return h, w


def derive_arch_from_ckpt(ckpt):
    """Extract everything needed to instantiate the policy from a checkpoint:
    conv-layer spec, (IMAGE_H, IMAGE_W), CNN_FLATTEN_DIM, RGB_PROJ_DIM, N_STATE.
    Handles all three training-time encoder variants (H=16, H=32/36, H=64)
    and the 4:3 / 16:9 / square input aspect ratios used across runs.
    """
    layers = detect_encoder_arch(ckpt["encoder"])
    final_c = layers[-1]["out_c"]

    rgb_proj_w = ckpt["actor"]["proj.rgb_proj.0.weight"]
    rgb_proj_dim, cnn_flatten_dim = rgb_proj_w.shape
    n_state = ckpt["actor"]["proj.state_proj.0.weight"].shape[1]

    spatial = cnn_flatten_dim // final_c
    if cnn_flatten_dim % final_c != 0:
        raise RuntimeError(
            f"CNN flatten dim {cnn_flatten_dim} is not a multiple of final channels "
            f"{final_c}; architecture mismatch with this script."
        )

    candidates = [
        # H=16 family (square, early training)
        (16, 16),
        # H=32 family — old 4:3 wrist camera (32×42 = flatten 1792)
        (32, 42), (24, 32), (32, 32),
        # H=36 family — current 16:9 real-camera calibration (¼ / ÷30 of 1920×1080)
        (36, 64), (45, 80), (27, 48), (18, 32), (72, 128), (36, 48),
        # H=64 family — 3-layer with bigger first kernel (kernel 8, stride 4 on L1)
        (64, 64), (64, 86), (80, 144), (80, 80), (64, 128),
        # Misc square sizes
        (24, 24), (40, 40), (48, 48), (48, 64), (60, 80),
    ]
    for h, w in candidates:
        out_h, out_w = _forward_spatial(layers, h, w)
        if out_h > 0 and out_w > 0 and out_h * out_w == spatial:
            return {
                "image_h": h, "image_w": w,
                "cnn_flatten_dim": cnn_flatten_dim,
                "rgb_proj_dim": rgb_proj_dim,
                "n_state": n_state,
                "layers": layers,
            }
    raise RuntimeError(
        f"Could not back-solve image (H, W) from CNN flatten dim {cnn_flatten_dim} "
        f"(= {final_c} × {spatial}, conv kernels {[l['kernel'] for l in layers]}, "
        f"strides {[l['stride'] for l in layers]}). Add the input size to the "
        f"candidates list in derive_arch_from_ckpt()."
    )


class CNNEncoder(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        # Default: H=32 / H=36 family (3 conv layers, kernels 4/4/3, strides 2/2/1).
        # In main() we override this with the spec detected from the checkpoint.
        if layers is None:
            layers = [
                {"in_c": 3, "out_c": 32, "kernel": (4, 4), "stride": 2},
                {"in_c": 32, "out_c": 64, "kernel": (4, 4), "stride": 2},
                {"in_c": 64, "out_c": 64, "kernel": (3, 3), "stride": 1},
            ]
        modules = []
        for layer in layers:
            modules.append(nn.Conv2d(
                layer["in_c"], layer["out_c"],
                kernel_size=layer["kernel"][0], stride=layer["stride"],
            ))
            modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        self.conv = nn.Sequential(*modules)

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


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw,
             control_hz=None, camera_hz=None, new_frames=None):
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
    if control_hz is not None:
        rr.log("perf/control_hz", rr.Scalars([float(control_hz)]))
    if camera_hz is not None:
        rr.log("perf/camera_hz", rr.Scalars([float(camera_hz)]))
    if new_frames is not None:
        rr.log("perf/new_frames_per_step", rr.Scalars([float(new_frames)]))


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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    # Declared up-front so reads in argparse defaults compose with later writes.
    global ROBOT_PORT, CAMERA_INDEX, IMAGE_H, IMAGE_W, CNN_FLATTEN_DIM, RGB_PROJ_DIM

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to ckpt.pt")
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.15, help="safety multiplier on policy action (lower = slower)")
    p.add_argument("--episode_steps", type=int, default=600, help="control steps per episode (600 @ 30 Hz = 20s)")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="open a Rerun viewer with live camera + joint plots (--no-viz to disable)")
    p.add_argument("--n_episodes", type=int, default=0, help="if >0, run this many episodes back-to-back without waiting for Enter")
    p.add_argument("--log_dir", type=str, default=None, help="if set, dump per-step npz logs there (one file per episode)")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"),
                   help="bowl position fed to the policy when the checkpoint expects a 21-d state (default: 0.25 0.10 0.00)")
    p.add_argument("--robot_port", type=str, default=ROBOT_PORT,
                   help=f"serial device for the SO101 follower (default: {ROBOT_PORT}; see `ls /dev/tty{{ACM,USB}}*`)")
    p.add_argument("--camera_index", type=int, default=CAMERA_INDEX,
                   help=f"V4L2 device index (default: {CAMERA_INDEX}; see `v4l2-ctl --list-devices`)")
    args = p.parse_args()

    # Allow CLI overrides without editing the file. These are read in
    # create_real_robot() / RealRobotAgent.__init__() via module globals.
    ROBOT_PORT = args.robot_port
    CAMERA_INDEX = args.camera_index

    if args.viz:
        init_viz()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load policy (only encoder + actor are needed; critic/log_alpha are training-only).
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Auto-detect the architecture from the checkpoint weights. This overwrites
    # the module-level defaults so:
    #   - Projection/Actor (which read these as globals at __init__) build to
    #     the checkpoint's exact widths;
    #   - preprocess_image() downsamples the camera frame to the right (H, W).
    arch = derive_arch_from_ckpt(ckpt)
    n_state_ckpt = arch["n_state"]
    use_bowl_xyz = n_state_ckpt == 21
    if n_state_ckpt not in (18, 21):
        raise RuntimeError(f"Unsupported state size in checkpoint: {n_state_ckpt} (expected 18 or 21)")

    IMAGE_H, IMAGE_W = arch["image_h"], arch["image_w"]
    CNN_FLATTEN_DIM = arch["cnn_flatten_dim"]
    RGB_PROJ_DIM = arch["rgb_proj_dim"]

    encoder = CNNEncoder(layers=arch["layers"]).to(device).eval()
    actor = Actor(n_state=n_state_ckpt).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    kernels = [l["kernel"][0] for l in arch["layers"]]
    strides = [l["stride"] for l in arch["layers"]]
    print(
        f"Loaded checkpoint (trained to step {ckpt.get('global_step', '?')}): "
        f"input={IMAGE_H}×{IMAGE_W}, cnn_flatten={CNN_FLATTEN_DIM}, "
        f"rgb_proj={RGB_PROJ_DIM}, n_state={n_state_ckpt}, "
        f"conv kernels={kernels}, strides={strides}"
        + (f" → feeding bowl_xyz={args.bowl_xyz}" if use_bowl_xyz else "")
    )

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

            ema_alpha = 0.2
            ema_control_hz, ema_camera_hz = None, None
            cam = agent.cameras["base_camera"]
            prev_cam_count = cam.frame_count
            prev_step_t = None

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                qpos = agent.get_qpos().cpu().numpy().flatten()
                agent.capture_sensor_data()
                rgb = agent.get_sensor_data()["base_camera"]["rgb"]
                cur_cam_count = cam.frame_count
                new_frames = cur_cam_count - prev_cam_count
                prev_cam_count = cur_cam_count

                if prev_step_t is not None:
                    dt = t0 - prev_step_t
                    if dt > 0:
                        inst_control_hz = 1.0 / dt
                        inst_camera_hz = new_frames / dt
                        ema_control_hz = inst_control_hz if ema_control_hz is None else (1 - ema_alpha) * ema_control_hz + ema_alpha * inst_control_hz
                        ema_camera_hz = inst_camera_hz if ema_camera_hz is None else (1 - ema_alpha) * ema_camera_hz + ema_alpha * inst_camera_hz
                prev_step_t = t0

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
                        control_hz=ema_control_hz,
                        camera_hz=ema_camera_hz,
                        new_frames=new_frames,
                    )
                if log_dir:
                    log_qpos.append(qpos.copy())
                    log_target.append(target_qpos.copy())
                    log_action_raw.append(raw_action.copy())
                    log_policy_rgb.append(obs_rgb[0].cpu().numpy().copy())

                if step % 30 == 0 and ema_control_hz is not None:
                    print(f"  step {step:4d}  control={ema_control_hz:5.1f} Hz  camera={ema_camera_hz:5.1f} Hz  new_frames={new_frames}")

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
        # Stop the camera thread before releasing the device — otherwise the
        # background reader races cap.release() at process exit and segfaults.
        for cam in agent.cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        agent.reset(REST_QPOS)
        robot.disconnect()


if __name__ == "__main__":
    main()
