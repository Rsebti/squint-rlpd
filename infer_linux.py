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
import collections
import json
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pure-numpy FK: fingertip-midpoint z for the grasp gate + IK nudge for the
# perpendicular "back" grasp correction.
from so101_fk import tcp_pos, fk_frames, nudge_arm_joints


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
CONTROL_HZ = 75          # control-loop rate (Hz). NB: the eval1 policy was
                         # trained at control_freq=10 Hz, so 75 Hz oversamples
                         # the policy 7.5×. DELTA_CAP is derived from a per-second
                         # velocity envelope below, so changing this rate keeps
                         # the same max joint velocity (finer, smoother loop)
                         # rather than scaling the robot's speed with the rate.

# Per-joint velocity envelope (rad/s): arm 1.0 (57 deg/s), gripper 3.0
# (172 deg/s, 3× arm). DELTA_CAP (rad/step) = velocity / CONTROL_HZ, so the
# policy's normalized action maps to the same per-second velocity at any rate.
ARM_VEL_LIMIT = 1.0
GRIPPER_VEL_LIMIT = 3.0
DELTA_CAP = np.array(
    [ARM_VEL_LIMIT / CONTROL_HZ] * 5 + [GRIPPER_VEL_LIMIT / CONTROL_HZ],
    dtype=np.float32,
)
# Joint limits from so101.urdf, order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
JOINT_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
JOINT_UPPER = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])
# SO101 "start" keyframe — robot rest pose. Must match envs/robot/so101.py keyframes["start"].
# pan centred at 0 (was -2.242°) to symmetrize the wrist-camera footprint.
# Gripper at 120° = URDF upper limit, fully open (updated 2026-05-19 in commit b08096a).
REST_QPOS = np.deg2rad(
    np.array([0.0, -80.791, 36.747, 86.901, -82.154, 120.0], dtype=np.float32)
)

# Gripper snap-to-close (latched). The eval1 policy commands the gripper as
# hard as it can (raw action ≈ −1) during the WHOLE approach, so the gripper
# closes gradually 115°→~10° as the arm descends; it only settles (~14°) once
# the arm reaches the cube, where the policy stops pushing and the raw action
# rises from −1 to oscillating ±. The integrated target never reaches full
# close (−10° sim) so it doesn't physically clamp.
#
# Fix: LATCH a full-close override at the grasp moment, detected by the policy
# letting off the close command (raw action > GRIPPER_LATCH_ACTION) WHILE the
# gripper target is already closed (≤ GRIPPER_SNAP_BELOW_DEG). Latching on the
# threshold ALONE fired ~20 steps too early — mid-approach, before the cube was
# reached — slamming the gripper shut on the way down. Once latched, the servo
# is driven to GRIPPER_FULL_CLOSE_DEG for the rest of the episode (pick-only:
# no release). The policy's own target_qpos observation is left untouched.
GRIPPER_SNAP_ENABLED = False      # snap-to-close disabled by default (off for now)
GRIPPER_SNAP_BELOW_DEG = 20.0     # sim deg; gripper target must be ≤ this to latch
GRIPPER_LATCH_ACTION = -0.5       # raw gripper action must rise above this to latch
                                  # (policy stopped pushing full-close = grasp reached)
GRIPPER_FULL_CLOSE_DEG = -10.0    # sim deg; the latched target (= sim full close)

# ── FK-gated hardcoded grasp (on top of the RL policy) ─────────────────────
# The policy aligns the arm to the cube but the gripper never reliably closes
# on it. The policy keeps FULL control of the gripper during the approach — its
# open/close motion is part of how it aligns the jaw to the cube, so we must NOT
# pin it. The ONLY hardcoded part is a firm final close, triggered purely from
# forward kinematics: the cube sits on the table (top face ~2 cm above z=0;
# base_link is also at z=0), so "fingertip 2 cm below the cube top" ==
# fingertip-midpoint z at table level. Sequence per attempt: approach (policy,
# gripper free) → z-gate → more policy → nudge (one IK move: shift TCP 1 cm in
# xy perpendicular to the finger line toward the base to fix the systematic
# one-finger miss, AND descend the tip to z=0.1 cm) → close → verify via gripper
# present-position → if empty, replay the last 2 s of approach commands in
# REVERSE (retreat, no IK) → rerun.
GRASP_ENABLED = True              # FK-gated hardcoded grasp (--no-grasp to disable)
GRASP_GATE_Z = 0.004              # m; base gate height ABOVE the calibrated table plane.
                                  # Effective gate = GRASP_GATE_Z + GRASP_GATE_Z_SLOPE·r.
GRASP_GATE_Z_SLOPE = 0.0          # m of extra gate height per metre of reach r (TCP xy
                                  # distance from base). The policy bottoms out higher when
                                  # extended, so far cubes need a looser gate; raise this
                                  # until far cubes trigger. 0 = constant gate (near cubes).
# Stall trigger: rather than rely on the arm hitting an absolute height (fragile
# for far cubes / imperfect table calib), also fire the gate when the descent
# PLATEAUS — the policy has taken the arm as low as it will go. Robust and needs
# no per-distance tuning; the nudge IK then finishes the descent to the table.
GRASP_GATE_STALL = True           # also fire the gate when the descent plateaus
GRASP_STALL_S = 1.0               # s of no further descent (new low) before firing
GRASP_STALL_EPS = 0.003           # m; a drop smaller than this doesn't count as descending
GRASP_ENGAGE_Z = 0.06             # m; only allow the stall-fire once the TCP is within this
                                  # height of the table (so it can't fire at the rest pose)
GRASP_CLOSE_DEG = -10.0           # sim deg; commanded full close at the grasp moment
GRASP_WAIT_S = 3.0                # s of extra policy run after the gate before closing
GRASP_NUDGE_M = 0.01              # m; corrective TCP shift before closing, in the xy
                                  # plane PERPENDICULAR to the finger-connecting line,
                                  # toward the robot base ("back"). Negate to flip.
GRASP_NUDGE_Z = -0.01             # m; target TCP height RELATIVE to the calibrated table plane
                                  # for the corrected pose (-0.01 = press 1 cm below the table).
                                  # Merged into the same IK move as the back nudge: the tip ends
                                  # 1 cm back AND at this height. Raise toward 0 if it digs in.
GRASP_NUDGE_SETTLE_S = 0.5        # s to let the servos reach the nudged pose before closing
GRASP_CLOSE_S = 1.0               # s allotted for the gripper to close + settle
GRASP_EMPTY_BELOW_DEG = -5.0      # after closing, if measured gripper > this it
                                  # stalled on an object = grasped; ≤ this = empty.
                                  # Measured: cube stalls at ~-0.5° sim, full-close
                                  # target is -10°, so -5° is the midpoint.
GRASP_MAX_RETRIES = 3             # back-off + retry attempts before giving up
GRASP_RETREAT_UP_M = 0.10         # on a miss, IK the TCP this far UP …
GRASP_RETREAT_BACK_M = 0.05       # … and this far BACK toward the base (to view where the
                                  # cube went), then rerun the policy
GRASP_RETREAT_SPEED = 0.20        # m/s the TCP eases through the back-off move
GRASP_HOLD_S = 0.0                # s to hold the closed grasp before lifting (0 = lift immediately)
GRASP_LIFT_M = 0.05               # m to raise the TCP (cube) after a confirmed grasp
GRASP_LIFT_S = 1.5                # s allotted to complete the lift before ending the episode

# Table-plane calibration: the FK z that actually corresponds to "touching the
# table" drifts with reach, so z_table(r) = TABLE_Z_A·r + TABLE_Z_B (r = TCP
# horizontal distance from base). The gate/nudge are taken RELATIVE to this
# plane. Loaded from table_z_calib.json (examples/table_z_calib.py); both 0 ⇒
# flat z=0 assumption (pre-calibration behaviour).
TABLE_Z_CALIB_PATH = Path(__file__).parent / "table_z_calib.json"
TABLE_Z_A = 0.0
TABLE_Z_B = 0.0


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
        """Send a joint-angle target (sim radians) to the servos. The gripper
        snap-to-close is applied by the caller (main loop) via the latch, NOT
        here, so the open-gripper reset ramp isn't fought by the override."""
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


# ── Table background masking ─────────────────────────────────────────────────
# The sim trains with a controlled background; the real camera sees whatever
# room/wall/curtain is behind the table edge. mask_background_to_table() detects
# the table (white/desaturated OR coloured) and paints every pixel OUTSIDE the
# table's convex hull with the table's mean colour, so the policy sees a clean
# background matching the sim. Set via --table_mask. Tuning: --table_val_band
# (main curtain knob; lower = excludes dimmer background), --table_sat_band.
TABLE_MASK_ENABLED = False
TABLE_SAT_BAND = 45
TABLE_VAL_BAND = 80
TABLE_WHITE_SAT_THRESH = 60
TABLE_HUE_TOL = 14
TABLE_DETECT_W = 320          # detection runs at this width (fast); mask is then
                              # applied and the result downsampled to the CNN size
LOWER_TABLE_Y = 0.45          # the robot/camera are fixed to the table, so the LOWER part of
                              # the frame is always table (+ the gripper rising from the bottom
                              # edge), never background. Force everything below this fraction to
                              # count as table so the background mask can't grey the gripper.
LAST_MASKED_VIZ = None        # last masked detection-res RGB frame, for the viewer

# Non-goal cube masking: on top of the table mask, grey ONLY the pixels that
# match the OTHER cubes' colours (the goal cube is never touched, so all of its
# faces — including dark ones — survive). A pixel is a non-goal cube if it's
# saturated and its NEAREST palette colour is one of the non-goal colours. The
# table, gripper, bowl and the goal cube are all left as-is.
COLOR_DISTRACTOR_MASK = True  # grey non-goal cube colours on top of the table mask
DISTRACTOR_SAT_MIN = 80       # a pixel is a coloured cube only if S ≥ this (0–255);
                              # the table is desaturated and stays below it
NONGOAL_V_MIN = 35            # ...AND bright enough (V ≥ this), to drop the near-black gripper.
                              # Kept low so a cube that falls into shadow / vignette when the
                              # gripper closes in off-centre (V can drop to ~50) is still masked.
                              # The gripper's near-black S=255 artifact is now excluded mainly by
                              # the (hue,sat) COLOR_DIST_TOL test, so this floor only needs to clear
                              # true black (V≈3) with margin.
COLOR_DIST_TOL = 26           # max (hue,sat) distance to a cube centroid to count the pixel as
                              # that cube (so random saturated stuff isn't greyed). Combined
                              # distance: sqrt(huedist^2 + (SAT_DIST_W*satdist_scaled)^2).
SAT_DIST_W = 1.0              # weight on saturation in the colour classifier. Saturation is what
                              # separates hue-adjacent cubes (blue vs purple) when shadow/close-
                              # range drift their hues together; raise to lean harder on it.
LOW_SAT_FLOOR = 45            # connectivity grow floor: from a confident bright non-goal face,
                              # absorb TOUCHING coloured pixels down to this saturation (the
                              # cube's darker faces). Kept above the table's saturation so the
                              # grow can't bleed into the table; the goal cube is excluded too.
MASK_GROW_PX = 3              # grow the greyed non-goal regions this many px (detect-res) to
                              # cover their borders
GOAL_PROTECT_FRAC = 0.5       # no-mask halo around the goal cube, as a fraction of the cube's
                              # width — keeps the gripper↔cube gap from being masked at grasp
# Aggressive close-range mode: once the gripper tip is below MASK_AGG_Z above the
# table, the goal cube sits between the fingers (centre-bottom of the wrist view).
# We then identify the goal by POSITION (the central blob) — confirmed it colour-
# matches the goal — and mask EVERY other saturated cube blob wholesale, instead of
# per-pixel colour-classifying distractors (which is fragile in shadow / at edges).
MASK_AGGRESSIVE = False        # set per-step by set_mask_aggressive() from the tip height
MASK_AGG_Z = 0.08              # tip height above table (m) below which aggressive mode is on
AGG_CUBE_TOL = 45              # looser (hue,sat) distance for "is this pixel any cube" in aggressive
                               # mode — catches shadow-shifted distractors while excluding the
                               # near-black gripper (whose random hue stays far from every centroid)
AGG_MIN_AREA = 25              # ignore blobs smaller than this many detect-res px (specks/noise)
AGG_GOAL_FRAC = 0.40           # central blob must have ≥ this fraction of pixels classed as the
                               # goal colour to confirm it's the goal (else fall back to colour mask)
AGG_ZONE_Y = 0.72             # vertical position of the grasp point (between the fingers), as a
                              # fraction of image height — the central goal blob sits near here
# OpenCV hues (0–179) of the COLOR_PALETTE in envs/place.py, index = goal_color.
#                0 red 1 blue 2 green 3 yellow 4 purple 5 orange
GOAL_HUE_CV = [4, 112, 64, 26, 149, 8]
# OpenCV saturations (0–255) of the same palette, used together with hue to tell
# hue-adjacent cubes apart (blue↔purple). Calibration overrides these per rig.
GOAL_SAT_CV = [249, 253, 242, 242, 235, 246]
# Optional measured override (final_utils/calib_colors.py) — real cube hue+sat under
# the actual camera/lighting beat the palette-derived defaults above.
HUE_CALIB_PATH = Path(__file__).parent / "hue_calib.json"


def load_hue_calib():
    """If hue_calib.json exists, override GOAL_HUE_CV / GOAL_SAT_CV with measured values."""
    global GOAL_HUE_CV, GOAL_SAT_CV
    if HUE_CALIB_PATH.exists():
        d = json.loads(HUE_CALIB_PATH.read_text())
        GOAL_HUE_CV = [int(round(h)) for h in d["hues"]]
        if "sat" in d:
            GOAL_SAT_CV = [int(round(s)) for s in d["sat"]]
        print(f"Loaded colour calib (measured): hue={GOAL_HUE_CV} sat={GOAL_SAT_CV}")
        return True
    return False


def set_mask_aggressive(qpos):
    """Flip the close-range aggressive mask on when the FK gripper tip drops below
    MASK_AGG_Z above the (reach-corrected) table. Call once per control step before
    preprocess_image(); mask_background_to_table() reads the MASK_AGGRESSIVE global."""
    global MASK_AGGRESSIVE
    tcp = tcp_pos(qpos)
    r = float(np.hypot(tcp[0], tcp[1]))
    z_table = TABLE_Z_A * r + TABLE_Z_B
    MASK_AGGRESSIVE = (float(tcp[2]) - z_table) < MASK_AGG_Z
    return MASK_AGGRESSIVE


def mask_background_to_table(rgb_img, goal_color=None, aggressive=None):
    """Two-step mask, in RGB:

    1. Table mask — detect the table (connected region from a bottom-centre seed)
       and paint everything OUTSIDE its convex hull with the table's mean colour
       (greys the room/wall/desk behind the table). Everything ON the table stays.
    2. Non-goal cubes — if `goal_color` is given and COLOR_DISTRACTOR_MASK is on,
       grey the pixels that match the OTHER cubes' colours (saturated AND nearest
       palette colour ≠ goal). The GOAL cube is never targeted, so all of its
       faces survive; the table, gripper and bowl are left as-is.

    Returns the masked RGB image, or the original frame if detection fails.
    """
    h, w = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    y_lo = int(0.55 * h)
    x_lo, x_hi = int(0.30 * w), int(0.70 * w)
    if S[y_lo:, x_lo:x_hi].size == 0:
        return rgb_img
    sH = int(np.median(H[y_lo:, x_lo:x_hi]))
    sS = int(np.median(S[y_lo:, x_lo:x_hi]))
    sV = int(np.median(V[y_lo:, x_lo:x_hi]))
    seed_pt = (w // 2, int(0.80 * h))

    if sS < TABLE_WHITE_SAT_THRESH:
        table_px = (S <= min(255, sS + TABLE_SAT_BAND)) & (V >= max(0, sV - TABLE_VAL_BAND))
    else:
        lo, hi = sH - TABLE_HUE_TOL, sH + TABLE_HUE_TOL
        if lo < 0:
            h_mask = (H >= (180 + lo)) | (H <= hi)
        elif hi > 179:
            h_mask = (H >= lo) | (H <= (hi - 180))
        else:
            h_mask = (H >= lo) & (H <= hi)
        table_px = h_mask & (S >= max(0, sS - TABLE_SAT_BAND)) & (V >= max(0, sV - TABLE_VAL_BAND))

    kern = np.ones((5, 5), np.uint8)
    table_u8 = table_px.astype(np.uint8) * 255
    table_u8 = cv2.morphologyEx(table_u8, cv2.MORPH_CLOSE, kern, iterations=3)
    table_u8 = cv2.morphologyEx(table_u8, cv2.MORPH_OPEN, kern, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(table_u8, 8)
    if num <= 1:
        return rgb_img
    sx, sy = int(np.clip(seed_pt[0], 0, w - 1)), int(np.clip(seed_pt[1], 0, h - 1))
    seed_label = int(labels[sy, sx]) or (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    if stats[seed_label, cv2.CC_STAT_AREA] < 0.05 * H.size:
        return rgb_img
    table_mask = (labels == seed_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb_img
    hull = cv2.convexHull(np.concatenate(contours))
    hull_mask = np.zeros_like(table_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    hull_mask[int(LOWER_TABLE_Y * h):, :] = 255     # lower band is always table (gripper-safe)

    mean_rgb = rgb_img[table_mask > 0].reshape(-1, 3).mean(axis=0)
    out = rgb_img.copy()
    # 1. Background → ONE clean region (smoothed, hole-free), filled with the table mean.
    bg = cv2.morphologyEx((hull_mask == 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern, iterations=2)
    out[bg > 0] = mean_rgb

    # 2. Mask the non-goal cubes (never the goal cube), absorbing their darker
    #    faces by growing from each confident bright face into the touching,
    #    same-ish-colour pixels, then INPAINTING from neighbouring pixels so each
    #    filled patch takes the LOCAL table colour, not the global mean.
    if goal_color is not None and COLOR_DISTRACTOR_MASK:
        gi = int(goal_color)
        # Classify each pixel by its nearest cube in (hue, saturation) space, NOT hue
        # alone — hue-adjacent cubes (blue↔purple) drift together under shadow/close
        # range, but their saturations stay apart, so sat is what keeps them separate.
        # Saturation (0–255) is rescaled to hue units (0–180) so the two are comparable.
        Hi = H.astype(np.float32)
        Si = S.astype(np.float32)
        sat_scale = 180.0 / 255.0
        hue_d = np.stack(
            [np.minimum(np.abs(Hi - hc), 180 - np.abs(Hi - hc)) for hc in GOAL_HUE_CV], axis=0)
        sat_d = np.stack([np.abs(Si - sc) for sc in GOAL_SAT_CV], axis=0) * sat_scale
        dstack = np.sqrt(hue_d ** 2 + (SAT_DIST_W * sat_d) ** 2)
        nearest = dstack.argmin(axis=0)
        dmin = dstack.min(axis=0)

        # ── Aggressive close-range mode ─────────────────────────────────────────
        # Once the gripper is closing in, the goal cube sits between the fingers
        # (centre-bottom). Identify it by POSITION (the central blob), confirm it is
        # the goal colour, then mask EVERY other saturated cube blob wholesale — no
        # fragile per-pixel colour test on shadowed / frame-edge distractors.
        agg = MASK_AGGRESSIVE if aggressive is None else aggressive
        if agg:
            # All cube material (any colour): saturated, bright, near SOME cube centroid
            # — so the near-black gripper (random hue, far from every centroid) and the
            # desaturated table are excluded.
            cube = (S >= DISTRACTOR_SAT_MIN) & (V >= NONGOAL_V_MIN) & (dmin <= AGG_CUBE_TOL)
            cube_u8 = cv2.morphologyEx(cube.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            # The goal cube = the GOAL-COLOURED blob nearest the grasp point (between the
            # fingers). Pick among goal-coloured blobs so a stray highlight that misreads as
            # another colour can't preempt the real goal cube.
            goalpx = cv2.morphologyEx(((cube_u8 > 0) & (nearest == gi)).astype(np.uint8) * 255,
                                      cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nG, labG, statsG, centsG = cv2.connectedComponentsWithStats(goalpx, 8)
            gpt = (0.5 * w, AGG_ZONE_Y * h)
            best, bestd = 0, 1e18
            for L in range(1, nG):
                if statsG[L, cv2.CC_STAT_AREA] < AGG_MIN_AREA:
                    continue
                cx, cy = centsG[L]
                dd = (cx - gpt[0]) ** 2 + (cy - gpt[1]) ** 2
                if dd < bestd:
                    bestd, best = dd, L
            if best > 0:                              # a central goal cube is present → aggressive
                central = labG == best
                # Protect the goal cube + a halo (covers its own faces/highlights that may
                # misread as another colour) so we never paint a hole in the goal.
                area = int(central.sum())
                r_goal = (area / np.pi) ** 0.5
                pr = max(2, int(round(GOAL_PROTECT_FRAC * 2 * r_goal)))
                gk = cv2.dilate(central.astype(np.uint8) * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pr + 1, 2 * pr + 1)))
                # Mask EVERY cube pixel outside the goal-cube region — whole distractor blobs,
                # regardless of colour/shadow, with no fragile per-pixel test.
                ng_u8 = cube_u8.copy()
                if MASK_GROW_PX > 0:
                    r = int(MASK_GROW_PX)
                    ng_u8 = cv2.dilate(ng_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                ng_u8[gk > 0] = 0
                if ng_u8.any():
                    out = cv2.inpaint(out, ng_u8, 4, cv2.INPAINT_TELEA)
                return out
            # no central goal cube → fall back to the colour mask below

        # Confident bright non-goal faces = seeds (bright enough to exclude the gripper).
        seed = (S >= DISTRACTOR_SAT_MIN) & (V >= NONGOAL_V_MIN) & (nearest != gi) & (dmin <= COLOR_DIST_TOL)
        # Candidate cube material: any coloured non-goal pixel down to a floor kept
        # above the table's saturation (so it can't bleed into the table) and above
        # near-black (so the gripper is excluded); the goal cube is excluded too.
        floor = max(LOW_SAT_FLOOR, sS + 20)
        cand = (S >= floor) & (V >= NONGOAL_V_MIN) & (nearest != gi)
        cand_u8 = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_lab, lab = cv2.connectedComponents(cand_u8, 8)
        keep_labels = np.unique(lab[seed & (lab > 0)])          # components touching a seed
        ng = np.isin(lab, keep_labels) & (lab > 0)
        ng_u8 = ng.astype(np.uint8) * 255
        if MASK_GROW_PX > 0:
            r = int(MASK_GROW_PX)
            ng_u8 = cv2.dilate(ng_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        # No-mask zone around the goal cube: protect the cube + a halo (~GOAL_PROTECT_FRAC
        # of its width) so the gripper↔cube gap isn't masked as the grasp closes in.
        goal_keep = (S >= DISTRACTOR_SAT_MIN) & (V >= NONGOAL_V_MIN) & (nearest == gi) & (dstack[gi] <= COLOR_DIST_TOL)
        gk = cv2.morphologyEx(goal_keep.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        area = int((gk > 0).sum())
        if area > 0:
            r_goal = (area / np.pi) ** 0.5                # equivalent cube radius (px)
            pr = max(1, int(round(GOAL_PROTECT_FRAC * 2 * r_goal)))
            gk = cv2.dilate(gk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pr + 1, 2 * pr + 1)))
        # Protect only the gripper↔goal gap (boundary/grow pixels), NEVER re-reveal a
        # confident non-goal cube face that the halo happens to overlap — the policy is
        # colour-blind, so an un-masked neighbour cube (e.g. purple next to blue) would
        # capture it. Keep the non-goal seeds masked even inside the protect zone.
        seed_u8 = cv2.dilate(seed.astype(np.uint8) * 255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ng_u8[(gk > 0) & (seed_u8 == 0)] = 0
        if ng_u8.any():                                  # inpaint from neighbouring table pixels
            out = cv2.inpaint(out, ng_u8, 4, cv2.INPAINT_TELEA)
    return out


# ── Observation / action helpers ────────────────────────────────────────────
def preprocess_image(rgb, goal_color=None):
    """Real camera frame (1,H,W,3) uint8 → (1, IMAGE_H, IMAGE_W, 3) uint8 tensor.

    The real wrist camera was calibrated 2026-05-19 at 1920×1080 (16:9). The
    sim renders 640×360 (¼ res, same 16:9). Both are area-downsampled to
    (IMAGE_H, IMAGE_W), an exact 16:9 grid. No center-crop — the 16:9 aspect is
    preserved end-to-end with uniform pooling.

    When TABLE_MASK_ENABLED, the background is replaced with the table's mean
    colour BEFORE downsampling: detection runs at TABLE_DETECT_W (fast), the
    mask is applied, then the masked frame is downsampled to the CNN size.
    """
    global LAST_MASKED_VIZ
    img = rgb[0].cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb[0])
    if TABLE_MASK_ENABLED:
        det_w = TABLE_DETECT_W
        det_h = int(round(det_w * img.shape[0] / img.shape[1]))
        det = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_AREA)
        det = mask_background_to_table(det, goal_color)
        LAST_MASKED_VIZ = det           # full detection-res masked frame, for the viewer
        img = det
    else:
        LAST_MASKED_VIZ = None
    img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(torch.uint8)


def init_viz(memory_limit: str = "256MB"):
    """Spawn a Rerun viewer for live camera + joint plots. Returns True if the
    viewer started, False if rerun isn't installed (deploy then continues
    without the viewer instead of crashing).

    `memory_limit` caps the viewer's in-memory store: when exceeded, Rerun
    drops the OLDEST logged data first, so a long episode can't grow unbounded
    and OOM-crash the viewer. With the downsampled raw frame (~170 KB) +
    masked + policy frames (~380 KB/step total), 256 MB holds a rolling window
    of several seconds at 30 Hz. Lower it (e.g. 64MB ≈ 3 s) for a tighter
    window, raise it for more scrollback."""
    if rr is None:
        print("[viz] rerun not installed (pip install rerun-sdk) — continuing without viewer.")
        return False
    rr.init("squint_infer")
    rr.spawn(memory_limit=memory_limit)
    return True


def log_step(step, raw_rgb, policy_rgb, qpos, target_qpos, action_raw,
             control_hz=None, camera_hz=None, new_frames=None):
    """Push one timestep to the Rerun viewer."""
    if rr is None:
        return
    rr.set_time("step", sequence=step)
    rr.set_time("wall", timestamp=time.time())
    # Downsample the raw frame before logging — a full 1920×1080 frame is ~6 MB
    # and at 30 Hz floods the viewer's memory store within seconds. 320 px wide
    # (~170 KB) keeps the stream light so the memory-limit GC has slack.
    raw = np.asarray(raw_rgb)
    if raw.shape[1] > 320:
        rh = int(round(320 * raw.shape[0] / raw.shape[1]))
        raw = cv2.resize(raw, (320, rh), interpolation=cv2.INTER_AREA)
    rr.log("camera/raw", rr.Image(raw))
    # Masked frame (what the CNN sees, before the final downsample) — clearest
    # view of the table-masking quality at deploy.
    if LAST_MASKED_VIZ is not None:
        rr.log("camera/masked", rr.Image(LAST_MASKED_VIZ))
    rr.log("camera/policy_input", rr.Image(policy_rgb))
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


def back_nudge_joint_target(qpos, target_qpos, nudge_m, target_z):
    """Shift target_qpos to the corrected grasp pose: TCP moved `nudge_m` in the
    xy plane PERPENDICULAR to the finger-connecting line, in the 'back' sense (the
    perpendicular pointing toward the robot base), AND down to `target_z`. Both are
    merged into one damped-LS IK move (FK Jacobian).

    Returns (new_target_qpos, info_str).
    """
    cur = tcp_pos(qpos)
    delta = np.array([0.0, 0.0, target_z - cur[2]])   # z descent always applied
    dir_info = "no-xy"
    if nudge_m:
        frames = fk_frames(qpos)
        f1 = frames["finger1_tip"][:3, 3]
        f2 = frames["finger2_tip"][:3, 3]
        line = (f2 - f1)[:2]                          # finger-connecting line in xy
        n = float(np.linalg.norm(line))
        if n >= 1e-9:
            line /= n
            perp = np.array([-line[1], line[0]])      # perpendicular in xy
            if np.dot(perp, -cur[:2]) < 0:            # pick 'back' (toward base, at xy origin)
                perp = -perp
            delta[0], delta[1] = perp[0] * nudge_m, perp[1] * nudge_m
            dir_info = f"dir=({perp[0]:+.2f},{perp[1]:+.2f}) {nudge_m*100:.2f}cm"
    dq = nudge_arm_joints(qpos, delta)
    out = target_qpos.copy()
    out[:5] = np.clip(out[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
    return out, f"{dir_info} z→{target_z*100:.2f}cm"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    # Declared up-front so reads in argparse defaults compose with later writes.
    global ROBOT_PORT, CAMERA_INDEX, IMAGE_H, IMAGE_W, CNN_FLATTEN_DIM, RGB_PROJ_DIM
    global TABLE_MASK_ENABLED, TABLE_SAT_BAND, TABLE_VAL_BAND, TABLE_WHITE_SAT_THRESH, COLOR_DISTRACTOR_MASK
    global GRIPPER_SNAP_ENABLED, GRIPPER_SNAP_BELOW_DEG, GRIPPER_FULL_CLOSE_DEG, GRIPPER_LATCH_ACTION
    global GRASP_ENABLED, GRASP_GATE_Z, GRASP_GATE_Z_SLOPE, GRASP_WAIT_S, GRASP_CLOSE_S, GRASP_EMPTY_BELOW_DEG
    global GRASP_GATE_STALL, GRASP_STALL_S, GRASP_STALL_EPS, GRASP_ENGAGE_Z
    global GRASP_MAX_RETRIES, GRASP_CLOSE_DEG, GRASP_NUDGE_M, GRASP_NUDGE_Z, GRASP_NUDGE_SETTLE_S
    global GRASP_HOLD_S, GRASP_LIFT_M, GRASP_LIFT_S
    global TABLE_Z_A, TABLE_Z_B

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to ckpt.pt")
    p.add_argument("--goal_color", type=int, default=0, help="0 red 1 blue 2 green 3 yellow 4 purple 5 orange")
    p.add_argument("--action_scale", type=float, default=0.15, help="safety multiplier on policy action (lower = slower)")
    p.add_argument("--episode_steps", type=int, default=600, help="control steps per episode (600 @ 30 Hz = 20s)")
    p.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="open a Rerun viewer with live camera + joint plots (--no-viz to disable)")
    p.add_argument("--viz_memory_limit", type=str, default="256MB",
                   help="Rerun viewer memory cap; oldest data is dropped past this so long "
                        "episodes can't OOM-crash the viewer (e.g. 64MB for a ~3 s window).")
    p.add_argument("--n_episodes", type=int, default=0, help="if >0, run this many episodes back-to-back without waiting for Enter")
    p.add_argument("--log_dir", type=str, default=None, help="if set, dump per-step npz logs there (one file per episode)")
    p.add_argument("--bowl_xyz", type=float, nargs=3, default=[0.25, 0.10, 0.00],
                   metavar=("X", "Y", "Z"),
                   help="bowl position fed to the policy when the checkpoint expects a 21-d state (default: 0.25 0.10 0.00)")
    p.add_argument("--robot_port", type=str, default=ROBOT_PORT,
                   help=f"serial device for the SO101 follower (default: {ROBOT_PORT}; see `ls /dev/tty{{ACM,USB}}*`)")
    p.add_argument("--camera_index", type=int, default=CAMERA_INDEX,
                   help=f"V4L2 device index (default: {CAMERA_INDEX}; see `v4l2-ctl --list-devices`)")
    p.add_argument("--table_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="mask the background to the table's mean colour before the CNN (default ON)")
    p.add_argument("--distractor_mask", action=argparse.BooleanOptionalAction, default=COLOR_DISTRACTOR_MASK,
                   help="grey the non-goal cubes (and the aggressive close-range mask) on top of the table "
                        f"mask (default {'ON' if COLOR_DISTRACTOR_MASK else 'OFF'}; --no-distractor_mask keeps "
                        "ALL cubes visible while still masking the background)")
    p.add_argument("--table_val_band", type=int, default=TABLE_VAL_BAND,
                   help=f"white-table value band; LOWER excludes dimmer background like curtains (default {TABLE_VAL_BAND})")
    p.add_argument("--table_sat_band", type=int, default=TABLE_SAT_BAND,
                   help=f"saturation band around the table seed (default {TABLE_SAT_BAND})")
    p.add_argument("--table_white_sat_thresh", type=int, default=TABLE_WHITE_SAT_THRESH,
                   help=f"seed saturation below this → white-table mode (default {TABLE_WHITE_SAT_THRESH})")
    p.add_argument("--gripper_snap", action=argparse.BooleanOptionalAction, default=False,
                   help="snap the gripper to full close once the policy commands at/below the threshold (default OFF; pass --gripper_snap to enable)")
    p.add_argument("--gripper_snap_below_deg", type=float, default=GRIPPER_SNAP_BELOW_DEG,
                   help=f"gripper target (sim deg) must be ≤ this to latch the snap (default {GRIPPER_SNAP_BELOW_DEG})")
    p.add_argument("--gripper_latch_action", type=float, default=GRIPPER_LATCH_ACTION,
                   help=f"raw gripper action must rise above this to latch — i.e. the policy stopped pushing full-close = grasp reached (default {GRIPPER_LATCH_ACTION})")
    p.add_argument("--gripper_full_close_deg", type=float, default=GRIPPER_FULL_CLOSE_DEG,
                   help=f"latched target in sim deg; -10 = sim full close, lower = harder clamp (default {GRIPPER_FULL_CLOSE_DEG})")
    p.add_argument("--grasp", action=argparse.BooleanOptionalAction, default=GRASP_ENABLED,
                   help="FK-gated hardcoded grasp on top of the policy: hold gripper open during approach, "
                        "close when the fingertip midpoint descends to the cube, verify, retreat+retry on a miss "
                        f"(default {'ON' if GRASP_ENABLED else 'OFF'}; --no-grasp to disable)")
    p.add_argument("--grasp_gate_z", type=float, default=GRASP_GATE_Z,
                   help=f"base gate height (m) above the calibrated table; gate = this + slope·r (default {GRASP_GATE_Z})")
    p.add_argument("--grasp_gate_z_slope", type=float, default=GRASP_GATE_Z_SLOPE,
                   help=f"extra gate height (m) per metre of reach r, so far/extended cubes trigger (default {GRASP_GATE_Z_SLOPE})")
    p.add_argument("--grasp_gate_stall", action=argparse.BooleanOptionalAction, default=GRASP_GATE_STALL,
                   help=f"also fire the gate when the descent plateaus near the table (robust, no tuning; default {'ON' if GRASP_GATE_STALL else 'OFF'})")
    p.add_argument("--grasp_stall_s", type=float, default=GRASP_STALL_S,
                   help=f"seconds of no further descent before the stall trigger fires (default {GRASP_STALL_S})")
    p.add_argument("--grasp_stall_eps", type=float, default=GRASP_STALL_EPS,
                   help=f"a descent smaller than this (m) doesn't count as still descending (default {GRASP_STALL_EPS})")
    p.add_argument("--grasp_engage_z", type=float, default=GRASP_ENGAGE_Z,
                   help=f"stall trigger only allowed once TCP is within this height (m) of the table (default {GRASP_ENGAGE_Z})")
    p.add_argument("--grasp_wait_s", type=float, default=GRASP_WAIT_S,
                   help=f"keep running the policy this long after the gate before closing (default {GRASP_WAIT_S}s)")
    p.add_argument("--grasp_nudge_m", type=float, default=GRASP_NUDGE_M,
                   help="corrective TCP shift (m) before closing, in the xy plane perpendicular to the "
                        f"finger line, toward the base; negate to flip direction, 0 to disable (default {GRASP_NUDGE_M})")
    p.add_argument("--grasp_nudge_z", type=float, default=GRASP_NUDGE_Z,
                   help=f"target TCP height (m) RELATIVE to the calibrated table for the corrected pose, "
                        f"merged into the nudge IK move; negative presses into the table (default {GRASP_NUDGE_Z})")
    p.add_argument("--grasp_nudge_settle_s", type=float, default=GRASP_NUDGE_SETTLE_S,
                   help=f"time to let the servos reach the nudged pose before closing (default {GRASP_NUDGE_SETTLE_S}s)")
    p.add_argument("--grasp_close_s", type=float, default=GRASP_CLOSE_S,
                   help=f"time allotted for the gripper to close + settle before verifying (default {GRASP_CLOSE_S}s)")
    p.add_argument("--grasp_empty_below_deg", type=float, default=GRASP_EMPTY_BELOW_DEG,
                   help=f"after closing, measured gripper > this = stalled on object = grasped; ≤ this = empty (default {GRASP_EMPTY_BELOW_DEG})")
    p.add_argument("--grasp_max_retries", type=int, default=GRASP_MAX_RETRIES,
                   help=f"reopen+retreat+rerun attempts before giving up (default {GRASP_MAX_RETRIES})")
    p.add_argument("--grasp_hold_s", type=float, default=GRASP_HOLD_S,
                   help=f"hold the closed grasp this long before lifting (default {GRASP_HOLD_S}s)")
    p.add_argument("--grasp_lift_m", type=float, default=GRASP_LIFT_M,
                   help=f"raise the cube this far (m) after a confirmed grasp, then end the episode (default {GRASP_LIFT_M})")
    p.add_argument("--grasp_lift_s", type=float, default=GRASP_LIFT_S,
                   help=f"time to complete the lift before ending the episode (default {GRASP_LIFT_S}s)")
    args = p.parse_args()

    # Allow CLI overrides without editing the file. These are read in
    # create_real_robot() / RealRobotAgent.__init__() via module globals.
    ROBOT_PORT = args.robot_port
    CAMERA_INDEX = args.camera_index
    TABLE_MASK_ENABLED = args.table_mask
    COLOR_DISTRACTOR_MASK = args.distractor_mask
    TABLE_VAL_BAND = args.table_val_band
    TABLE_SAT_BAND = args.table_sat_band
    TABLE_WHITE_SAT_THRESH = args.table_white_sat_thresh
    GRIPPER_SNAP_ENABLED = args.gripper_snap
    GRIPPER_SNAP_BELOW_DEG = args.gripper_snap_below_deg
    GRIPPER_LATCH_ACTION = args.gripper_latch_action
    GRIPPER_FULL_CLOSE_DEG = args.gripper_full_close_deg
    GRASP_ENABLED = args.grasp
    GRASP_GATE_Z = args.grasp_gate_z
    GRASP_GATE_Z_SLOPE = args.grasp_gate_z_slope
    GRASP_GATE_STALL = args.grasp_gate_stall
    GRASP_STALL_S = args.grasp_stall_s
    GRASP_STALL_EPS = args.grasp_stall_eps
    GRASP_ENGAGE_Z = args.grasp_engage_z
    GRASP_WAIT_S = args.grasp_wait_s
    GRASP_NUDGE_M = args.grasp_nudge_m
    GRASP_NUDGE_Z = args.grasp_nudge_z
    GRASP_NUDGE_SETTLE_S = args.grasp_nudge_settle_s
    GRASP_CLOSE_S = args.grasp_close_s
    GRASP_EMPTY_BELOW_DEG = args.grasp_empty_below_deg
    GRASP_MAX_RETRIES = args.grasp_max_retries
    GRASP_HOLD_S = args.grasp_hold_s
    GRASP_LIFT_M = args.grasp_lift_m
    GRASP_LIFT_S = args.grasp_lift_s

    # Table-plane calibration (examples/table_z_calib.py). Absent ⇒ flat z=0.
    if TABLE_Z_CALIB_PATH.exists():
        c = json.loads(TABLE_Z_CALIB_PATH.read_text())
        TABLE_Z_A, TABLE_Z_B = float(c["a"]), float(c["b"])
        print(f"Table-z calib: z_table(r) = {TABLE_Z_A:.4f}·r + {TABLE_Z_B:.4f} "
              f"(n={c.get('n')}, rmse={c.get('rmse_m', 0)*100:.2f} cm, "
              f"r∈[{c.get('r_min', 0)*100:.0f},{c.get('r_max', 0)*100:.0f}] cm)")
    else:
        print("No table_z_calib.json — using flat z=0 table assumption "
              "(run examples/table_z_calib.py to fix reach-dependent table height).")
    load_hue_calib()

    viz_on = init_viz(memory_limit=args.viz_memory_limit) if args.viz else False

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
            gripper_latched = False                      # snap-to-close latch (per episode)

            # FK-gated hardcoded grasp state (per episode). Phases:
            #   approach → wait → close → (hold | retreat) ; retreat → approach
            grasp_phase = "approach"
            grasp_phase_ctr = 0
            grasp_retries = 0
            grasp_close_rad = float(np.deg2rad(GRASP_CLOSE_DEG))
            grasp_open_rad = float(np.deg2rad(120.0))
            grasp_wait_steps = max(1, int(round(GRASP_WAIT_S * CONTROL_HZ)))
            grasp_nudge_steps = max(1, int(round(GRASP_NUDGE_SETTLE_S * CONTROL_HZ)))
            grasp_close_steps = max(1, int(round(GRASP_CLOSE_S * CONTROL_HZ)))
            grasp_hold_steps = max(1, int(round(GRASP_HOLD_S * CONTROL_HZ)))
            grasp_lift_steps = max(1, int(round(GRASP_LIFT_S * CONTROL_HZ)))
            grasp_retreat_target = None                  # set on a miss (IK back-off target)
            grasp_result = None                          # "success" | "failed" when terminal
            grasp_min_above = float("inf")               # lowest tcp_above seen (stall detector)
            grasp_stall_ctr = 0
            grasp_stall_steps = max(1, int(round(GRASP_STALL_S * CONTROL_HZ)))
            tcp_z = float("nan")

            log_qpos, log_target, log_action_raw, log_policy_rgb = [], [], [], []

            ema_alpha = 0.2
            ema_control_hz, ema_camera_hz = None, None
            cam = agent.cameras["base_camera"]
            prev_cam_count = cam.frame_count
            prev_step_t = None

            for step in range(args.episode_steps):
                t0 = time.perf_counter()

                qpos = agent.get_qpos().cpu().numpy().flatten()
                # Hide the snap-to-close clamp from the policy: once latched the
                # servo is forced to −10° but the policy commanded ~10°; that
                # huge gripper tracking error is off-distribution (never happens
                # in sim) and makes the policy "think" it grasped and abort the
                # arm approach. Report the gripper at its intended (commanded)
                # position so the arm keeps approaching exactly as un-clamped.
                if gripper_latched:
                    qpos[5] = target_qpos[5]
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

                set_mask_aggressive(qpos)            # aggressive mask when tip is close to the table
                obs_rgb = preprocess_image(rgb, args.goal_color).to(device)
                obs_state = build_state(
                    qpos, target_qpos, args.goal_color,
                    bowl_xyz=args.bowl_xyz if use_bowl_xyz else None,
                ).to(device)

                with torch.no_grad():
                    raw_action = actor(encoder(obs_rgb), obs_state)[0].cpu().numpy()

                action = np.clip(raw_action * args.action_scale, -1.0, 1.0)
                tcp_xyz = tcp_pos(qpos)                   # fingertip midpoint (base frame)
                tcp_z = float(tcp_xyz[2])
                tcp_r = float(np.hypot(tcp_xyz[0], tcp_xyz[1]))   # reach: xy distance from base
                z_table = TABLE_Z_A * tcp_r + TABLE_Z_B
                tcp_above = tcp_z - z_table               # height above the calibrated table
                gate_z_eff = GRASP_GATE_Z + GRASP_GATE_Z_SLOPE * tcp_r   # looser when extended

                if GRASP_ENABLED:
                    # ── FK-gated hardcoded grasp state machine ──────────────
                    if grasp_phase == "approach":
                        # Policy drives BOTH arm and gripper (the gripper's
                        # open/close motion is part of how it aligns to the cube).
                        target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                        # Track descent: reset the stall counter whenever we reach a new low.
                        if tcp_above < grasp_min_above - GRASP_STALL_EPS:
                            grasp_min_above = tcp_above
                            grasp_stall_ctr = 0
                        else:
                            grasp_stall_ctr += 1
                        stalled = (GRASP_GATE_STALL and grasp_min_above <= GRASP_ENGAGE_Z
                                   and grasp_stall_ctr >= grasp_stall_steps)
                        if tcp_above <= gate_z_eff or stalled:
                            grasp_phase, grasp_phase_ctr = "wait", 0
                            why = "stalled" if (stalled and tcp_above > gate_z_eff) else "height"
                            print(f"  [grasp] gate reached ({why}): {tcp_above*100:.1f} cm above table "
                                  f"(gate {gate_z_eff*100:.1f} cm @ r={tcp_r*100:.0f} cm) → wait {GRASP_WAIT_S:.1f}s")
                    elif grasp_phase == "wait":
                        # Keep running the policy (arm + gripper) past the gate
                        # before the nudge + hardcoded close.
                        target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                        grasp_phase_ctr += 1
                        if grasp_phase_ctr >= grasp_wait_steps:
                            target_qpos, info = back_nudge_joint_target(qpos, target_qpos, GRASP_NUDGE_M, z_table + GRASP_NUDGE_Z)
                            grasp_phase, grasp_phase_ctr = "nudge", 0
                            print(f"  [grasp] nudge back {info} → settle {GRASP_NUDGE_SETTLE_S:.1f}s")
                    elif grasp_phase == "nudge":
                        # Hold the corrected pose so the servos arrive before the jaws move.
                        grasp_phase_ctr += 1
                        if grasp_phase_ctr >= grasp_nudge_steps:
                            grasp_phase, grasp_phase_ctr = "close", 0
                            print("  [grasp] closing")
                    elif grasp_phase == "close":
                        target_qpos[5] = grasp_close_rad          # command full close
                        grasp_phase_ctr += 1
                        if grasp_phase_ctr >= grasp_close_steps:
                            grip_deg = float(np.rad2deg(qpos[5]))
                            if grip_deg > GRASP_EMPTY_BELOW_DEG:
                                grasp_phase, grasp_phase_ctr = "hold", 0
                                print(f"  [grasp] GRASPED (gripper stalled at {grip_deg:.1f}° > {GRASP_EMPTY_BELOW_DEG:.0f}°) → hold {GRASP_HOLD_S:.1f}s")
                            elif grasp_retries < GRASP_MAX_RETRIES:
                                grasp_retries += 1
                                cur = tcp_pos(qpos)
                                rad = float(np.hypot(cur[0], cur[1]))
                                back = (-cur[:2] / rad * GRASP_RETREAT_BACK_M) if rad > 1e-6 else np.zeros(2)
                                grasp_retreat_target = cur + np.array([back[0], back[1], GRASP_RETREAT_UP_M])
                                grasp_phase, grasp_phase_ctr = "retreat", 0
                                print(f"  [grasp] empty (gripper {grip_deg:.1f}°); back off "
                                      f"+{GRASP_RETREAT_UP_M*100:.0f}cm up/{GRASP_RETREAT_BACK_M*100:.0f}cm back, "
                                      f"retry {grasp_retries}/{GRASP_MAX_RETRIES}")
                            else:
                                grasp_phase = "hold_open"
                                grasp_result = "failed"
                                print(f"  [grasp] FAILED after {GRASP_MAX_RETRIES} retries")
                    elif grasp_phase == "retreat":
                        # IK the TCP up + back toward the base (gripper open) to get a
                        # view of where the cube went, then rerun the policy.
                        vec = grasp_retreat_target - tcp_pos(qpos)
                        if float(np.linalg.norm(vec)) <= 0.01:
                            grasp_min_above, grasp_stall_ctr = float("inf"), 0   # restart stall detector
                            grasp_phase = "approach"               # rerun policy
                            print("  [grasp] backed off → rerunning policy")
                        else:
                            step_vec = vec * min(1.0, (GRASP_RETREAT_SPEED / CONTROL_HZ) / float(np.linalg.norm(vec)))
                            dq = nudge_arm_joints(qpos, step_vec)
                            target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                            target_qpos[5] = grasp_open_rad
                    elif grasp_phase == "hold":
                        target_qpos[5] = grasp_close_rad           # keep object clamped
                        grasp_phase_ctr += 1
                        if grasp_phase_ctr >= grasp_hold_steps:
                            grasp_phase, grasp_phase_ctr = "lift", 0
                            print(f"  [grasp] lifting cube {GRASP_LIFT_M*100:.0f} cm")
                    elif grasp_phase == "lift":
                        # Ramp the lift in tiny per-step IK increments (a single
                        # 5 cm IK jump is unreliable near awkward poses; small
                        # closed-loop steps stay accurate and move smoothly).
                        dz = GRASP_LIFT_M / grasp_lift_steps
                        dq = nudge_arm_joints(qpos, np.array([0.0, 0.0, dz]))
                        target_qpos[:5] = np.clip(target_qpos[:5] + dq[:5], JOINT_LOWER[:5], JOINT_UPPER[:5])
                        target_qpos[5] = grasp_close_rad           # stay clamped while lifting
                        grasp_phase_ctr += 1
                        if grasp_phase_ctr >= grasp_lift_steps:
                            grasp_result = "success"
                    # "hold_open": freeze target_qpos as-is
                    cmd_qpos = target_qpos.copy()
                else:
                    target_qpos = np.clip(target_qpos + action * DELTA_CAP, JOINT_LOWER, JOINT_UPPER)
                    # Gripper snap-to-close latch: arm at the cube once the policy
                    # stops pushing full-close (raw action rises above the latch
                    # level) while the gripper is already closed. Latch then forces
                    # the SERVO fully closed for the rest of the episode. We send a
                    # COPY so target_qpos (and thus the policy's next observation)
                    # is left as the policy intended.
                    if (GRIPPER_SNAP_ENABLED and not gripper_latched
                            and np.rad2deg(target_qpos[5]) <= GRIPPER_SNAP_BELOW_DEG
                            and raw_action[5] > GRIPPER_LATCH_ACTION):
                        gripper_latched = True
                    cmd_qpos = target_qpos.copy()
                    if gripper_latched:
                        cmd_qpos[5] = np.deg2rad(GRIPPER_FULL_CLOSE_DEG)
                agent.set_target_qpos(torch.from_numpy(cmd_qpos))

                if viz_on:
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
                    extra = (f"  r={tcp_r*100:4.0f}cm  above_table={tcp_above*100:5.1f}cm  "
                             f"gate={gate_z_eff*100:4.1f}cm  phase={grasp_phase}") if GRASP_ENABLED else ""
                    print(f"  step {step:4d}  control={ema_control_hz:5.1f} Hz  camera={ema_camera_hz:5.1f} Hz  new_frames={new_frames}{extra}")

                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

                if grasp_result is not None:
                    print(f"  [grasp] episode {grasp_result.upper()} at step {step}")
                    break

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
            status = grasp_result if (GRASP_ENABLED and grasp_result) else "ended"
            print(f"Episode {status} ({step + 1} steps).")
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
