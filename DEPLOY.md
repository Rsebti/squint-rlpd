# Deploying the PlaceCube Policy on the SO101 Robot

`infer.py` is a **fully standalone** inference script — the policy network, the
obs/action contract, and the robot driver are all in that one file. 

## Setup (Mac)

```bash
conda create -n squint python=3.10 -y && conda activate squint
pip install torch torchvision numpy opencv-python "lerobot[feetech]==0.4.3"
```

Then edit the marked block at the top of `infer.py`:
- `ROBOT_PORT` → your Mac's serial port (`ls /dev/cu.*`, e.g. `/dev/cu.usbmodem14401`)
- `CAMERA_INDEX` → webcam index (`0`, `1`, or `2`)
- `CALIBRATION_ID` / `CALIBRATION_DIR` → your arm's calibration `.json`

## Architecture

```
camera (16x16x3) ─► CNNEncoder ─► 1024 ─┐
                                        ├─► Actor (MLP) ─► action (6)
state (21) ─────────────────────────────┘
```

- **CNNEncoder**: 2 conv layers, outputs a 1024-d image feature.
- **Actor**: projects image+state, 3 hidden layers (256), outputs a 6-d action in `[-1, 1]`.
- Only `encoder` + `actor` weights from `ckpt.pt` are used (critic is training-only).

## Input (what the policy consumes each step)

| Part | Shape | Content |
|---|---|---|
| `rgb` | `(1,16,16,3)` uint8 | wrist camera, center-cropped, resized 128→16 |
| `state` | `(1,21)` float | `[qpos(6), controller_target_qpos(6), goal_onehot(6), bowl_xyz_robot_frame(3)]` |

- `qpos` — measured joint angles (radians), order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
- `controller_target_qpos` — the running target the controller accumulates (starts at rest qpos).
- `goal_onehot` — which cube color to pick: `0 red  1 blue  2 green  3 yellow  4 purple  5 orange`.
- `bowl_xyz_robot_frame` — bowl centre `(x, y, z)` in metres, expressed in the robot base frame. Measure once per deploy with a tape (≈ `[0.25, 0.10, 0.0]` for the SO101 with bowl in the typical workspace), pass via `--bowl_xyz`. **Old 18-d checkpoints**: `infer.py` falls back to a 18-d state automatically when you omit `--bowl_xyz`.

## Output (what to do with the action)

The actor returns a 6-d vector in `[-1, 1]`. To turn it into a robot command:

```
action      = clip(action * action_scale, -1, 1)             # action_scale = safety multiplier
delta       = action * [0.0333]*5 + [0.0667]                 # per-joint rad/step caps @ 30 Hz
target_qpos = clip(target_qpos + delta, joint_limits)        # accumulate onto running target
robot.set_target_qpos(target_qpos)                           # driver converts rad→deg + gripper map
```

`infer.py` does all of this. The robot runs at **30 Hz** (the calibrated control rate matching the sim's `dt_ctrl = 33.3 ms`).

## Run

```bash
python infer.py --checkpoint ckpt.pt --goal_color 0 --action_scale 0.15 --bowl_xyz 0.25 0.10 0.00
```

- `--goal_color` — cube color index (0–5)
- `--action_scale` — start at `0.1` for the first runs (smaller/slower moves), raise once it looks safe
- `--bowl_xyz` — bowl centre in robot frame, metres. Omit only when running an old 18-d checkpoint.
- `--episode_steps` — control steps per episode (default 150 ≈ 5 s at 30 Hz; use 100 for tighter cycles)

Press `Enter` to start each episode; `Ctrl+C` quits and returns the arm to rest.

## Safety

- Start with `--action_scale 0.1`, stay near the e-stop.
- The policy is sim-trained — transfer is imperfect.
- On `Ctrl+C` or exit the arm returns to its rest pose; if it crashes first, power-cycle.
