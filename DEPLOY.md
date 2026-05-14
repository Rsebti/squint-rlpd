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
state (18) ─────────────────────────────┘
```

- **CNNEncoder**: 2 conv layers, outputs a 1024-d image feature.
- **Actor**: projects image+state, 3 hidden layers (256), outputs a 6-d action in `[-1, 1]`.
- Only `encoder` + `actor` weights from `ckpt.pt` are used (critic is training-only).

## Input (what the policy consumes each step)

| Part | Shape | Content |
|---|---|---|
| `rgb` | `(1,16,16,3)` uint8 | wrist camera, center-cropped, resized 128→16 |
| `state` | `(1,18)` float | `[qpos(6), controller_target_qpos(6), goal_onehot(6)]` |

- `qpos` — measured joint angles (radians), order: pan, lift, elbow, wrist_flex, wrist_roll, gripper.
- `controller_target_qpos` — the running target the controller accumulates (starts at rest qpos).
- `goal_onehot` — which cube color to pick: `0 red  1 blue  2 green  3 yellow  4 purple  5 orange`.

## Output (what to do with the action)

The actor returns a 6-d vector in `[-1, 1]`. To turn it into a robot command:

```
action      = clip(action * action_scale, -1, 1)      # action_scale = safety multiplier (0.15)
delta       = action * [0.1,0.1,0.1,0.1,0.1, 0.2]     # per-joint rad/step caps (gripper = 0.2)
target_qpos = clip(target_qpos + delta, joint_limits) # accumulate onto running target
robot.set_target_qpos(target_qpos)                    # driver converts rad→deg + gripper map
```

`infer.py` does all of this. The robot runs at **10 Hz**.

## Run

```bash
python infer.py --checkpoint ckpt.pt --goal_color 0 --action_scale 0.15
```

- `--goal_color` — cube color index (0–5)
- `--action_scale` — start at `0.1` for the first runs (smaller/slower moves), raise once it looks safe
- `--episode_steps` — control steps per episode (default 150)

Press `Enter` to start each episode; `Ctrl+C` quits and returns the arm to rest.

## Safety

- Start with `--action_scale 0.1`, stay near the e-stop.
- The policy is sim-trained — transfer is imperfect.
- On `Ctrl+C` or exit the arm returns to its rest pose; if it crashes first, power-cycle.
