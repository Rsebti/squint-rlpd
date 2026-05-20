# RLPD on top of Squint

This file documents `train_rlpd.py` — our implementation of **RLPD** (Ball et al.,
*Efficient Online Reinforcement Learning with Offline Data*, ICML 2023) layered
on top of the Squint repo. We keep Squint's ManiSkill3 sim, SO-101 robot,
parallel-env training pipeline, and `torch.compile`/`cudagraphs` optimisations,
and replace only the SAC core with RLPD.

The plan: use RLPD to learn a **robust visual grasp** policy in sim; the place
phase will be solved separately with a scripted FK/IK controller.

## What changed vs `train_squint.py`

| Aspect | Squint (SAC + C51) | RLPD |
|---|---|---|
| Q head | Distributional C51 (101 atoms) | Scalar Q |
| Critic loss | Cross-entropy against the projected categorical target | MSE TD(0) |
| Q-ensemble size | `num_q=2` | `num_q=10` (default) |
| Target aggregation | Mean over all 2 Q-nets | Min over a random subset of size `subset_size=2` (REDQ-style) |
| Actor aggregation | Mean over the full ensemble | Min over the full ensemble |
| Offline data | None | Optional, loaded into a second `ReplayBuffer` |
| Batch | 100% online | 50% online / 50% offline (symmetric sampling) |
| LayerNorm in critic | Already yes | Still yes (RLPD requires it) |

The CNN encoder, projection, Actor architecture, entropy autotune, target
network EMA, partial reset, and the env wrappers (downsample, jitter, video
recording) are all reused from Squint unchanged.

## Files

- `train_rlpd.py` — main training script (forked from `train_squint.py`, ~5
  surgical edits documented inline).
- `rlpd_utils.py` — offline buffer loader and a CLI helper that builds a
  synthetic offline bundle by rolling out a Squint checkpoint.
- `parse_check_rlpd.py` — pure-syntax check (runs anywhere, no GPU/maniskill
  needed). Use this on the laptop before pushing.
- `test_rlpd_smoke.py` — exercises Critic forward / SAC target / offline bundle
  roundtrip on CPU. Run on the 5090 (needs torch + tensordict but not
  maniskill) before the first long run.

## Running

### 1. Sanity check the file before launching

```bash
# Laptop (no GPU required):
python parse_check_rlpd.py

# 5090 (needs torch + tensordict in the squint env):
python test_rlpd_smoke.py
```

Both must pass before you launch a real run.

### 2. Pure-online (no offline data yet)

```bash
python train_rlpd.py \
    --env_id=SO101LiftCube-v1 \
    --pick_only_reward \
    --num_envs=256 \
    --num_q=10 \
    --subset_size=2 \
    --track \
    --wandb_entity=YOUR_WANDB_USERNAME
```

With `--offline_path` unset, RLPD degenerates to REDQ (SAC + LayerNorm + big
ensemble + sample-then-min). This is the baseline to compare against once
demos are wired in.

### 3. Synthetic offline data from a trained Squint checkpoint

While real teleop demos are being recorded, you can bootstrap a small offline
buffer by rolling out a partially-trained Squint policy:

```bash
# First: train Squint until it gets ~40-60% success (~5-10 min on the 5090).
python train_squint.py --env_id=SO101LiftCube-v1 --pick_only_reward

# Then: roll it out and keep only the successful episodes.
python rlpd_utils.py from_ckpt \
    --ckpt=runs/<run_name>/ckpt_best.pt \
    --env_id=SO101LiftCube-v1 \
    --out=offline_bundles/liftcube_synth.pt \
    --num_episodes=200 \
    --pick_only_reward

# Now train RLPD with the synthetic offline data.
python train_rlpd.py \
    --env_id=SO101LiftCube-v1 \
    --pick_only_reward \
    --num_envs=256 \
    --offline_path=offline_bundles/liftcube_synth.pt \
    --offline_ratio=0.5 \
    --track
```

### 4. Real teleop demos — `Rsebti/projet3_demos_v1` (39 episodes, ~12k frames)

The LeRobot HF loader is wired up in
`rlpd_utils.py::_load_lerobot_dataset`. Expected source schema:

| Field | dtype | Shape | Notes |
|---|---|---|---|
| `action` | float32 | (6,) | absolute joint TARGETS, degrees (arm) + degrees (gripper, assumed) |
| `observation.state` | float32 | (6,) | current joint positions, same units |
| `observation.images.wrist` | uint8 video | (480, 640, 3) | 30 FPS, AV1-encoded |
| `episode_index`, `frame_index`, `timestamp` | int64 / float32 | scalar | bookkeeping |

Conversions applied:

| Sim field (58D state slice) | Source | Notes |
|---|---|---|
| `noisy_qpos[0:6]` | `observation.state` → rad | per-joint deg2rad |
| `controller_target[6:12]` | `action` → rad | the running PD target ≈ next-step target |
| `goal_color[12:18]` | one-hot[0] = 1 | demos don't store the target color → default; retrain on a multi-goal dataset for Eval-2 |
| `bowl_xyz_robot_frame[18:21]` | zeros | no bowl in grasp-only demos |
| `qvel[21:27]` | finite-diff (qpos[t+1] − qpos[t]) × 30 | uses recorded 30 FPS |
| `is_item_grasped[27]` | `gripper_qpos > grasp_threshold_rad` | tune via the printed gripper qpos range |
| `tcp_pose[42:49]` | FK from qpos (`so101_fk.py::tcp_pos`) | xyz from FK, quat = identity (Squint's Sim2RealEnv keeps the privileged quat unobservable at deploy too) |
| `item_pose`, `bin_pose`, `tcp_to_*`, `item_to_bin_pos` | zeros | privileged sim-only info; same handling as Squint's deploy path |
| `action_normalised[6]` | `clip((act_rad − state_rad) / step_range, -1, 1)` | arm step = ±0.05 rad, gripper = ±0.20 rad |
| `rgb[80, 144, 3]` | area-resize 480×640 → 80×144, uint8 | matches `DownsampleObsWrapper` |

**Two-step workflow:**

```bash
# (a) Decode the HF dataset once and stash a CPU .pt bundle. Run on the 5090
# the first time (decord/torchcodec install lives there); the .pt is then
# copied to wherever you train. ~50 MB for the v1 dataset at 80×144.
python rlpd_utils.py from_lerobot \
    --repo_id=Rsebti/projet3_demos_v1 \
    --out=offline_bundles/projet3_v1_80x144.pt \
    --image_height=80 --image_width=144

# (b) Train RLPD against the bundle.
python train_rlpd.py \
    --env_id=SO101LiftCube-v1 \
    --pick_only_reward \
    --no-env_domain_randomization \
    --offline_path=offline_bundles/projet3_v1_80x144.pt \
    --offline_ratio=0.5 \
    --num_envs=256 \
    --track
```

You can also point `--offline_path` directly at the HF id
(`Rsebti/projet3_demos_v1`); the loader runs at training startup. The two-step
form is just faster for iterating on hyperparameters because the video decode
only happens once.

**Verification before training.** The loader prints summary stats at the end
of conversion:
* `gripper qpos` — `min/median/max` of the 6th joint after deg→rad.
  Should look like radians (~0 fully open, ~1.5 fully closed for SO-101's
  gripper joint range). If the values look like degrees instead (e.g.
  median ≈ 70-90) pass `--no-gripper_in_degrees`.
* `pre-norm delta` — the raw per-step delta in rad. Anything more than
  ~5% of arm deltas clipping at ±0.05 means either the recorded FPS is
  different from 30 Hz (rare) or the unit assumption is wrong.
* `fraction grasped` — the rate at which `is_item_grasped=1` fires given
  `grasp_threshold_rad`. Tune the threshold so it's ~30-60% (rare at the
  start, common late in the episode). Calibrate by eyeballing the
  printed gripper-qpos distribution.

**Known caveats of the v1 loader:**
* The `goal_color` slice is fixed to one-hot index 0. The grasp-only run is
  fine with this; for Eval-2 (color-conditioned grasp) you need either a
  multi-goal recording or to add the goal-color metadata to the demo schema
  in v2.
* The quaternion components of `tcp_pose` are left as the identity quat —
  the encoder is RGB-led and Squint's deploy path doesn't measure tcp
  orientation either, so this is not a regression. If you observe Q value
  divergence specifically when offline-batch ratio is high, suspect the
  zero-padded privileged slices and either ramp `offline_ratio` slowly or
  switch to sim-replay state reconstruction (see TODO in
  `_load_lerobot_dataset`).
* Every demo is labelled as a success (sparse reward = +1 at the last
  frame). If the recording session has any unintended bad demos, filter
  episode indices with `--max_episodes` or `keep_episode_indices=...` in
  the Python API.

## Hyperparameter defaults

`num_q=10`, `subset_size=2`, `offline_ratio=0.5` match the canonical RLPD
paper. The other SAC-side knobs (`policy_lr`, `q_lr`, `gamma`, `tau`,
`num_updates`, etc.) come from Squint and are left untouched — Squint already
uses an aggressive update-to-data ratio (`num_updates=256` per parallel-env
step), which RLPD is explicitly designed to benefit from.

If RLPD overestimates Q-values catastrophically (you'll see q_max blow up to
> 100 in wandb), the first knob to turn is `subset_size`: bumping it from 2
to e.g. 5 makes the target less conservative; dropping it to 1 makes it more
conservative. The paper sticks to 2.

## Why no BC regulariser?

RLPD's design point is "no behavioral cloning loss", just symmetric sampling
+ LayerNorm + big ensemble. Adding a BC term would be a divergence from the
paper; if you want a BC-regularised offline-to-online algorithm, look at
RLPD-BC, IQL+SAC, or TD3+BC in the literature — but evaluate the vanilla
version first so you know what the BC term actually buys you.

## Known limitations

- The LeRobot → ManiSkill demo loader is intentionally a stub (see the
  `_load_lerobot_dataset` docstring). The exact mapping depends on the demo
  schema the team produces, so we wire it in once the first real demo lands.
- `update_main` does a fresh `torch.randint` for `subset_idx` every grad
  step. Inside a `cudagraphs`-captured function this should be fine (random
  ops are graph-friendly with newer torch), but if you see "graph requires
  static input" errors, move the random index sampling to the outer Python
  loop and pass `subset_idx` in as a TensorDict field.
- The synthetic demo collector in `rlpd_utils.collect_synthetic_offline_from_ckpt`
  uses an eval (deterministic) actor — that's intentional, you want clean
  rollouts in the offline buffer, not exploration noise. If you want a mix,
  sample stochastic actions via `actor.get_action(...)[0]` instead of
  `actor.get_eval_action(...)`.
