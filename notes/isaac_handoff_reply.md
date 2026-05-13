# Reply to `isaac_env_handoff_to_squint.md`

Squint side. Answers grounded in the actual code in this repo, not training-day memory. File:line references are all under `squint/`.

---

## TL;DR — what to change on the Isaac side

1. **Turn gravity back ON on the robot.** Your `disable_gravity=True` is the opposite of our setting. The settle behaviour you matched was coincidence, not principle. (Q1 below.)
2. **Sample distractor + goal color per episode** uniformly from the 6-color palette (distractor ≠ goal). Your fixed-blue distractor and fixed-red goal are off-distribution for the new checkpoint. (Q2, Q3.)
3. **Don't clip the integrated controller target to soft joint limits.** Our controller does not clip; yours does. (Q5.)
4. **Checkpoint path has moved**: latest is `runs/placecube_realgravity_distractor_run1/ckpt.pt`, not `placecube_flattable_woodcube_run1/`. The run name change ("realgravity") was deliberate after the gravity fix.

Everything else in your alignment table is correct.

---

## Question-by-question answers (your §8)

### §8.1 — Gravity confirmation

**The robot has gravity ON.** Not coincidentally — explicitly.

`envs/robot/so101.py:122-132`:

```python
# Wrap each controller in a dict with balance_passive_force=False so
# ManiSkill does NOT disable gravity on the robot links. The default
# (balance_passive_force=True) is a workaround for PhysX's lack of
# gravity compensation; with it on, every robot link gets
# disable_gravity=True, which makes the sim a poor match for Isaac.
controller_configs = dict(
    pd_joint_delta_pos=dict(arm=pd_joint_delta_pos, balance_passive_force=False),
    pd_joint_pos=dict(arm=pd_joint_pos, balance_passive_force=False),
    pd_joint_target_delta_pos=dict(arm=pd_joint_target_delta_pos, balance_passive_force=False),
    pd_joint_vel=dict(arm=pd_joint_vel, balance_passive_force=False),
)
```

The `qvel = 0` settle in `settle_behavior.txt` is what PD with `stiffness=1000, damping=100` looks like holding the home pose against gravity — the home configuration happens to have shoulder_lift ≈ 0 (arm horizontal but very short moment arm at the base), elbow ≈ 0, wrist roughly self-balancing, so the static torque the PD has to apply is small. Your audit's joint-axis probe confirms gravity-on dynamics qualitatively: `shoulder_lift +0.1×10` moves the tip Δz=-0.0167 m, which is the joint working a load.

**Action on Isaac side:** remove `RigidBodyPropertiesCfg(disable_gravity=True)` from `squint_robot.py`. Then re-run the settle audit — expect small non-zero drift in the first few steps, then steady state under PD. That's what we get.

### §8.2 — Distractor color sampling

Per-episode, uniform over `{0…NUM_COLORS-1} \ {goal_idx}`. See `envs/place.py:501-505`:

```python
# distractor color: uniform over {0..NUM_COLORS-1} \ {goal_idx}, vectorized.
# Trick: sample in {0..NUM_COLORS-2} then shift up by 1 wherever >= goal.
offset = torch.randint(NUM_COLORS - 1, (b,), device=self.device, dtype=torch.long)
distractor_idx = offset + (offset >= goal_idx).long()
```

`COLOR_PALETTE` (`envs/place.py:28-38`): red=0, blue=1, green=2, yellow=3, purple=4, orange=5.

Your fixed-blue distractor is fine *only* when goal=red, which happens to match your fixed deploy goal. So for the current single-color deploy it doesn't matter, but if you want to eval the policy across goals, sample the distractor too.

### §8.3 — Goal color sampling at reset

Per-episode, uniform over the 6-color palette. Or pinned via `env.reset(options={"goal_color_idx": <int or 1-D tensor>})` (`envs/place.py:486-505`). The `goal_color` one-hot in the obs reflects whatever was sampled.

**For eval that matches training distribution**, re-sample each episode. The new checkpoint was trained with full color-conditioning randomization, so always-red biases eval toward one slice of behavior.

### §8.4 — qpos noise scale

Two distinct noise sources — your `0.02` matches **only** the reset-time one. There is a separate one applied to the policy obs:

**Reset-time** (`envs/base_random_env.py:55` + `envs/place.py:515-517`):
```python
initial_qpos_noise_scale: float = 0.02    # rad, per-joint independent
# applied as: self.rest_qpos + torch.randn(b, 6) * 0.02
```
Per-joint independent gaussian. ✓ Your Isaac side matches this exactly.

**Obs-time** (`envs/place.py:589-594`, gated on `domain_randomization=True`):
```python
robot_qpos_noise_std: float = np.deg2rad(5)   # ≈ 0.0873 rad
# applied as: qpos = qpos + torch.randn_like(qpos) * 0.0873
```
This is added to `obs["agent"]["noisy_qpos"]` only — the controller target and the underlying physics qpos are clean. **The new checkpoint was trained with DR=True**, so during training the policy saw `noisy_qpos` with σ ≈ 5°. For sim-to-sim deploy this is debatable; if you want to mirror training conditions, add noise=σ to your `joint_pos_with_noise` term. If you want clean deploy obs (your current setup), the policy will see slightly cleaner state than training — usually that's an advantage, not a problem.

### §8.5 — Soft joint limit clipping

We **do not** clip the integrated target to joint limits. ManiSkill's `PDJointPosControllerConfig` with `use_target=True` stores `_target_qpos` and just adds the delta — no per-step clip. PhysX naturally can't drive past the URDF qlimit but the stored target can drift past it; subsequent negative deltas can return it to inside the limit without the PD ever applying a max-torque saturation.

Your `DeltaTargetJointPositionAction._target = torch.clamp(self._target + delta, self._joint_lo, self._joint_hi)` in `squint_actions.py` is stricter. Concretely, this matters in two regimes:

- Near a limit, an over-driving policy that intends "saturate against the wall" will have its target snap back inside the limit on your side, vs. drift outside on ours. When it later issues a return delta, your controller responds 1 step faster than ours.
- The `target_qpos` observation that goes into the policy diverges from training distribution near limits — by up to one delta-bound worth of error (≤ 0.1 rad on arm joints, ≤ 0.2 on gripper).

**Action:** remove the clamp; just integrate freely. (`squint_actions.py` `process_actions`.) Hard joint limits from the URDF are enforced by PhysX regardless.

### §8.6 — RGB normalization / color jitter

For the deployed checkpoint specifically: **`apply_overlay=False`** (`train_squint.py:87` and `train_squint.py:643-645`). No greenscreen at training time. RGB pipeline is just `F.interpolate(rgb_128, size=(16,16), mode='area')` and uint8 — your `wrist_rgb_16` matches.

**But:** the run *was* trained with `domain_randomization=True`, and that DR path applies the visual randomizations declared in `DefaultRandomizationConfig` (see `envs/base_random_env.py`): wrist-camera FOV / pos / rot noise per episode, robot color randomization, lighting variation, etc. None of these touch the RGB tensor post-render — they perturb the scene/cam at reset and the renderer takes care of the rest. So no normalization or color-jitter on the final RGB tensor at any point. Your "raw 16×16 area-downsampled uint8" obs is correct.

The RGB-channel mismatch your audit table shows (you: R=181 G=178 B=178 neutral; us: R=181 G=166 B=163 warm) is from your dome+distant lights being white and our scene-neutral table at `(0xB8, 0xAD, 0xA9)` recoloring the floor and ground too (see `envs/place.py:201` `_recolor_entities_to(self.table_scene.scene_objects, SCENE_NEUTRAL_RGB)`). Probably worth replicating in your scene — repaint the ground plane too if you have one, or live with the +12-15 on G/B.

---

## Your §9 — what we'll send

| Item | Status |
|---|---|
| 18-d ckpt | At `runs/placecube_realgravity_distractor_run1/ckpt.pt`. Path moved since your doc was written. Same head structure. |
| Gravity confirmation | Above — turn it on. |
| Bit-identical inference test (one numerical reference: state in → action out) | Happy to provide. Send your post-reset 18-d state vector + the cam RGB you read after settle, and I'll feed them through our policy and return the resulting action. The discriminator is whether your post-reset state matches ours within float32 noise — if it does, identical action means the gap is purely physics/visual, not policy. If it doesn't, we know to look at reset/cam first. |
| `black_overlay.png` | Already in this repo at `envs/black_overlay.png` (16×16 uint8 PNG). Unused for this checkpoint but available. |

---

## Two minor corrections to your alignment table (§7)

- **"Cube half size 0.0125"** — your table notes `mid 0.0125`. The DR range is `(0.018/2, 0.022/2)` (`envs/place.py:97`), so the mid is `0.010`. With DR off it spawns at exactly `0.010` (20 mm cube), not `0.0125`. Your scene uses `(0.020, 0.020, 0.020)` ✓.
- **"Density 200"** — your scene comment in `squint_scene.py` says density 200, but you actually use 700 in code (correct: matches `item_density_range = (700, 700)` in `envs/place.py:107`). Just update the comment.

---

## Open follow-ups (from us to you)

1. After you flip gravity, send the new settle audit + obs dump. We'd like to verify the post-reset state diverges from ours by no more than what gravity-on physics integration should produce in 1 control step (sub-mm tip drift).
2. The cam world-pos discrepancy your audit shows (your `(+0.288, +0.007, +0.109)` vs our `(+0.286, -0.006, +0.123)`) — y differs by ~13 mm and z by ~14 mm. Possible causes worth probing: (a) `gripper` body in the converted USD has a different local-origin than ManiSkill's `gripper_link`, (b) you're snapshotting the cam pose 1 control-step late because of `force_render=False` in `_update_wrist_cam_pose`. We can sanity-check by sending you the exact `gripper_link.pose` we read at home and you can compare to your `gripper` body pose pre-cam-update.

---

# Part 2 — Updates and training-side reference

This second half captures (a) changes since the inference reply above was written, and (b) the full training-implementation reference needed if you want to **train from scratch** on Isaac, not just deploy our checkpoint.

## Inference-side corrections (supersede Part 1 where they conflict)

- **Action delta bounds halved.** Arm joints are now **±0.05 rad/step** (was ±0.1). Gripper still ±0.2. See [envs/robot/so101.py:97-100](envs/robot/so101.py#L97-L100). Update your `DeltaTargetJointPositionActionCfg.bounds` to `[0.05, 0.05, 0.05, 0.05, 0.05, 0.2]` — otherwise you command motions 2× faster than training.

- **Wrist camera FOV bug on your side.** [squint_scene.py](sim/eval2/envs/squint_native/squint_scene.py) declares `WRIST_CAM_FOV_RAD = math.radians(71)` but **never uses it**. The actual cam is built with `PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955)` → FOV ≈ **47.2°**. We render at 71°. Fix: `focal_length = horizontal_aperture / (2 * tan(71° / 2)) ≈ 14.69`, or keep `focal_length=24` and set `horizontal_aperture ≈ 34.3`. Verify by reading FOV back from `cam.cfg.spawn` after build. This is probably ~1.5× the impact of the gravity bug — same priority.

- **Episode length is now 75 control steps, not 50.** `@register_env("SO101PlaceCube-v1", max_episode_steps=75)` ([envs/place.py:935](envs/place.py#L935)). 7.5 s at 10 Hz. PlaceCan stays at 50.

- **Goal site z is OFFSET above the bin** (in case you were placing your visualization sphere at item-half height). [envs/place.py:794-795](envs/place.py#L794-L795):
  ```python
  goal_xyz[:, 2] = self.bin_thickness + self.item_half_sizes + self.target_z_above_floor
  ```
  with `target_z_above_floor = 2 * bowl_half_z + 0.04` when `use_real_bowl=True` ([place.py:449](envs/place.py#L449)). So for our bowl (half_z ≈ 0.0265): goal z ≈ `0 + 0.01 + 0.053 + 0.04 = 0.103 m` above the table floor. That's `4 cm above the bowl rim`, not 4 cm above the cube. If your green sphere looks too high, that's why.

- **The bin is now a real bowl mesh** (`envs/meshes/bowl.obj`) via CoACD decomposition, density 500, dynamic, friction 0.5/0.5/0. `use_real_bowl=True` is the new default. The 5-box tray is still selectable via `use_real_bowl=False`.

- **n_distractors is configurable** (0-4, default 1). The current checkpoint was trained with 1 distractor (face-to-face with target, distinct palette color).

## Sim/scene additions you'll want for parity

- **Table friction is now per-env randomized** (was hardcoded). [envs/place.py:146](envs/place.py#L146): `table_friction_range = (0.05, 0.4)`. Mid = 0.225 for DR-off. The table is a kinematic box, friction sampled per env.

- **Cube friction range widened**: `(0.05, 0.6)` instead of `(0.1, 0.5)`. [envs/place.py:143](envs/place.py#L143).

- **Cube density range widened**: `(600, 1000)` kg/m³ instead of `(700, 700)`. [envs/place.py:144](envs/place.py#L144).

- **Distractor cubes** use the same per-env friction/density as the target cube (one scalar sampled per env, shared across all cubes in that env).

- **Gripper material** on the robot URDF is **`(2.0, 2.0, 0.0)`** with `patch_radius=0.1`, `min_patch_radius=0.1` — applied to `gripper_link`, `moving_jaw_so101_v1_link`, `finger1_tip`, `finger2_tip`. ([envs/robot/so101.py:28-46](envs/robot/so101.py#L28-L46)). Maps to Isaac body names `gripper`, `jaw`, `finger1_tip`, `finger2_tip`. **PhysX `torsional_patch_radius` and `min_torsional_patch_radius` need to be set to 0.1 on these bodies.**

- **PhysX friction combine mode is MIN** in SAPIEN. Isaac Lab can default to multiply or average — set `friction_combine_mode="min"` and `restitution_combine_mode="min"` on all PhysicsMaterials.

(Full per-actor friction/density/contact-offset table was sent separately — request it if you don't have it.)

---

## Training implementation reference

If you want to retrain on Isaac, the items below define what the algorithm and signals actually are. Everything here was extracted from the current `master` of this repo.

### 1. The algorithm is SAC + distributional C51 — NOT PPO

[train_squint.py](train_squint.py) is **off-policy SAC** with a **distributional C51 critic**. Replay buffer, soft target update via `lerp_(tau)`, entropy autotuning, two-Q ensemble. If your Isaac training stack assumed PPO, the whole loop is different.

Concretely:

- Replay buffer size: 1,000,000
- Batch size: 512
- Updates per env-step block: 256
- Learning starts at: 5,000 timesteps
- `policy_lr = q_lr = alpha_lr = 3e-4`
- `policy_frequency = 4` (actor updated every 4 critic updates)
- `target_network_frequency = 1`, `tau = 0.01`
- `gamma = 0.9` (**unusually low** — ~10-step credit horizon. If you use rl_games/rsl_rl defaults of 0.99, behavior diverges.)
- `num_q = 2` (CDQ-style ensemble, **mean-reduced** for target — not min!)
- C51: `num_atoms = 101`, `v_min = -20`, `v_max = 20`
- Entropy autotune: `target_entropy = -n_act = -6`, log_alpha initialized to 0 (alpha=1)
- Optimizer: Adam, no weight decay
- `num_envs = 2048`, `total_timesteps = 1.5e6`

The critic in our ckpt is **not portable to a scalar V(s) head** — it's `Q(s, a)` with 101-atom categorical distribution. If you migrate to PPO, you'll need a fresh value head; if you stay with SAC, you can reuse our critic structure.

### 2. Observation structure (training-time)

After `FlattenRGBDObservationWrapper(rgb=True, depth=False, state=True)` + `DownsampleObsWrapper(target_size=16)` the policy sees a dict with **exactly two keys**:

- `obs["rgb"]`: `(N, 16, 16, 3)` uint8, HWC. Sim renders 128×128, downsampled with `F.interpolate(mode='area')` to 16×16, cast back to uint8. **Normalization happens inside the CNN**: `x = x/255 - 0.5`. Don't pre-normalize on your side.

- `obs["state"]`: `(N, 18)` float32 = `[noisy_qpos(6), controller_target_qpos(6), goal_color_one_hot(6)]`. Order is dict-iteration order from `_get_obs_agent` ([envs/place.py:791-807](envs/place.py#L791-L807)); `flatten_state_dict` is key-order dependent so this order is load-bearing.

There is **no privileged observation, no asymmetric critic.** `_get_obs_extra` is gated on `obs_mode_struct.state`, which is False when `obs_mode="rgb"` (the train default). So `item_pose`, `bin_pose`, `tcp_pose`, frictions, densities — none of it reaches actor OR critic. Don't bother plumbing them.

### 3. Network architecture (exact)

CNN encoder ([train_squint.py:278-316](train_squint.py#L278-L316)) — 16×16 input:
```
Conv2d(3, 32, k=4, s=2)  → ReLU   # → (B, 32, 7, 7)
Conv2d(32, 64, k=4, s=1) → ReLU   # → (B, 64, 4, 4)
Flatten                            # → (B, 1024)
```
No padding, no pooling, no BatchNorm. Output is **1024-d** raw features. ~34k params.

Projection head (`Projection` class, [train_squint.py:319-331](train_squint.py#L319-L331)):
```
rgb_proj   = Linear(1024, 50)  → LayerNorm  → Tanh
state_proj = Linear(18,   256) → LayerNorm  → ReLU
concat → (B, 306)
```
Actor and critic each instantiate their own Projection. Only the CNN encoder is shared (and its gradients come from the critic optimizer).

Actor ([train_squint.py:334-387](train_squint.py#L334-L387)):
```
proj (306-d)
→ Linear(306, 256) → LayerNorm → ReLU
→ Linear(256, 256) → LayerNorm → ReLU
→ Linear(256, 256) → LayerNorm → ReLU
├─→ fc_mean   = Linear(256, 6)
└─→ fc_logstd = Linear(256, 6) → tanh → remap to [-5, +2]
sample: Normal(mean, exp(log_std)).rsample()
→ tanh → action_scale * tanh + action_bias  ∈ [low, high]
```
State-dependent log_std (NOT a learned scalar). For inference: deterministic — just `tanh(mean) * scale + bias`. `fc_logstd` is in the checkpoint but unused at deploy.

Critic ([train_squint.py:390-505](train_squint.py#L390-L505)):
- Own Projection (1024→50 ‖ 18→256 = 306-d)
- Concat with 6-d action → 312-d
- 2 separate Q-nets, each: `Linear(312, 512) → LN → ReLU → Linear(512, 512) → LN → ReLU → Linear(512, 512) → LN → ReLU → Linear(512, 101)`
- Q-value = `softmax(logits) · linspace(-20, 20, 101)`
- Stacked via `tensordict.from_modules`, dispatched with `torch.vmap`
- Target critic Polyak-updated with `tau=0.01` every gradient step

Init: **orthogonal**, gain=1 for Linear, gain=√2 (`calculate_gain('relu')`) for Conv2d. Biases zero. LayerNorm at PyTorch default.

### 4. Reward function — boolean OVERWRITE branching

[envs/place.py:875-927](envs/place.py#L875-L927). What the algorithm sees is `compute_normalized_dense_reward = compute_dense_reward / 9`. ManiSkill's default `reward_mode="normalized_dense"` is what `env.step()` returns. PPO/SAC consumes the divided-by-9 value.

```python
reward = reaching_reward                                       # [0, 2]
if is_item_grasped:    reward = 3 + place_reward              # [3, 5]  OVERWRITE
if is_item_above_bin:  reward = 4 + place_reward
                              + is_item_dropped
                              + gripper_openness
                              + static_robot_reward            # [4, 9]  OVERWRITE
if success:            reward = 9                              # HARD SET
reward -= 6 * robot_touching_table
reward -= 3 * robot_touching_bin
reward -= 1 * (not item_lifted)
```

Component formulas:
- `reaching_reward = 2 * (1 - tanh(5 * ||tcp_pos - item_pos||))`
- `place_reward = place_reward_final + place_reward_z` (each in [0, 1])
  - `place_reward_final = 1 - tanh(5 * ||goal_xyz - item_pos||)`
  - `place_reward_z` uses a **far-vs-close** branch based on `xy_dist <= bin_radius`. Far: target z is `bin_top + 2*bin_half_z + 0.03` (hover altitude). Close: target z is `bin_thickness + item_half_size`. Then `1 - tanh(10 * |dz|)` (scale 10, steeper than xy).
- `gripper_openness = (gripper_qpos - qmin) / (qmax - qmin)` ∈ [0, 1]
- `static_robot_reward = 1 - tanh(10 * ||qvel[:, :-1]||)` — **excludes the gripper joint velocity**
- `is_item_dropped = (~robot_touching_item).float()`

**Critical gotcha for Isaac**: `RewardTermCfg` composes additively (`Σ w_i * f_i`), but our reward uses boolean **overwrites** at three places. Don't decompose into multiple terms or you'll silently double-count. **Implement as one custom Python function bound to a single `RewardTermCfg` with `weight=1/9`.**

### 5. Contact predicates

All in [envs/robot/so101.py:154-193](envs/robot/so101.py#L154-L193). Map to Isaac body names: `gripper_link → gripper`, `moving_jaw_so101_v1_link → jaw`. The tip links (`finger1_tip`, `finger2_tip`) are massless TCP frames with **no collision geometry** — **don't wire contact sensors to them**.

**`is_touching(obj)`**: `||F||₂ ≥ 0.01 N` on EITHER `gripper` or `jaw` (logical OR). Force = per-pair filtered contact force in world frame.

**`is_grasping(obj, min_force=0.5, max_angle=110)`**: BOTH jaws have `||F||₂ ≥ 0.5 N` AND `angle(F, inward_dir) ≤ 110°`. Inward direction is column-1 of the body rotation matrix in world frame: `link.pose.to_transformation_matrix()[..., :3, 1]`. For `gripper` use `+y` (column-1 directly), for `jaw` use `-y` (sign-flip column-1). **This +y convention is URDF-specific** — after URDF→USD conversion, the local axis may flip or permute. Verify with a debug arrow viz on Isaac side. Wrong axis → angle is computed against the wrong vector → `is_grasping` returns ~always False → training collapses silently.

**`is_static(threshold=0.15)`**: `max(|qvel[:, :-1]|) ≤ 0.15` rad/s. **Excludes the gripper joint** (last index). Critical: if Isaac uses `||qvel||₂` over all joints, the gripper opening to release the cube will keep `is_robot_static=False` and `success` will never fire.

SAPIEN API: `scene.get_pairwise_contact_forces(linkA, linkB)` returns `(N_envs, 3)` per-pair world-frame force, summed over all contact points between the two bodies. The Isaac analog is `ContactSensorCfg` with `filter_prim_paths_expr=[<other prim>]` — **NOT** unfiltered `ArticulationView.get_contact_forces()`, which aggregates over all contacts and would leak table contact into `is_grasping(item)`.

Units: SAPIEN returns `impulse / dt`. If Isaac's sensor returns raw impulses, divide by dt or your 0.5 N / 0.01 N thresholds are 100× too high.

### 6. Termination / success criterion

```python
success = is_item_above_bin & (~robot_touching_item) & is_robot_static & (~robot_touching_bin)
```

- `is_item_above_bin = (|item_x - bin_x| < bin_half_x) & (|item_y - bin_y| < bin_half_y)`. **L∞ box check (rectangle) aligned to world axes.** NOT a radial L2 check, NOT bin-local. The bin's random yaw is **ignored** for the success check. z is NOT checked.
- `is_robot_static` — see above.
- `robot_touching_item` / `robot_touching_bin` — `is_touching` on the respective actors.

`terminated = success`; `truncated = elapsed_steps >= 75`. No other termination — no joint-limit failure, no fall-off-table, no self-collision failure.

The **bin pose is read live every step** (`self.bin.pose.p`). The bowl is a dynamic actor and can move when the robot bumps it. Don't cache the spawn pose; the success polygon translates with the bowl.

Bowl half-sizes are read from the mesh AABB at load time. Fallback values if the AABB read fails: `(0.074, 0.0745, 0.0265)` ([envs/place.py:436](envs/place.py#L436)).

### 7. Training-loop quirks

- `ignore_terminations=True` (because `partial_reset=False` is the default). The vec env does **not** auto-reset on `success=True` — episodes always run to truncation.
- `bootstrap_at_done='always'` (the default). `dones` is **forced all-False** for value targets. Training never cuts the value bootstrap on done or truncation. Whatever PPO/SAC infra you use on Isaac must match this — otherwise the TD targets diverge at episode boundaries.

### 8. What's training-only and can be ignored if you only want to deploy

| Component | Need for inference? |
|---|---|
| CNN encoder | YES |
| Projection (rgb_proj + state_proj) | YES |
| Actor trunk + fc_mean | YES |
| Actor `fc_logstd` | NO at inference, but YES in ckpt (strict load) |
| `action_scale`, `action_bias` buffers | YES |
| Critic (Projection + Q-ensemble + atoms) | NO |
| `log_alpha`, `critic_target` | NO |
| Contact predicates | NO |
| Reward function | NO |
| `_get_obs_extra` privileged keys | NO (never emitted at train time anyway) |

Deploy-time checkpoint surgery: only `ckpt['encoder']` and `ckpt['actor']` are loaded. `ckpt['critic']`, `ckpt['log_alpha']`, `ckpt['global_step']` are ignored ([train_squint.py:529-542](train_squint.py#L529-L542)).

### 9. "Silently wrong on Isaac side" checklist for training

- [ ] Contact sensors are on `gripper` and `jaw` bodies, NOT `finger1_tip` / `finger2_tip` (no collision on the tips).
- [ ] Contact force is **filtered per-pair**, not net per-body.
- [ ] Contact force units are Newtons (impulse / dt), not raw impulses.
- [ ] Inward-direction sign is verified: `gripper` local +y points INTO the gap; `jaw` local +y points OUT (use -y). After URDF→USD this may flip. Debug-viz with an arrow.
- [ ] `is_grasping` angle threshold is **110°**, not 90°. Critical, our value is intentionally lenient.
- [ ] `is_robot_static` uses `qvel[:, :-1]` (excludes gripper) with L∞ max, threshold 0.15 rad/s.
- [ ] Success uses **independent box check** `|dx|<hx & |dy|<hy`, not radial.
- [ ] Bin pose tracked live (dynamic bowl).
- [ ] No z check in success.
- [ ] `RewardTermCfg` is ONE custom function, not decomposed (overwrite semantics).
- [ ] Reward includes the `/9` normalization (or you scale your value bounds accordingly).
- [ ] `gamma=0.9` if you want our credit horizon. rl_games/rsl_rl default 0.99 gives different behavior.
- [ ] `ignore_terminations=True` and `bootstrap_at_done='always'` in your rollout/replay collector.
- [ ] State dict ordering: `[noisy_qpos, controller_target_qpos, goal_color_one_hot]` (load-bearing).

---

*Anything in this doc that references training-time setup is keyed off the `placecube_realgravity_distractor_run1` checkpoint. If you're deploying an older 12-d ckpt, the goal-color answers don't apply.*
