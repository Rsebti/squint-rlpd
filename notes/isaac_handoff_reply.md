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

*Anything in this doc that references training-time setup is keyed off the `placecube_realgravity_distractor_run1` checkpoint. If you're deploying an older 12-d ckpt, the goal-color answers don't apply.*
