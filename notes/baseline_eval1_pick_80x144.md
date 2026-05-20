# Baseline — 1st working eval1 pick-only policy (`eval1_pick_80x144`)

**Date trained:** 2026-05-19 → 2026-05-20
**Wandb run:** `fedecominelli04_robot/maniskill-so101/8utndjvd`
**Checkpoint in repo:** `runs/eval1_pick_80x144/ckpt.pt` (embedded `global_step = 800,768`)
**Status on real robot:** works well, not perfect — first eval1 policy with a usable real-world pick.

This is the **base point** for further improvements. Any future eval1 pick-only run should start from this exact config and change *one knob at a time* so regressions are attributable.

---

## Reconstructable launch command

Fresh Brev RTX Pro 6000 VM, one-line bootstrap (env vars **after** the pipe so they reach `bash`):

```bash
curl -fsSL https://raw.githubusercontent.com/fedecomi04/squint/master/scripts/brev_bootstrap_rtx6000.sh | \
  LAUNCHER=scripts/brev_run_ablation.sh \
  PICK_ONLY=true \
  SIDE_APPROACH=false \
  N_DISTRACTORS=0 \
  SIM_FREQ=100 \
  LATENCY=off \
  EP_STEPS=100 \
  IMAGE_HEIGHT=80 IMAGE_WIDTH=144 \
  RENDER_HEIGHT=360 RENDER_WIDTH=640 \
  NUM_ENVS=1024 \
  BUFFER_SIZE=500000 \
  BATCH_SIZE=512 \
  TOTAL_TIMESTEPS=10000000 \
  EXP_NAME=eval1_pick_80x144 \
  bash
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now baked into
`scripts/brev_run_ablation.sh`, so the OOM-by-fragmentation that bit two
earlier runs no longer needs to be set manually.

## Full training config (from wandb)

| group | key | value | notes |
|---|---|---|---|
| **task** | `env_id` | `SO101PlaceCube-v1` | |
| | `n_distractors` | 0 | eval1 = no distractors |
| | `pick_only_reward` | True | reach → grasp → close hard → stable for 1 s |
| | `pick_side_approach` | False | NOT using the fixed-finger-first curriculum |
| | `eval_max_episode_steps` | 100 | 10 s @ 10 Hz |
| **physics** | `sim_freq` | **100 Hz** | per [project_sim2real_ablation_2026-05-20](../) — 100 Hz beat 300 Hz |
| | `control_freq` | 10 Hz | |
| | `camera_lag_substeps_min/max` | 0 / 0 | **no latency** (also the ablation winner) |
| **vision** | `image_height` × `image_width` | **80 × 144** | 16:9, matches calibrated 1920×1080 real cam |
| | `render_height` × `render_width` | 360 × 640 | full-res render → area-pool to 80×144 |
| **memory** | `num_envs` | 1024 | 2048 + 1 M OOMs in HSV jitter, see [project_brev_memory_knobs](../) |
| | `buffer_size` | 500 000 | 1 M OOMs at pre-alloc |
| | `batch_size` | 512 | |
| | `num_updates` | 256 | |
| **SAC** | `policy_lr` / `q_lr` / `alpha_lr` | 3e-4 | |
| | `gamma` | 0.9 | |
| | `tau` | 0.01 | |
| | `num_q` | 2 | |
| | `num_atoms` | 101 | C51-style critic |
| | `v_min` / `v_max` | -20 / 20 | |
| | `policy_frequency` | 4 | actor updated every 4 critic updates |
| | `learning_starts` | 5 000 | |
| | `autotune` | True | α auto-tuned from `target_entropy` |
| | `freeze_encoder_after_frac` | 0.9 | encoder frozen for last 10 % of training |
| **DR** | `domain_randomization` | True | wrist camera ±3 mm / ±2°, item friction/mass, table friction, lighting, jitter |
| | `apply_jitter` | True | torchvision color jitter — this is the augmentation that needed `expandable_segments` |
| **infra** | `total_timesteps` | 10 M | targeted; checkpoint saved at 800 k |
| | `eval_freq` | 400 000 | every 400 k steps |
| | `seed` | 1 | |
| | `cuda` / `compile` / `cudagraphs` | all True | |
| | `track` / `save_model` | True / True | |

## Code state used

All required code changes are in commits up to and including:
- `1adc269` — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` baked into launcher
- `777f191` — `CNNEncoder` accepts any `H ≥ 56` (so `H = 80` works)
- `a286bda` — side-approach curriculum added (unused here — `SIDE_APPROACH=false`)
- `30813fc` — checkpoint committed at step 800 k

Pulling the repo at `HEAD` (or any commit after `1adc269`) and running the
launch command above reproduces this exact training.

## Deploying this checkpoint

Edit `infer.py` so the network shapes match the 80×144 input (defaults
are still tuned for the older 36×64 checkpoint):

| constant | current default (36×64) | this checkpoint (80×144) |
|---|---|---|
| `IMAGE_H` | 36 | **80** |
| `IMAGE_W` | 64 | **144** |
| `CNN_FLATTEN_DIM` | 3840 | **5376** (= 64 × 6 × 14, Atari profile) |
| `RGB_PROJ_DIM` | 50 | 50 (unchanged) |

The real camera resolution stays at `1920×1080`; the downsample to
`80×144` happens via `cv2.resize(img, (144, 80), INTER_AREA)`.

## What works / what doesn't

- ✓ Real-robot pick succeeds on most attempts.
- ✗ Not perfect — failures likely in the same family as the original
  motivation for the side-approach curriculum: gripper arrives slightly
  pre-closed, moving finger taps the cube top.
- → Next experiments (changing **one** knob at a time vs this baseline):
  1. ✅ **`DROP_PENALTY_COEF=3.0`** — converged to the one-shot-grasp
     behaviour the user wanted. See follow-up below.
  2. `SIDE_APPROACH=true SIDE_APPROACH_OPEN_COEF=0.3` — addresses the
     pre-closed gripper failure mode (untested).
  3. Wider cube size range in `PlaceRandomizationConfig.cube_half_size_range`
     to better bracket the real cube.
  4. `LATENCY=on` (10–50 ms camera lag) to add robustness if real-camera
     pipeline lag is suspected.

## Variant: `eval1_pick_80x144_dropPen3` (drop-penalty 3.0)

**Same config as the baseline + `DROP_PENALTY_COEF=3.0`** (penalty applied
on every `is_grasped` True→False transition). In sim, converged to the
*one-shot* one-attempt-per-episode behaviour the user wanted.

⚠️ **REAL-ROBOT VERDICT (2026-05-20): worse than the baseline.** Tested
on the physical SO101 — **~80 % of attempts failed to grasp the cube**
(grasp success ≈ 20 %). The penalty made the policy too conservative:
it commits to a single grasp attempt and won't recover if the first try
fails. So the in-sim convergence to "one-shot" did not transfer; the
baseline (no drop penalty, allowed to retry) remains the preferred
deploy target until a different fix is tried.

- **Wandb run:** `fedecominelli04_robot/maniskill-so101/a2huk9qa`
- **Checkpoint:** `runs/eval1_pick_80x144_dropPen3/ckpt.pt`, embedded
  `global_step = 2,000,896`
- **Trained:** stopped at ~2.15 M steps after convergence (run completed
  ~2.7 k seconds wall time).

Reward dynamics observed (eval/return, step-aligned vs baseline):

| step    | baseline | dropPen3 | gap   |
|---------|----------|----------|-------|
| 0       | 1.00     | 0.96     | ≈0    |
| 400 k   | 11.98    | 9.06     | −2.9  |
| 800 k   | 13.22    | 9.43     | −3.8  |
| 1.2 M   | 13.19    | 12.79    | −0.4  |
| 2.0 M   | 13.22    | (converged) | — |

Interpretation: at 400–800 k the policy was averaging ≈1 drop/episode
(−3 per ep), then learned to one-shot the grasp by 1.2 M, eliminating
the gap.

To deploy this ckpt, use the same `infer.py` constants as the baseline
(`IMAGE_H=80, IMAGE_W=144, CNN_FLATTEN_DIM=5376, RGB_PROJ_DIM=50`).

## Cross-references

- Wandb sweep history: see [reference_wandb](../../memory) for entity / project / artifact pattern.
- Ablation memory (sim_freq × latency winners): [project_sim2real_ablation_2026-05-20](../../memory).
- Memory-knobs memory (RTX Pro 6000): [project_brev_memory_knobs](../../memory).
