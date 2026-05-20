#!/usr/bin/env bash
# Parametric launcher for the sim_freq × latency 2x2 ablation.
#
# Inputs (env vars, all optional):
#   SIM_FREQ        100 or 300       (physics substep rate)
#   LATENCY         on or off        ("on" → camera lag ∈ [10ms,50ms];
#                                     "off" → camera lag = 0ms)
#   SEED            RNG seed         (default 1)
#   N_DISTRACTORS   distractors      (default 1; n=0→eval1, n=1→eval2, n=3→eval3)
#   EXP_NAME        run name         (default: auto from SIM_FREQ/LATENCY/STAGE)
#
# Usage:
#   SIM_FREQ=100 LATENCY=on  bash scripts/brev_run_ablation.sh
#   SIM_FREQ=100 LATENCY=off bash scripts/brev_run_ablation.sh
#   SIM_FREQ=300 LATENCY=on  bash scripts/brev_run_ablation.sh
#   SIM_FREQ=300 LATENCY=off bash scripts/brev_run_ablation.sh
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ── Ablation knobs (the 4 runs) ─────────────────────────────────────────────
SIM_FREQ="${SIM_FREQ:-100}"
LATENCY="${LATENCY:-off}"

# Map LATENCY → camera_lag_substeps range. At sim_freq=100, 1 substep = 10 ms;
# at sim_freq=300, 1 substep = 3.33 ms. Keep the *wall-clock* lag range at
# 10–50 ms across both sim_freq values.
if [ "$LATENCY" = "off" ]; then
  CAM_LAG_MIN=0
  CAM_LAG_MAX=0
  LAT_TAG="nolat"
elif [ "$LATENCY" = "on" ]; then
  if [ "$SIM_FREQ" = "100" ]; then
    CAM_LAG_MIN=1; CAM_LAG_MAX=5     # 10–50 ms
  elif [ "$SIM_FREQ" = "300" ]; then
    CAM_LAG_MIN=3; CAM_LAG_MAX=15    # 10–50 ms (3/300 = 10 ms, 15/300 = 50 ms)
  else
    echo "ERROR: SIM_FREQ must be 100 or 300, got $SIM_FREQ" >&2; exit 1
  fi
  LAT_TAG="lat"
else
  echo "ERROR: LATENCY must be 'on' or 'off', got $LATENCY" >&2; exit 1
fi

# ── Standard knobs ──────────────────────────────────────────────────────────
SEED="${SEED:-1}"
N_DISTRACTORS="${N_DISTRACTORS:-1}"
case "$N_DISTRACTORS" in
  0) STAGE=eval1 ;;
  1) STAGE=eval2 ;;
  3) STAGE=eval3 ;;
  *) STAGE="eval_n${N_DISTRACTORS}" ;;
esac

# Episode length in control steps (1 step = 1/CONTROL_FREQ s = 100 ms @ 10 Hz).
# Defaults to 10 s (100 steps). Override for pick-only (e.g. EP_STEPS=80 = 8 s).
CONTROL_FREQ=10
EP_STEPS="${EP_STEPS:-100}"

# Pick-only mode: when true, env's reward = grasp+stay-still-for-1s (no place).
# Episode auto-terminates on success. False → full pick-and-place reward.
PICK_ONLY="${PICK_ONLY:-false}"
if [ "$PICK_ONLY" = "true" ]; then
  PICK_ONLY_FLAG="--pick_only_reward"
  PICK_TAG="pick"
elif [ "$PICK_ONLY" = "false" ]; then
  PICK_ONLY_FLAG="--no-pick_only_reward"
  PICK_TAG="place"
else
  echo "ERROR: PICK_ONLY must be 'true' or 'false', got $PICK_ONLY" >&2; exit 1
fi

# Side-approach curriculum (pick-only only). When true, the policy must land
# the fixed finger on the cube first before the normal grasp ladder kicks in.
SIDE_APPROACH="${SIDE_APPROACH:-false}"
if [ "$SIDE_APPROACH" = "true" ]; then
  SIDE_APPROACH_FLAG="--pick_side_approach"
  SIDE_TAG="_side"
elif [ "$SIDE_APPROACH" = "false" ]; then
  SIDE_APPROACH_FLAG="--no-pick_side_approach"
  SIDE_TAG=""
else
  echo "ERROR: SIDE_APPROACH must be 'true' or 'false', got $SIDE_APPROACH" >&2; exit 1
fi
SIDE_APPROACH_OPEN_COEF="${SIDE_APPROACH_OPEN_COEF:-0.3}"

# Drop penalty (pick-only only). Penalty applied on every grasped→not-grasped
# transition (each fumble). Default 0 = off; set e.g. 3.0 to push one-shot grasps.
DROP_PENALTY_COEF="${DROP_PENALTY_COEF:-0.0}"

# Directional-light shadows in the DR env (envs/base_random_env.py:shadows).
# Default ON — high sim2real lighting variation. Set SHADOWS=false to disable
# (saves a lot of GPU shadow-map memory; the difference between "fits" and
# "scene-init OOM" at 1024+ envs × 360x640 render).
SHADOWS="${SHADOWS:-true}"
if [ "$SHADOWS" = "true" ]; then
  SHADOWS_FLAG="--env_shadows"
elif [ "$SHADOWS" = "false" ]; then
  SHADOWS_FLAG="--no-env_shadows"
else
  echo "ERROR: SHADOWS must be 'true' or 'false', got $SHADOWS" >&2; exit 1
fi

# Warm-start: path to a checkpoint to load encoder/actor/critic/log_alpha
# from before training (or the literal "wandb" to pull the agent's :latest
# artifact). Empty = train from scratch. The render resolution does NOT
# affect the network, so a ckpt trained at one render res loads cleanly at
# another — used to change render/num_envs without losing the learned policy.
CHECKPOINT="${CHECKPOINT:-}"
if [ -n "$CHECKPOINT" ]; then
  CHECKPOINT_FLAG="--checkpoint=$CHECKPOINT"
else
  CHECKPOINT_FLAG=""
fi

ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-36}"
IMAGE_WIDTH="${IMAGE_WIDTH:-64}"           # exact 16:9 — matches real cam 1920x1080 (calibrated 2026-05-19)
# Sim render resolution. Default 72x128 (9:16 landscape, longest side=128).
# Memory ≈ 25× less than the train_squint.py default 360x640. Aspect 0.5625
# is exact 9/16, EXACTLY matches the policy input aspect (36x64 = 0.5625).
# 128→64 is ÷2 and 72→36 is ÷2 — uniform integer area-pool.
RENDER_HEIGHT="${RENDER_HEIGHT:-72}"
RENDER_WIDTH="${RENDER_WIDTH:-128}"

# RTX 6000 96 GB knobs (mirror brev_run_rtx6000_32x32.sh).
NUM_ENVS="${NUM_ENVS:-6144}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-4000000}"
NUM_UPDATES="${NUM_UPDATES:-256}"
BATCH_SIZE="${BATCH_SIZE:-768}"

EXP_NAME="${EXP_NAME:-${STAGE}_${PICK_TAG}${SIDE_TAG}_sim${SIM_FREQ}_${LAT_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-ABLATION-${STAGE}-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

# Avoid fragmentation-induced OOM during color-jitter peaks at higher image
# dims. The default caching allocator can fragment until a 100-300 MiB temp
# can't find a contiguous slot even when GBs are free. Expandable segments
# let the allocator grow contiguous regions on demand.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ""
echo "================================================================"
echo "  Ablation run: $EXP_NAME"
echo "  mode=$PICK_TAG (pick_only_reward=$PICK_ONLY)   side_approach=$SIDE_APPROACH (open_coef=$SIDE_APPROACH_OPEN_COEF)   drop_penalty=$DROP_PENALTY_COEF   shadows=$SHADOWS"
echo "  sim_freq=$SIM_FREQ Hz   control_freq=$CONTROL_FREQ Hz   ep_steps=$EP_STEPS ($((EP_STEPS / CONTROL_FREQ)) s)"
echo "  latency=$LATENCY (camera_lag substeps in [$CAM_LAG_MIN, $CAM_LAG_MAX])"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE"
echo "  num_updates=$NUM_UPDATES  batch_size=$BATCH_SIZE"
echo "  image=${IMAGE_HEIGHT}x${IMAGE_WIDTH}   render=${RENDER_HEIGHT}x${RENDER_WIDTH}   warm_start=${CHECKPOINT:-<none>}"
echo "  group=$WANDB_GROUP"
echo "================================================================"

python train_squint.py \
    --env_id="$ENV_ID" \
    --exp_name="$EXP_NAME" \
    --agent_name="$EXP_NAME" \
    --seed="$SEED" \
    --n_distractors="$N_DISTRACTORS" \
    --total_timesteps="$TOTAL_TIMESTEPS" \
    --eval_max_episode_steps="$EP_STEPS" \
    --sim_freq="$SIM_FREQ" \
    --control_freq="$CONTROL_FREQ" \
    --camera_lag_substeps_min="$CAM_LAG_MIN" \
    --camera_lag_substeps_max="$CAM_LAG_MAX" \
    --num_envs="$NUM_ENVS" \
    --num_eval_envs="$NUM_EVAL_ENVS" \
    --buffer_size="$BUFFER_SIZE" \
    --num_updates="$NUM_UPDATES" \
    --batch_size="$BATCH_SIZE" \
    --image_height="$IMAGE_HEIGHT" \
    --image_width="$IMAGE_WIDTH" \
    --render_height="$RENDER_HEIGHT" \
    --render_width="$RENDER_WIDTH" \
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model \
    --pick_side_approach_open_coef="$SIDE_APPROACH_OPEN_COEF" \
    --drop_penalty_coef="$DROP_PENALTY_COEF" \
    $PICK_ONLY_FLAG \
    $SIDE_APPROACH_FLAG \
    $SHADOWS_FLAG \
    $CHECKPOINT_FLAG

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
