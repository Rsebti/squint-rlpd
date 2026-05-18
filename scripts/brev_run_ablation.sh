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
LATENCY="${LATENCY:-on}"

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

# 7 s episode @ 10 Hz control = 70 steps. (Sim_freq does NOT change this.)
CONTROL_FREQ=10
EP_STEPS=70

ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-32}"
IMAGE_WIDTH="${IMAGE_WIDTH:-42}"   # landscape, aspect-preserved

# RTX 6000 96 GB knobs (mirror brev_run_rtx6000_32x32.sh).
NUM_ENVS="${NUM_ENVS:-6144}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-4000000}"
NUM_UPDATES="${NUM_UPDATES:-256}"
BATCH_SIZE="${BATCH_SIZE:-768}"

EXP_NAME="${EXP_NAME:-${STAGE}_sim${SIM_FREQ}_${LAT_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-ABLATION-${STAGE}-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

echo ""
echo "================================================================"
echo "  Ablation run: $EXP_NAME"
echo "  sim_freq=$SIM_FREQ Hz   control_freq=$CONTROL_FREQ Hz   ep_steps=$EP_STEPS (7 s)"
echo "  latency=$LATENCY (camera_lag substeps in [$CAM_LAG_MIN, $CAM_LAG_MAX])"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE"
echo "  num_updates=$NUM_UPDATES  batch_size=$BATCH_SIZE  image=${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
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
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
