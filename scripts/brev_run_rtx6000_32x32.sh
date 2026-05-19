#!/usr/bin/env bash
# One-shot launcher tuned for the RTX Pro 6000 96 GB VM (Blackwell).
# 96 GB VRAM + 192 GB RAM lets us push num_envs and buffer harder than on
# H100; Blackwell tensor cores like wider batches so batch_size goes up too.
# Image resolution updated 2026-05 from 32x32 → 36x64 (16:9, matches the
# calibrated 1920x1080 real camera). Filename kept for the bootstrap default.
#
# Usage on the VM:
#   tmux new -s squint
#   bash ~/squint/scripts/brev_setup.sh         # one-time install
#   bash ~/squint/scripts/brev_run_rtx6000_32x32.sh
set -euo pipefail

# WANDB key hardcoded per user request (rotate later).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ── Knobs (tuned for RTX Pro 6000 96 GB / Blackwell) ────────────────────────
# Policy input resolution (16:9, matches calibrated 1920x1080 real camera).
export IMAGE_HEIGHT="${IMAGE_HEIGHT:-36}"
export IMAGE_WIDTH="${IMAGE_WIDTH:-64}"
# Sim render resolution (uniform integer area-pool to image_height/width).
export RENDER_HEIGHT="${RENDER_HEIGHT:-72}"
export RENDER_WIDTH="${RENDER_WIDTH:-128}"
export NUM_ENVS="${NUM_ENVS:-6144}"          # 96 GB VRAM headroom over H100's 80 GB
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
export BUFFER_SIZE="${BUFFER_SIZE:-4000000}" # 4 M transitions (~28 GB VRAM at 36x64 uint8)
export NUM_UPDATES="${NUM_UPDATES:-256}"     # Paper default; Blackwell makes 384 unnecessary
export BATCH_SIZE="${BATCH_SIZE:-768}"       # Wider batch → better tensor-core utilisation

N_DISTRACTORS="${N_DISTRACTORS:-1}"
# Curriculum-stage naming: n=0 → eval1, n=1 → eval2, n=3 → eval3.
case "$N_DISTRACTORS" in
  0) STAGE=eval1 ;;
  1) STAGE=eval2 ;;
  3) STAGE=eval3 ;;
  *) STAGE="eval_n${N_DISTRACTORS}" ;;
esac

EXP_NAME="${EXP_NAME:-${STAGE}_rtx6000_32x32}"
SEED="${SEED:-1}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
EP_STEPS="${EP_STEPS:-75}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-RTX6000-32x32-${STAGE}-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

echo ""
echo "================================================================"
echo "  RTX Pro 6000 / 32x32 / $EXP_NAME"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS  ep=$EP_STEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE"
echo "  num_updates=$NUM_UPDATES  batch_size=$BATCH_SIZE  image=${IMAGE_HEIGHT}x${IMAGE_WIDTH}  render=${RENDER_HEIGHT}x${RENDER_WIDTH}"
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
    --save_model

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
