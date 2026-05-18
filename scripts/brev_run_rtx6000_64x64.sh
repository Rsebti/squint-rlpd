#!/usr/bin/env bash
# One-shot launcher for a second RTX Pro 6000 96 GB VM running the 64x64
# image-input variant. 64x64 = 4× pixels vs 32x32 → buffer-per-transition
# also 4×, so BUFFER_SIZE must drop (4M no longer fits in 96 GB VRAM).
#
# Usage on the VM:
#   tmux new -s squint
#   bash ~/squint/scripts/brev_setup.sh         # one-time install
#   bash ~/squint/scripts/brev_run_rtx6000_64x64.sh
set -euo pipefail

# WANDB key hardcoded per user request.
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ── Knobs (tuned for RTX Pro 6000 96 GB at 64x64) ────────────────────────────
export IMAGE_SIZE=64
export NUM_ENVS="${NUM_ENVS:-6144}"          # 96 GB VRAM has headroom (32x32 used 54%)
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
export BUFFER_SIZE="${BUFFER_SIZE:-2000000}" # 2M transitions ~50 GB at 64x64 uint8 (vs 25 GB at 32x32)
export NUM_UPDATES="${NUM_UPDATES:-256}"     # Same as 32x32; encoder is heavier so don't pile on
export BATCH_SIZE="${BATCH_SIZE:-1024}"      # Wider batch absorbs GPU idle headroom; Blackwell-friendly

N_DISTRACTORS="${N_DISTRACTORS:-1}"
# Curriculum-stage naming: n=0 → eval1, n=1 → eval2, n=3 → eval3.
case "$N_DISTRACTORS" in
  0) STAGE=eval1 ;;
  1) STAGE=eval2 ;;
  3) STAGE=eval3 ;;
  *) STAGE="eval_n${N_DISTRACTORS}" ;;
esac

EXP_NAME="${EXP_NAME:-${STAGE}_rtx6000_64x64}"
SEED="${SEED:-1}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
EP_STEPS="${EP_STEPS:-75}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-RTX6000-64x64-${STAGE}-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

echo ""
echo "================================================================"
echo "  RTX Pro 6000 / 64x64 / $EXP_NAME"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS  ep=$EP_STEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE"
echo "  num_updates=$NUM_UPDATES  batch_size=$BATCH_SIZE  image_size=$IMAGE_SIZE"
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
    --image_size="$IMAGE_SIZE" \
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
