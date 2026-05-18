#!/usr/bin/env bash
# One-shot launcher tuned for the RTX Pro 6000 96 GB VM (Blackwell).
# 96 GB VRAM + 192 GB RAM lets us push num_envs and buffer harder than on
# H100; Blackwell tensor cores like wider batches so batch_size goes up too.
# image_size stays at 32 for the higher-resolution visual experiment.
#
# Usage on the VM:
#   tmux new -s squint
#   bash ~/squint/scripts/brev_setup.sh         # one-time install
#   bash ~/squint/scripts/brev_run_rtx6000_32x32.sh
set -euo pipefail

# WANDB key hardcoded per user request (rotate later).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ── Knobs (tuned for RTX Pro 6000 96 GB / Blackwell) ────────────────────────
export IMAGE_SIZE=32
export NUM_ENVS="${NUM_ENVS:-6144}"          # 96 GB VRAM headroom over H100's 80 GB
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
export BUFFER_SIZE="${BUFFER_SIZE:-4000000}" # 4 M transitions (~28 GB VRAM at 32x32 uint8)
export NUM_UPDATES="${NUM_UPDATES:-256}"     # Paper default; Blackwell makes 384 unnecessary
export BATCH_SIZE="${BATCH_SIZE:-768}"       # Wider batch → better tensor-core utilisation

EXP_NAME="${EXP_NAME:-eval1_rtx6000_32x32}"
SEED="${SEED:-1}"
N_DISTRACTORS="${N_DISTRACTORS:-1}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
EP_STEPS="${EP_STEPS:-75}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-RTX6000-32x32-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

echo ""
echo "================================================================"
echo "  RTX Pro 6000 / 32x32 / $EXP_NAME"
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
