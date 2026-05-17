#!/usr/bin/env bash
# One-shot launcher for the H100 VM (`mighty-maroon-pelican`) with image
# input bumped from 16x16 to 32x32. Tags exp_name so it's distinguishable
# in wandb from the A100 / 16x16 baselines.
#
# Usage on the VM (after `brev shell mighty-maroon-pelican`):
#   tmux new -s squint
#   bash ~/squint/scripts/brev_setup.sh        # one-time, idempotent
#   bash ~/squint/scripts/brev_run_h100_32x32.sh
set -euo pipefail

# WANDB key hardcoded per user request (they accept the leak; rotate later).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

export IMAGE_SIZE=32
export NUM_ENVS="${NUM_ENVS:-4096}"   # H100 80 GB has more VRAM headroom than A100 80 GB; 4096 is a safer ramp
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
export WANDB_GROUP="${WANDB_GROUP:-SQUINT-H100-32x32-$(date +%Y%m%d-%H%M)}"

# Stage-name suffix so wandb/runs/ dir cleanly identifies this experiment.
# We override the eval1 alias by editing the STAGES map indirectly: simplest
# way is to launch a fresh stage with a different name.
ORIG_DIR="${REPO_DIR:-$HOME/squint}"
cd "$ORIG_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint

EXP_NAME="eval1_h100_32x32"
SEED="${SEED:-1}"
N_DISTRACTORS="${N_DISTRACTORS:-0}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
EP_STEPS="${EP_STEPS:-75}"
BUFFER_SIZE="${BUFFER_SIZE:-3000000}"
NUM_UPDATES="${NUM_UPDATES:-384}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"

echo ""
echo "================================================================"
echo "  H100 / 32x32 / $EXP_NAME"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS  ep=$EP_STEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE  updates=$NUM_UPDATES"
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
    --image_size="$IMAGE_SIZE" \
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
