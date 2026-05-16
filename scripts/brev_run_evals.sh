#!/usr/bin/env bash
# 3-stage training chain on Brev/NVIDIA VM:
#   eval1: 20M steps, fresh
#   eval2: 20M steps, warm-start from eval1 ckpt (different seed)
#   eval3: 20M steps, warm-start from eval1 ckpt (different seed)
#
# Each stage saves to runs/{exp_name}/ckpt.pt and uploads to wandb as
# artifact name `model_{exp_name}_{env_id}_{seed}` (see Logger.upload_checkpoint).
# Local files are the source of truth for the warm-start chain; wandb is the
# off-machine backup. If the VM dies between stages, run scripts/brev_pull_ckpt.sh
# to re-download eval1 from wandb before re-launching eval2/eval3.
#
# Usage (after brev_setup.sh has run successfully):
#   export WANDB_ENTITY=fedecomi04                       # your wandb user/team
#   bash scripts/brev_run_evals.sh                       # all 3 stages
#   bash scripts/brev_run_evals.sh eval2 eval3           # only the warm-start ones
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

# ── Knobs (edit here, not on the command line) ────────────────────────────────
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"   # 20M per stage
EP_STEPS="${EP_STEPS:-100}"                      # 3.3 s @ 30 Hz. Matches local training; bump to 150 for longer cycles.
NUM_ENVS="${NUM_ENVS:-3072}"           # A100 80GB sweet spot; drop to 2048 on L40S/48GB
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-32}"   # halves eval-metric noise vs the trainer's default 16

# Unique group tag for this launch — all 3 stages share it, so they appear
# together in wandb under a single, timestamped group (eval1/eval2/eval3 still
# get distinct run IDs + names). Override via WANDB_GROUP if you want a fixed
# label across re-launches.
WANDB_GROUP="${WANDB_GROUP:-SQUINT-$(date +%Y%m%d-%H%M)}"
# WANDB_ENTITY is optional — leave empty to let wandb pick the default entity
# associated with the API key (your personal user). Set explicitly only if you
# want runs under a team/workspace and that workspace exists on wandb.ai.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"

EVAL1_CKPT="runs/eval1/ckpt.pt"

# ── Single-stage launcher ─────────────────────────────────────────────────────
run_stage() {
  local exp_name="$1"; local seed="$2"; local ckpt_path="${3:-}"
  local extra=()
  if [ -n "$ckpt_path" ]; then
    if [ ! -f "$ckpt_path" ]; then
      echo "ERROR: warm-start ckpt missing at $ckpt_path. Run scripts/brev_pull_ckpt.sh first." >&2
      exit 1
    fi
    extra+=(--checkpoint="$ckpt_path")
  fi
  if [ -n "$WANDB_ENTITY" ]; then
    extra+=(--wandb_entity="$WANDB_ENTITY")
  fi

  echo ""
  echo "================================================================"
  echo "  $exp_name | seed=$seed | total=$TOTAL_TIMESTEPS | ep=$EP_STEPS"
  echo "  warm-start: ${ckpt_path:-<none>}    wandb_entity: ${WANDB_ENTITY:-<default>}"
  echo "================================================================"

  python train_squint.py \
    --env_id="$ENV_ID" \
    --exp_name="$exp_name" \
    --agent_name="$exp_name" \
    --seed="$seed" \
    --total_timesteps="$TOTAL_TIMESTEPS" \
    --eval_max_episode_steps="$EP_STEPS" \
    --num_envs="$NUM_ENVS" \
    --num_eval_envs="$NUM_EVAL_ENVS" \
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model \
    "${extra[@]}"
}

# ── Stage selector (default: all three, sequential) ───────────────────────────
declare -A STAGES=(
  [eval1]="1 "
  [eval2]="2 $EVAL1_CKPT"
  [eval3]="3 $EVAL1_CKPT"
)

REQUESTED=("$@")
if [ "${#REQUESTED[@]}" -eq 0 ]; then
  REQUESTED=(eval1 eval2 eval3)
fi

for stage in "${REQUESTED[@]}"; do
  if [ -z "${STAGES[$stage]:-}" ]; then
    echo "Unknown stage '$stage'. Valid: eval1 eval2 eval3" >&2; exit 1
  fi
  # shellcheck disable=SC2086
  run_stage "$stage" ${STAGES[$stage]}
done

echo ""
echo "All requested stages finished. Checkpoints in runs/{eval1,eval2,eval3}/ckpt.pt"
echo "Also uploaded to wandb under $WANDB_ENTITY/$WANDB_PROJECT (artifact model_<exp>_<env>_<seed>:latest)."
