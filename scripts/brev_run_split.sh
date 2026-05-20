#!/usr/bin/env bash
# Launcher for the "split" task: train a policy to push the TWO cubes apart
# (no grasping) until the surface gap between them reaches SPLIT_TARGET_GAP,
# then hold both cubes still. This is the eval2 pre-step before the pick +
# IK-to-bowl pipeline — separating the cubes so one can be isolated.
#
# Reward (envs/place.py:_compute_dense_reward_split):
#   reach nearest cube + split_sep_coef·separation_progress
#   − table-touch penalty − bowl-touch penalty
#   success (separated + both cubes static 0.5 s) → terminal bonus, early stop.
#
# Usage (on the VM):
#   bash scripts/brev_run_split.sh
#   SPLIT_TARGET_GAP=0.04 bash scripts/brev_run_split.sh
# Or via the one-line bootstrap:
#   LAUNCHER=scripts/brev_run_split.sh curl -fsSL .../brev_bootstrap_rtx6000.sh | bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ── Split-specific knobs ────────────────────────────────────────────────────
# SPLIT_TARGET_GAP: surface-to-surface gap (m) the two cubes must reach. The
# cubes spawn touching (~2 cm centre-to-centre); 0.03 m gap ≈ 0.05 m centres.
SPLIT_TARGET_GAP="${SPLIT_TARGET_GAP:-0.03}"
SPLIT_SEP_COEF="${SPLIT_SEP_COEF:-1.0}"

# ── Task / stage ────────────────────────────────────────────────────────────
# Split is the eval2 setup: exactly two cubes (1 distractor). The reward
# requires n_distractors >= 1; default to 1.
N_DISTRACTORS="${N_DISTRACTORS:-1}"
SEED="${SEED:-1}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"

# ── Physics / control (winner config: 100 Hz sim, 10 Hz control, no latency) ─
SIM_FREQ="${SIM_FREQ:-100}"
CONTROL_FREQ="${CONTROL_FREQ:-10}"
CAM_LAG_MIN="${CAM_LAG_MIN:-0}"
CAM_LAG_MAX="${CAM_LAG_MAX:-0}"

# Episode length in control steps. Registered max is 100 (=10 s); the terminal
# bonus accounting assumes 100, so keep eval at 100 too. Successful episodes
# auto-terminate early on separation.
EP_STEPS="${EP_STEPS:-100}"

# Directional-light shadows. Default OFF — same as the proven 80×144 / 2048-env
# savage-DR runs (shadows×lights×cameras OOMs the parallel renderer at this
# env count / render res). Set SHADOWS=true only with fewer envs.
SHADOWS="${SHADOWS:-false}"
if [ "$SHADOWS" = "true" ]; then
  SHADOWS_FLAG="--env_shadows"
elif [ "$SHADOWS" = "false" ]; then
  SHADOWS_FLAG="--no-env_shadows"
else
  echo "ERROR: SHADOWS must be 'true' or 'false', got $SHADOWS" >&2; exit 1
fi

# Warm-start from a checkpoint (or literal "wandb"). Empty = from scratch.
CHECKPOINT="${CHECKPOINT:-}"
if [ -n "$CHECKPOINT" ]; then
  CHECKPOINT_FLAG="--checkpoint=$CHECKPOINT"
else
  CHECKPOINT_FLAG=""
fi

# ── Resolution (proven 80×144 policy input, 160×288 render) ──────────────────
IMAGE_HEIGHT="${IMAGE_HEIGHT:-80}"
IMAGE_WIDTH="${IMAGE_WIDTH:-144}"
RENDER_HEIGHT="${RENDER_HEIGHT:-160}"
RENDER_WIDTH="${RENDER_WIDTH:-288}"

# ── RTX 6000 96 GB knobs (mirror the proven 80×144 savage-DR runs) ──────────
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-500000}"
NUM_UPDATES="${NUM_UPDATES:-256}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"

EXP_NAME="${EXP_NAME:-eval2_split_80x144}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"
WANDB_GROUP="${WANDB_GROUP:-SQUINT-SPLIT-eval2-$(date +%Y%m%d-%H%M)}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ""
echo "================================================================"
echo "  Split run: $EXP_NAME"
echo "  target_gap=$SPLIT_TARGET_GAP m   sep_coef=$SPLIT_SEP_COEF   shadows=$SHADOWS"
echo "  sim_freq=$SIM_FREQ Hz   control_freq=$CONTROL_FREQ Hz   ep_steps=$EP_STEPS"
echo "  cam_lag substeps in [$CAM_LAG_MIN, $CAM_LAG_MAX]"
echo "  seed=$SEED  n_distractors=$N_DISTRACTORS  total=$TOTAL_TIMESTEPS"
echo "  num_envs=$NUM_ENVS  num_eval_envs=$NUM_EVAL_ENVS  buffer=$BUFFER_SIZE"
echo "  image=${IMAGE_HEIGHT}x${IMAGE_WIDTH}  render=${RENDER_HEIGHT}x${RENDER_WIDTH}  warm_start=${CHECKPOINT:-<none>}"
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
    --split_only_reward \
    --no-pick_only_reward \
    --split_target_gap="$SPLIT_TARGET_GAP" \
    --split_sep_coef="$SPLIT_SEP_COEF" \
    --track \
    --wandb_project_name="$WANDB_PROJECT" \
    --wandb_group="$WANDB_GROUP" \
    --save_model \
    $SHADOWS_FLAG \
    $CHECKPOINT_FLAG

echo ""
echo "Done. Checkpoint at runs/$EXP_NAME/ckpt.pt"
