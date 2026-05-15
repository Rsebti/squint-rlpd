#!/usr/bin/env bash
# One-shot entrypoint: provision VM + run all 3 training stages.
#
# On a fresh Brev/MassedCompute L40S VM:
#   1. SSH in
#   2. curl -L https://raw.githubusercontent.com/fedecomi04/squint/master/scripts/brev_run_all.sh -o run_all.sh
#   3. export WANDB_API_KEY=...   WANDB_ENTITY=...
#   4. tmux new -s squint
#   5. bash run_all.sh
#   6. Ctrl-b d to detach. Reconnect anytime with `tmux attach -t squint`.
#
# Total wall-time: ~5–9 h on L40S (3× 20M @ 2048 envs). All checkpoints upload
# to wandb at the end of each stage, so VM destruction after completion is safe.
set -euo pipefail

: "${WANDB_API_KEY:?export WANDB_API_KEY before running}"
# WANDB_ENTITY is optional — empty/unset means "use the default entity
# associated with the API key" (your personal user on wandb.ai).
export WANDB_ENTITY="${WANDB_ENTITY:-}"

REPO_DIR="${REPO_DIR:-$HOME/squint}"
SQUINT_REMOTE="${SQUINT_REMOTE:-https://github.com/fedecomi04/squint.git}"

# Stage 1: provision the VM (idempotent — safe to re-run).
if [ ! -f "$REPO_DIR/scripts/brev_setup.sh" ]; then
  echo ">> Bootstrap: cloning repo to fetch setup script..."
  git clone "$SQUINT_REMOTE" "$REPO_DIR"
fi
bash "$REPO_DIR/scripts/brev_setup.sh"

# Stage 2: run the 3-stage training chain.
bash "$REPO_DIR/scripts/brev_run_evals.sh"

echo ""
echo "================================================================"
echo " All 3 stages complete. Checkpoints are in wandb:"
echo "   $WANDB_ENTITY/${WANDB_PROJECT:-maniskill-so101}"
echo "   artifacts: model_eval1_<env>_1, model_eval2_<env>_2, model_eval3_<env>_3"
echo " You can now delete the VM."
echo "================================================================"
