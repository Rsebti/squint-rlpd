#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  ONE-SHOT BOOTSTRAP for a fresh RTX Pro 6000 Blackwell VM on Brev.
#
#  Run on the VM with a single line:
#    curl -fsSL https://raw.githubusercontent.com/fedecomi04/squint/master/scripts/brev_bootstrap_rtx6000.sh | bash
#
#  Handles, in order:
#    1. Clone / fast-forward the repo
#    2. apt deps (libvulkan, libnvidia-gl with apt-pin override)
#    3. Miniforge + conda env
#    4. Torch swap to 2.7.1+cu128 if running on Blackwell (sm_100+)
#    5. wandb login (key baked in below)
#    6. CUDA + ManiSkill sanity check (fails fast before training)
#    7. Launches training in a DETACHED tmux session named "squint"
#
#  After this script finishes, do:
#    tmux attach -t squint        # to watch training
#    tail -f ~/training.log       # alternative: just stream the log
#
#  Override training knobs by exporting BEFORE piping to bash, e.g.:
#    N_DISTRACTORS=1 EXP_NAME=eval1_n1 curl -fsSL ... | bash
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Hardcoded credentials & training defaults (override via env vars) ───────
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export N_DISTRACTORS="${N_DISTRACTORS:-0}"
export EXP_NAME="${EXP_NAME:-eval1_rtx6000_32x32_n0}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
export WANDB_GROUP="${WANDB_GROUP:-SQUINT-RTX6000-32x32-n0-$(date +%Y%m%d-%H%M)}"

REPO_DIR="$HOME/squint"
SQUINT_REMOTE="https://github.com/fedecomi04/squint.git"

# ── 1. Clone or update repo ────────────────────────────────────────────────
echo "==[BOOT] git: clone or pull"
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch --quiet
    git checkout master --quiet
    git pull --ff-only --quiet
else
    git clone --quiet "$SQUINT_REMOTE" "$REPO_DIR"
fi

# ── 2-5. Run setup (apt deps, miniforge, conda env, torch swap, wandb) ─────
echo "==[BOOT] running brev_setup.sh (installs deps + handles Blackwell)"
bash "$REPO_DIR/scripts/brev_setup.sh"

# ── 6. Sanity check before burning compute ─────────────────────────────────
echo "==[BOOT] sanity check"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
python - <<'PY'
import torch, sapien, mani_skill
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"  torch  {torch.__version__}  archs={torch.cuda.get_arch_list()}")
print(f"  gpu    {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print(f"  sapien {sapien.__version__}")
torch.zeros(4, device='cuda').sum().item()  # crash here if torch arch wrong
print("  CUDA op OK")
PY

# ── 7. Launch training in detached tmux ────────────────────────────────────
if tmux has-session -t squint 2>/dev/null; then
    echo ""
    echo "  WARNING: tmux session 'squint' already exists. Kill it first if you want to restart:"
    echo "    tmux kill-session -t squint"
    echo "  Then re-run this bootstrap."
    exit 1
fi

tmux new-session -d -s squint \
    "cd $REPO_DIR && \
     export WANDB_API_KEY='$WANDB_API_KEY' && \
     export N_DISTRACTORS='$N_DISTRACTORS' && \
     export EXP_NAME='$EXP_NAME' && \
     export TOTAL_TIMESTEPS='$TOTAL_TIMESTEPS' && \
     export WANDB_GROUP='$WANDB_GROUP' && \
     bash scripts/brev_run_rtx6000_32x32.sh 2>&1 | tee $HOME/training.log"

echo ""
echo "================================================================"
echo "  Training started in detached tmux session 'squint'."
echo ""
echo "  exp_name=$EXP_NAME"
echo "  n_distractors=$N_DISTRACTORS  total_timesteps=$TOTAL_TIMESTEPS"
echo "  wandb_group=$WANDB_GROUP"
echo ""
echo "  Watch:    tmux attach -t squint     (detach with Ctrl-B then D)"
echo "  Tail log: tail -f ~/training.log"
echo "  Stop:     tmux kill-session -t squint"
echo "================================================================"
