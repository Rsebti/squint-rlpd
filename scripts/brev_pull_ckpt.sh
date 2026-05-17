#!/usr/bin/env bash
# Re-download a stage's checkpoint from wandb to runs/{exp_name}/ckpt.pt.
# Use this after a Brev VM gets re-created and you need eval1's ckpt back
# before launching eval2/eval3.
#
# Usage:
#   bash scripts/brev_pull_ckpt.sh eval1
#   bash scripts/brev_pull_ckpt.sh eval2          # if you want to inspect eval2
set -euo pipefail

EXP_NAME="${1:?usage: brev_pull_ckpt.sh <exp_name>}"
ENV_ID="${ENV_ID:-SO101PlaceCube-v1}"
case "$EXP_NAME" in
  eval1) SEED_FOR_STAGE=1 ;;
  eval2) SEED_FOR_STAGE=2 ;;
  eval3) SEED_FOR_STAGE=3 ;;
  *)     SEED_FOR_STAGE=1 ;;
esac
SEED="${SEED:-$SEED_FOR_STAGE}"

WANDB_ENTITY="${WANDB_ENTITY:?set WANDB_ENTITY first}"
WANDB_PROJECT="${WANDB_PROJECT:-maniskill-so101}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate squint
cd "${REPO_DIR:-$HOME/squint}"

mkdir -p "runs/$EXP_NAME"

python - "$WANDB_ENTITY" "$WANDB_PROJECT" "$EXP_NAME" "$ENV_ID" "$SEED" <<'PY'
import sys, shutil, pathlib, wandb
entity, project, exp_name, env_id, seed = sys.argv[1:6]
art = f"{entity}/{project}/model_{exp_name}_{env_id}_{seed}:latest"
print(f"Downloading {art} ...")
api = wandb.Api()
artifact = api.artifact(art)
out = artifact.download(root=f"runs/{exp_name}")
# train_squint expects runs/{exp_name}/ckpt.pt — if the artifact has another
# filename, copy it to ckpt.pt so the warm-start path matches.
dst = pathlib.Path(out) / "ckpt.pt"
if not dst.exists():
    cands = list(pathlib.Path(out).glob("*.pt"))
    if not cands:
        raise SystemExit(f"No .pt file found inside artifact at {out}")
    shutil.copy(cands[0], dst)
print(f"OK -> {dst}")
PY
