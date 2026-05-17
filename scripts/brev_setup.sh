#!/usr/bin/env bash
# Provision a Brev (or any Ubuntu+CUDA) NVIDIA VM for Squint training.
#
# Usage (on the VM, after `ssh brev-...`):
#   export WANDB_API_KEY=...        # required
#   export HF_TOKEN=...             # optional (HF checkpoint mirror)
#   export SQUINT_REMOTE=git@github.com:fedecomi04/squint.git   # or https://...
#   bash brev_setup.sh
#
# Idempotent: safe to re-run on the same VM to refresh code or re-login.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/squint}"
SQUINT_REMOTE="${SQUINT_REMOTE:-https://github.com/fedecomi04/squint.git}"

echo "==[1/4] miniforge + base tooling ============================"
sudo apt-get update -qq
# libvulkan1 + vulkan-tools: SAPIEN's renderer logs a fallback warning
# when libvulkan isn't installed system-wide ("Failed to find system
# libvulkan. Fallback to SAPIEN builtin libvulkan."). Installing the
# loader silences it and uses the NVIDIA driver's Vulkan ICD instead of
# the bundled fallback.
# nvtop intentionally omitted — not in stock Ubuntu repos on every image.
sudo apt-get install -y -qq \
    git wget tmux htop ffmpeg \
    libvulkan1 vulkan-tools

# NVIDIA Vulkan ICD (libGLX_nvidia + nvidia_icd.json). Brev's H100 images
# ship a HEADLESS NVIDIA driver — compute libs only — and pin `libnvidia-gl*`
# at priority -1 so apt refuses to install it. Without this lib SAPIEN can't
# create a Vulkan instance: `vk::createInstanceUnique: ErrorIncompatibleDriver`.
# On compute-only A100 images this lib is already pre-installed, so the check
# is a no-op there.
if ! ls /usr/share/vulkan/icd.d/ 2>/dev/null | grep -q nvidia_icd.json; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  MAJOR="${DRV%%.*}"
  if [ -n "$MAJOR" ]; then
    echo "  (no NVIDIA Vulkan ICD; installing libnvidia-gl-${MAJOR} via pin override)"
    # If the cuda-keyring isn't present yet, pull the .deb.
    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
      wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
      sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null
      sudo apt-get update -y >/dev/null
    fi
    printf 'Package: libnvidia-gl-%s\nPin: version *\nPin-Priority: 1000\n' "$MAJOR" \
      | sudo tee /etc/apt/preferences.d/zz-nvidia-gl-override > /dev/null
    sudo apt-get install -y --allow-downgrades "libnvidia-gl-${MAJOR}=${DRV}-1ubuntu1" \
      || sudo apt-get install -y "libnvidia-gl-${MAJOR}"
  fi
fi

if [ ! -d "$HOME/miniforge3" ]; then
  # NB: not /tmp — many cloud VM images mount /tmp noexec, which breaks
  # the miniforge self-extractor.
  INSTALLER="$HOME/miniforge.sh"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$INSTALLER"
  bash "$INSTALLER" -b -p "$HOME/miniforge3"
  rm -f "$INSTALLER"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

echo "==[2/4] clone / update repo =================================="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$SQUINT_REMOTE" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only

echo "==[3/4] conda env from environment.yaml ====================="
if ! conda env list | awk '{print $1}' | grep -qx squint; then
  conda env create -f environment.yaml
else
  echo "  (env 'squint' already exists, skipping create)"
fi
conda activate squint

# Belt-and-suspenders: install coacd into already-existing squint envs that
# were created from an older environment.yaml (before coacd was listed).
python -c "import coacd" 2>/dev/null || pip install -q coacd

echo "==[4/4] wandb / HF login ====================================="
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is required. Get it from https://wandb.ai/authorize" >&2
  exit 1
fi
wandb login --relogin "$WANDB_API_KEY"

if [ -n "${HF_TOKEN:-}" ]; then
  pip install -q huggingface_hub
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Sanity-check CUDA + parallel-env build.
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
importlib.import_module("mani_skill")
print("mani_skill: ok")
PY

echo ""
echo "==> Setup complete. Next: bash scripts/brev_run_evals.sh"
