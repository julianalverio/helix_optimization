#!/usr/bin/env bash
# Pod-side bootstrap for BoltzGen de novo binder design.
# No conda. Just pip install boltzgen, pre-fetch weights, run.
set -euo pipefail

REPO=/workspace/twistr
export HF_HOME=/workspace/cache/huggingface
export BOLTZGEN_CACHE=/workspace/cache/boltzgen
mkdir -p "$HF_HOME" "$BOLTZGEN_CACHE"

echo "==> pip install boltzgen[cuda]"
# The pre-shipped torchvision in runpod/pytorch:cuda12.4.1 is ABI-pinned to
# torch 2.4.1+cu124. `pip install boltzgen` upgrades torch, breaking
# torchvision with `AttributeError: torchvision has no attribute 'extension'`.
# Nuke stale torchvision/torchaudio first, install boltzgen[cuda] (pulls a
# coherent torch/vision/audio bundle).
pip uninstall -y torchvision torchaudio || true
pip install --no-cache-dir 'boltzgen[cuda]' \
    || { echo "FATAL: boltzgen install failed"; exit 2; }

# boltzgen[cuda] pulls torch 2.11+ which JIT-compiles fused kernels via
# nvrtc 13 (libnvrtc-builtins.so.13.0). NO PyPI wheel ships this lib —
# the unsuffixed nvidia-cuda-nvrtc wheel turned out to ship only CUDA 12.4.
# Install CUDA 13 nvrtc from NVIDIA's apt repo (only need the small
# cuda-nvrtc-13-0 package, not the full toolkit).
echo "==> installing CUDA 13 nvrtc via NVIDIA apt repo"
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq wget gnupg ca-certificates >/dev/null 2>&1
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    || { echo "FATAL: cuda-keyring download failed"; exit 6; }
dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq cuda-nvrtc-13-0 \
    || { echo "FATAL: apt install cuda-nvrtc-13-0 failed"; exit 6; }

# Trace every command from here so we can pinpoint silent failures.
set -x

# Use dpkg -L to get the package's installed-file list (no full-FS find).
NVRTC_BUILTINS=$(dpkg -L cuda-nvrtc-13-0 | grep 'libnvrtc-builtins.so.13' | head -1 || true)
if [[ -z "$NVRTC_BUILTINS" ]]; then
    echo "FATAL: dpkg -L did not list libnvrtc-builtins.so.13 — fallback file listing:"
    dpkg -L cuda-nvrtc-13-0 | grep -E '\.so' || true
    exit 5
fi
NVRTC_LIB_DIR=$(dirname "$NVRTC_BUILTINS")
export LD_LIBRARY_PATH="$NVRTC_LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "==> NVRTC_LIB_DIR=$NVRTC_LIB_DIR"
echo "==> nvrtc-builtins file: $NVRTC_BUILTINS"
ls -la "$NVRTC_LIB_DIR" | grep nvrtc | head

set +x

echo "==> nvidia-smi"
nvidia-smi -L
which boltzgen || { echo "FATAL: boltzgen CLI not on PATH"; exit 3; }

# Pre-fetch all model weights so per-face runs don't race on first download
# and so a download failure surfaces here (before we burn GPU time).
echo "==> boltzgen download all"
boltzgen download all --cache "$BOLTZGEN_CACHE" \
    || { echo "FATAL: boltzgen weight download failed"; exit 4; }

cd "$REPO"
echo "==> running BoltzGen runner (num_designs=${BOLTZGEN_NUM_DESIGNS:-2})"
python -m twistr.dev.tools.runpod.boltzgen.run_boltzgen \
    --specs-dir /workspace/boltzgen_specs \
    --outputs-dir /workspace/boltzgen_outputs \
    --num-designs "${BOLTZGEN_NUM_DESIGNS:-2}" \
    --cache-dir "$BOLTZGEN_CACHE" || {
        echo "==> runner failed; dumping per-face logs"
        for f in /workspace/boltzgen_status/*.log; do
            echo "----- $f -----"
            tail -100 "$f" || true
        done
        exit 1
    }
