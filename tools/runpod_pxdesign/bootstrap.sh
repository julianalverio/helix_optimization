#!/usr/bin/env bash
# Pod-side bootstrap. Sets up PXDesign's conda env and weights (cached on
# /workspace/cache when a network volume is attached), then runs the
# wrapper. Exits when the wrapper does — the launcher streams stdout,
# scp's results back, then terminates the pod via try/finally.
#
# Caching strategy:
#   - micromamba lives at $CACHE_DIR/bin/micromamba (one curl)
#   - the conda env lives at $CACHE_DIR/micromamba/envs/pxdesign (set
#     via MAMBA_ROOT_PREFIX so install.sh writes through to the volume)
#   - tool_weights/ and release_data/ are symlinked from $PXD into
#     $CACHE_DIR/ so download_tool_weights.sh writes through too
#   - CUTLASS_PATH points at $CACHE_DIR/cutlass
#   - PROTENIX_DATA_ROOT_DIR points at $CACHE_DIR/release_data
#
# When CACHE_DIR is the container disk (no volume), every artifact is
# rebuilt; when it's a network volume, the markers + content persist.
set -euo pipefail

REPO="/workspace/twistr"
PXD="$REPO/twistr/external/PXDesign"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
mkdir -p "$CACHE_DIR/bin" "$CACHE_DIR/tool_weights" "$CACHE_DIR/release_data"

ENV_MARKER="$CACHE_DIR/pxdesign-env-ready"
WEIGHTS_MARKER="$CACHE_DIR/pxdesign-weights-ready"
MICROMAMBA_BIN="$CACHE_DIR/bin/micromamba"

export MAMBA_ROOT_PREFIX="$CACHE_DIR/micromamba"
export CUTLASS_PATH="$CACHE_DIR/cutlass"
export PROTENIX_DATA_ROOT_DIR="$CACHE_DIR/release_data"

# Symlink PXDesign's expected data dirs into the cache so install.sh and
# download_tool_weights.sh write through to the persistent volume. -n on
# ln avoids creating a nested link if the target already exists.
ln -sfn "$CACHE_DIR/tool_weights" "$PXD/tool_weights"
ln -sfn "$CACHE_DIR/release_data" "$PXD/release_data"

# 1. micromamba (one curl, cached on volume)
if [[ ! -x "$MICROMAMBA_BIN" ]]; then
    echo "==> installing micromamba"
    curl -sLo "$MICROMAMBA_BIN" \
        https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
    chmod +x "$MICROMAMBA_BIN"
fi
export PATH="$CACHE_DIR/bin:$PATH"
eval "$(micromamba shell hook -s bash)"

# 2. PXDesign conda env (~10-20 min first time, instant on cache hit)
if [[ ! -f "$ENV_MARKER" ]]; then
    # Self-heal a partial install left over from a prior failed prewarm:
    # if the env dir exists without a marker, it's incomplete and would
    # cause install.sh's `micromamba create` to error on duplicate name.
    if [[ -d "$MAMBA_ROOT_PREFIX/envs/pxdesign" ]]; then
        echo "==> partial env detected without marker — cleaning before fresh install"
        rm -rf "$MAMBA_ROOT_PREFIX/envs/pxdesign"
    fi
    echo "==> creating PXDesign env (~10-20 min, cached)"
    cd "$PXD"
    # CUDA 12.1 wheels (cu121): PXDesign pins torch==2.3.1; pytorch.org only
    # serves that pin under cu121 (cu124 was withdrawn after pytorch >= 2.4).
    # cu121 binaries run fine on the 12.4 driver via NVIDIA forward compat.
    bash install.sh --pkg_manager micromamba --cuda-version 12.1 --env pxdesign \
        || { echo "==> install.sh failed"; exit 1; }
    touch "$ENV_MARKER"
else
    echo "==> reusing cached PXDesign env from $MAMBA_ROOT_PREFIX"
fi

micromamba activate pxdesign

# 2b. Downgrade deepspeed: install.sh installs deepspeed unpinned (>=0.15.1),
# but recent versions (>=0.16) use torch.library.custom_op which only exists
# in torch >=2.4. PXDesign pins torch==2.3.1 → AttributeError at import.
# Pin to deepspeed<0.16 to stay compatible.
DEEPSPEED_FIX_MARKER="$CACHE_DIR/pxdesign-deepspeed-pinned"
if [[ ! -f "$DEEPSPEED_FIX_MARKER" ]]; then
    echo "==> pinning deepspeed<0.16 for torch 2.3.1 compatibility"
    pip install --no-cache-dir 'deepspeed<0.16' \
        || { echo "==> deepspeed pin failed"; exit 1; }
    touch "$DEEPSPEED_FIX_MARKER"
fi

# 3. Model weights (~10 GB, cached via symlink)
if [[ ! -f "$WEIGHTS_MARKER" ]]; then
    echo "==> downloading model weights (~10 GB, cached)"
    cd "$PXD"
    bash download_tool_weights.sh \
        || { echo "==> download_tool_weights.sh failed"; exit 1; }
    touch "$WEIGHTS_MARKER"
else
    echo "==> reusing cached model weights"
fi

# 4. Sanity + run
echo "==> nvidia-smi"
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Prewarm path: tools/runpod_pxdesign/prewarm.py sets PXD_SETUP_ONLY=1 to
# populate the network volume without paying for a design run.
if [[ "${PXD_SETUP_ONLY:-0}" == "1" ]]; then
    echo "==> setup complete (PXD_SETUP_ONLY=1, skipping design wrapper)"
    exit 0
fi

cd "$REPO"
echo "==> running PXDesign wrapper"
python -m tools.runpod_pxdesign.run_pxdesign --config config_pxdesign.yaml
