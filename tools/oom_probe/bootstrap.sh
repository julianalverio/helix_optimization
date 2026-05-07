#!/usr/bin/env bash
# Pod-side bootstrap for the OOM probe. The repo (incl. the lengths sidecar
# and the manifest/cluster parquets — but NOT the bulky npz examples) is
# already rsync'd to /workspace/twistr by tools/oom_probe/launch.py.
set -euo pipefail

cd /workspace/twistr

echo "==> installing twistr[ml]"
pip install --quiet --upgrade pip
pip install --quiet -e ".[ml]"

echo "==> nvidia-smi"
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "==> running OOM probe (writes .cache/batch_calibration.json)"
python -m twistr.ml.training.probe --config config_ml.yaml

echo "==> probe complete"
