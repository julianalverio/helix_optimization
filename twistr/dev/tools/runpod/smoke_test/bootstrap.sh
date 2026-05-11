#!/usr/bin/env bash
# Pod-side bootstrap. The repo (incl. submodules + dev/tools/runpod/smoke_test/subset/
# + config_smoke.yaml) is already rsync'd to /workspace/twistr by launch.py.
# Environment: WANDB_API_KEY exported by the SSH command.
set -euo pipefail

cd /workspace/twistr

echo "==> installing twistr[ml]"
pip install --quiet --upgrade pip
pip install --quiet -e ".[ml]"

echo "==> nvidia-smi"
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "==> running smoke training (2 steps, batch_size=2)"
python -m twistr.pipeline.training.train --config dev/tools/runpod/smoke_test/config_smoke.yaml

echo "==> smoke test complete"
