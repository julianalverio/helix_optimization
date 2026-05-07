#!/usr/bin/env bash
# Pod-side bootstrap. Runs INSIDE a detached tmux session started by
# tools/runpod_train/launch.py. Repo + data/module3/ already uploaded.
#
# Env vars set by launch.py via /tmp/run_train.sh wrapper:
#   WANDB_API_KEY  — wandb.init() picks this up; no `wandb login` needed
#   RUNPOD_API_KEY — used to self-terminate the pod after training exits
#   RUNPOD_POD_ID  — this pod's ID (target of the terminate mutation)
#
# Self-termination: regardless of whether training succeeds or crashes,
# we POST a podTerminate mutation to RunPod's GraphQL API after training.
# The pod dies, sshd drops the launcher's tail-F, and the user sees a
# clean end to the log stream.
#
# We deliberately do NOT use `set -e` — training failure should still
# trigger termination. Instead we capture the training exit code and
# always reach the terminate block.
set -uo pipefail

cd /workspace/twistr

# Dataset lives on a persistent network volume mounted at /data (see
# launch.py: NETWORK_VOLUME_ID / VOLUME_MOUNT_PATH). Symlink it into the
# source tree so config-relative paths like `data/module3/...` resolve.
ln -sfn /data data

echo "==> installing twistr[ml]"
pip install --quiet --upgrade pip
pip install -e ".[ml]"

echo "==> nvidia-smi"
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "==> running training (config_ml.yaml)"
python -m twistr.ml.training.train --config config_ml.yaml
TRAIN_EXIT=$?
echo "==> training exited with code $TRAIN_EXIT"

echo "==> self-terminating pod $RUNPOD_POD_ID"
curl -s -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
  https://api.runpod.io/graphql
echo
