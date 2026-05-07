"""Launch a training job on RunPod.

PLACEHOLDER. RunPod credentials are not configured yet — this script's body
intentionally does not call the RunPod API. When credentials are ready:

  1. Set RUNPOD_API_KEY in the environment.
  2. Install the runpod SDK:  uv pip install runpod
  3. Implement launch_runpod() to:
       - filter RunPod's GPU catalog to PCIe form factor only (skip SXM/HBM3
         variants — they're 1.5–2× the price for the same memory and we don't
         need NVLink bandwidth for our workload)
       - allocate a pod with cfg.num_gpus GPU(s) of the chosen PCIe model
       - sync this repo + the relevant slice of data/ to the pod
       - run `python -m twistr.ml.training.train --config <remote-path>` remotely
       - stream logs back; pull checkpoints on completion

For now, calling this script raises with a message describing what would happen.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from twistr.ml.config import load_ml_config

# Cost guard: only PCIe-attached GPUs (e.g. "NVIDIA A100 80GB PCIe", "NVIDIA
# H100 PCIe"). Skip SXM / HBM3 / SXM4 variants — they cost ~1.5-2× more and
# the inter-GPU bandwidth advantage doesn't matter for our DDP-only workload.
GPU_INTERCONNECT = "PCIe"
DISALLOWED_INTERCONNECT_TOKENS = ("SXM", "SXM4", "SXM5", "HBM3")


def is_pcie_gpu_type(gpu_type_name: str) -> bool:
    """Return True iff a RunPod GPU type name is PCIe (not SXM/HBM3). Used by
    the launcher to filter the catalog before allocating a pod."""
    name = gpu_type_name.upper()
    if any(tok in name for tok in DISALLOWED_INTERCONNECT_TOKENS):
        return False
    return GPU_INTERCONNECT.upper() in name


def launch_runpod(config_path: Path) -> None:
    cfg = load_ml_config(config_path)
    if cfg.num_gpus == 0:
        raise ValueError(
            "num_gpus=0 means local CPU training; the RunPod launcher is for GPU jobs."
        )
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set — RunPod credentials are not configured yet.\n"
            "Once configured, this launcher will:\n"
            f"  - filter the GPU catalog to {GPU_INTERCONNECT}-only "
            f"(skip {'/'.join(DISALLOWED_INTERCONNECT_TOKENS)} variants to save cost)\n"
            f"  - allocate a pod with {cfg.num_gpus} GPU(s) of a PCIe model\n"
            "  - sync this repo + relevant data/\n"
            "  - run `python -m twistr.ml.training.train --config <path>` remotely\n"
            "  - stream logs back and pull artifacts on completion"
        )
    raise NotImplementedError(
        "RunPod launcher body not implemented. Fill in once credentials work. "
        "Use is_pcie_gpu_type() to filter the catalog before allocating."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_ml.yaml"))
    args = parser.parse_args()
    launch_runpod(args.config)


if __name__ == "__main__":
    main()
