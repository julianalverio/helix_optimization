"""Populate the PXDesign network volume with the conda env + model
weights so subsequent design runs skip the ~20 min cold-cache install.

Spins up one pod against `cfg.network_volume_id`, runs `bootstrap.sh`
with `PXD_SETUP_ONLY=1` (skips the design wrapper at the end), then
terminates. Cost: ~$0.50 one-time. After this completes, every future
`launch.py` pod that mounts the same volume starts in seconds instead
of paying the full install + weights download.

Usage:
    python -m twistr.dev.tools.runpod.pxdesign.prewarm --config runtime/configs/pxdesign.yaml

The config must define both `network_volume_id` and `data_center_id` —
otherwise there's no persistent target for the install.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from twistr.dev.tools.runpod.pxdesign.config import load_pxdesign_config
from twistr.dev.tools.runpod.pxdesign.launch import (
    _build_tarball,
    _check_prereqs,
    _create_pod,
    _send_tarball,
    _ssh_endpoint,
    _ssh_run_detached,
    _terminate,
    _wait_for_sshd,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_pxdesign_config(args.config)
    if not cfg.network_volume_id:
        sys.exit(
            "prewarm requires network_volume_id (and data_center_id) in the "
            "config — without a persistent volume the install isn't reused."
        )
    _check_prereqs()

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tarball = Path(tmp.name)
    try:
        _build_tarball(tarball)
        pod_id, gpu_id = _create_pod(cfg)
        try:
            host, port = _ssh_endpoint(pod_id)
            print(f"ssh endpoint: {host}:{port} — waiting for sshd", flush=True)
            _wait_for_sshd(host, port)
            _send_tarball(host, port, tarball, label="prewarm")
            rc = _ssh_run_detached(
                host, port,
                "PXD_SETUP_ONLY=1 bash /workspace/twistr/dev/tools/runpod/pxdesign/bootstrap.sh",
                label="prewarm",
                timeout_min=60,  # cold install + ~10 GB weights download
            )
            if rc != 0:
                sys.exit(f"prewarm bootstrap failed on {gpu_id} (exit {rc})")
            print(
                f"prewarm complete on {gpu_id} — volume {cfg.network_volume_id} "
                "is ready; future runs will skip env install + weights download.",
                flush=True,
            )
        finally:
            _terminate(pod_id)
    finally:
        tarball.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
