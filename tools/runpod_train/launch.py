"""Mac-side orchestrator for a full RunPod training run.

Pod lifecycle:
  - Tries GPU types in order (A100 PCIe → A100 SXM → H100); first one with
    capacity wins. Pod creation is pinned to the data center hosting the
    persistent dataset volume (see NETWORK_VOLUME_ID / DATA_CENTER_ID
    below), and the volume is attached at /data so the dataset is
    available without a per-run upload. bootstrap.sh symlinks /data into
    the source tree as data/.
  - Uploads the source tree (gzip stream, excluding data/) over SSH.
  - Starts training inside a detached `tmux` session on the pod, then
    streams the log file back. Disconnecting the launcher (Ctrl-C, SSH
    drop, laptop sleep) does NOT kill training — tmux owns it.
  - The pod self-terminates via the RunPod GraphQL API when training exits,
    success or failure (see bootstrap.sh). The launcher's try/finally is a
    safety net for the upload-phase failures only.

Auth:
  - WANDB_API_KEY: read from ~/.netrc (machine api.wandb.ai), passed to
    the pod env. wandb.init() picks it up — no `wandb login` step needed.
  - RUNPOD_API_KEY: read from ~/.runpod/config.toml (where runpodctl
    stores it) or the RUNPOD_API_KEY env var. Passed to the pod for
    self-termination.

The launcher highlights the wandb run URL as soon as it appears in the
streamed log, so you don't have to scroll through pip output to find it.
"""
from __future__ import annotations

import json
import netrc
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNPOD_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
BOOTSTRAP = HERE / "bootstrap.sh"
TRAIN_CONFIG = REPO_ROOT / "config_ml.yaml"

# Persistent network volume holding the dataset (data/module3/). Pod creation
# is pinned to the volume's data-center; bootstrap.sh symlinks the volume
# mount into the source tree so config-relative paths resolve. To replace the
# dataset, spin up a one-off CPU pod with this volume attached at /data and
# `tar | ssh` into /data — see tools/runpod_train/README in repo history.
NETWORK_VOLUME_ID = "mw9ymxpdwd"
DATA_CENTER_ID = "US-KS-2"
VOLUME_MOUNT_PATH = "/data"

# Ordered fallback list. Each is tried in turn; first one with capacity wins.
# Strings must match runpodctl's exact GPU IDs. All three are stocked in
# US-KS-2 (the volume's DC) per `runpodctl datacenter list`.
GPU_FALLBACKS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 60
CLOUD_TYPE = "SECURE"

TOP_LEVEL_SKIP = {".git", ".venv", ".venv-rosetta", ".cache", ".claude",
                  "wandb", "logs", "data", ".pytest_cache", ".ruff_cache",
                  "dist", "build", "node_modules"}
ANYWHERE_SKIP = {"__pycache__", ".DS_Store"}
SUFFIX_SKIP = {".pyc"}
# Specific deeper paths to skip. Each entry is a "/"-joined prefix; any tar
# entry whose path begins with one of these is dropped. Used for upstream
# packages that live in the repo for reference but aren't imported by the
# training pipeline. Kept (because read at module load):
#   - twistr/external/Protenix → chi_angles.py reads constants.py
#   - twistr/external/alphafold → sidechain.py reads residue_constants.py
PATH_PREFIX_SKIP = (
    "twistr/external/masif",
    "twistr/external/ScanNet",
    "twistr/external/PXDesign",
)

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
]

WANDB_URL_RE = re.compile(r"https://wandb\.ai/[\w./-]+/runs/[\w./-]+")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        sys.exit(f"{cmd[0]} failed (exit {e.returncode}): {msg or '<no output>'}")


def _wandb_key() -> str:
    n = netrc.netrc()
    auth = n.authenticators("api.wandb.ai")
    if not auth:
        sys.exit("WANDB_API_KEY not found: no entry for api.wandb.ai in ~/.netrc")
    return auth[2]


def _runpod_api_key() -> str:
    """Pod self-termination needs an API key. Prefer the env var; fall back
    to runpodctl's local config."""
    env = os.environ.get("RUNPOD_API_KEY")
    if env:
        return env
    if RUNPOD_CONFIG.exists():
        with open(RUNPOD_CONFIG, "rb") as f:
            cfg = tomllib.load(f)
        for key in ("apikey", "api_key", "apiKey"):
            if key in cfg:
                return cfg[key]
    sys.exit(
        "RUNPOD_API_KEY not found. Either export RUNPOD_API_KEY in your "
        f"shell or check {RUNPOD_CONFIG} for an apikey field."
    )


def _check_prereqs() -> None:
    for tool in ("runpodctl", "ssh"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    _run(["runpodctl", "me"])
    for p in (RUNPOD_SSH_KEY, BOOTSTRAP, TRAIN_CONFIG):
        if not p.exists():
            sys.exit(f"missing prereq: {p}")
    if RUNPOD_SSH_KEY.stat().st_mode & 0o077:
        sys.exit(f"{RUNPOD_SSH_KEY} permissions too open — run `chmod 600 {RUNPOD_SSH_KEY}`")
    _wandb_key()
    _runpod_api_key()


def _try_create(gpu_id: str) -> dict | None:
    proc = subprocess.run(
        ["runpodctl", "pod", "create",
         "--cloud-type", CLOUD_TYPE,
         "--gpu-id", gpu_id,
         "--image", IMAGE,
         "--container-disk-in-gb", str(CONTAINER_DISK_GB),
         "--ports", "22/tcp",
         "--data-center-ids", DATA_CENTER_ID,
         "--network-volume-id", NETWORK_VOLUME_ID,
         "--volume-mount-path", VOLUME_MOUNT_PATH,
         "--name", f"twistr-train-{int(time.time())}"],
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _create_pod() -> tuple[str, str]:
    """Try every GPU type in the fallback list; if all are out, sleep and
    retry. Capacity in a single data center can clear in seconds-to-
    minutes, and the volume pin (DATA_CENTER_ID) makes us wait there. The
    full sweep already takes ~30s due to per-call API latency, so a 60s
    inter-sweep sleep keeps us responsive without hammering the API."""
    sleep_seconds = 60
    while True:
        for gpu_id in GPU_FALLBACKS:
            print(f"trying {gpu_id}...")
            info = _try_create(gpu_id)
            if info is not None:
                print(f"pod {info['id']} created on {gpu_id} "
                      f"({info['machine'].get('location', '?')}) at ${info['costPerHr']}/hr")
                return info["id"], gpu_id
            print(f"  {gpu_id}: no capacity")
        print(f"all GPU types out of capacity in {DATA_CENTER_ID}; "
              f"sleeping {sleep_seconds}s before retry")
        time.sleep(sleep_seconds)


def _ssh_endpoint(pod_id: str) -> tuple[str, int]:
    deadline = time.time() + 300
    while time.time() < deadline:
        out = _run(["runpodctl", "ssh", "info", pod_id])
        info = json.loads(out.stdout)
        ip, port = info.get("ip"), info.get("port")
        if ip and port:
            return ip, int(port)
        time.sleep(5)
    sys.exit(f"timed out waiting for SSH endpoint on pod {pod_id}")


def _wait_for_sshd(host: str, port: int) -> None:
    deadline = time.time() + 300
    while time.time() < deadline:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
             f"root@{host}", "echo ready"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            return
        time.sleep(5)
    sys.exit(f"sshd never came up on {host}:{port}")


def _tree_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = tar_info.name.split("/")
    if parts and parts[0] in TOP_LEVEL_SKIP:
        return None
    if any(p in ANYWHERE_SKIP for p in parts):
        return None
    if any(parts[-1].endswith(s) for s in SUFFIX_SKIP):
        return None
    if any(tar_info.name == p or tar_info.name.startswith(p + "/")
           for p in PATH_PREFIX_SKIP):
        return None
    return tar_info


def _upload_source(host: str, port: int) -> None:
    print(f"uploading source tree {REPO_ROOT} → /workspace/twistr (tar | gzip | ssh)")
    ssh = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", "mkdir -p /workspace/twistr && tar -xzC /workspace/twistr"],
        stdin=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=ssh.stdin, mode="w|gz") as tar:
            for entry in sorted(REPO_ROOT.iterdir()):
                tar.add(entry, arcname=entry.name, filter=_tree_filter)
    finally:
        ssh.stdin.close()
    rc = ssh.wait()
    if rc != 0:
        sys.exit(f"source upload failed (ssh exit {rc})")


def _start_training_in_tmux(host: str, port: int, env: dict[str, str]) -> None:
    """Write a wrapper script on the pod that exports the auth env vars and
    runs the bootstrap, then start a detached tmux session that runs it.
    Tee'd into train.log so the launcher can stream-and-detach."""
    quoted = {k: shlex.quote(v) for k, v in env.items()}
    remote_cmd = f"""
set -e
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y tmux >/dev/null 2>&1
cat > /tmp/run_train.sh <<'EOF'
#!/usr/bin/env bash
export WANDB_API_KEY={quoted['WANDB_API_KEY']}
export RUNPOD_API_KEY={quoted['RUNPOD_API_KEY']}
export RUNPOD_POD_ID={quoted['RUNPOD_POD_ID']}
bash /workspace/twistr/tools/runpod_train/bootstrap.sh 2>&1 | tee /workspace/twistr/train.log
EOF
chmod +x /tmp/run_train.sh
: > /workspace/twistr/train.log
tmux new-session -d -s train /tmp/run_train.sh
"""
    rc = subprocess.call([
        "ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
        f"root@{host}", remote_cmd,
    ])
    if rc != 0:
        sys.exit(f"failed to start tmux session (ssh exit {rc})")


def _stream_logs(host: str, port: int) -> None:
    """tail -F train.log on the pod; highlight the wandb run URL the first
    time it appears. Ctrl-C detaches without killing training."""
    proc = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", "tail -F /workspace/twistr/train.log 2>/dev/null"],
        stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    url_seen = False
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not url_seen:
                m = WANDB_URL_RE.search(line)
                if m:
                    print(f"\n{'=' * 60}")
                    print(f"  WANDB RUN: {m.group(0)}")
                    print(f"{'=' * 60}\n")
                    url_seen = True
        print("\nlog stream ended — pod terminated or SSH dropped")
    except KeyboardInterrupt:
        print("\ndetached from log stream — training continues; "
              "pod will self-terminate when training exits")
    finally:
        proc.terminate()


def _terminate(pod_id: str) -> None:
    print(f"terminating pod {pod_id}")
    try:
        _run(["runpodctl", "pod", "delete", pod_id])
    except subprocess.CalledProcessError as e:
        print(f"WARNING: delete failed — verify manually with `runpodctl pod list`. {e}")


def main() -> None:
    _check_prereqs()
    wandb_key = _wandb_key()
    runpod_key = _runpod_api_key()
    pod_id, gpu_id = _create_pod()
    kicked_off = False
    try:
        host, port = _ssh_endpoint(pod_id)
        print(f"ssh endpoint: {host}:{port} — waiting for sshd")
        _wait_for_sshd(host, port)
        _upload_source(host, port)
        _start_training_in_tmux(host, port, env={
            "WANDB_API_KEY": wandb_key,
            "RUNPOD_API_KEY": runpod_key,
            "RUNPOD_POD_ID": pod_id,
        })
        kicked_off = True
        print(f"\n{'=' * 60}")
        print(f"training started in tmux session 'train' on pod {pod_id} ({gpu_id})")
        print(f"  attach to monitor: ssh -p {port} root@{host} 'tmux attach -t train'")
        print(f"  abort:             runpodctl pod delete {pod_id}")
        print(f"\nstreaming logs (Ctrl-C detaches; training keeps running; "
              f"pod self-terminates when training exits)...")
        print(f"{'=' * 60}\n")
        _stream_logs(host, port)
    finally:
        # Only clean up from the Mac side if we never reached the kickoff.
        # After kickoff, the pod owns its lifecycle (bootstrap.sh calls the
        # RunPod API to terminate itself when training exits).
        if not kicked_off:
            _terminate(pod_id)


if __name__ == "__main__":
    main()
