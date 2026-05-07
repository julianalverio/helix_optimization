"""Mac-side orchestrator for the RunPod ML smoke test.

Creates an A40 PCIe pod (secure cloud), rsyncs the local working tree (incl.
submodules and the data subset under tools/runpod_smoke_test/subset/) to the
pod, runs `python -m twistr.ml.training.train` for max_steps=2, streams logs
back, and terminates the pod (even on failure / Ctrl-C).

We rsync the working tree rather than `git clone` so the smoke test exercises
exactly what's on disk, not whatever is currently pushed to GitHub. (The
deploy key under .deploy_key is left in place for future use but unused here.)

Prereqs (the script checks all of these and bails fast if anything's missing):
  - runpodctl on PATH and authed (sanity-checked via `runpodctl me`)
  - rsync on PATH (system default on macOS is fine)
  - ~/.runpod/ssh/RunPod-Key-Go private key (auto-created by runpodctl)
  - tools/runpod_smoke_test/subset/ (run make_subset.py first)
  - WANDB_API_KEY in ~/.netrc under machine api.wandb.ai
"""
from __future__ import annotations

import json
import netrc
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNPOD_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
SUBSET_DIR = HERE / "subset"
BOOTSTRAP = HERE / "bootstrap.sh"
SMOKE_CONFIG = HERE / "config_smoke.yaml"

GPU_ID = "NVIDIA A40"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 30
CLOUD_TYPE = "SECURE"

# Top-level directories to skip (path-anchored). `data/` is the 50 GB pdb
# tree — must NOT match nested `data/` dirs like Protenix/protenix/data, which
# is exactly the trap bsdtar's unanchored --exclude fell into.
TOP_LEVEL_SKIP = {".git", ".venv", "wandb", "logs", "data", ".pytest_cache",
                  "dist", "build"}
# Skipped wherever they appear (cached/derived/OS junk).
ANYWHERE_SKIP = {"__pycache__", ".DS_Store"}
SUFFIX_SKIP = {".pyc"}

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def _wandb_key() -> str:
    n = netrc.netrc()
    auth = n.authenticators("api.wandb.ai")
    if not auth:
        sys.exit("WANDB_API_KEY not found: no entry for api.wandb.ai in ~/.netrc")
    return auth[2]


def _check_prereqs() -> None:
    for tool in ("runpodctl", "ssh"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    _run(["runpodctl", "me"])
    for p in (RUNPOD_SSH_KEY, BOOTSTRAP, SMOKE_CONFIG):
        if not p.exists():
            sys.exit(f"missing prereq: {p}")
    if not (SUBSET_DIR / "module3_manifest.parquet").exists():
        sys.exit(f"missing data subset — run: python {HERE / 'make_subset.py'}")
    _wandb_key()


def _create_pod() -> str:
    out = _run([
        "runpodctl", "pod", "create",
        "--cloud-type", CLOUD_TYPE,
        "--gpu-id", GPU_ID,
        "--image", IMAGE,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--ports", "22/tcp",
        "--name", f"twistr-smoke-{int(time.time())}",
    ])
    info = json.loads(out.stdout)
    print(f"pod {info['id']} created on {info.get('machine', {}).get('gpuDisplayName', '?')} "
          f"({info['machine'].get('location', '?')}) at ${info['costPerHr']}/hr")
    return info["id"]


def _ssh_endpoint(pod_id: str) -> tuple[str, int]:
    """Poll runpodctl ssh info until it returns ip + port (was 'pod not ready')."""
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


def _tar_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Path-anchored exclude: skip top-level data/.git/etc. by exact basename
    match on the FIRST path component, plus __pycache__/*.pyc anywhere."""
    parts = tar_info.name.split("/")
    if parts and parts[0] in TOP_LEVEL_SKIP:
        return None
    if any(p in ANYWHERE_SKIP for p in parts):
        return None
    if any(parts[-1].endswith(s) for s in SUFFIX_SKIP):
        return None
    return tar_info


def _upload_tree(host: str, port: int, src: Path, dst: str) -> None:
    """Stream a Python-built tarball through ssh into `tar -x` on the pod. Uses
    Python's tarfile module so we get path-anchored excludes (bsdtar's
    --exclude is unanchored and was eating Protenix/protenix/data/)."""
    ssh = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", f"mkdir -p {dst} && tar -xzC {dst}"],
        stdin=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=ssh.stdin, mode="w|gz") as tar:
            for entry in sorted(src.iterdir()):
                tar.add(entry, arcname=entry.name, filter=_tar_filter)
    finally:
        ssh.stdin.close()
    rc = ssh.wait()
    if rc != 0:
        sys.exit(f"upload failed (ssh exit {rc})")


def _ssh_exec_streaming(host: str, port: int, env: dict[str, str], remote_cmd: str) -> int:
    env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
    full = f"{env_prefix} {remote_cmd}" if env_prefix else remote_cmd
    return subprocess.call([
        "ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
        f"root@{host}", full,
    ])


def _terminate(pod_id: str) -> None:
    print(f"terminating pod {pod_id}")
    try:
        _run(["runpodctl", "pod", "delete", pod_id])
    except subprocess.CalledProcessError as e:
        print(f"WARNING: delete failed — verify manually with `runpodctl pod list`. {e}")


def main() -> None:
    _check_prereqs()
    wandb_key = _wandb_key()
    pod_id = _create_pod()
    try:
        host, port = _ssh_endpoint(pod_id)
        print(f"ssh endpoint: {host}:{port} — waiting for sshd")
        _wait_for_sshd(host, port)
        print(f"uploading working tree {REPO_ROOT} → /workspace/twistr (tar | ssh)")
        _upload_tree(host, port, REPO_ROOT, "/workspace/twistr")
        rc = _ssh_exec_streaming(
            host, port,
            env={"WANDB_API_KEY": wandb_key},
            remote_cmd="bash /workspace/twistr/tools/runpod_smoke_test/bootstrap.sh",
        )
        if rc != 0:
            sys.exit(f"smoke test failed on pod (exit code {rc})")
        print("smoke test PASSED")
    finally:
        _terminate(pod_id)


if __name__ == "__main__":
    main()
