"""Mac-side orchestrator for the OOM-probe RunPod job.

Tries to allocate an A100 80GB PCIe pod; falls back to A100 80GB SXM if
PCIe is out of stock. Tars the working tree (excluding bulky data and
submodules), uploads via ssh, runs `python -m twistr.ml.training.probe`,
rsyncs `.cache/batch_calibration.json` back into the local repo, then
terminates the pod (even on failure / Ctrl-C).

Prereqs:
  - runpodctl on PATH and authed (sanity-checked via `runpodctl me`)
  - tar, ssh, scp on PATH
  - ~/.runpod/ssh/RunPod-Key-Go (auto-created by runpodctl)
  - <examples_root>/.lengths.json populated locally — run
      `python -m twistr.ml.training.probe --compute-lengths-only --config config_ml.yaml`
    first if missing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNPOD_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
BOOTSTRAP = HERE / "bootstrap.sh"
DEFAULT_CONFIG = REPO_ROOT / "config_ml.yaml"
LOCAL_CACHE = REPO_ROOT / ".cache" / "batch_calibration.json"

# A100 PCIe preferred; SXM fallback if PCIe is out of stock. RunPod's GPU
# IDs occasionally rename — check `runpodctl pod list-gpus` if both slots
# return out-of-stock and the catalog has different names.
GPU_PREFERENCES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 30
CLOUD_TYPE = "SECURE"

# Path-anchored excludes (matched against the FIRST component of the entry's
# arcname). Anything in this set is dropped along with its entire subtree.
TOP_LEVEL_SKIP = {
    ".git", ".venv", "wandb", "logs", ".pytest_cache",
    ".cache",  # local-only; the pod will write its own .cache/batch_calibration.json
    "dist", "build",
}
# Skipped wherever they appear (cached/derived/OS junk).
ANYWHERE_SKIP = {"__pycache__", ".DS_Store"}
SUFFIX_SKIP = {".pyc"}

# `data/` is otherwise excluded entirely, but the probe needs three small
# files inside it: the manifest, the cluster weights, and the lengths
# sidecar. Everything else under data/ — npz examples, raw module1/module2
# files, logs — is dropped.
DATA_KEEP = {
    "data/module3/module3_manifest.parquet",
    "data/module3/helix_clusters.parquet",
    "data/module3/.lengths.json",
}

# Per-path-prefix excludes. Protenix and FAFE under `twistr/external/` are
# git submodules and ARE needed at module-import time (chi-angle constants
# are loaded from Protenix in `twistr/ml/features/chi_angles.py`), so don't
# add them here. `tests/` is omitted to keep the upload minimal — the probe
# never runs pytest.
PATH_PREFIX_SKIP = {
    "tests/",
}

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def _check_prereqs(config_path: Path) -> Path:
    for tool in ("runpodctl", "ssh", "scp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    _run(["runpodctl", "me"])
    for p in (RUNPOD_SSH_KEY, BOOTSTRAP, config_path):
        if not p.exists():
            sys.exit(f"missing prereq: {p}")

    cfg = yaml.safe_load(config_path.read_text())
    examples_root = REPO_ROOT / cfg["examples_root"]
    lengths_sidecar = examples_root / ".lengths.json"
    if not lengths_sidecar.exists():
        sys.exit(
            f"missing lengths sidecar: {lengths_sidecar}\n"
            "Run this first to populate it:\n"
            "  .venv/bin/python -m twistr.ml.training.probe "
            f"--compute-lengths-only --config {config_path}"
        )
    return lengths_sidecar


def _try_create_pod(gpu_id: str) -> dict | None:
    """Try to create a pod with the given GPU id. Returns the pod info dict
    on success, or None if the GPU type is unavailable / out of stock. Other
    failures raise."""
    name = f"twistr-oom-probe-{int(time.time())}"
    try:
        out = _run([
            "runpodctl", "pod", "create",
            "--cloud-type", CLOUD_TYPE,
            "--gpu-id", gpu_id,
            "--image", IMAGE,
            "--container-disk-in-gb", str(CONTAINER_DISK_GB),
            "--ports", "22/tcp",
            "--name", name,
        ])
    except subprocess.CalledProcessError as e:
        msg = ((e.stdout or "") + (e.stderr or "")).lower()
        if any(tok in msg for tok in (
            "no longer", "out of stock", "unavailable",
            "no available", "no gpu", "insufficient",
        )):
            return None
        raise
    return json.loads(out.stdout)


def _create_pod() -> tuple[str, dict]:
    last_error: str | None = None
    for gpu_id in GPU_PREFERENCES:
        print(f"trying GPU: {gpu_id}", flush=True)
        try:
            info = _try_create_pod(gpu_id)
        except subprocess.CalledProcessError as e:
            last_error = f"{gpu_id}: {(e.stderr or e.stdout or '').strip()}"
            print(f"  hard failure on {gpu_id}: {last_error}", flush=True)
            continue
        if info is None:
            print(f"  {gpu_id} not available; trying next", flush=True)
            continue
        machine = info.get("machine", {})
        print(
            f"pod {info['id']} created on {machine.get('gpuDisplayName', gpu_id)} "
            f"({machine.get('location', '?')}) at ${info['costPerHr']}/hr",
            flush=True,
        )
        return info["id"], info
    sys.exit(
        "no GPU from preference list could be allocated. "
        f"Last error: {last_error or 'all entries returned out-of-stock'}"
    )


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


def _tar_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Path-anchored exclude. The arcname has no leading `./`. Tested against:
      - first path component in TOP_LEVEL_SKIP  → drop subtree
      - any path component in ANYWHERE_SKIP    → drop entry
      - basename ending in any SUFFIX_SKIP     → drop entry
      - `data/` entries except those in DATA_KEEP → drop
      - any path beginning with a PATH_PREFIX_SKIP entry → drop subtree
    """
    name = tar_info.name
    parts = name.split("/")
    if parts and parts[0] in TOP_LEVEL_SKIP:
        return None
    if any(p in ANYWHERE_SKIP for p in parts):
        return None
    if any(name.endswith(s) for s in SUFFIX_SKIP):
        return None
    if any(name.startswith(prefix) for prefix in PATH_PREFIX_SKIP):
        return None
    if parts[0] == "data":
        # Allow the directory entries leading to a kept file so tar can
        # recreate the path, but block everything else under data/.
        if name in DATA_KEEP:
            return tar_info
        if tar_info.isdir() and any(
            kept.startswith(name + "/") for kept in DATA_KEEP
        ):
            return tar_info
        return None
    return tar_info


def _upload_tree(host: str, port: int, src: Path, dst: str) -> None:
    """Stream a Python-built tarball through ssh into `tar -xz` on the pod.
    Python's tarfile module gives path-anchored excludes (bsdtar's --exclude
    is unanchored and that's exactly the trap that ate 50 GB of npz files
    on the first run)."""
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
        if ssh.stdin is not None:
            ssh.stdin.close()
    rc = ssh.wait()
    if rc != 0:
        sys.exit(f"upload failed (ssh exit {rc})")


def _ssh_exec_streaming(host: str, port: int, remote_cmd: str) -> int:
    return subprocess.call([
        "ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
        f"root@{host}", remote_cmd,
    ])


def _scp_pull(host: str, port: int, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([
        "scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
        f"root@{host}:{remote}", str(local),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARNING: failed to pull {remote}: {r.stderr.strip()}")
        return False
    return True


def _terminate(pod_id: str) -> None:
    print(f"terminating pod {pod_id}", flush=True)
    try:
        _run(["runpodctl", "pod", "delete", pod_id])
    except subprocess.CalledProcessError as e:
        print(f"WARNING: delete failed — verify with `runpodctl pod list`. {e}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    _check_prereqs(args.config)
    pod_id, _ = _create_pod()
    try:
        host, port = _ssh_endpoint(pod_id)
        print(f"ssh endpoint: {host}:{port} — waiting for sshd", flush=True)
        _wait_for_sshd(host, port)
        print(f"uploading working tree {REPO_ROOT} → /workspace/twistr (tarfile | ssh)", flush=True)
        _upload_tree(host, port, REPO_ROOT, "/workspace/twistr")
        rc = _ssh_exec_streaming(
            host, port,
            "bash /workspace/twistr/tools/oom_probe/bootstrap.sh",
        )
        if rc != 0:
            sys.exit(f"probe failed on pod (exit code {rc})")
        print(f"pulling /workspace/twistr/.cache/batch_calibration.json → {LOCAL_CACHE}", flush=True)
        if not _scp_pull(host, port, "/workspace/twistr/.cache/batch_calibration.json", LOCAL_CACHE):
            sys.exit("probe finished but cache pull failed")
        print(f"OOM probe complete; cache written to {LOCAL_CACHE}", flush=True)
    finally:
        _terminate(pod_id)


if __name__ == "__main__":
    main()
