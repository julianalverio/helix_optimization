"""Launch parallel RunPod pods for BoltzGen de novo binder design.

One pod per face (face1, face2). Tarball is tiny (<200 KB) — only
scripts + the renumbered target CIF + per-face spec YAML. BoltzGen
weights (~6 GB) are pulled by the pod itself from HuggingFace.

Two-phase usage:
  Smoke (2 designs/face):      python -m tools.runpod_boltzgen.launch_boltzgen --num-designs 2
  Production (50 designs/face): python -m tools.runpod_boltzgen.launch_boltzgen --num-designs 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.runpod_pxdesign.launch import (
    REPO_ROOT,
    RUNPOD_SSH_KEY,
    SSH_OPTS,
    _ssh_endpoint,
    _wait_for_sshd,
    _terminate,
    IMAGE,
    CONTAINER_DISK_GB,
    _NO_CAPACITY_TOKENS,
    _TRANSIENT_TOKENS,
    _CREATE_RETRY_MAX,
    _CREATE_RETRY_BACKOFF_SEC,
)


HERE = Path(__file__).resolve().parent
BOOTSTRAP = HERE / "boltzgen_bootstrap.sh"
SPEC_DIR_LOCAL = REPO_ROOT / "boltzgen_specs"
DEFAULT_GPU_PREFS = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
)


def _try_create_pod(gpu_id: str, cloud_type: str) -> dict | None:
    cmd = [
        "runpodctl", "pod", "create",
        "--cloud-type", cloud_type,
        "--gpu-id", gpu_id,
        "--image", IMAGE,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--ports", "22/tcp",
    ]
    last = ""
    for attempt in range(1, _CREATE_RETRY_MAX + 1):
        invocation = cmd + ["--name", f"twistr-boltzgen-{int(time.time())}"]
        proc = subprocess.run(invocation, text=True, capture_output=True)
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                sys.exit(f"could not parse pod create output: {proc.stdout!r}")
        last = ((proc.stdout or "") + (proc.stderr or "")).strip()
        msg_lc = last.lower()
        if any(tok in msg_lc for tok in _NO_CAPACITY_TOKENS):
            return None
        if any(tok in msg_lc for tok in _TRANSIENT_TOKENS) and attempt < _CREATE_RETRY_MAX:
            time.sleep(_CREATE_RETRY_BACKOFF_SEC * attempt)
            continue
        sys.exit(f"runpodctl pod create failed: {last.splitlines()[-1] if last else '<no output>'}")
    sys.exit(f"runpodctl pod create transient error after {_CREATE_RETRY_MAX} attempts: "
             f"{last.splitlines()[-1] if last else '<no output>'}")


def _build_tarball(dest: Path, face: str) -> None:
    """Pack tools/runpod_boltzgen + the renumbered target CIF + the
    per-face spec YAML. Per-pod tarball: only the spec for *this* face."""
    target_cif = SPEC_DIR_LOCAL / "3erd_chainA_renumbered.cif"
    spec_yaml = SPEC_DIR_LOCAL / f"{face}.yaml"
    if not target_cif.is_file():
        sys.exit(f"missing renumbered target CIF: {target_cif}")
    if not spec_yaml.is_file():
        sys.exit(f"missing spec YAML: {spec_yaml}")

    print(f"[{face}] building tarball", flush=True)
    with tarfile.open(dest, "w:gz") as tar:
        # scripts
        scripts = REPO_ROOT / "tools/runpod_boltzgen"
        tar.add(scripts, arcname="tools/runpod_boltzgen",
                filter=lambda ti: None
                if any(p in ("__pycache__", ".DS_Store") for p in ti.name.split("/"))
                or ti.name.endswith(".pyc")
                else ti)
        # tools/__init__.py — needed for `python -m tools.runpod_boltzgen.run_boltzgen`
        tools_init = REPO_ROOT / "tools" / "__init__.py"
        if tools_init.is_file():
            tar.add(tools_init, arcname="tools/__init__.py")
        twistr_init = REPO_ROOT / "twistr" / "__init__.py"
        if twistr_init.is_file():
            tar.add(twistr_init, arcname="twistr/__init__.py")
        # renumbered target CIF (path matches what the spec YAML embeds)
        tar.add(target_cif, arcname=str(target_cif.relative_to(REPO_ROOT)))
        # per-face spec — pod sees it under boltzgen_specs/<face>.yaml inside REPO,
        # but the runner reads from /workspace/boltzgen_specs (separate dir),
        # so we extract a copy there too via the bootstrap step (see _send_specs).
        tar.add(spec_yaml, arcname=str(spec_yaml.relative_to(REPO_ROOT)))
    size_kb = dest.stat().st_size / 1024
    print(f"[{face}] tarball ready: {size_kb:.1f} KB", flush=True)


def _send_tarball(host: str, port: int, tarball: Path, face: str,
                  max_attempts: int = 4) -> None:
    print(f"[{face}] uploading tarball ({tarball.stat().st_size / 1024:.1f} KB) "
          f"→ {host}:{port}", flush=True)
    rsync_bin = "/opt/homebrew/bin/rsync" if Path("/opt/homebrew/bin/rsync").exists() else "rsync"
    install_rsync = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}",
         "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1"],
        capture_output=True, text=True, timeout=120,
    )
    if install_rsync.returncode != 0:
        print(f"[{face}] rsync install on pod failed: "
              f"{(install_rsync.stderr or install_rsync.stdout).strip()[:200]}", flush=True)
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 "
        f"-o ServerAliveCountMax=3 -i {RUNPOD_SSH_KEY} -p {port}"
    )
    for attempt in range(1, max_attempts + 1):
        rsync = subprocess.run(
            [rsync_bin, "--partial", "--inplace", "--append-verify",
             "--no-perms", "--no-owner", "--no-group",
             "-e", ssh_cmd, str(tarball), f"root@{host}:/tmp/twistr_upload.tgz"],
            capture_output=True, text=True,
        )
        if rsync.returncode == 0:
            break
        msg = (rsync.stderr or rsync.stdout or "").strip().splitlines()[-3:]
        print(f"[{face}] rsync attempt {attempt}/{max_attempts} failed: {' | '.join(msg)}",
              flush=True)
        if attempt == max_attempts:
            sys.exit(f"[{face}] tarball rsync failed")
        time.sleep(min(60, 10 * attempt))
    extract = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}",
         "mkdir -p /workspace/twistr && tar -xzC /workspace/twistr -f /tmp/twistr_upload.tgz "
         "&& rm /tmp/twistr_upload.tgz "
         "&& mkdir -p /workspace/boltzgen_specs "
         "&& cp /workspace/twistr/boltzgen_specs/*.yaml /workspace/boltzgen_specs/"],
        capture_output=True, text=True,
    )
    if extract.returncode != 0:
        sys.exit(f"[{face}] extract failed: {(extract.stderr or extract.stdout).strip()}")
    print(f"[{face}] tarball uploaded and extracted", flush=True)


def _start_detached(host: str, port: int, face: str, num_designs: int) -> None:
    log_path = "/workspace/boltzgen.log"
    sentinel = "/workspace/boltzgen.done"
    setup = f"""
set -eu
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y tmux
fi
rm -f {log_path} {sentinel}
mkdir -p /workspace/boltzgen_status /workspace/boltzgen_outputs
tmux new-session -d -s boltzgen \
    "BOLTZGEN_NUM_DESIGNS={num_designs} bash /workspace/twistr/tools/runpod_boltzgen/boltzgen_bootstrap.sh > {log_path} 2>&1; echo \\$? > {sentinel}"
echo "tmux session started"
"""
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", setup],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"[{face}] failed to start tmux: {(proc.stderr or proc.stdout).strip()}")


def _poll_status(host: str, port: int) -> tuple[str, list[str], list[str], str]:
    """Return (sentinel_text, ok_list, fail_list, last_log_line)."""
    cmd = (
        "cat /workspace/boltzgen.done 2>/dev/null; "
        "echo '---'; ls /workspace/boltzgen_status/*.ok 2>/dev/null | xargs -n1 -I{} basename {} .ok; "
        "echo '---'; ls /workspace/boltzgen_status/*.fail 2>/dev/null | xargs -n1 -I{} basename {} .fail; "
        "echo '---'; tail -1 /workspace/boltzgen.log 2>/dev/null | tr -d '\\r' | head -c 200"
    )
    r = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", cmd],
        capture_output=True, text=True, timeout=30,
    )
    parts = (r.stdout or "").split("---")
    sentinel = parts[0].strip() if len(parts) > 0 else ""
    oks = [s.strip() for s in (parts[1].splitlines() if len(parts) > 1 else []) if s.strip()]
    fails = [s.strip() for s in (parts[2].splitlines() if len(parts) > 2 else []) if s.strip()]
    tail = parts[3].strip() if len(parts) > 3 else ""
    return sentinel, oks, fails, tail


def _fetch_outputs(host: str, port: int, local_root: Path, face: str) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    print(f"[{face}] pulling /workspace/boltzgen_outputs → {local_root}", flush=True)
    subprocess.run(
        ["scp", *SSH_OPTS, "-r", "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/boltzgen_outputs/.", str(local_root)],
        capture_output=True, text=True,
    )
    subprocess.run(
        ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/boltzgen.log", str(local_root / f"{face}.boltzgen.log")],
        capture_output=True, text=True,
    )
    status_dir = local_root / "_status"
    status_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["scp", *SSH_OPTS, "-r", "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/boltzgen_status/.", str(status_dir)],
        capture_output=True, text=True,
    )


def _run_one_pod(face: str, gpu_prefs: tuple[str, ...], num_designs: int,
                 out_dir: Path, poll_interval: int, timeout_min: int,
                 stock_retry_min: int) -> bool:
    print(f"[{face}] allocating pod (num_designs={num_designs})", flush=True)
    pod_id = None
    try:
        deadline = time.time() + stock_retry_min * 60
        attempt = 0
        while time.time() < deadline and pod_id is None:
            attempt += 1
            for cloud in ("SECURE", "COMMUNITY"):
                for gpu_id in gpu_prefs:
                    info = _try_create_pod(gpu_id, cloud)
                    if info is not None:
                        pod_id = info["id"]
                        machine = info.get("machine", {})
                        print(f"[{face}] pod {pod_id} on {machine.get('gpuDisplayName', gpu_id)} "
                              f"({cloud}) @ ${info['costPerHr']}/hr", flush=True)
                        break
                    print(f"[{face}] {cloud}/{gpu_id}: no stock (attempt {attempt})", flush=True)
                if pod_id is not None:
                    break
            if pod_id is None:
                remaining = int(deadline - time.time())
                if remaining <= 0:
                    break
                wait = min(60, remaining)
                print(f"[{face}] all GPUs stocked out; sleeping {wait}s "
                      f"(~{remaining // 60} min budget remaining)", flush=True)
                time.sleep(wait)
        if pod_id is None:
            print(f"[{face}] FAIL: no GPU after {stock_retry_min} min", flush=True)
            return False

        host, port = _ssh_endpoint(pod_id)
        print(f"[{face}] ssh {host}:{port}", flush=True)
        _wait_for_sshd(host, port)

        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tf:
            tarball = Path(tf.name)
        try:
            _build_tarball(tarball, face)
            _send_tarball(host, port, tarball, face)
        finally:
            tarball.unlink(missing_ok=True)

        _start_detached(host, port, face, num_designs)

        deadline = time.time() + timeout_min * 60
        last_summary = (-1, -1, "")
        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                sentinel, oks, fails, tail = _poll_status(host, port)
            except subprocess.TimeoutExpired:
                print(f"[{face}] poll: ssh timeout, retrying", flush=True)
                continue
            summary = (len(oks), len(fails), tail)
            if summary != last_summary:
                print(f"[{face}] poll: ok={len(oks)} fail={len(fails)} | tail: {tail[:160]}",
                      flush=True)
                last_summary = summary
            if sentinel.isdigit():
                rc = int(sentinel)
                print(f"[{face}] sentinel rc={rc}; ok={len(oks)} fail={len(fails)}", flush=True)
                _fetch_outputs(host, port, out_dir / face, face)
                return rc == 0 and len(fails) == 0
        print(f"[{face}] TIMEOUT after {timeout_min} min", flush=True)
        _fetch_outputs(host, port, out_dir / face, face)
        return False
    finally:
        if pod_id is not None:
            _terminate(pod_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-designs", type=int, required=True,
                        help="Designs per face (smoke=2, production=50)")
    parser.add_argument("--faces", nargs="+", default=["face1", "face2"])
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "boltzgen_outputs")
    parser.add_argument("--gpu", nargs="+", default=list(DEFAULT_GPU_PREFS))
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    parser.add_argument("--timeout-min", type=int, default=90)
    parser.add_argument("--stock-retry-min", type=int, default=30)
    args = parser.parse_args()

    for tool in ("runpodctl", "ssh", "scp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    if not RUNPOD_SSH_KEY.exists():
        sys.exit(f"missing ssh key: {RUNPOD_SSH_KEY}")
    if not BOOTSTRAP.is_file():
        sys.exit(f"missing bootstrap: {BOOTSTRAP}")
    for face in args.faces:
        if not (SPEC_DIR_LOCAL / f"{face}.yaml").is_file():
            sys.exit(f"missing spec: {SPEC_DIR_LOCAL / f'{face}.yaml'} — "
                     f"run `python -m tools.runpod_boltzgen.build_specs ...` first")
    if not (SPEC_DIR_LOCAL / "3erd_chainA_renumbered.cif").is_file():
        sys.exit(f"missing renumbered target CIF in {SPEC_DIR_LOCAL}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpu_prefs = tuple(args.gpu)
    with ThreadPoolExecutor(max_workers=len(args.faces)) as pool:
        futures = {
            pool.submit(_run_one_pod, face, gpu_prefs, args.num_designs,
                        args.out_dir, args.poll_interval_sec, args.timeout_min,
                        args.stock_retry_min): face
            for face in args.faces
        }
        ok_count = 0
        for fut in as_completed(futures):
            face = futures[fut]
            try:
                ok = fut.result()
            except BaseException as e:
                ok = False
                print(f"[{face}] worker raised: {e}", flush=True)
            if ok:
                ok_count += 1
            print(f"[{face}] {'COMPLETED' if ok else 'FAILED'}", flush=True)
    print(f"==> {ok_count}/{len(args.faces)} pods succeeded; outputs at {args.out_dir}/",
          flush=True)
    sys.exit(0 if ok_count == len(args.faces) else 1)


if __name__ == "__main__":
    main()
