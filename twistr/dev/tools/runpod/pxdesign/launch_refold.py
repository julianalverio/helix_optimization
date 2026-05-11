"""Launch parallel RunPod pods to run Protenix all-atom refolds on
PXDesign-designed binders. Bypasses PXDesign's filter pipeline (which
hangs on AF2 eval) — calls `protenix pred` directly per design.

Assumes:
  - Inputs already built via `build_refold_inputs.py`.
  - The pod will pip-install protenix and download Protenix weights from
    Bytedance CDN on first `pred` call (RunPod's pod-side network is fast).

Reuses pod-create / ssh / detached-run helpers from launch.py. Tarball
is very lean (~3-5 MB by default): just code + 2 a3m files per chain +
target CIF. Pass --include-protenix-cache to revert to the 2 GB cached-
weights tarball (slow upload from home internet; fallback only).

Usage:
  python -m twistr.dev.tools.runpod.pxdesign.launch_refold \\
      --inputs-dir runtime/outputs/refold_inputs \\
      --target runtime/data/pdb/3ERD.cif \\
      --network-volume-id <id> --data-center-id <dc> \\
      --n-parallel 4 \\
      --out-dir refold_outputs
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

from twistr.dev.tools.runpod.pxdesign.launch import (
    REPO_ROOT,
    RUNPOD_SSH_KEY,
    SSH_OPTS,
    MSA_CACHE_DIR,
    msa_cache_key,
    _ssh_endpoint,
    _wait_for_sshd,
    _terminate,
    _try_create_pod,
    IMAGE,
    CONTAINER_DISK_GB,
    _NO_CAPACITY_TOKENS,
    _TRANSIENT_TOKENS,
    _CREATE_RETRY_MAX,
    _CREATE_RETRY_BACKOFF_SEC,
)
from twistr.dev.tools.runpod.pxdesign.config import PXDesignConfig


def _try_create_pod_v2(cfg: PXDesignConfig, gpu_id: str, cloud_type: str,
                       use_volume: bool = True) -> dict | None:
    """Like launch._try_create_pod, but with explicit cloud_type ('SECURE' or
    'COMMUNITY') and an optional `use_volume=False` to skip the network-volume
    mount (frees us from the volume's DC pinning when stock is dry there)."""
    cmd = [
        "runpodctl", "pod", "create",
        "--cloud-type", cloud_type,
        "--gpu-id", gpu_id,
        "--image", IMAGE,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--ports", "22/tcp",
    ]
    if use_volume and cfg.network_volume_id:
        cmd += [
            "--network-volume-id", cfg.network_volume_id,
            "--volume-mount-path", "/workspace/cache",
            "--data-center-ids", cfg.data_center_id,
        ]
    last_msg = ""
    for attempt in range(1, _CREATE_RETRY_MAX + 1):
        invocation = cmd + ["--name", f"twistr-refold-{int(time.time())}"]
        proc = subprocess.run(invocation, text=True, capture_output=True)
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                sys.exit(f"could not parse pod create output: {proc.stdout!r}")
        last_msg = ((proc.stdout or "") + (proc.stderr or "")).strip()
        msg_lc = last_msg.lower()
        if any(tok in msg_lc for tok in _NO_CAPACITY_TOKENS):
            return None
        if any(tok in msg_lc for tok in _TRANSIENT_TOKENS) and attempt < _CREATE_RETRY_MAX:
            time.sleep(_CREATE_RETRY_BACKOFF_SEC * attempt)
            continue
        sys.exit(f"runpodctl pod create failed: {last_msg.splitlines()[-1] if last_msg else '<no output>'}")
    sys.exit(f"runpodctl pod create transient error after {_CREATE_RETRY_MAX} attempts: "
             f"{last_msg.splitlines()[-1] if last_msg else '<no output>'}")

REFOLD_BOOTSTRAP = REPO_ROOT / "dev/tools/runpod/pxdesign/refold_bootstrap.sh"

# Lean tarball: only what the pod needs to run `protenix pred`.
#   dev/tools/runpod/pxdesign/{run_protenix_refold.py, refold_bootstrap.sh, ...}
#   tools/__init__.py, twistr/__init__.py (for python -m twistr.dev.tools.runpod.pxdesign)
#   .cache/pxdesign_msa/<key>/ — the cached target MSA (referenced in JSONs)
#   runtime/data/pdb/<target> — the target file (only needed for sanity / reference)
#
# Per-pod input JSON shards are scp'd separately after extraction.
_TARBALL_INCLUDE = [
    "dev/tools/runpod/pxdesign",
]


def _build_lean_tarball(dest: Path, target_path: Path,
                        include_protenix_cache: bool = False) -> None:
    """Pack only the files the refold pod actually needs."""
    target_path = (REPO_ROOT / target_path).resolve() if not target_path.is_absolute() else target_path
    msa_dir = MSA_CACHE_DIR / msa_cache_key(target_path, "A")
    if not msa_dir.is_dir():
        sys.exit(f"missing MSA cache for {target_path}: {msa_dir}")

    protenix_cache = REPO_ROOT / ".cache" / "protenix"
    if include_protenix_cache:
        if not protenix_cache.is_dir():
            sys.exit(f"missing protenix cache at {protenix_cache}; pre-download first.")
        # sanity: confirm key files exist
        required = [
            protenix_cache / "common/components.cif",
            protenix_cache / "common/components.cif.rdkit_mol.pkl",
            protenix_cache / "checkpoint/protenix_base_default_v1.0.0.pt",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            sys.exit(f"protenix cache incomplete; missing: {[str(p) for p in missing]}")

    # tarfile gz is slow on multi-GB inputs; uncompressed tar is fine — model
    # weights are already incompressible and the wire is fast enough.
    print(f"building lean refold tarball from {REPO_ROOT}", flush=True)
    mode = "w:gz" if not include_protenix_cache else "w"
    with tarfile.open(dest, mode) as tar:
        for rel in _TARBALL_INCLUDE:
            p = REPO_ROOT / rel
            if not p.exists():
                sys.exit(f"missing tarball entry: {p}")
            tar.add(p, arcname=rel, filter=lambda ti: None
                    if any(part in ("__pycache__", ".DS_Store") for part in ti.name.split("/"))
                    or ti.name.endswith(".pyc")
                    else ti)
        twistr_init = REPO_ROOT / "twistr" / "__init__.py"
        if twistr_init.is_file():
            tar.add(twistr_init, arcname="twistr/__init__.py")
        # Only ship the two .a3m files Protenix actually reads; skip ~5 MB of
        # intermediate files our MSA builder leaves alongside (pdb70.m8,
        # uniref.a3m, bfd.*.a3m, msa.sh).
        for a3m in ("pairing.a3m", "non_pairing.a3m"):
            src = msa_dir / a3m
            if not src.is_file():
                sys.exit(f"missing required MSA file {src}")
            tar.add(src, arcname=str(src.relative_to(REPO_ROOT)))
        tar.add(target_path, arcname=str(target_path.relative_to(REPO_ROOT)))
        if include_protenix_cache:
            tar.add(protenix_cache, arcname=".cache/protenix")
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"refold tarball ready: {size_mb:.1f} MB", flush=True)


def _send_tarball(host: str, port: int, tarball: Path, label: str,
                  max_attempts: int = 6) -> None:
    """Upload via rsync with --partial --append-verify so a dropped connection
    resumes from the partial bytes already on the pod (the user's home
    internet is intermittent — see user_intermittent_internet memory)."""
    print(f"[{label}] uploading tarball ({tarball.stat().st_size / 1e9:.2f} GB) → {host}:{port}", flush=True)
    # rsync needs to be present on BOTH ends. Pod images may not include it.
    install_rsync = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}",
         "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1"],
        capture_output=True, text=True, timeout=120,
    )
    if install_rsync.returncode != 0:
        print(f"[{label}] rsync install on pod failed: "
              f"{(install_rsync.stderr or install_rsync.stdout).strip()[:200]}", flush=True)
    ssh_cmd = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 "
        f"-o ServerAliveCountMax=3 -i {RUNPOD_SSH_KEY} -p {port}"
    )
    # macOS ships openrsync (protocol 2.6.9) which lacks --append-verify and
    # --info=progress2. Prefer brew's modern rsync; fall back to plain rsync.
    rsync_bin = "/opt/homebrew/bin/rsync" if Path("/opt/homebrew/bin/rsync").exists() else "rsync"
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
        print(f"[{label}] rsync attempt {attempt}/{max_attempts} failed: {' | '.join(msg)}",
              flush=True)
        if attempt == max_attempts:
            sys.exit(f"[{label}] tarball rsync failed after {max_attempts} attempts")
        time.sleep(min(60, 10 * attempt))
    extract = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}",
         "mkdir -p /workspace/twistr && tar -xC /workspace/twistr -f /tmp/twistr_upload.tgz "
         "&& rm /tmp/twistr_upload.tgz"],
        capture_output=True, text=True,
    )
    if extract.returncode != 0:
        sys.exit(f"[{label}] tarball extract failed: {(extract.stderr or extract.stdout).strip()}")
    print(f"[{label}] tarball uploaded and extracted", flush=True)


def _send_inputs(host: str, port: int, json_paths: list[Path], label: str) -> None:
    """Ship this pod's assigned input JSONs to /workspace/refold_inputs/."""
    print(f"[{label}] shipping {len(json_paths)} input JSONs", flush=True)
    mk = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", "mkdir -p /workspace/refold_inputs /workspace/refold_outputs /workspace/refold_status"],
        capture_output=True, text=True,
    )
    if mk.returncode != 0:
        sys.exit(f"[{label}] mkdir failed: {(mk.stderr or mk.stdout).strip()}")
    scp = subprocess.run(
        ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         *[str(p) for p in json_paths],
         f"root@{host}:/workspace/refold_inputs/"],
        capture_output=True, text=True,
    )
    if scp.returncode != 0:
        sys.exit(f"[{label}] inputs scp failed: {(scp.stderr or scp.stdout).strip()}")


def _start_detached(host: str, port: int, label: str) -> None:
    """Kick off the refold bootstrap inside tmux. Doesn't wait — caller polls."""
    log_path = "/workspace/refold.log"
    sentinel = "/workspace/refold.done"
    setup = f"""
set -eu
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y tmux
fi
rm -f {log_path} {sentinel}
tmux new-session -d -s refold "bash /workspace/twistr/dev/tools/runpod/pxdesign/refold_bootstrap.sh > {log_path} 2>&1; echo \\$? > {sentinel}"
echo "tmux session started"
"""
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", setup],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"[{label}] failed to start tmux: {(proc.stderr or proc.stdout).strip()}")


def _poll_status(host: str, port: int) -> tuple[str, list[str], list[str]]:
    """Return (sentinel_text, ok_list, fail_list). sentinel_text is empty if not done."""
    cmd = (
        "cat /workspace/refold.done 2>/dev/null; "
        "echo '---'; ls /workspace/refold_status/*.ok 2>/dev/null | xargs -n1 -I{} basename {} .ok; "
        "echo '---'; ls /workspace/refold_status/*.fail 2>/dev/null | xargs -n1 -I{} basename {} .fail"
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
    return sentinel, oks, fails


def _fetch_outputs(host: str, port: int, local_root: Path, label: str) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] pulling /workspace/refold_outputs → {local_root}", flush=True)
    r = subprocess.run(
        ["scp", *SSH_OPTS, "-r", "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/refold_runtime/outputs/.", str(local_root)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[{label}] WARNING: outputs scp returned {r.returncode}: {r.stderr.strip()}", flush=True)
    # also pull the bootstrap log and per-design status logs for diagnosis
    subprocess.run(
        ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/refold.log", str(local_root / f"{label}.refold.log")],
        capture_output=True, text=True,
    )
    status_dir = local_root / "_status"
    status_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["scp", *SSH_OPTS, "-r", "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         f"root@{host}:/workspace/refold_status/.", str(status_dir)],
        capture_output=True, text=True,
    )


def _run_one_pod(label: str, json_paths: list[Path], tarball: Path,
                 cfg: PXDesignConfig, out_dir: Path,
                 poll_interval_sec: int, timeout_min: int,
                 stock_retry_min: int = 30,
                 use_volume: bool = True) -> bool:
    print(f"[{label}] allocating pod for {len(json_paths)} designs", flush=True)
    pod_id = None
    try:
        # Stockout retry loop. The volume is pinned to a DC so we can't migrate;
        # poll the GPU preferences every 60s for up to stock_retry_min minutes.
        deadline = time.time() + stock_retry_min * 60
        attempt = 0
        while time.time() < deadline and pod_id is None:
            attempt += 1
            for cloud in ("SECURE", "COMMUNITY"):
                for gpu_id in cfg.gpu_preferences:
                    info = _try_create_pod_v2(cfg, gpu_id, cloud, use_volume=use_volume)
                    if info is not None:
                        pod_id = info["id"]
                        machine = info.get("machine", {})
                        vol_tag = "+vol" if use_volume else "no-vol"
                        print(f"[{label}] pod {pod_id} on {machine.get('gpuDisplayName', gpu_id)} "
                              f"({cloud}, {vol_tag}) @ ${info['costPerHr']}/hr", flush=True)
                        break
                    print(f"[{label}] {cloud}/{gpu_id}: no stock (attempt {attempt})", flush=True)
                if pod_id is not None:
                    break
            if pod_id is None:
                remaining = int(deadline - time.time())
                if remaining <= 0:
                    break
                wait = min(60, remaining)
                print(f"[{label}] all GPUs stocked out; sleeping {wait}s before retry "
                      f"(~{remaining // 60} min budget remaining)", flush=True)
                time.sleep(wait)
        if pod_id is None:
            print(f"[{label}] FAIL: no GPU available after {stock_retry_min} min", flush=True)
            return False

        host, port = _ssh_endpoint(pod_id)
        print(f"[{label}] ssh {host}:{port}", flush=True)
        _wait_for_sshd(host, port)
        _send_tarball(host, port, tarball, label)
        _send_inputs(host, port, json_paths, label)
        _start_detached(host, port, label)

        deadline = time.time() + timeout_min * 60
        last_summary = (-1, -1)
        while time.time() < deadline:
            time.sleep(poll_interval_sec)
            try:
                sentinel, oks, fails = _poll_status(host, port)
            except subprocess.TimeoutExpired:
                print(f"[{label}] poll: ssh timeout, retrying", flush=True)
                continue
            summary = (len(oks), len(fails))
            if summary != last_summary:
                print(f"[{label}] poll: ok={len(oks)}/{len(json_paths)} fail={len(fails)}", flush=True)
                last_summary = summary
            if sentinel.isdigit():
                rc = int(sentinel)
                print(f"[{label}] sentinel rc={rc}; ok={len(oks)} fail={len(fails)}", flush=True)
                _fetch_outputs(host, port, out_dir / label, label)
                return rc == 0 and len(fails) == 0
        print(f"[{label}] TIMEOUT after {timeout_min} min", flush=True)
        _fetch_outputs(host, port, out_dir / label, label)
        return False
    finally:
        if pod_id is not None:
            _terminate(pod_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--network-volume-id", default=None,
                        help="Optional. When set, mount this volume (warm env path).")
    parser.add_argument("--data-center-id", default=None,
                        help="DC for the network volume; required when --network-volume-id is set.")
    parser.add_argument("--no-volume", action="store_true",
                        help="Don't pin to the volume's DC; install protenix fresh per pod. "
                             "Pays ~5 min/pod overhead but unblocks when the volume's DC is stocked out.")
    parser.add_argument("--n-parallel", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "refold_outputs")
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    parser.add_argument("--timeout-min", type=int, default=120)
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=["NVIDIA H100 80GB HBM3", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    )
    parser.add_argument(
        "--include-protenix-cache", action="store_true",
        help="Ship the local 2 GB Protenix weight cache in the tarball. "
             "Default: skip (pod downloads weights from Bytedance CDN). Use only "
             "as a fallback if Bytedance stalls.",
    )
    args = parser.parse_args()

    for tool in ("runpodctl", "ssh", "scp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    if not RUNPOD_SSH_KEY.exists():
        sys.exit(f"missing ssh key: {RUNPOD_SSH_KEY}")
    if not REFOLD_BOOTSTRAP.is_file():
        sys.exit(f"missing refold_bootstrap.sh at {REFOLD_BOOTSTRAP}")

    target_abs = args.target if args.target.is_absolute() else (REPO_ROOT / args.target).resolve()
    if not target_abs.is_file():
        sys.exit(f"target file not found: {target_abs}")

    inputs = sorted(args.inputs_dir.glob("*.json"))
    if not inputs:
        sys.exit(f"no input JSONs in {args.inputs_dir}")

    n = min(args.n_parallel, len(inputs))
    shards: list[list[Path]] = [[] for _ in range(n)]
    for i, p in enumerate(inputs):
        shards[i % n].append(p)
    print(f"sharding {len(inputs)} designs across {n} pods: {[len(s) for s in shards]}", flush=True)

    # Minimal cfg-shaped object for `_try_create_pod` (it reads gpu_preferences,
    # network_volume_id, data_center_id only).
    use_volume = not args.no_volume and args.network_volume_id is not None
    if use_volume and not args.data_center_id:
        sys.exit("--data-center-id required when --network-volume-id is set (use --no-volume to skip)")
    cfg = PXDesignConfig(
        binder_length=80,
        target=__import__("twistr.dev.tools.runpod.pxdesign.config", fromlist=["Target"]).Target(
            file=str(target_abs.relative_to(REPO_ROOT)), chains={"A": "all"},
        ),
        network_volume_id=args.network_volume_id if use_volume else None,
        data_center_id=args.data_center_id if use_volume else None,
        gpu_preferences=tuple(args.gpu),
    )

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tf:
        tarball = Path(tf.name)
    try:
        _build_lean_tarball(tarball, args.target,
                            include_protenix_cache=args.include_protenix_cache)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(_run_one_pod, f"pod{i}", shards[i], tarball,
                            cfg, args.out_dir,
                            args.poll_interval_sec, args.timeout_min,
                            30, use_volume): i
                for i in range(n)
            }
            ok_count = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                ok = fut.result()
                if ok:
                    ok_count += 1
                print(f"[pod{idx}] {'COMPLETED' if ok else 'FAILED'}", flush=True)
        print(f"==> {ok_count}/{n} pods succeeded; outputs at {args.out_dir}/", flush=True)
        sys.exit(0 if ok_count == n else 1)
    finally:
        tarball.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
