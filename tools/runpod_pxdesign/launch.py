"""Mac-side orchestrator for a PXDesign RunPod job.

Reads `config_pxdesign.yaml`, allocates a pod from `cfg.gpu_preferences`
(first available wins), ships the working tree (PXDesign submodule and
the `.cache/pxdesign_msa/` cache included), runs
`tools/runpod_pxdesign/bootstrap.sh`, fetches the high-value outputs
(`summary.csv` + `passing-*` folders), and terminates the pod via
try/finally — even on failure / Ctrl-C.

When `cfg.network_volume_id` is set, the volume mounts at
/workspace/cache so the conda env + ~10 GB of weights survive across
runs (~20 min first-run cost, instant on warm cache).

MSAs are NOT built here — the launcher pre-flight hard-errors if any
chain lacks both an explicit `msa:` and a cache hit. Build them first:
    python -m tools.runpod_pxdesign.build_msas <config>

Prereqs:
  - runpodctl on PATH and authed (sanity-checked via `runpodctl me`)
  - tar, ssh, scp on PATH
  - ~/.runpod/ssh/RunPod-Key-Go (auto-created by runpodctl)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile

import yaml
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from tools.runpod_pxdesign.config import PXDesignConfig, load_pxdesign_config

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNPOD_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
BOOTSTRAP = HERE / "bootstrap.sh"
DEFAULT_CONFIG = REPO_ROOT / "config_pxdesign.yaml"
MSA_CACHE_DIR = REPO_ROOT / ".cache" / "pxdesign_msa"

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 80
CLOUD_TYPE = "SECURE"

# `.cache/` and `data/` are otherwise excluded entirely; we keep specific
# subtrees the pod needs:
#   - `.cache/pxdesign_msa/`: cached MSAs the wrapper reads via msa_cache_key
#   - `data/pdb/`: target PDB/CIF files the wrapper reads to compute the
#     MSA cache key (cfg.target.file lives here)
TOP_LEVEL_SKIP = {".git", ".venv", ".venv-rosetta", "venv", "wandb", "logs",
                  "data", ".pytest_cache", "dist", "build", ".idea", ".vscode"}
CACHE_KEEP_PREFIX = ".cache/pxdesign_msa"
DATA_KEEP_PREFIX = "data/pdb"
ANYWHERE_SKIP = {"__pycache__", ".DS_Store"}
SUFFIX_SKIP = {".pyc"}

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",  # dead-connection detected in ~45s
]

# Captured pod stock-out / capacity errors from runpodctl. Anything matching
# these substrings means "try the next GPU"; everything else is a hard error.
_NO_CAPACITY_TOKENS = (
    "no longer", "out of stock", "unavailable",
    "no available", "no gpu", "insufficient",
)

# Transient RunPod-side errors that warrant an automatic retry on the same
# GPU (vs. falling through to the next GPU type). We've seen the generic
# "something went wrong" hit ~70% of pod-create requests during a brief
# API-side outage; retry-with-backoff lets the launcher self-heal instead
# of failing the worker.
_TRANSIENT_TOKENS = (
    "something went wrong",
    "internal server error",
    "service unavailable",
    "timeout",
    "502 bad gateway",
    "503",
)
_CREATE_RETRY_MAX = 4
_CREATE_RETRY_BACKOFF_SEC = 15


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        sys.exit(f"{cmd[0]} failed (exit {e.returncode}): {msg or '<no output>'}")


def msa_cache_key(target_file: Path, chain_id: str) -> str:
    """Stable hash of (target file bytes, chain id). Crop is intentionally
    excluded: PXDesign's MSA is built on the full-length target sequence
    even when the structure is cropped at design time (PXDesign README:
    "Why must the MSA correspond to the full-length sequence?")."""
    h = hashlib.sha256()
    h.update(target_file.read_bytes())
    h.update(b"\0")
    h.update(chain_id.encode())
    return h.hexdigest()[:16]


def _check_msa_cache(cfg: PXDesignConfig) -> None:
    """Hard-error if any chain that needs an MSA has neither an explicit
    msa: path nor a populated cache entry. Preview preset skips MSA, so
    this is a no-op there."""
    if cfg.preset == "preview":
        return
    target_path = REPO_ROOT / cfg.target.file
    if not target_path.is_file():
        sys.exit(f"target file not found: {target_path}")

    missing: list[tuple[str, str]] = []
    for cid, chain in cfg.target.chains.items():
        if isinstance(chain, str):
            continue
        if chain.msa is not None:
            if not (REPO_ROOT / chain.msa).is_dir():
                missing.append((cid, f"explicit msa path missing: {chain.msa}"))
            continue
        cache_path = MSA_CACHE_DIR / msa_cache_key(target_path, cid)
        if not (cache_path.is_dir() and any(cache_path.iterdir())):
            missing.append((cid, f"no cache entry at {cache_path.relative_to(REPO_ROOT)}"))

    if missing:
        details = "\n".join(f"  chain {c}: {reason}" for c, reason in missing)
        sys.exit(
            f"missing MSA(s):\n{details}\n"
            "Build them first:\n"
            f"  python -m tools.runpod_pxdesign.build_msas {DEFAULT_CONFIG.name}"
        )


def _check_prereqs() -> None:
    for tool in ("runpodctl", "ssh", "scp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    _run(["runpodctl", "me"])
    for p in (RUNPOD_SSH_KEY, BOOTSTRAP):
        if not p.exists():
            sys.exit(f"missing prereq: {p}")
    if RUNPOD_SSH_KEY.stat().st_mode & 0o077:
        sys.exit(f"{RUNPOD_SSH_KEY} permissions too open — run `chmod 600 {RUNPOD_SSH_KEY}`")
    pxd_dir = REPO_ROOT / "twistr" / "external" / "PXDesign"
    if not (pxd_dir / "install.sh").is_file():
        sys.exit(
            f"PXDesign submodule not initialised at {pxd_dir}. Run:\n"
            "  git submodule update --init twistr/external/PXDesign"
        )


def _try_create_pod(cfg: PXDesignConfig, gpu_id: str) -> dict | None:
    """Try to allocate a pod with `gpu_id`. Returns the pod info on success,
    None on stock-out (caller falls through to next GPU). Retries with
    exponential backoff on transient RunPod-API errors before raising."""
    cmd = [
        "runpodctl", "pod", "create",
        "--cloud-type", CLOUD_TYPE,
        "--gpu-id", gpu_id,
        "--image", IMAGE,
        "--container-disk-in-gb", str(CONTAINER_DISK_GB),
        "--ports", "22/tcp",
    ]
    if cfg.network_volume_id:
        cmd += [
            "--network-volume-id", cfg.network_volume_id,
            "--volume-mount-path", "/workspace/cache",
            "--data-center-ids", cfg.data_center_id,
        ]

    last_msg = ""
    for attempt in range(1, _CREATE_RETRY_MAX + 1):
        invocation = cmd + ["--name", f"twistr-pxdesign-{int(time.time())}"]
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
            sleep_sec = _CREATE_RETRY_BACKOFF_SEC * attempt
            print(
                f"transient runpod error on {gpu_id} (attempt {attempt}/{_CREATE_RETRY_MAX}); "
                f"retrying in {sleep_sec}s",
                flush=True,
            )
            time.sleep(sleep_sec)
            continue
        sys.exit(f"runpodctl pod create failed: {last_msg.splitlines()[-1] if last_msg else '<no output>'}")
    sys.exit(f"runpodctl pod create transient error after {_CREATE_RETRY_MAX} attempts: "
             f"{last_msg.splitlines()[-1] if last_msg else '<no output>'}")


def _create_pod(cfg: PXDesignConfig) -> tuple[str, str]:
    for gpu_id in cfg.gpu_preferences:
        print(f"trying GPU: {gpu_id}", flush=True)
        info = _try_create_pod(cfg, gpu_id)
        if info is not None:
            machine = info.get("machine", {})
            print(
                f"pod {info['id']} created on {machine.get('gpuDisplayName', gpu_id)} "
                f"({machine.get('location', '?')}) at ${info['costPerHr']}/hr",
                flush=True,
            )
            return info["id"], gpu_id
        print(f"  {gpu_id}: no capacity, trying next", flush=True)
    sys.exit(f"no GPU from preference list could be allocated: {list(cfg.gpu_preferences)}")


def _ssh_endpoint(pod_id: str) -> tuple[str, int]:
    """Poll runpodctl until the pod's SSH endpoint is provisioned. Tolerates
    transient runpodctl failures during the early-boot window where the pod
    exists but hasn't fully registered with the control plane yet."""
    deadline = time.time() + 600  # 10 min — some pods take 5-7 min to come up
    while time.time() < deadline:
        proc = subprocess.run(
            ["runpodctl", "ssh", "info", pod_id],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            try:
                info = json.loads(proc.stdout)
            except json.JSONDecodeError:
                info = {}
            ip, port = info.get("ip"), info.get("port")
            if ip and port:
                return ip, int(port)
        time.sleep(10)
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
    """Path-anchored exclude. The arcname has no leading `./`. `.cache/`
    and `data/` are dropped except for the specific subtrees the wrapper
    needs (cached MSAs + target PDB files)."""
    name = tar_info.name
    parts = name.split("/")
    if any(p in ANYWHERE_SKIP for p in parts):
        return None
    if any(name.endswith(s) for s in SUFFIX_SKIP):
        return None
    if parts[0] == ".cache":
        if name == ".cache" or name.startswith(CACHE_KEEP_PREFIX):
            return tar_info
        return None
    if parts[0] == "data":
        # Keep directory entries up to data/pdb/, plus any FILE directly in
        # data/pdb/ (i.e., data/pdb/<file>). We deliberately exclude
        # subdirectories under data/pdb/ — twistr's curation pipeline mirrors
        # the entire RCSB PDB there (~46 GB of <2char>/<id>.cif.gz files)
        # which we never need shipped to PXDesign pods.
        if name == "data" or name == "data/pdb":
            return tar_info
        if len(parts) == 3 and parts[1] == "pdb" and not tar_info.isdir():
            return tar_info
        return None
    if parts[0] in TOP_LEVEL_SKIP:
        return None
    return tar_info


def _build_tarball(dest: Path) -> None:
    """Pack the working tree (with `_tar_filter` exclusions) into `dest`.
    Built once per launcher invocation; reused across N parallel pods."""
    print(f"building tarball from {REPO_ROOT}", flush=True)
    with tarfile.open(dest, "w:gz") as tar:
        for entry in sorted(REPO_ROOT.iterdir()):
            tar.add(entry, arcname=entry.name, filter=_tar_filter)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"tarball ready: {size_mb:.1f} MB", flush=True)


def _send_tarball(host: str, port: int, tarball: Path, label: str = "") -> None:
    """scp `tarball` to the pod and extract into /workspace/twistr.
    Build-then-ship avoids the failure mode where a `tar | ssh stdin` pipe
    silently hangs for hours when SSH's underlying TCP connection dies —
    scp surfaces network failures cleanly via its own timeouts."""
    tag = f"[{label}] " if label else ""
    print(f"{tag}uploading → {host}:{port}", flush=True)
    scp = subprocess.run(
        ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
         str(tarball), f"root@{host}:/tmp/twistr_upload.tgz"],
        capture_output=True, text=True,
    )
    if scp.returncode != 0:
        sys.exit(f"{tag}scp upload failed: {(scp.stderr or scp.stdout).strip()}")
    extract = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}",
         "mkdir -p /workspace/twistr && tar -xzC /workspace/twistr -f /tmp/twistr_upload.tgz "
         "&& rm /tmp/twistr_upload.tgz"],
        capture_output=True, text=True,
    )
    if extract.returncode != 0:
        sys.exit(f"{tag}remote extract failed: {(extract.stderr or extract.stdout).strip()}")


def _ssh_run(host: str, port: int, remote_cmd: str) -> int:
    """Run a remote command with stdout streamed live. Returns the exit code."""
    return subprocess.call([
        "ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
        f"root@{host}", remote_cmd,
    ])


def _ssh_run_detached(host: str, port: int, remote_cmd: str, label: str = "",
                      timeout_min: int = 90) -> int:
    """Run `remote_cmd` in a detached tmux session on the pod, stream the
    log via tail -F (best-effort, may drop), and poll for a completion
    sentinel file. Resilient to SSH stream disconnects: bootstrap
    continues running on the pod even if the local SSH dies.

    Mirrors the pattern in tools/runpod_train/launch.py:223-249. Returns
    the remote command's exit code; sys.exits if the pod never writes
    the sentinel within `timeout_min`."""
    log_path = "/workspace/twistr/.bootstrap.log"
    sentinel = "/workspace/twistr/.bootstrap_done"
    tag = f"[{label}] " if label else ""

    # Install tmux only if missing, then start the work detached. The
    # session writes its exit code to `sentinel` so we can poll for
    # completion from a fresh SSH connection (no long-lived stream needed).
    setup = f"""
set -eu
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y tmux
fi
rm -f {log_path} {sentinel}
cat > /tmp/_pxdwork.sh <<'WORKERSH'
#!/usr/bin/env bash
{remote_cmd}
WORKERSH
chmod +x /tmp/_pxdwork.sh
tmux new-session -d -s pxdwork "/tmp/_pxdwork.sh > {log_path} 2>&1; echo \\$? > {sentinel}"
echo "tmux session started"
"""
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", setup],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            f"{tag}failed to start detached session (rc={proc.returncode})\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )

    # Stream log in a separate process — purely informational; if it dies
    # mid-run that's fine because the work is in tmux.
    tailer = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
         f"root@{host}", f"tail -F {log_path} 2>/dev/null"],
    )
    deadline = time.time() + timeout_min * 60
    rc: int | None = None
    try:
        while time.time() < deadline:
            time.sleep(20)
            r = subprocess.run(
                ["ssh", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-p", str(port),
                 f"root@{host}", f"cat {sentinel} 2>/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            text = r.stdout.strip()
            if text.isdigit():
                rc = int(text)
                break
        if rc is None:
            sys.exit(f"{tag}detached session did not finish within {timeout_min} min")
    finally:
        tailer.terminate()
        try:
            tailer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tailer.kill()

    # On failure, pull the full remote log so the caller has diagnostic info.
    # The streamed log via tail -F is best-effort and often truncated when
    # the session ends abruptly.
    if rc != 0:
        local_log = Path(f"/tmp/pxdwork_{label or 'run'}_{int(time.time())}.log")
        pull = subprocess.run(
            ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
             f"root@{host}:{log_path}", str(local_log)],
            capture_output=True, text=True,
        )
        if pull.returncode == 0 and local_log.exists():
            tail = local_log.read_text().splitlines()[-80:]
            print(f"{tag}--- remote log tail ({local_log}) ---", flush=True)
            for line in tail:
                print(f"  {line}", flush=True)
            print(f"{tag}--- end remote log ---", flush=True)
        else:
            print(f"{tag}WARNING: could not pull remote log: {(pull.stderr or pull.stdout).strip()}", flush=True)
    return rc


def _scp_pull(host: str, port: int, remote: str, local: Path, recursive: bool) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-r"] if recursive else []
    r = subprocess.run([
        "scp", *SSH_OPTS, *flags, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
        f"root@{host}:{remote}", str(local),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARNING: failed to pull {remote}: {r.stderr.strip()}")
        return False
    return True


def _fetch_results(host: str, port: int, cfg: PXDesignConfig) -> None:
    """Pull the entire design_outputs/<subdir>/ from the pod. Captures
    raw diffusion outputs (`global_run_0/...`), summary.csv, passing-*
    folders, and target_pred/ — whichever happen to exist. We err on the
    side of fetching everything because PXDesign's structure varies
    by preset and pipeline-completion state, and partial pipelines can
    still produce useful raw designs we'd otherwise lose."""
    remote_root = f"/workspace/twistr/design_outputs/{cfg.output_subdir}"
    local_root = REPO_ROOT / "design_outputs"
    local_root.mkdir(parents=True, exist_ok=True)
    print(f"fetching results from {remote_root} → {local_root / cfg.output_subdir}", flush=True)
    _scp_pull(host, port, remote_root, local_root, recursive=True)


def _terminate(pod_id: str) -> None:
    print(f"terminating pod {pod_id}", flush=True)
    r = subprocess.run(["runpodctl", "pod", "delete", pod_id],
                       capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "<no output>").strip()
        print(f"WARNING: delete failed — verify with `runpodctl pod list`. {msg}")


def _ship_per_pod_config(host: str, port: int, cfg: PXDesignConfig, label: str) -> None:
    """Write the per-pod cfg as YAML to a tempfile and scp it onto the pod
    as /workspace/twistr/config_pxdesign.yaml. Bootstrap reads from that
    fixed path; without this the per-pod output_subdir suffix is ignored
    and every parallel pod writes to the master config's output dir."""
    from dataclasses import asdict
    payload = asdict(cfg)
    # Convert the chains dict's TargetChain values back to plain dicts;
    # asdict already does this for nested dataclasses.
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
        yaml.safe_dump(payload, tmp, sort_keys=False)
        local_cfg = Path(tmp.name)
    try:
        scp = subprocess.run(
            ["scp", *SSH_OPTS, "-i", str(RUNPOD_SSH_KEY), "-P", str(port),
             str(local_cfg), f"root@{host}:/workspace/twistr/config_pxdesign.yaml"],
            capture_output=True, text=True,
        )
        if scp.returncode != 0:
            sys.exit(f"[{label}] failed to ship per-pod config: {(scp.stderr or scp.stdout).strip()}")
    finally:
        local_cfg.unlink(missing_ok=True)


def _run_one_pod(cfg: PXDesignConfig, tarball: Path) -> bool:
    """Allocate a pod, ship the prebuilt tarball, run bootstrap, fetch
    results, terminate. Returns True on success, False if the pipeline
    exited non-zero. Always terminates the pod via try/finally."""
    label = cfg.output_subdir
    pod_id, gpu_id = _create_pod(cfg)
    try:
        host, port = _ssh_endpoint(pod_id)
        print(f"[{label}] ssh endpoint: {host}:{port} — waiting for sshd", flush=True)
        _wait_for_sshd(host, port)
        _send_tarball(host, port, tarball, label=label)
        # Override the master config_pxdesign.yaml with this pod's sub_cfg
        # so bootstrap.sh's hardcoded `--config config_pxdesign.yaml` reads
        # the per-pod values (notably the suffixed output_subdir).
        _ship_per_pod_config(host, port, cfg, label)
        rc = _ssh_run_detached(
            host, port,
            "bash /workspace/twistr/tools/runpod_pxdesign/bootstrap.sh",
            label=label,
        )
        if rc != 0:
            print(f"[{label}] WARNING: bootstrap exited {rc} — attempting partial result fetch", flush=True)
        _fetch_results(host, port, cfg)
        if rc != 0:
            print(f"[{label}] FAILED on {gpu_id} (exit {rc})", flush=True)
            return False
        print(f"[{label}] complete on {gpu_id}", flush=True)
        return True
    finally:
        _terminate(pod_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight (config load, prereqs) and exit without "
             "creating a pod.",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of pods to run in parallel against the same config "
             "(default 1). Each pod gets a unique output_subdir suffix "
             "(_a, _b, _c, …) and shares one locally-built tarball.",
    )
    args = parser.parse_args()

    if not 1 <= args.n_parallel <= 26:
        sys.exit(f"--n-parallel must be in [1, 26], got {args.n_parallel}")

    cfg = load_pxdesign_config(args.config)
    _check_msa_cache(cfg)
    _check_prereqs()
    if args.dry_run:
        print("dry-run OK: pre-flight passed")
        return

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tarball = Path(tmp.name)
    try:
        _build_tarball(tarball)
        if args.n_parallel == 1:
            ok = _run_one_pod(cfg, tarball)
            if not ok:
                sys.exit("PXDesign run failed")
            return

        sub_cfgs = [
            replace(cfg, output_subdir=f"{cfg.output_subdir}_{string.ascii_lowercase[i]}")
            for i in range(args.n_parallel)
        ]
        failed = 0
        with ThreadPoolExecutor(max_workers=args.n_parallel) as pool:
            futures = {pool.submit(_run_one_pod, sc, tarball): sc.output_subdir for sc in sub_cfgs}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    if not fut.result():
                        failed += 1
                except BaseException as e:  # SystemExit propagates from helpers
                    failed += 1
                    print(f"[{label}] worker raised: {e}", flush=True)
        if failed:
            sys.exit(f"{failed}/{args.n_parallel} pod(s) failed; see above")
        print(f"all {args.n_parallel} pods completed", flush=True)
    finally:
        tarball.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
