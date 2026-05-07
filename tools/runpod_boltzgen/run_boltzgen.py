"""Pod-side wrapper: invoke `boltzgen run` for each spec in --specs-dir.

Writes per-face SUCCESS/FAIL markers to /workspace/boltzgen_status/<face>.{ok,fail}
plus per-face logs so the launcher can poll completion.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _run_one(spec: Path, outputs_dir: Path, status_dir: Path,
             cache_dir: Path, num_designs: int) -> tuple[str, bool, float]:
    name = spec.stem  # "face1" / "face2"
    out = outputs_dir / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = status_dir / f"{name}.log"
    cmd = [
        "boltzgen", "run", str(spec),
        "--output", str(out),
        "--protocol", "protein-anything",
        "--num_designs", str(num_designs),
        "--budget", str(num_designs),
        "--devices", "1",
        "--cache", str(cache_dir),
        # Auto-mode would enable cuequivariance kernels on H100 (capability 9),
        # which JIT-compile against CUDA 13 nvrtc — the RunPod base image only
        # ships CUDA 12.4, so the JIT crashes with
        # `libnvrtc-builtins.so.13.0` not found. Explicit false sidesteps it.
        "--use_kernels", "false",
    ]
    print(f"==> [{name}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    marker = status_dir / (f"{name}.ok" if rc == 0 else f"{name}.fail")
    marker.write_text(f"rc={rc}\nseconds={dt:.1f}\n")
    return name, rc == 0, dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--num-designs", type=int,
                        default=int(os.environ.get("BOLTZGEN_NUM_DESIGNS", "2")))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/workspace/cache/boltzgen"))
    args = parser.parse_args()

    status_dir = Path("/workspace/boltzgen_status")
    status_dir.mkdir(parents=True, exist_ok=True)

    specs = sorted(args.specs_dir.glob("face*.yaml"))
    if not specs:
        print(f"FATAL: no face*.yaml specs in {args.specs_dir}", flush=True)
        sys.exit(2)
    print(f"==> {len(specs)} face spec(s); num_designs={args.num_designs}; "
          f"outputs → {args.outputs_dir}", flush=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    for spec in specs:
        name, ok, dt = _run_one(spec, args.outputs_dir, status_dir,
                                args.cache_dir, args.num_designs)
        if ok:
            n_ok += 1
            print(f"==> [{name}] OK in {dt:.0f}s", flush=True)
        else:
            n_fail += 1
            print(f"==> [{name}] FAIL in {dt:.0f}s — see {status_dir / (name + '.log')}",
                  flush=True)
    print(f"==> done. ok={n_ok} fail={n_fail}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
