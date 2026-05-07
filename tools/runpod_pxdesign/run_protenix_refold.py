"""Pod-side: run `protenix pred` on each input JSON in --inputs-dir.

Writes per-design SUCCESS markers to /workspace/refold_status/<name>.{ok,fail}
so the launcher can poll completion. Outputs go to --outputs-dir/<name>/.

Single-sample, default cycles, no AF2/PXDesign filter — just the all-atom
prediction + confidence JSON. The MSA paths inside each input JSON were
written by the local builder to point at /workspace/twistr/.cache/...
which is where the tarball extraction lands them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_one(input_json: Path, outputs_dir: Path, status_dir: Path,
             n_sample: int, n_step: int) -> tuple[str, bool, float]:
    name = input_json.stem
    out = outputs_dir / name
    out.mkdir(parents=True, exist_ok=True)
    log_path = status_dir / f"{name}.log"
    cmd = [
        "protenix", "pred",
        "-i", str(input_json),
        "-o", str(out),
        "-n", "protenix_base_default_v1.0.0",
        "-e", str(n_sample),
        "-p", str(n_step),
        "--use_msa", "True",
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
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--n-sample", type=int, default=1,
                        help="Diffusion samples per design. Default 1 — minimum for fastest wall.")
    parser.add_argument("--n-step", type=int, default=200,
                        help="Diffusion steps. Default 200 (Protenix default).")
    args = parser.parse_args()

    status_dir = Path("/workspace/refold_status")
    status_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.inputs_dir.glob("*.json"))
    if not inputs:
        print(f"FATAL: no input JSONs in {args.inputs_dir}", flush=True)
        sys.exit(2)
    print(f"==> {len(inputs)} designs to refold; outputs → {args.outputs_dir}", flush=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for inp in inputs:
        name, ok, dt = _run_one(inp, args.outputs_dir, status_dir,
                                args.n_sample, args.n_step)
        if ok:
            n_ok += 1
            print(f"==> [{name}] OK in {dt:.0f}s", flush=True)
        else:
            n_fail += 1
            print(f"==> [{name}] FAIL in {dt:.0f}s — see {status_dir / (name + '.log')}", flush=True)
    print(f"==> done. ok={n_ok} fail={n_fail}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
