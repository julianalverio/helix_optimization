"""Run one per-linker Remodel job by shelling out to the isolated
PyRosetta interpreter."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).parent / "_remodel_script.py"


def run_remodel(
    rosetta_python: str,
    subpose_pdb: Path,
    blueprint: Path,
    out_dir: Path,
    nstruct: int,
    num_trajectory: int,
    linker_lo: int,
    linker_hi: int,
    max_chainbreak_a: float = 0.5,
) -> list[dict]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "subpose_pdb": str(subpose_pdb.resolve()),
        "blueprint": str(blueprint.resolve()),
        "out_dir": str(out_dir),
        "nstruct": nstruct,
        "num_trajectory": num_trajectory,
        "linker_lo": linker_lo,
        "linker_hi": linker_hi,
        "max_chainbreak_a": max_chainbreak_a,
    }
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2))

    # absolute() instead of resolve() — the venv python is a symlink to the
    # base interpreter, and following it loses the venv's site-packages.
    python_bin = str(Path(os.path.expanduser(rosetta_python)).absolute())
    log_path = out_dir / "remodel.log"
    with log_path.open("w") as log:
        subprocess.run(
            [python_bin, str(_SCRIPT.resolve()), str(job_path)],
            check=True, stdout=log, stderr=subprocess.STDOUT,
            cwd=out_dir,
        )

    return json.loads((out_dir / "scores.json").read_text())
