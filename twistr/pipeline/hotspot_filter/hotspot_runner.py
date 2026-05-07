"""Runner for PPI-hotspotID's critires.sh.

Invokes critires.sh as a subprocess; expected to run inside an environment
with AmberTools, FreeSASA, DSSP, and the AutoGluon-loadable model already
present (the Modal worker image, or a comparably provisioned local env).

The output of critires.sh is `results.txt` with rows of the form:
    <chain> <resnum> <score>
(possibly with an icode appended to resnum, e.g. "27A").
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..epitopes.filter import ResidueId
from ..scannet_filter.filter import _split_seq_icode


def parse_results_txt(path: Path) -> dict[ResidueId, float]:
    """Read PPI-hotspotID's results.txt → {ResidueId: score}.

    Tolerates either whitespace- or comma-separated columns. Each row should
    contain at least chain, residue index (with optional icode), score."""
    out: dict[ResidueId, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in (line.split(",") if "," in line else line.split())]
            if len(parts) < 3:
                continue
            chain = parts[0]
            res_idx = parts[1]
            try:
                score = float(parts[-1])
            except ValueError:
                continue
            seq, icode = _split_seq_icode(res_idx)
            out[ResidueId(chain=chain, seq=seq, icode=icode)] = score
    return out


def run_ppi_hotspot_critires(
    pdb_path: Path, scratch_dir: Path,
) -> dict[ResidueId, float]:
    """Run PPI-hotspotID's critires.sh on `pdb_path` from inside an
    environment that already has all of AmberTools / FreeSASA / DSSP /
    AutoGluon installed (i.e. inside the Modal worker, NOT on the host).

    Stages the PDB into `scratch_dir/input.pdb`, runs critires.sh, parses
    `scratch_dir/results.txt`."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdb_path, scratch_dir / "input.pdb")
    # `critires.sh` lives in the PPI-hotspot repo cloned into the image; the
    # Modal worker prepends its directory to PATH.
    result = subprocess.run(
        ["critires.sh"], cwd=scratch_dir,
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PPI-hotspotID critires.sh failed: exit={result.returncode}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}"
        )
    results_path = scratch_dir / "results.txt"
    if not results_path.exists():
        raise RuntimeError(
            f"PPI-hotspotID produced no results.txt:\n"
            f"--- stdout tail ---\n{result.stdout[-2000:]}"
        )
    return parse_results_txt(results_path)
