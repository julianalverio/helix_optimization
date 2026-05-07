"""Wrapper around ScanNet's predict_bindingsites.py running inside the
official Docker image (jertubiana/scannet).

Pipeline per PDB:
  1. Stage the input PDB into a per-PDB scratch dir mounted at /work.
  2. In the container:
       cd /scannet
       python predict_bindingsites.py /work/<pdbid>.pdb \\
         --mode <mode> [--assembly] --noMSA --name <pdbid> \\
         --predictions_folder /work/scannet_out
  3. Glob for `predictions_<pdbid>.csv` under the predictions folder.
  4. Parse the CSV and return {ResidueId: score} for model 0.

CSV columns (per `write_predictions` in predict_bindingsites.py):
    Model, Chain, Residue Index, Sequence, Binding site probability
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from .._cache import is_valid, mark, signature
from ..epitopes.filter import ResidueId
from .filter import _split_seq_icode


# `jertubiana/scannet` image puts the repo at /ScanNet (its WORKDIR).
SCANNET_ROOT = "/ScanNet"


def _container_script(pdbid_lower: str, mode: str, assembly: bool) -> str:
    extra = "--assembly " if assembly else ""
    return (
        "set -euo pipefail\n"
        f"cd {SCANNET_ROOT}\n"
        "mkdir -p /work/scannet_out\n"
        "python predict_bindingsites.py "
        f"/work/{pdbid_lower}.pdb "
        f"--mode {mode} --noMSA {extra}"
        f"--name {pdbid_lower} "
        "--predictions_folder /work/scannet_out\n"
    )


def run_scannet(
    pdb_path: Path,
    pdb_id: str,
    scratch_dir: Path,
    image: str,
    platform: str,
    mode: str,
    assembly: bool,
) -> dict[ResidueId, float]:
    """Score every residue in `pdb_path` with ScanNet. Returns {ResidueId: score}.

    The CSV's residue identifier becomes ResidueId(chain, seq, icode).
    Multi-model PDBs are restricted to model 0."""
    pdbid_lower = pdb_id.lower()
    scratch_dir = scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir = scratch_dir / "scannet_out"

    # Cache hit: outputs exist AND the cache sidecar matches the signature
    # of the current input PDB + mode + assembly + docker image tag.
    sidecar = out_dir / ".cache_sig"
    sig = signature(pdb_path, mode, str(assembly), image)
    if out_dir.exists():
        cached = sorted(out_dir.rglob(f"predictions_{pdbid_lower}.csv"))
        if cached and is_valid(sidecar, sig):
            return _parse_predictions_csv(cached[0])

    shutil.copy(pdb_path, scratch_dir / f"{pdbid_lower}.pdb")
    script = _container_script(pdbid_lower, mode, assembly)
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--platform", platform,
            "-v", f"{scratch_dir}:/work",
            image, "bash", "-lc", script,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ScanNet docker failed for {pdb_id}: exit={result.returncode}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}"
        )

    csvs = sorted(out_dir.rglob(f"predictions_{pdbid_lower}.csv"))
    if not csvs:
        raise RuntimeError(
            f"ScanNet produced no predictions CSV for {pdb_id}\n"
            f"stdout tail:\n{result.stdout[-1500:]}"
        )
    mark(sidecar, sig)
    return _parse_predictions_csv(csvs[0])


def _parse_predictions_csv(path: Path) -> dict[ResidueId, float]:
    """ScanNet CSV → {ResidueId: score}. Skips rows whose Model column is not 0."""
    out: dict[ResidueId, float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        score_col = "Binding site probability"
        if score_col not in reader.fieldnames:
            # Fall back to first non-metadata column if the model emits multi-output.
            score_col = next(c for c in reader.fieldnames
                             if c not in ("Model", "Chain", "Residue Index", "Sequence"))
        for row in reader:
            model = row.get("Model", "0").strip()
            if model not in ("", "0"):
                continue
            chain = row["Chain"].strip()
            res_index = row["Residue Index"].strip()
            seq, icode = _split_seq_icode(res_index)
            try:
                score = float(row[score_col])
            except (TypeError, ValueError):
                continue
            out[ResidueId(chain=chain, seq=seq, icode=icode)] = score
    return out


