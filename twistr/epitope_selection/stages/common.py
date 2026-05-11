"""Helpers shared across pipeline stages."""
from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path

import gemmi
import pandas as pd

from twistr.epitope_selection.epitopes.config import EpitopesConfig
from twistr.epitope_selection.epitopes.filter import (
    build_residue_records, compute_allowed_residues, parse_dssp, run_dssp,
)

logger = logging.getLogger(__name__)


# Drop chains shorter than this in multi-chain inputs (catches stray
# co-activator peptides in "_receptor" PDBs that break MaSIF's MSMS).
_MIN_CHAIN_AA = 30

# Bump on changes to modal_image.py's critires.sh, AmberTools, or the
# AutoGluon model — invalidates per-PDB hotspot caches.
_CRITIRES_VERSION = "critires_v1"


def stage_pdb(pdb_id: str, pdb_dir: Path, work_root: Path) -> tuple[bytes, list[str], Path]:
    """Decompress mmCIF → on-disk PDB, sanitized.

    Removes alternate conformations, hydrogens, waters, and ligands; drops
    residues missing CA/N/C; for multi-chain inputs drops chains shorter
    than `_MIN_CHAIN_AA` residues. Returns (pdb_bytes, chain_ids, pdb_path).
    """
    cif_gz = pdb_dir / pdb_id.lower()[1:3] / f"{pdb_id.lower()}.cif.gz"
    if not cif_gz.exists():
        raise FileNotFoundError(str(cif_gz))
    with gzip.open(cif_gz, "rt") as f:
        text = f.read()
    structure = gemmi.read_structure_string(text, format=gemmi.CoorFormat.Mmcif)
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_waters()
    structure.remove_ligands_and_waters()
    backbone = {"CA", "N", "C"}
    for chain in structure[0]:
        bad = [i for i, r in enumerate(chain)
               if not backbone.issubset({a.name for a in r})]
        for i in reversed(bad):
            del chain[i]

    chain_aa: dict[str, int] = {}
    for chain in structure[0]:
        aa = sum(
            1 for r in chain
            if (info := gemmi.find_tabulated_residue(r.name)) is not None
            and info.is_amino_acid()
        )
        if aa > 0:
            chain_aa[chain.name] = aa
    if not chain_aa:
        raise RuntimeError(f"{pdb_id}: no protein chains found")

    if len(chain_aa) > 1:
        kept = {n for n, c in chain_aa.items() if c >= _MIN_CHAIN_AA} or set(chain_aa)
        dropped = set(chain_aa) - kept
        if dropped:
            logger.info(
                "%s: dropping small chains %s (< %d aa) from %s",
                pdb_id, sorted(dropped), _MIN_CHAIN_AA,
                {n: chain_aa[n] for n in sorted(chain_aa)},
            )
            for cn in dropped:
                structure[0].remove_chain(cn)
        chains = sorted(kept)
    else:
        chains = list(chain_aa)

    work = work_root / pdb_id.lower()
    work.mkdir(parents=True, exist_ok=True)
    pdb_path = work / f"{pdb_id.lower()}.pdb"
    structure.write_pdb(str(pdb_path))
    return pdb_path.read_bytes(), chains, pdb_path


def build_records(pdb_path: Path, ep_cfg: EpitopesConfig):
    """Run DSSP on `pdb_path`; return (records, core, halo)."""
    structure = gemmi.read_structure(str(pdb_path))
    dssp_path = pdb_path.with_suffix(".dssp")
    run_dssp(pdb_path, dssp_path)
    records = build_residue_records(structure, parse_dssp(dssp_path))
    core, halo = compute_allowed_residues(records, ep_cfg)
    return records, core, halo


def clean_pdb_for_critires(pdb_bytes: bytes) -> bytes:
    """Strip HETATM and non-blank/non-A altlocs (critires.sh's check_bb.py
    rejects both); keep header lines the parser tolerates."""
    cleaned: list[str] = []
    for line in pdb_bytes.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM  "):
            if line.startswith(("HEADER", "CRYST1", "MODEL", "ENDMDL", "TER", "END")):
                cleaned.append(line)
            continue
        altloc = line[16] if len(line) > 16 else " "
        if altloc not in (" ", "A"):
            continue
        cleaned.append(line)
    return ("\n".join(cleaned) + "\n").encode("utf-8")


def write_parquet(rows: list[dict], path: Path) -> None:
    """Write `rows` to `path` and fsync so the next stage can't observe a
    half-written file even if interrupted."""
    pd.DataFrame(rows).to_parquet(path, index=False)
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def stamp_pdb_path(rows: list[dict], pdb_path: Path) -> None:
    abs_path = str(pdb_path.resolve())
    for row in rows:
        row["pdb_path"] = abs_path


def write_diagnostics(diagnostics: list[dict], path: Path) -> None:
    path.write_text(json.dumps(diagnostics, indent=2, default=str))


def derive_pdb_ids(parquet_path: Path) -> list[str]:
    """PDB list for stages run without `--pdb-list`, taken from upstream parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"cannot derive PDB list — {parquet_path} does not exist; "
            f"upstream stages must run first or this stage must be invoked "
            f"with `--pdb-list` and a stages list that starts at 'masif'."
        )
    df = pd.read_parquet(parquet_path)
    if df.empty or "pdb_id" not in df.columns:
        return []
    seen: list[str] = []
    for pid in df["pdb_id"]:
        if pid not in seen:
            seen.append(pid)
    return seen
