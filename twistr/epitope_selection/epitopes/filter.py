"""Pre-MaSIF residue filter: helix detection (DSSP), SASA, halo expansion.

Returns the set of "allowed" residues whose surface vertices we keep when
masking the MaSIF output. Allowed = core ∪ halo, where:
  - core = helix (DSSP H/G) AND rSASA ≥ core_min_relative_sasa
  - halo = any-SS residue with at least one heavy atom within halo_distance_a
           of any core heavy atom AND rSASA ≥ halo_min_relative_sasa,
           one-hop only (no neighbors-of-neighbors), cross-chain allowed.

rSASA = (DSSP per-residue SASA) / (Tien 2013 theoretical max for that AA).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from .config import EpitopesConfig

# Tien et al. 2013, Table 1 "theoretical" max SASA (Å²) for Gly-X-Gly tripeptide.
TIEN2013_MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# DSSP one-letter → three-letter (for residues DSSP reports as standard AAs).
_AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


@dataclass(frozen=True)
class ResidueId:
    chain: str
    seq: int
    icode: str  # blank string if none

    def __str__(self) -> str:
        return f"{self.chain}/{self.seq}{self.icode}".rstrip()


@dataclass
class ResidueRecord:
    rid: ResidueId
    resname: str          # 3-letter PDB code
    ss: str               # DSSP code (H, G, I, E, B, T, S, ' ')
    sasa: float           # Å²
    rsasa: float          # SASA / Tien-2013 max; np.nan if non-standard residue
    heavy_xyz: np.ndarray    # (n_heavy, 3) all heavy-atom coords (incl. backbone)
    sidechain_xyz: np.ndarray  # heavy-atom coords excluding backbone (N, CA, C, O,
                               # OXT). ALA → CB only. GLY → empty (no patch anchor).


def run_dssp(pdb_path: Path, out_path: Path) -> None:
    """Run mkdssp; produces legacy DSSP-format text at `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mkdssp", "--output-format", "dssp", str(pdb_path), str(out_path)],
        check=True, capture_output=True, text=True,
    )


def parse_dssp(dssp_path: Path) -> dict[tuple[str, int, str], tuple[str, float]]:
    """Return {(chain, seq, icode): (ss_code, sasa_A2)}. Skips chain breaks."""
    out: dict[tuple[str, int, str], tuple[str, float]] = {}
    in_body = False
    with dssp_path.open() as f:
        for line in f:
            if not in_body:
                if line.startswith("  #  RESIDUE"):
                    in_body = True
                continue
            # Chain break marker line is shorter and has '!' in column 14.
            if len(line) < 39 or line[13] == "!":
                continue
            seq_field = line[5:11].strip()
            if not seq_field:
                continue
            seq = int(seq_field)
            icode = line[10:11].strip()
            chain = line[11:12]
            ss = line[16]
            try:
                sasa = float(line[34:38])
            except ValueError:
                continue
            out[(chain, seq, icode)] = (ss, sasa)
    return out


_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


def _heavy_atom_coords(residue: gemmi.Residue) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_heavy, sidechain_heavy) coords. GLY's sidechain set is empty
    (no β-carbon → no patch anchor)."""
    all_heavy: list[tuple[float, float, float]] = []
    sidechain: list[tuple[float, float, float]] = []
    is_gly = residue.name == "GLY"
    for atom in residue:
        if atom.element.name == "H":
            continue
        xyz = (atom.pos.x, atom.pos.y, atom.pos.z)
        all_heavy.append(xyz)
        if not is_gly and atom.name not in _BACKBONE_ATOM_NAMES:
            sidechain.append(xyz)
    arr_all = np.asarray(all_heavy, dtype=np.float64) if all_heavy else np.empty((0, 3))
    arr_sc = np.asarray(sidechain, dtype=np.float64) if sidechain else np.empty((0, 3))
    return arr_all, arr_sc


def build_residue_records(
    structure: gemmi.Structure,
    dssp_records: dict[tuple[str, int, str], tuple[str, float]],
) -> list[ResidueRecord]:
    """Walk the structure; pair each amino-acid residue with its DSSP entry."""
    records: list[ResidueRecord] = []
    for chain in structure[0]:
        for res in chain:
            if not _is_amino(res):
                continue
            seqid = res.seqid
            key = (chain.name, seqid.num, seqid.icode.strip())
            dssp = dssp_records.get(key)
            if dssp is None:
                continue
            ss, sasa = dssp
            resname = res.name
            max_sasa = TIEN2013_MAX_SASA.get(resname)
            rsasa = float(sasa / max_sasa) if max_sasa else float("nan")
            heavy_xyz, sc_xyz = _heavy_atom_coords(res)
            records.append(ResidueRecord(
                rid=ResidueId(chain=chain.name, seq=seqid.num, icode=seqid.icode.strip()),
                resname=resname, ss=ss, sasa=float(sasa), rsasa=rsasa,
                heavy_xyz=heavy_xyz, sidechain_xyz=sc_xyz,
            ))
    return records


def compute_allowed_residues(
    records: list[ResidueRecord], cfg: EpitopesConfig,
) -> tuple[set[ResidueId], set[ResidueId]]:
    """Returns (core, halo). core = DSSP helix residues with rSASA ≥ cfg.
    core_min_relative_sasa. halo = any-SS residues NOT in core, with at least
    one heavy atom within cfg.halo_distance_a of any core heavy atom AND
    rSASA ≥ cfg.halo_min_relative_sasa. The allowed set is core ∪ halo."""
    helix_codes = set(cfg.helix_codes)
    core_idx = [
        i for i, r in enumerate(records)
        if r.ss in helix_codes
        and not np.isnan(r.rsasa)
        and r.rsasa >= cfg.core_min_relative_sasa
        and r.heavy_xyz.size > 0
    ]
    if not core_idx:
        return set(), set()

    core_atoms = np.vstack([records[i].heavy_xyz for i in core_idx])
    core_tree = cKDTree(core_atoms)

    core: set[ResidueId] = {records[i].rid for i in core_idx}
    halo: set[ResidueId] = set()
    for r in records:
        if r.rid in core:
            continue
        if np.isnan(r.rsasa) or r.rsasa < cfg.halo_min_relative_sasa:
            continue
        if r.heavy_xyz.size == 0:
            continue
        hits = core_tree.query_ball_point(r.heavy_xyz, r=cfg.halo_distance_a)
        if any(h for h in hits):
            halo.add(r.rid)
    return core, halo


