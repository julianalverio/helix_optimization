"""Emit one PyMOL command file per PXDesign design.

Each .txt file: reinitializes PyMOL, loads the design PDB (target chain
A + designed binder chain B in the same file), colors the hotspot
residues on the target AND the binder residues that contact them by
amino-acid class, and shows sticks for those residues only.

AA class colors (per user spec):
  yellow = hydrophobic (incl. aromatic)
  red    = positive
  blue   = negative
  green  = polar

Usage:
  python -m twistr.dev.tools.runpod.pxdesign.pymol_views \\
      --config runtime/configs/pxdesign.yaml \\
      --inputs runtime/outputs/design_runtime/outputs/3erd_b2_a runtime/outputs/design_runtime/outputs/3erd_b2_b ... \\
      --out runtime/outputs/pymol_views/face1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from twistr.dev.tools.runpod.pxdesign.config import TargetChain, load_pxdesign_config

CONTACT_RADIUS = 5.0  # Å, heavy-atom; matches rank_designs.py

AA_CLASS_COLORS = {
    "hydrophobic": "yellow",
    "positive": "red",
    "negative": "blue",
    "polar": "green",
}

# Three-letter → class. P, G grouped with hydrophobic; C with polar; H with
# positive — conventional biochemistry-textbook partition. Aromatic
# (F/W/Y) goes to hydrophobic per user spec.
AA_TO_CLASS: dict[str, str] = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic",
    "ILE": "hydrophobic", "MET": "hydrophobic", "PRO": "hydrophobic",
    "GLY": "hydrophobic", "PHE": "hydrophobic", "TRP": "hydrophobic",
    "TYR": "hydrophobic",
    "LYS": "positive", "ARG": "positive", "HIS": "positive",
    "ASP": "negative", "GLU": "negative",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar",
    "CYS": "polar",
}


def _heavy_atoms(chain: gemmi.Chain) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[tuple[float, float, float]] = []
    seq: list[int] = []
    for res in chain:
        for atom in res:
            if atom.element.name == "H":
                continue
            xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
            seq.append(res.seqid.num)
    return np.asarray(xyz, dtype=np.float64), np.asarray(seq, dtype=np.int32)


def _residue_name(chain: gemmi.Chain, seq_num: int) -> str | None:
    for res in chain:
        if res.seqid.num == seq_num:
            return res.name
    return None


def _contacting_binder_residues(
    target: gemmi.Chain, binder: gemmi.Chain, hotspots: list[int],
) -> set[int]:
    binder_xyz, binder_seq = _heavy_atoms(binder)
    target_xyz, target_seq = _heavy_atoms(target)
    binder_tree = cKDTree(binder_xyz)
    contacting: set[int] = set()
    for hs in hotspots:
        idxs = np.where(target_seq == hs)[0]
        if len(idxs) == 0:
            continue
        for hit_list in binder_tree.query_ball_point(target_xyz[idxs], r=CONTACT_RADIUS):
            for j in hit_list:
                contacting.add(int(binder_seq[j]))
    return contacting


def _group_by_class(
    chain: gemmi.Chain, residue_nums: list[int],
) -> dict[str, list[int]]:
    """Return {class_name: sorted_residue_nums} for the given residues."""
    grouped: dict[str, list[int]] = {}
    for n in residue_nums:
        name = _residue_name(chain, n)
        if name is None:
            continue
        cls = AA_TO_CLASS.get(name.upper())
        if cls is None:
            continue
        grouped.setdefault(cls, []).append(n)
    return {k: sorted(v) for k, v in grouped.items()}


def _resi_selector(chain_id: str, residues: list[int]) -> str:
    return f"chain {chain_id} and resi {'+'.join(str(n) for n in residues)}"


def _build_pml(
    pdb_path: Path,
    design_id: str,
    target_chain: str,
    binder_chain: str,
    hotspots: list[int],
    contacting_binder: list[int],
    grouped_target: dict[str, list[int]],
    grouped_binder: dict[str, list[int]],
) -> str:
    obj = design_id.replace("/", "_")
    lines: list[str] = [
        "reinitialize",
        f"load {pdb_path}, {obj}",
        "hide everything",
        "show cartoon",
        f"color grey80, chain {target_chain}",
        f"color grey60, chain {binder_chain}",
        "",
        f"# Hotspots on chain {target_chain} — by AA class",
    ]
    for cls, residues in grouped_target.items():
        if not residues:
            continue
        lines.append(f"color {AA_CLASS_COLORS[cls]}, {_resi_selector(target_chain, residues)}")
    lines.append("")
    lines.append(f"# Contacting binder residues on chain {binder_chain} — by AA class")
    for cls, residues in grouped_binder.items():
        if not residues:
            continue
        lines.append(f"color {AA_CLASS_COLORS[cls]}, {_resi_selector(binder_chain, residues)}")
    lines.append("")

    target_sel = _resi_selector(target_chain, hotspots)
    if contacting_binder:
        binder_sel = _resi_selector(binder_chain, contacting_binder)
        focus = f"({target_sel}) or ({binder_sel})"
    else:
        focus = f"({target_sel})"
    lines += [
        f"show sticks, {focus}",
        f"orient {focus}",
        f"zoom {focus}, 5",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir; one .txt per design will be written here.")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="B")
    args = parser.parse_args()

    cfg = load_pxdesign_config(args.config)
    chain_entry = cfg.target.chains.get(args.target_chain)
    if not isinstance(chain_entry, TargetChain) or chain_entry.hotspots is None:
        raise SystemExit(f"config has no hotspots on chain {args.target_chain}")
    hotspots = list(chain_entry.hotspots)

    pdbs: list[Path] = []
    for d in args.inputs:
        pdbs.extend(sorted(d.glob("**/predictions/converted_pdbs/*.pdb")))
    if not pdbs:
        raise SystemExit(f"no PDBs found under {args.inputs}")

    args.out.mkdir(parents=True, exist_ok=True)
    for pdb in pdbs:
        structure = gemmi.read_structure(str(pdb))
        model = structure[0]
        target = model[args.target_chain]
        binder = model[args.binder_chain]

        contacting = sorted(_contacting_binder_residues(target, binder, hotspots))
        grouped_target = _group_by_class(target, hotspots)
        grouped_binder = _group_by_class(binder, contacting)

        parts = pdb.parts
        run_part = parts[parts.index("design_outputs") + 1] if "design_outputs" in parts else parts[-6]
        seed_part = next((p for p in parts if p.startswith("seed_")), "seed_?")
        sample_part = pdb.stem.replace("pxdesign_target_", "")
        design_id = f"{run_part}/{seed_part}/{sample_part}"

        pml = _build_pml(
            pdb_path=pdb.resolve(),
            design_id=design_id,
            target_chain=args.target_chain,
            binder_chain=args.binder_chain,
            hotspots=hotspots,
            contacting_binder=contacting,
            grouped_target=grouped_target,
            grouped_binder=grouped_binder,
        )
        out_path = args.out / f"{design_id.replace('/', '_')}.txt"
        out_path.write_text(pml)

    print(f"wrote {len(pdbs)} PyMOL command files to {args.out}/")


if __name__ == "__main__":
    main()
