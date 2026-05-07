"""For face1's best design, find the 13 consecutive all-helical binder
residues with max hotspot-contact count, and write a PDB containing the
full target plus only those 13 binder residues.

Usage:
  python -m tools.runpod_boltzgen.crop_best_helix \\
      --scoring scoring/face1.csv \\
      --hotspots 44,48,51,52,55,59 \\
      --out cropped/face1_best.pdb
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from twistr.pipeline.tensors.dssp import run_dssp

TARGET, BINDER = "B", "A"
R, W = 5.0, 10  # contact radius (Å), helix window length


def _heavy(chain: gemmi.Chain) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for r in chain:
        pts = [(a.pos.x, a.pos.y, a.pos.z) for a in r if a.element.name != "H"]
        if pts:
            out[r.seqid.num] = np.asarray(pts)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoring", type=Path, required=True,
                    help="CSV from score_designs.py; first row is treated as best")
    ap.add_argument("--hotspots", required=True,
                    help="Comma-separated target seqids, e.g. 44,48,51,52,55,59")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hotspots = [int(x) for x in args.hotspots.split(",")]
    best = next(csv.DictReader(args.scoring.open()))
    cif = Path(best["cif"])
    print(f"best: {best['design_id']} ← {cif}")

    s = gemmi.read_structure(str(cif))
    target = s[0][TARGET]
    binder = s[0][BINDER]

    # Per-binder-residue: number of distinct hotspots within R of any heavy atom.
    t_atoms, b_atoms = _heavy(target), _heavy(binder)
    sids = sorted(b_atoms)
    pts = np.vstack([b_atoms[i] for i in sids])
    sid_per_atom = np.concatenate([np.full(len(b_atoms[i]), i) for i in sids])
    tree = cKDTree(pts)
    contact = {i: 0 for i in sids}
    for h in hotspots:
        if h not in t_atoms:
            continue
        contacted = {int(sid_per_atom[a])
                     for atom_idxs in tree.query_ball_point(t_atoms[h], r=R)
                     for a in atom_idxs}
        for i in contacted:
            contact[i] += 1

    # SS8 code 0 == H per twistr.pipeline.tensors.constants.SS8_CODES.
    ss = run_dssp(s).ss_map
    is_h = lambda i: (k := ss.get((BINDER, i))) is not None and k[1] == 0

    # Consecutive length-W all-helical window with max sum-of-contact-counts.
    best_win: list[int] | None = None
    best_score = -1
    for i in range(len(sids) - W + 1):
        win = sids[i:i + W]
        if win[-1] - win[0] != W - 1:
            continue
        if not all(is_h(j) for j in win):
            continue
        score = sum(contact[j] for j in win)
        if score > best_score:
            best_win, best_score = win, score
    if best_win is None:
        raise SystemExit(f"no {W}-residue all-helical consecutive window in chain {BINDER}")
    print(f"binder residues {best_win[0]}..{best_win[-1]}  hotspot-contacts={best_score}")

    # Build cropped structure: full target chain + only kept binder residues.
    keep = set(best_win)
    out = gemmi.Structure()
    out.name = s.name
    m = gemmi.Model("1")
    nt = gemmi.Chain(TARGET)
    for r in target:
        nt.add_residue(r)
    m.add_chain(nt)
    nb = gemmi.Chain(BINDER)
    for r in binder:
        if r.seqid.num in keep:
            nb.add_residue(r)
    m.add_chain(nb)
    out.add_model(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
