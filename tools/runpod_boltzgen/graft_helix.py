"""Graft our cropped designed helix into a scaffold by Kabsch-aligning to a
target residue range and replacing those residues. Useful for substituting
a sdAb scaffold loop with a designed binder helix.

Usage:
  python -m tools.runpod_boltzgen.graft_helix \\
      --helix-pdb cropped/face1_best.pdb --helix-chain A --helix-resi 20-29 \\
      --scaffold-pdb ~/Desktop/helical_sdAb_scaffold_examples/93ite.pdb \\
      --scaffold-chain A --scaffold-resi 13-22 \\
      --out grafted/face1_into_93ite.pdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemmi
import numpy as np


def _range(s: str) -> list[int]:
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def _ca(chain: gemmi.Chain, seqids: list[int]) -> np.ndarray:
    coords = []
    for sid in seqids:
        r = next((r for r in chain if r.seqid.num == sid), None)
        if r is None:
            raise SystemExit(f"residue {sid} not in chain {chain.name}")
        ca = next((a for a in r if a.name == "CA"), None)
        if ca is None:
            raise SystemExit(f"residue {sid} in chain {chain.name} has no CA")
        coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return np.asarray(coords)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that (R @ P.T).T + t aligns P onto Q."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Qc - R @ Pc
    return R, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--helix-pdb", type=Path, required=True)
    ap.add_argument("--helix-chain", default="A")
    ap.add_argument("--helix-resi", required=True, help="inclusive, e.g. 20-29")
    ap.add_argument("--scaffold-pdb", type=Path, required=True)
    ap.add_argument("--scaffold-chain", default="A")
    ap.add_argument("--scaffold-resi", required=True, help="inclusive, e.g. 13-22")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    h_seqs = _range(args.helix_resi)
    s_seqs = _range(args.scaffold_resi)
    if len(h_seqs) != len(s_seqs):
        raise SystemExit(f"length mismatch: helix {len(h_seqs)} vs scaffold {len(s_seqs)}")

    helix = gemmi.read_structure(str(args.helix_pdb.expanduser()))
    scaffold = gemmi.read_structure(str(args.scaffold_pdb.expanduser()))
    h_chain = helix[0][args.helix_chain]
    s_chain = scaffold[0][args.scaffold_chain]

    P = _ca(h_chain, h_seqs)
    Q = _ca(s_chain, s_seqs)
    R, t = _kabsch(P, Q)
    aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))
    print(f"Kabsch RMSD over {len(P)} CAs: {rmsd:.3f} A")

    # Build replacement chain: scaffold residues before s_seqs, transformed
    # helix residues renumbered to s_seqs, scaffold residues after s_seqs.
    smin, smax = min(s_seqs), max(s_seqs)
    new_chain = gemmi.Chain(args.scaffold_chain)
    for r in s_chain:
        if r.seqid.num < smin:
            new_chain.add_residue(r)
    for hs, ss in zip(h_seqs, s_seqs):
        hr = next(rr for rr in h_chain if rr.seqid.num == hs)
        new_r = gemmi.Residue()
        new_r.name = hr.name
        new_r.seqid = gemmi.SeqId(ss, " ")
        new_r.entity_type = gemmi.EntityType.Polymer
        for a in hr:
            new_a = gemmi.Atom()
            new_a.name = a.name
            new_a.element = a.element
            p = np.array([a.pos.x, a.pos.y, a.pos.z])
            new_pos = R @ p + t
            new_a.pos = gemmi.Position(*new_pos)
            new_a.b_iso = a.b_iso
            new_a.occ = a.occ
            new_r.add_atom(new_a)
        new_chain.add_residue(new_r)
    for r in s_chain:
        if r.seqid.num > smax:
            new_chain.add_residue(r)

    out = scaffold.clone()
    out[0].remove_chain(args.scaffold_chain)
    out[0].add_chain(new_chain)

    # Also bring along every OTHER chain from helix-pdb (the bound target),
    # transformed by the same (R, t) so the complex stays in the bound pose
    # in the scaffold frame.
    for hc in helix[0]:
        if hc.name == args.helix_chain:
            continue
        bound = gemmi.Chain(hc.name)
        for r in hc:
            new_r = gemmi.Residue()
            new_r.name = r.name
            new_r.seqid = r.seqid
            new_r.entity_type = r.entity_type
            for a in r:
                new_a = gemmi.Atom()
                new_a.name = a.name
                new_a.element = a.element
                p = np.array([a.pos.x, a.pos.y, a.pos.z])
                new_pos = R @ p + t
                new_a.pos = gemmi.Position(*new_pos)
                new_a.b_iso = a.b_iso
                new_a.occ = a.occ
                new_r.add_atom(new_a)
            bound.add_residue(new_r)
        out[0].add_chain(bound)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(args.out.expanduser()))
    print(f"wrote {args.out.expanduser()}")


if __name__ == "__main__":
    main()
