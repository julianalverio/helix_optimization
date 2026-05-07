"""Score BoltzGen designs on three geometric criteria + ipTM:
  1. fraction of hotspot residues contacted by the binder (heavy atoms ≤5 Å)
  2. total buried surface area on the hotspot residues (target side)
  3. DSSP helix-fraction across the binder residues that contact hotspots
  4. ipTM (joined from BoltzGen's final_designs_metrics_*.csv by name)

Usage:
  python -m tools.runpod_boltzgen.score_designs \\
      --hotspots 44,48,51,52,55,59 \\
      --metrics-csv boltzgen_outputs/face1/face1/final_ranked_designs/final_designs_metrics_50.csv \\
      --cifs boltzgen_outputs/face1/face1/final_ranked_designs/final_50_designs/rank*.cif \\
      --out scoring/face1.csv
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import freesasa
import gemmi
import numpy as np
from scipy.spatial import cKDTree

from twistr.pipeline.tensors.dssp import run_dssp

# BoltzGen output: target chain on B, designed binder on A.
TARGET_CHAIN, BINDER_CHAIN = "B", "A"
CONTACT_R = 5.0  # Å, heavy-atom


def _heavy_atom_xyz(chain: gemmi.Chain) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for r in chain:
        pts = [(a.pos.x, a.pos.y, a.pos.z) for a in r if a.element.name != "H"]
        if pts:
            out[r.seqid.num] = np.asarray(pts)
    return out


def _contacts(target: dict[int, np.ndarray], binder: dict[int, np.ndarray],
              hotspots: list[int]) -> dict[int, set[int]]:
    seqids = sorted(binder)
    pts = np.vstack([binder[s] for s in seqids])
    sid = np.concatenate([np.full(len(binder[s]), s) for s in seqids])
    tree = cKDTree(pts)
    return {
        h: (
            {int(sid[i]) for atom_idxs in tree.query_ball_point(target[h], r=CONTACT_R) for i in atom_idxs}
            if h in target else set()
        )
        for h in hotspots
    }


def _hotspot_bsa(struct: gemmi.Structure, hotspots: list[int], work: Path) -> float:
    """SASA loss on the target hotspot residues when the binder is present."""
    complex_pdb = work / "complex.pdb"
    target_pdb = work / "target.pdb"
    struct.write_pdb(str(complex_pdb))
    target_only = struct.clone()
    for c in list(target_only[0]):
        if c.name != TARGET_CHAIN:
            target_only[0].remove_chain(c.name)
    target_only.write_pdb(str(target_pdb))
    sc = freesasa.Structure(str(complex_pdb)); rc = freesasa.calc(sc)
    st = freesasa.Structure(str(target_pdb));  rt = freesasa.calc(st)
    sels = [f"h{h}, chain {TARGET_CHAIN} and resi {h}" for h in hotspots]
    ac = freesasa.selectArea(sels, sc, rc)
    at = freesasa.selectArea(sels, st, rt)
    return sum(max(0.0, at[f"h{h}"] - ac[f"h{h}"]) for h in hotspots)


def _interface_helix_frac(struct: gemmi.Structure, binder_seqids: set[int]) -> float:
    if not binder_seqids:
        return float("nan")
    # SS8_CODES: ("H", "G", "I", "E", "B", "T", "S", "-", "?") — H is index 0.
    o = run_dssp(struct)
    helix = sum(
        1 for s in binder_seqids
        if (k := o.ss_map.get((BINDER_CHAIN, s))) is not None and k[1] == 0
    )
    return helix / len(binder_seqids)


def _score(cif_path: Path, hotspots: list[int], work: Path) -> tuple[float, float, float]:
    s = gemmi.read_structure(str(cif_path))
    t = _heavy_atom_xyz(s[0][TARGET_CHAIN])
    b = _heavy_atom_xyz(s[0][BINDER_CHAIN])
    contacts = _contacts(t, b, hotspots)
    frac_hit = sum(1 for v in contacts.values() if v) / len(hotspots)
    bsa = _hotspot_bsa(s, hotspots, work)
    union: set[int] = set().union(*contacts.values()) if contacts else set()
    helix = _interface_helix_frac(s, union)
    return frac_hit, bsa, helix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hotspots", required=True, help="e.g. 44,48,51,52,55,59 (BoltzGen-frame target seqids)")
    ap.add_argument("--cifs", nargs="+", required=True, help="design CIF files to score")
    ap.add_argument("--metrics-csv", type=Path, required=True,
                    help="BoltzGen final_designs_metrics_*.csv (for ipTM lookup by 'name')")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hotspots = [int(x) for x in args.hotspots.split(",")]
    iptm_by_name = {row["id"]: float(row["iptm"]) for row in csv.DictReader(args.metrics_csv.open())}

    rows: list[tuple[str, float, float, float, float, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for cif in (Path(p) for p in args.cifs):
            fh, bsa, hf = _score(cif, hotspots, work)
            # CIF stem looks like "rank01_face1_20"; the metrics-CSV name is "face1_20".
            name = cif.stem.split("_", 1)[1] if "_" in cif.stem else cif.stem
            iptm = iptm_by_name.get(name, float("nan"))
            rows.append((cif.stem, fh, bsa, hf, iptm, str(cif)))
            print(f"{cif.stem}: hit={fh:.2f} bsa={bsa:.0f} hxF={hf:.2f} iptm={iptm:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r[1], r[2], 0.0 if r[3] != r[3] else r[3], r[4]), reverse=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design_id", "frac_hotspots_hit", "hotspot_bsa_a2",
                    "helix_fraction_at_interface", "iptm", "cif"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
