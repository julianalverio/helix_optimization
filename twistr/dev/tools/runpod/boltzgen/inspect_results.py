"""Quick local inspector for BoltzGen outputs.

Reads each face's `final_ranked_designs/final_designs_metrics_*.csv`
(or `all_designs_metrics.csv` if the budget run didn't filter) and
prints the top designs by ipTM with the matching CIF paths.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _find_metrics_csv(face_root: Path) -> Path | None:
    cands = sorted(face_root.rglob("final_designs_metrics_*.csv"))
    if cands:
        return cands[0]
    cands = sorted(face_root.rglob("all_designs_metrics.csv"))
    return cands[0] if cands else None


def _find_cif_dir(face_root: Path) -> Path | None:
    for sub in ("final_designs", "intermediate_ranked"):
        cands = sorted(face_root.rglob(f"final_*_designs"))
        if cands:
            return cands[-1]
    cands = sorted(face_root.rglob("*.cif"))
    return cands[0].parent if cands else None


def _inspect(face_root: Path, top_n: int) -> None:
    print(f"\n=== {face_root} ===")
    metrics = _find_metrics_csv(face_root)
    if metrics is None:
        print(f"  no metrics CSV found under {face_root}")
        cifs = sorted(face_root.rglob("*.cif"))
        print(f"  {len(cifs)} CIF(s) on disk")
        for c in cifs[:top_n]:
            print(f"    {c}")
        return
    print(f"  metrics: {metrics}")
    rows = list(csv.DictReader(metrics.open()))
    print(f"  {len(rows)} designs")
    sort_key = "iptm" if rows and "iptm" in rows[0] else "ipTM" if rows and "ipTM" in rows[0] else None
    if sort_key:
        rows.sort(key=lambda r: float(r.get(sort_key, 0) or 0), reverse=True)
        print(f"  top {min(top_n, len(rows))} by {sort_key}:")
        for i, r in enumerate(rows[:top_n], start=1):
            name = r.get("name") or r.get("design_id") or r.get("id") or "?"
            iptm = r.get(sort_key, "?")
            plddt = r.get("plddt") or r.get("plddt_binder") or r.get("complex_plddt") or "?"
            print(f"    {i:>2}. {name}  ipTM={iptm}  pLDDT={plddt}")
    cifs = sorted(face_root.rglob("*.cif"))
    print(f"  CIFs on disk: {len(cifs)}")
    for c in cifs[:3]:
        print(f"    {c}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("boltzgen_outputs"))
    parser.add_argument("--faces", nargs="+", default=["face1", "face2"])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    for face in args.faces:
        face_root = args.root / face
        if not face_root.is_dir():
            print(f"[{face}] no directory at {face_root}")
            continue
        _inspect(face_root, args.top)


if __name__ == "__main__":
    main()
