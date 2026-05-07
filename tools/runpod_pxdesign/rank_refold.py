"""Rank refolded designs by Protenix confidence.

Reads `refold_outputs/<pod_label>/<design_name>/<dataset>/<sample>/seed_*/predictions/`
which contains:
  - <name>_sample_0.cif (the all-atom predicted complex)
  - <name>_summary_confidence_sample_0.json (pTM, ipTM, pLDDT, ranking_score, ...)

Writes a CSV sorted by ranking_score (Protenix's own composite: 0.8*ipTM + 0.2*pTM).

Usage:
  python -m tools.runpod_pxdesign.rank_refold \\
      --refold-dir refold_outputs \\
      --face-prefix 3erd_b2 \\
      --out rankings/face1_refold.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _gather(refold_dir: Path, face_prefix: str) -> list[dict]:
    """Walk the refold tree; for each prediction, pair the CIF with its
    confidence JSON and tag with the design_id."""
    rows: list[dict] = []
    for conf_json in sorted(refold_dir.rglob("*_summary_confidence_sample_*.json")):
        name_with_sample = conf_json.stem.replace("_summary_confidence", "")
        # name_with_sample looks like: <design_id_flat>_sample_<i>
        # The design_id_flat is everything before "_sample_<digit(s)>".
        parts = name_with_sample.rsplit("_sample_", 1)
        if len(parts) != 2:
            continue
        design_flat, sample_idx = parts
        if not design_flat.startswith(face_prefix):
            continue
        cif = conf_json.parent / f"{design_flat}_sample_{sample_idx}.cif"
        if not cif.is_file():
            continue
        try:
            data = json.loads(conf_json.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({
            "design_id": design_flat,
            "sample": int(sample_idx),
            "cif": str(cif.resolve()),
            "iptm": float(data.get("iptm", 0.0)),
            "ptm": float(data.get("ptm", 0.0)),
            "plddt": float(data.get("plddt", 0.0)),
            "ranking_score": float(data.get("ranking_score", 0.0)),
            "disorder": float(data.get("disorder", 0.0)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refold-dir", type=Path, required=True)
    parser.add_argument("--face-prefix", required=True,
                        help="Filter by design_id prefix, e.g. '3erd_b2' (face-1) or '3erd2' (face-2)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = _gather(args.refold_dir, args.face_prefix)
    if not rows:
        raise SystemExit(f"no refold predictions found under {args.refold_dir} matching prefix {args.face_prefix!r}")

    rows.sort(key=lambda r: r["ranking_score"], reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "design_id", "sample", "ranking_score", "iptm", "ptm",
                    "plddt", "disorder", "cif"])
        for i, r in enumerate(rows, start=1):
            w.writerow([
                i, r["design_id"], r["sample"],
                f"{r['ranking_score']:.4f}", f"{r['iptm']:.4f}", f"{r['ptm']:.4f}",
                f"{r['plddt']:.2f}", f"{r['disorder']:.4f}", r["cif"],
            ])
    print(f"wrote {args.out} ({len(rows)} predictions)")
    print("top 5:")
    for i, r in enumerate(rows[:5], start=1):
        print(f"  {i:>2}. {r['design_id']}  rank={r['ranking_score']:.3f}  "
              f"ipTM={r['iptm']:.3f}  pTM={r['ptm']:.3f}  pLDDT={r['plddt']:.1f}")


if __name__ == "__main__":
    main()
