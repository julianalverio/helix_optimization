"""Materialize a tiny on-disk slice of runtime/data/examples/ for the RunPod smoke test.

Picks the first N manifest entries, copies their NPZs (preserving the
`examples/<two_letter>/<id>.npz` layout the dataset expects), and writes a
filtered manifest pointing at them. Everything lands under
dev/tools/runpod/smoke_test/subset/ which is .gitignored — re-run any time."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "data" / "module3"
SOURCE_MANIFEST = SOURCE_ROOT / "module3_manifest.parquet"
DEST_ROOT = REPO_ROOT / "tools" / "runpod_smoke_test" / "subset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of examples to copy")
    args = parser.parse_args()

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(args.n).reset_index(drop=True)

    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    (DEST_ROOT / "examples").mkdir(parents=True)

    for rel_path in manifest["path_example"]:
        src = SOURCE_ROOT / rel_path
        dst = DEST_ROOT / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest.to_parquet(DEST_ROOT / "module3_manifest.parquet", index=False)

    # Datamodule's WeightedRandomSampler needs a (path_example, weight) parquet.
    # Uniform weight=1.0 is fine for the smoke test — we're not measuring sampling.
    clusters = pd.DataFrame({"path_example": manifest["path_example"], "weight": 1.0})
    clusters.to_parquet(DEST_ROOT / "helix_clusters.parquet", index=False)

    total_bytes = sum(p.stat().st_size for p in DEST_ROOT.rglob("*") if p.is_file())
    print(f"wrote {len(manifest)} examples to {DEST_ROOT} ({total_bytes/1024:.0f} KB)")


if __name__ == "__main__":
    main()
