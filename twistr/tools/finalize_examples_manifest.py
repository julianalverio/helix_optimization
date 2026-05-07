"""Rebuild examples-pipeline manifests from on-disk example files when the driver
couldn't complete its end-of-run write step (typically due to Modal app disconnect
during the final 1% of batches). Walks every example .npz, reconstructs both
module3_manifest.parquet and module3_entry_status.parquet, then writes them
atomically and emits a summary.

Usage:
    python -m twistr.tools.finalize_examples_manifest
"""
from __future__ import annotations

import argparse
import io
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..pipeline.examples.constants import ENTRY_STATUS_COLUMNS, EXAMPLE_MANIFEST_COLUMNS
from .. import paths


def _resolve_pipeline_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _decode(x):
    return str(x.item()) if hasattr(x, "item") else str(x)


def _scan_example(path: Path) -> dict:
    d = np.load(path)
    is_helix = d["is_helix"].astype(bool)
    is_iface = d["is_interface_residue"].astype(bool)
    chain_slot = d["chain_slot"]
    n_total = is_helix.shape[0]
    n_helix_residues = int(is_helix.sum())
    n_partner_residues = n_total - n_helix_residues
    n_partner_chains = int(len(set(int(c) for c in chain_slot[~is_helix])))
    n_helix_contacts = int((is_helix & is_iface).sum())
    n_partner_interface = int(((~is_helix) & is_iface).sum())
    return {
        "pdb_id": _decode(d["pdb_id"]),
        "assembly_id": int(d["assembly_id"]),
        "example_id": int(d["example_id"]),
        "helix_seqres_start": int(d["helix_seqres_start"]),
        "helix_seqres_end": int(d["helix_seqres_end"]),
        "helix_sequence": _decode(d["helix_sequence"]),
        "n_helix_residues": n_helix_residues,
        "n_partner_residues": n_partner_residues,
        "n_partner_chains": n_partner_chains,
        "n_helix_contacts": n_helix_contacts,
        "n_partner_interface_residues": n_partner_interface,
        "n_residues_total": n_total,
        "resolution": float(d["resolution"]) if not np.isnan(d["resolution"]) else None,
        "r_free": float(d["r_free"]) if not np.isnan(d["r_free"]) else None,
        "source_method": _decode(d["source_method"]),
        "sasa_used": bool(d["sasa_used"]),
    }


def _example_row(rec: dict, rel_path: str, pipeline_version: str, processing_date: str) -> dict:
    return {
        "example_id_full": f"{rec['pdb_id']}_{rec['assembly_id']}_{rec['example_id']}",
        "pdb_id": rec["pdb_id"],
        "assembly_id": rec["assembly_id"],
        "example_id": rec["example_id"],
        "helix_seqres_start": rec["helix_seqres_start"],
        "helix_seqres_end": rec["helix_seqres_end"],
        "helix_length": rec["helix_seqres_end"] - rec["helix_seqres_start"] + 1,
        "n_helix_residues": rec["n_helix_residues"],
        "n_partner_residues": rec["n_partner_residues"],
        "n_partner_chains": rec["n_partner_chains"],
        "n_helix_contacts": rec["n_helix_contacts"],
        "n_partner_interface_residues": rec["n_partner_interface_residues"],
        "n_residues_total": rec["n_residues_total"],
        "helix_sequence": rec["helix_sequence"],
        "resolution": rec["resolution"],
        "r_free": rec["r_free"],
        "source_method": rec["source_method"],
        "sasa_used": rec["sasa_used"],
        "path_example": rel_path,
        "pipeline_version": pipeline_version,
        "config_hash": "",
        "processing_date": processing_date,
    }


def _coerce_example_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        "example_id_full": "string",
        "pdb_id": "string",
        "helix_sequence": "string",
        "source_method": "string",
        "path_example": "string",
        "pipeline_version": "string",
        "config_hash": "string",
        "assembly_id": "int8",
        "example_id": "int32",
        "helix_seqres_start": "int32",
        "helix_seqres_end": "int32",
        "helix_length": "int16",
        "n_helix_residues": "int16",
        "n_partner_residues": "int16",
        "n_partner_chains": "int8",
        "n_helix_contacts": "int16",
        "n_partner_interface_residues": "int16",
        "n_residues_total": "int32",
        "resolution": "float32",
        "r_free": "float32",
        "sasa_used": "bool",
    })
    df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


def _coerce_entry_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        "pdb_id": "string",
        "processing_status": "string",
        "drop_reason": "string",
        "assembly_id": "int8",
        "n_helix_segments": "Int32",
        "n_interacting_helices": "Int32",
        "n_windows_before_filter": "Int32",
        "n_examples_emitted": "Int32",
    })
    df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/module3"))
    args = parser.parse_args()

    output_dir = args.output_dir
    examples_root = output_dir / "examples"
    markers_root = output_dir / "markers"

    pipeline_version = _resolve_pipeline_version()
    processing_date = datetime.now(timezone.utc).isoformat()

    print(f"Scanning {examples_root}...")
    npz_files = sorted(examples_root.rglob("*.npz"))
    print(f"  {len(npz_files)} example .npz files")

    rows: list[dict] = []
    examples_per_entry: dict[tuple[str, int], int] = {}
    t0 = time.time()
    for i, p in enumerate(npz_files, 1):
        rec = _scan_example(p)
        rel = str(p.relative_to(output_dir))
        rows.append(_example_row(rec, rel, pipeline_version, processing_date))
        key = (rec["pdb_id"], rec["assembly_id"])
        examples_per_entry[key] = examples_per_entry.get(key, 0) + 1
        if i % 50000 == 0:
            print(f"  scanned {i}/{len(npz_files)} ({(time.time() - t0):.0f}s)")

    example_df = _coerce_example_dtypes(pd.DataFrame(rows, columns=EXAMPLE_MANIFEST_COLUMNS))
    example_path = output_dir / "module3_manifest.parquet"
    tmp = example_path.with_suffix(".parquet.tmp")
    example_df.to_parquet(tmp, index=False)
    tmp.replace(example_path)
    print(f"Wrote {example_path} ({len(example_df)} rows)")

    print(f"Scanning {markers_root}...")
    marker_files = sorted(markers_root.rglob("*.marker"))
    print(f"  {len(marker_files)} markers")

    entry_rows: list[dict] = []
    for m in marker_files:
        stem = m.stem
        try:
            pdb_id, asm = stem.rsplit("_", 1)
            pdb_id = pdb_id.upper()
            assembly_id = int(asm)
        except ValueError:
            continue
        n_emitted = examples_per_entry.get((pdb_id, assembly_id), 0)
        if n_emitted > 0:
            status, reason = "ok", None
        else:
            status, reason = "dropped", "no_surviving_windows"
        entry_rows.append({
            "pdb_id": pdb_id,
            "assembly_id": assembly_id,
            "processing_status": status,
            "drop_reason": reason,
            "n_helix_segments": None,
            "n_interacting_helices": None,
            "n_windows_before_filter": None,
            "n_examples_emitted": n_emitted,
            "processing_date": processing_date,
        })

    entry_df = _coerce_entry_dtypes(pd.DataFrame(entry_rows, columns=ENTRY_STATUS_COLUMNS))
    entry_path = output_dir / "module3_entry_status.parquet"
    tmp = entry_path.with_suffix(".parquet.tmp")
    entry_df.to_parquet(tmp, index=False)
    tmp.replace(entry_path)
    print(f"Wrote {entry_path} ({len(entry_df)} rows)")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {len(example_df)}")
    print(f"Total entries with markers: {len(entry_df)}")
    print(f"Entries producing >=1 example: {sum(1 for v in examples_per_entry.values() if v > 0)}")
    print(f"Entries with 0 examples: {sum(1 for v in entry_df['n_examples_emitted'] if int(v) == 0)}")
    if len(example_df):
        print()
        print("helix_length:")
        print(example_df["helix_length"].describe().to_string())
        print()
        print("n_helix_contacts:")
        print(example_df["n_helix_contacts"].describe().to_string())
        print()
        print("n_partner_chains distribution:")
        print(example_df["n_partner_chains"].value_counts().sort_index().to_string())
        print()
        print(f"sasa_used: {int(example_df['sasa_used'].sum())} of {len(example_df)} examples")


if __name__ == "__main__":
    main()
