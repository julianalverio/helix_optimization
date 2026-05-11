from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import paths
from .config import (
    PIPELINE_VERSION,
    Config,
    config_as_dict,
    config_hash,
)

logger = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "pdb_id",
    "file_path",
    "sha256",
    "method",
    "resolution",
    "r_free",
    "rfree_missing",
    "multi_method",
    "n_polymer_entities",
    "n_instantiated_polymer_chains",
    "n_protein_chains",
    "has_dna",
    "has_rna",
    "has_ligands",
    "has_modified_residues",
    "has_short_peptide",
    "deposition_date",
    "release_date",
    "primary_assembly_id",
    "large_assembly",
    "n_unique_interfaces",
    "unique_interface_plan",
    "max_protein_observed_fraction",
    "min_protein_observed_fraction",
    "status",
    "obsoleted_from",
    "pipeline_version",
    "config_hash",
    "snapshot_date",
]


def _write_atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _compute_drop_reason(row, min_observed_fraction: float) -> str | None:
    phase_a = row.get("phase_a_drop_reason")
    if isinstance(phase_a, str) and phase_a:
        return phase_a
    if not bool(row.get("file_present", False)):
        return "download_missing"
    if not bool(row.get("parse_ok", False)):
        err = row.get("parse_error")
        err = err if isinstance(err, str) else ""
        type_part = err.split(":", 1)[0] if err else "unknown"
        return f"parse_error:{type_part}"
    max_obs = row.get("max_protein_observed_fraction")
    if max_obs is None or pd.isna(max_obs) or max_obs < min_observed_fraction:
        return "filter:observed_fraction"
    return None


def build_final_manifest(
    cfg: Config,
    data_root_path: Path,
    snapshot_date: datetime,
) -> tuple[Path, Path]:
    manifests = paths.manifests_dir(data_root_path)
    candidates = pd.read_parquet(manifests / "candidates.parquet")
    verify = pd.read_parquet(manifests / "verify_results.parquet")

    verify = verify.rename(
        columns={
            "method": "method_from_file",
            "resolution": "resolution_from_file",
            "r_free": "r_free_from_file",
        }
    )
    merged = candidates.merge(verify, on="pdb_id", how="left")

    pdb_root = paths.pdb_dir(data_root_path)
    merged["file_present"] = merged["pdb_id"].map(
        lambda pid: paths.mmcif_abs_path(data_root_path, pid).exists()
    )

    merged["drop_reason"] = merged.apply(
        lambda r: _compute_drop_reason(r, cfg.min_observed_fraction), axis=1
    )
    parse_ok = merged["parse_ok"].fillna(False)
    max_obs = merged["max_protein_observed_fraction"].fillna(0.0)
    passed_observed = max_obs >= cfg.min_observed_fraction
    merged["passed_parse"] = parse_ok
    merged["passed_observed_fraction_filter"] = passed_observed
    merged["passed_download"] = merged["file_present"]
    merged["passed_all_filters_final"] = merged["drop_reason"].isna()

    download_missing = merged[merged["drop_reason"] == "download_missing"]["pdb_id"].tolist()
    for pdb_id in download_missing:
        logger.warning("download missing for %s", pdb_id)

    audit = merged.copy()
    audit_path = manifests / "candidates_audit.parquet"
    _write_atomic_parquet(audit, audit_path)

    final_mask = merged["drop_reason"].isna()
    final = merged[final_mask].copy()
    final["pipeline_version"] = PIPELINE_VERSION
    final["config_hash"] = config_hash(cfg)
    final["snapshot_date"] = snapshot_date.date().isoformat()
    if "file_path" not in final.columns or final["file_path"].isna().any():
        final["file_path"] = final["pdb_id"].map(paths.mmcif_rel_path)
    else:
        final["file_path"] = final["pdb_id"].map(paths.mmcif_rel_path)

    if "unique_interface_plan" not in final.columns:
        final["unique_interface_plan"] = None
    if "n_unique_interfaces" not in final.columns:
        final["n_unique_interfaces"] = final["unique_interface_plan"].apply(
            lambda p: len(p) if isinstance(p, list) else 0
        )

    for col in FINAL_COLUMNS:
        if col not in final.columns:
            final[col] = None
    final = final[FINAL_COLUMNS]

    final_path = manifests / "module1_manifest.parquet"
    _write_atomic_parquet(final, final_path)

    metadata = {
        "snapshot_date": snapshot_date.isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": config_hash(cfg),
        "config": config_as_dict(cfg),
        "interface_source": "rcsb_interface_cluster",
    }
    (manifests / "module1_manifest.meta.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    return final_path, audit_path
