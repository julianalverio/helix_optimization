from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..curation import paths
from .config import TensorsConfig

CANONICAL_DROP_REASONS = {
    "contains_glycan",
    "contains_nucleic_acid",
    "contains_d_amino_acid",
    "contains_modified_residue",
    "non_protein_at_interface",
    "ca_only_structure",
    "insufficient_protein_chains_after_processing",
    "unk_dominated_structure",
    "unparseable_mmcif",
    "assembly_expansion_failed",
    "dssp_failed",
    "processing_error",
    "batch_retry_exhausted",
}

EXPECTED_MANIFEST_COLUMNS = {
    "pdb_id", "assembly_id", "processing_status", "drop_reason",
    "method", "resolution", "r_free", "deposition_date", "release_date",
    "n_chains_processed", "n_substantive_chains", "path_tensor",
    "pipeline_version", "config_hash", "processing_date",
}

EXPECTED_NPZ_KEYS = {
    "n_chains", "n_max_residues", "residue_index", "residue_type",
    "ss_3", "ss_8", "coordinates", "atom_mask", "protein_chain_names",
    "cofactor_coords", "cofactor_atom_names", "cofactor_elements",
    "cofactor_residue_names", "cofactor_residue_indices", "cofactor_chain_names",
}


def build_summary_report(
    manifest_df: pd.DataFrame,
    cfg: TensorsConfig,
    output_dir: Path,
    wall_time_sec: float | None = None,
) -> Path:
    total = len(manifest_df)
    status_counts = manifest_df["processing_status"].value_counts().to_dict()
    ok = int(status_counts.get("ok", 0))
    dropped = int(status_counts.get("dropped", 0))
    error = int(status_counts.get("error", 0))

    drop_reasons = (
        manifest_df[manifest_df["processing_status"] != "ok"]["drop_reason"]
        .value_counts(dropna=False)
        .to_dict()
    )

    lines: list[str] = []
    lines.append(f"# Tensors Pipeline Summary ({pd.Timestamp.utcnow().date().isoformat()})")
    lines.append("")
    lines.append(f"- Total entries: **{total}**")
    lines.append(f"- Processed successfully: **{ok}**")
    lines.append(f"- Dropped: **{dropped}**")
    lines.append(f"- Errors: **{error}**")
    if wall_time_sec is not None:
        lines.append(f"- Wall time: **{wall_time_sec / 60.0:.1f} min**")
    lines.append("")
    lines.append("## Drops and errors by reason")
    for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        pct = (count / total * 100) if total else 0.0
        lines.append(f"- `{reason}`: **{count}** ({pct:.2f}%)")

    dssp_fail_count = int(drop_reasons.get("dssp_failed", 0))
    lines.append("")
    lines.append(f"- Whole-DSSP failures: **{dssp_fail_count}**")

    ok_rows = manifest_df[manifest_df["processing_status"] == "ok"]
    if len(ok_rows):
        n_sub = ok_rows["n_substantive_chains"].dropna()
        if len(n_sub):
            lines.append("")
            lines.append("## Successful entries — n_substantive_chains")
            lines.append(f"- mean/median/min/max: {n_sub.mean():.2f} / {n_sub.median():.0f} / {int(n_sub.min())} / {int(n_sub.max())}")

        n_max = _collect_n_max_residues(ok_rows, output_dir)
        if n_max.size:
            lines.append("")
            lines.append("## Successful entries — n_max_residues")
            lines.append(f"- mean/median/min/max: {n_max.mean():.2f} / {np.median(n_max):.0f} / {int(n_max.min())} / {int(n_max.max())}")

    lines.append("")
    lines.append(f"- Pipeline version: `{manifest_df.iloc[0]['pipeline_version'] if total else ''}`")
    lines.append(f"- Config hash: `{manifest_df.iloc[0]['config_hash'] if total else ''}`")

    out_path = output_dir / "summary_report.md"
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_path)
    return out_path


def _collect_n_max_residues(ok_rows: pd.DataFrame, output_dir: Path) -> np.ndarray:
    values: list[int] = []
    for rel in ok_rows["path_tensor"].dropna():
        path = output_dir / rel
        if not path.exists():
            continue
        try:
            data = np.load(path)
            values.append(int(data["n_max_residues"]))
        except Exception:
            continue
    return np.array(values, dtype=np.int32)


def build_test_summary(manifest_df: pd.DataFrame, cfg: TensorsConfig, output_dir: Path) -> Path:
    lines: list[str] = ["# Test Summary", ""]
    all_pass = True

    ok_rows = manifest_df[manifest_df["processing_status"] == "ok"]
    if len(ok_rows) >= 1:
        lines.append(f"- [x] At least one entry successfully processed ({len(ok_rows)})")
    else:
        lines.append("- [ ] At least one entry successfully processed (NONE)")
        all_pass = False

    seen_reasons = set(manifest_df[manifest_df["processing_status"] != "ok"]["drop_reason"].dropna())
    unknown = seen_reasons - CANONICAL_DROP_REASONS
    if not unknown:
        lines.append("- [x] All drop_reason values are canonical")
    else:
        lines.append(f"- [ ] Unknown drop reasons: {sorted(unknown)}")
        all_pass = False

    missing_cols = EXPECTED_MANIFEST_COLUMNS - set(manifest_df.columns)
    if not missing_cols:
        lines.append("- [x] Manifest has all expected columns")
    else:
        lines.append(f"- [ ] Missing manifest columns: {sorted(missing_cols)}")
        all_pass = False

    sample_ok, sample_message = _sample_npz_check(ok_rows, output_dir)
    if sample_ok:
        lines.append(f"- [x] Sample tensor file loadable: {sample_message}")
    else:
        lines.append(f"- [ ] Tensor sample check failed: {sample_message}")
        all_pass = False

    shapes_ok, shapes_message = _shape_check(ok_rows, output_dir)
    if shapes_ok:
        lines.append(f"- [x] Tensor shapes/dtypes match spec: {shapes_message}")
    else:
        lines.append(f"- [ ] Tensor shape/dtype check failed: {shapes_message}")
        all_pass = False

    lines.append("")
    lines.append(f"## Overall: {'PASS' if all_pass else 'FAIL'}")

    out_path = output_dir / "test_summary.md"
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_path)
    return out_path


def _sample_npz_check(ok_rows: pd.DataFrame, output_dir: Path) -> tuple[bool, str]:
    if not len(ok_rows):
        return False, "no successful entries"
    first = ok_rows.iloc[0]
    rel = first["path_tensor"]
    if not rel:
        return False, "path_tensor is null"
    path = output_dir / rel
    if not path.exists():
        return False, f"file missing: {path}"
    try:
        data = np.load(path)
        keys = set(data.files)
    except Exception as exc:
        return False, f"load error: {exc}"
    missing = EXPECTED_NPZ_KEYS - keys
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    return True, f"{rel} keys={sorted(keys)}"


def _shape_check(ok_rows: pd.DataFrame, output_dir: Path) -> tuple[bool, str]:
    if not len(ok_rows):
        return False, "no successful entries"
    path = output_dir / ok_rows.iloc[0]["path_tensor"]
    data = np.load(path)
    coords = data["coordinates"]
    atom_mask = data["atom_mask"]
    if coords.dtype != np.float16:
        return False, f"coordinates dtype {coords.dtype} != float16"
    if coords.shape[-2:] != (14, 3):
        return False, f"coordinates shape {coords.shape} last-2 != (14, 3)"
    if atom_mask.dtype != np.int8:
        return False, f"atom_mask dtype {atom_mask.dtype} != int8"
    values = set(np.unique(atom_mask).tolist())
    if not values.issubset({-1, 0, 1}):
        return False, f"atom_mask values {values} not subset of -1/0/1"
    return True, f"coordinates {coords.shape} {coords.dtype}; atom_mask {atom_mask.shape} {atom_mask.dtype}"
