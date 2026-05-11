from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import paths
from .config import Config

FILTER_COLUMNS = [
    "passed_status_filter",
    "passed_method_filter",
    "passed_resolution_filter",
    "passed_rfree_filter",
    "passed_chains_filter",
    "passed_protein_chain_filter",
    "passed_protein_length_filter",
    "passed_date_filter",
    "passed_size_cap_filter",
]


def build_report(cfg: Config, data_root_path: Path, snapshot_date: datetime) -> Path:
    manifests = paths.manifests_dir(data_root_path)
    audit = pd.read_parquet(manifests / "candidates_audit.parquet")
    final = pd.read_parquet(manifests / "module1_manifest.parquet")

    total = len(audit)
    lines = [
        f"# Module 1 Summary ({snapshot_date.date().isoformat()})",
        "",
        f"- Total entries considered: **{total}**",
        f"- Passed all metadata filters: **{int(audit['passed_all_filters'].sum())}**",
        f"- Final dataset size: **{len(final)}**",
        "",
        "## Drops by reason",
    ]
    dropped = audit[audit["drop_reason"].notna()]
    reason_counts = dropped["drop_reason"].value_counts().to_dict()
    for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{reason}`: **{count}**")
    if not reason_counts:
        lines.append("- (no drops)")

    lines += [
        "",
        "## Drops by Phase A filter (per-filter booleans)",
    ]
    for col in FILTER_COLUMNS:
        failed = int((~audit[col].fillna(True)).sum())
        lines.append(f"- {col}: **{failed}** ({100 * failed / total:.1f}%)" if total else f"- {col}: 0")
    attempted = audit[audit["passed_all_filters"]]
    parse_failed = int((~attempted["passed_parse"].fillna(False)).sum())
    obs_failed = int((~attempted["passed_observed_fraction_filter"].fillna(False)).sum())
    lines.append(f"- passed_parse (of {len(attempted)} attempted): **{parse_failed}** failed")
    lines.append(f"- passed_observed_fraction_filter (of {len(attempted)} attempted): **{obs_failed}** failed")

    lines += [
        "",
        "## Final dataset distributions",
    ]
    if not final.empty:
        method_counts = final["method"].value_counts(dropna=False).to_dict()
        lines.append(f"- Methods: {method_counts}")
        lines.append(
            f"- Resolution (mean/min/max): "
            f"{final['resolution'].mean():.2f} / {final['resolution'].min():.2f} / {final['resolution'].max():.2f}"
        )
        lines.append(
            f"- Chain count (mean/min/max): "
            f"{final['n_instantiated_polymer_chains'].mean():.1f} / "
            f"{final['n_instantiated_polymer_chains'].min()} / "
            f"{final['n_instantiated_polymer_chains'].max()}"
        )
        lines.append(f"- Large assemblies: **{int(final['large_assembly'].sum())}**")
    else:
        lines.append("- Final dataset is empty.")

    lines += [
        "",
        "## Manual-review candidates",
    ]
    parse_errors = audit["parse_error"].notna() if "parse_error" in audit.columns else False
    obs_failed_mask = ~audit.get("passed_observed_fraction_filter", pd.Series(True, index=audit.index)).fillna(True)
    flagged = audit[
        audit["passed_all_filters"]
        & (audit["rfree_missing"] | parse_errors | obs_failed_mask)
    ]["pdb_id"].tolist()
    lines.append(f"- {len(flagged)} entries flagged: {flagged[:20]}{'...' if len(flagged) > 20 else ''}")

    out = manifests / "report.md"
    out.write_text("\n".join(lines) + "\n")
    return out
