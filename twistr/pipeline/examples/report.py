from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExamplesConfig
from .constants import (
    CANONICAL_DROP_REASONS,
    ENTRY_STATUS_COLUMNS,
    EXAMPLE_MANIFEST_COLUMNS,
    EXAMPLE_NPZ_KEYS,
)


def build_summary_report(
    example_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    cfg: ExamplesConfig,
    output_dir: Path,
    wall_time_sec: float | None = None,
) -> Path:
    lines: list[str] = []
    lines.append(f"# Examples Pipeline Summary ({pd.Timestamp.utcnow().date().isoformat()})")
    lines.append("")
    total_entries = len(entry_df)
    status_counts = entry_df["processing_status"].value_counts().to_dict() if total_entries else {}
    ok = int(status_counts.get("ok", 0))
    dropped = int(status_counts.get("dropped", 0))
    error = int(status_counts.get("error", 0))

    lines.append(f"- Module 2 entries considered: **{total_entries}**")
    lines.append(f"- Entries producing ≥1 example: **{ok}**")
    lines.append(f"- Entries dropped (no examples produced): **{dropped}**")
    lines.append(f"- Entries with infrastructure errors: **{error}**")
    lines.append(f"- Total training examples: **{len(example_df)}**")
    if wall_time_sec is not None:
        lines.append(f"- Wall time: **{wall_time_sec / 60.0:.1f} min**")

    if total_entries:
        lines.append("")
        lines.append("## Drop/error breakdown by reason")
        reasons = entry_df[entry_df["processing_status"] != "ok"]["drop_reason"].value_counts(dropna=False).to_dict()
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            pct = count / total_entries * 100.0
            lines.append(f"- `{reason}`: **{count}** ({pct:.2f}%)")

    if len(example_df):
        lines.append("")
        lines.append("## Example distributions")
        for col in ("helix_length", "n_helix_contacts", "n_partner_interface_residues", "n_partner_chains"):
            if col in example_df.columns:
                s = example_df[col].dropna()
                if len(s):
                    lines.append(
                        f"- `{col}`: mean={s.mean():.2f}, median={s.median():.0f}, min={int(s.min())}, max={int(s.max())}"
                    )
        sasa_fail = int((~example_df["sasa_used"].astype(bool)).sum()) if "sasa_used" in example_df.columns else 0
        lines.append(f"- Examples using SASA fallback (distance-only): **{sasa_fail}**")

    lines.append("")
    lines.append("## Known limitations")
    lines.append("- `chain_label` is gemmi's post-assembly short name from Module 2, not strictly the source mmCIF `label_asym_id`.")
    lines.append("- SASA computation considers only helix chain + partner chains; non-partner chains in the entry are excluded from the complex.")

    out_path = output_dir / "summary_report.md"
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_path)
    return out_path


def build_test_summary(
    example_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    lines: list[str] = ["# Test Summary", ""]
    all_pass = True

    if len(example_df) >= 1:
        lines.append(f"- [x] At least one example extracted ({len(example_df)})")
    else:
        lines.append("- [ ] At least one example extracted (NONE)")
        all_pass = False

    seen_reasons = set(entry_df[entry_df["processing_status"] != "ok"]["drop_reason"].dropna()) if len(entry_df) else set()
    unknown = seen_reasons - CANONICAL_DROP_REASONS
    if not unknown:
        lines.append("- [x] All drop_reason values are canonical")
    else:
        lines.append(f"- [ ] Unknown drop reasons: {sorted(unknown)}")
        all_pass = False

    missing_cols = set(EXAMPLE_MANIFEST_COLUMNS) - set(example_df.columns)
    if not missing_cols:
        lines.append("- [x] Example manifest has all expected columns")
    else:
        lines.append(f"- [ ] Missing example manifest columns: {sorted(missing_cols)}")
        all_pass = False

    missing_entry_cols = set(ENTRY_STATUS_COLUMNS) - set(entry_df.columns)
    if not missing_entry_cols:
        lines.append("- [x] Entry-status manifest has all expected columns")
    else:
        lines.append(f"- [ ] Missing entry-status columns: {sorted(missing_entry_cols)}")
        all_pass = False

    sample_ok, sample_msg = _sample_npz_check(example_df, output_dir)
    if sample_ok:
        lines.append(f"- [x] Sample example tensor loadable: {sample_msg}")
    else:
        lines.append(f"- [ ] Example tensor check failed: {sample_msg}")
        all_pass = False

    shape_ok, shape_msg = _shape_check(example_df, output_dir)
    if shape_ok:
        lines.append(f"- [x] Tensor shapes/dtypes match spec: {shape_msg}")
    else:
        lines.append(f"- [ ] Tensor shape/dtype check failed: {shape_msg}")
        all_pass = False

    order_ok, order_msg = _ordering_check(example_df, output_dir)
    if order_ok:
        lines.append(f"- [x] Helix-first ordering invariant holds: {order_msg}")
    else:
        lines.append(f"- [ ] Helix ordering check failed: {order_msg}")
        all_pass = False

    lines.append("")
    lines.append(f"## Overall: {'PASS' if all_pass else 'FAIL'}")

    out_path = output_dir / "test_summary.md"
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out_path)
    return out_path


def _first_loadable(example_df: pd.DataFrame, output_dir: Path):
    for _, row in example_df.iterrows():
        rel = row.get("path_example")
        if not rel:
            continue
        path = output_dir / str(rel)
        if not path.exists():
            continue
        try:
            return path, np.load(path)
        except Exception:
            continue
    return None, None


def _sample_npz_check(example_df: pd.DataFrame, output_dir: Path) -> tuple[bool, str]:
    if not len(example_df):
        return False, "no examples"
    path, data = _first_loadable(example_df, output_dir)
    if data is None:
        return False, "no loadable example"
    missing = EXAMPLE_NPZ_KEYS - set(data.files)
    if missing:
        return False, f"missing npz keys: {sorted(missing)}"
    return True, f"{path.name} keys={len(data.files)}"


def _shape_check(example_df: pd.DataFrame, output_dir: Path) -> tuple[bool, str]:
    if not len(example_df):
        return False, "no examples"
    path, data = _first_loadable(example_df, output_dir)
    if data is None:
        return False, "no loadable example"
    coords = data["coordinates"]
    amask = data["atom_mask"]
    if coords.dtype != np.float16:
        return False, f"coordinates dtype {coords.dtype} != float16"
    if coords.ndim != 3 or coords.shape[-2:] != (14, 3):
        return False, f"coordinates shape {coords.shape} != (n, 14, 3)"
    if amask.dtype != np.int8:
        return False, f"atom_mask dtype {amask.dtype} != int8"
    values = set(np.unique(amask).tolist())
    if not values.issubset({-1, 0, 1}):
        return False, f"atom_mask values {values} not subset of -1/0/1"
    return True, f"coords {coords.shape}; atom_mask {amask.shape}"


def _ordering_check(example_df: pd.DataFrame, output_dir: Path) -> tuple[bool, str]:
    if not len(example_df):
        return False, "no examples"
    path, data = _first_loadable(example_df, output_dir)
    if data is None:
        return False, "no loadable example"
    is_helix = data["is_helix"].astype(bool)
    chain_slot = data["chain_slot"]
    if not is_helix.any():
        return False, "no helix residues flagged"
    first_false = int(np.argmin(is_helix)) if not bool(is_helix[-1]) else len(is_helix)
    if not bool(is_helix[0]):
        return False, "chain_slot[0] is not helix"
    if first_false < len(is_helix) and is_helix[first_false:].any():
        return False, "non-contiguous helix region"
    if int(chain_slot[0]) != 0:
        return False, f"chain_slot[0] == {int(chain_slot[0])} != 0"
    if np.any(np.diff(chain_slot) < 0):
        return False, "chain_slot not monotonically non-decreasing"
    return True, f"n_helix={int(is_helix.sum())}, n_total={len(is_helix)}"
