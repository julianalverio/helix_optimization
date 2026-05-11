"""Hotspot stage: critires.sh on Modal (cached) → per-patch hotspot annotation."""
from __future__ import annotations

import json
import logging

import pandas as pd

from twistr.epitope_selection._cache import is_valid as _cache_valid, mark as _cache_mark, signature as _cache_signature
from twistr.epitope_selection.hotspot_filter.filter import filter_patches_for_pdb
from twistr.epitope_selection.scannet_filter.filter import parse_residue_id

from .common import (
    _CRITIRES_VERSION, build_records, clean_pdb_for_critires, stage_pdb,
    stamp_pdb_path, write_parquet,
)
from .context import PipelineContext

logger = logging.getLogger(__name__)


def run_batch(ctx: PipelineContext) -> None:
    if ctx.scannet_rows:
        sn_df = pd.DataFrame(ctx.scannet_rows)
    elif ctx.scannet_path.exists():
        sn_df = pd.read_parquet(ctx.scannet_path)
    else:
        raise FileNotFoundError(
            f"Hotspot stage cannot find ScanNet patches at {ctx.scannet_path}; "
            f"run the scannet stage first or supply an existing scannet parquet."
        )
    ctx.final_path.parent.mkdir(parents=True, exist_ok=True)
    if sn_df.empty:
        logger.info("Hotspot: input parquet has no rows; nothing to score")
        if not ctx.final_path.exists():
            write_parquet([], ctx.final_path)
        return

    # Lazy-import the Modal app so importing this module doesn't trigger
    # Modal SDK initialization in tests / non-pipeline callers.
    from twistr.epitope_selection.manager.modal_app import app, run_critires

    for pdb_id, pdb_patches in sn_df.groupby("pdb_id", sort=False):
        try:
            pdb_path = ctx.work_root / pdb_id.lower() / f"{pdb_id.lower()}.pdb"
            if not pdb_path.exists():
                _, _, pdb_path = stage_pdb(pdb_id, ctx.pdb_dir, ctx.work_root)
            cleaned = clean_pdb_for_critires(pdb_path.read_bytes())
            records, _, _ = build_records(pdb_path, ctx.epitopes_cfg)

            hs_scratch = ctx.work_root / pdb_id.lower() / "hotspot"
            hs_scratch.mkdir(parents=True, exist_ok=True)
            cache_file = hs_scratch / "hotspot_scores.json"
            sidecar = hs_scratch / ".cache_sig"
            sig = _cache_signature(cleaned, _CRITIRES_VERSION)
            if cache_file.exists() and _cache_valid(sidecar, sig):
                logger.info("%s: PPI-hotspot cache hit", pdb_id)
                hotspot_scores_str = json.loads(cache_file.read_text())
            else:
                logger.info("%s: running PPI-hotspot critires.sh (Modal)", pdb_id)
                with app.run():
                    hotspot_scores_str = run_critires.remote(pdb_id, cleaned)
                cache_file.write_text(json.dumps(hotspot_scores_str))
                _cache_mark(sidecar, sig)

            hotspot_scores = {
                parse_residue_id(s): v for s, v in hotspot_scores_str.items()
            }
            final_df = filter_patches_for_pdb(
                pdb_id, pdb_patches, hotspot_scores, records, ctx.hotspot_cfg,
            )
            rows = final_df.to_dict("records")
            stamp_pdb_path(rows, pdb_path)
            ctx.final_rows.extend(rows)
            write_parquet(ctx.final_rows, ctx.final_path)
            logger.info(
                "%s: hotspot wrote %d patches → %s (cumulative %d)",
                pdb_id, len(rows), ctx.final_path, len(ctx.final_rows),
            )
        except Exception as e:
            logger.exception("%s: hotspot stage failed", pdb_id)
            ctx.skipped.append((pdb_id, f"{type(e).__name__}: {e}"))

    if not ctx.final_path.exists():
        write_parquet([], ctx.final_path)
