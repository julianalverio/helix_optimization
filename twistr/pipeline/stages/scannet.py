"""ScanNet stage: per-residue Docker scoring → per-patch annotation."""
from __future__ import annotations

import logging

import pandas as pd

from twistr.pipeline.scannet_filter.filter import filter_patches_for_pdb
from twistr.pipeline.scannet_filter.scannet_runner import run_scannet

from .common import stage_pdb, stamp_pdb_path, write_parquet
from .context import PipelineContext

logger = logging.getLogger(__name__)


def run_batch(ctx: PipelineContext) -> None:
    if ctx.masif_rows:
        masif_df = pd.DataFrame(ctx.masif_rows)
    elif ctx.masif_path.exists():
        masif_df = pd.read_parquet(ctx.masif_path)
    else:
        raise FileNotFoundError(
            f"ScanNet stage cannot find MaSIF patches at {ctx.masif_path}; "
            f"run the masif stage first or supply an existing masif parquet."
        )
    ctx.scannet_path.parent.mkdir(parents=True, exist_ok=True)
    if masif_df.empty:
        logger.info("ScanNet: input parquet has no rows; nothing to score")
        if not ctx.scannet_path.exists():
            write_parquet([], ctx.scannet_path)
        return

    for pdb_id, pdb_patches in masif_df.groupby("pdb_id", sort=False):
        try:
            pdb_path = ctx.work_root / pdb_id.lower() / f"{pdb_id.lower()}.pdb"
            if not pdb_path.exists():
                _, _, pdb_path = stage_pdb(pdb_id, ctx.pdb_dir, ctx.work_root)
            logger.info("%s: running ScanNet (Docker)", pdb_id)
            scannet_scores = run_scannet(
                pdb_path=pdb_path, pdb_id=pdb_id,
                scratch_dir=ctx.work_root / pdb_id.lower() / "scannet",
                image=ctx.scannet_cfg.scannet_image,
                platform=ctx.scannet_cfg.scannet_platform,
                mode=ctx.scannet_cfg.scannet_mode,
                assembly=ctx.scannet_cfg.scannet_assembly,
            )
            sn_df = filter_patches_for_pdb(
                pdb_id, pdb_patches, scannet_scores, ctx.scannet_cfg,
            )
            rows = sn_df.to_dict("records")
            stamp_pdb_path(rows, pdb_path)
            ctx.scannet_rows.extend(rows)
            write_parquet(ctx.scannet_rows, ctx.scannet_path)
            logger.info(
                "%s: ScanNet wrote %d patches → %s (cumulative %d)",
                pdb_id, len(rows), ctx.scannet_path, len(ctx.scannet_rows),
            )
        except Exception as e:
            logger.exception("%s: ScanNet stage failed", pdb_id)
            ctx.skipped.append((pdb_id, f"{type(e).__name__}: {e}"))

    if not ctx.scannet_path.exists():
        write_parquet([], ctx.scannet_path)
