"""MaSIF stage: per-PDB Docker run → residue-graph patches → parquet row(s)."""
from __future__ import annotations

import json
import logging

from twistr.epitope_selection.epitopes.masif_runner import run_masif_site
from twistr.epitope_selection.epitopes.patches import extract_patches, patches_to_rows

from .common import (
    build_records, stage_pdb, stamp_pdb_path, write_diagnostics, write_parquet,
)
from .context import PipelineContext

logger = logging.getLogger(__name__)


def run_batch(ctx: PipelineContext) -> None:
    if ctx.pdb_ids is None:
        raise ValueError(
            "MaSIF stage requires a PDB list; pass --pdb-list to epitope-selection-run"
        )
    ctx.masif_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = ctx.masif_path.parent / f"{ctx.masif_path.stem}_debug"

    for pdb_id in ctx.pdb_ids:
        diag: dict = {"pdb_id": pdb_id, "stages": {}, "error": None}
        try:
            pdb_bytes, chains, pdb_path = stage_pdb(pdb_id, ctx.pdb_dir, ctx.work_root)
            if any(len(c) != 1 for c in chains):
                raise RuntimeError(f"multi-char chains not supported: {chains}")
            chain_concat = "".join(chains)
            logger.info("%s: PDB staged at %s; running MaSIF (Docker)", pdb_id, pdb_path)

            vertices, faces, scores = run_masif_site(
                pdb_path, pdb_id, chain_concat,
                scratch_dir=ctx.work_root / pdb_id.lower() / "masif",
                image=ctx.epitopes_cfg.masif_image,
                platform=ctx.epitopes_cfg.masif_platform,
            )
            records, core, halo = build_records(pdb_path, ctx.epitopes_cfg)
            diag["stages"]["dssp"] = {
                "n_records": len(records), "n_core": len(core), "n_halo": len(halo),
            }
            diag["stages"]["masif"] = {
                "n_vertices": int(vertices.shape[0]),
                "score_max": float(scores.max() if scores.size else 0.0),
            }

            masif_diag: dict = {"pdb_id": pdb_id, "pdb_path": str(pdb_path.resolve())}
            masif_patches = extract_patches(
                vertices, faces, scores, records, core, halo, ctx.epitopes_cfg,
                diag=masif_diag,
            )
            diag["stages"]["masif_patches"] = len(masif_patches)
            rows = patches_to_rows(pdb_id, chain_concat, masif_patches)
            stamp_pdb_path(rows, pdb_path)
            ctx.masif_rows.extend(rows)
            write_parquet(ctx.masif_rows, ctx.masif_path)

            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{pdb_id.lower()}.json").write_text(
                json.dumps(masif_diag, indent=2, default=str))
            logger.info(
                "%s: MaSIF wrote %d patches → %s (cumulative %d); debug → %s",
                pdb_id, len(rows), ctx.masif_path, len(ctx.masif_rows),
                debug_dir / f"{pdb_id.lower()}.json",
            )
        except Exception as e:
            logger.exception("%s: MaSIF stage failed", pdb_id)
            err = f"{type(e).__name__}: {e}"
            ctx.skipped.append((pdb_id, err))
            diag["error"] = err
        ctx.diagnostics.append(diag)
        write_diagnostics(ctx.diagnostics, ctx.diagnostics_path)

    if not ctx.masif_path.exists():
        write_parquet([], ctx.masif_path)
