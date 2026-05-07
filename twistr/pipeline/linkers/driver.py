"""End-to-end driver: 4 per-linker Remodel jobs in random order, then
splice the best of each into a final assembled PDB."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from .blueprint import write_blueprint
from .config import LinkersConfig, load_linkers_config
from .pose_builder import (
    SubposeLayout,
    assemble_full_pose,
    build_all_subposes,
)
from .remodel_runner import run_remodel

logger = logging.getLogger(__name__)

LINKER_IDS = ('linker1', 'linker2', 'linker3', 'linker4')


def _pick_best(scores: list[dict]) -> dict | None:
    successes = [s for s in scores if s["error"] is None and s["path"] is not None]
    if not successes:
        return None
    return min(successes, key=lambda s: s["total_score"])


def run_linkers(cfg: LinkersConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    work_dir = out_dir / "subposes"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building per-linker sub-poses → %s", work_dir)
    subposes: dict[str, tuple[Path, SubposeLayout]] = build_all_subposes(cfg, work_dir)

    rng = random.Random(cfg.seed)
    order = list(LINKER_IDS)
    rng.shuffle(order)
    logger.info("Linker design order (seed=%d): %s", cfg.seed, order)

    rows: list[dict] = []
    chosen: dict[str, Path] = {}
    for lid in order:
        sub_pdb, layout = subposes[lid]
        bp_path = work_dir / lid / "linker.blueprint"
        write_blueprint(layout, cfg.linker_aa_whitelist, bp_path)

        designs_dir = out_dir / "designs" / lid
        logger.info("[%s] designing %d candidates × %d trajectories",
                    lid, cfg.nstruct, cfg.num_trajectory)
        scores = run_remodel(
            rosetta_python=cfg.rosetta_python,
            subpose_pdb=sub_pdb,
            blueprint=bp_path,
            out_dir=designs_dir,
            nstruct=cfg.nstruct,
            num_trajectory=cfg.num_trajectory,
            linker_lo=layout.linker[0],
            linker_hi=layout.linker[1],
        )
        best = _pick_best(scores)
        if best is None:
            raise RuntimeError(
                f"{lid}: no successful designs out of {cfg.nstruct} attempts. "
                f"See {designs_dir}/remodel.log."
            )
        chosen[lid] = Path(best["path"])
        logger.info("[%s] best design: %s (score=%.3f)",
                    lid, chosen[lid].name, best["total_score"])
        for s in scores:
            rows.append({
                "linker_id": lid,
                "design_index": s["index"],
                "path": s["path"],
                "total_score": s["total_score"],
                "error": s["error"],
                "chosen": s["path"] == best["path"],
            })

    layouts = {lid: subposes[lid][1] for lid in LINKER_IDS}
    final_pdb = out_dir / "final.pdb"
    assemble_full_pose(cfg, chosen, layouts, final_pdb)
    logger.info("Wrote final assembled pose: %s", final_pdb)

    designs_parquet = out_dir / "designs.parquet"
    pd.DataFrame(rows).to_parquet(designs_parquet, index=False)
    logger.info("Wrote design summary: %s", designs_parquet)

    return final_pdb


def main(config_path: Path) -> Path:
    cfg = load_linkers_config(config_path)
    return run_linkers(cfg)
