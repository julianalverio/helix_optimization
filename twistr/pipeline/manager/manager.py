"""Pipeline orchestrator: run the stages listed in `cfg.stages` in order.

Stages communicate via on-disk parquets, so any subset can run alone (e.g.
`stages: [viz]` regenerates .pml files from an existing final parquet).
`--pdb-list` is required only when the first stage is `masif`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from twistr.pipeline.epitope_viz.config import load_epitope_viz_config
from twistr.pipeline.epitopes.config import EpitopesConfig
from twistr.pipeline.hotspot_filter.config import HotspotConfig
from twistr.pipeline.scannet_filter.config import ScanNetConfig
from twistr.pipeline.stages import hotspot, masif, scannet, viz
from twistr.pipeline.stages.common import (
    derive_pdb_ids, write_diagnostics, write_parquet,
)
from twistr.pipeline.stages.context import PipelineContext

logger = logging.getLogger(__name__)


_STAGE_REGISTRY = {
    "masif": masif.run_batch,
    "scannet": scannet.run_batch,
    "hotspot": hotspot.run_batch,
    "viz": viz.run_batch,
}


@dataclass(frozen=True)
class PipelineConfig:
    stages: list[str]
    epitopes_config: str
    scannet_config: str
    hotspot_config: str
    epitope_viz_config: str
    output_path: str
    pdb_dir: str = "data/pdb"
    masif_output_path: str = ""
    scannet_output_path: str = ""


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(PipelineConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown pipeline config keys: {sorted(unknown)}")
    cfg = PipelineConfig(**raw)
    if not cfg.stages:
        raise ValueError("`stages` must be a non-empty list")
    bad = [s for s in cfg.stages if s not in _STAGE_REGISTRY]
    if bad:
        raise ValueError(
            f"Unknown stage(s): {bad}. Valid: {sorted(_STAGE_REGISTRY)}"
        )
    return cfg


def read_pdb_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _build_context(cfg: PipelineConfig, pdb_list_path: Path | None) -> PipelineContext:
    sub = {}
    for key, path in [("epitopes", cfg.epitopes_config),
                      ("scannet", cfg.scannet_config),
                      ("hotspot", cfg.hotspot_config)]:
        with open(path) as f:
            sub[key] = yaml.safe_load(f) or {}

    out_path = Path(cfg.output_path)
    masif_path = Path(cfg.masif_output_path or sub["epitopes"]["output_path"])
    scannet_path = Path(cfg.scannet_output_path or sub["scannet"]["output_path"])
    for p in (masif_path, scannet_path, out_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    work_root = Path(sub["epitopes"].get("work_dir", "data/epitopes/work"))
    work_root.mkdir(parents=True, exist_ok=True)

    viz_cfg = replace(
        load_epitope_viz_config(cfg.epitope_viz_config),
        patches_parquet=str(out_path),
    )
    pdb_ids = read_pdb_list(pdb_list_path) if pdb_list_path is not None else None
    return PipelineContext(
        pdb_ids=pdb_ids,
        pdb_dir=Path(cfg.pdb_dir),
        work_root=work_root,
        masif_path=masif_path,
        scannet_path=scannet_path,
        final_path=out_path,
        epitopes_cfg=EpitopesConfig(**sub["epitopes"]),
        scannet_cfg=ScanNetConfig(**sub["scannet"]),
        hotspot_cfg=HotspotConfig(**sub["hotspot"]),
        viz_cfg=viz_cfg,
        diagnostics_path=out_path.parent / f"{out_path.stem}.diagnostics.json",
        error_log=out_path.parent / f"{out_path.stem}.errors.log",
    )


def run_pipeline(pdb_list_path: Path | None, config_path: Path) -> Path:
    cfg = load_pipeline_config(config_path)
    if cfg.stages[0] == "masif" and pdb_list_path is None:
        raise ValueError("--pdb-list is required when the first stage is 'masif'")
    ctx = _build_context(cfg, pdb_list_path)
    if ctx.pdb_ids is None:
        # Pre-populate from upstream parquet so the per-PDB summary log can
        # report a real count.
        for stage, parquet in [("scannet", ctx.masif_path),
                                ("hotspot", ctx.scannet_path),
                                ("viz", ctx.final_path)]:
            if stage in cfg.stages:
                ctx.pdb_ids = derive_pdb_ids(parquet)
                break
    logger.info(
        "Pipeline: stages=%s, %s PDBs → %s",
        cfg.stages,
        "?" if ctx.pdb_ids is None else str(len(ctx.pdb_ids)),
        cfg.output_path,
    )

    for stage_name in cfg.stages:
        logger.info("=== stage: %s ===", stage_name)
        _STAGE_REGISTRY[stage_name](ctx)

    for path in (ctx.masif_path, ctx.scannet_path, ctx.final_path):
        if not path.exists():
            write_parquet([], path)

    write_diagnostics(ctx.diagnostics, ctx.diagnostics_path)
    with ctx.error_log.open("w") as ef:
        for pid, err in ctx.skipped:
            ef.write(f"{pid}\tStageError\t{err}\n")

    summary = [f"Pipeline complete (stages run: {cfg.stages}):"]
    if "masif" in cfg.stages:
        summary.append(f"  masif:   {len(ctx.masif_rows)} rows → {ctx.masif_path}")
    if "scannet" in cfg.stages:
        summary.append(f"  scannet: {len(ctx.scannet_rows)} rows → {ctx.scannet_path}")
    if "hotspot" in cfg.stages:
        summary.append(f"  final:   {len(ctx.final_rows)} rows → {ctx.final_path}")
    if "viz" in cfg.stages:
        pml_dir = Path(ctx.viz_cfg.output_dir)
        n_pml = len(list(pml_dir.glob("*.pml"))) if pml_dir.exists() else 0
        summary.append(f"  pml:     {n_pml} .pml files → {pml_dir}")
    summary.append(f"  diagnostics → {ctx.diagnostics_path}")
    summary.append(f"  errors → {ctx.error_log}")
    logger.info("\n".join(summary))
    return Path(cfg.output_path)
