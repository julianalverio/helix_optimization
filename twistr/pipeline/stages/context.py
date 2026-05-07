"""Shared `PipelineContext` passed to every stage's `run_batch`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from twistr.pipeline.epitope_viz.config import EpitopeVizConfig
from twistr.pipeline.epitopes.config import EpitopesConfig
from twistr.pipeline.hotspot_filter.config import HotspotConfig
from twistr.pipeline.scannet_filter.config import ScanNetConfig


@dataclass
class PipelineContext:
    pdb_ids: list[str] | None     # None when no --pdb-list and first stage isn't masif
    pdb_dir: Path
    work_root: Path
    masif_path: Path
    scannet_path: Path
    final_path: Path
    epitopes_cfg: EpitopesConfig
    scannet_cfg: ScanNetConfig
    hotspot_cfg: HotspotConfig
    viz_cfg: EpitopeVizConfig
    diagnostics_path: Path
    error_log: Path
    diagnostics: list[dict] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # Per-stage row accumulators — populated by upstream stages so downstream
    # stages can read in-memory rather than round-tripping through disk.
    masif_rows: list[dict] = field(default_factory=list)
    scannet_rows: list[dict] = field(default_factory=list)
    final_rows: list[dict] = field(default_factory=list)
