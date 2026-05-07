"""Viz stage: one .pml per row of the final parquet."""
from __future__ import annotations

from twistr.pipeline.epitope_viz.driver import run_epitope_viz

from .context import PipelineContext


def run_batch(ctx: PipelineContext) -> None:
    run_epitope_viz(ctx.viz_cfg)
