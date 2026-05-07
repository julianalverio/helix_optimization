"""Unit tests for the epitope pipeline manager (orchestration only).

These tests cover config loading, stage validation, and the
"--pdb-list optional when not starting at MaSIF" rule. They do not exercise
the Docker/Modal-bound stages — those are covered by their respective
stage-level tests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from twistr.pipeline.manager.manager import load_pipeline_config, run_pipeline


def _write_full_config_set(
    tmp_path: Path, *, stages: list[str],
    output_path: str | None = None,
) -> Path:
    """Drop a complete, minimal config set into `tmp_path` and return the
    pipeline.yaml path. Sub-configs use only the fields their dataclasses
    require so the manager can build a `PipelineContext` without crashing."""
    out = output_path or str(tmp_path / "out" / "patches_final.parquet")
    (tmp_path / "epitopes.yaml").write_text(yaml.safe_dump({
        "pdb_dir": str(tmp_path / "pdb"),
        "output_path": str(tmp_path / "out" / "patches.parquet"),
        "work_dir": str(tmp_path / "work"),
    }))
    (tmp_path / "scannet.yaml").write_text(yaml.safe_dump({
        "masif_parquet": str(tmp_path / "out" / "patches.parquet"),
        "pdb_dir": str(tmp_path / "pdb"),
        "output_path": str(tmp_path / "out" / "patches_scannet.parquet"),
        "work_dir": str(tmp_path / "scannet_work"),
    }))
    (tmp_path / "hotspot.yaml").write_text(yaml.safe_dump({
        "scannet_parquet": str(tmp_path / "out" / "patches_scannet.parquet"),
        "pdb_dir": str(tmp_path / "pdb"),
        "output_path": str(tmp_path / "out" / "patches_final.parquet"),
        "work_dir": str(tmp_path / "hotspot_work"),
    }))
    (tmp_path / "viz.yaml").write_text(yaml.safe_dump({
        "patches_parquet": out,
        "pdb_dir": str(tmp_path / "pdb"),
        "output_dir": str(tmp_path / "pml"),
    }))
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(yaml.safe_dump({
        "stages": stages,
        "epitopes_config": str(tmp_path / "epitopes.yaml"),
        "scannet_config": str(tmp_path / "scannet.yaml"),
        "hotspot_config": str(tmp_path / "hotspot.yaml"),
        "epitope_viz_config": str(tmp_path / "viz.yaml"),
        "pdb_dir": str(tmp_path / "pdb"),
        "output_path": out,
    }))
    return pipeline_path


def test_validates_unknown_stage_name(tmp_path: Path):
    pipeline_path = _write_full_config_set(tmp_path, stages=["bogus"])
    with pytest.raises(ValueError, match="Unknown stage"):
        load_pipeline_config(pipeline_path)


def test_validates_empty_stages_list(tmp_path: Path):
    pipeline_path = _write_full_config_set(tmp_path, stages=[])
    with pytest.raises(ValueError, match="non-empty"):
        load_pipeline_config(pipeline_path)


def test_validates_unknown_pipeline_config_key(tmp_path: Path):
    pipeline_path = _write_full_config_set(tmp_path, stages=["viz"])
    raw = yaml.safe_load(pipeline_path.read_text())
    raw["surprise_field"] = 42
    pipeline_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown pipeline config keys"):
        load_pipeline_config(pipeline_path)


def test_load_preserves_stage_order(tmp_path: Path):
    pipeline_path = _write_full_config_set(
        tmp_path, stages=["viz", "scannet", "masif"],
    )
    cfg = load_pipeline_config(pipeline_path)
    assert cfg.stages == ["viz", "scannet", "masif"]


def test_requires_pdb_list_when_masif_is_first(tmp_path: Path):
    pipeline_path = _write_full_config_set(tmp_path, stages=["masif"])
    with pytest.raises(ValueError, match="--pdb-list is required"):
        run_pipeline(None, pipeline_path)


def test_runs_viz_only_without_pdb_list(tmp_path: Path):
    """`stages: [viz]` reads the existing final parquet — no Docker/Modal.
    With an empty final parquet pre-staged, viz writes an empty pml dir."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    final_path = out_dir / "patches_final.parquet"
    pd.DataFrame([]).to_parquet(final_path, index=False)

    pipeline_path = _write_full_config_set(
        tmp_path, stages=["viz"], output_path=str(final_path),
    )

    out = run_pipeline(None, pipeline_path)
    assert out == final_path
    pml_dir = tmp_path / "pml"
    assert pml_dir.exists()
    assert not list(pml_dir.glob("*.pml"))   # no rows → no .pml files


def test_run_pipeline_creates_empty_parquets_when_no_stage_runs(tmp_path: Path):
    """Even with `stages: [viz]` and no input data, the manager's finalizer
    should leave masif/scannet/final parquets on disk so downstream tools
    can rely on their existence."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    final_path = out_dir / "patches_final.parquet"
    pd.DataFrame([]).to_parquet(final_path, index=False)

    pipeline_path = _write_full_config_set(
        tmp_path, stages=["viz"], output_path=str(final_path),
    )
    run_pipeline(None, pipeline_path)

    # All three stage parquets exist after the run.
    assert (out_dir / "patches.parquet").exists()
    assert (out_dir / "patches_scannet.parquet").exists()
    assert final_path.exists()
