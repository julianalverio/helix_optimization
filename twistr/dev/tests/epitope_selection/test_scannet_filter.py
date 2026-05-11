"""Unit tests for the ScanNet patch filter (no docker)."""
from __future__ import annotations

import pandas as pd
import pytest

from twistr.epitope_selection.epitopes.filter import ResidueId
from twistr.epitope_selection.scannet_filter.config import ScanNetConfig
from twistr.epitope_selection.scannet_filter.filter import (
    filter_patches_for_pdb,
    parse_residue_id,
)


def _cfg(**overrides) -> ScanNetConfig:
    base = dict(
        masif_parquet="ignored", pdb_dir="ignored",
        output_path="ignored", work_dir="ignored",
        residue_positive_threshold=0.5,
        patch_min_mean_score=0.4,
        patch_min_positive_fraction=0.25,
        patch_min_max_score=0.65,
        patch_min_residues=5,
    )
    base.update(overrides)
    return ScanNetConfig(**base)


def _patch_row(residue_ids: list[str], **extra) -> dict:
    base = dict(
        pdb_id="1xyz", chains="A", patch_id=0,
        n_vertices=10, area_a2=400.0, max_score=0.9,
        n_residues=len(residue_ids),
        residue_ids=residue_ids,
        vertex_indices=[0, 1, 2],
    )
    base.update(extra)
    return base


def test_parse_residue_id_basic():
    assert parse_residue_id("A/123") == ResidueId("A", 123, "")
    assert parse_residue_id("B/456") == ResidueId("B", 456, "")


def test_parse_residue_id_with_icode():
    assert parse_residue_id("A/123A") == ResidueId("A", 123, "A")
    assert parse_residue_id("H/27B") == ResidueId("H", 27, "B")


def test_parse_residue_id_negative_seq():
    assert parse_residue_id("A/-5") == ResidueId("A", -5, "")


def test_filter_passes_patch_meeting_all_thresholds():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    scores = {ResidueId("A", i, ""): 0.8 if i < 5 else 0.3 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["scannet_n_scored"] == 10
    assert row["scannet_mean"] == pytest.approx(0.55)
    assert row["scannet_max"] == pytest.approx(0.8)
    assert row["scannet_pos_frac"] == pytest.approx(0.5)


def test_filter_drops_patch_with_too_few_scored_residues():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    # Only 4 residues have scores → below patch_min_residues=5.
    scores = {ResidueId("A", i, ""): 0.95 for i in range(4)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert len(out) == 0


def test_filter_drops_patch_when_both_mean_and_pos_frac_fail():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    # mean=0.13 (< 0.4), pos_frac=0.10 (1/10 < 0.25), max=0.95 — both fail.
    scores = {ResidueId("A", i, ""): 0.95 if i == 0 else 0.04 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg(patch_min_max_score=0.5))
    assert len(out) == 0


def test_filter_keeps_patch_when_only_mean_passes():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    # mean=0.41 (≥ 0.4), pos_frac=0.0 (none ≥ 0.5) — kept via mean.
    scores = {ResidueId("A", i, ""): 0.49 if i == 0 else 0.41 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg(patch_min_max_score=0.4))
    assert len(out) == 1


def test_filter_keeps_patch_when_only_pos_frac_passes():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    # mean=0.30 (< 0.4), pos_frac=0.30 (3/10 ≥ 0.25) — kept via pos_frac.
    scores = {ResidueId("A", i, ""): 0.95 if i < 3 else 0.02 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert len(out) == 1


def test_filter_drops_patch_below_max_threshold():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)])])
    # Max only 0.6 (< 0.65 threshold), even though mean / pos_frac would pass.
    scores = {ResidueId("A", i, ""): 0.6 if i < 5 else 0.4 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert len(out) == 0


def test_filter_handles_missing_residues_in_scannet():
    # Residues with no ScanNet score are silently dropped from the count.
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(8)])])
    scores = {ResidueId("A", i, ""): 0.9 for i in range(6)}  # last 2 missing
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["scannet_n_scored"] == 6
    assert len(row["scannet_residue_scores"]) == 6
    assert len(row["scannet_scored_residue_ids"]) == 6


def test_filter_preserves_masif_columns_in_output():
    df = pd.DataFrame([_patch_row([f"A/{i}" for i in range(10)], patch_id=42, area_a2=512.0)])
    scores = {ResidueId("A", i, ""): 0.8 for i in range(10)}
    out = filter_patches_for_pdb("1xyz", df, scores, _cfg())
    assert out.iloc[0]["patch_id"] == 42
    assert out.iloc[0]["area_a2"] == 512.0
    assert "scannet_mean" in out.columns


def test_empty_patches_returns_empty_with_scannet_columns():
    empty = pd.DataFrame([_patch_row(["A/1"])]).iloc[0:0]
    out = filter_patches_for_pdb("1xyz", empty, {}, _cfg())
    assert len(out) == 0
    for col in ("scannet_n_scored", "scannet_mean", "scannet_max",
                "scannet_pos_frac", "scannet_residue_scores",
                "scannet_scored_residue_ids"):
        assert col in out.columns
