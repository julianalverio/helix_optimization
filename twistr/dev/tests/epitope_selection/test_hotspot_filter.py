"""Unit tests for the PPI-hotspotID patch acceptance logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twistr.epitope_selection.epitopes.filter import ResidueId, ResidueRecord
from twistr.epitope_selection.hotspot_filter.config import HotspotConfig
from twistr.epitope_selection.hotspot_filter.filter import (
    HotspotDecision,
    _classify_cluster,
    _sidechain_neighbors,
    evaluate_patch,
    filter_patches_for_pdb,
    one_letter,
    select_hotspots,
)


def _cfg(**overrides) -> HotspotConfig:
    base = dict(
        scannet_parquet="ignored", pdb_dir="ignored",
        output_path="ignored", work_dir="ignored",
    )
    base.update(overrides)
    return HotspotConfig(**base)


def _residue(
    rid: str, resname: str, *, sc_xyz=None, has_sidechain: bool = True,
) -> ResidueRecord:
    chain, rest = rid.split("/")
    seq = int(rest)
    if has_sidechain and sc_xyz is None:
        sc_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    elif not has_sidechain:
        sc_xyz = np.empty((0, 3), dtype=np.float64)
    sc = np.asarray(sc_xyz, dtype=np.float64)
    heavy = sc if sc.size > 0 else np.array([[0.0, 0.0, 0.0]])
    return ResidueRecord(
        rid=ResidueId(chain=chain, seq=seq, icode=""),
        resname=resname, ss="H", sasa=100.0, rsasa=0.5,
        heavy_xyz=heavy, sidechain_xyz=sc,
    )


# ---------------------- one_letter ----------------------

def test_one_letter_standard_aas():
    assert one_letter("ALA") == "A"
    assert one_letter("TRP") == "W"
    assert one_letter("XYZ") == "?"


# ---------------------- select_hotspots: cutoff vs fallback ----------------------

def test_select_hotspots_uses_threshold_when_enough_qualify():
    scores = {ResidueId("A", i, ""): (0.8 if i < 6 else 0.1) for i in range(10)}
    cfg = _cfg(hotspot_score_threshold=0.5, hotspot_min_residues_for_cutoff=5)
    hits = select_hotspots(scores, cfg)
    assert len(hits) == 6


def test_select_hotspots_falls_back_to_top_decile_when_few_qualify():
    # Only 2 residues at ≥ 0.5, but min_for_cutoff=5 → fall back to top 10%.
    scores = {ResidueId("A", i, ""): (0.99 if i < 2 else 0.1) for i in range(20)}
    cfg = _cfg(
        hotspot_score_threshold=0.5,
        hotspot_min_residues_for_cutoff=5,
        hotspot_top_fraction_fallback=0.10,
    )
    hits = select_hotspots(scores, cfg)
    # ceil(20 * 0.1) = 2 → top-2 are the same A/0 and A/1.
    assert len(hits) == 2
    assert hits == {ResidueId("A", 0, ""), ResidueId("A", 1, "")}


def test_select_hotspots_empty_input():
    assert select_hotspots({}, _cfg()) == set()


# ---------------------- _classify_cluster ----------------------

def test_cluster_hydrophobic_aromatic_passes_with_two():
    assert _classify_cluster({"L", "I", "S"}, _cfg()) == "hydrophobic_aromatic"


def test_cluster_aromatic_passes_with_one():
    # Just one hydrophobic/aromatic (F) + S (not in any set) → hydrophobic
    # check fails (count 1 < 2), aromatic check passes (F ∈ {F,Y,W}).
    assert _classify_cluster({"F", "S"}, _cfg()) == "aromatic"
    assert _classify_cluster({"F"}, _cfg()) == "aromatic"
    assert _classify_cluster({"Y"}, _cfg()) == "aromatic"
    assert _classify_cluster({"W"}, _cfg()) == "aromatic"


def test_cluster_charged_polar_passes_with_two():
    # R, K both in charged_polar; neither is in hydrophobic_aromatic or aromatic.
    assert _classify_cluster({"R", "K"}, _cfg()) == "charged_polar"


def test_cluster_mixed_passes():
    # One hydrophobic-aromatic (M) + one polar (E) → hydrophobic check fails (only 1),
    # aromatic fails (M ∉ aromatic set), charged_polar fails (only 1 from set),
    # mixed passes.
    assert _classify_cluster({"M", "E"}, _cfg()) == "mixed"


def test_cluster_no_match_returns_none():
    # A, S, T, P, G, C are not in any of the cluster sets.
    assert _classify_cluster({"A", "S", "T"}, _cfg()) is None
    assert _classify_cluster(set(), _cfg()) is None


def test_cluster_precedence_hydrophobic_aromatic_first():
    """When N satisfies multiple, the earlier-listed cluster wins. The order
    in the spec is hydrophobic/aromatic → aromatic → charged/polar → mixed."""
    # Two L's → hydrophobic_aromatic wins even though E,K could form charged_polar.
    assert _classify_cluster({"L", "I", "E", "K"}, _cfg()) == "hydrophobic_aromatic"


# ---------------------- _sidechain_neighbors ----------------------

def test_sidechain_neighbors_distance_threshold():
    hot = _residue("A/1", "LEU", sc_xyz=[[0.0, 0.0, 0.0]])
    near = _residue("A/2", "ILE", sc_xyz=[[2.0, 0.0, 0.0]])
    far = _residue("A/3", "VAL", sc_xyz=[[10.0, 0.0, 0.0]])
    out = _sidechain_neighbors(hot, [hot, near, far], distance_a=5.0)
    assert {n.rid for n in out} == {ResidueId("A", 2, "")}


def test_sidechain_neighbors_excludes_glycine_with_no_sidechain():
    hot = _residue("A/1", "LEU", sc_xyz=[[0.0, 0.0, 0.0]])
    gly = _residue("A/2", "GLY", has_sidechain=False)
    out = _sidechain_neighbors(hot, [gly], distance_a=5.0)
    assert out == []


# ---------------------- evaluate_patch ----------------------

def _patch_with(residue_ids: list[ResidueId]) -> list[ResidueId]:
    return residue_ids


def test_two_hotspots_accepted_no_cluster_check():
    rids = [ResidueId("A", i, "") for i in range(1, 6)]
    pdb_scores = {rid: 0.9 for rid in rids[:2]}
    pdb_scores.update({rid: 0.1 for rid in rids[2:]})
    hotspots = {rids[0], rids[1]}
    records = [_residue(f"A/{r.seq}", "ALA") for r in rids]
    decision = evaluate_patch(rids, pdb_scores, hotspots, records, _cfg())
    assert decision.accepted
    assert decision.n_hotspots == 2
    assert decision.accept_path == "two_hotspots"
    assert decision.cluster_neighbors == []


def test_zero_hotspots_rejected():
    rids = [ResidueId("A", i, "") for i in range(1, 6)]
    records = [_residue(f"A/{r.seq}", "ALA") for r in rids]
    decision = evaluate_patch(rids, {}, set(), records, _cfg())
    assert not decision.accepted
    assert decision.accept_path == "rejected_no_hotspot"


def test_one_hotspot_with_valid_cluster_accepted():
    # Hotspot is LEU at origin. Two neighbors ILE and VAL within 3 Å → hydrophobic/aromatic.
    hotspot = _residue("A/1", "LEU", sc_xyz=[[0.0, 0.0, 0.0]])
    near1 = _residue("A/2", "ILE", sc_xyz=[[2.0, 0.0, 0.0]])
    near2 = _residue("A/3", "VAL", sc_xyz=[[0.0, 2.0, 0.0]])
    far = _residue("A/4", "ALA", sc_xyz=[[20.0, 0.0, 0.0]])
    records = [hotspot, near1, near2, far]
    rids = [hotspot.rid, near1.rid, near2.rid, far.rid]
    pdb_scores = {hotspot.rid: 0.9}
    decision = evaluate_patch(rids, pdb_scores, {hotspot.rid}, records, _cfg())
    assert decision.accepted
    assert decision.n_hotspots == 1
    assert decision.accept_path == "one_hotspot_hydrophobic_aromatic"
    assert {ResidueId("A", 2, ""), ResidueId("A", 3, "")} <= set(decision.cluster_neighbors)


def test_one_hotspot_no_cluster_rejected():
    # Hotspot LEU surrounded by ALA/SER/THR (none in any cluster set).
    hot = _residue("A/1", "LEU", sc_xyz=[[0.0, 0.0, 0.0]])
    near_a = _residue("A/2", "ALA", sc_xyz=[[2.0, 0.0, 0.0]])
    near_b = _residue("A/3", "SER", sc_xyz=[[0.0, 2.0, 0.0]])
    records = [hot, near_a, near_b]
    rids = [r.rid for r in records]
    pdb_scores = {hot.rid: 0.9}
    decision = evaluate_patch(rids, pdb_scores, {hot.rid}, records, _cfg())
    # Note: LEU itself isn't in N (excluded as the hotspot). But the single
    # neighbor LEU... wait, LEU is the hotspot. ALA/SER are alone insufficient.
    # ALA ∈ none of the cluster sets. SER ∈ none either.
    # → no cluster passes → reject.
    assert not decision.accepted
    assert decision.accept_path == "rejected_no_cluster"


def test_one_hotspot_glycine_rejected():
    """GLY hotspot has no sidechain → cannot form a cluster."""
    gly_hot = _residue("A/1", "GLY", has_sidechain=False)
    near = _residue("A/2", "LEU", sc_xyz=[[0.5, 0.0, 0.0]])
    records = [gly_hot, near]
    decision = evaluate_patch(
        [gly_hot.rid, near.rid], {gly_hot.rid: 0.9}, {gly_hot.rid}, records, _cfg(),
    )
    assert not decision.accepted
    assert decision.accept_path == "rejected_hotspot_no_sidechain"


def test_one_hotspot_charged_polar_cluster():
    # Hotspot HIS, two charged neighbors (R, K).
    hot = _residue("A/1", "HIS", sc_xyz=[[0.0, 0.0, 0.0]])
    n1 = _residue("A/2", "ARG", sc_xyz=[[2.0, 0.0, 0.0]])
    n2 = _residue("A/3", "LYS", sc_xyz=[[0.0, 2.0, 0.0]])
    records = [hot, n1, n2]
    decision = evaluate_patch(
        [r.rid for r in records], {hot.rid: 0.9}, {hot.rid}, records, _cfg(),
    )
    assert decision.accepted
    assert decision.accept_path == "one_hotspot_charged_polar"


def test_one_hotspot_mixed_cluster():
    # Hotspot CYS (not in any cluster set itself; doesn't matter — N is examined).
    # N has 1 hydrophobic (LEU) + 1 polar (GLU).
    hot = _residue("A/1", "CYS", sc_xyz=[[0.0, 0.0, 0.0]])
    n_hp = _residue("A/2", "LEU", sc_xyz=[[2.0, 0.0, 0.0]])
    n_pol = _residue("A/3", "GLU", sc_xyz=[[0.0, 2.0, 0.0]])
    records = [hot, n_hp, n_pol]
    decision = evaluate_patch(
        [r.rid for r in records], {hot.rid: 0.9}, {hot.rid}, records, _cfg(),
    )
    assert decision.accepted
    # LEU alone passes hydrophobic check? LEU ∈ hydrophobic_aromatic (count 1)
    # → not ≥ 2 → fails. GLU ∈ charged_polar (count 1) → fails. Mixed: 1 hp + 1 polar → pass.
    assert decision.accept_path == "one_hotspot_mixed"


# ---------------------- end-to-end via filter_patches_for_pdb ----------------------

def test_filter_patches_for_pdb_keeps_all_patches_with_hotspot_stats():
    """Both input patches survive: hotspot stats annotate them for ranking,
    and `accept_path` records what the legacy filter would have decided."""
    patch_rows = [
        {"pdb_id": "1abc", "chains": "A", "patch_id": 0,
         "n_residues": 5, "residue_ids": ["A/1", "A/2", "A/3", "A/4", "A/5"]},
        {"pdb_id": "1abc", "chains": "A", "patch_id": 1,
         "n_residues": 5, "residue_ids": ["A/10", "A/11", "A/12", "A/13", "A/14"]},
    ]
    df = pd.DataFrame(patch_rows)
    pdb_scores = {ResidueId("A", i, ""): (0.9 if i in (1, 2) else 0.1)
                  for i in list(range(1, 6)) + list(range(10, 15))}
    records = [_residue(f"A/{i}", "ALA")
               for i in list(range(1, 6)) + list(range(10, 15))]
    out = filter_patches_for_pdb("1abc", df, pdb_scores, records,
                                 _cfg(hotspot_min_residues_for_cutoff=2))
    assert len(out) == 2
    by_id = {row["patch_id"]: row for _, row in out.iterrows()}
    assert by_id[0]["n_hotspots"] == 2
    assert by_id[0]["accept_path"] == "two_hotspots"
    assert by_id[1]["n_hotspots"] == 0
    assert by_id[1]["accept_path"] == "rejected_no_hotspot"
