"""Unit tests for the residue-graph patch construction in epitopes.patches.

Tests cover:
  - vertex_voronoi_areas (still used for area reporting)
  - compute_residue_masif_scores (top-quartile mean, MaSIF-unsupported gating)
  - find_helix_segments (contiguous H/G runs per chain)
  - build_residue_graph (helix-face + spatial side-chain edges, same-helix
    spatial-edge suppression)
  - find_patch_residues (re-extraction)
  - end-to-end extract_patches: i/i+4 connectivity, helix-adjacent inclusion,
    same-helix non-i/i+k pairs not glued, residue count + max score gates.
"""
from __future__ import annotations

import numpy as np
import pytest

from twistr.pipeline.epitopes.config import EpitopesConfig
from twistr.pipeline.epitopes.filter import ResidueId, ResidueRecord
from twistr.pipeline.epitopes.patches import (
    build_residue_graph,
    compute_residue_masif_scores,
    extract_patches,
    find_helix_segments,
    find_patch_residues,
    vertex_voronoi_areas,
)


def _grid_mesh(n: int, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat n×n grid in z=0 plane, triangulated into 2(n-1)² triangles."""
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
    faces: list[list[int]] = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = a + n; d = c + 1
            faces.append([a, b, c]); faces.append([b, d, c])
    return verts, np.asarray(faces, dtype=np.int64)


def _cfg(**overrides) -> EpitopesConfig:
    base = dict(
        pdb_dir="ignored", output_path="ignored", work_dir="ignored",
        vertex_to_residue_distance_a=1.5,
        score_aggregation_min_vertices=3,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_sidechain_distance_a=5.0,
        helix_node_score_threshold=0.55,
        halo_node_score_threshold=0.50,
        core_score_threshold=0.70,
        strong_score_threshold=0.85,
        mean_anchor_score_threshold=0.55,
        component_min_anchor_residues=2,
        # Tests use synthetic records so re-extraction is unbounded; tests
        # that rely on `expanded_patch_min_residues` set it explicitly.
        expanded_patch_min_residues=1,
        patch_residue_min_relative_sasa=0.0,
        strict_mode=False,
    )
    base.update(overrides)
    return EpitopesConfig(**base)


def _split_core_halo(
    records: list[ResidueRecord], helix_codes=("H", "G"),
) -> tuple[set[ResidueId], set[ResidueId]]:
    """Treat helix records as core, others as halo. Synthetic test helper."""
    core = {r.rid for r in records if r.ss in helix_codes}
    halo = {r.rid for r in records if r.ss not in helix_codes}
    return core, halo


def _residue(
    chain: str, seq: int, x: float, y: float, z: float, *,
    ss: str = "H", rsasa: float = 1.0, resname: str = "ALA",
    sidechain_offset: tuple[float, float, float] = (0.5, 0.0, 0.5),
) -> ResidueRecord:
    """Single-atom backbone + single side-chain atom offset from it."""
    bb = np.array([[x, y, z]], dtype=np.float64)
    if resname == "GLY":
        sc = np.empty((0, 3), dtype=np.float64)
        heavy = bb
    else:
        sc = np.array([[x + sidechain_offset[0], y + sidechain_offset[1],
                        z + sidechain_offset[2]]], dtype=np.float64)
        heavy = np.vstack([bb, sc])
    return ResidueRecord(
        rid=ResidueId(chain=chain, seq=seq, icode=""),
        resname=resname, ss=ss, sasa=100.0, rsasa=rsasa,
        heavy_xyz=heavy, sidechain_xyz=sc,
    )


# ---------------------- vertex Voronoi areas ----------------------

def test_voronoi_areas_sum_to_total_surface_area():
    verts, faces = _grid_mesh(4)
    areas = vertex_voronoi_areas(verts, faces)
    assert areas.sum() == pytest.approx(9.0, rel=1e-9)
    assert areas[0] == pytest.approx(1.0 / 6.0, rel=1e-9)
    assert areas[5] == pytest.approx(1.0, rel=1e-9)


# ---------------------- residue MaSIF score aggregation ----------------------

def test_residue_score_top_quartile_mean():
    # 8 vertices clustered at the side-chain location, scores: 1.0..0.3.
    sc = (1.0, 0.0, 0.0)  # side-chain offset
    rec = _residue("A", 1, 0.0, 0.0, 0.0, sidechain_offset=sc)
    n = 8
    verts = np.array([[1.0, 0.0, 0.0]] * n, dtype=np.float64)
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    rs, vidx = compute_residue_masif_scores(
        verts, scores, [rec], allowed={rec.rid},
        distance_a=1.5, min_vertices=3,
    )
    # k = max(3, ceil(0.25 * 8)) = max(3, 2) = 3 → mean of top-3 (1.0+0.9+0.8)/3
    assert rs[rec.rid] == pytest.approx((1.0 + 0.9 + 0.8) / 3, rel=1e-9)
    assert sorted(int(i) for i in vidx[rec.rid]) == list(range(8))


def test_residue_score_unsupported_when_too_few_vertices():
    rec = _residue("A", 1, 0.0, 0.0, 0.0, sidechain_offset=(1.0, 0.0, 0.0))
    verts = np.array([[1.0, 0.0, 0.0], [1.05, 0.0, 0.0]])  # only 2 vertices
    scores = np.array([0.95, 0.95])
    rs, _ = compute_residue_masif_scores(
        verts, scores, [rec], allowed={rec.rid}, distance_a=1.5, min_vertices=3,
    )
    assert rec.rid not in rs


def test_residue_score_skips_glycine():
    """GLY has no side-chain heavy atoms → no patch anchor."""
    gly = _residue("A", 1, 0.0, 0.0, 0.0, resname="GLY")
    assert gly.sidechain_xyz.size == 0
    verts = np.array([[0.0, 0.0, 0.0]] * 5)
    scores = np.full(5, 0.95)
    rs, _ = compute_residue_masif_scores(
        verts, scores, [gly], allowed={gly.rid}, distance_a=2.0, min_vertices=3,
    )
    assert gly.rid not in rs


def test_residue_score_top_quartile_k_clamped_to_min_vertices():
    # 4 vertices → ceil(0.25 * 4) = 1, but k floor = min_vertices = 3 → mean of top-3.
    rec = _residue("A", 1, 0.0, 0.0, 0.0, sidechain_offset=(1.0, 0.0, 0.0))
    verts = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.5, 0.1])
    rs, _ = compute_residue_masif_scores(
        verts, scores, [rec], allowed={rec.rid}, distance_a=1.5, min_vertices=3,
    )
    assert rs[rec.rid] == pytest.approx((0.9 + 0.8 + 0.5) / 3, rel=1e-9)


# ---------------------- helix segments ----------------------

def test_helix_segments_split_by_chain_break_and_non_helix():
    records = [
        _residue("A", 1, 0, 0, 0, ss="H"),
        _residue("A", 2, 0, 0, 0, ss="H"),
        _residue("A", 3, 0, 0, 0, ss="E"),  # break by SS
        _residue("A", 4, 0, 0, 0, ss="H"),
        _residue("A", 5, 0, 0, 0, ss="H"),
        _residue("B", 1, 0, 0, 0, ss="H"),  # break by chain
        _residue("B", 2, 0, 0, 0, ss="H"),
    ]
    segments, rid_to_seg = find_helix_segments(records, helix_codes=("H", "G"))
    assert len(segments) == 3
    assert [len(s) for s in segments] == [2, 2, 2]
    assert rid_to_seg[ResidueId("A", 1, "")] == (0, 0)
    assert rid_to_seg[ResidueId("A", 4, "")] == (1, 0)
    assert rid_to_seg[ResidueId("B", 1, "")] == (2, 0)


def test_helix_segments_treats_h_and_g_as_continuous_helix():
    records = [
        _residue("A", 1, 0, 0, 0, ss="H"),
        _residue("A", 2, 0, 0, 0, ss="G"),
        _residue("A", 3, 0, 0, 0, ss="H"),
    ]
    segments, _ = find_helix_segments(records, helix_codes=("H", "G"))
    assert len(segments) == 1
    assert len(segments[0]) == 3


# ---------------------- residue graph: helix-face edges ----------------------

def test_helix_face_edges_for_i_and_i_plus_4():
    # 5 helix residues at sequential positions 0..4 within one segment.
    records = [_residue("A", i + 1, 0, 0, 0) for i in range(5)]
    nodes = {r.rid for r in records}
    segments, rid_to_seg = find_helix_segments(records, helix_codes=("H", "G"))
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_distance_a=0.001,  # disable spatial edges
    )
    rid0 = ResidueId("A", 1, "")
    rid4 = ResidueId("A", 5, "")  # position 4 in the segment
    assert rid4 in adj[rid0], "i and i+4 must be helix-face-connected"


def test_no_helix_face_edge_for_i_and_i_plus_2():
    records = [_residue("A", i + 1, 0, 0, 0) for i in range(5)]
    nodes = {r.rid for r in records}
    segments, rid_to_seg = find_helix_segments(records, helix_codes=("H", "G"))
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_distance_a=0.001,
    )
    rid0 = ResidueId("A", 1, "")
    rid2 = ResidueId("A", 3, "")  # position 2 in the segment
    assert rid2 not in adj[rid0]


# ---------------------- residue graph: spatial side-chain edges ----------------------

def test_spatial_side_chain_edge_between_helix_and_loop():
    helix = _residue("A", 1, 0, 0, 0, ss="H", sidechain_offset=(0.5, 0, 0))
    loop = _residue("A", 50, 1.0, 0, 0, ss="T", sidechain_offset=(1.5, 0, 0))
    nodes = {helix.rid, loop.rid}
    records = [helix, loop]
    segments, rid_to_seg = find_helix_segments(records, helix_codes=("H", "G"))
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_distance_a=2.0,
    )
    # helix CB at (0.5,0,0); loop CB at (2.5,0,0). Distance 2.0 → on the boundary.
    assert loop.rid in adj[helix.rid]


def test_no_spatial_edge_within_same_helix_segment():
    """The same-helix spatial-edge suppression rule (prevents gluing
    opposite helix faces via close i/i+1/i+2 distances)."""
    a = _residue("A", 1, 0, 0, 0, ss="H", sidechain_offset=(0.1, 0, 0))
    b = _residue("A", 2, 0, 0, 0, ss="H", sidechain_offset=(0.2, 0, 0))
    nodes = {a.rid, b.rid}
    records = [a, b]
    segments, rid_to_seg = find_helix_segments([a, b], helix_codes=("H", "G"))
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_distance_a=10.0,  # huge spatial cutoff would otherwise connect
    )
    # i and i+1 in same helix and offset 1 ∉ {3,4,7,8} → no edge.
    assert b.rid not in adj[a.rid]


def test_cross_segment_spatial_edge_between_two_helices():
    """Two residues each in a DIFFERENT helix segment can be spatially connected."""
    h1 = _residue("A", 1, 0, 0, 0, ss="H", sidechain_offset=(0.0, 0, 0))
    h1b = _residue("A", 2, 1, 0, 0, ss="H", sidechain_offset=(0.0, 0, 0))
    loop = _residue("A", 3, 5, 0, 0, ss="T")  # break the segment
    h2 = _residue("A", 4, 1.5, 0, 0, ss="H", sidechain_offset=(0.0, 0, 0))
    records = [h1, h1b, loop, h2]
    nodes = {h1.rid, h1b.rid, h2.rid}
    segments, rid_to_seg = find_helix_segments(records, helix_codes=("H", "G"))
    assert len(segments) == 2  # split by the loop
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=(3, 4, 7, 8),
        spatial_distance_a=2.0,
    )
    # h1b at x=1, h2 at x=1.5 → distance 0.5 ≤ 2.0; different segments → spatial edge OK.
    assert h2.rid in adj[h1b.rid]


# ---------------------- find_patch_residues (re-extraction, unchanged behavior) ----------------------

def test_find_patch_residues_uses_per_residue_rsasa_floor():
    patch_xyz = np.array([[0., 0., 0.]])
    near_high = _residue("A", 1, 0.5, 0.0, 0.0, rsasa=0.5)
    near_low = _residue("A", 2, 0.5, 0.5, 0.0, rsasa=0.05)
    far_high = _residue("A", 3, 5.0, 0.0, 0.0, rsasa=1.0)
    out = find_patch_residues(
        patch_xyz, [near_high, near_low, far_high], distance_a=1.0, min_rsasa=0.15,
    )
    assert [str(r) for r in out] == ["A/1"]


# ---------------------- end-to-end extract_patches ----------------------

def _vertex_cluster_at(xyz: tuple[float, float, float], n: int = 8) -> np.ndarray:
    return np.array([list(xyz)] * n, dtype=np.float64)


def test_A_i_and_i_plus_4_sparse_helix_face_seed_passes():
    """Spec Test A: i and i+4 in the same patch, both core, max strong; i+1
    i+2 i+3 not usable. Expected: 1 surviving 2-anchor patch."""
    records = [
        _residue("A", i + 1, float(i), 0, 0, ss="H",
                 sidechain_offset=(0.0, 1.0, 0.0))
        for i in range(5)
    ]
    # Vertices clustered around segment positions 0 and 4 only.
    verts = np.vstack([
        _vertex_cluster_at((0.0, 1.0, 0.0)),
        _vertex_cluster_at((4.0, 1.0, 0.0)),
    ])
    scores = np.full(16, 0.95)
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        spatial_sidechain_distance_a=0.001,
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,  # only 2 records exist for re-extraction
    )
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 1
    component = {str(r) for r in patches[0].component_residue_ids}
    assert component == {"A/1", "A/5"}


def test_B_weak_two_anchor_pair_fails_core_gate():
    """Spec Test B: two graph-edge-connected anchors with scores below
    `core_score_threshold` (0.70). The component forms but the 2-anchor
    'both must be core' check rejects."""
    records = [
        _residue("A", i + 1, float(i), 0, 0, ss="H",
                 sidechain_offset=(0.0, 1.0, 0.0))
        for i in range(5)
    ]
    # Anchors at segment positions 0 and 4. Scores tuned to clear the helix
    # node threshold (0.55) but fail the core threshold (0.70).
    cluster0 = _vertex_cluster_at((0.0, 1.0, 0.0))
    cluster4 = _vertex_cluster_at((4.0, 1.0, 0.0))
    verts = np.vstack([cluster0, cluster4])
    # All 8 vertices in each cluster get the same score → top-quartile mean
    # equals that score.
    scores = np.concatenate([np.full(8, 0.56), np.full(8, 0.61)])
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        spatial_sidechain_distance_a=0.001,
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,
    )
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 0


def test_C_same_helix_i_and_i_plus_1_not_connected_by_spatial_edge():
    """Spec Test C: same-segment i and i+1, both strong. They must NOT be
    connected by a spatial edge (offset 1 ∉ {3,4,7,8} → no helix-face edge
    either). Two singleton components → both fail the size gate."""
    a = _residue("A", 1, 0, 0, 0, ss="H", sidechain_offset=(0.1, 0, 0))
    b = _residue("A", 2, 0, 0, 0, ss="H", sidechain_offset=(0.2, 0, 0))
    records = [a, b]
    cluster_a = _vertex_cluster_at((0.1, 0, 0))
    cluster_b = _vertex_cluster_at((0.2, 0, 0))
    verts = np.vstack([cluster_a, cluster_b])
    scores = np.full(16, 0.95)
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.05,  # tight enough to keep clusters separate
        spatial_sidechain_distance_a=10.0,  # big enough that the suppression rule matters
        component_min_anchor_residues=2,
    )
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 0


def test_D_helix_adjacent_patch_via_spatial_edge():
    """Spec Test D: one core helix residue + one halo loop residue, side-chain
    distance ≤ 5 Å, both meet their respective node thresholds. Connected by
    a spatial side-chain edge → one surviving 2-anchor patch."""
    helix = _residue("A", 1, 0.0, 0.0, 0.0, ss="H",
                     sidechain_offset=(0.0, 1.0, 0.0))   # side-chain at (0,1,0)
    loop = _residue("A", 50, 0.0, 2.5, 0.0, ss="T",
                    sidechain_offset=(0.0, 1.0, 0.0))    # side-chain at (0,3.5,0)
    records = [helix, loop]
    verts = np.vstack([
        _vertex_cluster_at((0.0, 1.0, 0.0)),  # → helix
        _vertex_cluster_at((0.0, 3.5, 0.0)),  # → loop
    ])
    scores = np.concatenate([
        np.full(8, 0.95),  # helix score 0.95 ≥ core
        np.full(8, 0.90),  # loop  score 0.90 ≥ core; max ≥ strong
    ])
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        spatial_sidechain_distance_a=3.0,  # 2.5 Å between side-chains → edge
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 1
    component = {str(r) for r in patches[0].component_residue_ids}
    assert component == {"A/1", "A/50"}


def test_extract_patches_filters_by_strong_score():
    """≥ 3-anchor component fails when no anchor reaches strong_score_threshold."""
    records = [
        _residue("A", i + 1, float(i), 0, 0, ss="H",
                 sidechain_offset=(0.0, 1.0, 0.0))
        for i in range(8)
    ]
    # Three anchors at segment positions 0, 3, 4 (so connectivity 0↔3 and 0↔4).
    verts = np.vstack([
        _vertex_cluster_at((0.0, 1.0, 0.0)),
        _vertex_cluster_at((3.0, 1.0, 0.0)),
        _vertex_cluster_at((4.0, 1.0, 0.0)),
    ])
    # All anchors at 0.75 → mean ≥ 0.55 ✓, but max < 0.85 ✗.
    scores = np.full(verts.shape[0], 0.75)
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,
    )
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 0


def test_extract_patches_filters_by_component_size():
    """A singleton component fails the component_min_anchor_residues gate."""
    rec = _residue("A", 1, 0.0, 0.0, 0.0, ss="H",
                   sidechain_offset=(0.0, 1.0, 0.0))
    verts = _vertex_cluster_at((0.0, 1.0, 0.0))
    scores = np.full(verts.shape[0], 0.95)
    core, halo = _split_core_halo([rec])
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        component_min_anchor_residues=2,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, [rec], core, halo, cfg)
    assert len(patches) == 0


def test_extract_patches_filters_by_expanded_residue_count():
    """A passing 2-anchor component is still rejected if its re-extracted
    bystander set is smaller than expanded_patch_min_residues."""
    records = [
        _residue("A", i + 1, float(i), 0, 0, ss="H",
                 sidechain_offset=(0.0, 1.0, 0.0))
        for i in range(5)
    ]
    verts = np.vstack([
        _vertex_cluster_at((0.0, 1.0, 0.0)),
        _vertex_cluster_at((4.0, 1.0, 0.0)),
    ])
    scores = np.full(16, 0.95)
    core, halo = _split_core_halo(records)
    cfg = _cfg(
        vertex_to_residue_distance_a=0.5,
        spatial_sidechain_distance_a=0.001,
        component_min_anchor_residues=2,
        # Only the two anchors fall within 0.5 Å of the patch vertices, so
        # re-extracted set = 2; raising the gate to 3 must reject.
        expanded_patch_min_residues=3,
    )
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg)
    assert len(patches) == 0


def test_extract_patches_strict_mode_for_three_anchors():
    """≥ 3-anchor component passes first-pass but fails strict mode when
    fewer than 2 anchors are at core threshold."""
    records = [
        _residue("A", i + 1, float(i), 0, 0, ss="H",
                 sidechain_offset=(0.0, 1.0, 0.0))
        for i in range(8)
    ]
    # Three anchors at segment positions 0, 3, 4 connected 0↔3, 0↔4.
    verts = np.vstack([
        _vertex_cluster_at((0.0, 1.0, 0.0)),
        _vertex_cluster_at((3.0, 1.0, 0.0)),
        _vertex_cluster_at((4.0, 1.0, 0.0)),
    ])
    # One strong anchor at 0.95 satisfies max ≥ 0.85; the other two at 0.62
    # are graph nodes (≥ 0.55) but only 1 anchor ≥ core (0.70). Mean = 0.73.
    scores = np.concatenate([np.full(8, 0.95), np.full(8, 0.62), np.full(8, 0.62)])
    core, halo = _split_core_halo(records)
    faces = np.array([[0, 1, 2], [13, 14, 15]], dtype=np.int64)

    cfg_first_pass = _cfg(
        vertex_to_residue_distance_a=0.5,
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,
        strict_mode=False,
    )
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg_first_pass)
    assert len(patches) == 1, "first-pass should accept (mean 0.73 ≥ 0.55, max 0.95 ≥ 0.85)"

    cfg_strict = _cfg(
        vertex_to_residue_distance_a=0.5,
        component_min_anchor_residues=2,
        expanded_patch_min_residues=2,
        strict_mode=True,
    )
    patches = extract_patches(verts, faces, scores, records, core, halo, cfg_strict)
    assert len(patches) == 0, "strict mode requires ≥ 2 anchors at core threshold"
