"""Residue-level patch construction for the MaSIF epitope detector.

Patches are connected components in a RESIDUE graph (not a surface-mesh graph),
because helix face residues at sequence positions i and i+4 share a binding
face but are typically NOT connected by high-scoring mesh vertices.

Pipeline (replaces the old mesh-CC patch logic):

  1. Aggregate MaSIF vertex scores → per-residue MaSIF scores using a
     top-quartile mean over vertices within `vertex_to_residue_distance_a`
     of the residue's side-chain heavy atoms.
       k = max(score_aggregation_min_vertices, ceil(0.25 * n_nearby))
       residue_score = mean(top-k nearby vertex scores)
     Residues with fewer than `score_aggregation_min_vertices` nearby
     vertices are MaSIF-unsupported.

  2. Graph nodes use class-specific score thresholds:
       - core (DSSP helix + rSASA ≥ core_min_relative_sasa) requires
         residue MaSIF score ≥ `helix_node_score_threshold` (default 0.55).
       - halo (immediate neighbor + rSASA ≥ halo_min_relative_sasa) requires
         residue MaSIF score ≥ `halo_node_score_threshold` (default 0.50).
     The threshold split lets a moderate-score halo residue join a graph
     even when no single threshold would admit it without also flooding the
     graph with too many helix nodes.

  3. Two edge types:
     a) helix-face: same DSSP helix segment, |i-j| ∈ helix_face_offsets.
        Position is the sequential index within the segment, NOT the PDB
        residue number — handles insertion codes / numbering gaps.
     b) spatial side-chain: residues NOT both in the same helix segment,
        with min side-chain heavy-atom distance ≤ spatial_sidechain_distance_a.

  4. Patches = connected components of this residue graph.

  5. Patch acceptance is size-dependent. A component must:
       - have ≥ `component_min_anchor_residues` anchors (default 2)
       - have ≥ `expanded_patch_min_residues` re-extracted residues (default 5)
     Then, for 2-anchor components: both anchor scores ≥ `core_score_threshold`
     AND max ≥ `strong_score_threshold`. For ≥ 3-anchor components:
     mean ≥ `mean_anchor_score_threshold` AND max ≥ `strong_score_threshold`.
     With `strict_mode=True`, ≥3-anchor also requires mean ≥ 0.60 AND ≥ 2
     anchors at score ≥ `core_score_threshold`. The mesh area filter is
     dropped (residue-CC patches don't have a single contiguous footprint).

  6. Residue re-extraction: for each surviving patch the parquet's
     `residue_ids` column is the union of residues whose heavy atoms are
     within `vertex_to_residue_distance_a` of any "patch vertex" (a vertex
     used to score any component residue) AND rSASA ≥
     `patch_residue_min_relative_sasa`. Component residues themselves are
     always included (they sourced the patch vertices). The graph members
     are reported separately as `component_residue_ids`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .config import EpitopesConfig
from .filter import ResidueId, ResidueRecord


@dataclass(frozen=True)
class Patch:
    patch_id: int
    vertex_indices: np.ndarray         # union of vertices used to score component residues
    area_a2: float                     # Σ Voronoi areas over vertex_indices (reporting only)
    max_score: float                   # max residue MaSIF score over component
    residue_ids: list[ResidueId]       # re-extracted (within 4 Å of vertex + rSASA ≥ floor)
    component_residue_ids: list[ResidueId]      # graph CC members
    residue_masif_scores: list[float]  # aligned with component_residue_ids


def patches_to_rows(pdb_id: str, chains: str, patches: list["Patch"]) -> list[dict]:
    """Serialize MaSIF patches into the parquet row schema."""
    return [
        {
            "pdb_id": pdb_id.lower(),
            "chains": chains,
            "patch_id": p.patch_id,
            "n_vertices": int(len(p.vertex_indices)),
            "area_a2": p.area_a2,
            "max_score": p.max_score,
            "n_residues": len(p.residue_ids),
            "residue_ids": [str(r) for r in p.residue_ids],
            "vertex_indices": p.vertex_indices.astype(int).tolist(),
            "n_component_residues": len(p.component_residue_ids),
            "component_residue_ids": [str(r) for r in p.component_residue_ids],
            "residue_masif_scores": p.residue_masif_scores,
        }
        for p in patches
    ]


def vertex_voronoi_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex area = (1/3) × sum of incident triangle areas."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    out = np.zeros(vertices.shape[0], dtype=np.float64)
    np.add.at(out, faces[:, 0], tri_area / 3.0)
    np.add.at(out, faces[:, 1], tri_area / 3.0)
    np.add.at(out, faces[:, 2], tri_area / 3.0)
    return out


def compute_residue_masif_scores(
    vertices: np.ndarray,
    scores: np.ndarray,
    records: list[ResidueRecord],
    allowed: set[ResidueId],
    distance_a: float,
    min_vertices: int,
) -> tuple[dict[ResidueId, float], dict[ResidueId, np.ndarray]]:
    """Top-quartile mean of MaSIF vertex scores within `distance_a` of each
    allowed residue's side-chain heavy atoms.

    Returns (residue_scores, residue_vertex_indices). A residue is omitted
    from both dicts if it has no side-chain heavy atoms (GLY) or fewer than
    `min_vertices` MaSIF vertices nearby."""
    if vertices.shape[0] == 0:
        return {}, {}
    tree = cKDTree(vertices)
    residue_scores: dict[ResidueId, float] = {}
    residue_vertex_indices: dict[ResidueId, np.ndarray] = {}
    for rec in records:
        if rec.rid not in allowed or rec.sidechain_xyz.size == 0:
            continue
        nbr_lists = tree.query_ball_point(rec.sidechain_xyz, r=distance_a)
        # Deduplicate vertex indices across the residue's side-chain atoms.
        idx_set: set[int] = set()
        for sub in nbr_lists:
            idx_set.update(int(i) for i in sub)
        if len(idx_set) < min_vertices:
            continue
        idx = np.fromiter(idx_set, dtype=np.int64, count=len(idx_set))
        sorted_scores = np.sort(scores[idx])[::-1]
        k = max(min_vertices, math.ceil(0.25 * idx.size))
        k = min(k, idx.size)
        residue_scores[rec.rid] = float(sorted_scores[:k].mean())
        residue_vertex_indices[rec.rid] = idx
    return residue_scores, residue_vertex_indices


def find_helix_segments(
    records: list[ResidueRecord], helix_codes: tuple[str, ...],
) -> tuple[list[list[ResidueId]], dict[ResidueId, tuple[int, int]]]:
    """Walk records (assumed to be in chain → sequence order) and return:
      - segments: list of contiguous helix runs, each a list of ResidueIds.
        A helix break = chain change OR ss not in helix_codes.
      - rid_to_seg: rid → (segment_index, position_within_segment).

    Records with rid.chain change always start a new segment, so segments do
    not span chains."""
    helix_set = set(helix_codes)
    segments: list[list[ResidueId]] = []
    current: list[ResidueId] = []
    last_chain: str | None = None
    for rec in records:
        is_helix = rec.ss in helix_set
        chain_break = rec.rid.chain != last_chain
        if not is_helix or chain_break:
            if current:
                segments.append(current)
                current = []
        if is_helix:
            current.append(rec.rid)
        last_chain = rec.rid.chain
    if current:
        segments.append(current)

    rid_to_seg: dict[ResidueId, tuple[int, int]] = {}
    for seg_id, seg in enumerate(segments):
        for pos, rid in enumerate(seg):
            rid_to_seg[rid] = (seg_id, pos)
    return segments, rid_to_seg


def build_residue_graph(
    nodes: set[ResidueId],
    records: list[ResidueRecord],
    segments: list[list[ResidueId]],
    rid_to_seg: dict[ResidueId, tuple[int, int]],
    helix_face_offsets: tuple[int, ...],
    spatial_distance_a: float,
) -> dict[ResidueId, set[ResidueId]]:
    """Hybrid residue graph. Two edge types:
      - helix-face: both endpoints in the same helix segment AND
                    |position_i - position_j| ∈ helix_face_offsets
      - spatial side-chain: NOT both in the same helix segment AND
                            min side-chain heavy-atom distance ≤ spatial_distance_a
    Same-helix pairs whose offset isn't in `helix_face_offsets` are NOT
    spatially connected — by design, to avoid gluing opposite helix faces
    via short i/i+1/i+2 distances."""
    adj: dict[ResidueId, set[ResidueId]] = {n: set() for n in nodes}
    sc_by_rid = {rec.rid: rec.sidechain_xyz for rec in records}
    offset_set = set(helix_face_offsets)

    # Helix-face edges (segment-restricted).
    for segment in segments:
        seg_nodes = [(pos, rid) for pos, rid in enumerate(segment) if rid in nodes]
        for a in range(len(seg_nodes)):
            pa, rida = seg_nodes[a]
            for b in range(a + 1, len(seg_nodes)):
                pb, ridb = seg_nodes[b]
                if abs(pa - pb) in offset_set:
                    adj[rida].add(ridb)
                    adj[ridb].add(rida)

    # Spatial side-chain edges (cross-segment / non-helix only).
    # Nodes pass through compute_residue_masif_scores, which already drops
    # residues with empty side chains; sc_by_rid is guaranteed to have a
    # non-empty array for every rid in nodes.
    nodes_list = list(nodes)
    for i, rida in enumerate(nodes_list):
        sca = sc_by_rid[rida]
        seg_a = rid_to_seg.get(rida, (-1, -1))[0]
        for ridb in nodes_list[i + 1:]:
            seg_b = rid_to_seg.get(ridb, (-1, -1))[0]
            if seg_a >= 0 and seg_b >= 0 and seg_a == seg_b:
                continue  # same helix segment → handled by helix-face edges only
            if cdist(sca, sc_by_rid[ridb]).min() <= spatial_distance_a:
                adj[rida].add(ridb)
                adj[ridb].add(rida)
    return adj


def _residue_connected_components(
    adj: dict[ResidueId, set[ResidueId]],
) -> list[list[ResidueId]]:
    visited: set[ResidueId] = set()
    components: list[list[ResidueId]] = []
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: list[ResidueId] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nbr in adj[v]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        components.append(comp)
    return components


def find_patch_residues(
    patch_vertex_xyz: np.ndarray,
    records: list[ResidueRecord],
    distance_a: float,
    min_rsasa: float,
) -> list[ResidueId]:
    """Re-extract residues whose heavy atoms come within `distance_a` of any
    patch vertex AND satisfy rSASA ≥ `min_rsasa`. Runs over ALL records, not
    just the allowed set: patches advertise their actual contact residues,
    which may be helix or non-helix. Also captures bystander residues that
    weren't strong enough to be component anchors but touch the patch surface."""
    candidates = [
        r for r in records
        if not np.isnan(r.rsasa) and r.rsasa >= min_rsasa and r.heavy_xyz.size > 0
    ]
    if not candidates or patch_vertex_xyz.shape[0] == 0:
        return []
    patch_tree = cKDTree(patch_vertex_xyz)
    out: list[ResidueId] = []
    for r in candidates:
        hits = patch_tree.query_ball_point(r.heavy_xyz, r=distance_a)
        if any(h for h in hits):
            out.append(r.rid)
    return out


def _component_check(comp_scores: list[float], cfg: EpitopesConfig) -> tuple[bool, str]:
    """Apply size-dependent acceptance rules to a component. Returns
    (passed, reason). `reason` is "" on pass, else the failing sub-gate."""
    n = len(comp_scores)
    max_s = max(comp_scores)
    if n == 2:
        if min(comp_scores) < cfg.core_score_threshold:
            return False, (f"min anchor score {min(comp_scores):.3f} < "
                           f"core_score_threshold {cfg.core_score_threshold}")
        if max_s < cfg.strong_score_threshold:
            return False, (f"max {max_s:.3f} < strong_score_threshold "
                           f"{cfg.strong_score_threshold}")
        return True, ""
    mean_s = sum(comp_scores) / n
    if mean_s < cfg.mean_anchor_score_threshold:
        return False, (f"mean {mean_s:.3f} < mean_anchor_score_threshold "
                       f"{cfg.mean_anchor_score_threshold}")
    if max_s < cfg.strong_score_threshold:
        return False, (f"max {max_s:.3f} < strong_score_threshold "
                       f"{cfg.strong_score_threshold}")
    if cfg.strict_mode:
        if mean_s < 0.60:
            return False, f"strict_mode: mean {mean_s:.3f} < 0.60"
        n_core = sum(1 for s in comp_scores if s >= cfg.core_score_threshold)
        if n_core < 2:
            return False, (f"strict_mode: only {n_core} anchors at core grade "
                           f"≥ {cfg.core_score_threshold} (need 2)")
    return True, ""


def extract_patches(
    vertices: np.ndarray,
    faces: np.ndarray,
    scores: np.ndarray,
    records: list[ResidueRecord],
    core: set[ResidueId],
    halo: set[ResidueId],
    cfg: EpitopesConfig,
    diag: dict | None = None,
) -> list[Patch]:
    """Build residue-level patches with class-specific node thresholds and
    size-dependent acceptance rules. See module docstring.

    If `diag` is given, fill it with intermediate state (per-residue scores,
    node membership, per-component pass/fail with reasons) so downstream
    callers can write a debug report — especially useful when 0 patches
    survive."""
    allowed = core | halo
    residue_scores, residue_vertex_idx = compute_residue_masif_scores(
        vertices, scores, records, allowed,
        distance_a=cfg.vertex_to_residue_distance_a,
        min_vertices=cfg.score_aggregation_min_vertices,
    )
    nodes: set[ResidueId] = set()
    node_status: dict[ResidueId, str] = {}
    for rid, s in residue_scores.items():
        if rid in core and s >= cfg.helix_node_score_threshold:
            nodes.add(rid)
            node_status[rid] = "node_core"
        elif rid in halo and s >= cfg.halo_node_score_threshold:
            nodes.add(rid)
            node_status[rid] = "node_halo"
        elif rid in core:
            node_status[rid] = (f"core_below_threshold "
                                f"({s:.3f} < {cfg.helix_node_score_threshold})")
        else:
            node_status[rid] = (f"halo_below_threshold "
                                f"({s:.3f} < {cfg.halo_node_score_threshold})")

    rec_by_rid = {r.rid: r for r in records}
    if diag is not None:
        diag["thresholds"] = {
            "vertex_to_residue_distance_a": cfg.vertex_to_residue_distance_a,
            "score_aggregation_min_vertices": cfg.score_aggregation_min_vertices,
            "helix_node_score_threshold": cfg.helix_node_score_threshold,
            "halo_node_score_threshold": cfg.halo_node_score_threshold,
            "core_score_threshold": cfg.core_score_threshold,
            "strong_score_threshold": cfg.strong_score_threshold,
            "mean_anchor_score_threshold": cfg.mean_anchor_score_threshold,
            "component_min_anchor_residues": cfg.component_min_anchor_residues,
            "expanded_patch_min_residues": cfg.expanded_patch_min_residues,
            "patch_residue_min_relative_sasa": cfg.patch_residue_min_relative_sasa,
            "strict_mode": cfg.strict_mode,
        }
        diag["totals"] = {
            "n_records": len(records),
            "n_core_residues": len(core),
            "n_halo_residues": len(halo),
            "n_residues_scored": len(residue_scores),
            "n_nodes": len(nodes),
            "vertex_score_min": float(scores.min()) if scores.size else None,
            "vertex_score_mean": float(scores.mean()) if scores.size else None,
            "vertex_score_max": float(scores.max()) if scores.size else None,
        }
        scored = []
        for rid, s in residue_scores.items():
            rec = rec_by_rid[rid]
            scored.append({
                "rid": str(rid), "resname": rec.resname, "ss": rec.ss,
                "rsasa": float(rec.rsasa) if not np.isnan(rec.rsasa) else None,
                "is_core": rid in core, "is_halo": rid in halo,
                "n_nearby_vertices": int(len(residue_vertex_idx[rid])),
                "masif_score": float(s),
                "is_node": rid in nodes,
                "node_status": node_status[rid],
            })
        scored.sort(key=lambda r: r["masif_score"], reverse=True)
        diag["scored_residues"] = scored

    if not nodes:
        if diag is not None:
            diag["totals"]["n_helix_segments"] = 0
            diag["totals"]["n_components"] = 0
            diag["totals"]["n_final_patches"] = 0
            diag["components"] = []
        return []

    segments, rid_to_seg = find_helix_segments(records, cfg.helix_codes)
    adj = build_residue_graph(
        nodes, records, segments, rid_to_seg,
        helix_face_offsets=cfg.helix_face_offsets,
        spatial_distance_a=cfg.spatial_sidechain_distance_a,
    )
    components = _residue_connected_components(adj)
    voronoi = vertex_voronoi_areas(vertices, faces)
    if diag is not None:
        diag["totals"]["n_helix_segments"] = len(segments)
        diag["totals"]["n_edges"] = sum(len(v) for v in adj.values()) // 2
        diag["totals"]["n_components"] = len(components)

    comp_diags: list[dict] = []
    patches: list[Patch] = []
    next_id = 0
    for comp in components:
        comp_scores = [residue_scores[rid] for rid in comp]
        cd: dict = {
            "size": len(comp),
            "anchor_rids": [str(r) for r in comp],
            "anchor_scores": [float(s) for s in comp_scores],
            "max_score": float(max(comp_scores)),
            "mean_score": float(sum(comp_scores) / len(comp_scores)),
            "passed_size_gate": len(comp) >= cfg.component_min_anchor_residues,
            "passed_score_gate": None,
            "score_gate_reason": "",
            "expanded_residue_ids": None,
            "passed_expanded_gate": None,
            "rejected_at": None,
        }
        if not cd["passed_size_gate"]:
            cd["rejected_at"] = "size"
            cd["score_gate_reason"] = (
                f"size {len(comp)} < component_min_anchor_residues "
                f"{cfg.component_min_anchor_residues}")
            comp_diags.append(cd)
            continue

        passed_score, reason = _component_check(comp_scores, cfg)
        cd["passed_score_gate"] = passed_score
        if not passed_score:
            cd["rejected_at"] = "score"
            cd["score_gate_reason"] = reason
            comp_diags.append(cd)
            continue

        vert_idx_set: set[int] = set()
        for rid in comp:
            vert_idx_set.update(int(v) for v in residue_vertex_idx[rid])
        vert_idx = np.fromiter(vert_idx_set, dtype=np.int64, count=len(vert_idx_set))
        vert_idx.sort()
        patch_residues = find_patch_residues(
            vertices[vert_idx], records,
            distance_a=cfg.vertex_to_residue_distance_a,
            min_rsasa=cfg.patch_residue_min_relative_sasa,
        )
        cd["expanded_residue_ids"] = [str(r) for r in patch_residues]
        cd["passed_expanded_gate"] = (
            len(patch_residues) >= cfg.expanded_patch_min_residues)
        if not cd["passed_expanded_gate"]:
            cd["rejected_at"] = "expanded"
            cd["score_gate_reason"] = (
                f"expanded patch has {len(patch_residues)} residues "
                f"< expanded_patch_min_residues {cfg.expanded_patch_min_residues}")
            comp_diags.append(cd)
            continue

        cd["rejected_at"] = None
        comp_diags.append(cd)
        patches.append(Patch(
            patch_id=next_id,
            vertex_indices=vert_idx,
            area_a2=float(voronoi[vert_idx].sum()),
            max_score=float(max(comp_scores)),
            residue_ids=patch_residues,
            component_residue_ids=list(comp),
            residue_masif_scores=[float(s) for s in comp_scores],
        ))
        next_id += 1

    if diag is not None:
        diag["components"] = comp_diags
        diag["totals"]["n_final_patches"] = len(patches)
    return patches
