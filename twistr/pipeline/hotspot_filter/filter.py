"""Per-patch hotspot stats (ranking, not rejection).

PPI-hotspotID is used to *score and annotate* every patch, not to drop any.
For each patch we count the hotspot residues, capture their scores, and run
the legacy accept-path classifier so its verdict is still available as a
ranking diagnostic in the `accept_path` column. No patches are filtered out
here — every input patch flows to the output with its hotspot stats stamped
on.

Hotspot identification (unchanged):
  1. Pick a hotspot cutoff:
     - tool-recommended threshold (`hotspot_score_threshold`, default 0.5)
     - fall back to top `hotspot_top_fraction_fallback` of scored residues
       per PDB if fewer than `hotspot_min_residues_for_cutoff` qualify
  2. For each patch, hotspots = patch residues whose score passes the cutoff.

Accept-path classifier (now informational only — populates `accept_path`):
       n_hotspots ≥ 2  →  "two_hotspots"
       n_hotspots == 1 →  define N = residues whose sidechain heavy atoms come
                          within `cluster_neighbor_distance_a` of the hotspot's
                          sidechain heavy atoms (over the FULL structure, not
                          just the patch).
                          → "one_hotspot_<cluster>" if N satisfies any of
                            (hydrophobic_aromatic / aromatic / charged_polar /
                            mixed); else "rejected_no_cluster".
                          GLY (no sidechain) → "rejected_hotspot_no_sidechain".
       n_hotspots == 0 →  "rejected_no_hotspot"
The "rejected_*" labels are vestigial naming from when this stage was a
filter; they now describe what the legacy filter *would have* done.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from ..epitopes.filter import ResidueId, ResidueRecord
from .config import HotspotConfig


# Standard 3-letter → 1-letter amino acid mapping.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def one_letter(resname: str) -> str:
    """3-letter PDB resname → 1-letter code; '?' for non-standard."""
    return _THREE_TO_ONE.get(resname, "?")


@dataclass(frozen=True)
class HotspotDecision:
    """Per-patch hotspot stats. `accepted` and `accept_path` are now
    informational only — every patch is kept regardless. The fields exist
    for ranking and for downstream consumers that want to know what the
    legacy filter would have decided."""
    accepted: bool
    n_hotspots: int
    hotspot_residue_ids: list[ResidueId]
    hotspot_scores: list[float]
    accept_path: str           # "two_hotspots" | "one_hotspot_<cluster>" | "rejected_*"
    cluster_neighbors: list[ResidueId]  # populated for the 1-hotspot path


def select_hotspots(
    scores: dict[ResidueId, float], cfg: HotspotConfig,
) -> set[ResidueId]:
    """Return the set of residue IDs that count as hotspots for this PDB.
    Uses cfg.hotspot_score_threshold; if fewer than
    cfg.hotspot_min_residues_for_cutoff qualify, falls back to the top
    cfg.hotspot_top_fraction_fallback of residues by score."""
    if not scores:
        return set()
    above_cutoff = {rid for rid, s in scores.items() if s >= cfg.hotspot_score_threshold}
    if len(above_cutoff) >= cfg.hotspot_min_residues_for_cutoff:
        return above_cutoff
    # Top-fraction fallback. ceil so that small structures still have a few hotspots.
    n_top = max(1, math.ceil(len(scores) * cfg.hotspot_top_fraction_fallback))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {rid for rid, _ in ranked[:n_top]}


def _sidechain_neighbors(
    hotspot: ResidueRecord,
    candidates: list[ResidueRecord],
    distance_a: float,
) -> list[ResidueRecord]:
    """All residues with at least one sidechain heavy atom within distance_a
    of any hotspot sidechain heavy atom. Excludes the hotspot itself."""
    if hotspot.sidechain_xyz.size == 0:
        return []
    out: list[ResidueRecord] = []
    for cand in candidates:
        if cand.rid == hotspot.rid or cand.sidechain_xyz.size == 0:
            continue
        if cdist(hotspot.sidechain_xyz, cand.sidechain_xyz).min() <= distance_a:
            out.append(cand)
    return out


def _classify_cluster(
    neighbor_resnames_one: set[str], cfg: HotspotConfig,
) -> str | None:
    """Return the first cluster type that the neighbor set satisfies, or None."""
    n_hp = len(neighbor_resnames_one & set(cfg.cluster_hydrophobic_aromatic_set))
    if n_hp >= 2:
        return "hydrophobic_aromatic"
    n_ar = len(neighbor_resnames_one & set(cfg.cluster_aromatic_set))
    if n_ar >= 1:
        return "aromatic"
    n_cp = len(neighbor_resnames_one & set(cfg.cluster_charged_polar_set))
    if n_cp >= 2:
        return "charged_polar"
    has_mix_hp = bool(neighbor_resnames_one & set(cfg.cluster_mixed_hydrophobic_set))
    has_mix_pol = bool(neighbor_resnames_one & set(cfg.cluster_mixed_polar_set))
    if has_mix_hp and has_mix_pol:
        return "mixed"
    return None


def evaluate_patch(
    patch_residue_ids: list[ResidueId],
    pdb_scores: dict[ResidueId, float],
    hotspot_set: set[ResidueId],
    records: list[ResidueRecord],
    cfg: HotspotConfig,
) -> HotspotDecision:
    """Apply the spec's accept/reject rules to one patch."""
    record_by_rid = {r.rid: r for r in records}
    patch_set = set(patch_residue_ids)
    patch_hotspots = sorted(patch_set & hotspot_set, key=lambda r: (r.chain, r.seq, r.icode))
    hotspot_scores = [pdb_scores.get(rid, float("nan")) for rid in patch_hotspots]
    n_hot = len(patch_hotspots)

    if n_hot >= 2:
        return HotspotDecision(
            accepted=True, n_hotspots=n_hot,
            hotspot_residue_ids=patch_hotspots, hotspot_scores=hotspot_scores,
            accept_path="two_hotspots", cluster_neighbors=[],
        )
    if n_hot == 0:
        return HotspotDecision(
            accepted=False, n_hotspots=0,
            hotspot_residue_ids=[], hotspot_scores=[],
            accept_path="rejected_no_hotspot", cluster_neighbors=[],
        )

    # n_hot == 1 → cluster check.
    hotspot_rid = patch_hotspots[0]
    hotspot_rec = record_by_rid.get(hotspot_rid)
    if hotspot_rec is None or hotspot_rec.sidechain_xyz.size == 0:
        # Hotspot has no sidechain (GLY) → no cluster can form → reject.
        return HotspotDecision(
            accepted=False, n_hotspots=1,
            hotspot_residue_ids=patch_hotspots, hotspot_scores=hotspot_scores,
            accept_path="rejected_hotspot_no_sidechain", cluster_neighbors=[],
        )

    neighbors = _sidechain_neighbors(
        hotspot_rec, records, cfg.cluster_neighbor_distance_a,
    )
    neighbor_resnames = {one_letter(n.resname) for n in neighbors}
    cluster_hit = _classify_cluster(neighbor_resnames, cfg)
    if cluster_hit is None:
        return HotspotDecision(
            accepted=False, n_hotspots=1,
            hotspot_residue_ids=patch_hotspots, hotspot_scores=hotspot_scores,
            accept_path="rejected_no_cluster", cluster_neighbors=[n.rid for n in neighbors],
        )
    return HotspotDecision(
        accepted=True, n_hotspots=1,
        hotspot_residue_ids=patch_hotspots, hotspot_scores=hotspot_scores,
        accept_path=f"one_hotspot_{cluster_hit}",
        cluster_neighbors=[n.rid for n in neighbors],
    )


def filter_patches_for_pdb(
    pdb_id: str,
    pdb_patches,                                     # pd.DataFrame
    pdb_scores: dict[ResidueId, float],
    records: list[ResidueRecord],
    cfg: HotspotConfig,
):
    """Augment every scannet patch row with per-patch hotspot stats. No row
    is dropped — `accept_path` is the legacy filter's verdict and is now a
    ranking diagnostic only."""
    import pandas as pd
    from ..scannet_filter.filter import parse_residue_id

    if pdb_patches.empty:
        return pdb_patches.iloc[0:0].copy()

    hotspot_set = select_hotspots(pdb_scores, cfg)
    rows: list[dict] = []
    for _, patch in pdb_patches.iterrows():
        residue_ids = [parse_residue_id(s) for s in patch["residue_ids"]]
        decision = evaluate_patch(
            residue_ids, pdb_scores, hotspot_set, records, cfg,
        )
        rows.append({
            **{k: patch[k] for k in patch.index},
            "n_hotspots": decision.n_hotspots,
            "hotspot_residue_ids": [str(r) for r in decision.hotspot_residue_ids],
            "hotspot_scores": [float(s) for s in decision.hotspot_scores],
            "accept_path": decision.accept_path,
            "cluster_neighbor_residue_ids": [str(r) for r in decision.cluster_neighbors],
            "n_cluster_neighbors": len(decision.cluster_neighbors),
        })
    if not rows:
        empty = pdb_patches.iloc[0:0].copy()
        for col, dt in [
            ("n_hotspots", "int64"), ("hotspot_residue_ids", "object"),
            ("hotspot_scores", "object"), ("accept_path", "object"),
            ("cluster_neighbor_residue_ids", "object"), ("n_cluster_neighbors", "int64"),
        ]:
            empty[col] = pd.Series(dtype=dt)
        return empty
    return pd.DataFrame(rows)
