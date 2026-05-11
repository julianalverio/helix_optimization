from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SpatialIndex:
    tree: cKDTree
    atom_chain: np.ndarray
    atom_residue: np.ndarray
    atom_slot: np.ndarray


def build_spatial_index(coordinates: np.ndarray, atom_mask: np.ndarray) -> SpatialIndex:
    """coordinates: (n_chains, n_max, 14, 3) float16. atom_mask: (n_chains, n_max, 14) int8.
    Indexes all atoms where atom_mask == 1."""
    chain_idx, res_idx, slot_idx = np.where(atom_mask == 1)
    points = coordinates[chain_idx, res_idx, slot_idx].astype(np.float32)
    tree = cKDTree(points)
    return SpatialIndex(
        tree=tree,
        atom_chain=chain_idx.astype(np.int32),
        atom_residue=res_idx.astype(np.int32),
        atom_slot=slot_idx.astype(np.int32),
    )


def _residue_heavy_atom_points(coordinates: np.ndarray, atom_mask: np.ndarray,
                               chain_index: int, residue_position: int) -> np.ndarray:
    slots = np.where(atom_mask[chain_index, residue_position] == 1)[0]
    if len(slots) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return coordinates[chain_index, residue_position, slots].astype(np.float32)


def mark_contacting_residues(
    index: SpatialIndex,
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    chain_index: int,
    segment_start: int,
    segment_end: int,
    radius: float,
) -> np.ndarray:
    """Return bool array of length (segment_end - segment_start + 1) marking residues
    with at least one heavy atom within `radius` of a heavy atom on a different chain."""
    n = segment_end - segment_start + 1
    out = np.zeros(n, dtype=bool)
    for i, res_pos in enumerate(range(segment_start, segment_end + 1)):
        pts = _residue_heavy_atom_points(coordinates, atom_mask, chain_index, res_pos)
        if pts.size == 0:
            continue
        hits = index.tree.query_ball_point(pts, r=radius)
        for idxs in hits:
            if not idxs:
                continue
            neighbor_chains = index.atom_chain[idxs]
            if np.any(neighbor_chains != chain_index):
                out[i] = True
                break
    return out


def partner_chains_for_window(
    index: SpatialIndex,
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    helix_chain_index: int,
    window_residue_positions: list[int],
    radius: float,
) -> list[int]:
    """Return sorted list of chain indices contacted by any window residue, excluding the helix chain."""
    partners: set[int] = set()
    for res_pos in window_residue_positions:
        pts = _residue_heavy_atom_points(coordinates, atom_mask, helix_chain_index, res_pos)
        if pts.size == 0:
            continue
        hits = index.tree.query_ball_point(pts, r=radius)
        for idxs in hits:
            if not idxs:
                continue
            neighbor_chains = index.atom_chain[idxs]
            for c in np.unique(neighbor_chains):
                if int(c) != helix_chain_index:
                    partners.add(int(c))
    return sorted(partners)


def distance_interface_partners(
    index: SpatialIndex,
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    helix_chain_index: int,
    window_residue_positions: list[int],
    partner_chain_index: int,
    radius: float,
) -> set[int]:
    """Return set of residue positions on partner_chain that have any heavy atom within `radius`
    of any helix window heavy atom."""
    hit_residues: set[int] = set()
    for res_pos in window_residue_positions:
        pts = _residue_heavy_atom_points(coordinates, atom_mask, helix_chain_index, res_pos)
        if pts.size == 0:
            continue
        hits = index.tree.query_ball_point(pts, r=radius)
        for idxs in hits:
            if not idxs:
                continue
            neighbor_chains = index.atom_chain[idxs]
            neighbor_residues = index.atom_residue[idxs]
            mask = neighbor_chains == partner_chain_index
            if np.any(mask):
                hit_residues.update(int(r) for r in neighbor_residues[mask])
    return hit_residues
