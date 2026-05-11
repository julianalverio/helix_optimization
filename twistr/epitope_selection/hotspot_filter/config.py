from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HotspotConfig:
    scannet_parquet: str
    pdb_dir: str
    output_path: str
    work_dir: str

    # Hotspot cutoff. AutoGluon emits a per-residue probability; residues whose
    # score ≥ this are hotspots. If too few qualify, fall back to top decile.
    hotspot_score_threshold: float = 0.5
    hotspot_min_residues_for_cutoff: int = 5
    hotspot_top_fraction_fallback: float = 0.10

    # Neighbor-set construction: residues with sidechain heavy atoms within
    # `cluster_neighbor_distance_a` of the hotspot's sidechain heavy atoms.
    cluster_neighbor_distance_a: float = 5.0

    # AA sets per cluster type (one-letter codes).
    cluster_hydrophobic_aromatic_set: tuple[str, ...] = ("L", "I", "V", "M", "F", "Y", "W")
    cluster_aromatic_set: tuple[str, ...] = ("F", "Y", "W")
    cluster_charged_polar_set: tuple[str, ...] = ("K", "R", "H", "D", "E", "N", "Q", "Y", "W")
    # The "mixed interaction cluster" needs at least one large hydrophobic/
    # aromatic residue AND at least one charged-or-strong-polar residue.
    cluster_mixed_hydrophobic_set: tuple[str, ...] = ("L", "I", "V", "M", "F", "Y", "W")
    cluster_mixed_polar_set: tuple[str, ...] = ("K", "R", "H", "D", "E", "N", "Q", "Y", "W")
