from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpitopesConfig:
    pdb_dir: str
    output_path: str
    work_dir: str

    helix_codes: tuple[str, ...] = ("H", "G")
    core_min_relative_sasa: float = 0.3
    halo_distance_a: float = 6.0
    halo_min_relative_sasa: float = 0.15

    # Distance from a residue's side-chain heavy atoms to MaSIF mesh vertices,
    # used both for residue MaSIF score aggregation and for the patch residue
    # re-extraction step.
    vertex_to_residue_distance_a: float = 4.0
    # Minimum vertices needed near a residue's side-chain to compute its score;
    # if fewer are present, the residue is "MaSIF-unsupported" and excluded.
    score_aggregation_min_vertices: int = 3
    # Helix-face position offsets (|i-j| in this set → helix-face edge between
    # nodes in the same DSSP helix segment).
    helix_face_offsets: tuple[int, ...] = (3, 4, 7, 8)
    # Max side-chain heavy-atom min-distance for spatial side-chain edges.
    spatial_sidechain_distance_a: float = 5.0

    # ---------------- Graph-node thresholds (split by residue class) ---------
    # A core residue (DSSP helix + rSASA ≥ core_min_relative_sasa) becomes a
    # graph node when its top-quartile-mean MaSIF score is ≥ this.
    helix_node_score_threshold: float = 0.55
    # A halo residue (immediate neighbor + rSASA ≥ halo_min_relative_sasa)
    # becomes a graph node when its top-quartile-mean score is ≥ this.
    halo_node_score_threshold: float = 0.50

    # ---------------- Patch-acceptance thresholds ---------------------------
    # "Core" anchor — used by the 2-anchor "both must be core" check.
    core_score_threshold: float = 0.70
    # "Strong" anchor — patch must contain at least one residue at or above this.
    strong_score_threshold: float = 0.85
    # First-pass mean-score gate for ≥ 3-anchor components.
    mean_anchor_score_threshold: float = 0.55

    # ---------------- Component / expanded size gates -----------------------
    # Minimum graph-component (anchor) size. Sparse helix-face seeds (i/i+4)
    # are valid 2-anchor patches and survive when 2-anchor rules pass.
    component_min_anchor_residues: int = 2
    # Minimum |residue_ids| (re-extracted bystander-inclusive set) per patch.
    expanded_patch_min_residues: int = 5
    # rSASA floor for re-extracted residues.
    patch_residue_min_relative_sasa: float = 0.15

    # Opt-in stricter mode for ≥ 3-anchor components. When True, additionally
    # require mean_anchor_score ≥ 0.60 AND ≥ 2 anchors with score ≥ 0.70.
    strict_mode: bool = False

    masif_image: str = "pablogainza/masif:latest"
    masif_platform: str = "linux/amd64"
