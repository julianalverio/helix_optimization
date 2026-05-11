"""Render a PyMOL script for one valid epitope patch.

The script is self-contained: `reinitialize` clears any prior session, then
the structure is loaded with the chains shown as gray cartoon on a black
background. Patch residues are shown as sticks colored by a 4-class
biochemical grouping (hydrophobic/negative/positive/polar)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..scannet_filter.filter import parse_residue_id
from .aa_classes import classify
from .config import EpitopeVizConfig

logger = logging.getLogger(__name__)


def build_pml(
    pdb_id: str,
    patch_id: str,
    patch_residue_ids: list[str],
    pdb_path: Path,
    aa_lookup: dict[tuple[str, int, str], str],
    cfg: EpitopeVizConfig,
    hotspot_residue_ids: list[str] | None = None,
) -> str:
    """Build the .pml script text for one patch.

    `aa_lookup` maps (chain, seq, icode) → one-letter amino acid code.
    Residues whose AA can't be looked up or doesn't fit any of the four
    canonical categories are skipped (with a warning) — they won't get sticks.

    If `hotspot_residue_ids` is given, those residues are re-colored with
    `cfg.hotspot_color` *after* the 4-class coloring, so they override and
    visually pop out from the rest of the patch.
    """
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for rid_str in patch_residue_ids:
        rid = parse_residue_id(rid_str)
        aa = aa_lookup.get((rid.chain, rid.seq, rid.icode))
        if aa is None:
            logger.warning("%s/%s: residue %s not found in PDB; skipping",
                           pdb_id, patch_id, rid_str)
            continue
        cat = classify(aa)
        if cat is None:
            logger.warning("%s/%s: residue %s (%s) is non-canonical; skipping",
                           pdb_id, patch_id, rid_str, aa)
            continue
        buckets.setdefault((rid.chain, cat), []).append((rid.seq, rid.icode))

    lines = [
        "reinitialize",
        f"bg_color {cfg.background_color}",
        f"load {pdb_path.absolute()}, {pdb_id}",
        "hide everything",
        "show cartoon",
        f"color {cfg.chain_color}, {pdb_id}",
        "",
    ]

    sel_names: list[str] = []
    for (chain, cat) in sorted(buckets):
        seqs = buckets[(chain, cat)]
        resi_str = "+".join(f"{s}{ic}" if ic else f"{s}"
                            for s, ic in sorted(set(seqs)))
        sel_name = f"patch_{chain}_{cat}"
        color = cfg.sticks_colors[cat]
        lines.extend([
            f"select {sel_name}, {pdb_id} and chain {chain} and resi {resi_str}",
            f"show sticks, {sel_name}",
            f"color {color}, {sel_name}",
            "",
        ])
        sel_names.append(sel_name)

    # Hotspot override (per chain). Done after the 4-class loop so hotspot
    # color wins. Hotspots are a subset of patch residues and are already
    # shown as sticks above; we only need to recolor.
    hotspots_by_chain: dict[str, list[tuple[int, str]]] = {}
    for rid_str in hotspot_residue_ids or []:
        rid = parse_residue_id(rid_str)
        hotspots_by_chain.setdefault(rid.chain, []).append((rid.seq, rid.icode))
    for chain in sorted(hotspots_by_chain):
        seqs = hotspots_by_chain[chain]
        resi_str = "+".join(f"{s}{ic}" if ic else f"{s}"
                            for s, ic in sorted(set(seqs)))
        sel_name = f"patch_{chain}_hotspot"
        lines.extend([
            f"select {sel_name}, {pdb_id} and chain {chain} and resi {resi_str}",
            f"color {cfg.hotspot_color}, {sel_name}",
            "",
        ])
        sel_names.append(sel_name)

    if sel_names:
        lines.extend([
            f"select patch_all, {' or '.join(sel_names)}",
            "orient patch_all",
            f"zoom patch_all, {cfg.zoom_padding_a}",
        ])

    return "\n".join(lines) + "\n"
