"""Patch-level ScanNet stats — used for ranking, not rejection by default.

Reads MaSIF's `patches.parquet`, runs ScanNet once per PDB to score every
residue, then for each patch:
  - Looks up scores for the patch's `residue_ids` (the re-extracted set
    from the MaSIF stage).
  - Computes mean / max / positive fraction (where positive = score ≥ τ)
    and stamps them onto the row for downstream ranking.
  - Drops patches with fewer than `patch_min_residues` scored residues
    (sanity gate, not a score-based reject).

The score-threshold knobs (`patch_min_mean_score`, `patch_min_max_score`,
`patch_min_positive_fraction`) remain in the config and are still applied
here, but the shipped defaults are 0.0 — i.e. ScanNet is currently a
ranker, not a filter. Setting them non-zero re-enables hard rejection.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..epitopes.filter import ResidueId
from .config import ScanNetConfig

logger = logging.getLogger(__name__)


def _split_seq_icode(s: str) -> tuple[int, str]:
    """`123` → (123, ''); `123A` → (123, 'A'); `-5` → (-5, '')."""
    end = len(s)
    while end > 0 and not s[end - 1].isdigit() and s[end - 1] != "-":
        end -= 1
    return int(s[:end]), s[end:].strip()


def parse_residue_id(s: str) -> ResidueId:
    """Inverse of `ResidueId.__str__`. `"A/123A"` → ResidueId("A",123,"A")."""
    chain, rest = s.split("/", 1)
    seq, icode = _split_seq_icode(rest)
    return ResidueId(chain=chain, seq=seq, icode=icode)


def filter_patches_for_pdb(
    pdb_id: str,
    pdb_patches: pd.DataFrame,
    scannet_scores: dict[ResidueId, float],
    cfg: ScanNetConfig,
) -> pd.DataFrame:
    """Augment one PDB's patch rows with ScanNet stats. Drops rows with
    fewer than `cfg.patch_min_residues` scored residues; otherwise also
    drops rows that fail the (mean OR pos_frac) gate or the max-score floor
    if those thresholds are non-zero in `cfg`. With the shipped zero
    defaults the function is annotation-only."""
    rows: list[dict] = []
    for _, patch in pdb_patches.iterrows():
        kept = [(s, scannet_scores[r])
                for s in patch["residue_ids"]
                if (r := parse_residue_id(s)) in scannet_scores]
        n_scored = len(kept)
        if n_scored < cfg.patch_min_residues:
            continue
        kept_strs = [s for s, _ in kept]
        scores = np.fromiter((v for _, v in kept), dtype=np.float64, count=n_scored)
        mean = float(scores.mean())
        max_ = float(scores.max())
        pos_frac = float((scores >= cfg.residue_positive_threshold).mean())
        # Score gates: require (mean ≥ threshold) OR (pos_frac ≥ threshold).
        # Either signal alone is enough — a uniformly-decent patch passes
        # via mean, a strongly-bimodal patch with a hot core passes via
        # pos_frac. Then enforce the max-score floor independently.
        if mean < cfg.patch_min_mean_score and pos_frac < cfg.patch_min_positive_fraction:
            continue
        if max_ < cfg.patch_min_max_score:
            continue
        rows.append({
            **{k: patch[k] for k in patch.index},
            "scannet_n_scored": n_scored,
            "scannet_mean": mean,
            "scannet_max": max_,
            "scannet_pos_frac": pos_frac,
            "scannet_residue_scores": [float(s) for s in scores],
            "scannet_scored_residue_ids": kept_strs,
        })
    if not rows:
        return pdb_patches.iloc[0:0].assign(
            scannet_n_scored=pd.Series(dtype="int64"),
            scannet_mean=pd.Series(dtype="float64"),
            scannet_max=pd.Series(dtype="float64"),
            scannet_pos_frac=pd.Series(dtype="float64"),
            scannet_residue_scores=pd.Series(dtype="object"),
            scannet_scored_residue_ids=pd.Series(dtype="object"),
        )
    return pd.DataFrame(rows)
