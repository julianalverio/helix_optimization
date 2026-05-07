from __future__ import annotations

import torch

from twistr.ml.config import MLConfig

from .chi_angles import chi_mask
from .conditioning import (
    conditioning_chi_sincos,
    conditioning_chi_validity,
    conditioning_frame_6d,
    conditioning_frame_6d_validity,
    conditioning_mask,
    conditioning_translation,
    conditioning_translation_validity,
)
from .interactions import clean_interaction_matrix, conditioning_interaction_matrix
from .residue_type import one_hot_residue_type


def build_features(
    batch: dict[str, torch.Tensor],
    cfg: MLConfig,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Compute every input feature the model consumes from a raw batch produced
    by ExamplesDataset. Add new features by importing the function above and
    assigning into the returned dict — one feature per function, named here.

    `target_interaction_matrix` is the clean binary (B, N, N, 6) label
    matrix; the loss uses it as the BCE / geometric-loss target. The
    conditioning input is the same matrix with noise applied (plus an
    augmentation-mask bit and a padding-mask bit, → (B, N, N, 8)) —
    both share the single `clean_interaction_matrix` call to avoid
    recomputing the detector."""
    target_im = clean_interaction_matrix(batch)
    padding_mask = batch["padding_mask"]
    return {
        "residue_type_one_hot": one_hot_residue_type(batch["residue_type"]),
        "chi_mask": chi_mask(batch["residue_type"]),
        "conditioning_mask": conditioning_mask(batch["is_helix"], padding_mask),
        "conditioning_translation": conditioning_translation(batch),
        "conditioning_translation_validity": conditioning_translation_validity(batch),
        "conditioning_frame_6d": conditioning_frame_6d(batch),
        "conditioning_frame_6d_validity": conditioning_frame_6d_validity(batch),
        "conditioning_chi_sincos": conditioning_chi_sincos(batch),
        "conditioning_chi_validity": conditioning_chi_validity(batch),
        "target_interaction_matrix": target_im,
        "conditioning_interaction_matrix": conditioning_interaction_matrix(target_im, batch, cfg, generator),
        "padding_mask": padding_mask,
        "chain_slot": batch["chain_slot"],
    }
