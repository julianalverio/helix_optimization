from __future__ import annotations

import torch

from twistr.pipeline.models.rotation import frame_from_three_points, rotation_to_6d

from .chi_angles import chi_sincos, compute_chi_angles


def conditioning_mask(is_helix: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Per-residue boolean indicating which residues have their coordinates
    supplied as conditioning. Default policy: every real residue on a chain
    other than the helix's chain (i.e. partner residues) is conditioned — the
    model is given the partner structure and predicts the helix in response.
    Padded residues (padding_mask == False) are excluded — they have no
    coordinates to condition on."""
    return (~is_helix.bool()) & padding_mask.bool()


def conditioning_translation_validity(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, N) bool: True iff the residue is conditioning AND its CA atom is
    present. The companion `conditioning_translation` returns zeros wherever
    this is False — model uses this flag to ignore those positions."""
    cond = conditioning_mask(batch["is_helix"], batch["padding_mask"])
    ca_present = batch["atom_mask"][..., 1] == 1
    return cond & ca_present


def conditioning_frame_6d_validity(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, N) bool: True iff the residue is conditioning AND all three frame-
    defining atoms (N, CA, C) are present."""
    cond = conditioning_mask(batch["is_helix"], batch["padding_mask"])
    n_present = batch["atom_mask"][..., 0] == 1
    ca_present = batch["atom_mask"][..., 1] == 1
    c_present = batch["atom_mask"][..., 2] == 1
    return cond & n_present & ca_present & c_present


def conditioning_translation(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """CA position (in the dataset's centered + /10 frame) for residues with
    valid conditioning data (conditioning AND CA present), zeros elsewhere.
    Shape (B, N, 3). Pair with `conditioning_translation_validity`."""
    ca = batch["coordinates"][..., 1, :]
    valid = conditioning_translation_validity(batch).unsqueeze(-1)
    return torch.where(valid, ca, torch.zeros_like(ca))


def conditioning_frame_6d(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """6D rotation parameterization (Zhou et al. 2019) of each residue's
    frame for residues with valid conditioning data (conditioning AND all
    of N/CA/C present), zeros elsewhere. Shape (B, N, 6)."""
    coords = batch["coordinates"]
    N, CA, C = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    R = frame_from_three_points(N, CA, C)
    rot_6d = rotation_to_6d(R)
    valid = conditioning_frame_6d_validity(batch).unsqueeze(-1)
    return torch.where(valid, rot_6d, torch.zeros_like(rot_6d))


def conditioning_chi_sincos(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sidechain conformation of each conditioned residue, encoded as (sin, cos)
    pairs of the four chi dihedrals. Shape (B, N, 4, 2). Zeroed where the chi
    doesn't exist for the residue type, where any defining atom is missing,
    or where the residue itself is not conditioning. Companion validity mask
    is exposed as conditioning_chi_validity."""
    angles, validity = compute_chi_angles(
        batch["coordinates"], batch["residue_type"], batch["atom_mask"],
    )
    sincos = chi_sincos(angles)
    cond = conditioning_mask(batch["is_helix"], batch["padding_mask"]).unsqueeze(-1)
    keep = (validity & cond).unsqueeze(-1)
    return torch.where(keep, sincos, torch.zeros_like(sincos))


def conditioning_chi_validity(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-chi boolean: True iff the chi exists for the residue type, all four
    defining atoms are present, AND the residue is conditioning. Shape
    (B, N, 4). The model uses this to ignore zeroed chi slots."""
    _, validity = compute_chi_angles(
        batch["coordinates"], batch["residue_type"], batch["atom_mask"],
    )
    cond = conditioning_mask(batch["is_helix"], batch["padding_mask"]).unsqueeze(-1)
    return validity & cond
