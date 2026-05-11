from __future__ import annotations

import math

import torch

from twistr.pipeline.features.chi_angles import atan2_dihedral

# Alpha-helix Ramachandran region. Inside the box the loss is zero;
# outside, the penalty grows linearly in radians of out-of-range distance.
HELIX_PHI_RANGE_DEG = (-100.0, -30.0)
HELIX_PSI_RANGE_DEG = (-80.0, -10.0)
HELIX_OMEGA_TOL_DEG = 10.0  # |π − |ω|| ≤ tol → trans peptide bond


def helix_dihedral_loss(
    atom14: torch.Tensor,
    is_helix: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Flat-bottomed linear penalty on backbone (φ, ψ, ω) for residues in
    `is_helix`. Inputs: atom14 (B, N, 14, 3) — slots 0/1/2 are N/CA/C as
    placed by `apply_torsions_to_atom14`; is_helix (B, N) bool;
    padding_mask (B, N) bool with True = real residue. Returns a scalar —
    per-example mean over interior helix residues, then mean over the batch.
    Boundary helix residues (no left or right neighbour) are excluded by
    requiring the dihedral triple to be all-helix; padded residues (any of
    the triple is padding) are excluded via padding_mask."""
    N_atom = atom14[..., 0, :]
    CA = atom14[..., 1, :]
    C = atom14[..., 2, :]

    phi = atan2_dihedral(C[:, :-2], N_atom[:, 1:-1], CA[:, 1:-1], C[:, 1:-1])
    psi = atan2_dihedral(N_atom[:, 1:-1], CA[:, 1:-1], C[:, 1:-1], N_atom[:, 2:])
    omega = atan2_dihedral(CA[:, :-2], C[:, :-2], N_atom[:, 1:-1], CA[:, 1:-1])

    real_triple = padding_mask[:, :-2] & padding_mask[:, 1:-1] & padding_mask[:, 2:]
    mask = is_helix[:, :-2] & is_helix[:, 1:-1] & is_helix[:, 2:] & real_triple

    phi_lo, phi_hi = math.radians(HELIX_PHI_RANGE_DEG[0]), math.radians(HELIX_PHI_RANGE_DEG[1])
    psi_lo, psi_hi = math.radians(HELIX_PSI_RANGE_DEG[0]), math.radians(HELIX_PSI_RANGE_DEG[1])
    omega_tol = math.radians(HELIX_OMEGA_TOL_DEG)

    phi_loss = (phi_lo - phi).clamp_min(0) + (phi - phi_hi).clamp_min(0)
    psi_loss = (psi_lo - psi).clamp_min(0) + (psi - psi_hi).clamp_min(0)
    omega_loss = ((math.pi - omega.abs()) - omega_tol).clamp_min(0)

    per_res = phi_loss + psi_loss + omega_loss
    mask_f = mask.to(per_res.dtype)
    denom = mask_f.sum(dim=-1)                                                  # (B,)
    per_example = (per_res * mask_f).sum(dim=-1) / denom.clamp_min(1.0)
    has_signal = (denom > 0).to(per_res.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
