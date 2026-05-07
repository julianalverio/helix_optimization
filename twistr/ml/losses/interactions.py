"""Per-interaction-type geometric losses on predicted full-atom coordinates.

Each loss is **two-sided** and **flat-bottomed linear**:
  - For pairs with GT=1 (interaction must form): zero loss inside the geometric
    band, linear in physical units (Å, cosine) outside. Reduced as the
    minimum violation over atom-level alternatives — at least one alternative
    must satisfy the band.
  - For pairs with GT=0 (interaction must not form): zero loss outside the
    band, linear in physical units inside. Reduced as the maximum in-band
    margin — no alternative may satisfy the band.

The detector at `twistr/ml/features/interactions.py` defines the bands; this
module recomputes the underlying distances/cosines/centroids directly (the
detector returns sigmoid-banded scores, not raw out-of-range distances).

Inputs are expected in **Angstroms** (caller multiplies by COORD_SCALE_ANGSTROMS).
Backbone O (atom14 slot 3) is placed by the model's predicted psi torsion
via the AF2 psi rigid group — see `twistr/ml/models/sidechain.py:apply_torsions_to_atom14`.
No peptide-plane reconstruction is done here."""
from __future__ import annotations

import torch

from twistr.ml.features.interactions import (
    AROM_PARA_HI,
    AROM_PARA_LO,
    AROM_PD_D,
    AROM_PD_DPAR,
    AROM_SANDWICH_D,
    AROM_SANDWICH_DPAR_HI,
    AROM_T_D,
    AROMATIC_ATOM_MASK,
    AROMATIC_RING_SLOTS,
    ELEMENT_VDW,
    HBOND_ACCEPTORS_ATOM14,
    HBOND_ACCEPTORS_MASK,
    HBOND_COS_DAY_THRESH,
    HBOND_COS_XDA_THRESH,
    HBOND_DIST_HI_A,
    HBOND_DIST_LO_A,
    HBOND_DONORS_ATOM14,
    HBOND_DONORS_MASK,
    IS_AROMATIC,
    VDW_BAND_HI_OFFSET_A,
    VDW_BAND_LO_OFFSET_A,
    VDW_RADII,
    _gather_atom14,
    _ring_centroid_normal,
    _safe_norm,
    _safe_normalize,
)


# ----------------------------------------------------------------------
# VDW loss.

def vdw_interaction_loss(
    coords_atom14_ang: torch.Tensor,    # (B, N, 14, 3) in Å
    residue_type: torch.Tensor,         # (B, N) long
    atom_mask: torch.Tensor,            # (B, N, 14) int8 in {-1, 0, 1}
    target_vdw: torch.Tensor,           # (B, N, N) bool/float in {0, 1}
    padding_mask: torch.Tensor,         # (B, N) bool, True = real residue
) -> torch.Tensor:
    """Per-example mean over off-diagonal real-residue pairs of the VDW
    geometric loss, then mean over the batch — length-invariant."""
    B, N = residue_type.shape
    device = coords_atom14_ang.device
    vdw_r = VDW_RADII.to(device)[residue_type]                                # (B, N, 14)

    sidechain = torch.zeros(14, dtype=torch.bool, device=device)
    sidechain[4:] = True
    atom_present = (atom_mask == 1) & sidechain

    ca = coords_atom14_ang[:, :, None, :, None, :]                            # (B, N, 1, 14, 1, 3)
    cb = coords_atom14_ang[:, None, :, None, :, :]
    d = (ca - cb).norm(dim=-1)                                                # (B, N, N, 14, 14)

    r_sum = vdw_r[:, :, None, :, None] + vdw_r[:, None, :, None, :]
    lo = r_sum + VDW_BAND_LO_OFFSET_A
    hi = r_sum + VDW_BAND_HI_OFFSET_A

    pa = atom_present[:, :, None, :, None]
    pb = atom_present[:, None, :, None, :]
    valid = pa & pb                                                           # (B, N, N, 14, 14)

    # Per-(a,b) violation: how far outside the band.
    viol = (lo - d).clamp_min(0) + (d - hi).clamp_min(0)
    # Per-(a,b) margin: how far inside the band (positive ⇒ in band).
    margin = torch.minimum(d - lo, hi - d).clamp_min(0)

    # Where invalid, neutralise: GT=1 reduction is min, so set viol = +inf;
    # GT=0 reduction is max-of-margin, so set margin = 0.
    INF = torch.tensor(1e6, device=device, dtype=d.dtype)
    viol = torch.where(valid, viol, INF.expand_as(viol))
    margin = torch.where(valid, margin, torch.zeros_like(margin))

    pair_loss_pos = viol.amin(dim=(-1, -2))                                   # GT=1 (B, N, N)
    any_valid = valid.any(dim=-1).any(dim=-1)
    pair_loss_pos = torch.where(any_valid, pair_loss_pos, torch.zeros_like(pair_loss_pos))
    pair_loss_neg = margin.amax(dim=(-1, -2))                                 # GT=0 (B, N, N)

    target = target_vdw.to(d.dtype)
    pair_loss = target * pair_loss_pos + (1.0 - target) * pair_loss_neg

    # Denominator counts only VDW-evaluable pairs (≥1 sidechain–sidechain atom
    # pair present). Pairs without sidechain atoms (e.g. GLY–GLY) contribute 0
    # to the numerator and are excluded from the average. Examples with no
    # VDW-evaluable pairs are dropped from the batch mean entirely.
    pair_real = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)
    not_diag = ~torch.eye(N, device=device, dtype=torch.bool)
    applicable = any_valid & pair_real & not_diag                              # (B, N, N)
    applicable_f = applicable.to(pair_loss.dtype)
    denom = applicable_f.sum(dim=(-2, -1))                                     # (B,)
    per_example = (pair_loss * applicable_f).sum(dim=(-2, -1)) / denom.clamp_min(1.0)
    has_signal = (denom > 0).to(pair_loss.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)


# ----------------------------------------------------------------------
# H-bond loss.

def hbond_interaction_loss(
    coords_atom14_ang: torch.Tensor,    # in Å — slot 3 (O) placed via psi torsion
    residue_type: torch.Tensor,
    atom_mask: torch.Tensor,
    target_hbond: torch.Tensor,
    padding_mask: torch.Tensor,         # (B, N) bool, True = real residue
) -> torch.Tensor:
    """Per-example mean over off-diagonal real-residue pairs of the h-bond
    loss, then mean over the batch. Symmetrized: each pair considers (donor in
    i, acceptor in j) ∪ (donor in j, acceptor in i). Slot 3 (O) is taken from
    the input coords as placed by the model's predicted psi torsion (see
    `apply_torsions_to_atom14`)."""
    B, N = residue_type.shape
    device = coords_atom14_ang.device
    atom_present = (atom_mask == 1)

    donor_table = HBOND_DONORS_ATOM14.to(device)[residue_type]
    donor_typed = HBOND_DONORS_MASK.to(device)[residue_type]
    accept_table = HBOND_ACCEPTORS_ATOM14.to(device)[residue_type]
    accept_typed = HBOND_ACCEPTORS_MASK.to(device)[residue_type]

    D_slots, X_slots = donor_table[..., 0], donor_table[..., 1]
    A_slots, Y_slots = accept_table[..., 0], accept_table[..., 1]

    D = _gather_atom14(coords_atom14_ang, D_slots)
    X = _gather_atom14(coords_atom14_ang, X_slots)
    A = _gather_atom14(coords_atom14_ang, A_slots)
    Y = _gather_atom14(coords_atom14_ang, Y_slots)

    donor_valid = donor_typed & atom_present.gather(-1, D_slots) & atom_present.gather(-1, X_slots)
    accept_valid = accept_typed & atom_present.gather(-1, A_slots) & atom_present.gather(-1, Y_slots)

    Di = D[:, :, None, :, None, :]                              # (B, N, 1, max_d, 1, 3)
    Xi = X[:, :, None, :, None, :]
    Aj = A[:, None, :, None, :, :]
    Yj = Y[:, None, :, None, :, :]
    DA = Aj - Di
    XD = Xi - Di
    AD = -DA
    YA = Yj - Aj

    d_DA = DA.norm(dim=-1)                                       # (B, N, N, max_d, max_a)
    cos_xda = (XD * DA).sum(-1) / (_safe_norm(XD).squeeze(-1) * _safe_norm(DA).squeeze(-1))
    cos_day = (AD * YA).sum(-1) / (_safe_norm(AD).squeeze(-1) * _safe_norm(YA).squeeze(-1))

    viol_dist = (HBOND_DIST_LO_A - d_DA).clamp_min(0) + (d_DA - HBOND_DIST_HI_A).clamp_min(0)
    viol_xda = (cos_xda - HBOND_COS_XDA_THRESH).clamp_min(0)
    viol_day = (cos_day - HBOND_COS_DAY_THRESH).clamp_min(0)
    per_pair_viol = viol_dist + viol_xda + viol_day

    margin_dist = torch.minimum(d_DA - HBOND_DIST_LO_A, HBOND_DIST_HI_A - d_DA)
    margin_xda = HBOND_COS_XDA_THRESH - cos_xda
    margin_day = HBOND_COS_DAY_THRESH - cos_day
    per_pair_margin = torch.minimum(
        torch.minimum(margin_dist, margin_xda), margin_day,
    ).clamp_min(0)

    atom_pair_valid = donor_valid[:, :, None, :, None] & accept_valid[:, None, :, None, :]

    INF = torch.tensor(1e6, device=device, dtype=d_DA.dtype)
    per_pair_viol_ij = torch.where(atom_pair_valid, per_pair_viol, INF.expand_as(per_pair_viol))
    per_pair_margin_ij = torch.where(atom_pair_valid, per_pair_margin, torch.zeros_like(per_pair_margin))

    viol_ij = per_pair_viol_ij.amin(dim=(-1, -2))                # (B, N, N) — i donates to j
    margin_ij = per_pair_margin_ij.amax(dim=(-1, -2))
    valid_ij = atom_pair_valid.any(dim=-1).any(dim=-1)
    viol_ij = torch.where(valid_ij, viol_ij, torch.zeros_like(viol_ij))

    # Symmetrize: best donor-acceptor configuration in either direction.
    viol_sym = torch.minimum(viol_ij, viol_ij.transpose(-1, -2))
    margin_sym = torch.maximum(margin_ij, margin_ij.transpose(-1, -2))

    target = target_hbond.to(d_DA.dtype)
    pair_loss = target * viol_sym + (1.0 - target) * margin_sym

    # Denominator counts only h-bond-evaluable pairs (≥1 donor-acceptor combo
    # present in either i→j or j→i direction). Pairs without compatible atoms
    # contribute 0 to the numerator and are excluded from the average.
    pair_real = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)
    not_diag = ~torch.eye(N, device=device, dtype=torch.bool)
    valid_sym = valid_ij | valid_ij.transpose(-1, -2)
    applicable = valid_sym & pair_real & not_diag
    applicable_f = applicable.to(pair_loss.dtype)
    denom = applicable_f.sum(dim=(-2, -1))
    per_example = (pair_loss * applicable_f).sum(dim=(-2, -1)) / denom.clamp_min(1.0)
    has_signal = (denom > 0).to(pair_loss.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)


# ----------------------------------------------------------------------
# Aromatic loss.

def aromatic_subtype_losses(
    coords_atom14_ang: torch.Tensor,
    residue_type: torch.Tensor,
    atom_mask: torch.Tensor,
    target_aromatic_subtypes: torch.Tensor,   # (B, N, N, 3) — channels [pd, sandwich, t_shaped]
    padding_mask: torch.Tensor,               # (B, N) bool, True = real residue
) -> dict[str, torch.Tensor]:
    """Three independent two-sided flat-bottomed losses, one per π-stacking
    sub-type. Geometry is computed once and reused. Each sub-type's GT slice
    independently drives its own band penalty: GT=1 → linear violation outside
    the sub-type's band, GT=0 → linear margin inside it. Returns a dict with
    keys 'parallel_displaced', 'sandwich', 't_shaped', each a scalar — per-
    example mean over off-diagonal real-residue pairs, then mean over batch."""
    B, N = residue_type.shape
    device = coords_atom14_ang.device
    atom_present = (atom_mask == 1)

    ring_slots = AROMATIC_RING_SLOTS.to(device)[residue_type]                 # (B, N, 6)
    ring_typed = AROMATIC_ATOM_MASK.to(device)[residue_type]
    is_arom = IS_AROMATIC.to(device)[residue_type]                            # (B, N)

    ring_pos = _gather_atom14(coords_atom14_ang, ring_slots)
    eff_mask = ring_typed & atom_present.gather(-1, ring_slots)
    arom_valid = is_arom & eff_mask[..., 0] & eff_mask[..., 1] & eff_mask[..., 2]

    centroid, normal = _ring_centroid_normal(ring_pos, eff_mask)

    c_i = centroid[:, :, None, :]
    c_j = centroid[:, None, :, :]
    n_i = normal[:, :, None, :]
    n_j = normal[:, None, :, :]
    r12 = c_j - c_i
    d = r12.norm(dim=-1)
    n_dot = (n_i * n_j).sum(-1, keepdim=True)
    parallel = n_dot.squeeze(-1).abs()
    # Sign-align n_j with n_i before averaging so anti-parallel ring pairs
    # (a common π-stacking orientation) don't collapse n_avg to zero.
    sign = torch.sign(n_dot)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n_avg = _safe_normalize((n_i + sign * n_j) / 2)
    d_perp = (r12 * n_avg).sum(-1).abs()
    d_par = (d.pow(2) - d_perp.pow(2)).clamp_min(1e-8).sqrt()

    # Sandwich: parallel ≥ 0.85, d ∈ [3.0, 4.5], d_par ≤ 1.5
    viol_sw = (
        (AROM_PARA_LO - parallel).clamp_min(0)
        + (AROM_SANDWICH_D[0] - d).clamp_min(0)
        + (d - AROM_SANDWICH_D[1]).clamp_min(0)
        + (d_par - AROM_SANDWICH_DPAR_HI).clamp_min(0)
    )
    margin_sw = torch.minimum(
        torch.minimum(parallel - AROM_PARA_LO, d - AROM_SANDWICH_D[0]),
        torch.minimum(AROM_SANDWICH_D[1] - d, AROM_SANDWICH_DPAR_HI - d_par),
    ).clamp_min(0)

    # Parallel-displaced: parallel ≥ 0.85, d ∈ [3.5, 6.5], d_par ∈ [1.5, 3.5]
    viol_pd = (
        (AROM_PARA_LO - parallel).clamp_min(0)
        + (AROM_PD_D[0] - d).clamp_min(0)
        + (d - AROM_PD_D[1]).clamp_min(0)
        + (AROM_PD_DPAR[0] - d_par).clamp_min(0)
        + (d_par - AROM_PD_DPAR[1]).clamp_min(0)
    )
    margin_pd = torch.minimum(
        torch.minimum(parallel - AROM_PARA_LO, d - AROM_PD_D[0]),
        torch.minimum(
            AROM_PD_D[1] - d,
            torch.minimum(d_par - AROM_PD_DPAR[0], AROM_PD_DPAR[1] - d_par),
        ),
    ).clamp_min(0)

    # T-shaped: parallel ≤ 0.4, d ∈ [4.5, 7.0]. (No d_par constraint.)
    viol_t = (
        (parallel - AROM_PARA_HI).clamp_min(0)
        + (AROM_T_D[0] - d).clamp_min(0)
        + (d - AROM_T_D[1]).clamp_min(0)
    )
    margin_t = torch.minimum(
        torch.minimum(AROM_PARA_HI - parallel, d - AROM_T_D[0]),
        AROM_T_D[1] - d,
    ).clamp_min(0)

    # Denominator counts only aromatic-aromatic real off-diagonal pairs.
    # Non-aromatic pairs contribute 0 to the numerator (arom_f gate below) and
    # are excluded from the average. Examples with no aromatic-aromatic pairs
    # are dropped from the batch mean entirely.
    pair_arom = arom_valid[:, :, None] & arom_valid[:, None, :]               # (B, N, N)
    pair_real = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)
    not_diag = ~torch.eye(N, device=device, dtype=torch.bool)
    applicable = pair_arom & pair_real & not_diag
    applicable_f = applicable.to(d.dtype)
    denom_per_example = applicable_f.sum(dim=(-2, -1))                        # (B,)
    has_signal = (denom_per_example > 0).to(d.dtype)
    denom_safe = denom_per_example.clamp_min(1.0)
    n_active = has_signal.sum().clamp_min(1.0)

    arom_f = pair_arom.to(d.dtype)
    viol_pd = viol_pd * arom_f
    viol_sw = viol_sw * arom_f
    viol_t = viol_t * arom_f
    margin_pd = margin_pd * arom_f
    margin_sw = margin_sw * arom_f
    margin_t = margin_t * arom_f

    def _reduce(viol: torch.Tensor, margin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.to(d.dtype)
        pair_loss = t * viol + (1.0 - t) * margin
        per_example = (pair_loss * applicable_f).sum(dim=(-2, -1)) / denom_safe
        return (per_example * has_signal).sum() / n_active

    return {
        "parallel_displaced": _reduce(viol_pd, margin_pd, target_aromatic_subtypes[..., 0]),
        "sandwich": _reduce(viol_sw, margin_sw, target_aromatic_subtypes[..., 1]),
        "t_shaped": _reduce(viol_t, margin_t, target_aromatic_subtypes[..., 2]),
    }


# ----------------------------------------------------------------------
# Top-level helper: compute all three losses.

def interaction_geometry_losses(
    coords_atom14: torch.Tensor,        # (B, N, 14, 3) in DATASET units
    residue_type: torch.Tensor,
    atom_mask: torch.Tensor,
    target_im: torch.Tensor,            # (B, N, N, 6) — [vdw, hbond, pd, sandwich, t_shaped, none]
    padding_mask: torch.Tensor,         # (B, N) bool, True = real residue
) -> dict[str, torch.Tensor]:
    """Compute all geometric losses. `coords_atom14` is in dataset units
    (Å / COORD_SCALE_ANGSTROMS). The returned dict has keys 'vdw', 'hbond',
    'parallel_displaced', 'sandwich', 't_shaped', each a scalar tensor.
    Padded residues (padding_mask == False) are excluded from each loss's
    pair-level reduction."""
    from twistr.ml.constants import COORD_SCALE_ANGSTROMS

    coords_ang = coords_atom14 * COORD_SCALE_ANGSTROMS
    # Slot 3 (O) is placed by the model's predicted psi via the AF2 psi rigid
    # group (see twistr/ml/models/sidechain.py); pass `coords_ang` directly
    # to the h-bond loss with no peptide-O reconstruction needed.
    arom = aromatic_subtype_losses(
        coords_ang, residue_type, atom_mask, target_im[..., 2:5], padding_mask,
    )
    return {
        "vdw": vdw_interaction_loss(coords_ang, residue_type, atom_mask, target_im[..., 0], padding_mask),
        "hbond": hbond_interaction_loss(coords_ang, residue_type, atom_mask, target_im[..., 1], padding_mask),
        "parallel_displaced": arom["parallel_displaced"],
        "sandwich": arom["sandwich"],
        "t_shaped": arom["t_shaped"],
    }
