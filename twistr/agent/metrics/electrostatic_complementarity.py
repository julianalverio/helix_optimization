"""Electrostatic complementarity across the interface.

McCoy et al. (1997) defined EC as the (anti-)correlation of Poisson-
Boltzmann electrostatic potentials computed on opposite sides of the
interaction surface: a well-complementary interface presents one side's
potential as the negative of the other's at matched surface points, so
the linear correlation between the two potentials is near -1. Computing
PB potentials requires APBS or equivalent and is heavy for an inner-loop
scorer.

We use a residue-charge Coulomb-sum proxy that captures the same
asymmetry-favouring signal at negligible compute cost. Identify charged
residues on each side (D / E = -1; K / R = +1; H = +0.5 at physiological
pH), then sum -q_i * q_j / d_ij over cross-interface charged residue
pairs within `distance_cutoff`. Pairs of opposite charges contribute
positively (favourable complementarity), like pairs contribute
negatively. The raw sum is passed through tanh to map to [-1, +1] for
direct comparability with the SC metric.

Returns NaN if neither side has any interface charged residue (the
metric is undefined for purely-hydrophobic interfaces).
"""
from __future__ import annotations

import math

import torch

from twistr.tensors.constants import RESIDUE_TYPE_INDEX

_CB_SLOT = 4
_CA_SLOT = 1

_CHARGE = {
    "ASP": -1.0, "GLU": -1.0,
    "LYS": +1.0, "ARG": +1.0,
    "HIS": +0.5,
}
_CHARGE_BY_INDEX = torch.zeros(20, dtype=torch.float32)
for resname, q in _CHARGE.items():
    _CHARGE_BY_INDEX[RESIDUE_TYPE_INDEX[resname]] = q


def electrostatic_complementarity(
    atoms_atom14_ang: torch.Tensor,         # (1, N, 14, 3) Å
    atom_mask: torch.Tensor,                # (1, N, 14) int8
    residue_type: torch.Tensor,             # (1, N) long
    is_helix: torch.Tensor,                 # (1, N) bool
    is_interface_residue: torch.Tensor,     # (1, N) bool
    distance_cutoff: float = 12.0,
) -> float:
    coords = atoms_atom14_ang[0]
    mask = atom_mask[0]
    rtype = residue_type[0]
    helix = is_helix[0]
    iface = is_interface_residue[0]

    charge_table = _CHARGE_BY_INDEX.to(rtype.device)
    charges = charge_table[rtype]                           # (N,)

    # Use Cβ where present (sidechain anchor); fall back to Cα for
    # glycine and any residue missing its Cβ.
    pos = torch.where(
        (mask[..., _CB_SLOT] == 1).unsqueeze(-1),
        coords[..., _CB_SLOT, :],
        coords[..., _CA_SLOT, :],
    )                                                       # (N, 3)
    valid_pos = (mask[..., _CA_SLOT] == 1)

    is_charged = (charges != 0.0) & valid_pos
    helix_charged = helix & iface & is_charged
    target_charged = (~helix) & iface & is_charged

    if not helix_charged.any() or not target_charged.any():
        return float("nan")

    q_h = charges[helix_charged]                            # (Nh,)
    p_h = pos[helix_charged]                                # (Nh, 3)
    q_t = charges[target_charged]
    p_t = pos[target_charged]

    d = torch.cdist(p_h, p_t).clamp_min(1e-2)               # (Nh, Nt) Å
    mask_within = d <= distance_cutoff
    qq = q_h.unsqueeze(1) * q_t.unsqueeze(0)                # (Nh, Nt)

    # Favourable (opposite-charge) pairs have qq < 0; we want positive EC
    # for those. Coulomb energy ~ +qq/d (positive for like, negative for
    # unlike); EC = -energy = -qq/d.
    contributions = (-qq / d) * mask_within.to(qq.dtype)
    raw = contributions.sum().item()

    # Normalize by the number of charged pairs that could have
    # contributed, so EC is intensive rather than extensive.
    n_pairs = int(mask_within.sum().item())
    if n_pairs == 0:
        return 0.0
    normalized = raw / n_pairs
    return float(math.tanh(normalized * 5.0))
