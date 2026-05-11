"""Hydrophobic-stub & aromatic-ring packing loss.

For each interface residue's packing atoms (aliphatic stub Cs ∪ aromatic
ring atoms — see `PACKING_ATOMS` in `features/interaction_matrix.py`), count
heavy-atom neighbors lying in the VDW band [d_lo, d_hi] Å on other residues
with supervised positions. Penalty is one-sided `relu(n_target - count)`
per packing atom: encourages tight interface packing, never penalises
over-packing (`clash_loss` handles the lower bound).

Neighbor scope (Option III — atoms with supervised positions):
  - any atom on a helix residue,
  - antigen backbone (atom14 slots 0-3 of any non-helix real residue),
  - antigen interface-residue sidechain (slots 4-13 of non-helix interface).

Non-interface antigen sidechain atoms are excluded so the model can't fake
packing against drifted, unsupervised sidechain positions.
"""
from __future__ import annotations

import torch

from twistr.pipeline.features.interaction_matrix import IS_PACKING_ATOM, _band


def packing_neighbor_loss(
    coords_atom14_ang: torch.Tensor,        # (B, N, 14, 3) in Å
    residue_type: torch.Tensor,             # (B, N) long
    atom_mask: torch.Tensor,                # (B, N, 14) int8 in {-1, 0, 1}
    is_interface_residue: torch.Tensor,     # (B, N) bool
    is_helix: torch.Tensor,                 # (B, N) bool
    padding_mask: torch.Tensor,             # (B, N) bool, True = real residue
    n_target: float = 4.0,
    d_lo: float = 3.3,
    d_hi: float = 5.5,
    tau: float = 0.3,
) -> torch.Tensor:
    """Per-atom mean → per-example mean over examples with ≥1 from-atom →
    batch mean. Returns a scalar tensor."""
    device = coords_atom14_ang.device
    B, N = residue_type.shape

    atom_present = (atom_mask == 1)                                            # (B, N, 14)
    real = padding_mask.bool()                                                  # (B, N)
    interface = is_interface_residue.bool()
    helix = is_helix.bool()

    # FROM atoms: packing-mask atoms on interface, real residues, present.
    is_packing = IS_PACKING_ATOM.to(device)[residue_type]                       # (B, N, 14)
    from_atom_mask = (
        is_packing
        & atom_present
        & interface.unsqueeze(-1)
        & real.unsqueeze(-1)
    )                                                                            # (B, N, 14)

    # TO atoms (Option III). Equivalently: real & present & (residue is helix
    # OR residue is interface OR slot is backbone). Antigen non-interface
    # sidechain atoms are excluded.
    sidechain_slot = torch.zeros(14, dtype=torch.bool, device=device)
    sidechain_slot[4:] = True
    backbone_slot = ~sidechain_slot                                              # (14,)
    res_supervised = (helix | interface) & real                                  # (B, N)
    # Per-(res, slot): supervised residue allows any slot; non-supervised residue
    # allows only backbone slots (still subject to `real`).
    neighbor_atom_mask = atom_present & real.unsqueeze(-1) & (
        res_supervised.unsqueeze(-1) | backbone_slot.view(1, 1, 14)
    )                                                                            # (B, N, 14)

    # Pairwise heavy-atom distances. (B, N_from, N_to, 14_from, 14_to)
    p_i = coords_atom14_ang[:, :, None, :, None, :]                             # (B, N, 1, 14, 1, 3)
    p_j = coords_atom14_ang[:, None, :, None, :, :]                             # (B, 1, N, 1, 14, 3)
    d = (p_i - p_j).norm(dim=-1)                                                # (B, N, N, 14, 14)

    # Smooth band indicator: product of two sigmoids centred at d_lo and d_hi.
    in_band = _band(d, d_lo, d_hi, k=tau)                                       # (B, N, N, 14, 14)

    # Exclude self-residue pairs (i == j) — a residue's stub atoms shouldn't
    # count its own atoms as neighbors.
    not_self = ~torch.eye(N, device=device, dtype=torch.bool)                   # (N, N)
    pair_mask = (
        from_atom_mask[:, :, None, :, None]                                     # (B, N, 1, 14, 1)
        & neighbor_atom_mask[:, None, :, None, :]                               # (B, 1, N, 1, 14)
        & not_self[None, :, :, None, None]                                      # (1, N, N, 1, 1)
    )
    in_band = in_band * pair_mask.to(in_band.dtype)

    # Soft neighbor count per (b, i, a). Sum over j (dim 2) and to-slot (dim 4).
    count = in_band.sum(dim=(2, 4))                                              # (B, N, 14)
    loss_per_atom = torch.relu(n_target - count)                                 # (B, N, 14)

    # Per-example mean over from-atoms, batch mean over examples that have
    # ≥1 from-atom (examples with no interface packing atoms contribute 0).
    from_atom_f = from_atom_mask.to(loss_per_atom.dtype)
    per_example_sum = (loss_per_atom * from_atom_f).sum(dim=(-2, -1))            # (B,)
    per_example_denom = from_atom_f.sum(dim=(-2, -1))                            # (B,)
    per_example = per_example_sum / per_example_denom.clamp_min(1.0)
    has_signal = (per_example_denom > 0).to(per_example.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
