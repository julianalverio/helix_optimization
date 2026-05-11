"""Per-atom14 MSE loss in Å². Used as a structural anchor: predicted atom14
positions are compared against GT atom14 positions in the dataset frame
(no Kabsch — predictions and GT live in the same frame because both are
subject to the same per-example random rotation in `ExamplesDataset`).

The caller passes a per-atom-slot inclusion mask so different residue
classes can be supervised on different atom subsets — e.g. one call
covering the antigen backbone only, another covering helix all-atoms plus
non-helix interface sidechains. See `lightning_module._compute_losses`
for the two masks the training code constructs."""
from __future__ import annotations

import torch

from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS


def coord_mse_loss(
    pred_atoms_atom14: torch.Tensor,    # (B, N, 14, 3) in dataset units (Å / 10)
    gt_atoms_atom14: torch.Tensor,      # (B, N, 14, 3) in dataset units
    atom_mask: torch.Tensor,            # (B, N, 14) int8 in {-1, 0, 1}; only ==1 counts
    atom_inclusion: torch.Tensor,       # (B, N, 14) bool — atoms to include in this loss
) -> torch.Tensor:
    """Mean-over-batch of (mean-over-residues of (mean-over-valid-atoms of Å²)).
    `atom_inclusion` selects atoms per (residue, slot) so the caller can
    apply different rules to backbone vs sidechain or to different residue
    classes. The (atom_mask == 1) gate is applied internally so callers
    don't need to mix in atom presence. Each residue with ≥1 valid atom
    contributes equally regardless of how many slots it contributes; each
    example contributes equally regardless of length; examples with no
    valid atoms anywhere in their inclusion are dropped from the batch
    mean."""
    diff_ang = (pred_atoms_atom14 - gt_atoms_atom14) * COORD_SCALE_ANGSTROMS
    sq = diff_ang.pow(2).sum(dim=-1)                                            # (B, N, 14) Å² per atom

    valid_atom = (atom_mask == 1) & atom_inclusion                              # (B, N, 14)
    valid_atom_f = valid_atom.to(sq.dtype)
    atoms_per_res = valid_atom_f.sum(dim=-1)                                    # (B, N)
    per_residue = (sq * valid_atom_f).sum(dim=-1) / atoms_per_res.clamp_min(1.0)

    res_has_signal = (atoms_per_res > 0).to(sq.dtype)
    res_count = res_has_signal.sum(dim=-1)                                      # (B,)
    per_example = (per_residue * res_has_signal).sum(dim=-1) / res_count.clamp_min(1.0)
    has_signal = (res_count > 0).to(sq.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
