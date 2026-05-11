"""AF2-style steric clash loss.

Reference: Jumper et al. 2021, AlphaFold supplementary §1.9.11, Algorithm 28
``between_residue_clash_loss``. Reference implementations consulted:
  - DeepMind AlphaFold:        alphafold/model/folding.py
  - OpenFold:                  openfold/utils/loss.py
  - FAFE (vendored):           twistr/external/FAFE/losses/violation.py:86

Each non-bonded inter-residue atom pair contributes
``ReLU(r_a + r_b - tolerance - d_ab)`` to the loss; the AF2 default
``overlap_tolerance = 1.5 Å`` is used here. Reduction is per-example
``sum / count(valid_pairs)``, then mean over the batch — every example
contributes equally regardless of length. Excluded pairs:
  - same residue
  - missing atoms (atom14 atom_mask != 1)
  - the C(i)-N(i+1) peptide bond, in both directions of the symmetric tensor
  - the SG-SG slot for any CYS-CYS pair (covers all disulfides)

VdW radii come from ``VDW_RADII`` in twistr/pipeline/features/interaction_matrix.py,
which builds an atom14 table from element-keyed values sourced from
Protenix's ``rdkit_van_der_waals_radius`` (twistr/external/Protenix/
protenix/data/constants.py:215). Atom14 slot conventions are read from
``ATOM14_SLOT_INDEX`` in twistr/pipeline/tensors/constants.py."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from twistr.pipeline.features.interaction_matrix import VDW_RADII
from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

_CYS_INDEX = RESIDUE_TYPE_NAMES.index("CYS")
_CYS_SG_SLOT = ATOM14_SLOT_INDEX["CYS"]["SG"]
_BACKBONE_N_SLOT = ATOM14_SLOT_INDEX["ALA"]["N"]
_BACKBONE_C_SLOT = ATOM14_SLOT_INDEX["ALA"]["C"]


def steric_clash_loss(
    coords_atom14_ang: torch.Tensor,    # (B, N, 14, 3) in Å
    atom_mask: torch.Tensor,            # (B, N, 14) int8 in {-1, 0, 1}; only `==1` slots count
    residue_type: torch.Tensor,         # (B, N) long
    overlap_tolerance: float = 1.5,
) -> torch.Tensor:
    B, N = residue_type.shape
    device = coords_atom14_ang.device

    atom_r = VDW_RADII.to(device)[residue_type]                              # (B, N, 14)

    diffs = (
        coords_atom14_ang[:, :, None, :, None, :]
        - coords_atom14_ang[:, None, :, None, :, :]
    )
    dists = (diffs.pow(2).sum(-1) + 1e-10).sqrt()                            # (B, N, N, 14, 14)

    lower = atom_r[:, :, None, :, None] + atom_r[:, None, :, None, :]
    viol = F.relu(lower - overlap_tolerance - dists)

    atom_present = atom_mask == 1
    pair_present = (
        atom_present[:, :, None, :, None]
        & atom_present[:, None, :, None, :]
    )

    res_idx = torch.arange(N, device=device)
    diff_res = (res_idx[:, None] != res_idx[None, :])[None, :, :, None, None]

    # Adjacent residues along the backbone (i, i+1).
    i_idx = torch.arange(N - 1, device=device)
    j_idx = i_idx + 1
    bond_pair = torch.zeros(N, N, 14, 14, dtype=torch.bool, device=device)
    bond_pair[i_idx, j_idx, _BACKBONE_C_SLOT, _BACKBONE_N_SLOT] = True
    bond_pair[j_idx, i_idx, _BACKBONE_N_SLOT, _BACKBONE_C_SLOT] = True
    bond_pair = bond_pair.unsqueeze(0)

    is_cys = residue_type == _CYS_INDEX
    cys_pair = is_cys[:, :, None] & is_cys[:, None, :]                       # (B, N, N)
    cys_excl = torch.zeros(B, N, N, 14, 14, dtype=torch.bool, device=device)
    cys_excl[..., _CYS_SG_SLOT, _CYS_SG_SLOT] = cys_pair

    valid = pair_present & diff_res & ~bond_pair & ~cys_excl
    mask_f = valid.to(viol.dtype)
    denom = mask_f.sum(dim=(-4, -3, -2, -1))                                    # (B,)
    per_example = (viol * mask_f).sum(dim=(-4, -3, -2, -1)) / denom.clamp_min(1.0)
    has_signal = (denom > 0).to(viol.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
