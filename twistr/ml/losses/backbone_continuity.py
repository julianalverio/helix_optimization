"""AF2-style between-residue bond-geometry loss.

Penalises any deviation of the peptide-bond geometry between residue `i`
and residue `i+1` from the empirical PDB reference values, using a
flat-bottom relu — see Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44–45.
The model emits an independent (R, t) per residue, so the chain can
split at arbitrary places without anything else in the loss set noticing.

The three geometric metrics tracked:

  1. ‖C(i) – N(i+1)‖             vs `between_res_bond_length_c_n`
                                   (1.329 Å non-Pro, 1.341 Å when i+1 = PRO)
  2. cos∠CA(i)–C(i)–N(i+1)        vs `between_res_cos_angles_ca_c_n`
                                   (–0.4473, σ=0.0311)
  3. cos∠C(i)–N(i+1)–CA(i+1)      vs `between_res_cos_angles_c_n_ca`
                                   (–0.5203, σ=0.0353)

Reference values are AST-extracted from the AlphaFold submodule
(`twistr/external/alphafold/alphafold/common/residue_constants.py`,
lines 516–521) — the same pattern `twistr/ml/models/sidechain.py:65-77`
uses for the rigid-group constants. We deliberately don't import AF2
(it pulls in JAX) and don't copy values into this repo (so they stay in
sync with the submodule)."""
from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from twistr.pipeline.tensors.constants import RESIDUE_TYPE_INDEX

_AF2_RESIDUE_CONSTANTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "external" / "alphafold" / "alphafold" / "common" / "residue_constants.py"
)


def _extract_af2_assign(name: str):
    if not _AF2_RESIDUE_CONSTANTS_PATH.exists():
        raise RuntimeError(
            f"AlphaFold submodule not found at {_AF2_RESIDUE_CONSTANTS_PATH}. "
            "Run: git submodule update --init --recursive"
        )
    tree = ast.parse(_AF2_RESIDUE_CONSTANTS_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {_AF2_RESIDUE_CONSTANTS_PATH}")


# [non-PRO, PRO] — the C–N peptide bond into proline is shorter due to the ring.
_C_N_BOND_LENGTH: list = _extract_af2_assign("between_res_bond_length_c_n")
_C_N_BOND_LENGTH_STDDEV: list = _extract_af2_assign("between_res_bond_length_stddev_c_n")
# (cos, stddev) pairs.
_CA_C_N_COS: list = _extract_af2_assign("between_res_cos_angles_ca_c_n")
_C_N_CA_COS: list = _extract_af2_assign("between_res_cos_angles_c_n_ca")

_PRO_INDEX = RESIDUE_TYPE_INDEX["PRO"]

# atom14 layout — N=0, CA=1, C=2 — is fixed across all residue types
# (see twistr/pipeline/tensors/constants.py:_ATOM14).
_N_SLOT, _CA_SLOT, _C_SLOT = 0, 1, 2

_EPS = 1e-6


def _flat_bottom(
    metric: torch.Tensor,           # (B, N-1)
    gt,
    gt_stddev,
    mask: torch.Tensor,             # (B, N-1)
    tolerance: float,
) -> torch.Tensor:
    """relu(|metric − gt| − tolerance · stddev), per-example mean over
    masked entries, then mean over examples that had ≥1 valid pair —
    matches the reduction pattern used by `steric_clash_loss` and
    `helix_dihedral_loss` so each example contributes equally regardless
    of length. `gt` and `gt_stddev` may be Python scalars or per-pair
    tensors. The sqrt-with-eps form of |·| is the AF2 convention; keeps
    the gradient finite at metric == gt exactly."""
    error = torch.sqrt(_EPS + (metric - gt).pow(2))
    per_pair = F.relu(error - tolerance * gt_stddev)
    pair_count = mask.sum(dim=-1)
    per_example = (per_pair * mask).sum(dim=-1) / pair_count.clamp_min(1.0)
    has_signal = (pair_count > 0).to(per_example.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)


def backbone_continuity_loss(
    atoms_atom14_ang: torch.Tensor,    # (B, N, 14, 3) IN ANGSTROMS
    atom_mask: torch.Tensor,           # (B, N, 14) int8 in {-1, 0, 1}
    residue_type: torch.Tensor,        # (B, N) — RESIDUE_TYPE_NAMES order
    chain_slot: torch.Tensor,          # (B, N) — same chain ⇒ peptide-bond candidate
    is_helix: torch.Tensor,            # (B, N) bool — helix-only gate
    padding_mask: torch.Tensor,        # (B, N) bool, True = real residue
    tolerance: float = 12.0,
) -> torch.Tensor:
    """Sum of c_n_loss + ca_c_n_loss + c_n_ca_loss; each averaged
    independently over its own masked-pair set so an example with all
    three sets empty (pure chain breaks / pads) returns 0 cleanly. The
    non-Pro / Pro reference is selected per pair from `residue_type[i+1]`.

    **Helix-only gate.** The peptide-bond geometry only makes sense
    between residues that are actually adjacent in the original
    structure. Our antigen comes in as a *crop* — fragments of the
    surrounding partner with arbitrary chain breaks at the crop
    boundaries — so two antigen residues that look adjacent in the N
    axis may have an arbitrary gap in the original protein. Penalising
    "C–N too long" on those would emit meaningless gradients. The helix
    is contiguous, so we restrict the loss to helix-helix adjacent
    pairs only. (`same_chain` is also kept as defence-in-depth, even
    though `both_helix` is strictly stronger when the helix is one
    chain.)"""
    dtype = atoms_atom14_ang.dtype

    # (B, N-1, 3) for each side of the bond.
    this_ca = atoms_atom14_ang[..., :-1, _CA_SLOT, :]
    this_c  = atoms_atom14_ang[..., :-1, _C_SLOT,  :]
    next_n  = atoms_atom14_ang[..., 1:,  _N_SLOT,  :]
    next_ca = atoms_atom14_ang[..., 1:,  _CA_SLOT, :]

    # atom_mask == 1 → real heavy atom present at this slot.
    am1 = (atom_mask == 1).to(dtype)
    this_ca_m = am1[..., :-1, _CA_SLOT]
    this_c_m  = am1[..., :-1, _C_SLOT]
    next_n_m  = am1[..., 1:,  _N_SLOT]
    next_ca_m = am1[..., 1:,  _CA_SLOT]

    # Pair must be: both real residues, same chain, AND both helix.
    real_pair  = padding_mask[..., :-1] & padding_mask[..., 1:]
    same_chain = chain_slot[..., :-1] == chain_slot[..., 1:]
    both_helix = is_helix.bool()[..., :-1] & is_helix.bool()[..., 1:]
    no_gap = (real_pair & same_chain & both_helix).to(dtype)

    # 1) C–N bond length. Pro/non-Pro reference selected by the i+1 residue.
    next_is_pro = (residue_type[..., 1:] == _PRO_INDEX).to(dtype)
    cn_gt = (1.0 - next_is_pro) * _C_N_BOND_LENGTH[0] + next_is_pro * _C_N_BOND_LENGTH[1]
    cn_sd = (1.0 - next_is_pro) * _C_N_BOND_LENGTH_STDDEV[0] + next_is_pro * _C_N_BOND_LENGTH_STDDEV[1]

    cn_len = torch.sqrt(_EPS + (this_c - next_n).pow(2).sum(-1))
    cn_loss = _flat_bottom(
        cn_len, cn_gt, cn_sd,
        this_c_m * next_n_m * no_gap, tolerance,
    )

    # 2) & 3) Bond-angle cosines via unit vectors at the bond.
    ca_c_len = torch.sqrt(_EPS + (this_ca - this_c).pow(2).sum(-1))
    n_ca_len = torch.sqrt(_EPS + (next_n - next_ca).pow(2).sum(-1))
    c_ca_u = (this_ca - this_c) / ca_c_len.unsqueeze(-1)
    c_n_u  = (next_n - this_c) / cn_len.unsqueeze(-1)
    n_ca_u = (next_ca - next_n) / n_ca_len.unsqueeze(-1)

    cancn_loss = _flat_bottom(
        (c_ca_u * c_n_u).sum(-1),
        _CA_C_N_COS[0],
        _CA_C_N_COS[1],
        this_ca_m * this_c_m * next_n_m * no_gap,
        tolerance,
    )

    cnca_loss = _flat_bottom(
        ((-c_n_u) * n_ca_u).sum(-1),
        _C_N_CA_COS[0],
        _C_N_CA_COS[1],
        this_c_m * next_n_m * next_ca_m * no_gap,
        tolerance,
    )

    return cn_loss + cancn_loss + cnca_loss
