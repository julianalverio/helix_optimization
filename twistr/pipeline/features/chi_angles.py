from __future__ import annotations

import ast
from pathlib import Path

import torch

from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

# The chi-defining atom-name tuples are the source of truth and we don't want
# to type them by hand. We extract them from Protenix's source via AST at
# module load — same effect as a direct import, but without pulling in
# rdkit/torch/etc. that Protenix's constants module triggers on real import.
_PROTENIX_CONSTANTS = (
    Path(__file__).resolve().parents[2]
    / "external" / "Protenix" / "protenix" / "data" / "constants.py"
)


def _extract_protenix_dict(name: str):
    if not _PROTENIX_CONSTANTS.exists():
        raise RuntimeError(
            f"Protenix submodule not found at {_PROTENIX_CONSTANTS}. "
            "Run: git submodule update --init --recursive"
        )
    tree = ast.parse(_PROTENIX_CONSTANTS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {_PROTENIX_CONSTANTS}")


# Sourced from Protenix protenix/data/constants.py:_CHI_ANGLES_ATOMS.
# Maps amino-acid 3-letter name → list of 0..4 tuples of 4 atom names. Each
# tuple defines one chi dihedral via the four atoms it spans.
_CHI_ANGLES_ATOMS: dict = _extract_protenix_dict("_CHI_ANGLES_ATOMS")


def _build_chi_atom14_indices() -> torch.Tensor:
    """Translate _CHI_ANGLES_ATOMS atom-name tuples into atom14 slot indices
    using our ATOM14_SLOT_INDEX. Result shape (20, 4, 4) long: for each
    residue type and chi index, the four atom14 slots that define the chi.
    Slots for non-existent chis are zero — masked out by chi_mask."""
    table = torch.zeros((20, 4, 4), dtype=torch.long)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for chi_idx, atom_names in enumerate(_CHI_ANGLES_ATOMS[res_name]):
            for atom_idx, atom_name in enumerate(atom_names):
                table[res_idx, chi_idx, atom_idx] = ATOM14_SLOT_INDEX[res_name][atom_name]
    return table


def _build_chi_mask_table() -> torch.Tensor:
    """Per-residue chi validity, derived from the same Protenix table:
    chi_i is defined for a residue iff i < len(_CHI_ANGLES_ATOMS[res])."""
    mask = torch.zeros((20, 4), dtype=torch.bool)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        n = len(_CHI_ANGLES_ATOMS[res_name])
        if n > 0:
            mask[res_idx, :n] = True
    return mask


CHI_ATOM14_INDICES: torch.Tensor = _build_chi_atom14_indices()
CHI_ANGLES_MASK_TABLE: torch.Tensor = _build_chi_mask_table()


def chi_mask(residue_type: torch.Tensor) -> torch.Tensor:
    """Per-residue chi-angle validity mask. Input shape (..., N), output shape
    (..., N, 4) of bool. mask[..., i] is True iff chi_(i+1) is defined for
    that residue's amino-acid type. Does NOT account for missing atoms in the
    structure — combine with atom presence in compute_chi_angles for that."""
    return CHI_ANGLES_MASK_TABLE.to(device=residue_type.device)[residue_type]


def atan2_dihedral(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
) -> torch.Tensor:
    """Standard atan2-based dihedral angle around the p2-p3 axis. All inputs
    shape (..., 3); output shape (...). Convention matches gemmi's
    calculate_dihedral and AF2/OpenFold."""
    a = p2 - p1
    b = p3 - p2
    c = p4 - p3
    u = torch.linalg.cross(a, b, dim=-1)
    v = torch.linalg.cross(b, c, dim=-1)
    b_unit = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y = (torch.linalg.cross(u, v, dim=-1) * b_unit).sum(dim=-1)
    x = (u * v).sum(dim=-1)
    return torch.atan2(y, x)


def compute_chi_angles(
    coordinates: torch.Tensor,
    residue_type: torch.Tensor,
    atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the four chi dihedral angles per residue from atom14 coordinates.

    Inputs:
      coordinates  (B, N, 14, 3) float
      residue_type (B, N) long, values in [0, 19]
      atom_mask    (B, N, 14) int8 with values in {-1, 0, 1}; only `==1` counts as present

    Outputs:
      angles   (B, N, 4) float, radians in (-π, π]. Zero where validity is False.
      validity (B, N, 4) bool. True iff chi exists for the residue type AND
               all four defining atoms have atom_mask == 1.
    """
    B, N = residue_type.shape
    indices = CHI_ATOM14_INDICES.to(coordinates.device)[residue_type]      # (B, N, 4, 4)

    # Gather the 4 chi-atom positions per residue per chi via advanced indexing.
    bi = torch.arange(B, device=coordinates.device)[:, None, None, None].expand(B, N, 4, 4)
    ni = torch.arange(N, device=coordinates.device)[None, :, None, None].expand(B, N, 4, 4)
    points = coordinates[bi, ni, indices]                                  # (B, N, 4, 4, 3)

    angles = atan2_dihedral(
        points[..., 0, :], points[..., 1, :], points[..., 2, :], points[..., 3, :],
    )                                                                       # (B, N, 4)

    presence = (atom_mask[bi, ni, indices] == 1)                           # (B, N, 4, 4)
    atoms_all_present = presence.all(dim=-1)                               # (B, N, 4)
    type_mask = chi_mask(residue_type)                                     # (B, N, 4)
    validity = type_mask & atoms_all_present
    angles = torch.where(validity, angles, torch.zeros_like(angles))
    return angles, validity


def chi_sincos(angles: torch.Tensor) -> torch.Tensor:
    """(sin, cos) encoding of chi angles. Input shape (..., 4); output (..., 4, 2).
    Periodic angles map to the unit circle, eliminating the [0, 2π) wraparound."""
    return torch.stack([angles.sin(), angles.cos()], dim=-1)
