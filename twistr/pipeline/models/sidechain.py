"""Place predicted full-atom (atom14) coordinates from the model's per-residue
output (R, t, chi_sincos) using AF2's canonical chi=0 reference geometry.

Algorithm (matches AlphaFold 2's all_atom convention):
  1. Each residue has a canonical "atom14 layout at chi=0" — atom positions
     in the residue's backbone frame with all chi dihedrals set to zero.
     Stored as `ATOM14_LOCAL_CHI0` (20, 14, 3). Computed deterministically at
     module load by composing AF2's `restype_rigid_group_default_frame` with
     `restype_atom14_rigid_group_positions` — same arithmetic as AF2's
     `_make_rigid_group_constants` in `alphafold/common/residue_constants.py`.
  2. Apply chi rotations: for chi_i (i = 1..4), rotate every atom whose
     `ATOM14_CHI_DEPTH` is ≥ i around the chi_i axis bond. The chi_i axis is
     the middle bond of the chi_i dihedral (atoms 1 and 2 of the chi tuple,
     in the same orientation as AF2's chi_1-frame x-axis).
  3. Apply the predicted backbone frame (R, t).

`ATOM14_CHI_DEPTH` is derived from AF2's group indices:
  group 0 (backbone) and group 3 (psi/O) → depth 0
  group 4 (chi_1) → depth 1, group 5 → 2, group 6 → 3, group 7 → 4
At AF2's chi=0, an atom in chi_i's group has its dihedral wrt the previous
chi axis set to zero by the default-frame construction. Because we don't
predict psi, slot 3 (O) carries AF2's psi=0 reference position; callers
needing physical O (e.g., the h-bond loss) reconstruct it from the next
residue's N via `_compute_backbone_o_from_peptide`.

AF2 source: `twistr/external/alphafold/alphafold/common/residue_constants.py`
(submodule of google-deepmind/alphafold). Constants are extracted via AST
(same pattern as our Protenix `_CHI_ANGLES_ATOMS` extraction) so we never
copy AF2 values into this repo and never import the AF2 module (which pulls
in jax).

KNOWN FOLLOW-UP: chi π-periodicity is not handled here. AF2 lists the
2-fold-symmetric chis in `chi_pi_periodic` (ASP χ₂, GLU χ₃, PHE χ₂,
TYR χ₂, HIS χ₂ — and ARG's NH1↔NH2 atom-level swap). Our current losses
(VDW, h-bond, aromatic) all reduce by min/max over atom pairs, donor-
acceptor pairs, or sub-types, so they are naturally invariant to the
swap and don't need π-periodicity handling. If we ever add a direct chi
regression loss or a per-atom RMSD/FAPE loss, this needs to be added —
build the swap table from `chi_pi_periodic` + AF2 atom equivalences and
take min(loss(GT), loss(swapped GT))."""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from twistr.pipeline.features.chi_angles import CHI_ATOM14_INDICES
from twistr.tensors.constants import (
    ATOM14_SLOT_INDEX,
    RESIDUE_TYPE_NAMES,
    _ATOM14,
)


# ----------------------------------------------------------------------
# AST-extract AF2 constants.

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


# residue_3-letter → list of [atom_name, group_idx, (x, y, z)]; positions are
# in the local frame of `group_idx`. group 0 = backbone, 3 = psi, 4-7 = chi_1..chi_4.
_AF2_RIGID_GROUP_ATOM_POSITIONS: dict = _extract_af2_assign("rigid_group_atom_positions")
# residue_3-letter → list of 4-tuples of atom names (one per chi).
_AF2_CHI_ANGLES_ATOMS: dict = _extract_af2_assign("chi_angles_atoms")
# 20 lists of 4 floats, one per residue type in AF2 restype order.
_AF2_CHI_ANGLES_MASK: list = _extract_af2_assign("chi_angles_mask")
# 20-letter string of one-letter codes in AF2's restype order.
_AF2_RESTYPES: list = _extract_af2_assign("restypes")
# one-letter → three-letter dict.
_AF2_RESTYPE_1TO3: dict = _extract_af2_assign("restype_1to3")


# ----------------------------------------------------------------------
# Compose default frames + atom local positions to get backbone-frame atom14
# positions at chi=0.

def _make_rigid_transformation_4x4(
    ex: torch.Tensor, ey: torch.Tensor, translation: torch.Tensor,
) -> torch.Tensor:
    """Direct port of AF2's `_make_rigid_transformation_4x4` (residue_constants.py)."""
    ex_n = ex / ex.norm().clamp_min(1e-12)
    ey_n = ey - (ey @ ex_n) * ex_n
    ey_n = ey_n / ey_n.norm().clamp_min(1e-12)
    ez_n = torch.linalg.cross(ex_n, ey_n, dim=-1)
    m = torch.eye(4, dtype=torch.float64)
    m[:3, 0] = ex_n
    m[:3, 1] = ey_n
    m[:3, 2] = ez_n
    m[:3, 3] = translation
    return m


def _build_residue_default_frames(
    atom_positions: dict[str, torch.Tensor],
    chi_atoms: list[tuple[str, ...]],
    chi_mask: list[float],
) -> torch.Tensor:
    """(8, 4, 4) per-group default frames (relative to the parent group), in
    AF2 convention. Identity for group 0 (backbone) and group 1 (pre-omega).
    See AF2 `_make_rigid_group_constants`."""
    frames = torch.eye(4, dtype=torch.float64).repeat(8, 1, 1)
    frames[2] = _make_rigid_transformation_4x4(
        ex=atom_positions["N"] - atom_positions["CA"],
        ey=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        translation=atom_positions["N"],
    )
    frames[3] = _make_rigid_transformation_4x4(
        ex=atom_positions["C"] - atom_positions["CA"],
        ey=atom_positions["CA"] - atom_positions["N"],
        translation=atom_positions["C"],
    )
    if chi_mask[0]:
        bp = [atom_positions[name] for name in chi_atoms[0]]
        frames[4] = _make_rigid_transformation_4x4(
            ex=bp[2] - bp[1], ey=bp[0] - bp[1], translation=bp[2],
        )
    for chi_idx in range(1, 4):
        if chi_mask[chi_idx]:
            axis_end = atom_positions[chi_atoms[chi_idx][2]]
            frames[4 + chi_idx] = _make_rigid_transformation_4x4(
                ex=axis_end,
                ey=torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
                translation=axis_end,
            )
    return frames


def _compose_group_to_backbone(default_frames: torch.Tensor) -> torch.Tensor:
    """For each of 8 groups, return the 4x4 transform that maps group-local
    coords directly to backbone-local coords (i.e., compose chi-chain frames).
    Groups 0, 1, 2, 3 are already group→backbone; groups 4..7 are
    group→parent and must be composed with ancestors."""
    out = default_frames.clone()
    out[5] = default_frames[4] @ default_frames[5]
    out[6] = out[5] @ default_frames[6]
    out[7] = out[6] @ default_frames[7]
    return out


def _build_atom14_layout_and_groups() -> tuple[torch.Tensor, torch.Tensor]:
    """(20, 14, 3) all-torsion-zero atom14 layout in backbone frame and
    (20, 14) per-atom rigid-group index. Group encoding (matches AF2):
      0 backbone, 1 pre-omega, 2 phi, 3 psi, 4..7 chi_1..chi_4.
    For atom14 (heavy atoms only), groups 1 and 2 are unpopulated."""
    layout = torch.zeros((20, 14, 3), dtype=torch.float32)
    group = torch.zeros((20, 14), dtype=torch.long)

    af2_restype_to_three = {one: _AF2_RESTYPE_1TO3[one] for one in _AF2_RESTYPES}

    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        try:
            af2_idx = next(
                i for i, one in enumerate(_AF2_RESTYPES)
                if af2_restype_to_three[one] == res_name
            )
        except StopIteration as exc:
            raise RuntimeError(f"AF2 doesn't list residue type {res_name}") from exc

        chi_mask = _AF2_CHI_ANGLES_MASK[af2_idx]
        chi_atoms = _AF2_CHI_ANGLES_ATOMS[res_name]
        af2_atoms = _AF2_RIGID_GROUP_ATOM_POSITIONS[res_name]

        atom_positions = {
            name: torch.tensor(pos, dtype=torch.float64)
            for name, _, pos in af2_atoms
        }
        default_frames = _build_residue_default_frames(atom_positions, chi_atoms, chi_mask)
        composed = _compose_group_to_backbone(default_frames)

        for atom_name, group_idx, local_pos in af2_atoms:
            if atom_name not in ATOM14_SLOT_INDEX[res_name]:
                continue
            slot = ATOM14_SLOT_INDEX[res_name][atom_name]
            local_h = torch.tensor(list(local_pos) + [1.0], dtype=torch.float64)
            backbone_h = composed[group_idx] @ local_h
            layout[res_idx, slot] = backbone_h[:3].to(torch.float32)
            group[res_idx, slot] = group_idx

    return layout, group


def _group_to_chi_depth(group: torch.Tensor) -> torch.Tensor:
    """Convert per-atom group index → chi depth: groups 0/1/2/3 (backbone /
    pre-omega / phi / psi) → depth 0; groups 4..7 (chi_1..chi_4) → depth 1..4."""
    return torch.where(group >= 4, group - 3, torch.zeros_like(group))


def _build_atom14_used_mask() -> torch.Tensor:
    """(20, 14) bool: True for atom14 slots populated by each residue type."""
    mask = torch.zeros((20, 14), dtype=torch.bool)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        mask[res_idx, : len(_ATOM14[res_name])] = True
    return mask


# Module-load: deterministic, no dataset I/O.
_LAYOUT_ANG, ATOM14_GROUP_IDX = _build_atom14_layout_and_groups()
ATOM14_CHI_DEPTH = _group_to_chi_depth(ATOM14_GROUP_IDX)
# Cast layout to dataset units (Å / COORD_SCALE_ANGSTROMS) so it matches the
# model's translation output and matches the dataset's coordinate scale.
from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS  # noqa: E402
ATOM14_LOCAL_CHI0 = _LAYOUT_ANG / COORD_SCALE_ANGSTROMS
ATOM14_USED_MASK: torch.Tensor = _build_atom14_used_mask()

# Index map into the model's (B, N, 7, 2) `torsion_sincos` tensor.
# Slot order matches AF2's restype_rigid_group_default_frame group ordering.
TORSION_OMEGA = 0
TORSION_PHI = 1
TORSION_PSI = 2
TORSION_CHI1 = 3
TORSION_CHI2 = 4
TORSION_CHI3 = 5
TORSION_CHI4 = 6


# ----------------------------------------------------------------------
# Forward kinematics: place atom14 from (R, t, torsion_sincos).

def _rotate_atoms(
    local: torch.Tensor,        # (B, N, 14, 3)
    pivot: torch.Tensor,        # (B, N, 3)
    axis_dir: torch.Tensor,     # (B, N, 3) unit
    sin_a: torch.Tensor,        # (B, N)
    cos_a: torch.Tensor,        # (B, N)
    rotate_mask: torch.Tensor,  # (B, N, 14) bool
) -> torch.Tensor:
    """Apply Rodrigues rotation to atoms where `rotate_mask` is True; leave
    others unchanged."""
    rel = local - pivot.unsqueeze(-2)                                 # (B, N, 14, 3)
    sin_b = sin_a.unsqueeze(-1).unsqueeze(-1)
    cos_b = cos_a.unsqueeze(-1).unsqueeze(-1)
    cross = torch.linalg.cross(axis_dir.unsqueeze(-2).expand_as(rel), rel, dim=-1)
    dot = (axis_dir.unsqueeze(-2) * rel).sum(dim=-1, keepdim=True)
    rotated = rel * cos_b + cross * sin_b + axis_dir.unsqueeze(-2) * dot * (1 - cos_b)
    rotated = rotated + pivot.unsqueeze(-2)
    return torch.where(rotate_mask.unsqueeze(-1), rotated, local)


def apply_torsions_to_atom14(
    R: torch.Tensor,             # (B, N, 3, 3) backbone rotation
    t: torch.Tensor,             # (B, N, 3) backbone translation, dataset units
    torsion_sincos: torch.Tensor,# (B, N, 7, 2) — omega, phi, psi, chi_1..chi_4
    residue_type: torch.Tensor,  # (B, N) long
) -> torch.Tensor:
    """Place predicted atom14 coordinates in dataset units. Output shape
    (B, N, 14, 3). Applies psi (rotates O around the CA→C axis via the AF2
    psi rigid group) and chi_1..chi_4 (rotates sidechain atoms around their
    respective chi axes). Phi and omega torsions have no atom14 atoms in
    AF2's heavy-atom layout (only hydrogens, which we don't model) so they
    do not affect placement.

    Slots beyond each residue's atom14 length are populated but should be
    ignored downstream via atom_mask / ATOM14_USED_MASK."""
    device = R.device
    layout = ATOM14_LOCAL_CHI0.to(device=device, dtype=R.dtype)
    group_idx = ATOM14_GROUP_IDX.to(device=device)
    chi_indices = CHI_ATOM14_INDICES.to(device=device)

    B, N = residue_type.shape
    local = layout[residue_type]                              # (B, N, 14, 3)
    chi_idx_per_res = chi_indices[residue_type]                # (B, N, 4, 4)
    group_per_res = group_idx[residue_type]                    # (B, N, 14)

    bi = torch.arange(B, device=device)[:, None].expand(B, N)
    ni = torch.arange(N, device=device)[None, :].expand(B, N)

    # Psi rotation: rotates atoms in group 3 (just O in atom14) around the
    # CA→C axis at pivot C, matching AF2's default psi-frame x-axis.
    psi_pivot = local[bi, ni, 2]                              # C (slot 2)
    psi_axis_vec = local[bi, ni, 2] - local[bi, ni, 1]         # C - CA
    psi_axis_dir = psi_axis_vec / psi_axis_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    psi_mask = (group_per_res == 3)
    local = _rotate_atoms(
        local, psi_pivot, psi_axis_dir,
        torsion_sincos[..., TORSION_PSI, 0],
        torsion_sincos[..., TORSION_PSI, 1],
        psi_mask,
    )

    # Chi rotations: chi_i rotates atoms in groups (4 + i_zero_indexed) and
    # downstream around the chi-tuple a1→a2 bond.
    for chi_i in range(4):
        slot_a1 = chi_idx_per_res[..., chi_i, 1]
        slot_a2 = chi_idx_per_res[..., chi_i, 2]
        pivot = local[bi, ni, slot_a1]
        axis_end = local[bi, ni, slot_a2]
        axis_vec = axis_end - pivot
        axis_dir = axis_vec / axis_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotate_mask = (group_per_res >= 4 + chi_i)
        local = _rotate_atoms(
            local, pivot, axis_dir,
            torsion_sincos[..., TORSION_CHI1 + chi_i, 0],
            torsion_sincos[..., TORSION_CHI1 + chi_i, 1],
            rotate_mask,
        )

    global_pos = torch.einsum("...ij,...kj->...ki", R, local) + t.unsqueeze(-2)
    return global_pos


