"""Cross-validate the chi-to-atom14 placement: applying it with ground-truth
backbone frames and ground-truth chi angles must reconstruct the ground-truth
atom14 coordinates. This catches sign-convention bugs in the chi rotation,
mistakes in the chi-depth table, and frame-composition errors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from twistr.ml.constants import COORD_SCALE_ANGSTROMS
from twistr.ml.features.chi_angles import chi_sincos as encode_sincos, compute_chi_angles
from twistr.ml.models.rotation import frame_from_three_points
from twistr.ml.models.sidechain import (
    ATOM14_CHI_DEPTH,
    ATOM14_USED_MASK,
    apply_torsions_to_atom14,
)
from twistr.pipeline.tensors.constants import RESIDUE_TYPE_NAMES

EXAMPLE_NPZ = Path("data/module3/examples/br/1brs_1_0.npz")
# The layout uses AF2's canonical (idealized) bond geometry. Real residues
# differ from idealized by ~0.05 Å per bond; errors compound along the chi
# chain. We assert depth-stratified bounds so that gross bugs (sign-flipped
# chi rotation, wrong axis, table corruption — all of which produce >2 Å
# errors at depth 1) are caught even though depth-4 atoms have intrinsic
# ~1.3 Å slack from real-vs-AF2 geometry.
DEPTH_TOLERANCES = {
    0: 0.04,   # backbone (N/CA/C/CB) — tight: 0.4 Å
    1: 0.07,   # chi_1 atoms — 0.7 Å
    2: 0.12,   # chi_2 — 1.2 Å (rings, branched)
    3: 0.10,   # chi_3 — 1.0 Å
    4: 0.15,   # chi_4 — 1.5 Å (ARG NH1/NH2, LYS NZ)
}


def _load_dataset_units(path: Path):
    data = np.load(path)
    coords = torch.from_numpy(data["coordinates"].astype(np.float32)) / COORD_SCALE_ANGSTROMS
    residue_type = torch.from_numpy(data["residue_type"]).long()
    atom_mask = torch.from_numpy(data["atom_mask"])
    return coords, residue_type, atom_mask


def _mask_out_degenerate_in_source(coords: torch.Tensor, used: torch.Tensor) -> torch.Tensor:
    """Float16 storage collapses some atom positions onto each other (e.g., LEU
    CD1 = CD2 in residues where they're already <0.05 Å apart). Round-trip
    can't recover information that wasn't in the source — exclude such atoms."""
    n_res, n_slot = used.shape
    degenerate = torch.zeros_like(used, dtype=torch.bool)
    for i in range(n_res):
        n_atoms = int(used[i].sum())
        if n_atoms < 2:
            continue
        pos = coords[i, :n_atoms]
        pairwise = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        eye = torch.eye(n_atoms, dtype=torch.bool)
        too_close = (pairwise < 0.05) & ~eye
        degenerate[i, :n_atoms] = too_close.any(dim=-1)
    return degenerate


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_round_trip_recovers_gt_atom14():
    coords, residue_type, atom_mask = _load_dataset_units(EXAMPLE_NPZ)

    R = frame_from_three_points(coords[:, 0], coords[:, 1], coords[:, 2])
    t = coords[:, 1]
    angles, _ = compute_chi_angles(coords[None], residue_type[None], atom_mask[None])
    chi_sc = encode_sincos(angles[0])

    # Build full 7-torsion sincos: [omega, phi, psi, chi_1..chi_4].
    # For the round-trip test we use measured chis and identity
    # omega/phi/psi (cos=1, sin=0). Slot 3 (O) is psi-dependent so will be
    # placed at the AF2 psi=0 reference position; the round-trip test
    # already excludes slot 3 for this reason.
    n_res = chi_sc.shape[0]
    torsions = torch.zeros(n_res, 7, 2)
    torsions[..., 1] = 1.0  # cos = 1 → angle = 0 for omega/phi/psi
    torsions[:, 3:7] = chi_sc

    placed = apply_torsions_to_atom14(R[None], t[None], torsions[None], residue_type[None])[0]

    used = ATOM14_USED_MASK[residue_type]
    present = (atom_mask == 1)
    degenerate = _mask_out_degenerate_in_source(coords, used)
    # Exclude slot 3 (carbonyl O) — its position is set by psi (the next-residue
    # backbone dihedral), not by the residue's own chi angles, so this module
    # cannot place it accurately. Backbone h-bond losses must use a different
    # source for O (e.g., derive from N(i+1)) or skip O entirely.
    is_o = torch.zeros(14, dtype=torch.bool)
    is_o[3] = True
    check = used & present & ~degenerate & ~is_o.unsqueeze(0)

    diff = (placed - coords).norm(dim=-1)
    depth = ATOM14_CHI_DEPTH[residue_type]
    assert (check & (depth == 0)).sum() > 50, "expected backbone atoms in test"
    for d, bound in DEPTH_TOLERANCES.items():
        mask_d = check & (depth == d)
        if mask_d.sum() == 0:
            continue
        max_err = diff[mask_d].max().item()
        assert max_err < bound, (
            f"depth {d} max round-trip error: {max_err:.4e} dataset units "
            f"({max_err * 10:.4f} Å), bound {bound} ({bound * 10:.2f} Å) — "
            f"chi rotation or chi-depth table is wrong"
        )


def test_chi_depth_consistency_with_chi_tuples():
    """chi-tuple atom 3 (the 4th, 0-indexed atom) is the first atom defined by
    chi_i — its chi depth must equal i. This catches typos in the depth table."""
    from twistr.pipeline.tensors.constants import ATOM14_SLOT_INDEX
    from twistr.ml.features.chi_angles import _CHI_ANGLES_ATOMS

    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for chi_idx, atoms in enumerate(_CHI_ANGLES_ATOMS[res_name]):
            slot = ATOM14_SLOT_INDEX[res_name][atoms[3]]
            depth = ATOM14_CHI_DEPTH[res_idx, slot].item()
            assert depth == chi_idx + 1, (
                f"{res_name} atom {atoms[3]} (slot {slot}) is the chi_{chi_idx + 1} "
                f"atom, expected depth {chi_idx + 1}, got {depth}"
            )


def test_chi_depth_zero_for_backbone_atoms():
    """N, CA, C, O are always backbone (depth 0)."""
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for slot in (0, 1, 2, 3):
            assert ATOM14_CHI_DEPTH[res_idx, slot].item() == 0
        # CB (slot 4) is depth 0 for all non-GLY residues.
        if res_name != "GLY":
            assert ATOM14_CHI_DEPTH[res_idx, 4].item() == 0


def test_chi_depth_does_not_exceed_chi_count():
    """No residue's atom can have depth > number of chi angles defined."""
    from twistr.ml.features.chi_angles import _CHI_ANGLES_ATOMS

    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        n_chis = len(_CHI_ANGLES_ATOMS[res_name])
        max_depth = ATOM14_CHI_DEPTH[res_idx].max().item()
        assert max_depth <= n_chis, (
            f"{res_name} has {n_chis} chis but max depth = {max_depth}"
        )
