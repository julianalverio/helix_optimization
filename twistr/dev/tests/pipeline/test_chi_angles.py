"""Cross-validate our chi-angle computation against gemmi (independent C++
dihedral implementation). If our implementation has a sign-convention bug,
gather mistake, atom-index mistake, or formula bug, the values will disagree
on a real residue and this test will catch it."""
from __future__ import annotations

import math
from pathlib import Path

import gemmi
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.features.chi_angles import (
    CHI_ATOM14_INDICES,
    _CHI_ANGLES_ATOMS,
    chi_mask,
    compute_chi_angles,
)
from twistr.tensors.constants import (
    ATOM14_SLOT_INDEX,
    RESIDUE_TYPE_NAMES,
)

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")
ANGULAR_TOL = 1e-4  # radians


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular distance between a and b, in (-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def test_chi_atom14_indices_arg():
    """Spot-check that index translation is sane for ARG."""
    arg_idx = RESIDUE_TYPE_NAMES.index("ARG")
    # ARG chi1: (N, CA, CB, CG) → atom14 slots (0, 1, 4, 5)
    assert CHI_ATOM14_INDICES[arg_idx, 0].tolist() == [0, 1, 4, 5]
    # ARG chi2: (CA, CB, CG, CD) → (1, 4, 5, 6)
    assert CHI_ATOM14_INDICES[arg_idx, 1].tolist() == [1, 4, 5, 6]
    # ARG chi3: (CB, CG, CD, NE) → (4, 5, 6, 7)
    assert CHI_ATOM14_INDICES[arg_idx, 2].tolist() == [4, 5, 6, 7]
    # ARG chi4: (CG, CD, NE, CZ) → (5, 6, 7, 8)
    assert CHI_ATOM14_INDICES[arg_idx, 3].tolist() == [5, 6, 7, 8]


def test_chi_mask_ala_gly_zero_lys_full():
    rtype = torch.tensor(
        [RESIDUE_TYPE_NAMES.index(r) for r in ("ALA", "GLY", "LYS", "VAL", "SER")],
        dtype=torch.long,
    )
    m = chi_mask(rtype)
    assert m[0].tolist() == [False, False, False, False]   # ALA
    assert m[1].tolist() == [False, False, False, False]   # GLY
    assert m[2].tolist() == [True, True, True, True]       # LYS
    assert m[3].tolist() == [True, False, False, False]    # VAL (1 chi)
    assert m[4].tolist() == [True, False, False, False]    # SER (1 chi)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_chi_matches_gemmi_on_real_data():
    """Compute chi for every (residue, chi) where validity is True via two
    independent paths — ours and gemmi.calculate_dihedral — and require
    agreement to <1e-4 rad. This exercises:
      - the vendored _CHI_ANGLES_ATOMS dict (any name typo → angle disagrees)
      - the atom-name → atom14 slot translation (any layout drift → disagrees)
      - the dihedral formula and sign convention
      - the gather / advanced-indexing logic"""
    data = np.load(EXAMPLE_NPZ)
    coords_np = data["coordinates"].astype(np.float32)
    residue_type_np = data["residue_type"]
    atom_mask_np = data["atom_mask"]

    coords = torch.from_numpy(coords_np)
    residue_type = torch.from_numpy(residue_type_np).long()
    atom_mask = torch.from_numpy(atom_mask_np)

    angles_ours, validity_ours = compute_chi_angles(
        coords[None], residue_type[None], atom_mask[None],
    )
    angles_ours = angles_ours[0]      # (N, 4)
    validity_ours = validity_ours[0]  # (N, 4)

    n_checked = 0
    for i in range(int(residue_type.shape[0])):
        rtype = int(residue_type[i])
        if not (0 <= rtype < 20):
            continue
        res_name = RESIDUE_TYPE_NAMES[rtype]
        for chi_idx, atom_names in enumerate(_CHI_ANGLES_ATOMS[res_name]):
            if not bool(validity_ours[i, chi_idx]):
                continue
            slots = [ATOM14_SLOT_INDEX[res_name][n] for n in atom_names]
            pts = [gemmi.Position(*coords_np[i, s].tolist()) for s in slots]
            chi_gemmi = gemmi.calculate_dihedral(*pts)
            chi_ours = float(angles_ours[i, chi_idx])
            diff = _angle_diff(chi_ours, chi_gemmi)
            assert diff < ANGULAR_TOL, (
                f"{res_name} residue {i} chi{chi_idx + 1}: "
                f"ours={chi_ours:.6f} gemmi={chi_gemmi:.6f} diff={diff:.6e}"
            )
            n_checked += 1

    # Sanity: we actually exercised some chis.
    assert n_checked >= 5, f"only {n_checked} chis checked — test data may be wrong"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_chi_validity_ala_gly_always_false():
    data = np.load(EXAMPLE_NPZ)
    coords = torch.from_numpy(data["coordinates"].astype(np.float32))
    residue_type = torch.from_numpy(data["residue_type"]).long()
    atom_mask = torch.from_numpy(data["atom_mask"])

    _, validity = compute_chi_angles(coords[None], residue_type[None], atom_mask[None])
    validity = validity[0]
    ala_idx = RESIDUE_TYPE_NAMES.index("ALA")
    gly_idx = RESIDUE_TYPE_NAMES.index("GLY")
    is_alagly = (residue_type == ala_idx) | (residue_type == gly_idx)
    if is_alagly.any():
        assert not validity[is_alagly].any()


def test_sincos_round_trip():
    """sin/cos encoding → atan2 reconstruction matches the original angle."""
    angles = torch.tensor([-3.0, -1.5, 0.0, 0.5, 1.5, 3.0])
    from twistr.pipeline.features.chi_angles import chi_sincos
    sc = chi_sincos(angles.unsqueeze(-1).expand(-1, 4))  # (6, 4, 2)
    recovered = torch.atan2(sc[..., 0], sc[..., 1])  # (6, 4)
    assert torch.allclose(recovered, angles.unsqueeze(-1).expand(-1, 4), atol=1e-6)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_conditioning_chi_gating():
    """The conditioning wrappers must (a) zero out non-conditioned residues
    and (b) pass the un-gated computation through unchanged for conditioned
    residues. conditioning_mask = ~is_helix, so helix residues are zeroed
    and partner residues are passed through."""
    from twistr.pipeline.features.chi_angles import chi_sincos
    from twistr.pipeline.features.conditioning import (
        conditioning_chi_sincos,
        conditioning_chi_validity,
    )

    data = np.load(EXAMPLE_NPZ)
    coords = torch.from_numpy(data["coordinates"].astype(np.float32))
    residue_type = torch.from_numpy(data["residue_type"]).long()
    atom_mask = torch.from_numpy(data["atom_mask"])
    is_helix = torch.from_numpy(np.ascontiguousarray(data["is_helix"]))

    batch = {
        "coordinates": coords[None],
        "residue_type": residue_type[None],
        "atom_mask": atom_mask[None],
        "is_helix": is_helix[None],
        "padding_mask": torch.ones(1, is_helix.shape[0], dtype=torch.bool),
    }

    angles, validity = compute_chi_angles(
        batch["coordinates"], batch["residue_type"], batch["atom_mask"],
    )
    expected_sincos = chi_sincos(angles)
    gated_sincos = conditioning_chi_sincos(batch)
    gated_validity = conditioning_chi_validity(batch)

    helix = is_helix.bool()
    partner = ~helix
    assert helix.any() and partner.any()

    # Helix residues (non-conditioned): everything zeroed.
    assert (gated_sincos[0, helix] == 0).all()
    assert not gated_validity[0, helix].any()

    # Partner residues (conditioned): pass-through, with invalid chis still
    # zeroed (matches the wrapper's combined keep = validity & cond).
    partner_validity = validity[0, partner]
    expected_partner_sincos = torch.where(
        partner_validity.unsqueeze(-1),
        expected_sincos[0, partner],
        torch.zeros_like(expected_sincos[0, partner]),
    )
    assert torch.equal(gated_sincos[0, partner], expected_partner_sincos)
    assert torch.equal(gated_validity[0, partner], partner_validity)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_conditioning_translation_and_frame_validity_on_real_data():
    """Validity flags are True only for partner residues with the relevant
    backbone atoms present. On 1BRS all backbone heavy atoms are present, so
    validity should match `~is_helix` exactly."""
    from twistr.pipeline.features.conditioning import (
        conditioning_frame_6d,
        conditioning_frame_6d_validity,
        conditioning_translation,
        conditioning_translation_validity,
    )

    data = np.load(EXAMPLE_NPZ)
    coords = torch.from_numpy(data["coordinates"].astype(np.float32))
    atom_mask = torch.from_numpy(data["atom_mask"])
    is_helix = torch.from_numpy(np.ascontiguousarray(data["is_helix"]))
    batch = {
        "coordinates": coords[None],
        "residue_type": torch.from_numpy(data["residue_type"]).long()[None],
        "atom_mask": atom_mask[None],
        "is_helix": is_helix[None],
        "padding_mask": torch.ones(1, is_helix.shape[0], dtype=torch.bool),
    }

    partner = ~is_helix.bool()
    t_valid = conditioning_translation_validity(batch)[0]
    f_valid = conditioning_frame_6d_validity(batch)[0]
    assert torch.equal(t_valid, partner)
    assert torch.equal(f_valid, partner)
    # Translation/frame zeros where invalid; non-zero where valid.
    t = conditioning_translation(batch)[0]
    f = conditioning_frame_6d(batch)[0]
    assert (t[~partner] == 0).all()
    assert (f[~partner] == 0).all()
    assert (t[partner].abs().sum(dim=-1) > 0).any()
    assert (f[partner].abs().sum(dim=-1) > 0).any()


def test_conditioning_translation_and_frame_validity_handles_missing_atoms():
    """Synthesize a batch with a partner residue that has CA missing and
    another with N missing. Translation validity drops only when CA is
    missing; frame validity drops when ANY of N/CA/C is missing. The
    corresponding output cells are zeroed."""
    from twistr.pipeline.features.conditioning import (
        conditioning_frame_6d,
        conditioning_frame_6d_validity,
        conditioning_translation,
        conditioning_translation_validity,
    )

    coords = torch.randn(1, 4, 14, 3)                                            # 4 partner residues
    atom_mask = torch.ones(1, 4, 14, dtype=torch.int8)
    atom_mask[0, 1, 1] = 0                                                       # residue 1: CA missing
    atom_mask[0, 2, 0] = 0                                                       # residue 2: N missing
    is_helix = torch.zeros(1, 4, dtype=torch.bool)                               # all partners
    padding_mask = torch.ones(1, 4, dtype=torch.bool)
    batch = {"coordinates": coords, "atom_mask": atom_mask, "is_helix": is_helix, "padding_mask": padding_mask}

    t_valid = conditioning_translation_validity(batch)[0]
    f_valid = conditioning_frame_6d_validity(batch)[0]

    assert t_valid.tolist() == [True, False, True, True]                         # CA-missing → False
    assert f_valid.tolist() == [True, False, False, True]                        # CA OR N missing → False

    t = conditioning_translation(batch)[0]
    f = conditioning_frame_6d(batch)[0]
    assert (t[1] == 0).all()
    assert (f[1] == 0).all() and (f[2] == 0).all()
