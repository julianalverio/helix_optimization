"""Tests for the AF2-style backbone chain-continuity loss."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from twistr.ml.losses.backbone_continuity import (
    _CA_C_N_COS,
    _C_N_BOND_LENGTH,
    _C_N_BOND_LENGTH_STDDEV,
    _C_N_CA_COS,
    _PRO_INDEX,
    backbone_continuity_loss,
)
from twistr.pipeline.tensors.constants import RESIDUE_TYPE_NAMES

ALA = RESIDUE_TYPE_NAMES.index("ALA")
PRO = RESIDUE_TYPE_NAMES.index("PRO")
N_SLOT, CA_SLOT, C_SLOT = 0, 1, 2

# Canonical backbone lengths used for placement (Å). Magnitudes don't affect
# any of the angle losses (they're about cosines of bond directions); only the
# C–N bond length is scored against a reference.
_CA_C_LEN = 1.52
_N_CA_LEN = 1.46


def _two_residue_batch(
    cn_bond_length: float,
    next_is_pro: bool = False,
    same_chain: bool = True,
    pad_residue_1: bool = False,
    both_helix: bool = True,
):
    """Build a (B=1, N=2) batch with the C/CA of res 0 and N/CA of res 1
    placed at AF2-canonical bond *angles*, with the C–N bond stretched to
    `cn_bond_length` Å. atom_mask=1 only for the four backbone atoms used."""
    coords = torch.zeros(1, 2, 14, 3, dtype=torch.float32)
    atom_mask = torch.zeros(1, 2, 14, dtype=torch.int8)

    # Place this_c at origin and next_n at (cn, 0, 0). The C→N unit vector
    # along +x is what the cos-angle reference frame is built off.
    cn = cn_bond_length
    coords[0, 0, C_SLOT] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1, N_SLOT] = torch.tensor([cn, 0.0, 0.0])

    # CA(0): direction v from this_c such that v · (+x) = cos(CA-C-N)
    cos_ca_c_n = _CA_C_N_COS[0]                # -0.4473
    v = torch.tensor([cos_ca_c_n, math.sqrt(1 - cos_ca_c_n**2), 0.0])
    coords[0, 0, CA_SLOT] = _CA_C_LEN * v

    # CA(1): direction w from next_n such that (-(+x)) · w = cos(C-N-CA)
    # i.e. w_x = -cos(C-N-CA).
    cos_c_n_ca = _C_N_CA_COS[0]                # -0.5203
    w_x = -cos_c_n_ca                          #  0.5203
    w = torch.tensor([w_x, math.sqrt(1 - w_x**2), 0.0])
    coords[0, 1, CA_SLOT] = torch.tensor([cn, 0.0, 0.0]) + _N_CA_LEN * w

    atom_mask[0, 0, C_SLOT] = 1
    atom_mask[0, 0, CA_SLOT] = 1
    atom_mask[0, 1, N_SLOT] = 1
    atom_mask[0, 1, CA_SLOT] = 1
    if pad_residue_1:
        atom_mask[0, 1, :] = -1

    residue_type = torch.tensor(
        [[ALA, PRO if next_is_pro else ALA]], dtype=torch.long,
    )
    chain_slot = torch.tensor(
        [[0, 0 if same_chain else 1]], dtype=torch.long,
    )
    is_helix = torch.tensor(
        [[both_helix, both_helix]], dtype=torch.bool,
    )
    padding_mask = torch.tensor(
        [[True, not pad_residue_1]], dtype=torch.bool,
    )
    return coords, atom_mask, residue_type, chain_slot, is_helix, padding_mask


def test_zero_loss_at_canonical_geometry():
    args = _two_residue_batch(cn_bond_length=_C_N_BOND_LENGTH[0])
    loss = backbone_continuity_loss(*args)
    # sqrt-eps floor for |·| keeps loss tiny but non-zero; confirm it's well
    # under the per-pair tolerance band.
    assert loss.item() < 1e-4, f"expected ~0 at canonical geometry, got {loss.item()}"


def test_c_n_too_long_triggers_loss():
    bond = 1.6
    loss = backbone_continuity_loss(*_two_residue_batch(cn_bond_length=bond))
    # Only the C–N bond length contributes (angles preserved by construction).
    expected = max(0.0, abs(bond - _C_N_BOND_LENGTH[0]) - 12.0 * _C_N_BOND_LENGTH_STDDEV[0])
    assert loss.item() == pytest.approx(expected, abs=1e-3)
    assert loss.item() > 0.0


def test_proline_uses_proline_reference():
    """With i+1 = PRO and bond at the proline reference (1.341 Å), loss
    should be zero. Tightened tolerance forces the proline branch to be
    distinguishable from the non-PRO 1.329 Å reference."""
    bond = _C_N_BOND_LENGTH[1]                                                   # 1.341 Å
    args = _two_residue_batch(cn_bond_length=bond, next_is_pro=True)
    loss_pro = backbone_continuity_loss(*args, tolerance=0.1)
    assert loss_pro.item() < 1e-3

    # Same bond length, but lying about the residue type — non-PRO 1.329 Å
    # reference now says the bond is wrong.
    coords, atom_mask, _, chain_slot, is_helix, padding_mask = args
    residue_type_nonpro = torch.tensor([[ALA, ALA]], dtype=torch.long)
    loss_nonpro = backbone_continuity_loss(
        coords, atom_mask, residue_type_nonpro, chain_slot, is_helix, padding_mask,
        tolerance=0.1,
    )
    assert loss_nonpro.item() > 1e-3


def test_chain_break_skipped():
    """Even a wildly broken bond should produce zero loss when the two
    residues are flagged as different chains — there's no peptide bond."""
    args = _two_residue_batch(cn_bond_length=10.0, same_chain=False)
    loss = backbone_continuity_loss(*args)
    assert loss.item() == 0.0


def test_antigen_pair_skipped():
    """Antigen residues are cropped fragments where adjacent N-axis
    positions may be chain breaks in the original protein. Even if the
    geometry looks broken, the loss must NOT fire on non-helix pairs."""
    args = _two_residue_batch(cn_bond_length=10.0, both_helix=False)
    loss = backbone_continuity_loss(*args)
    assert loss.item() == 0.0


def test_padded_residue_skipped():
    args = _two_residue_batch(cn_bond_length=10.0, pad_residue_1=True)
    loss = backbone_continuity_loss(*args)
    assert loss.item() == 0.0


def test_padding_invariance():
    """Appending a padded residue to a 2-residue batch must not change the
    loss. Same contract every other masked loss in this codebase obeys."""
    bond = 1.6
    args2 = _two_residue_batch(cn_bond_length=bond)
    coords2, atom_mask2, rtype2, chain2, helix2, pad2 = args2
    base_loss = backbone_continuity_loss(*args2)

    coords3 = torch.cat([coords2, torch.zeros(1, 1, 14, 3)], dim=1)
    atom_mask3 = torch.cat(
        [atom_mask2, -torch.ones(1, 1, 14, dtype=torch.int8)], dim=1,
    )
    rtype3 = torch.cat([rtype2, torch.tensor([[ALA]])], dim=1)
    chain3 = torch.cat([chain2, torch.tensor([[0]])], dim=1)
    helix3 = torch.cat([helix2, torch.tensor([[False]])], dim=1)
    pad3 = torch.cat([pad2, torch.tensor([[False]])], dim=1)

    padded_loss = backbone_continuity_loss(
        coords3, atom_mask3, rtype3, chain3, helix3, pad3,
    )
    assert padded_loss.item() == pytest.approx(base_loss.item(), abs=1e-6)


def test_pro_index_matches_af2():
    """The PRO index used in the loss must match the AF2 restypes ordering;
    the loss reads the i+1 residue type and selects the PRO bond reference
    based on that index."""
    assert _PRO_INDEX == PRO
