"""Tests for the AF2-style steric clash loss."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.features.interaction_matrix import VDW_RADII
from twistr.pipeline.losses.steric_clash import steric_clash_loss
from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

ALA = RESIDUE_TYPE_NAMES.index("ALA")
GLY = RESIDUE_TYPE_NAMES.index("GLY")
CYS = RESIDUE_TYPE_NAMES.index("CYS")
SG_SLOT = ATOM14_SLOT_INDEX["CYS"]["SG"]


def _empty_batch(n_res: int, residue_types: list[int]) -> dict[str, torch.Tensor]:
    """Build a batch with all atoms missing (atom_mask=0). Caller sets specific
    atoms to 1 and provides their coordinates."""
    coords = torch.zeros(1, n_res, 14, 3, dtype=torch.float32)
    atom_mask = torch.zeros(1, n_res, 14, dtype=torch.int8)
    res = torch.tensor([residue_types], dtype=torch.long)
    return {"coordinates": coords, "atom_mask": atom_mask, "residue_type": res}


def test_no_clash_when_atoms_well_separated():
    batch = _empty_batch(2, [ALA, ALA])
    # Place CB of res 0 and CB of res 1 far apart.
    batch["coordinates"][0, 0, 4] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 1, 4] = torch.tensor([100.0, 0.0, 0.0])
    batch["atom_mask"][0, 0, 4] = 1
    batch["atom_mask"][0, 1, 4] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() == 0.0


def test_explicit_clash_matches_formula():
    batch = _empty_batch(2, [ALA, ALA])
    # Two CB atoms at d = 1.0 Å. r_C + r_C = 3.4. tolerance = 1.5.
    # Expected per-pair violation = relu(3.4 - 1.5 - 1.0) = 0.9.
    # Mask sums over (i, j, a, b) symmetric tensor → 2 valid entries:
    # (0, 1, 4, 4) and (1, 0, 4, 4). Sum = 1.8, mean = 1.8 / 2 = 0.9.
    batch["coordinates"][0, 0, 4] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 1, 4] = torch.tensor([1.0, 0.0, 0.0])
    batch["atom_mask"][0, 0, 4] = 1
    batch["atom_mask"][0, 1, 4] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() == pytest.approx(0.9, abs=1e-5)


def test_peptide_bond_excluded():
    # Two adjacent residues with C(0) and N(1) at peptide-bond distance (1.33 Å).
    # r_C + r_N = 3.3. tolerance = 1.5. Strict violation = 3.3 - 1.5 - 1.33 = 0.47.
    # If the C-N bond exclusion works, total loss = 0.
    batch = _empty_batch(2, [ALA, ALA])
    batch["coordinates"][0, 0, 2] = torch.tensor([0.0, 0.0, 0.0])  # C of res 0
    batch["coordinates"][0, 1, 0] = torch.tensor([1.33, 0.0, 0.0])  # N of res 1
    batch["atom_mask"][0, 0, 2] = 1
    batch["atom_mask"][0, 1, 0] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() == 0.0


def test_non_adjacent_c_n_is_not_excluded():
    # C(0) and N(2) are NOT a peptide bond (residues are not sequentially adjacent).
    # If we put them at clash distance, the loss must be > 0.
    batch = _empty_batch(3, [ALA, ALA, ALA])
    batch["coordinates"][0, 0, 2] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 2, 0] = torch.tensor([1.33, 0.0, 0.0])
    batch["atom_mask"][0, 0, 2] = 1
    batch["atom_mask"][0, 2, 0] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() > 0


def test_cys_disulfide_excluded():
    # SG-SG between two CYS at disulfide distance (~2.05 Å).
    # r_S + r_S = 3.6. tolerance = 1.5. Strict violation = 3.6 - 1.5 - 2.05 = 0.05.
    # The CYS-CYS exclusion zeroes this.
    batch = _empty_batch(3, [CYS, ALA, CYS])  # use res 0 and 2 to avoid peptide-bond exclusion
    batch["coordinates"][0, 0, SG_SLOT] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 2, SG_SLOT] = torch.tensor([2.05, 0.0, 0.0])
    batch["atom_mask"][0, 0, SG_SLOT] = 1
    batch["atom_mask"][0, 2, SG_SLOT] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() == 0.0


def test_cys_ala_sg_pair_is_not_excluded():
    # An ALA at slot 5 (no atom defined there for ALA, but suppose it had one)
    # paired with a CYS-SG should NOT be excluded by the disulfide rule.
    # We test the converse: force a non-CYS at slot 5 with a clash to a CYS-SG.
    # Since ALA's slot 5 mask is 0, we use GLY → CYS(SG). A direct test of the
    # filter: only same-CYS-CYS-SG pairs are excluded.
    batch = _empty_batch(3, [CYS, ALA, ALA])
    # CYS-SG vs ALA-CB clash at 1.0 Å
    batch["coordinates"][0, 0, SG_SLOT] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 2, 4] = torch.tensor([1.0, 0.0, 0.0])
    batch["atom_mask"][0, 0, SG_SLOT] = 1
    batch["atom_mask"][0, 2, 4] = 1
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() > 0


def test_missing_atoms_excluded():
    batch = _empty_batch(2, [ALA, ALA])
    # Place atoms at clashing distance but leave atom_mask at 0 → excluded.
    batch["coordinates"][0, 0, 4] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 1, 4] = torch.tensor([1.0, 0.0, 0.0])
    loss = steric_clash_loss(batch["coordinates"], batch["atom_mask"], batch["residue_type"])
    assert loss.item() == 0.0


def test_gradient_flows():
    batch = _empty_batch(3, [ALA, ALA, ALA])
    batch["coordinates"][0, 0, 4] = torch.tensor([0.0, 0.0, 0.0])
    batch["coordinates"][0, 2, 4] = torch.tensor([1.0, 0.0, 0.0])
    batch["atom_mask"][0, 0, 4] = 1
    batch["atom_mask"][0, 2, 4] = 1
    coords = batch["coordinates"].clone().detach().requires_grad_(True)
    loss = steric_clash_loss(coords, batch["atom_mask"], batch["residue_type"])
    loss.backward()
    assert coords.grad.abs().sum() > 0


def test_uses_protenix_sourced_vdw_radii():
    # Sanity: the radii used inside the loss come from the same VDW_RADII table
    # that the detector and existing VDW loss use (single source of truth).
    # ALA-CB is element C → 1.7 Å.
    assert VDW_RADII[ALA, 4].item() == pytest.approx(1.7)
    # CYS-SG → 1.8 Å (S).
    assert VDW_RADII[CYS, SG_SLOT].item() == pytest.approx(1.8)
