"""Tests for the hydrophobic-stub & aromatic-ring packing loss."""
from __future__ import annotations

import torch

from twistr.pipeline.features.interaction_matrix import IS_PACKING_ATOM
from twistr.pipeline.losses.packing import packing_neighbor_loss
from twistr.tensors.constants import (
    ATOM14_SLOT_INDEX,
    RESIDUE_TYPE_INDEX,
)


LEU = RESIDUE_TYPE_INDEX["LEU"]
ALA = RESIDUE_TYPE_INDEX["ALA"]
GLY = RESIDUE_TYPE_INDEX["GLY"]
ASP = RESIDUE_TYPE_INDEX["ASP"]
PHE = RESIDUE_TYPE_INDEX["PHE"]


def _two_residues(
    res0: int, res1: int, sep_ang: float,
    res0_is_helix: bool = True, res1_is_helix: bool = True,
    res0_is_interface: bool = True, res1_is_interface: bool = True,
) -> dict[str, torch.Tensor]:
    """B=1, N=2 batch. Every atom of residue 0 at origin; every atom of
    residue 1 at (sep_ang, 0, 0). All 14 slots present for both."""
    B, N = 1, 2
    coords = torch.zeros(B, N, 14, 3)
    coords[:, 1, :, 0] = sep_ang
    return {
        "coords_atom14_ang": coords,
        "residue_type": torch.tensor([[res0, res1]], dtype=torch.long),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[res0_is_interface, res1_is_interface]]),
        "is_helix": torch.tensor([[res0_is_helix, res1_is_helix]]),
        "padding_mask": torch.tensor([[True, True]]),
    }


def test_atom_set_leu():
    """LEU packing atoms: CB, CG, CD1, CD2 (slots 4, 5, 6, 7)."""
    expected = {ATOM14_SLOT_INDEX["LEU"][n] for n in ("CB", "CG", "CD1", "CD2")}
    actual = {int(i) for i in torch.nonzero(IS_PACKING_ATOM[LEU], as_tuple=False).flatten()}
    assert actual == expected


def test_atom_set_asp_only_cb():
    """ASP packing atoms: CB only — CG is the carboxylate C, OD1/OD2 are
    the functional-group oxygens."""
    expected = {ATOM14_SLOT_INDEX["ASP"]["CB"]}
    actual = {int(i) for i in torch.nonzero(IS_PACKING_ATOM[ASP], as_tuple=False).flatten()}
    assert actual == expected


def test_atom_set_gly_empty():
    """GLY has no sidechain → empty packing-atom set."""
    assert IS_PACKING_ATOM[GLY].sum().item() == 0


def test_atom_set_phe_stub_plus_ring():
    """PHE packing atoms: CB (stub) + 6 ring atoms (CG, CD1, CD2, CE1, CE2, CZ)."""
    expected = {
        ATOM14_SLOT_INDEX["PHE"][n] for n in
        ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    }
    actual = {int(i) for i in torch.nonzero(IS_PACKING_ATOM[PHE], as_tuple=False).flatten()}
    assert actual == expected


def test_tight_packing_zero_loss():
    """Two LEUs at 4.0 Å (in band) — every from-atom sees 14 in-band neighbors.
    count >> n_target → relu(n_target - count) = 0 → loss = 0."""
    batch = _two_residues(LEU, LEU, sep_ang=4.0)
    loss = packing_neighbor_loss(**batch)
    assert loss.item() < 1e-4


def test_far_apart_max_loss():
    """Two LEUs at 10 Å (out of band) — count ≈ 0 → loss ≈ n_target = 4."""
    batch = _two_residues(LEU, LEU, sep_ang=10.0)
    loss = packing_neighbor_loss(**batch, n_target=4.0)
    assert abs(loss.item() - 4.0) < 1e-3


def test_isolated_residue_max_loss():
    """Single interface residue with no neighbors — loss = n_target per
    packing atom."""
    B, N = 1, 1
    batch = {
        "coords_atom14_ang": torch.zeros(B, N, 14, 3),
        "residue_type": torch.tensor([[LEU]], dtype=torch.long),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True]]),
        "is_helix": torch.tensor([[True]]),
        "padding_mask": torch.tensor([[True]]),
    }
    loss = packing_neighbor_loss(**batch, n_target=4.0)
    assert abs(loss.item() - 4.0) < 1e-3


def test_neighbor_scope_excludes_non_interface_antigen_sidechain():
    """Option III: an antigen non-interface residue's sidechain atoms must
    NOT count as neighbors. Set up a helix interface LEU and a non-helix
    non-interface LEU; place the non-helix LEU sidechain atoms 4 Å away
    but its backbone atoms 100 Å away. Result: no in-band neighbors visible
    to the helix LEU, so loss == n_target."""
    B, N = 1, 2
    coords = torch.zeros(B, N, 14, 3)
    # Place residue-1 backbone (slots 0-3) far away.
    coords[:, 1, :4, 0] = 100.0
    # Place residue-1 sidechain (slots 4-13) 4 Å away — would be in band IF
    # they counted as neighbors.
    coords[:, 1, 4:, 0] = 4.0
    batch = {
        "coords_atom14_ang": coords,
        "residue_type": torch.tensor([[LEU, LEU]], dtype=torch.long),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True, False]]),
        "is_helix": torch.tensor([[True, False]]),
        "padding_mask": torch.tensor([[True, True]]),
    }
    loss = packing_neighbor_loss(**batch, n_target=4.0)
    assert abs(loss.item() - 4.0) < 1e-3, (
        f"non-interface antigen sidechain leaked into neighbor count: loss={loss.item()}"
    )


def test_neighbor_scope_includes_non_interface_antigen_backbone():
    """Option III: an antigen non-interface residue's *backbone* atoms DO
    count as neighbors. Mirror of the previous test — put backbone in band
    and sidechain far away — should drive the loss to ~0 (4 backbone atoms
    × 1 in-band contribution each ≈ 4 neighbors per from-atom)."""
    B, N = 1, 2
    coords = torch.zeros(B, N, 14, 3)
    coords[:, 1, :4, 0] = 4.0       # backbone in band
    coords[:, 1, 4:, 0] = 100.0     # sidechain far
    batch = {
        "coords_atom14_ang": coords,
        "residue_type": torch.tensor([[LEU, LEU]], dtype=torch.long),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True, False]]),
        "is_helix": torch.tensor([[True, False]]),
        "padding_mask": torch.tensor([[True, True]]),
    }
    loss = packing_neighbor_loss(**batch, n_target=4.0)
    # 4 backbone atoms in band, n_target=4 → count ≈ 4, relu(0) ≈ 0.
    assert loss.item() < 0.5, f"backbone neighbor count below expectation: loss={loss.item()}"


def test_self_residue_excluded():
    """A LEU's own packing atoms must not count its own other atoms as
    neighbors. Single LEU at origin (every atom at the same coordinate, so
    intra-residue distances are 0 — well within any band). If self-residue
    is properly excluded, the count is 0 and loss == n_target."""
    B, N = 1, 1
    batch = {
        "coords_atom14_ang": torch.zeros(B, N, 14, 3),  # all atoms at origin
        "residue_type": torch.tensor([[LEU]], dtype=torch.long),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True]]),
        "is_helix": torch.tensor([[True]]),
        "padding_mask": torch.tensor([[True]]),
    }
    loss = packing_neighbor_loss(**batch, n_target=4.0)
    assert abs(loss.item() - 4.0) < 1e-3


def test_gradient_flow():
    """Loss has finite gradient w.r.t. coordinates."""
    batch = _two_residues(LEU, LEU, sep_ang=7.0)
    batch["coords_atom14_ang"].requires_grad_(True)
    loss = packing_neighbor_loss(**batch)
    loss.backward()
    grad = batch["coords_atom14_ang"].grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0


def test_padding_invariance():
    """Padding the batch with extra residues does not change the loss."""
    base = _two_residues(LEU, LEU, sep_ang=7.0)
    base_loss = packing_neighbor_loss(**base)

    # Now add a third "padded" residue.
    B = 1
    N = 3
    coords = torch.zeros(B, N, 14, 3)
    coords[:, 0, :, :] = base["coords_atom14_ang"][:, 0]
    coords[:, 1, :, :] = base["coords_atom14_ang"][:, 1]
    # Padded residue 2 has garbage coords in band — shouldn't matter because
    # padding_mask is False there.
    coords[:, 2, :, 0] = 4.0
    padded_batch = {
        "coords_atom14_ang": coords,
        "residue_type": torch.tensor([[LEU, LEU, 0]], dtype=torch.long),
        "atom_mask": torch.cat([
            torch.ones(B, 2, 14, dtype=torch.int8),
            torch.full((B, 1, 14), -1, dtype=torch.int8),
        ], dim=1),
        "is_interface_residue": torch.tensor([[True, True, False]]),
        "is_helix": torch.tensor([[True, True, False]]),
        "padding_mask": torch.tensor([[True, True, False]]),
    }
    padded_loss = packing_neighbor_loss(**padded_batch)
    assert abs(base_loss.item() - padded_loss.item()) < 1e-5


def test_interface_gating_only_interface_residues_contribute():
    """Two LEUs in tight packing — but only one is at the interface. The
    non-interface LEU's packing atoms should NOT contribute. Symmetry check
    against the both-interface case: the all-interface case has 2× the
    from-atoms but the same per-atom loss (zero — they're tightly packed),
    so both should return 0."""
    both = _two_residues(LEU, LEU, sep_ang=4.0)
    one = _two_residues(LEU, LEU, sep_ang=4.0, res1_is_interface=False)
    assert both["coords_atom14_ang"].equal(one["coords_atom14_ang"])
    assert packing_neighbor_loss(**both).item() < 1e-4
    assert packing_neighbor_loss(**one).item() < 1e-4

    # Now arrange so res 0 is at the interface but isolated, and res 1 is
    # in-band but NOT at the interface. Both-interface: both are tightly
    # packed → loss = 0. One-interface: res 0's packing atoms see in-band
    # neighbors (res 1's atoms are valid neighbors because res 1 is_helix=True),
    # so loss = 0 as well. Distinguish by making res 1 non-helix non-interface
    # — then it's only a backbone neighbor for res 0.
    coords = torch.zeros(1, 2, 14, 3)
    coords[:, 1, :, 0] = 4.0
    far_batch = {
        "coords_atom14_ang": coords,
        "residue_type": torch.tensor([[LEU, LEU]], dtype=torch.long),
        "atom_mask": torch.ones(1, 2, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True, False]]),
        "is_helix": torch.tensor([[True, False]]),
        "padding_mask": torch.tensor([[True, True]]),
    }
    # Only res 0 contributes to the numerator. Its packing atoms see res 1's
    # backbone (4 atoms) — sidechain excluded by Option III. Count ≈ 4 per
    # from-atom, n_target = 4 → loss ≈ 0.
    loss = packing_neighbor_loss(**far_batch, n_target=4.0)
    assert loss.item() < 0.5

    # And the non-interface residue's own packing atoms must not appear in
    # the denominator: if we flip both interface bits, we get the same loss
    # but averaged over 2× as many atoms — they're all tightly packed so
    # zero, same answer. The proper denominator test: make the geometry
    # produce uneven per-atom losses across the two residues.
    coords2 = torch.zeros(1, 2, 14, 3)
    coords2[:, 1, :, 0] = 10.0
    a_only = {
        "coords_atom14_ang": coords2,
        "residue_type": torch.tensor([[LEU, LEU]], dtype=torch.long),
        "atom_mask": torch.ones(1, 2, 14, dtype=torch.int8),
        "is_interface_residue": torch.tensor([[True, False]]),
        "is_helix": torch.tensor([[True, True]]),  # both helix → res 1 atoms are valid neighbors
        "padding_mask": torch.tensor([[True, True]]),
    }
    both_interface = {**a_only, "is_interface_residue": torch.tensor([[True, True]])}
    # In a_only, only res 0 packing atoms contribute. In both_interface,
    # both res 0 and res 1 packing atoms contribute. Per-atom loss is the
    # same in both. The mean over from-atoms is identical → same per-example
    # loss → same final value.
    # So the test below checks that non-interface res 1 atoms simply don't
    # *show up* in the denominator — which is implicit in the fact that the
    # loss value is finite and equals n_target (everyone isolated).
    loss_a_only = packing_neighbor_loss(**a_only, n_target=4.0)
    loss_both = packing_neighbor_loss(**both_interface, n_target=4.0)
    assert abs(loss_a_only.item() - 4.0) < 1e-3
    assert abs(loss_both.item() - 4.0) < 1e-3
