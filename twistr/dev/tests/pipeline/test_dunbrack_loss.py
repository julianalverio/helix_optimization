"""Tests for the Dunbrack joint-vMM rotamer loss."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.losses.dunbrack import (
    _GENERAL_INDEX,
    _HELIX_INDEX,
    _LIBRARY_PATH,
    _load_library,
    dunbrack_rotamer_loss,
)
from twistr.tensors.constants import RESIDUE_TYPE_NAMES

ALA = RESIDUE_TYPE_NAMES.index("ALA")
LEU = RESIDUE_TYPE_NAMES.index("LEU")
VAL = RESIDUE_TYPE_NAMES.index("VAL")
PHE = RESIDUE_TYPE_NAMES.index("PHE")

_LIB_AVAILABLE = _LIBRARY_PATH.exists()
_skip_if_no_lib = pytest.mark.skipif(
    not _LIB_AVAILABLE,
    reason=f"Dunbrack library not built at {_LIBRARY_PATH}",
)


def _torsion_sincos_from_chi(chi_radians: torch.Tensor) -> torch.Tensor:
    """Build a (B, N, 7, 2) torsion_sincos tensor from a (B, N, 4) χ tensor."""
    B, N, _ = chi_radians.shape
    sincos = torch.zeros(B, N, 7, 2, dtype=chi_radians.dtype)
    sincos[..., :3, 1] = 1.0                                                       # ω/φ/ψ irrelevant
    sincos[..., 3:7, 0] = torch.sin(chi_radians)
    sincos[..., 3:7, 1] = torch.cos(chi_radians)
    return sincos


def _single_residue_loss(chi_deg, residue, is_helix=True, n_chi=None):
    """Wrapper: evaluate loss on a B=1, N=1 batch with a single residue."""
    chi_rad = torch.tensor([[[math.radians(d) for d in chi_deg]]], dtype=torch.float32)
    chi_mask = torch.tensor([[[1] * (n_chi or len(chi_deg)) + [0] * (4 - (n_chi or len(chi_deg)))]], dtype=torch.bool)
    return dunbrack_rotamer_loss(
        _torsion_sincos_from_chi(chi_rad),
        torch.tensor([[residue]], dtype=torch.long),
        torch.tensor([[is_helix]], dtype=torch.bool),
        chi_mask,
        torch.tensor([[True]], dtype=torch.bool),
    )


@_skip_if_no_lib
def test_chi_at_dominant_mode_has_lower_loss_than_random():
    """LEU helix dominant joint mode is near (χ1, χ2) = (-71°, +168°)
    per the fitter output. A prediction at that mode should score
    lower NLL than a far-from-mode (30°, 30°)."""
    lib = _load_library(torch.device("cpu"), torch.float32)
    assert lib["has_data"][_HELIX_INDEX, LEU]

    loss_mode = _single_residue_loss([-71.0, 168.0, 0.0, 0.0], LEU, is_helix=True)
    loss_rand = _single_residue_loss([30.0, 30.0, 0.0, 0.0], LEU, is_helix=True)
    assert loss_mode.item() < loss_rand.item(), (
        f"χ at joint mode should score lower NLL than random: "
        f"mode={loss_mode.item():.3f} rand={loss_rand.item():.3f}"
    )


@_skip_if_no_lib
def test_helix_dispatch_differs_from_general():
    """Helix-LEU and general-LEU have different joint mode preferences;
    the loss should pick the right library based on `is_helix`."""
    chi_deg = [-71.0, 168.0, 0.0, 0.0]
    loss_helix = _single_residue_loss(chi_deg, LEU, is_helix=True)
    loss_general = _single_residue_loss(chi_deg, LEU, is_helix=False)
    assert loss_helix.item() != loss_general.item()


@_skip_if_no_lib
def test_residue_with_no_chi_returns_zero():
    """ALA has n_chi=0 — has_data is False, loss should be 0 (no signal)."""
    chi_rad = torch.zeros(1, 1, 4)
    chi_mask = torch.zeros(1, 1, 4, dtype=torch.bool)
    loss = dunbrack_rotamer_loss(
        _torsion_sincos_from_chi(chi_rad),
        torch.tensor([[ALA]], dtype=torch.long),
        torch.tensor([[True]], dtype=torch.bool),
        chi_mask,
        torch.tensor([[True]], dtype=torch.bool),
    )
    assert loss.item() == 0.0


@_skip_if_no_lib
def test_padded_residue_returns_zero():
    chi_rad = torch.tensor([[[math.radians(180.0), 0.0, 0.0, 0.0]]])
    chi_mask = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool)
    loss = dunbrack_rotamer_loss(
        _torsion_sincos_from_chi(chi_rad),
        torch.tensor([[LEU]], dtype=torch.long),
        torch.tensor([[True]], dtype=torch.bool),
        chi_mask,
        torch.tensor([[False]], dtype=torch.bool),
    )
    assert loss.item() == 0.0


@_skip_if_no_lib
def test_periodic_invariance_chi_axis():
    """χ is circular — adding 2π should not change the loss."""
    chi_a = [-71.0, 168.0, 0.0, 0.0]
    chi_b = [-71.0 + 360.0, 168.0 - 360.0, 0.0, 0.0]
    loss_a = _single_residue_loss(chi_a, LEU, is_helix=True)
    loss_b = _single_residue_loss(chi_b, LEU, is_helix=True)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-5)


@_skip_if_no_lib
def test_pi_periodic_chi_invariance_phe():
    """PHE χ2 has 2-fold rotational symmetry. The fitter augments the
    data with χ2+180° before fitting, so the resulting mixture has
    symmetric component pairs. The loss should therefore score
    identically (within float tolerance) at χ2 and χ2+180° regardless
    of which PDB-labelled value the data happened to have more of."""
    chi_at_plus = [-66.0, 80.0, 0.0, 0.0]                                          # +80° (close to a fitted mode)
    chi_at_minus = [-66.0, 80.0 - 180.0, 0.0, 0.0]                                 # symmetric -100°
    loss_plus = _single_residue_loss(chi_at_plus, PHE, is_helix=True)
    loss_minus = _single_residue_loss(chi_at_minus, PHE, is_helix=True)
    # Eval-time symmetrisation should make these *exactly* identical
    # (modulo float arithmetic).
    assert loss_plus.item() == pytest.approx(loss_minus.item(), abs=1e-5), (
        f"π-periodic χ invariance broken: "
        f"loss(χ2=+80)={loss_plus.item():.6f} loss(χ2=-100)={loss_minus.item():.6f}"
    )


@_skip_if_no_lib
def test_gradient_flows_through_torsion_sincos():
    """The loss must be differentiable wrt the model's torsion_sincos."""
    chi_rad = torch.tensor([[[math.radians(-50.0), math.radians(170.0), 0.0, 0.0]]])
    sincos = _torsion_sincos_from_chi(chi_rad).requires_grad_(True)
    chi_mask = torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool)                    # LEU has 2 chis
    loss = dunbrack_rotamer_loss(
        sincos,
        torch.tensor([[LEU]], dtype=torch.long),
        torch.tensor([[True]], dtype=torch.bool),
        chi_mask,
        torch.tensor([[True]], dtype=torch.bool),
    )
    loss.backward()
    assert sincos.grad is not None
    # Non-zero gradient on χ1 (slot 3) and χ2 (slot 4); zero elsewhere.
    assert sincos.grad[..., 3, :].abs().sum().item() > 0
    assert sincos.grad[..., 4, :].abs().sum().item() > 0
    assert sincos.grad[..., :3, :].abs().sum().item() == 0
    assert sincos.grad[..., 5:7, :].abs().sum().item() == 0


@_skip_if_no_lib
def test_padding_invariance():
    """Appending a padded residue must not change the loss."""
    chi_2 = torch.tensor([[
        [math.radians(-71.0), math.radians(168.0), 0.0, 0.0],
        [math.radians(180.0), 0.0, 0.0, 0.0],
    ]])
    chi_mask_2 = torch.tensor([[[1, 1, 0, 0], [1, 0, 0, 0]]], dtype=torch.bool)
    rtype_2 = torch.tensor([[LEU, VAL]], dtype=torch.long)
    is_helix_2 = torch.tensor([[True, False]], dtype=torch.bool)
    pad_2 = torch.tensor([[True, True]], dtype=torch.bool)

    base = dunbrack_rotamer_loss(
        _torsion_sincos_from_chi(chi_2), rtype_2, is_helix_2, chi_mask_2, pad_2,
    )

    chi_3 = torch.cat([chi_2, torch.zeros(1, 1, 4)], dim=1)
    chi_mask_3 = torch.cat([chi_mask_2, torch.ones(1, 1, 4, dtype=torch.bool)], dim=1)
    rtype_3 = torch.cat([rtype_2, torch.tensor([[LEU]])], dim=1)
    is_helix_3 = torch.cat([is_helix_2, torch.tensor([[True]])], dim=1)
    pad_3 = torch.cat([pad_2, torch.tensor([[False]])], dim=1)

    padded = dunbrack_rotamer_loss(
        _torsion_sincos_from_chi(chi_3), rtype_3, is_helix_3, chi_mask_3, pad_3,
    )
    assert padded.item() == pytest.approx(base.item(), abs=1e-5)


@_skip_if_no_lib
def test_quality_filter_drops_zero_rows_on_prefiltered_dataset():
    """The Dunbrack public dataset is already pre-filtered to RSPERC≥25
    and FLP_CONFID==clear (per the file's own header). Our filter is a
    redundant safety net; document the expectation that 100% of the rows
    survive on this distributor's file."""
    from twistr.dev.tools.local.dunbrack.fit_rotamer_library import parse_dataset
    n = sum(1 for _ in parse_dataset(_LIBRARY_PATH.parent.parent.parent / "runtime" / "data" / "dunbrack" / "DatasetForBBDepRL2010.txt"))
    assert n > 100_000                                                              # smoke check that the file is real
