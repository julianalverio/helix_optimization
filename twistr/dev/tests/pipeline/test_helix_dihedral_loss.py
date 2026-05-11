"""Tests for the alpha-helix dihedral loss. Builds synthetic backbones with
known φ, ψ, ω via NeRF (Natural Extension Reference Frame) and verifies the
flat-bottomed linear shape, masking, and gradient flow."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.features.chi_angles import atan2_dihedral
from twistr.pipeline.losses.helix_dihedral import helix_dihedral_loss
from twistr.pipeline.models.sidechain import apply_torsions_to_atom14


# Idealized peptide bond geometry (Å, radians). Matches AF2 residue_constants.
L_NCa, L_CaC, L_CN = 1.458, 1.524, 1.329
A_NCaC = math.radians(111.0)
A_CaCN = math.radians(116.0)
A_CNCa = math.radians(121.0)


def _place(p1, p2, p3, bond_length, bond_angle, dihedral):
    """Place atom p4 such that dihedral(p1, p2, p3, p4) = `dihedral`,
    bond angle p2-p3-p4 = `bond_angle`, |p3-p4| = `bond_length`. Convention
    matches `atan2_dihedral`."""
    bc = p3 - p2
    bc = bc / bc.norm()
    n = torch.cross(p2 - p1, bc, dim=-1)
    n = n / n.norm()
    m = torch.cross(n, bc, dim=-1)
    return p3 + bond_length * (
        -math.cos(bond_angle) * bc
        + math.sin(bond_angle) * math.cos(dihedral) * m
        + math.sin(bond_angle) * math.sin(dihedral) * n
    )


def _build_backbone(n_res: int, phi: float, psi: float, omega: float) -> torch.Tensor:
    """Build (n_res, 4, 3) backbone with N/CA/C in slots 0/1/2 (CB zeroed),
    such that all interior φ/ψ/ω equal the requested values (radians)."""
    dtype = torch.float64
    n0 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    ca0 = torch.tensor([L_NCa, 0.0, 0.0], dtype=dtype)
    c0 = ca0 + L_CaC * torch.tensor(
        [-math.cos(A_NCaC), math.sin(A_NCaC), 0.0], dtype=dtype,
    )
    atoms = [n0, ca0, c0]
    for _ in range(1, n_res):
        atoms.append(_place(atoms[-3], atoms[-2], atoms[-1], L_CN, A_CaCN, psi))
        atoms.append(_place(atoms[-3], atoms[-2], atoms[-1], L_NCa, A_CNCa, omega))
        atoms.append(_place(atoms[-3], atoms[-2], atoms[-1], L_CaC, A_NCaC, phi))

    n_arr = torch.stack(atoms[0::3])
    ca_arr = torch.stack(atoms[1::3])
    c_arr = torch.stack(atoms[2::3])
    cb_arr = torch.zeros_like(n_arr)
    return torch.stack([n_arr, ca_arr, c_arr, cb_arr], dim=1).to(torch.float32)


def _observed_dihedrals(bb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
    phi = atan2_dihedral(c[:-2], n[1:-1], ca[1:-1], c[1:-1])
    psi = atan2_dihedral(n[1:-1], ca[1:-1], c[1:-1], n[2:])
    omega = atan2_dihedral(ca[:-2], c[:-2], n[1:-1], ca[1:-1])
    return phi, psi, omega


def test_ideal_helix_zero_loss():
    phi, psi, omega = math.radians(-57.0), math.radians(-47.0), math.radians(180.0)
    bb = _build_backbone(10, phi, psi, omega)

    phi_obs, psi_obs, omega_obs = _observed_dihedrals(bb)
    # Compare via (sin, cos) so ±π and other 2π-equivalent angles match.
    def _circular_close(a: torch.Tensor, target: float) -> bool:
        t = torch.full_like(a, target)
        return torch.allclose(torch.sin(a), torch.sin(t), atol=1e-3) and \
               torch.allclose(torch.cos(a), torch.cos(t), atol=1e-3)
    assert _circular_close(phi_obs, phi)
    assert _circular_close(psi_obs, psi)
    assert _circular_close(omega_obs, omega)

    is_helix = torch.ones(10, dtype=torch.bool)
    loss = helix_dihedral_loss(bb.unsqueeze(0), is_helix.unsqueeze(0), torch.ones_like(is_helix.unsqueeze(0), dtype=torch.bool))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_extended_chain_positive_loss():
    # β-strand-like: φ=-120°, ψ=+120°, ω=180°. Both φ and ψ outside the helix
    # box; ω is canonical so contributes 0.
    phi, psi, omega = math.radians(-120.0), math.radians(120.0), math.radians(180.0)
    bb = _build_backbone(8, phi, psi, omega)
    is_helix = torch.ones(8, dtype=torch.bool)
    loss = helix_dihedral_loss(bb.unsqueeze(0), is_helix.unsqueeze(0), torch.ones_like(is_helix.unsqueeze(0), dtype=torch.bool))
    # φ below low end −100° by 20°; ψ above high end −10° by 130°.
    expected = math.radians(20.0) + math.radians(130.0)
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_non_helix_residues_excluded():
    # Bad backbone — but is_helix all False → loss is exactly zero.
    bb = _build_backbone(8, math.radians(0.0), math.radians(0.0), math.radians(180.0))
    is_helix = torch.zeros(8, dtype=torch.bool)
    loss = helix_dihedral_loss(bb.unsqueeze(0), is_helix.unsqueeze(0), torch.ones_like(is_helix.unsqueeze(0), dtype=torch.bool))
    assert loss.item() == 0.0

    # Three middle residues helix → only the central one (idx 4) has both
    # neighbours also helix, so only it contributes.
    is_helix[3:6] = True
    loss = helix_dihedral_loss(bb.unsqueeze(0), is_helix.unsqueeze(0), torch.ones_like(is_helix.unsqueeze(0), dtype=torch.bool))
    # φ=0° above hi −30° by 30°; ψ=0° above hi −10° by 10°; ω=180° → 0.
    expected = math.radians(30.0) + math.radians(10.0)
    assert loss.item() == pytest.approx(expected, abs=1e-4)


def test_gradient_flows_to_frame():
    n = 6
    torch.manual_seed(0)
    R = torch.eye(3).expand(1, n, 3, 3).contiguous().requires_grad_(True)
    t = (torch.randn(1, n, 3) * 0.3).requires_grad_(True)
    # Identity torsions: cos=1, sin=0 → angle = 0.
    torsions = torch.zeros(1, n, 7, 2)
    torsions[..., 1] = 1.0
    residue_type = torch.full((1, n), 0, dtype=torch.long)  # ALA
    atom14 = apply_torsions_to_atom14(R, t, torsions, residue_type)
    is_helix = torch.ones(1, n, dtype=torch.bool)
    loss = helix_dihedral_loss(atom14, is_helix, torch.ones_like(is_helix))
    assert loss.item() > 0
    loss.backward()
    assert R.grad.abs().sum() > 0
    assert t.grad.abs().sum() > 0
