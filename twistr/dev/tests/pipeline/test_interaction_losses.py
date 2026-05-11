"""Per-interaction-type geometric losses on predicted coords. The losses are
two-sided and flat-bottomed: zero when the geometry matches the binary GT
label, linear in physical units (Å, cosine) otherwise."""
from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS
from twistr.pipeline.features.chi_angles import chi_sincos as encode_sincos, compute_chi_angles
from twistr.pipeline.features.interaction_matrix import clean_interaction_matrix
from twistr.pipeline.losses.interaction_geometry import (
    aromatic_subtype_losses,
    hbond_interaction_loss,
    interaction_geometry_losses,
    vdw_interaction_loss,
)
from twistr.pipeline.models.rotation import frame_from_three_points
from twistr.pipeline.models.sidechain import apply_torsions_to_atom14
from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")


def _two_residue_batch(
    res_a: str, res_b: str, ca_b_pos: tuple[float, float, float],
    cb_a: tuple[float, float, float] = (1.5, 0.0, 0.0),
    cb_b_offset: tuple[float, float, float] = (-1.5, 0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a (1, 2, 14, 3) atom14 batch in Å with two residues. Residue 0 has
    CA at origin, CB at `cb_a`. Residue 1 has CA at `ca_b_pos`, CB at
    `ca_b_pos + cb_b_offset`. Other atoms left at zero (loss tests typically
    only use sidechain CB or specific atoms — caller fills more if needed)."""
    res_a_idx = RESIDUE_TYPE_NAMES.index(res_a)
    res_b_idx = RESIDUE_TYPE_NAMES.index(res_b)
    coords = torch.zeros(1, 2, 14, 3, dtype=torch.float32)
    atom_mask = torch.zeros(1, 2, 14, dtype=torch.int8)
    # Residue 0: CA at origin, CB at cb_a; backbone present.
    coords[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0])  # CA
    coords[0, 0, 0] = torch.tensor([-0.5, 1.36, 0.0])  # N
    coords[0, 0, 2] = torch.tensor([1.5, 0.0, 0.0])  # C
    coords[0, 0, 4] = torch.tensor(cb_a)  # CB
    atom_mask[0, 0, [0, 1, 2, 4]] = 1
    # Residue 1: similarly, offset.
    ca_b = torch.tensor(ca_b_pos)
    coords[0, 1, 1] = ca_b
    coords[0, 1, 0] = ca_b + torch.tensor([-0.5, 1.36, 0.0])
    coords[0, 1, 2] = ca_b + torch.tensor([1.5, 0.0, 0.0])
    coords[0, 1, 4] = ca_b + torch.tensor(cb_b_offset)
    atom_mask[0, 1, [0, 1, 2, 4]] = 1
    residue_type = torch.tensor([[res_a_idx, res_b_idx]], dtype=torch.long)
    return coords, residue_type, atom_mask


# ----------------------------------------------------------------------
# VDW

def test_vdw_loss_zero_when_in_band_and_target_one():
    """Place two ALA residues so CB-CB distance is r_sum (≈ 3.4 Å for C-C).
    Target VDW=1 → loss should be ~0."""
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(3.4, 0.0, 0.0),
    )
    # CB-CB will be 3.4 - 1.5 - 1.5 = 0.4 — too close. Move farther.
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(6.4, 0.0, 0.0),
    )
    # Now CB at (1.5, 0, 0) and (6.4 - 1.5, 0, 0) = (4.9, 0, 0). Distance = 3.4.
    target = torch.zeros(1, 2, 2)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    loss = vdw_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() < 1e-5, f"in-band VDW with target=1 should give ~0 loss, got {loss}"


def test_vdw_loss_nonzero_when_far_and_target_one():
    """Two residues 20 Å apart with target=1: loss should be > 0."""
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(20.0, 0.0, 0.0),
    )
    target = torch.zeros(1, 2, 2)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    loss = vdw_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() > 5.0, f"far-apart VDW with target=1 should give large loss, got {loss}"


def test_vdw_loss_zero_when_far_and_target_zero():
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(20.0, 0.0, 0.0),
    )
    target = torch.zeros(1, 2, 2)
    loss = vdw_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() < 1e-5


def test_vdw_loss_nonzero_when_in_band_and_target_zero():
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(6.4, 0.0, 0.0),
    )
    # CB-CB ≈ 3.4 Å; in band for ALA-ALA.
    target = torch.zeros(1, 2, 2)
    loss = vdw_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() > 0.0, "in-band VDW with target=0 should fire"


def test_vdw_gradient_pushes_toward_band_when_target_one():
    """When target=1 and atoms are too far, gradient on coords should pull
    them closer along the CB-CB axis."""
    coords, residue_type, atom_mask = _two_residue_batch(
        "ALA", "ALA", ca_b_pos=(20.0, 0.0, 0.0),
    )
    coords = coords.requires_grad_(True)
    target = torch.zeros(1, 2, 2)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    loss = vdw_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    cb0_grad_x = coords.grad[0, 0, 4, 0].item()
    cb1_grad_x = coords.grad[0, 1, 4, 0].item()
    # Residue 0 CB at (1.5, 0, 0); residue 1 CB at (18.5, 0, 0). Gradient should
    # push CB0 in +x (toward CB1) and CB1 in -x (toward CB0).
    assert cb0_grad_x < 0, "VDW pull: CB0 should want to move +x (so loss has -x grad)"
    assert cb1_grad_x > 0, "VDW pull: CB1 should want to move -x (so loss has +x grad)"


# ----------------------------------------------------------------------
# H-bond

def _hbond_geometry_batch(d_ang: float, x_d_a_deg: float, d_a_y_deg: float):
    """Construct an h-bond donor-acceptor geometry. Both residues are SER —
    using the OG sidechain OH as both donor and acceptor (slot 5 = OG, parent
    = CB at slot 4).
      D = O at (0,0,0); X = parent CB at (-1, 0, 0).
      A = at (d_ang, 0, 0); Y = parent at angle DAY off the AD axis.
    """
    ser = RESIDUE_TYPE_NAMES.index("SER")
    coords = torch.zeros(1, 2, 14, 3, dtype=torch.float32)
    atom_mask = torch.zeros(1, 2, 14, dtype=torch.int8)
    residue_type = torch.tensor([[ser, ser]], dtype=torch.long)

    # Need backbone (N, CA, C) and sidechain (CB, OG) populated. Slot indices
    # for SER: N=0, CA=1, C=2, O=3, CB=4, OG=5.
    coords[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0])           # CA dummy
    coords[0, 0, 0] = torch.tensor([-0.5, 2.36, 0.0])         # N dummy
    coords[0, 0, 2] = torch.tensor([1.5, 1.0, 0.0])           # C dummy
    coords[0, 0, 4] = torch.tensor([-1.0, 0.0, 0.0])          # CB at (-1, 0, 0) — parent X
    coords[0, 0, 5] = torch.tensor([0.0, 0.0, 0.0])           # OG = donor D at origin
    atom_mask[0, 0, [0, 1, 2, 4, 5]] = 1

    # X-D-A angle: X=(-1,0,0), D=(0,0,0), A=(d, 0, 0). XD = (-1, 0, 0); DA = (d, 0, 0).
    # cos(X-D-A) = XD·DA / (|XD||DA|) = -1 ⋅ d / (1 ⋅ d) = -1. Angle = 180°.
    # We want a specific x_d_a_deg. Place A at (cos(180-x_d_a) * d, sin(180-x_d_a) * d, 0)
    # so DA direction makes angle x_d_a_deg with XD = (-1, 0, 0).
    theta_xda = math.radians(x_d_a_deg)
    a_dir = torch.tensor([-math.cos(theta_xda), math.sin(theta_xda), 0.0])  # angle from XD
    A = a_dir * d_ang
    coords[0, 1, 5] = A                                                       # acceptor OG
    coords[0, 1, 1] = A + torch.tensor([0.0, -1.0, 0.0])                       # CA dummy

    # D-A-Y: D = (0,0,0), A = above, Y = parent of A. AD = D - A = -A_dir * d. We want
    # cos(D-A-Y) = cos(d_a_y_deg). Place Y such that Y - A makes angle d_a_y_deg with AD.
    theta_day = math.radians(d_a_y_deg)
    ad_unit = -a_dir
    # Pick a perpendicular-to-ad_unit direction (in xy plane).
    perp = torch.tensor([-ad_unit[1].item(), ad_unit[0].item(), 0.0])
    perp = perp / perp.norm().clamp_min(1e-8)
    y_dir = ad_unit * math.cos(theta_day) + perp * math.sin(theta_day)
    Y = A + y_dir * 1.5                                                       # arbitrary parent length
    coords[0, 1, 4] = Y                                                       # CB = parent of OG
    coords[0, 1, 0] = A + torch.tensor([-0.5, 1.36, 0.0])
    coords[0, 1, 2] = A + torch.tensor([1.5, 0.0, 0.0])
    atom_mask[0, 1, [0, 1, 2, 4, 5]] = 1
    return coords, residue_type, atom_mask


def test_hbond_loss_zero_when_in_band_and_target_one():
    coords, residue_type, atom_mask = _hbond_geometry_batch(
        d_ang=3.0, x_d_a_deg=170.0, d_a_y_deg=120.0,
    )
    target = torch.zeros(1, 2, 2)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    loss = hbond_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() < 1e-5, f"valid hbond with target=1 should be ~0, got {loss}"


def test_hbond_loss_violates_distance():
    coords, residue_type, atom_mask = _hbond_geometry_batch(
        d_ang=4.5, x_d_a_deg=170.0, d_a_y_deg=120.0,
    )
    target = torch.zeros(1, 2, 2)
    target[0, 0, 1] = 1.0
    target[0, 1, 0] = 1.0
    loss = hbond_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    # Two off-diagonal cells (0,1) and (1,0) each contribute 0.9 (= 4.5 - 3.6)
    # since both are target=1 with the same geometry; mean = 0.9 (sum 1.8 /
    # B*N*(N-1) = 2).
    assert abs(loss.item() - 0.9) < 0.01


def test_hbond_loss_zero_when_far_and_target_zero():
    coords, residue_type, atom_mask = _hbond_geometry_batch(
        d_ang=10.0, x_d_a_deg=170.0, d_a_y_deg=120.0,
    )
    target = torch.zeros(1, 2, 2)
    loss = hbond_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() < 1e-5


def test_hbond_loss_nonzero_when_in_band_and_target_zero():
    coords, residue_type, atom_mask = _hbond_geometry_batch(
        d_ang=3.0, x_d_a_deg=170.0, d_a_y_deg=120.0,
    )
    target = torch.zeros(1, 2, 2)
    loss = hbond_interaction_loss(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() > 0


# ----------------------------------------------------------------------
# Aromatic

def _phe_pair_batch(centroid_offset: torch.Tensor, n2_dir: torch.Tensor):
    """Build a (1, 2, 14, 3) batch with two PHE rings. Ring 1 is at the origin
    in the xy plane (normal = +z). Ring 2 is at `centroid_offset`, rotated so
    its normal points along `n2_dir`. Constructs all 6 ring atoms as a regular
    hexagon of radius 1.4 Å (mimicking the PHE benzene ring)."""
    phe = RESIDUE_TYPE_NAMES.index("PHE")
    coords = torch.zeros(1, 2, 14, 3, dtype=torch.float32)
    atom_mask = torch.zeros(1, 2, 14, dtype=torch.int8)
    residue_type = torch.tensor([[phe, phe]], dtype=torch.long)

    # Ring atoms at standard hexagon, in xy plane, radius 1.4 Å. Slot order
    # for PHE: ATOM14_SLOT_INDEX["PHE"] for CG, CD1, CD2, CE1, CE2, CZ.
    angles = torch.tensor([0.0, 60.0, -60.0, 120.0, -120.0, 180.0]) * math.pi / 180.0
    radius = 1.4
    ring_local = torch.stack([radius * angles.cos(), radius * angles.sin(), torch.zeros(6)], dim=-1)
    slots = [ATOM14_SLOT_INDEX["PHE"][n] for n in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]

    # Place ring 1 at origin.
    for k, slot in enumerate(slots):
        coords[0, 0, slot] = ring_local[k]
        atom_mask[0, 0, slot] = 1
    # Backbone for residue 0
    coords[0, 0, 0] = torch.tensor([-0.5, 1.36, 0.0])
    coords[0, 0, 1] = torch.tensor([0.0, 0.0, -2.0])  # CA below ring
    coords[0, 0, 2] = torch.tensor([1.5, 0.0, -2.0])
    coords[0, 0, 4] = torch.tensor([-0.5, -0.7, -2.0])  # CB
    atom_mask[0, 0, [0, 1, 2, 4]] = 1

    # Place ring 2 at `centroid_offset`, normal along `n2_dir`. Rotate xy plane
    # to a plane normal to n2_dir.
    n2 = n2_dir / n2_dir.norm()
    # Build orthonormal basis with n2 as third axis.
    ref = torch.tensor([1.0, 0.0, 0.0]) if abs(n2[0].item()) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    e1 = torch.linalg.cross(ref, n2, dim=-1)
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(n2, e1, dim=-1)
    R = torch.stack([e1, e2, n2], dim=-1)  # cols are ring x, y, normal
    for k, slot in enumerate(slots):
        rotated = R @ ring_local[k]
        coords[0, 1, slot] = rotated + centroid_offset
        atom_mask[0, 1, slot] = 1
    coords[0, 1, 0] = centroid_offset + torch.tensor([-0.5, 1.36, 0.0])
    coords[0, 1, 1] = centroid_offset + torch.tensor([0.0, 0.0, -2.0])
    coords[0, 1, 2] = centroid_offset + torch.tensor([1.5, 0.0, -2.0])
    coords[0, 1, 4] = centroid_offset + torch.tensor([-0.5, -0.7, -2.0])
    atom_mask[0, 1, [0, 1, 2, 4]] = 1

    return coords, residue_type, atom_mask


def _arom_target(positive: str | None) -> torch.Tensor:
    """(1, 2, 2, 3) sub-type GT for the (0, 1) and (1, 0) pair. Channel order
    is [parallel_displaced, sandwich, t_shaped]. `positive=None` → all zeros."""
    target = torch.zeros(1, 2, 2, 3)
    if positive is None:
        return target
    idx = {"parallel_displaced": 0, "sandwich": 1, "t_shaped": 2}[positive]
    target[0, 0, 1, idx] = 1.0
    target[0, 1, 0, idx] = 1.0
    return target


def test_aromatic_sandwich_satisfies_target_one():
    """Two PHE rings stacked at d=3.7 Å, parallel normals → sandwich."""
    coords, residue_type, atom_mask = _phe_pair_batch(
        centroid_offset=torch.tensor([0.0, 0.0, 3.7]),
        n2_dir=torch.tensor([0.0, 0.0, 1.0]),
    )
    target = _arom_target("sandwich")
    out = aromatic_subtype_losses(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert out["sandwich"].item() < 1e-5


def test_aromatic_t_shape_satisfies_target_one():
    """Two PHE rings, perpendicular normals, distance 5.5 Å → T-shape."""
    coords, residue_type, atom_mask = _phe_pair_batch(
        centroid_offset=torch.tensor([5.5, 0.0, 0.0]),
        n2_dir=torch.tensor([1.0, 0.0, 0.0]),  # perpendicular to ring 1 normal (+z)
    )
    target = _arom_target("t_shaped")
    out = aromatic_subtype_losses(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert out["t_shaped"].item() < 1e-5


def test_aromatic_far_apart_no_loss_when_target_zero():
    coords, residue_type, atom_mask = _phe_pair_batch(
        centroid_offset=torch.tensor([15.0, 0.0, 0.0]),
        n2_dir=torch.tensor([0.0, 0.0, 1.0]),
    )
    target = _arom_target(None)
    out = aromatic_subtype_losses(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert out["parallel_displaced"].item() < 1e-5
    assert out["sandwich"].item() < 1e-5
    assert out["t_shaped"].item() < 1e-5


def test_aromatic_in_band_with_target_zero_fires():
    """Sandwich geometry with all sub-type GTs = 0 → sandwich loss must fire
    (margin penalty for being inside the sandwich band)."""
    coords, residue_type, atom_mask = _phe_pair_batch(
        centroid_offset=torch.tensor([0.0, 0.0, 3.7]),
        n2_dir=torch.tensor([0.0, 0.0, 1.0]),
    )
    target = _arom_target(None)
    out = aromatic_subtype_losses(coords, residue_type, atom_mask, target, torch.ones(1, 2, dtype=torch.bool))
    assert out["sandwich"].item() > 0.0


# ----------------------------------------------------------------------
# Integration: GT-everything → losses ≈ 0 on real 1BRS

@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_loss_near_zero_with_gt_coords_on_1brs():
    """When predicted coords = GT coords (atom14 from chi+frame), the binary
    interaction matrix derived from GT is, by definition, satisfied by GT
    geometry. The losses should be ~0 (subject to placement precision)."""
    data = np.load(EXAMPLE_NPZ)
    coords_dataset = torch.from_numpy(data["coordinates"].astype(np.float32)) / COORD_SCALE_ANGSTROMS
    residue_type = torch.from_numpy(data["residue_type"]).long()
    atom_mask = torch.from_numpy(data["atom_mask"])

    # Build the model-output equivalents from GT.
    R = frame_from_three_points(coords_dataset[:, 0], coords_dataset[:, 1], coords_dataset[:, 2])
    t = coords_dataset[:, 1]
    angles, _ = compute_chi_angles(coords_dataset[None], residue_type[None], atom_mask[None])
    chi_sc = encode_sincos(angles[0])
    n_res = chi_sc.shape[0]
    torsions = torch.zeros(n_res, 7, 2)
    torsions[..., 1] = 1.0
    torsions[:, 3:7] = chi_sc
    placed = apply_torsions_to_atom14(R[None], t[None], torsions[None], residue_type[None])  # (1, N, 14, 3)

    # Use the actual GT coords (in dataset units) for the binary target.
    batch = {
        "coordinates": coords_dataset[None],
        "residue_type": residue_type[None],
        "atom_mask": atom_mask[None],
    }
    target_im = clean_interaction_matrix(batch)

    losses = interaction_geometry_losses(placed, residue_type[None], atom_mask[None], target_im, torch.ones(1, residue_type.shape[0], dtype=torch.bool))
    # ~1 Å placement precision means atoms can drift across ~1 Å of band edge.
    # Loose ceiling — bands are several Å wide, so deep-band cells stay 0;
    # only border cells contribute.
    assert losses["vdw"].item() < 0.5, f"VDW loss on GT: {losses['vdw'].item()}"
    assert losses["hbond"].item() < 0.5, f"H-bond loss on GT: {losses['hbond'].item()}"
    for key in ("parallel_displaced", "sandwich", "t_shaped"):
        assert losses[key].item() < 0.5, f"{key} loss on GT: {losses[key].item()}"
