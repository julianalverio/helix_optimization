"""Bug-proofing the differentiable interaction detector. Five layers:

  1. Independent NumPy oracle cross-check on real 1BRS data.
  2. Geometric invariants (translation/rotation invariance, symmetry, range).
  3. Sub-component spot-checks (centroid/normal, _band, VDW/donor tables).
  4. Hand-built canonical geometries for each interaction type.
  5. Differentiability — gradient flows back to coords."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.features.interaction_matrix import (
    AROMATIC_RING_ATOMS,
    AROMATIC_RING_SLOTS,
    CHANNELS,
    ELEMENT_VDW,
    HBOND_ACCEPTORS,
    HBOND_DONORS,
    HBOND_DONORS_ATOM14,
    HBOND_DONORS_MASK,
    IS_AROMATIC,
    VDW_RADII,
    _band,
    _ring_centroid_normal,
    interaction_matrix,
)
from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")
COORD_SCALE_ANGSTROMS = 10.0


# ----------------------------------------------------------------------
# Independent NumPy oracle. Loops over residue pairs with hard thresholds and
# named-atom dict lookups — no torch tensors, no atom14 slot tensors, no
# soft scoring. The differentiable detector and this oracle would have to
# share the exact same bug for both to pass on real data.

def _oracle(coords_ang: np.ndarray, residue_type: np.ndarray, atom_mask: np.ndarray) -> np.ndarray:
    """Returns (N, N, 5) bool tensor matching channels [vdw, hbond, pd, sandwich, t_shaped]."""
    N = len(residue_type)
    M = np.zeros((N, N, 5), dtype=bool)

    def _present(i, atom, ri):
        return atom in ATOM14_SLOT_INDEX[ri] and atom_mask[i, ATOM14_SLOT_INDEX[ri][atom]] == 1

    def _pos(i, atom, ri):
        return coords_ang[i, ATOM14_SLOT_INDEX[ri][atom]]

    backbone_atoms = {"N", "CA", "C", "O"}

    for i in range(N):
        ri = RESIDUE_TYPE_NAMES[residue_type[i]]
        for j in range(N):
            if i == j:
                continue
            rj = RESIDUE_TYPE_NAMES[residue_type[j]]

            # VDW: any sidechain-sidechain atom pair within the contact band.
            # Backbone atoms (N/CA/C/O) are excluded — sequence-adjacent
            # residues shouldn't get a vdw label from CA-CA backbone proximity.
            for atom_a in ATOM14_SLOT_INDEX[ri]:
                if atom_a in backbone_atoms or not _present(i, atom_a, ri):
                    continue
                pa = _pos(i, atom_a, ri)
                ra = ELEMENT_VDW[atom_a[0]]
                for atom_b in ATOM14_SLOT_INDEX[rj]:
                    if atom_b in backbone_atoms or not _present(j, atom_b, rj):
                        continue
                    pb = _pos(j, atom_b, rj)
                    rb = ELEMENT_VDW[atom_b[0]]
                    d = np.linalg.norm(pa - pb)
                    r_sum = ra + rb
                    if r_sum - 0.4 <= d <= r_sum + 0.5:
                        M[i, j, 0] = True
                        break
                if M[i, j, 0]:
                    break

            # H-bond: i donates to j.
            for d_atom, x_atom in HBOND_DONORS[ri]:
                if not (_present(i, d_atom, ri) and _present(i, x_atom, ri)):
                    continue
                D, X = _pos(i, d_atom, ri), _pos(i, x_atom, ri)
                for a_atom, y_atom in HBOND_ACCEPTORS[rj]:
                    if not (_present(j, a_atom, rj) and _present(j, y_atom, rj)):
                        continue
                    A, Y = _pos(j, a_atom, rj), _pos(j, y_atom, rj)
                    d_DA = np.linalg.norm(A - D)
                    if not (2.5 <= d_DA <= 3.6):
                        continue
                    XD, DA = X - D, A - D
                    cos_xda = np.dot(XD, DA) / (np.linalg.norm(XD) * np.linalg.norm(DA) + 1e-12)
                    if cos_xda >= math.cos(math.radians(110)):
                        continue
                    AD, YA = -DA, Y - A
                    cos_day = np.dot(AD, YA) / (np.linalg.norm(AD) * np.linalg.norm(YA) + 1e-12)
                    if cos_day >= 0.0:
                        continue
                    M[i, j, 1] = True
                    break
                if M[i, j, 1]:
                    break

            # Aromatic.
            if ri in AROMATIC_RING_ATOMS and rj in AROMATIC_RING_ATOMS:
                ring_i = [_pos(i, a, ri) for a in AROMATIC_RING_ATOMS[ri] if _present(i, a, ri)]
                ring_j = [_pos(j, a, rj) for a in AROMATIC_RING_ATOMS[rj] if _present(j, a, rj)]
                if len(ring_i) >= 3 and len(ring_j) >= 3:
                    ci = np.mean(ring_i, axis=0)
                    cj = np.mean(ring_j, axis=0)
                    ni = np.cross(ring_i[1] - ring_i[0], ring_i[2] - ring_i[0])
                    ni /= np.linalg.norm(ni) + 1e-12
                    nj = np.cross(ring_j[1] - ring_j[0], ring_j[2] - ring_j[0])
                    nj /= np.linalg.norm(nj) + 1e-12
                    r12 = cj - ci
                    d = np.linalg.norm(r12)
                    parallel = abs(np.dot(ni, nj))
                    n_avg = (ni + nj) / 2
                    n_avg /= np.linalg.norm(n_avg) + 1e-12
                    d_perp = abs(np.dot(r12, n_avg))
                    d_par = math.sqrt(max(d * d - d_perp * d_perp, 0.0))
                    if parallel > 0.85 and 3.0 <= d <= 4.5 and d_par < 1.5:
                        M[i, j, 3] = True
                    if parallel > 0.85 and 3.5 <= d <= 6.5 and 1.5 <= d_par <= 3.5:
                        M[i, j, 2] = True
                    if parallel < 0.4 and 4.5 <= d <= 7.0:
                        M[i, j, 4] = True

    M[:, :, 1] = M[:, :, 1] | M[:, :, 1].T
    return M


# ----------------------------------------------------------------------
# Geometry helpers for hand-built tests.

def _hexagon(center: np.ndarray, e1: np.ndarray, e2: np.ndarray, radius: float = 1.4) -> np.ndarray:
    """6 vertices of a regular hexagon in the plane spanned by orthonormal e1/e2."""
    return np.stack([
        center + radius * (math.cos(k * math.pi / 3) * e1 + math.sin(k * math.pi / 3) * e2)
        for k in range(6)
    ])


def _phe_residue(ring_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a (14, 3) coords slab and (14,) atom_mask for a PHE residue
    where only the ring atoms (slots 5-10: CG, CD1, CD2, CE1, CE2, CZ) are
    populated. The 6 input positions are placed in atom14 slots 5..10."""
    coords = np.zeros((14, 3), dtype=np.float32)
    atom_mask = np.zeros((14,), dtype=np.int8)
    coords[5:11] = ring_xyz
    atom_mask[5:11] = 1
    return coords, atom_mask


def _two_residue_batch(
    coords_a: np.ndarray, mask_a: np.ndarray, type_a: str,
    coords_b: np.ndarray, mask_b: np.ndarray, type_b: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = np.stack([coords_a, coords_b])[None]
    atom_mask = np.stack([mask_a, mask_b])[None]
    rt = np.array([RESIDUE_TYPE_NAMES.index(type_a), RESIDUE_TYPE_NAMES.index(type_b)])[None]
    return (
        torch.from_numpy(coords).float(),
        torch.from_numpy(rt).long(),
        torch.from_numpy(atom_mask).to(torch.int8),
    )


# ======================================================================
# Layer 1 — Oracle cross-check on real data.

@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_matches_oracle_on_real_data():
    data = np.load(EXAMPLE_NPZ)
    coords_scaled = data["coordinates"].astype(np.float32)
    atom_mask = data["atom_mask"]
    residue_type = data["residue_type"]

    real = atom_mask == 1
    centroid = coords_scaled[real].mean(axis=0)
    coords_scaled = (coords_scaled - centroid) / COORD_SCALE_ANGSTROMS
    coords_ang = coords_scaled * COORD_SCALE_ANGSTROMS  # un-scale to physical Å

    soft = interaction_matrix(
        torch.from_numpy(coords_ang)[None],
        torch.from_numpy(residue_type).long()[None],
        torch.from_numpy(atom_mask).to(torch.int8)[None],
    )[0]                                                                       # (N, N, 6)
    hard_pred = (soft[..., :5] > 0.5).numpy()                                  # (N, N, 5)
    hard_oracle = _oracle(coords_ang, residue_type.astype(int), atom_mask)     # (N, N, 5)

    # Compare per channel; allow up to 1% disagreement (boundary cases where
    # the soft score sits within ~0.1 of 0.5). On 1BRS this should be 0 or
    # very small.
    N = len(residue_type)
    total = N * (N - 1)
    for ch_name, ch in CHANNELS.items():
        if ch == 5:
            continue
        disagree = (hard_pred[..., ch] != hard_oracle[..., ch]).sum()
        assert disagree / total < 0.01, (
            f"channel {ch_name}: {disagree}/{total} pairs disagree between "
            f"differentiable detector and NumPy oracle"
        )

    # Sanity: oracle finds non-trivial counts of multiple interaction types
    # on a real interface — i.e. the system isn't returning all-zeros.
    assert hard_oracle[..., 0].sum() >= 5, "oracle found <5 VDW contacts on 1BRS"
    assert hard_oracle[..., 1].sum() >= 1, "oracle found <1 H-bond on 1BRS"


# ======================================================================
# Layer 2 — Geometric invariants.

def _build_random_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    # Two PHE residues with random ring placements at moderate distance.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    ring_a = _hexagon(np.array([0.0, 0.0, 0.0]), e1, e2)
    ring_b = _hexagon(np.array([0.0, 0.0, 4.0]) + rng.normal(scale=0.5, size=3), e1, e2)
    ca, ma = _phe_residue(ring_a.astype(np.float32))
    cb, mb = _phe_residue(ring_b.astype(np.float32))
    return _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")


def test_translation_invariance():
    coords, rt, mask = _build_random_pair()
    out_a = interaction_matrix(coords, rt, mask)
    out_b = interaction_matrix(coords + torch.tensor([3.7, -1.2, 5.5]), rt, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_rotation_invariance():
    coords, rt, mask = _build_random_pair()
    rng = np.random.default_rng(42)
    A = rng.normal(size=(3, 3))
    R, _ = np.linalg.qr(A)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    R_t = torch.from_numpy(R.astype(np.float32))
    out_a = interaction_matrix(coords, rt, mask)
    out_b = interaction_matrix(coords @ R_t.T, rt, mask)
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_symmetry():
    coords, rt, mask = _build_random_pair()
    out = interaction_matrix(coords, rt, mask)
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-6)


def test_self_interaction_is_none():
    coords, rt, mask = _build_random_pair()
    out = interaction_matrix(coords, rt, mask)
    expected = torch.zeros(6); expected[5] = 1.0
    for i in range(out.shape[1]):
        assert torch.allclose(out[0, i, i], expected, atol=1e-6)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_output_in_unit_interval():
    data = np.load(EXAMPLE_NPZ)
    coords_scaled = data["coordinates"].astype(np.float32)
    real = data["atom_mask"] == 1
    centroid = coords_scaled[real].mean(axis=0)
    coords_ang = (coords_scaled - centroid)  # already in Å (centroid-shifted)
    out = interaction_matrix(
        torch.from_numpy(coords_ang)[None],
        torch.from_numpy(data["residue_type"]).long()[None],
        torch.from_numpy(data["atom_mask"]).to(torch.int8)[None],
    )
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


# ======================================================================
# Layer 3 — Sub-component spot-checks.

def test_band_helper():
    assert abs(_band(torch.tensor(2.0), 1.0, 3.0, k=0.15).item() - 1.0) < 0.01
    assert abs(_band(torch.tensor(1.0), 1.0, 3.0, k=0.15).item() - 0.5) < 0.05
    assert abs(_band(torch.tensor(3.0), 1.0, 3.0, k=0.15).item() - 0.5) < 0.05
    assert _band(torch.tensor(-2.0), 1.0, 3.0, k=0.15).item() < 0.01
    assert _band(torch.tensor(6.0), 1.0, 3.0, k=0.15).item() < 0.01


def test_aromatic_ring_centroid_and_normal():
    """Flat regular hexagon at origin in xy-plane: centroid = origin, normal = ±ẑ."""
    hex_xy = _hexagon(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    ring_pos = torch.from_numpy(hex_xy).float()[None, None]                    # (1, 1, 6, 3)
    ring_mask = torch.ones(1, 1, 6, dtype=torch.bool)
    centroid, normal = _ring_centroid_normal(ring_pos, ring_mask)
    assert torch.allclose(centroid[0, 0], torch.zeros(3), atol=1e-5)
    assert abs(abs(normal[0, 0, 2].item()) - 1.0) < 1e-5
    assert abs(normal[0, 0, 0].item()) < 1e-5
    assert abs(normal[0, 0, 1].item()) < 1e-5


def test_vdw_table_elements():
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for atom_name, slot in ATOM14_SLOT_INDEX[res_name].items():
            assert math.isclose(
                VDW_RADII[res_idx, slot].item(), ELEMENT_VDW[atom_name[0]], abs_tol=1e-5,
            )


def test_donor_parent_table_arg():
    arg_idx = RESIDUE_TYPE_NAMES.index("ARG")
    pairs = HBOND_DONORS_ATOM14[arg_idx][HBOND_DONORS_MASK[arg_idx]].tolist()
    expected = [
        [ATOM14_SLOT_INDEX["ARG"]["N"], ATOM14_SLOT_INDEX["ARG"]["CA"]],
        [ATOM14_SLOT_INDEX["ARG"]["NE"], ATOM14_SLOT_INDEX["ARG"]["CD"]],
        [ATOM14_SLOT_INDEX["ARG"]["NH1"], ATOM14_SLOT_INDEX["ARG"]["CZ"]],
        [ATOM14_SLOT_INDEX["ARG"]["NH2"], ATOM14_SLOT_INDEX["ARG"]["CZ"]],
    ]
    assert pairs == expected


def test_is_aromatic_table():
    aromatic_names = {RESIDUE_TYPE_NAMES[i] for i in range(20) if IS_AROMATIC[i]}
    assert aromatic_names == {"PHE", "TYR", "TRP", "HIS"}


# ======================================================================
# Layer 4 — Hand-built canonical geometries.

def test_hand_built_sandwich():
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
    ring_a = _hexagon(np.zeros(3), e1, e2)
    ring_b = _hexagon(np.array([0.0, 0.0, 3.8]), e1, e2)
    ca, ma = _phe_residue(ring_a); cb, mb = _phe_residue(ring_b)
    coords, rt, mask = _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["sandwich"]] > 0.9, f"sandwich={out[CHANNELS['sandwich']]:.3f}"
    assert out[CHANNELS["parallel_displaced"]] < 0.2
    assert out[CHANNELS["t_shaped"]] < 0.1


def test_hand_built_sandwich_antiparallel_normals():
    """Same sandwich geometry but ring B's basis is flipped so its normal
    points in the opposite direction. Common in real protein structures.
    Regression: previously d_perp collapsed to 0 because n_avg = (n_i+n_j)/2
    was the zero vector, mis-classifying anti-parallel sandwiches as not-
    sandwich."""
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
    ring_a = _hexagon(np.zeros(3), e1, e2)
    # Flip e2 so cross(e1, e2_flipped) → -z (anti-parallel to ring_a's +z normal).
    ring_b = _hexagon(np.array([0.0, 0.0, 3.8]), e1, -e2)
    ca, ma = _phe_residue(ring_a); cb, mb = _phe_residue(ring_b)
    coords, rt, mask = _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["sandwich"]] > 0.9, f"sandwich={out[CHANNELS['sandwich']]:.3f}"
    assert out[CHANNELS["parallel_displaced"]] < 0.2
    assert out[CHANNELS["t_shaped"]] < 0.1


def test_hand_built_parallel_displaced():
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
    # Centroids 4.5 Å apart with d_par = 2.5 Å (centered in [1.5, 3.5] band),
    # d_perp ≈ 3.74 Å.
    ring_a = _hexagon(np.zeros(3), e1, e2)
    ring_b = _hexagon(np.array([2.5, 0.0, math.sqrt(4.5**2 - 2.5**2)]), e1, e2)
    ca, ma = _phe_residue(ring_a); cb, mb = _phe_residue(ring_b)
    coords, rt, mask = _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["parallel_displaced"]] > 0.9, f"pd={out[CHANNELS['parallel_displaced']]:.3f}"
    assert out[CHANNELS["sandwich"]] < 0.2
    assert out[CHANNELS["t_shaped"]] < 0.1


def test_hand_built_t_shape():
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    ring_a = _hexagon(np.zeros(3), e1, e2)                     # in xy-plane (normal = z)
    ring_b = _hexagon(np.array([5.0, 0.0, 0.0]), e1, e3)       # in xz-plane (normal = y)
    ca, ma = _phe_residue(ring_a); cb, mb = _phe_residue(ring_b)
    coords, rt, mask = _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["t_shaped"]] > 0.9, f"t_shape={out[CHANNELS['t_shaped']]:.3f}"
    assert out[CHANNELS["sandwich"]] < 0.1
    assert out[CHANNELS["parallel_displaced"]] < 0.1


def test_hand_built_hbond():
    """ASN ND2 (parent CG) donates to ASP OD1 (parent CG) at d=2.9 Å,
    X-D-A = 160°, D-A-Y = 120°."""
    asn_coords = np.zeros((14, 3), dtype=np.float32)
    asn_mask = np.zeros((14,), dtype=np.int8)
    asn_coords[ATOM14_SLOT_INDEX["ASN"]["CG"]] = [0.0, 0.0, 0.0]
    asn_coords[ATOM14_SLOT_INDEX["ASN"]["ND2"]] = [1.3, 0.0, 0.0]
    asn_mask[ATOM14_SLOT_INDEX["ASN"]["CG"]] = 1
    asn_mask[ATOM14_SLOT_INDEX["ASN"]["ND2"]] = 1

    da_dir = np.array([math.cos(math.radians(20)), math.sin(math.radians(20)), 0.0])
    od1 = asn_coords[ATOM14_SLOT_INDEX["ASN"]["ND2"]] + 2.9 * da_dir
    # D-A-Y = 120°: rotate (-da_dir) by 120° around z.
    c, s = math.cos(math.radians(120)), math.sin(math.radians(120))
    rotz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    asp_cg_dir = rotz @ (-da_dir)
    asp_cg = od1 + 1.3 * asp_cg_dir

    asp_coords = np.zeros((14, 3), dtype=np.float32)
    asp_mask = np.zeros((14,), dtype=np.int8)
    asp_coords[ATOM14_SLOT_INDEX["ASP"]["OD1"]] = od1
    asp_coords[ATOM14_SLOT_INDEX["ASP"]["CG"]] = asp_cg
    asp_mask[ATOM14_SLOT_INDEX["ASP"]["OD1"]] = 1
    asp_mask[ATOM14_SLOT_INDEX["ASP"]["CG"]] = 1

    coords, rt, mask = _two_residue_batch(asn_coords, asn_mask, "ASN", asp_coords, asp_mask, "ASP")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["hbond"]] > 0.9, f"hbond={out[CHANNELS['hbond']]:.3f}"


def test_hand_built_vdw():
    """Two CB carbons at 3.4 Å (= 2 × 1.7 Å)."""
    a = np.zeros((14, 3), dtype=np.float32); ma = np.zeros((14,), dtype=np.int8)
    b = np.zeros((14, 3), dtype=np.float32); mb = np.zeros((14,), dtype=np.int8)
    a[ATOM14_SLOT_INDEX["ALA"]["CB"]] = [0.0, 0.0, 0.0]; ma[ATOM14_SLOT_INDEX["ALA"]["CB"]] = 1
    b[ATOM14_SLOT_INDEX["ALA"]["CB"]] = [3.4, 0.0, 0.0]; mb[ATOM14_SLOT_INDEX["ALA"]["CB"]] = 1
    coords, rt, mask = _two_residue_batch(a, ma, "ALA", b, mb, "ALA")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["vdw"]] > 0.9, f"vdw={out[CHANNELS['vdw']]:.3f}"


def test_far_apart_is_none():
    a = np.zeros((14, 3), dtype=np.float32); ma = np.zeros((14,), dtype=np.int8)
    b = np.zeros((14, 3), dtype=np.float32); mb = np.zeros((14,), dtype=np.int8)
    a[ATOM14_SLOT_INDEX["ALA"]["CB"]] = [0.0, 0.0, 0.0]; ma[ATOM14_SLOT_INDEX["ALA"]["CB"]] = 1
    b[ATOM14_SLOT_INDEX["ALA"]["CB"]] = [50.0, 0.0, 0.0]; mb[ATOM14_SLOT_INDEX["ALA"]["CB"]] = 1
    coords, rt, mask = _two_residue_batch(a, ma, "ALA", b, mb, "ALA")
    out = interaction_matrix(coords, rt, mask)[0, 0, 1]
    assert out[CHANNELS["none"]] > 0.99
    assert out[:5].max() < 1e-3


# ======================================================================
# Layer 5 — Differentiability.

def test_gradient_flows():
    e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, 1.0, 0.0])
    ring_a = _hexagon(np.zeros(3), e1, e2)
    ring_b = _hexagon(np.array([0.0, 0.0, 3.8]), e1, e2)
    ca, ma = _phe_residue(ring_a); cb, mb = _phe_residue(ring_b)
    coords, rt, mask = _two_residue_batch(ca, ma, "PHE", cb, mb, "PHE")
    coords = coords.clone().requires_grad_(True)
    out = interaction_matrix(coords, rt, mask)
    out.sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    # Gradient should reach the ring atoms (slots 5..10).
    ring_grad = coords.grad[0, :, 5:11].abs().sum()
    assert ring_grad > 0
