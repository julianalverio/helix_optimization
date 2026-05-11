"""Differentiable detector for five protein-protein interaction types:

  channel 0: VDW contact            channel 1: hydrogen bond
  channel 2: parallel-displaced π   channel 3: sandwich π   channel 4: T-shaped π
  channel 5: no interaction (= 1 − max of channels 0–4)

Output shape is (B, N, N, 6). The matrix is symmetric (hbond directionality
is collapsed via max), the diagonal is forced to one-hot "none", and every
score is a smooth product of sigmoid bands and cosine terms — the whole
tensor is backward-differentiable w.r.t. coordinates. Used as both a target
generator on ground-truth coords (apply argmax) and a CE-loss term on
predicted coords.

Coordinates must be in physical Angstroms — the dataset's /10 normalization
must be undone before passing in. Thresholds inside use literature values in Å.

Neighbor search is dense: a (B, N, N, 14, 14) atom-pair distance is a few MB
at typical N≤200, so KDTree's non-differentiable indexing buys nothing here."""
from __future__ import annotations

import math

import torch

from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS
from twistr.tensors.constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_NAMES

CHANNELS: dict[str, int] = {
    "vdw": 0, "hbond": 1, "parallel_displaced": 2,
    "sandwich": 3, "t_shaped": 4, "none": 5,
}

# The conditioning feature mirrors the 6-channel target and adds two binary
# bits: an "augmentation mask" (=1 on cells flagged by the residue-masking
# augmentation steps; "user did not specify this entry") and a "padding mask"
# (=1 on cells where either residue is padding from pad_collate).
CONDITIONING_CHANNELS: dict[str, int] = {
    "vdw": 0, "hbond": 1, "parallel_displaced": 2,
    "sandwich": 3, "t_shaped": 4, "none": 5,
    "augmentation_mask": 6, "padding_mask": 7,
}

# Per-interaction band constants. Single source of truth — the soft detector
# below and the geometric losses (twistr/pipeline/losses/interaction_geometry.py)
# both use these so the binary GT and the loss bands stay in lockstep.
VDW_BAND_LO_OFFSET_A = -0.4
VDW_BAND_HI_OFFSET_A = 0.5

HBOND_DIST_LO_A = 2.5
HBOND_DIST_HI_A = 3.6
HBOND_COS_XDA_THRESH = math.cos(math.radians(110.0))   # X-D-A angle > 110° ⇔ cos < this
HBOND_COS_DAY_THRESH = 0.0                              # D-A-Y angle > 90° ⇔ cos < 0

# Aromatic sub-type bands (Å for distances, cosine for `parallel`).
AROM_PARA_LO = 0.85
AROM_PARA_HI = 0.40   # T-shape requires parallel ≤ this
AROM_SANDWICH_D = (3.0, 4.5)
AROM_SANDWICH_DPAR_HI = 1.5
AROM_PD_D = (3.5, 6.5)
AROM_PD_DPAR = (1.5, 3.5)
AROM_T_D = (4.5, 7.0)


# Element-wise VDW radii (Å). Sourced from the RDKit / OpenFold table cited
# at twistr/external/Protenix/protenix/data/constants.py:215.
ELEMENT_VDW: dict[str, float] = {"C": 1.7, "N": 1.6, "O": 1.55, "S": 1.8}

_BB_DONOR = ("N", "CA")
_BB_ACCEPTOR = ("O", "C")

# Donors are heavy atoms bonded to a donatable H; the parent atom X is the
# heavy-atom neighbor used to evaluate the X-D-A angle (H lies on the ray
# opposite from X). Backbone N is donor for every residue except PRO. Side-
# chain entries follow PDBe contacts / Stickle 1992.
HBOND_DONORS: dict[str, list[tuple[str, str]]] = {
    "ALA": [_BB_DONOR],
    "ARG": [_BB_DONOR, ("NE", "CD"), ("NH1", "CZ"), ("NH2", "CZ")],
    "ASN": [_BB_DONOR, ("ND2", "CG")],
    "ASP": [_BB_DONOR],
    "CYS": [_BB_DONOR],
    "GLN": [_BB_DONOR, ("NE2", "CD")],
    "GLU": [_BB_DONOR],
    "GLY": [_BB_DONOR],
    "HIS": [_BB_DONOR, ("ND1", "CG"), ("NE2", "CD2")],
    "ILE": [_BB_DONOR],
    "LEU": [_BB_DONOR],
    "LYS": [_BB_DONOR, ("NZ", "CE")],
    "MET": [_BB_DONOR],
    "PHE": [_BB_DONOR],
    "PRO": [],
    "SER": [_BB_DONOR, ("OG", "CB")],
    "THR": [_BB_DONOR, ("OG1", "CB")],
    "TRP": [_BB_DONOR, ("NE1", "CD1")],
    "TYR": [_BB_DONOR, ("OH", "CZ")],
    "VAL": [_BB_DONOR],
}

# Acceptors are heavy atoms with a free lone pair; parent atom Y is the
# heavy-atom neighbor used to evaluate the D-A-Y angle.
HBOND_ACCEPTORS: dict[str, list[tuple[str, str]]] = {
    "ALA": [_BB_ACCEPTOR],
    "ARG": [_BB_ACCEPTOR],
    "ASN": [_BB_ACCEPTOR, ("OD1", "CG")],
    "ASP": [_BB_ACCEPTOR, ("OD1", "CG"), ("OD2", "CG")],
    "CYS": [_BB_ACCEPTOR],
    "GLN": [_BB_ACCEPTOR, ("OE1", "CD")],
    "GLU": [_BB_ACCEPTOR, ("OE1", "CD"), ("OE2", "CD")],
    "GLY": [_BB_ACCEPTOR],
    "HIS": [_BB_ACCEPTOR, ("ND1", "CG"), ("NE2", "CD2")],
    "ILE": [_BB_ACCEPTOR],
    "LEU": [_BB_ACCEPTOR],
    "LYS": [_BB_ACCEPTOR],
    "MET": [_BB_ACCEPTOR],
    "PHE": [_BB_ACCEPTOR],
    "PRO": [_BB_ACCEPTOR],
    "SER": [_BB_ACCEPTOR, ("OG", "CB")],
    "THR": [_BB_ACCEPTOR, ("OG1", "CB")],
    "TRP": [_BB_ACCEPTOR],
    "TYR": [_BB_ACCEPTOR, ("OH", "CZ")],
    "VAL": [_BB_ACCEPTOR],
}

# Aromatic ring atoms. Atoms 0, 1, 2 must span the ring plane (used for the
# cross-product normal) — true for every entry below. PHE/TYR use the 6-ring;
# TRP uses the indole 6-ring; HIS uses the imidazole 5-ring (5 atoms only,
# slot 5 padded).
AROMATIC_RING_ATOMS: dict[str, tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
}

# Packing atoms: aliphatic stub carbons (non-functional sp3 Cs) ∪ aromatic
# ring atoms. Used by the packing-neighbor loss to identify which atoms of
# each residue should have several heavy-atom neighbors within the VDW band.
#
# Exclusions:
#   - Functional-group heteroatoms (OG, SG, NZ, guanidinium, OD/ND, OE/NE)
#     and the carbons bonded to them (Asp/Asn CG, Glu/Gln CD, Met SD).
#   - Pro CD: bonded to backbone N, not aliphatic in the same sense.
PACKING_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD"),
    "ASN": ("CB",),
    "ASP": ("CB",),
    "CYS": ("CB",),
    "GLN": ("CB", "CG"),
    "GLU": ("CB", "CG"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE"),
    "MET": ("CB", "CG", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG"),
    "SER": ("CB",),
    "THR": ("CB", "CG2"),
    "TRP": ("CB", "CG", "CD1", "NE1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "VAL": ("CB", "CG1", "CG2"),
}

_RING_SLOTS = 6  # max ring atoms across residues


# ----------------------------------------------------------------------
# Module-load: build atom14-indexed tensors. Mirrors the chi_angles pattern.

def _build_vdw_radii_table() -> torch.Tensor:
    table = torch.zeros((20, 14), dtype=torch.float32)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for atom_name, slot in ATOM14_SLOT_INDEX[res_name].items():
            table[res_idx, slot] = ELEMENT_VDW[atom_name[0]]
    return table


def _build_hbond_table(
    spec: dict[str, list[tuple[str, str]]], max_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = torch.zeros((20, max_n, 2), dtype=torch.long)
    mask = torch.zeros((20, max_n), dtype=torch.bool)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for k, (atom_name, parent_name) in enumerate(spec[res_name]):
            table[res_idx, k, 0] = ATOM14_SLOT_INDEX[res_name][atom_name]
            table[res_idx, k, 1] = ATOM14_SLOT_INDEX[res_name][parent_name]
            mask[res_idx, k] = True
    return table, mask


def _build_packing_atom_table() -> torch.Tensor:
    mask = torch.zeros((20, 14), dtype=torch.bool)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        for atom_name in PACKING_ATOMS[res_name]:
            mask[res_idx, ATOM14_SLOT_INDEX[res_name][atom_name]] = True
    return mask


def _build_aromatic_table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slots = torch.zeros((20, _RING_SLOTS), dtype=torch.long)
    atom_mask = torch.zeros((20, _RING_SLOTS), dtype=torch.bool)
    is_aromatic = torch.zeros((20,), dtype=torch.bool)
    for res_idx, res_name in enumerate(RESIDUE_TYPE_NAMES):
        if res_name not in AROMATIC_RING_ATOMS:
            continue
        is_aromatic[res_idx] = True
        for k, atom_name in enumerate(AROMATIC_RING_ATOMS[res_name]):
            slots[res_idx, k] = ATOM14_SLOT_INDEX[res_name][atom_name]
            atom_mask[res_idx, k] = True
    return slots, atom_mask, is_aromatic


_MAX_DONORS = max(len(v) for v in HBOND_DONORS.values())
_MAX_ACCEPTORS = max(len(v) for v in HBOND_ACCEPTORS.values())

VDW_RADII: torch.Tensor = _build_vdw_radii_table()
HBOND_DONORS_ATOM14, HBOND_DONORS_MASK = _build_hbond_table(HBOND_DONORS, _MAX_DONORS)
HBOND_ACCEPTORS_ATOM14, HBOND_ACCEPTORS_MASK = _build_hbond_table(HBOND_ACCEPTORS, _MAX_ACCEPTORS)
AROMATIC_RING_SLOTS, AROMATIC_ATOM_MASK, IS_AROMATIC = _build_aromatic_table()
IS_PACKING_ATOM: torch.Tensor = _build_packing_atom_table()


# ----------------------------------------------------------------------
# Math helpers.

def _band(
    x: torch.Tensor, lo: float | torch.Tensor, hi: float | torch.Tensor, k: float = 0.15,
) -> torch.Tensor:
    """Smooth indicator of x ∈ [lo, hi]. ≈ 1 inside, ≈ 0 outside, crosses 0.5
    at the boundaries. k = edge softness in the same units as x."""
    return torch.sigmoid((x - lo) / k) * torch.sigmoid((hi - x) / k)


def _safe_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / _safe_norm(v, eps)


def _gather_atom14(coords: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """coords: (B, N, 14, 3); slots: (B, N, K) long. Returns (B, N, K, 3)."""
    B, N, K = slots.shape
    bi = torch.arange(B, device=coords.device)[:, None, None].expand(B, N, K)
    ni = torch.arange(N, device=coords.device)[None, :, None].expand(B, N, K)
    return coords[bi, ni, slots]


def _ring_centroid_normal(
    ring_pos: torch.Tensor, ring_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ring_pos: (..., 6, 3); ring_mask: (..., 6) bool. Returns (centroid,
    normal) where centroid is the masked mean and normal is the **best-fit
    plane normal** through all populated ring atoms via the eigenvector of
    the smallest eigenvalue of the (3, 3) atom covariance matrix.

    Robust to ring puckering (atoms ~0.05 Å off perfect planarity) — a
    3-atom cross product would be sensitive to whichever 3 atoms it picks.
    Eigh's gradient is numerically stable here because the planar eigenvalue
    (~0.001 Å²) is well-separated from the in-plane eigenvalues (~ring
    radius² ≈ 2 Å²); the `1/(λ_i - λ_j)` terms in eigenvector backprop
    don't blow up.

    Sign-aligned with the 3-atom cross product so the normal direction is
    consistent across calls and the |n_i · n_j| signal is stable."""
    m = ring_mask.unsqueeze(-1).to(ring_pos.dtype)
    n_atoms = m.sum(dim=-2).clamp_min(1.0)                                    # (..., 1)
    centroid = (ring_pos * m).sum(dim=-2) / n_atoms                           # (..., 3)

    # Covariance of populated atoms about the centroid: C = Xᵀ M X / Σm
    # where M is the diagonal mask. Masked atoms contribute zero rows.
    centered = (ring_pos - centroid.unsqueeze(-2)) * m                        # (..., 6, 3)
    cov = centered.transpose(-1, -2) @ centered                               # (..., 3, 3)
    cov = cov / n_atoms.unsqueeze(-1)
    # Asymmetric diagonal regularization breaks eigenvalue degeneracy in
    # pathological cases (non-aromatic residues — cov is zero — would give
    # degenerate eigenvalues, and eigh's backward through 1/(λ_i - λ_j)
    # produces NaN gradients that pollute the entire batch even though the
    # output gets masked out downstream by `pair_arom`). For real aromatic
    # rings, λ_min is ~0.01 Å² and λ_in_plane is ~2 Å², so adding
    # diag([1e-4, 2e-4, 3e-4]) Å² is well below physical scale and
    # well-separated from those eigenvalues.
    reg = torch.tensor([1e-4, 2e-4, 3e-4], dtype=cov.dtype, device=cov.device)
    cov = cov + torch.diag_embed(reg.expand(*cov.shape[:-2], 3))
    # eigh returns eigenvalues in ASCENDING order; smallest = plane normal.
    _, eigvecs = torch.linalg.eigh(cov)                                       # (..., 3, 3)
    normal_eig = eigvecs[..., 0]                                              # (..., 3)

    # Sign-align with the 3-atom cross product (deterministic up to which
    # 3 atoms we pick) so the normal direction is stable across calls.
    v1 = ring_pos[..., 1, :] - ring_pos[..., 0, :]
    v2 = ring_pos[..., 2, :] - ring_pos[..., 0, :]
    cross_normal = torch.linalg.cross(v1, v2, dim=-1)
    sign = torch.sign((normal_eig * cross_normal).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    normal = _safe_normalize(normal_eig * sign)
    return centroid, normal


# ----------------------------------------------------------------------
# Per-interaction scorers.

def _vdw_score(
    coords: torch.Tensor, residue_type: torch.Tensor, atom_present: torch.Tensor,
) -> torch.Tensor:
    """(B, N, N) VDW score per residue pair. Sidechain atoms only (slots ≥ 4) —
    backbone atoms are excluded so sequence-adjacent residues don't get a vdw
    label from CA-CA / N-O backbone proximity. GLY has no sidechain → contributes
    no vdw signal. Atom-pair scores aggregated by max."""
    vdw_r = VDW_RADII.to(coords.device)[residue_type]                          # (B, N, 14)

    sidechain = torch.zeros(14, dtype=torch.bool, device=coords.device)
    sidechain[4:] = True                                                        # 0=N, 1=CA, 2=C, 3=O are backbone
    sidechain_present = atom_present & sidechain

    ca = coords[:, :, None, :, None, :]                                        # (B, N, 1, 14, 1, 3)
    cb = coords[:, None, :, None, :, :]                                        # (B, 1, N, 1, 14, 3)
    d = (ca - cb).norm(dim=-1)                                                 # (B, N, N, 14, 14)

    r_a = vdw_r[:, :, None, :, None]
    r_b = vdw_r[:, None, :, None, :]
    r_sum = r_a + r_b
    s = _band(d, r_sum + VDW_BAND_LO_OFFSET_A, r_sum + VDW_BAND_HI_OFFSET_A, k=0.15)

    pa = sidechain_present[:, :, None, :, None].float()
    pb = sidechain_present[:, None, :, None, :].float()
    s = s * pa * pb
    return s.amax(dim=(-1, -2))


def _hbond_score(
    coords: torch.Tensor, residue_type: torch.Tensor, atom_present: torch.Tensor,
) -> torch.Tensor:
    """(B, N, N) hydrogen-bond score, symmetrized over donor↔acceptor direction."""
    donor_table = HBOND_DONORS_ATOM14.to(coords.device)[residue_type]          # (B, N, max_d, 2)
    donor_typed = HBOND_DONORS_MASK.to(coords.device)[residue_type]            # (B, N, max_d)
    accept_table = HBOND_ACCEPTORS_ATOM14.to(coords.device)[residue_type]
    accept_typed = HBOND_ACCEPTORS_MASK.to(coords.device)[residue_type]

    D_slots = donor_table[..., 0]
    X_slots = donor_table[..., 1]
    A_slots = accept_table[..., 0]
    Y_slots = accept_table[..., 1]

    D = _gather_atom14(coords, D_slots)
    X = _gather_atom14(coords, X_slots)
    A = _gather_atom14(coords, A_slots)
    Y = _gather_atom14(coords, Y_slots)

    donor_valid = donor_typed & atom_present.gather(-1, D_slots) & atom_present.gather(-1, X_slots)
    accept_valid = accept_typed & atom_present.gather(-1, A_slots) & atom_present.gather(-1, Y_slots)

    Di = D[:, :, None, :, None, :]                                             # (B, N, 1, max_d, 1, 3)
    Xi = X[:, :, None, :, None, :]
    Aj = A[:, None, :, None, :, :]                                             # (B, 1, N, 1, max_a, 3)
    Yj = Y[:, None, :, None, :, :]

    DA = Aj - Di
    XD = Xi - Di
    AD = -DA
    YA = Yj - Aj

    d_DA = DA.norm(dim=-1)                                                     # (B, N, N, max_d, max_a)
    band_dist = _band(d_DA, HBOND_DIST_LO_A, HBOND_DIST_HI_A, k=0.15)

    cos_xda = (XD * DA).sum(-1) / (_safe_norm(XD).squeeze(-1) * _safe_norm(DA).squeeze(-1))
    cos_day = (AD * YA).sum(-1) / (_safe_norm(AD).squeeze(-1) * _safe_norm(YA).squeeze(-1))

    band_xda = torch.sigmoid((HBOND_COS_XDA_THRESH - cos_xda) / 0.1)
    band_day = torch.sigmoid((HBOND_COS_DAY_THRESH - cos_day) / 0.1)

    s = band_dist * band_xda * band_day
    valid_pair = donor_valid[:, :, None, :, None] & accept_valid[:, None, :, None, :]
    s = s * valid_pair.float()

    ij = s.amax(dim=(-1, -2))                                                   # (B, N, N) — i donates to j
    return torch.maximum(ij, ij.transpose(-1, -2))


def _aromatic_scores(
    coords: torch.Tensor, residue_type: torch.Tensor, atom_present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (parallel_displaced, sandwich, t_shaped), each (B, N, N)."""
    ring_slots = AROMATIC_RING_SLOTS.to(coords.device)[residue_type]            # (B, N, 6)
    ring_typed = AROMATIC_ATOM_MASK.to(coords.device)[residue_type]             # (B, N, 6)
    is_arom = IS_AROMATIC.to(coords.device)[residue_type]                       # (B, N)

    ring_pos = _gather_atom14(coords, ring_slots)
    eff_mask = ring_typed & atom_present.gather(-1, ring_slots)
    arom_valid = is_arom & eff_mask[..., 0] & eff_mask[..., 1] & eff_mask[..., 2]

    centroid, normal = _ring_centroid_normal(ring_pos, eff_mask)                # (B, N, 3) each

    c_i = centroid[:, :, None, :]
    c_j = centroid[:, None, :, :]
    n_i = normal[:, :, None, :]
    n_j = normal[:, None, :, :]

    r12 = c_j - c_i
    d = r12.norm(dim=-1)                                                        # (B, N, N)
    n_dot = (n_i * n_j).sum(-1, keepdim=True)
    parallel = n_dot.squeeze(-1).abs()
    # Sign-align n_j with n_i before averaging so anti-parallel ring pairs
    # (a common π-stacking orientation) don't collapse n_avg to zero.
    sign = torch.sign(n_dot)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n_avg = _safe_normalize((n_i + sign * n_j) / 2)
    d_perp = (r12 * n_avg).sum(-1).abs()
    d_par = (d.pow(2) - d_perp.pow(2)).clamp_min(1e-8).sqrt()

    sandwich = (
        torch.sigmoid((parallel - AROM_PARA_LO) / 0.05)
        * _band(d, AROM_SANDWICH_D[0], AROM_SANDWICH_D[1], k=0.15)
        * torch.sigmoid((AROM_SANDWICH_DPAR_HI - d_par) / 0.2)
    )
    parallel_displaced = (
        torch.sigmoid((parallel - AROM_PARA_LO) / 0.05)
        * _band(d, AROM_PD_D[0], AROM_PD_D[1], k=0.15)
        * _band(d_par, AROM_PD_DPAR[0], AROM_PD_DPAR[1], k=0.2)
    )
    t_shaped = (
        torch.sigmoid((AROM_PARA_HI - parallel) / 0.05)
        * _band(d, AROM_T_D[0], AROM_T_D[1], k=0.15)
    )

    pair_arom = (arom_valid[:, :, None] & arom_valid[:, None, :]).float()
    return parallel_displaced * pair_arom, sandwich * pair_arom, t_shaped * pair_arom


def interaction_matrix(
    coords: torch.Tensor, residue_type: torch.Tensor, atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Soft (B, N, N, 6) interaction tensor. Coords in Angstroms. See module
    docstring for channel order and the unit contract."""
    atom_present = (atom_mask == 1)                                             # (B, N, 14)

    vdw = _vdw_score(coords, residue_type, atom_present)
    hbond = _hbond_score(coords, residue_type, atom_present)
    pd, sandwich, t_shape = _aromatic_scores(coords, residue_type, atom_present)

    interact = torch.stack([vdw, hbond, pd, sandwich, t_shape], dim=-1)         # (B, N, N, 5)

    B, N = residue_type.shape
    not_diag = (~torch.eye(N, device=coords.device, dtype=torch.bool))[None, :, :, None].float()
    interact = interact * not_diag

    none_score = (1.0 - interact.amax(dim=-1)).clamp(0.0, 1.0)
    return torch.cat([interact, none_score.unsqueeze(-1)], dim=-1)


# ----------------------------------------------------------------------
# Conditioning feature: clean target + noisy augmented input.

def clean_interaction_matrix(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, N, N, 6) **binary** ground-truth interaction matrix with channels
    [vdw, hbond, parallel_displaced, sandwich, t_shaped, none]. Values are
    exactly 0.0 or 1.0. Multi-label: each channel is an independent indicator
    of "does this residue pair form that interaction type"; a pair can have
    vdw=1 AND hbond=1 simultaneously.

    This is the shared source of truth between the feature pipeline (where
    it gets noised + smoothed into the conditioning input) and the loss
    function (where each cell tells the loss which differentiable scoring
    function to fire on the predicted structure). Not backwards-differentiable
    by design — it's a label tensor, not a quantity.

    Channel definitions, derived by thresholding the soft detector at 0.5:
      vdw                = sidechain VDW band fires for at least one atom pair
      hbond              = some donor-acceptor pair satisfies distance + angle bands
      parallel_displaced = aromatic rings parallel, in PD distance/offset band
      sandwich           = aromatic rings parallel, in sandwich distance/offset band
      t_shaped           = aromatic rings perpendicular, in T-shape distance band
      none               = 1 iff every other channel is 0
    """
    coords_ang = batch["coordinates"] * COORD_SCALE_ANGSTROMS
    soft = interaction_matrix(coords_ang, batch["residue_type"], batch["atom_mask"])
    b_vdw = soft[..., 0] > 0.5
    b_hbond = soft[..., 1] > 0.5
    b_pd = soft[..., 2] > 0.5
    b_sandwich = soft[..., 3] > 0.5
    b_t_shaped = soft[..., 4] > 0.5
    b_none = ~(b_vdw | b_hbond | b_pd | b_sandwich | b_t_shaped)
    return torch.stack(
        [b_vdw, b_hbond, b_pd, b_sandwich, b_t_shaped, b_none], dim=-1,
    ).to(soft.dtype)


def _beta_sample(
    alpha: float, beta: float, shape: tuple[int, ...],
    device: torch.device, dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample Beta(alpha, beta). torch.distributions.Beta doesn't accept a
    Generator, so when one is supplied we draw a seed from it and run the
    Beta sample inside `fork_rng` — this isolates the seed override to the
    Beta call so we don't leak it into surrounding code (e.g., dropout in
    the model forward, which would otherwise couple to our chosen seed).
    Forks both CPU and CUDA states."""
    a = torch.full(shape, alpha, device=device, dtype=dtype)
    b = torch.full(shape, beta, device=device, dtype=dtype)
    if generator is None:
        return torch.distributions.Beta(a, b).sample()
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=device).item())
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return torch.distributions.Beta(a, b).sample()


def _apply_interaction_matrix_noise(
    clean: torch.Tensor,
    is_interface_residue: torch.Tensor,
    padding_mask: torch.Tensor,
    cfg,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the noise pipeline. Returns (B, N, N, 8) — 6 noisy probability
    channels + augmentation_mask (channel 6) + padding_mask (channel 7).
    All noise is symmetric in (i, j). Input `clean` is the binary
    (B, N, N, 6) ground-truth label tensor.

    Steps:
      1. Whole-residue augmentation mask: count drawn from
         interacting_residue_mask_count_min..max on residues with any
         interaction, plus per-non-interface residue rate
         non_interface_residue_mask_rate. Affected pair cells get random
         uniform modality values + augmentation_mask=1.
      2. For each example b: u_b ~ U(0, 1). p_01_b = u_b * max_zero_to_one_flip_rate;
         p_10_b = u_b * max_one_to_zero_flip_rate.
      3. For each pristine upper-triangle cell (i, j) and each of 6 channels c:
         flip clean[c]=0 → 1 with prob p_01_b; flip clean[c]=1 → 0 with prob p_10_b.
      4. Convert noisy_binary → soft probability by Beta sampling:
            0 → Beta(neg_mean*neg_conf, (1-neg_mean)*neg_conf)
            1 → Beta(pos_mean*pos_conf, (1-pos_mean)*pos_conf).
      5. Mirror upper triangle to lower triangle (symmetry).
      6. padding_mask channel: 1 iff i or j is padded.
      7. Diagonal forced to (0,0,0,0,0,1) modality, augmentation_mask=0,
         padding_mask=1 if i is padded else 0."""
    B, N, _, C = clean.shape
    device = clean.device
    dtype = clean.dtype
    none_idx = CHANNELS["none"]

    def _rand(*shape) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, device=device, dtype=dtype)

    def _randint(low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=generator, device=device).item())

    out = clean.clone()
    aug_mask = torch.zeros((B, N, N), device=device, dtype=dtype)
    not_diag = ~torch.eye(N, device=device, dtype=torch.bool)
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    pos_alpha = cfg.positive_beta_mean * cfg.positive_beta_confidence
    pos_beta = (1.0 - cfg.positive_beta_mean) * cfg.positive_beta_confidence
    neg_alpha = cfg.negative_beta_mean * cfg.negative_beta_confidence
    neg_beta = (1.0 - cfg.negative_beta_mean) * cfg.negative_beta_confidence

    for b in range(B):
        modified = torch.zeros((N, N), device=device, dtype=torch.bool)
        residue_masked = torch.zeros((N,), device=device, dtype=torch.bool)
        is_real = padding_mask[b]                                                # (N,)
        real_pair = is_real[:, None] & is_real[None, :]                          # (N, N)

        # Padded cells have clean[..., none] = 1, so is_positive is False at
        # padded positions — no extra padding gate needed for the residue-mask
        # candidate selection.
        is_positive = (clean[b, :, :, none_idx] < 0.5)                           # (N, N)
        any_positive_per_residue = is_positive.any(dim=-1)                       # (N,)

        # Step 1a: whole-residue mask drawn from residues with any positive interaction.
        interacting_idx = torch.nonzero(any_positive_per_residue, as_tuple=False).flatten()
        if interacting_idx.numel() > 0:
            count = _randint(cfg.interacting_residue_mask_count_min,
                             cfg.interacting_residue_mask_count_max + 1)
            count = min(count, interacting_idx.numel())
            if count > 0:
                perm = torch.randperm(interacting_idx.numel(), generator=generator, device=device)
                residue_masked[interacting_idx[perm[:count]]] = True

        # Step 1b: whole-residue mask on non-interface real residues.
        non_interface = (~is_interface_residue[b].bool()) & is_real
        if cfg.non_interface_residue_mask_rate > 0:
            roll = _rand(int(is_real.sum().item()))
            residue_masked[is_real] = residue_masked[is_real] | (non_interface[is_real] & (roll < cfg.non_interface_residue_mask_rate))

        if residue_masked.any():
            entries = (residue_masked[:, None] | residue_masked[None, :]) & not_diag & real_pair
            pair_idx = torch.nonzero(entries & upper, as_tuple=False)
            if pair_idx.numel() > 0:
                rows, cols = pair_idx[:, 0], pair_idx[:, 1]
                rand_vals = _rand(rows.numel(), C)
                out[b, rows, cols] = rand_vals
                out[b, cols, rows] = rand_vals
                aug_mask[b, rows, cols] = 1.0
                aug_mask[b, cols, rows] = 1.0
                modified[rows, cols] = True
                modified[cols, rows] = True

        # Step 2 & 3: per-example bit-flip on remaining pristine cells.
        pristine = (~modified) & upper & real_pair
        if pristine.any():
            pair_idx = torch.nonzero(pristine, as_tuple=False)
            i_idx, j_idx = pair_idx[:, 0], pair_idx[:, 1]
            K = pair_idx.shape[0]
            cells = out[b, i_idx, j_idx]                                         # (K, C), exactly 0/1

            u_b = _rand(1).item()
            p_01 = u_b * cfg.max_zero_to_one_flip_rate
            p_10 = u_b * cfg.max_one_to_zero_flip_rate
            flip_roll = _rand(K, C)
            flip_01 = (cells == 0) & (flip_roll < p_01)
            flip_10 = (cells == 1) & (flip_roll < p_10)
            noisy_binary = torch.where(flip_01 | flip_10, 1.0 - cells, cells)

            # Step 4: Beta sample per cell per channel.
            beta_pos = _beta_sample(pos_alpha, pos_beta, (K, C), device, dtype, generator)
            beta_neg = _beta_sample(neg_alpha, neg_beta, (K, C), device, dtype, generator)
            soft = torch.where(noisy_binary == 1, beta_pos, beta_neg)

            # Step 5: write upper triangle and mirror.
            out[b, i_idx, j_idx] = soft
            out[b, j_idx, i_idx] = soft

    # Step 6: padding mask channel — 1 iff i or j is padded.
    pad_mask = (~padding_mask).to(dtype)                                         # (B, N), 1 where padded
    pad_chan = (pad_mask[:, :, None] + pad_mask[:, None, :]).clamp_max(1.0)      # (B, N, N)

    out = torch.cat([out, aug_mask.unsqueeze(-1), pad_chan.unsqueeze(-1)], dim=-1)

    # Step 7: force diagonal. Modality = (0,...,0,1), augmentation_mask = 0.
    # padding_mask on the diagonal is naturally 1 iff i is padded (from pad_chan).
    eye = torch.eye(N, device=device, dtype=torch.bool)
    diag_value = torch.zeros((B, N, N, C + 2), device=device, dtype=dtype)
    diag_value[..., none_idx] = 1.0
    diag_value[..., C + 1] = pad_chan                                            # preserve padding_mask on diagonal
    return torch.where(eye[None, :, :, None], diag_value, out)


def conditioning_interaction_matrix(
    clean: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(B, N, N, 8) noisy interaction matrix used as the model's lead-
    optimization conditioning input. Channels 0–5 are noisy probability
    estimates of the six modalities; channel 6 is the augmentation mask
    (1 on cells flagged by residue-masking augmentation steps); channel 7
    is the padding mask (1 on cells where either residue is padding).
    Takes the precomputed clean target (compute it once via
    `clean_interaction_matrix(batch)` and reuse for both this conditioning
    input and the loss target) so the soft-probability detector isn't run
    twice per step."""
    return _apply_interaction_matrix_noise(
        clean, batch["is_interface_residue"], batch["padding_mask"], cfg, generator,
    )
