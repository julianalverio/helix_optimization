"""Buried-surface-area metric via freesasa.

BSA = SASA(helix alone) + SASA(target alone) − SASA(complex), evaluated
on the model's predicted heavy-atom coordinates. Division by two is the
common convention (each interface atom contributes to both sides), so we
report the half-sum directly. Output is in Å².

freesasa structures are built directly from the atom-14 tensors via
`addAtom`, sidestepping the disk round-trip a PDB writer would impose.
The default freesasa classifier assigns van der Waals radii by element
(C 1.70, N 1.55, O 1.52, S 1.80 Å) from atom names; missing-atom slots
are skipped.
"""
from __future__ import annotations

import freesasa
import torch

from twistr.tensors.constants import ATOM14_SLOT_NAMES, RESIDUE_TYPE_NAMES


def _build_freesasa_structure(
    coords: torch.Tensor,           # (N, 14, 3) Å
    atom_mask: torch.Tensor,        # (N, 14) int8
    residue_type: torch.Tensor,     # (N,) long
    include_mask: torch.Tensor,     # (N,) bool — which residues participate
) -> freesasa.Structure:
    """Build a `freesasa.Structure` directly from the atom-14 tensors.
    The default classifier picks radii from atom names — we only pass
    coords + names + residue identity."""
    s = freesasa.Structure()
    for i in range(coords.shape[0]):
        if not bool(include_mask[i]):
            continue
        r_idx = int(residue_type[i].item())
        names = ATOM14_SLOT_NAMES[r_idx]
        for slot in range(14):
            if atom_mask[i, slot] != 1 or not names[slot]:
                continue
            x = float(coords[i, slot, 0].item())
            y = float(coords[i, slot, 1].item())
            z = float(coords[i, slot, 2].item())
            s.addAtom(
                names[slot].ljust(4)[:4],
                RESIDUE_TYPE_NAMES[r_idx],
                str(i + 1).rjust(4)[:4],
                "A",
                x, y, z,
            )
    return s


def buried_surface_area(
    atoms_atom14_ang: torch.Tensor,         # (1, N, 14, 3) Å
    atom_mask: torch.Tensor,                # (1, N, 14) int8
    residue_type: torch.Tensor,             # (1, N) long
    is_helix: torch.Tensor,                 # (1, N) bool
    probe_radius: float = 1.4,
) -> float:
    coords = atoms_atom14_ang[0].detach().cpu()
    mask = atom_mask[0].detach().cpu()
    rtype = residue_type[0].detach().cpu()
    helix = is_helix[0].detach().cpu()
    target = ~helix

    real = torch.ones(coords.shape[0], dtype=torch.bool)
    params = freesasa.Parameters(
        {"probe-radius": probe_radius, "algorithm": freesasa.LeeRichards},
    )

    s_complex = _build_freesasa_structure(coords, mask, rtype, real)
    s_helix = _build_freesasa_structure(coords, mask, rtype, helix)
    s_target = _build_freesasa_structure(coords, mask, rtype, target)

    if (s_complex.nAtoms() == 0 or s_helix.nAtoms() == 0 or s_target.nAtoms() == 0):
        return 0.0

    sasa_complex = freesasa.calc(s_complex, params).totalArea()
    sasa_helix = freesasa.calc(s_helix, params).totalArea()
    sasa_target = freesasa.calc(s_target, params).totalArea()
    return float((sasa_helix + sasa_target - sasa_complex) / 2.0)
