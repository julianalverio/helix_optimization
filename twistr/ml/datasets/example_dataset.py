from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from twistr.ml.constants import COORD_SCALE_ANGSTROMS


def random_rotation_matrix(generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform-on-SO(3) rotation matrix. QR of a 3×3 Gaussian gives Haar-uniform
    on O(3) once R's diagonal signs are canonicalised; flipping a column when
    det Q < 0 restricts to SO(3)."""
    M = torch.randn(3, 3, generator=generator)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diagonal(R))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class ExamplesDataset(Dataset):
    """Loads a Module 3 example .npz and returns the subset of fields the model
    needs. Coordinates are upcast to fp32, zero-centered on the centroid of real
    heavy atoms (atom_mask == 1), then divided by 10 Å so a unit ≈ 1 nm. All
    masked-slot coordinates inherit the same translation/scaling — downstream
    code is expected to gate on atom_mask before using them.

    atom_mask values (int8, per (residue, atom14_slot)) — see
    data/module2_instructions.md for the canonical spec:
        1  — real heavy atom present in the source structure (non-zero occupancy).
        0  — no usable atom at this slot, for any reason: slot not part of this
             residue type's canonical atom set (e.g. CG2 on ALA), OR atom missing
             from the deposited structure, OR zero occupancy after altloc
             resolution. The three cases are intentionally collapsed.
       -1  — residue does not exist at this position. Module-2 raw tensors use
             this to chain-pad up to N_max; module-3 example files strip those
             rows so -1 normally won't appear when reading from disk. The ML
             batch collate (`pad_collate` in datamodule.py) re-introduces -1 to
             pad variable-length proteins to the batch's max N. Whenever this
             value appears, every other per-residue field at the same residue is
             also non-existent and must be excluded from any computation. The
             companion (B, N) bool `padding_mask` (True = real residue) is the
             canonical gate; the -1 here is a redundant atom-grain signal so
             existing `atom_mask == 1` checks naturally skip non-residue rows.

    If `random_rotate` is True, a fresh uniform-random SO(3) rotation is
    sampled per `__getitem__` call and applied after centering. Distances,
    angles, and dihedrals are invariant; the conditioning translation / 6D-
    frame features rotate equivariantly to drive augmentation."""

    def __init__(self, paths: list[Path | str], random_rotate: bool = False):
        self.paths = [Path(p) for p in paths]
        self.random_rotate = random_rotate

    def __len__(self) -> int:
        return len(self.paths)

    def length(self, idx: int) -> int:
        """N (residue count) of example `idx` without loading the full file —
        only the chain_slot array is decompressed from the npz."""
        with np.load(self.paths[idx]) as f:
            return int(f["chain_slot"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[idx])
        coords = data["coordinates"].astype(np.float32)
        atom_mask = data["atom_mask"]
        residue_type = data["residue_type"]
        chain_slot = data["chain_slot"]
        is_helix = data["is_helix"]
        is_interface_residue = data["is_interface_residue"]

        real = atom_mask == 1
        centroid = coords[real].mean(axis=0)
        coords = (coords - centroid) / COORD_SCALE_ANGSTROMS

        coords_t = torch.from_numpy(coords)
        if self.random_rotate:
            coords_t = coords_t @ random_rotation_matrix().T

        return {
            "coordinates": coords_t,
            "atom_mask": torch.from_numpy(atom_mask),
            "residue_type": torch.from_numpy(residue_type).long(),
            "chain_slot": torch.from_numpy(chain_slot).long(),
            "is_helix": torch.from_numpy(np.ascontiguousarray(is_helix)),
            "is_interface_residue": torch.from_numpy(np.ascontiguousarray(is_interface_residue)),
        }
