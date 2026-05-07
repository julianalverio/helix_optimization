from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_RESIDUE_TYPES = 20


def one_hot_residue_type(residue_type: torch.Tensor) -> torch.Tensor:
    """One-hot encode residue type indices for the 20 standard amino acids
    (ALA…VAL, indexed 0-19). Input shape (..., N), output shape (..., N, 20),
    dtype float32."""
    return F.one_hot(residue_type, num_classes=NUM_RESIDUE_TYPES).float()
