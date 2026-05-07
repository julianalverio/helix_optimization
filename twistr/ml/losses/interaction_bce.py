"""Per-channel BCE on the predicted interaction matrix with symmetric label
smoothing. The model head returns per-channel logits (B, N, N, 6); the target
is the clean binary (B, N, N, 6) from `clean_interaction_matrix`. Loss uses
`binary_cross_entropy_with_logits` so the sigmoid + log are fused for
numerical stability at the saturating ends.

The diagonal (i == j) is masked: by construction every diagonal cell has the
same label [0, 0, 0, 0, 0, 1], so it carries no learning signal."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def interaction_bce_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """padding_mask: (B, N) bool, True = real residue. Padded rows AND columns
    are excluded from both numerator and denominator. Reduction is per-example
    mean over real off-diagonal pairs, then mean over the batch (excluding any
    example that has no real pairs at all) — every example contributes equally
    regardless of length."""
    target_smooth = target * (1.0 - label_smoothing) + (1.0 - target) * label_smoothing
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_smooth, reduction="none")

    N = bce.shape[1]
    pair_real = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)         # (B, N, N)
    not_diag = ~torch.eye(N, dtype=torch.bool, device=bce.device)               # (N, N)
    valid = (pair_real & not_diag).unsqueeze(-1).to(bce.dtype)                  # (B, N, N, 1)
    valid_b = valid.expand_as(bce)
    denom = valid_b.sum(dim=(-3, -2, -1))                                       # (B,)
    per_example = (bce * valid).sum(dim=(-3, -2, -1)) / denom.clamp_min(1.0)
    has_signal = (denom > 0).to(bce.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
