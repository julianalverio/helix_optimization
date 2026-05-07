"""Tests for the interaction-matrix BCE loss with label smoothing.

The loss consumes per-channel **logits** and uses
`binary_cross_entropy_with_logits` internally for numerical stability.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from twistr.ml.losses.interaction_bce import interaction_bce_loss


def _bce_from_logit(z: float, t: float) -> float:
    p = 1.0 / (1.0 + math.exp(-z))
    return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _all_real(pred: torch.Tensor) -> torch.Tensor:
    return torch.ones(*pred.shape[:2], dtype=torch.bool)


def test_loss_matches_closed_form_no_smoothing():
    # 2 residues, 1 channel — only the off-diagonal pair (0,1) and (1,0)
    # contribute (diagonal masked). Logits 2.0 and -1.0; symmetric so the
    # mean over the two cells equals the BCE of one cell.
    pred = torch.tensor([[[[2.0], [-1.0]], [[-1.0], [2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0], [1.0]], [[1.0], [0.0]]]], dtype=torch.float32)
    loss = interaction_bce_loss(pred, target, _all_real(pred), label_smoothing=0.0)
    assert loss.item() == pytest.approx(_bce_from_logit(-1.0, 1.0), abs=1e-5)


def test_diagonal_is_masked():
    # Off-diagonal logits held fixed; diagonal target set to wildly mismatched
    # values. The two losses must agree because the diagonal is masked.
    pred = torch.zeros(1, 3, 3, 6)  # logit=0 → sigmoid=0.5
    target = torch.zeros(1, 3, 3, 6)
    loss_a = interaction_bce_loss(pred, target, _all_real(pred), label_smoothing=0.0)

    target_bad_diag = target.clone()
    for i in range(3):
        target_bad_diag[0, i, i, :] = 1.0
    loss_b = interaction_bce_loss(pred, target_bad_diag, _all_real(pred), label_smoothing=0.0)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-6)


def test_label_smoothing_floor_for_confident_correct():
    # With smoothing eps, the loss floor occurs when sigmoid(pred) equals the
    # smoothed target (eps for negatives, 1-eps for positives), giving the
    # binary entropy H(eps) = -eps*log(eps) - (1-eps)*log(1-eps) per cell.
    eps = 0.1
    target = torch.zeros(1, 4, 4, 6)
    target[..., 0] = 1.0
    pos_logit = _logit(1.0 - eps)
    pred = torch.where(target.bool(),
                       torch.full_like(target, pos_logit),
                       torch.full_like(target, -pos_logit))
    loss = interaction_bce_loss(pred, target, _all_real(pred), label_smoothing=eps)
    expected = -eps * math.log(eps) - (1.0 - eps) * math.log(1.0 - eps)
    assert loss.item() == pytest.approx(expected, rel=1e-3)


def test_smoothing_makes_overconfidence_costly():
    # With smoothing, a hugely positive logit on target=1 is *worse* than a
    # logit that lands at the smoothed target (1-eps). Sanity check: without
    # smoothing the larger logit strictly wins.
    target = torch.ones(1, 2, 2, 1)
    pred_overconfident = torch.full_like(target, 20.0)         # sigmoid ≈ 1.0
    pred_at_smoothed_target = torch.full_like(target, _logit(0.9))

    pm = _all_real(target)
    loss_no_smooth_over = interaction_bce_loss(pred_overconfident, target, pm, 0.0)
    loss_no_smooth_calib = interaction_bce_loss(pred_at_smoothed_target, target, pm, 0.0)
    assert loss_no_smooth_over.item() < loss_no_smooth_calib.item()

    loss_smooth_over = interaction_bce_loss(pred_overconfident, target, pm, 0.1)
    loss_smooth_calib = interaction_bce_loss(pred_at_smoothed_target, target, pm, 0.1)
    assert loss_smooth_calib.item() < loss_smooth_over.item()


def test_loss_is_symmetric_in_zero_one_swap():
    # Swapping (target, logit) ↔ (1 - target, -logit) should give the same
    # loss — symmetric label smoothing has this property and bce_with_logits
    # mirrors around logit=0 ⇔ probability=0.5.
    torch.manual_seed(0)
    pred = torch.randn(1, 5, 5, 6)                              # arbitrary logits
    target = (torch.rand(1, 5, 5, 6) > 0.5).float()
    eps = 0.07
    pm = _all_real(pred)
    loss_a = interaction_bce_loss(pred, target, pm, eps)
    loss_b = interaction_bce_loss(-pred, 1.0 - target, pm, eps)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-5)


def test_gradient_flows():
    pred = torch.randn(1, 6, 6, 6, requires_grad=True)
    target = (torch.rand(1, 6, 6, 6) > 0.5).float()
    loss = interaction_bce_loss(pred, target, _all_real(pred), label_smoothing=0.05)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
