"""Tests for the interaction-matrix noise pipeline. The clean target is
binary (B, N, N, 6); the conditioning input is (B, N, N, 8) — 6 noisy
probability channels (bit-flip + Beta sample), an augmentation mask
(channel 6, set by residue masking), and a padding mask (channel 7, 1
where either residue is padding)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.config import MLConfig
from twistr.pipeline.features.interaction_matrix import (
    clean_interaction_matrix,
    conditioning_interaction_matrix,
)

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")


def _zero_cfg(**overrides) -> MLConfig:
    """All noise disabled. Bit flip rates = 0; Beta defaults still active.
    Tests enable specific steps via overrides."""
    base = dict(
        interacting_residue_mask_count_min=0,
        interacting_residue_mask_count_max=0,
        non_interface_residue_mask_rate=0.0,
        max_zero_to_one_flip_rate=0.0,
        max_one_to_zero_flip_rate=0.0,
    )
    base.update(overrides)
    return dataclasses.replace(MLConfig(), **base)


def _real_batch() -> dict[str, torch.Tensor]:
    from twistr.pipeline.datasets.datamodule import pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset
    ds = ExamplesDataset([EXAMPLE_NPZ])
    return pad_collate([ds[0]])


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator(); g.manual_seed(seed)
    return g


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_clean_matrix_is_binary():
    """6 channels, each exactly 0.0 or 1.0. Multi-label (some pairs have
    multiple channels = 1). `none == 1` ⟺ vdw = hbond = pd = sandwich =
    t_shaped = 0. Diagonal is (0, 0, 0, 0, 0, 1)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    assert clean.shape[-1] == 6
    assert ((clean == 0) | (clean == 1)).all()
    any_other = (clean[..., :5] == 1).any(dim=-1)
    assert torch.equal(clean[..., 5] == 1, ~any_other)
    N = clean.shape[1]
    for i in range(N):
        assert torch.equal(clean[0, i, i], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    not_diag = ~torch.eye(N, dtype=torch.bool)
    multi_label = (clean[0, ..., :5] == 1).sum(dim=-1) > 1
    assert (multi_label & not_diag).any(), "expected at least some pairs with multiple active channels"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_aromatic_subtypes_do_not_co_fire():
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    aromatic_block = clean[..., 2:5]
    n_active = aromatic_block.sum(dim=-1)
    assert (n_active <= 1).all()


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_output_shape_is_eight_channels():
    """End-to-end smoke: (B, N, N, 8), finite, in [0, 1]."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    out = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(0))
    assert out.shape == (1, clean.shape[1], clean.shape[2], 8)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_no_flip_recovers_clean_per_channel():
    """With max_*_flip_rate = 0 and high Beta confidence, thresholding the
    soft probability at 0.5 recovers the binary clean per channel exactly
    (Beta means 0.25 / 0.75 are well-separated by 0.5)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(positive_beta_confidence=1000.0, negative_beta_confidence=1000.0)
    out = conditioning_interaction_matrix(clean, batch, cfg, _gen(0))
    N = clean.shape[1]
    not_diag = ~torch.eye(N, dtype=torch.bool)
    recovered = (out[0, ..., :6] > 0.5).float()
    mask = not_diag.unsqueeze(-1).expand_as(recovered)
    assert torch.equal(recovered[mask], clean[0][mask])


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_residue_mask_marks_one_full_row_and_column():
    """Step 1 isolated, count = 1: exactly one residue's full row+col gets
    augmentation_mask = 1 (channel 6)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(interacting_residue_mask_count_min=1, interacting_residue_mask_count_max=1)
    out = conditioning_interaction_matrix(clean, batch, cfg, _gen(0))[0]
    N = out.shape[0]
    mask_per_row = out[..., 6].sum(dim=-1)
    fully_masked = torch.nonzero(mask_per_row == N - 1, as_tuple=False).flatten()
    assert fully_masked.numel() == 1
    r = int(fully_masked.item())
    not_self = torch.arange(N) != r
    assert (out[r, not_self, 6] == 1).all()
    assert (out[not_self, r, 6] == 1).all()


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_symmetry():
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    out = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(0))[0]
    assert torch.equal(out, out.transpose(0, 1))


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_diagonal_preserved():
    """Diagonal: modality = (0,0,0,0,0,1), augmentation_mask = 0,
    padding_mask = 0 (no padding in this single-example batch)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    out = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(0))[0]
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    for i in range(out.shape[0]):
        assert torch.equal(out[i, i], expected)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_non_interface_only_masking():
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(non_interface_residue_mask_rate=1.0)
    out = conditioning_interaction_matrix(clean, batch, cfg, _gen(0))[0]
    N = out.shape[0]
    is_interface = batch["is_interface_residue"][0].bool()
    not_diag = ~torch.eye(N, dtype=torch.bool)
    for i in range(N):
        if not is_interface[i]:
            assert (out[i, not_diag[i], 6] == 1).all()
        else:
            for j in range(N):
                if i == j:
                    continue
                expected_mask = float(not is_interface[j])
                assert out[i, j, 6].item() == expected_mask


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_reproducible_across_runs():
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    a = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(123))
    b = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(123))
    assert torch.equal(a, b)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_padding_mask_channel_marks_padded_pairs():
    """In a B=2 batch where one example is shorter, padding_mask channel = 1
    on every cell where i or j is padded, 0 elsewhere."""
    from twistr.pipeline.datasets.datamodule import _PAD_FILL, pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset

    ds = ExamplesDataset([EXAMPLE_NPZ])
    sample = ds[0]
    n_real = sample["is_helix"].shape[0]
    pad_len = 7
    sample_long = {
        k: torch.cat([sample[k], torch.full((pad_len, *sample[k].shape[1:]), fill, dtype=sample[k].dtype)], dim=0)
        for k, fill in _PAD_FILL.items()
    }
    batch = pad_collate([sample, sample_long])
    clean = clean_interaction_matrix(batch)
    out = conditioning_interaction_matrix(clean, batch, MLConfig(), _gen(0))

    pm = batch["padding_mask"]                                                  # (2, N)
    pad = ~pm                                                                   # 1 where padded
    expected = (pad[:, :, None] | pad[:, None, :]).float()
    assert torch.equal(out[..., 7], expected)


def _padded_batch(sample: dict[str, torch.Tensor], pad_len: int) -> dict[str, torch.Tensor]:
    """Manually construct a B=1 batch where the last `pad_len` residues are
    padding. pad_collate alone won't produce padding for a single sample —
    we extend the per-field tensors with their _PAD_FILL value and then set
    padding_mask explicitly."""
    from twistr.pipeline.datasets.datamodule import _PAD_FILL
    n_real = sample["is_helix"].shape[0]
    extended = {
        k: torch.cat([sample[k], torch.full((pad_len, *sample[k].shape[1:]), fill, dtype=sample[k].dtype)], dim=0)
        for k, fill in _PAD_FILL.items()
    }
    batch = {k: v.unsqueeze(0) for k, v in extended.items()}
    pm = torch.zeros(1, n_real + pad_len, dtype=torch.bool)
    pm[0, :n_real] = True
    batch["padding_mask"] = pm
    return batch


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_padding_and_augmentation_masks_are_independent():
    """Augmentation off + padding present → only channel 7 fires.
    Augmentation on + no padding → only channel 6 fires."""
    from twistr.pipeline.datasets.datamodule import pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset

    ds = ExamplesDataset([EXAMPLE_NPZ])
    sample = ds[0]
    n_real = sample["is_helix"].shape[0]
    pad_len = 5
    batch_padded = _padded_batch(sample, pad_len)

    clean_padded = clean_interaction_matrix(batch_padded)
    out_padded = conditioning_interaction_matrix(clean_padded, batch_padded, _zero_cfg(), _gen(0))
    assert (out_padded[..., 6] == 0).all(), "augmentation mask should be 0 when residue masking is disabled"
    assert out_padded[..., 7][:, n_real:].sum() > 0, "padding mask should fire on padded rows"

    batch_unpadded = pad_collate([sample])
    clean_unpadded = clean_interaction_matrix(batch_unpadded)
    cfg_aug = _zero_cfg(interacting_residue_mask_count_min=1, interacting_residue_mask_count_max=1)
    out_unpadded = conditioning_interaction_matrix(clean_unpadded, batch_unpadded, cfg_aug, _gen(0))
    assert (out_unpadded[..., 7] == 0).all(), "padding mask should be 0 when no padding"
    assert (out_unpadded[..., 6] > 0).any(), "augmentation mask should fire on the masked residue"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_no_bit_flip_on_padding_channel_under_high_flip_rate():
    """High flip rates must not perturb the padding mask. With max rates 1.0,
    padding cells stay 1 and unpadded cells stay 0."""
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset

    ds = ExamplesDataset([EXAMPLE_NPZ])
    sample = ds[0]
    n_real = sample["is_helix"].shape[0]
    pad_len = 10
    batch = _padded_batch(sample, pad_len)
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(max_zero_to_one_flip_rate=1.0, max_one_to_zero_flip_rate=1.0)
    out = conditioning_interaction_matrix(clean, batch, cfg, _gen(0))[0]
    pm = batch["padding_mask"][0]
    pad = ~pm
    expected = (pad[:, None] | pad[None, :]).float()
    assert torch.equal(out[..., 7], expected)
    # Sanity: there's actually padding to test against (avoid the trivial
    # "all zeros == all zeros" pass).
    assert pad.any()


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_beta_sample_means():
    """High Beta confidence + zero flip rates → the modality-channel sample
    mean over all-zero cells matches negative_beta_mean and over all-one
    cells matches positive_beta_mean (within a small statistical tolerance)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(positive_beta_confidence=200.0, negative_beta_confidence=200.0)
    out = conditioning_interaction_matrix(clean, batch, cfg, _gen(0))
    N = clean.shape[1]
    not_diag = ~torch.eye(N, dtype=torch.bool)
    chan = out[0, ..., :6]                                                       # (N, N, 6)
    mask_diag = not_diag.unsqueeze(-1).expand_as(chan)
    ones = (clean[0] == 1) & mask_diag
    zeros = (clean[0] == 0) & mask_diag
    assert ones.sum() > 100 and zeros.sum() > 100, "need enough cells for stats"
    mean_pos = chan[ones].mean().item()
    mean_neg = chan[zeros].mean().item()
    assert abs(mean_pos - 0.75) < 0.02, f"positive Beta mean {mean_pos:.4f} != 0.75"
    assert abs(mean_neg - 0.25) < 0.02, f"negative Beta mean {mean_neg:.4f} != 0.25"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_flip_rate_distribution():
    """Aggregate over many seeds: per-channel flip fraction across pristine
    cells matches E[u] · max_rate = 0.5 · max_rate within tolerance.
    Uses 0/1 thresholding on the soft probability (Beta means 0.25/0.75
    don't cross 0.5 so threshold-recovers the noisy_binary)."""
    batch = _real_batch()
    clean = clean_interaction_matrix(batch)
    cfg = _zero_cfg(
        max_zero_to_one_flip_rate=0.40,
        max_one_to_zero_flip_rate=0.40,
        positive_beta_confidence=200.0, negative_beta_confidence=200.0,
    )
    N = clean.shape[1]
    not_diag = ~torch.eye(N, dtype=torch.bool)
    n_seeds = 30
    flip_counts_01 = 0
    flip_counts_10 = 0
    total_zero = 0
    total_one = 0
    for s in range(n_seeds):
        out = conditioning_interaction_matrix(clean, batch, cfg, _gen(s))[0]
        recovered = (out[..., :6] > 0.5).float()
        flipped = (recovered != clean[0])
        flipped = flipped & not_diag.unsqueeze(-1)
        flip_counts_01 += int((flipped & (clean[0] == 0)).sum())
        flip_counts_10 += int((flipped & (clean[0] == 1)).sum())
        total_zero += int(((clean[0] == 0) & not_diag.unsqueeze(-1)).sum())
        total_one += int(((clean[0] == 1) & not_diag.unsqueeze(-1)).sum())
    rate_01 = flip_counts_01 / total_zero
    rate_10 = flip_counts_10 / total_one
    expected = 0.5 * 0.40
    assert abs(rate_01 - expected) < 0.05, f"0→1 flip rate {rate_01:.4f} != {expected}"
    assert abs(rate_10 - expected) < 0.05, f"1→0 flip rate {rate_10:.4f} != {expected}"
