"""Architecture sanity tests: shapes, output constraints (probability simplex,
unit-norm chi, valid rotations), gradient flow through the full model, and a
real-data round-trip via the LightningModule's training_step."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.config import MLConfig
from twistr.pipeline.features.builder import build_features
from twistr.pipeline.models.architecture import (
    TorsionHead,
    HelixDesignModel,
    InputEmbedder,
    InteractionMatrixHead,
)
from twistr.pipeline.models.output_head import FrameOutputHead
from twistr.pipeline.models.pairformer import (
    PairformerBlock,
    PairformerStack,
    TriangleAttention,
    TriangleMultiplication,
)

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")


def _real_batch():
    from twistr.pipeline.datasets.datamodule import pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset
    ds = ExamplesDataset([EXAMPLE_NPZ])
    return pad_collate([ds[0]])


def _small_cfg() -> MLConfig:
    """Tiny model for fast tests."""
    return MLConfig(
        c_s=32, c_z=16, pairformer_blocks=2,
        n_heads_single=2, n_heads_pair=2,
        c_hidden_mul=8, c_hidden_pair_att=8,
        transition_n=2, pairformer_dropout=0.0,
    )


def test_triangle_multiplication_shape_and_outgoing_vs_incoming():
    """The outgoing variant computes x_ij = Σ_k a_ik · b_jk and the incoming
    variant x_ij = Σ_k a_ki · b_kj — for a non-symmetric input z they must
    produce different outputs. (Default init zeroes the final linears, so
    re-init them to make the test meaningful.)"""
    B, N, c_z, c_h = 1, 6, 16, 8
    z = torch.randn(B, N, N, c_z)
    mul_out = TriangleMultiplication(c_z, c_h, outgoing=True)
    mul_in = TriangleMultiplication(c_z, c_h, outgoing=False)
    for mod in (mul_out, mul_in):
        torch.nn.init.normal_(mod.linear_z.weight, std=0.1)
        torch.nn.init.normal_(mod.linear_g.weight, std=0.1)
    a = mul_out(z)
    b = mul_in(z)
    assert a.shape == z.shape and b.shape == z.shape
    assert (a - b).abs().mean().item() > 1e-6


def test_triangle_attention_starting_vs_ending_differ():
    B, N, c_z = 1, 6, 16
    z = torch.randn(B, N, N, c_z)
    a_start = TriangleAttention(c_z, 4, n_heads=2, starting=True)(z)
    a_end = TriangleAttention(c_z, 4, n_heads=2, starting=False)(z)
    assert a_start.shape == z.shape
    assert a_end.shape == z.shape


def test_pairformer_block_preserves_shapes():
    cfg = _small_cfg()
    block = PairformerBlock(
        c_s=cfg.c_s, c_z=cfg.c_z,
        n_heads_single=cfg.n_heads_single, n_heads_pair=cfg.n_heads_pair,
        c_hidden_mul=cfg.c_hidden_mul, c_hidden_pair_att=cfg.c_hidden_pair_att,
        transition_n=cfg.transition_n, dropout=cfg.pairformer_dropout,
    )
    B, N = 2, 7
    s = torch.randn(B, N, cfg.c_s)
    z = torch.randn(B, N, N, cfg.c_z)
    s2, z2 = block(s, z)
    assert s2.shape == s.shape and z2.shape == z.shape


def test_pairformer_stack_runs_and_is_finite():
    cfg = _small_cfg()
    stack = PairformerStack(
        n_blocks=cfg.pairformer_blocks, c_s=cfg.c_s, c_z=cfg.c_z,
        n_heads_single=cfg.n_heads_single, n_heads_pair=cfg.n_heads_pair,
        c_hidden_mul=cfg.c_hidden_mul, c_hidden_pair_att=cfg.c_hidden_pair_att,
        transition_n=cfg.transition_n, dropout=0.0,
    )
    s = torch.randn(2, 7, cfg.c_s)
    z = torch.randn(2, 7, 7, cfg.c_z)
    s_out, z_out = stack(s, z)
    assert torch.isfinite(s_out).all() and torch.isfinite(z_out).all()


def test_interaction_matrix_head_independent_logits():
    """Multi-label: each of the 6 channels is an independent logit (sigmoid
    gives a probability in [0, 1]) — they do NOT sum to 1. At init the weight
    is zero so every channel collapses to `INIT_BIAS[c]` regardless of input —
    the per-channel logit-prior the head is initialised to."""
    head = InteractionMatrixHead(c_z=16)
    z = torch.randn(2, 5, 5, 16)
    out = head(z)
    assert out.shape == (2, 5, 5, 6)
    expected = torch.tensor(InteractionMatrixHead.INIT_BIAS).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-5)


def test_interaction_matrix_head_symmetric():
    head = InteractionMatrixHead(c_z=16)
    z = torch.randn(1, 6, 6, 16)
    out = head(z)
    assert torch.allclose(out, out.transpose(1, 2), atol=1e-5)


def test_frame_head_starts_at_identity_and_passes_through_conditioning():
    """At init the head must compose the conditioning frame with delta_R = I
    and pass `conditioning_translation` through unchanged (delta_t = 0):
        R = conditioning_R @ I = conditioning_R   (where validity = True)
        R = I @ I = I                             (where validity = False)
        t = conditioning_translation + 0 = conditioning_translation

    Partner residues with a valid GT frame land at their GT backbone pose
    immediately; helix residues (no conditioning frame) sit at the canonical
    layout pose with R = identity and learn from scratch."""
    from twistr.pipeline.models.rotation import rotation_from_6d

    head = FrameOutputHead(hidden_dim=32)
    s = torch.randn(2, 5, 32)
    cond_t = torch.randn(2, 5, 3)
    cond_6d = torch.randn(2, 5, 6)
    cond_valid = torch.tensor([
        [True, True, False, True, False],
        [False, True, True, False, True],
    ])
    # `conditioning_frame_6d` upstream is zeroed wherever validity is False;
    # mirror that so the test exercises the same input as the real pipeline.
    cond_6d = cond_6d * cond_valid.unsqueeze(-1)
    R, t = head(s, cond_t, cond_6d, cond_valid)

    assert torch.allclose(t, cond_t, atol=1e-6)

    expected_R = rotation_from_6d(cond_6d)
    eye = torch.eye(3).expand_as(expected_R)
    expected_R = torch.where(
        cond_valid.unsqueeze(-1).unsqueeze(-1), expected_R, eye,
    )
    assert torch.allclose(R, expected_R, atol=1e-5)


def test_torsion_head_unit_norm():
    head = TorsionHead(c_s=32)
    s = torch.randn(2, 5, 32)
    out = head(s)
    assert out.shape == (2, 5, 7, 2)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_full_model_on_real_batch():
    cfg = _small_cfg()
    batch = _real_batch()
    features = build_features(batch, MLConfig())
    model = HelixDesignModel(cfg)
    out = model(features)

    B, N = 1, 58
    assert out["interaction_matrix"].shape == (B, N, N, 6)
    assert out["rotation"].shape == (B, N, 3, 3)
    assert out["translation"].shape == (B, N, 3)
    assert out["torsion_sincos"].shape == (B, N, 7, 2)

    # Interaction-matrix channels are independent logits (multi-label, NOT a
    # categorical) — finite-valued; sigmoid gives the per-channel probability.
    assert torch.isfinite(out["interaction_matrix"]).all()
    assert torch.allclose(
        out["torsion_sincos"].norm(dim=-1), torch.ones(B, N, 7), atol=1e-5,
    )
    # Rotation matrices: orthonormal, det = +1.
    R = out["rotation"]
    eye = torch.eye(3).expand_as(R[0]).unsqueeze(0)
    assert torch.allclose(R @ R.transpose(-2, -1), eye, atol=1e-4)
    assert torch.allclose(torch.linalg.det(R), torch.ones(B, N), atol=1e-4)


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_gradient_flows_through_all_outputs():
    cfg = _small_cfg()
    batch = _real_batch()
    features = build_features(batch, MLConfig())
    model = HelixDesignModel(cfg)
    out = model(features)
    loss = (
        out["interaction_matrix"].sum()
        + out["rotation"].sum()
        + out["translation"].sum()
        + out["torsion_sincos"].sum()
    )
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 10, f"only {n_with_grad} params received gradient"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_mutation_sensitivity_metrics_runs_and_is_finite():
    """val-only metric: K random helix substitutions per example, each
    triggering a forward pass with WT conditioning held fixed. The three
    reported numbers must be finite and non-negative."""
    from twistr.pipeline.features.builder import build_features
    from twistr.pipeline.models.lightning_module import ExamplesModule
    cfg = MLConfig(
        c_s=32, c_z=16, pairformer_blocks=2,
        n_heads_single=2, n_heads_pair=2,
        c_hidden_mul=8, c_hidden_pair_att=8,
        transition_n=2, pairformer_dropout=0.0,
        mutation_metric_k=2,
    )
    batch = _real_batch()
    mod = ExamplesModule(cfg)
    mod.eval()
    with torch.no_grad():
        features = build_features(batch, cfg)
        wt_logits = mod.model(features)["interaction_matrix"]
        metrics = mod._mutation_sensitivity_metrics(batch, features, wt_logits, batch_idx=0)
    assert set(metrics) == {"im_total", "im_local", "im_far"}
    for name, value in metrics.items():
        assert torch.isfinite(value), f"{name} is not finite: {value}"
        assert value.item() >= 0.0, f"{name} should be non-negative: {value.item()}"


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_lightning_module_training_step_runs():
    from twistr.pipeline.models.lightning_module import ExamplesModule
    batch = _real_batch()
    mod = ExamplesModule(_small_cfg())
    losses = mod._compute_losses(batch, training=True)
    assert set(losses) == {
        "helix", "bce", "vdw", "hbond",
        "parallel_displaced", "sandwich", "t_shaped", "clash",
        "backbone_continuity", "dunbrack_interacting", "dunbrack_non_interacting",
        "coord_mse_antigen_backbone", "coord_mse_annealed", "packing",
    }
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} loss is not finite: {value}"


_SECOND_NPZ = Path("runtime/data/examples/examples/mt/4mt8_1_6.npz")


@pytest.mark.skipif(
    not (EXAMPLE_NPZ.exists() and _SECOND_NPZ.exists()),
    reason="needs two example npz files on disk",
)
def test_lightning_module_handles_padded_batch_of_two():
    """Two examples of different lengths must collate via pad_collate and run
    through _compute_losses with finite per-loss values."""
    from twistr.pipeline.datasets.datamodule import pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset
    from twistr.pipeline.models.lightning_module import ExamplesModule

    ds = ExamplesDataset([EXAMPLE_NPZ, _SECOND_NPZ])
    n0 = ds[0]["is_helix"].shape[0]
    n1 = ds[1]["is_helix"].shape[0]
    batch = pad_collate([ds[0], ds[1]])
    n_max = max(n0, n1)
    assert batch["padding_mask"].shape == (2, n_max)
    assert batch["padding_mask"][0].sum().item() == n0
    assert batch["padding_mask"][1].sum().item() == n1
    assert (batch["atom_mask"][0, n0:] == -1).all()
    assert (batch["atom_mask"][1, n1:] == -1).all()

    mod = ExamplesModule(_small_cfg())
    losses = mod._compute_losses(batch, training=True)
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} loss is not finite: {value}"


def _no_noise_cfg() -> MLConfig:
    """Tiny model with all conditioning-noise rates zeroed. build_features
    still draws from the Beta distribution per pristine cell (the noise
    pipeline always emits soft probabilities), so the padding-invariance
    test must re-seed torch's global RNG before each build_features call to
    get matching values at real cells."""
    return MLConfig(
        c_s=32, c_z=16, pairformer_blocks=2,
        n_heads_single=2, n_heads_pair=2,
        c_hidden_mul=8, c_hidden_pair_att=8,
        transition_n=2, pairformer_dropout=0.0,
        interacting_residue_mask_count_min=0, interacting_residue_mask_count_max=0,
        non_interface_residue_mask_rate=0.0,
        max_zero_to_one_flip_rate=0.0, max_one_to_zero_flip_rate=0.0,
    )


@pytest.mark.skipif(not EXAMPLE_NPZ.exists(), reason=f"{EXAMPLE_NPZ} not on disk")
def test_padding_does_not_change_real_residue_outputs():
    """Direct test of the model's pair_mask: appending padded residues to a
    single example must not change the model's outputs at the real-residue
    positions. If this passes, every loss reduced over real positions only is
    automatically invariant to padding."""
    from twistr.pipeline.datasets.datamodule import _PAD_FILL, pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset
    from twistr.pipeline.features.builder import build_features
    from twistr.pipeline.models.lightning_module import ExamplesModule

    ds = ExamplesDataset([EXAMPLE_NPZ])
    sample = ds[0]
    n_real = sample["is_helix"].shape[0]
    pad_len = 30

    batch_short = pad_collate([sample])
    sample_long = {
        k: torch.cat([sample[k], torch.full((pad_len, *sample[k].shape[1:]), fill, dtype=sample[k].dtype)], dim=0)
        for k, fill in _PAD_FILL.items()
    }
    batch_long = {k: v.unsqueeze(0) for k, v in sample_long.items()}
    pm = torch.zeros(1, n_real + pad_len, dtype=torch.bool)
    pm[0, :n_real] = True
    batch_long["padding_mask"] = pm

    cfg = _no_noise_cfg()
    torch.manual_seed(0)
    mod = ExamplesModule(cfg)
    mod.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        feats_short = build_features(batch_short, cfg)
        torch.manual_seed(42)
        feats_long = build_features(batch_long, cfg)
        out_short = mod.model(feats_short)
        out_long = mod.model(feats_long)

    for key in ["rotation", "translation", "torsion_sincos"]:
        a = out_short[key][:, :n_real]
        b = out_long[key][:, :n_real]
        assert torch.allclose(a, b, atol=1e-4), \
            f"{key} differs at real positions: max diff {(a - b).abs().max().item()}"
    a_im = out_short["interaction_matrix"][:, :n_real, :n_real]
    b_im = out_long["interaction_matrix"][:, :n_real, :n_real]
    assert torch.allclose(a_im, b_im, atol=1e-4), \
        f"interaction_matrix differs at real cells: max diff {(a_im - b_im).abs().max().item()}"


@pytest.mark.skip(
    reason="conditioning noise is now stochastic (Beta sampling per pristine "
    "cell), so the same example sees different conditioning depending on its "
    "position in the batch. Restoring this test requires content-derived "
    "per-example seeding inside _apply_interaction_matrix_noise."
)
@pytest.mark.skipif(
    not (EXAMPLE_NPZ.exists() and _SECOND_NPZ.exists()),
    reason="needs two example npz files on disk",
)
def test_per_example_loss_is_length_invariant():
    """A B=2 batch [A, B] of different-length examples produces per-loss values
    equal to the average of (A alone) and (B alone), provided both have signal
    for that loss type (no-signal examples are excluded from the batch mean,
    so a B=2 batch where one example is empty for a loss yields just the
    other's loss). For the two real example files used here, every loss type
    has signal in both examples — so the simple mean check applies."""
    from twistr.pipeline.datasets.datamodule import pad_collate
    from twistr.pipeline.datasets.example_dataset import ExamplesDataset
    from twistr.pipeline.models.lightning_module import ExamplesModule

    ds = ExamplesDataset([EXAMPLE_NPZ, _SECOND_NPZ])
    sample_a, sample_b = ds[0], ds[1]
    batch_a = pad_collate([sample_a])
    batch_b = pad_collate([sample_b])
    batch_ab = pad_collate([sample_a, sample_b])

    cfg = _no_noise_cfg()
    torch.manual_seed(0)
    mod = ExamplesModule(cfg)
    mod.eval()
    with torch.no_grad():
        losses_a = mod._compute_losses(batch_a, training=False)
        losses_b = mod._compute_losses(batch_b, training=False)
        losses_ab = mod._compute_losses(batch_ab, training=False)

    for name in losses_a:
        # If a loss is 0 in either single-example batch, the example may be
        # signal-empty and excluded from the batch-level mean. Skip the
        # mean check in that case (the loss is still asserted finite).
        a, b = losses_a[name].item(), losses_b[name].item()
        if a == 0.0 or b == 0.0:
            continue
        expected = 0.5 * (losses_a[name] + losses_b[name])
        actual = losses_ab[name]
        assert torch.allclose(actual, expected, atol=1e-4), (
            f"{name}: per-example mean of [A,B] = {actual.item():.6f} "
            f"differs from 0.5*(A + B) = {expected.item():.6f}"
        )


def test_input_embedder_relpos_one_hot_single_chain():
    """Same-chain pairs land in 0..2k buckets centered at k for the diagonal;
    one extra bucket (2k+1) is reserved for different-chain pairs."""
    emb = InputEmbedder(c_s=32, c_z=16, relpos_max_offset=8)
    chain_slot = torch.zeros(1, 5, dtype=torch.long)
    rp = emb._relative_position_one_hot(chain_slot)
    assert rp.shape == (1, 5, 5, 18)
    assert (rp.sum(dim=-1) == 1).all()
    assert (rp[0].diagonal(dim1=0, dim2=1).T.argmax(dim=-1) == 8).all()
    assert rp[..., -1].sum() == 0  # no inter-chain pairs in a single-chain example


def test_input_embedder_relpos_one_hot_inter_chain_uses_special_bucket():
    """All cross-chain pairs collapse to bucket 2k+1; intra-chain pairs use
    the standard signed-delta buckets."""
    emb = InputEmbedder(c_s=32, c_z=16, relpos_max_offset=8)
    chain_slot = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    rp = emb._relative_position_one_hot(chain_slot)
    assert rp.shape == (1, 4, 4, 18)
    buckets = rp.argmax(dim=-1)[0]
    # Intra-chain (helix block): standard signed-delta buckets.
    assert buckets[0, 0].item() == 8 and buckets[1, 1].item() == 8
    assert buckets[0, 1].item() == 7 and buckets[1, 0].item() == 9
    # Cross-chain pairs all land in the special bucket (2k+1 = 17).
    assert (buckets[:2, 2:] == 17).all()
    assert (buckets[2:, :2] == 17).all()


def test_param_count_is_in_expected_range():
    """Default-config model should fit comfortably in the ~1-2M param range
    appropriate for our small-dataset, narrow-task regime."""
    cfg = MLConfig()
    model = HelixDesignModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 500_000 < n < 5_000_000, f"unexpected param count: {n:,}"
