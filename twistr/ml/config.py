from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MLConfig:
    num_gpus: int = 1

    manifest_path: str = "data/module3/module3_manifest.parquet"
    examples_root: str = "data/module3"
    cluster_path: str = "data/module3/helix_clusters.parquet"
    val_count: int = 1000
    seed: int = 0

    batch_size: int = 1
    num_workers: int = 0

    # When True, the train/val DataLoaders use LengthBucketBatchSampler, sized
    # via an empirical (N_max → max_B) lookup table calibrated once per GPU +
    # cfg at the start of training (see twistr/ml/training/batch_calibration.py).
    # When False, falls back to the static `batch_size` above.
    dynamic_batch_size: bool = True
    # N quantiles probed during calibration (computed from the actual training
    # set's per-example N distribution). The lookup table maps each N to its
    # largest fitting B; sample-time interpolation rounds N up to the next
    # sweep point (conservative).
    calibration_n_quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 1.0)
    # Hard cap on per-batch B regardless of the calibration result. The OOM
    # probe maximises GPU memory use, but on a network-filesystem dataset a
    # single dataloader worker has to load every example in a batch
    # sequentially, so very large B starves the GPU on I/O. 0 disables the
    # cap.
    max_B_cap: int = 32

    max_epochs: int = 1
    max_steps: int = -1
    val_check_interval: float = 0.05
    # ModelCheckpoint sink. Saves best-by-val/loss_total + last; one-time write
    # of `last.ckpt` lets a crashed run resume from the most recent step.
    checkpoint_dir: str = "checkpoints"

    # Interaction-matrix conditioning noise. See twistr/ml/features/interactions.py
    # for the augmentation pipeline these drive. The clean target is binary
    # (0/1 per channel). Pristine cells go through per-example bit flips (rate
    # = u · max_*_flip_rate, where u ~ U(0,1) is one scalar per example) then
    # Beta(μ·ν, (1−μ)·ν) sampling to convert binary to a soft probability.
    interacting_residue_mask_count_min: int = 1
    interacting_residue_mask_count_max: int = 2
    non_interface_residue_mask_rate: float = 0.25
    max_zero_to_one_flip_rate: float = 0.20
    max_one_to_zero_flip_rate: float = 0.20
    positive_beta_mean: float = 0.75
    positive_beta_confidence: float = 15.0
    negative_beta_mean: float = 0.25
    negative_beta_confidence: float = 15.0

    # Architecture hyperparameters. Defaults sized for our regime — helix-
    # mediated interfaces (N ≤ ~200), no MSA / templates, lead-optimization
    # task closer to denoising than de-novo prediction. Aggressive scaledown
    # vs Protenix's defaults (c_s=384, c_z=128, n_blocks=48). Scale up if
    # underfitting; the trunk is small enough that doubling is cheap.
    c_s: int = 128                   # single representation channel dim
    c_z: int = 64                    # pair representation channel dim
    pairformer_blocks: int = 4
    n_heads_single: int = 4          # AttentionPairBias heads
    n_heads_pair: int = 4            # TriangleAttention heads
    c_hidden_mul: int = 32           # TriangleMultiplication hidden
    c_hidden_pair_att: int = 16      # TriangleAttention per-head hidden
    transition_n: int = 2            # FFN expansion factor
    pairformer_dropout: float = 0.1
    relpos_max_offset: int = 32      # ± bins for relative-position one-hot

    learning_rate: float = 1e-3
    # Linear LR warmup: lr ramps from 0 → learning_rate over the first
    # `lr_warmup_steps` optimizer steps, then constant. Many last-layer
    # weights are zero-init, so the first few hundred steps see asymmetric
    # gradient flow and Adam at full lr can blow up without a ramp.
    lr_warmup_steps: int = 1000

    helix_dihedral_weight: float = 1.0
    interaction_bce_weight: float = 5.0
    interaction_label_smoothing: float = 0.05

    # Geometric interaction losses on predicted full-atom coords. Each is a
    # flat-bottomed linear penalty (in Å / cosine units) — see
    # twistr/ml/losses/interactions.py. Down-weighted vs BCE because at init
    # every pair distance is multi-Å outside its band, so these losses are
    # O(10–100) per cell while BCE is O(0.7). Each π-stacking sub-type has
    # its own weight so they can be tuned independently.
    vdw_loss_weight: float = 0.1
    hbond_loss_weight: float = 0.1
    parallel_displaced_loss_weight: float = 0.1
    sandwich_loss_weight: float = 0.1
    t_shaped_loss_weight: float = 0.1

    # AF2-style steric clash penalty (Jumper et al. 2021, Suppl. Alg. 28).
    clash_loss_weight: float = 0.1

    # AF2-style between-residue bond geometry (Jumper et al. 2021, Suppl. Sec.
    # 1.9.11, eq 44–45). Penalises C(i)–N(i+1) bond length and adjacent bond
    # angles between consecutive same-chain residues — the chain can otherwise
    # split because the model's per-residue (R, t) heads are independent.
    # See twistr/ml/losses/backbone_continuity.py.
    backbone_continuity_weight: float = 0.1

    # Dunbrack rotamer-plausibility prior (-log p(χ | residue, χ_idx,
    # ss_class)). Two libraries fitted offline by
    # tools/dunbrack/fit_rotamer_library.py: 'helix' applied to is_helix
    # residues, 'general' to the antigen partner. Per-cell vMM with K=3
    # components. See twistr/ml/losses/dunbrack.py.
    dunbrack_weight: float = 0.1

    # Atom14 coord MSE (Å²) — structural anchor against GT. Two separate
    # losses with different masks and schedules:
    #
    #   coord_mse_antigen_backbone — partner backbone (slots 0-3 of every
    #     non-helix residue). Constant `_weight` throughout: we always
    #     want the antigen backbone pinned to the GT pose.
    #
    #   coord_mse_annealed — helix residues (all 14 slots) plus non-helix
    #     interface-residue sidechains (slots 4-13). Linearly anneals from
    #     `_start_weight` to `_floor_weight` over `_steps` training steps,
    #     so early training imitates the GT design and late training is
    #     free to find any band-/clash-/dihedral-valid configuration.
    #     Non-helix non-interface sidechains are penalized by neither loss.
    #
    # Validation always weights the annealed term at `_start_weight` so val
    # totals are comparable across the whole run; train follows the
    # schedule.
    coord_mse_antigen_backbone_weight: float = 0.1
    coord_mse_annealed_start_weight: float = 1.0
    coord_mse_annealed_floor_weight: float = 0.01
    coord_mse_annealed_steps: int = 10_000

    # Validation-only point-mutation sensitivity metric (logged under
    # `mutation_sensitivity/` in W&B). For each val example, sample
    # `mutation_metric_k` random helix single-residue substitutions, run
    # inference with the WT conditioning held fixed, and report mean-absolute-
    # difference of the predicted interaction-matrix probabilities — split into
    # near-mutation (within `mutation_metric_locality_radius` residues by
    # index) and far. Not a training loss.
    mutation_metric_k: int = 4
    mutation_metric_locality_radius: int = 5


def load_ml_config(path: Path | str) -> MLConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(MLConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown ml config keys: {sorted(unknown)}")
    cfg = MLConfig(**raw)
    if not 0 <= cfg.num_gpus <= 8:
        raise ValueError(f"num_gpus must be in [0, 8], got {cfg.num_gpus}")
    return cfg
