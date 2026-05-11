# `twistr.pipeline.datasets` — training data for in-distribution helix-mediated interfaces

Training examples are per-interface crops from the PDB: an α-helix
plus its partner chain restricted to residues within a contact-radius
window. The deployment regime is lead optimization on PXDesign / Boltz
proposals — the agent's inference distribution is helix-mediated
interfaces on the same fold manifold the model is trained on — so the
sampling and validation strategies are tuned to **in-distribution
generalization**, not OOD.

## Sampling: cluster-balanced, in-distribution

Helix sequences cluster heavily by fold family. Without weighting, common
fold families dominate the gradient and rare ones go undertrained — bad
for an agent that may propose mutations in any direction across the
manifold. `WeightedRandomSampler(weights = 1 / cluster_size,
replacement=True)` gives balanced exposure.

Clustering itself (`cluster_helices.py`) is non-trivial at 424K
sequences: the full pairwise Levenshtein matrix would be 89B entries,
so the pipeline first builds a sparse candidate graph (pairs with
length-normalized distance `d_norm ≤ τ_max` via chunked rapidfuzz
`cdist`), takes connected components, and runs dense complete-linkage
clustering inside each component. Pairs absent from the candidate graph
have `d > τ_max` and never merge under complete linkage at any
`τ ≤ τ_max`, so this is **exact** (not an approximation). The threshold
τ is selected by silhouette over a τ-grid.

## Validation: sequence-disjoint, not cluster-disjoint

The val-time mutation-sensitivity metric requires held-out helix
sequences to be *literally absent* from train, so the val set holds out
unique helix sequences rather than entire clusters. Examples whose
sequence is chosen for val but which are not the single representative
are dropped from both sides — putting them in train re-leaks the
held-out sequence, putting them in val violates uniqueness. About 1.5K
examples drop from the 424K pool.

Cluster-level OOD generalization is intentionally not a goal. In the
deployed pipeline, the agent proposes small mutations to known-binding
helices, so inference stays on the same fold manifold seen in training.
Holding out clusters would test a property — generalization to new fold
families — that doesn't matter for the deployment regime.

## Length-aware dynamic batching

Pair-tensor and triangle-attention memory in the Pairformer scale as
O(B·N²·c_z) and O(B·N³·H). A fixed `batch_size` is therefore a poor fit
for variable-length proteins (sized for worst-case `N_max`, the GPU
sits underutilized on short batches). `LengthBucketBatchSampler` sizes
each batch's B to its `N_max` via the empirical calibration table from
`training/batch_calibration.py`. Two modes:

- **Wrapped (train).** Drains the cluster-weighted upstream sampler
  each epoch, sorts indices by N, walks emitting variable-B buckets at
  the running `N_max`. Bucket order is *shuffled within the epoch*
  rather than emitted short-to-long: a previous training run surfaced
  step-function spikes in losses with N-dependent denominators (VDW,
  aromatic stacking) at the short→long transition; shuffling within
  length-uniform buckets preserves the inside-bucket uniformity while
  decorrelating order from training step.
- **Standalone (val).** Sorts the index set once at construction,
  emits deterministic buckets, shuffles bucket order per epoch with a
  seeded RNG. Each example appears exactly once per epoch.

DDP-native: each rank yields exactly `floor(total_buckets / num_replicas)`
buckets per epoch so gradient sync stays balanced. Standalone-mode
buckets are round-robin partitioned across ranks; wrapped-mode ranks
drive independent per-rank-seeded torch generators with deterministic
tail-padding if a rank's draws come up short.

## Coordinate normalization

`ExamplesDataset.__getitem__` upcasts to fp32, centers on the centroid
of real heavy atoms, divides by 10 Å so one unit ≈ 1 nm, and (if
`random_rotate=True`) applies a uniformly-distributed SO(3) rotation
sampled per fetch (QR of a 3×3 Gaussian, restricted to det = +1 by
column flip). Distances and angles are invariant under this
transformation; predictions and GT share the same per-example frame,
so no Kabsch alignment is needed at loss time.

## File index

| File | Role |
|---|---|
| `example_dataset.py` | `ExamplesDataset` — disk reader, fp32 upcast, centering, /10 normalization, optional SO(3) augmentation. |
| `datamodule.py` | Lightning `DataModule` — manifest + cluster loading, sequence-disjoint split, sampler construction, calibration trigger. |
| `batch_sampler.py` | `LengthBucketBatchSampler` (wrapped + standalone, DDP-native); `compute_lengths` with portable on-disk sidecar cache. |
| `cluster_helices.py` | Standalone clustering script: sparse-candidate-graph + complete-linkage hierarchical clustering, silhouette-selected τ. |
| `val_split.py` | Sequence-disjoint val split with drop-to-preserve-both-invariants logic. |
