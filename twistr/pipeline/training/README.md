# `twistr.pipeline.training` — training orchestration

Entry point, empirical GPU-memory calibration, and remote-launch
plumbing. The training step itself lives in `ExamplesModule` (in
`models/`); this directory owns everything around it — how a run is
configured, how batch sizes are picked for the current hardware, and
how a job lands on a rented GPU.

## `train.py` — entry point

Loads the YAML config, builds the data module + model + Wandb logger,
hands them to `pl.Trainer`. Three details specific to this setup:

- **Per-launch checkpoint subdirectory.** Each launch writes to
  `<checkpoint_dir>/run_<unix_timestamp>/`. Lightning's strict
  state-dict loading would fail an automatic `last.ckpt` resume across
  runs with architecture or hyperparameter changes; per-launch
  isolation makes crash recovery opt-in via an explicit `--resume`
  flag rather than implicit-and-broken.
- **`use_distributed_sampler = False` on DDP.** Lightning cannot
  auto-wrap our custom `batch_sampler=` (the `LengthBucketBatchSampler`)
  into a `DistributedSampler`. The batch sampler reads
  `trainer.world_size` / `trainer.global_rank` itself and emits a
  balanced bucket count per rank — we own the per-rank partition.
- **`num_sanity_val_steps = 0`.** Lightning's pre-training val pass
  loads the first val buckets at full calibrated B; on a network-
  filesystem-mounted dataset that's hundreds of file loads before
  training can start. Skipping it doesn't lose signal we care about.

## `batch_calibration.py` — empirical (N_max → max_B) lookup

The Pairformer's pair tensor and triangle-attention logits dominate
memory, with roughly cubic dependence on `N_max`. A static `batch_size`
is therefore a poor fit for variable-length proteins: sized for
worst-case `N_max`, the GPU sits underutilized on every short-helix
batch. The right knob is a per-batch `B = max_B(N_max)`.

The calibration module runs a **doubling-then-bisect** OOM probe at
each of a small set of N quantiles drawn from the actual training set.
At each `N_max`, the probe synthesizes a batch matching `pad_collate`'s
output schema (same dtypes, same feature semantics) so probed memory
reflects what real training will see, then doubles B until OOM and
bisects between the last fitting and first failing B. The resulting
`{N_max → max_B}` table is consumed by `LengthBucketBatchSampler` to
size each batch at run time.

Cached to disk keyed by `sha256(GPU memory class + memory-relevant
cfg fields)`. The hash deliberately excludes loss weights, learning
rate, data paths, and training-loop knobs, so hyperparameter sweeps
over those reuse the calibration. The **GPU memory class** normalizes
the device name so different SKUs of the same GPU + memory tier share
calibration: `NVIDIA A100 80GB PCIe` and `NVIDIA A100-SXM4-80GB` both
map to `A100-80GB`. A hard `max_B_cap` is applied on top of the
empirical result: on a network-filesystem-mounted dataset a single
dataloader worker has to load every example in a batch sequentially,
so at large B the GPU starves on I/O before memory becomes the
constraint.

CPU runs fall back to `{max_n: 1}` (no calibration possible without a
GPU and none needed).

## `probe.py` — standalone calibration

Same calibration path as `train.py` but as a free-standing script.
Includes a `--compute-lengths-only` mode that skips the GPU probe and
just populates the per-example lengths sidecar — used to pre-compute
lengths on a laptop before launching a remote training job. The
sidecar is portable across machines because keys are relative paths.

## Remote launch

The Mac-side RunPod orchestrator lives at `dev/tools/runpod/train/launch.py` —
upload, pod allocation, log streaming, and self-termination. Nothing in
this directory is responsible for talking to RunPod.

## File index

| File | Role |
|---|---|
| `train.py` | Training entry point — config load, data module, Lightning Trainer, checkpoint policy. |
| `batch_calibration.py` | Empirical `{N_max → max_B}` calibration with content-hashed disk cache. |
| `probe.py` | Standalone probe + `--compute-lengths-only` mode. |
