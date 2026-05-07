"""Per-stage modules for the epitope pipeline.

Each module exposes `run_batch(ctx)` — the canonical implementation of one
pipeline stage. The manager (`twistr/pipeline/manager/manager.py`) reads a
YAML's `stages: [...]` list and calls the corresponding `run_batch` in
order. Stages communicate strictly through on-disk parquets, so any subset
of stages can run alone (e.g. `stages: [viz]` regenerates .pml files from
an existing final parquet without re-running upstream stages).
"""
