"""Standalone OOM probe — runs `load_or_build_calibration` and exits.

Two modes:

  --compute-lengths-only:
      Reads the manifest + cluster files, splits train/val, calls
      `compute_lengths` for both. Populates `<examples_root>/.lengths.json`
      so a subsequent remote probe doesn't need the bulky npz files
      uploaded.

  default (GPU probe):
      Triggers the full calibration path by calling
      `ExamplesDataModule.setup()`. On GPU the doubling-then-bisect probe
      runs at each N quantile and writes `.cache/batch_calibration.json`.
      On CPU the calibration falls back to `{max_n: 1}` (see
      `batch_calibration.py`).

Run: `python -m twistr.pipeline.training.probe --config runtime/configs/ml.yaml`
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from twistr.pipeline.config import load_ml_config
from twistr.pipeline.datasets.batch_sampler import compute_lengths
from twistr.pipeline.datasets.datamodule import ExamplesDataModule
from twistr.pipeline.datasets.example_dataset import ExamplesDataset

# Make the calibrator's cache-hit / per-N max_B messages visible.
logging.getLogger("twistr.pipeline.training.batch_calibration").addHandler(
    logging.StreamHandler()
)
logging.getLogger("twistr.pipeline.training.batch_calibration").setLevel(logging.INFO)


def _compute_lengths_only(config_path: Path) -> int:
    """Replicates the manifest + split logic of
    `ExamplesDataModule.setup()` (no calibration, no probe module
    construction) and writes the lengths sidecar."""
    cfg = load_ml_config(config_path)
    manifest_path = Path(cfg.manifest_path)
    examples_root = Path(cfg.examples_root)
    cluster_path = Path(cfg.cluster_path)

    manifest = pd.read_parquet(manifest_path, columns=["path_example"])
    clusters = pd.read_parquet(cluster_path, columns=["path_example", "weight"])
    weight_lookup = dict(zip(clusters["path_example"], clusters["weight"]))
    rel_paths = manifest["path_example"].tolist()
    missing = [p for p in rel_paths if p not in weight_lookup]
    if missing:
        sys.exit(
            f"{len(missing)} manifest paths missing from {cluster_path}. "
            f"First few: {missing[:3]}"
        )
    pairs = [(examples_root / p, weight_lookup[p]) for p in rel_paths]
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_val = min(cfg.val_count, len(pairs) - 1)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    val_ds = ExamplesDataset([p for p, _ in val_pairs])
    train_ds = ExamplesDataset([p for p, _ in train_pairs])

    sidecar = examples_root / ".lengths.json"
    train_lengths = compute_lengths(train_ds, cache_path=sidecar)
    val_lengths = compute_lengths(val_ds, cache_path=sidecar)
    print(
        f"computed lengths for {len(train_lengths)} train + {len(val_lengths)} val "
        f"examples → {sidecar}"
    )
    print(
        f"  train N: min={min(train_lengths)} max={max(train_lengths)} "
        f"mean={sum(train_lengths) / len(train_lengths):.1f}"
    )
    return 0


def _gpu_probe(config_path: Path) -> int:
    cfg = load_ml_config(config_path)
    if not cfg.dynamic_batch_size:
        sys.exit(
            "probe requires cfg.dynamic_batch_size=True (the calibration is "
            "what produces the table); set it in the config and re-run."
        )
    dm = ExamplesDataModule(cfg)
    dm.setup()
    sampler = dm.train_batch_sampler
    if sampler is None:
        sys.exit("expected train_batch_sampler to be populated after setup()")
    print("calibration table:")
    for n in sorted(sampler._max_b_table):
        print(f"  N={n:>4} → max_B={sampler._max_b_table[n]}")
    print(f"  buckets in train epoch (deterministic estimate): {len(sampler)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("runtime/configs/ml.yaml"))
    parser.add_argument(
        "--compute-lengths-only",
        action="store_true",
        help="Populate <examples_root>/.lengths.json and exit; skip GPU probe.",
    )
    args = parser.parse_args()
    if args.compute_lengths_only:
        sys.exit(_compute_lengths_only(args.config))
    sys.exit(_gpu_probe(args.config))


if __name__ == "__main__":
    main()
