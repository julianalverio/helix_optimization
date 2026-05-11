from __future__ import annotations

import logging
import time
from pathlib import Path

import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from twistr.pipeline.config import MLConfig

from .batch_sampler import LengthBucketBatchSampler, compute_lengths
from .example_dataset import ExamplesDataset
from .val_split import unique_sequence_val_split

logger = logging.getLogger(__name__)


# Per-field padding fill values. Anything per-residue (first axis = N) gets
# padded along axis 0 up to N_max. atom_mask gets -1 (the spec's "residue does
# not exist at this position" sentinel — see runtime/data/tensors_instructions.md). The
# bool flags go to False, integer ids and coordinates to zero. The companion
# padding_mask (True = real residue) is the canonical gate downstream; per-field
# fills are chosen so existing `atom_mask == 1` / `is_helix` checks naturally
# exclude padded residues without any extra logic.
_PAD_FILL = {
    "coordinates": 0.0,
    "atom_mask": -1,
    "residue_type": 0,
    "chain_slot": 0,
    "is_helix": False,
    "is_interface_residue": False,
}


def pad_collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length per-residue tensors to the batch's max N and emit
    a (B, N_max) bool padding_mask (True = real residue). The residue axis is
    axis 0 in every per-sample tensor."""
    n_max = max(s["is_helix"].shape[0] for s in samples)
    out: dict[str, torch.Tensor] = {}
    for key, fill in _PAD_FILL.items():
        padded = []
        for s in samples:
            t = s[key]
            pad_len = n_max - t.shape[0]
            if pad_len == 0:
                padded.append(t)
                continue
            pad_shape = (pad_len, *t.shape[1:])
            padded.append(torch.cat([t, torch.full(pad_shape, fill, dtype=t.dtype)], dim=0))
        out[key] = torch.stack(padded, dim=0)

    padding_mask = torch.zeros(len(samples), n_max, dtype=torch.bool)
    for i, s in enumerate(samples):
        padding_mask[i, : s["is_helix"].shape[0]] = True
    out["padding_mask"] = padding_mask
    return out


class ExamplesDataModule(pl.LightningDataModule):
    """Reads the examples-pipeline manifest, joins per-example cluster weights,
    splits into train/val, and constructs DataLoaders. Variable-length proteins
    are zero-padded by `pad_collate` to the batch's max N; downstream code reads
    the (B, N) `padding_mask` (True = real) to exclude padded residues from
    attention and loss reductions.

    Train sampling: `WeightedRandomSampler(weights=1/cluster_size,
    replacement=True)` for diversity. When `cfg.dynamic_batch_size` is True
    (default), the weighted sampler is wrapped by `LengthBucketBatchSampler`
    so each batch's B is sized to its N_max via the calibration table; the
    weighted draws still drive *which* indices appear (cluster balance
    preserved).

    Val sampling: standalone length-bucketed (each example once, no weighting)
    when dynamic, else sequential `batch_size`.
    """

    def __init__(self, cfg: MLConfig):
        super().__init__()
        self.cfg = cfg
        self.manifest_path = Path(cfg.manifest_path)
        self.examples_root = Path(cfg.examples_root)
        self.cluster_path = Path(cfg.cluster_path)
        self.train_dataset: ExamplesDataset | None = None
        self.val_dataset: ExamplesDataset | None = None
        self.train_weights: list[float] | None = None
        self.train_batch_sampler: LengthBucketBatchSampler | None = None
        self.val_batch_sampler: LengthBucketBatchSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        t0 = time.time()
        logger.info("reading manifest %s", self.manifest_path)
        manifest = pd.read_parquet(
            self.manifest_path, columns=["path_example", "helix_sequence"],
        )
        clusters = pd.read_parquet(self.cluster_path, columns=["path_example", "weight"])
        rel_paths = manifest["path_example"].tolist()
        helix_sequences = manifest["helix_sequence"].tolist()
        weight_lookup = dict(zip(clusters["path_example"], clusters["weight"]))
        missing = [p for p in rel_paths if p not in weight_lookup]
        if missing:
            raise RuntimeError(
                f"{len(missing)} manifest paths missing from cluster file "
                f"{self.cluster_path}. First few: {missing[:3]}. Re-run "
                f"`python -m twistr.pipeline.datasets.cluster_helices` to refresh."
            )
        logger.info(
            "manifest+clusters loaded in %.1fs (%d examples)",
            time.time() - t0, len(rel_paths),
        )
        t0 = time.time()
        # Split so val helix sequences are disjoint from train (and unique
        # within val) — required by the mutation-sensitivity val metric to
        # actually be measuring sensitivity rather than memorisation.
        val_indices, train_indices = unique_sequence_val_split(
            rel_paths, helix_sequences,
            val_count=self.cfg.val_count, seed=self.cfg.seed,
        )
        n_dropped = len(rel_paths) - len(val_indices) - len(train_indices)
        val_paths = [self.examples_root / rel_paths[i] for i in val_indices]
        train_paths = [self.examples_root / rel_paths[i] for i in train_indices]
        self.val_dataset = ExamplesDataset(val_paths)
        self.train_dataset = ExamplesDataset(train_paths, random_rotate=True)
        self.train_weights = [weight_lookup[rel_paths[i]] for i in train_indices]
        logger.info(
            "datasets built in %.1fs (train=%d, val=%d, dropped=%d duplicate-sequence examples)",
            time.time() - t0, len(self.train_dataset), len(self.val_dataset), n_dropped,
        )

        if self.cfg.dynamic_batch_size:
            self._setup_dynamic_batching()

    def _setup_dynamic_batching(self) -> None:
        from twistr.pipeline.models.lightning_module import ExamplesModule
        from twistr.pipeline.training.batch_calibration import load_or_build_calibration

        t0 = time.time()
        logger.info("computing per-example lengths")
        lengths_cache = self.examples_root / ".lengths.json"
        train_lengths = compute_lengths(self.train_dataset, cache_path=lengths_cache)
        val_lengths = compute_lengths(self.val_dataset, cache_path=lengths_cache)
        logger.info("lengths ready in %.1fs", time.time() - t0)

        probe = ExamplesModule(self.cfg)
        if torch.cuda.is_available():
            probe = probe.cuda()
        try:
            # `data/` is volume-backed (RunPod network-volume symlink at
            # boot — see dev/tools/runpod/train/bootstrap.sh), so `data/cache/`
            # persists across pods. We deliberately keep this *outside*
            # `runtime/data/examples/`: module3 is dataset payload, this is
            # training infrastructure.
            table = load_or_build_calibration(
                probe, train_lengths, self.cfg,
                Path("data") / "cache" / "batch_calibration.json",
            )
        finally:
            del probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # DDP: each rank draws a 1/world_size slice of the per-epoch weighted
        # samples (so cluster-wide expected example frequency matches the
        # single-process case). The upstream WeightedRandomSampler is given a
        # rank-specific torch generator so every rank gets independent draws —
        # otherwise all ranks would replay the same indices.
        trainer = getattr(self, "trainer", None)
        world_size = trainer.world_size if trainer is not None else 1
        rank = trainer.global_rank if trainer is not None else 0
        n_per_rank = max(1, len(self.train_dataset) // world_size)
        weighted_gen = torch.Generator().manual_seed(
            self.cfg.seed * 1_000_003 + rank,
        )
        weighted = WeightedRandomSampler(
            self.train_weights,
            num_samples=n_per_rank,
            replacement=True,
            generator=weighted_gen,
        )
        self.train_batch_sampler = LengthBucketBatchSampler(
            train_lengths, table, sampler=weighted,
            num_replicas=world_size, rank=rank, seed=self.cfg.seed,
        )
        self.val_batch_sampler = LengthBucketBatchSampler(
            val_lengths, table, shuffle=False,
            num_replicas=world_size, rank=rank, seed=self.cfg.seed,
        )

    def train_dataloader(self) -> DataLoader:
        if self.cfg.dynamic_batch_size:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.cfg.num_workers,
                persistent_workers=self.cfg.num_workers > 0,
                collate_fn=pad_collate,
            )
        sampler = WeightedRandomSampler(
            self.train_weights, num_samples=len(self.train_dataset), replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=pad_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.cfg.dynamic_batch_size:
            return DataLoader(
                self.val_dataset,
                batch_sampler=self.val_batch_sampler,
                num_workers=self.cfg.num_workers,
                persistent_workers=self.cfg.num_workers > 0,
                collate_fn=pad_collate,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=pad_collate,
        )
