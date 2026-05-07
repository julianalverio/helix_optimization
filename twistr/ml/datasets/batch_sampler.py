"""Length-aware batch sampling for variable-N protein examples.

The Pairformer's pair tensor and triangle-attention logits dominate memory
(O(B·N_max²·c_z) and O(B·N_max³·H) respectively). With variable-length
proteins padded to the batch's N_max, the right knob is a per-batch
B = max_B(N_max). This module provides:

  - `compute_lengths`: read N for every example via the dataset's
    `length(idx)` accessor, with an optional disk sidecar so subsequent
    runs skip re-decompressing chain_slot from each npz.
  - `LengthBucketBatchSampler`: groups indices into batches sized to the
    (N_max → max_B) lookup table. Two modes:
      * Wrapped — takes an upstream `Iterable[int]` (e.g. a
        `WeightedRandomSampler`); each epoch drains the upstream sampler,
        sorts the drawn indices by N, walks emitting buckets. Preserves
        the upstream sampler's selection semantics (weighting, with/without
        replacement) while sizing each batch's B to its N_max.
      * Standalone — no upstream sampler; sorts the full index set once
        at construction, emits deterministic buckets, shuffles bucket
        order per epoch via a seeded RNG. Each example appears exactly
        once per epoch.

DDP support: pass `num_replicas` (= world size) and `rank` to opt in.
Each rank yields exactly `floor(total_buckets / num_replicas)` buckets per
epoch — equal across ranks — so DDP gradient sync stays balanced. In
standalone mode, the deterministic bucket set is round-robin partitioned
across ranks (each example appears on exactly one rank per epoch). In
wrapped mode, each rank drives its own upstream sampler (callers must
seed those independently per rank); if a rank's draws produce fewer
buckets than the per-rank target, the tail is padded with deterministic
buckets to maintain the count invariant.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from torch.utils.data import Sampler

from .example_dataset import ExamplesDataset


def compute_lengths(
    dataset: ExamplesDataset, cache_path: Path | None = None
) -> list[int]:
    """Return N for every example in `dataset`. If `cache_path` is given, use
    it as a JSON sidecar to skip decompressing chain_slot from npz files seen
    on prior runs.

    Keys are paths relative to `cache_path.parent` so the sidecar is portable
    across machines (e.g. between a Mac that pre-computes lengths and a
    RunPod pod that runs the probe). Paths that fall outside the cache
    directory are stored absolute as a fallback.
    """
    cache: dict[str, int] = {}
    base = cache_path.parent.resolve() if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    # Fast path: most paths share a prefix with cache_path.parent (string
    # equality, no filesystem I/O). Falling back to .resolve() per path is
    # 482K stat round-trips on a network filesystem and stalls startup.
    prefix_str = (
        str(cache_path.parent) + "/" if cache_path is not None else None
    )

    def _key(p: Path) -> str:
        s = str(p)
        if prefix_str is not None and s.startswith(prefix_str):
            return s[len(prefix_str):]
        if base is not None:
            try:
                return str(p.resolve().relative_to(base))
            except ValueError:
                pass
        return str(p.resolve())

    out: list[int] = []
    dirty = False
    for i, p in enumerate(dataset.paths):
        key = _key(Path(p))
        n = cache.get(key)
        if n is None:
            n = dataset.length(i)
            cache[key] = n
            dirty = True
        out.append(n)

    if dirty and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return out


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batches dataset indices into groups whose B fits each batch's N_max.

    With `sampler=None` (standalone mode): sort indices by N once,
    deterministic buckets, shuffled order per epoch.

    With `sampler` provided (wrapped mode): each epoch drains the
    upstream sampler, sorts those indices by N, then emits buckets.

    DDP: pass `num_replicas` (world size) and `rank` (caller's rank).
    Each rank yields exactly `floor(total_buckets / num_replicas)`
    buckets per epoch so gradient sync stays balanced. Default
    (num_replicas=1, rank=0) is single-process and a no-op.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        max_b_table: dict[int, int],
        sampler: Iterable[int] | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not max_b_table:
            raise ValueError("max_b_table must contain at least one entry")
        if min(max_b_table.values()) < 1:
            raise ValueError(f"max_b_table values must be ≥1: {max_b_table}")
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"invalid DDP config: rank={rank}, num_replicas={num_replicas}"
            )
        self.lengths = list(lengths)
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._sweep_points = sorted(max_b_table)
        self._max_b_table = dict(max_b_table)
        self._sorted_indices = sorted(
            range(len(self.lengths)), key=lambda i: self.lengths[i]
        )
        # Build deterministic buckets from the full sorted index set. Used in
        # standalone mode to enumerate, in wrapped mode for tail-padding when
        # weighted draws come up short, and in either mode as a stable __len__
        # — divided evenly across ranks for DDP balance.
        self._buckets: list[list[int]] = list(self._yield_buckets(self._sorted_indices))
        if num_replicas > 1 and len(self._buckets) < num_replicas:
            raise ValueError(
                f"only {len(self._buckets)} buckets across {num_replicas} ranks "
                "— increase dataset size, lower num_replicas, or relax "
                "calibration_n_quantiles to produce more buckets"
            )
        self._buckets_per_rank = len(self._buckets) // num_replicas

    def _max_b_for(self, n: int) -> int:
        """Round n up to the next sweep point and look up max_B. Conservative:
        an N between sweep points uses the next-larger sweep point's smaller
        max_B."""
        for sp in self._sweep_points:
            if n <= sp:
                return self._max_b_table[sp]
        return self._max_b_table[self._sweep_points[-1]]

    def _yield_buckets(self, ordered_indices: Iterable[int]) -> Iterator[list[int]]:
        """Walk an iterable of indices already sorted by N and emit buckets
        sized to max_B at the running N_max."""
        bucket: list[int] = []
        for idx in ordered_indices:
            n = self.lengths[idx]
            cap = self._max_b_for(n)
            if bucket and len(bucket) + 1 > cap:
                yield bucket
                bucket = []
            bucket.append(idx)
        if bucket:
            yield bucket

    def __iter__(self) -> Iterator[list[int]]:
        target = self._buckets_per_rank
        if self.sampler is None:
            # Standalone: shuffle bucket order with a seed shared across ranks
            # (so all ranks agree on the permutation), then take this rank's
            # round-robin stripe truncated to `target` for balance.
            order = list(range(len(self._buckets)))
            if self.shuffle:
                rng = random.Random(self.seed + self._epoch)
                rng.shuffle(order)
            self._epoch += 1
            my_indices = order[self.rank :: self.num_replicas][:target]
            for i in my_indices:
                yield self._buckets[i]
            return
        # Wrapped mode: drain this rank's upstream sampler, sort by N to form
        # length-uniform buckets, then SHUFFLE the bucket order. Without the
        # shuffle, every epoch is a length curriculum that runs short→long;
        # the boundary between "longest of epoch N" and "shortest of epoch N+1"
        # is a step-function change in N (and therefore in per-batch loss
        # magnitudes), which manifests as a loss spike at every epoch
        # boundary. Bucket-internal length uniformity is preserved.
        drawn = list(iter(self.sampler))
        drawn.sort(key=lambda i: self.lengths[i])
        buckets = list(self._yield_buckets(drawn))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(buckets)
        self._epoch += 1
        emitted = 0
        for bucket in buckets:
            if emitted >= target:
                return
            yield bucket
            emitted += 1
        # Pad. Use a per-rank offset into the deterministic bucket list so
        # different ranks pad with different buckets (avoids redundant work).
        for i in range(target - emitted):
            yield self._buckets[(self.rank * target + i) % len(self._buckets)]

    def __len__(self) -> int:
        return self._buckets_per_rank
