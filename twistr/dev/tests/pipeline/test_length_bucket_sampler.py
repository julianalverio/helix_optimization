"""LengthBucketBatchSampler — bucket sizing, coverage, ordering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from twistr.pipeline.datasets.batch_sampler import LengthBucketBatchSampler, compute_lengths
from twistr.pipeline.datasets.example_dataset import ExamplesDataset


def _bucket_invariant(buckets: list[list[int]], lengths: list[int], table: dict[int, int]) -> None:
    sweep = sorted(table)

    def cap_for(n: int) -> int:
        for sp in sweep:
            if n <= sp:
                return table[sp]
        return table[sweep[-1]]

    for b in buckets:
        n_max = max(lengths[i] for i in b)
        assert len(b) <= cap_for(n_max), (
            f"bucket of size {len(b)} at N_max={n_max} exceeds cap={cap_for(n_max)}"
        )


def test_standalone_mode_each_index_appears_once_per_epoch():
    lengths = [50] * 10 + [100] * 10 + [150] * 10
    table = {50: 8, 100: 4, 150: 2}
    sampler = LengthBucketBatchSampler(lengths, table, shuffle=False)
    seen: list[int] = []
    buckets: list[list[int]] = []
    for batch in sampler:
        buckets.append(batch)
        seen.extend(batch)
    assert sorted(seen) == list(range(30))
    _bucket_invariant(buckets, lengths, table)


def test_standalone_mode_respects_cap_at_each_n():
    """A new sweep point with a smaller cap forces the prior bucket to close
    before its size exceeds the new cap."""
    lengths = [10] * 6 + [20] * 6 + [30] * 6
    table = {10: 6, 20: 3, 30: 1}
    sampler = LengthBucketBatchSampler(lengths, table, shuffle=False)
    buckets = list(sampler)
    _bucket_invariant(buckets, lengths, table)
    # All-N=30 indices each get their own bucket (cap=1).
    n30_buckets = [b for b in buckets if all(lengths[i] == 30 for i in b)]
    assert len(n30_buckets) == 6
    assert all(len(b) == 1 for b in n30_buckets)


def test_standalone_mode_shuffle_changes_order_but_preserves_buckets():
    lengths = [10, 20, 10, 20, 30, 10, 20, 30, 10, 30]
    table = {10: 4, 20: 2, 30: 1}
    s1 = LengthBucketBatchSampler(lengths, table, shuffle=True, seed=0)
    s2 = LengthBucketBatchSampler(lengths, table, shuffle=True, seed=1)
    b1 = list(s1)
    b2 = list(s2)
    assert sorted(map(tuple, b1)) == sorted(map(tuple, b2))
    # Different seeds should produce different orderings (with overwhelming probability).
    assert b1 != b2 or len(b1) == 1


def test_wrapped_mode_drains_upstream_each_epoch():
    """In wrapped mode, the upstream sampler is iterated fresh per epoch and
    the drawn indices are length-bucketed."""
    lengths = [50, 100, 50, 100, 150, 50, 100, 150]
    table = {50: 4, 100: 2, 150: 1}

    class CountingSampler:
        def __init__(self, indices: list[int]):
            self.indices = indices
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    upstream = CountingSampler([0, 0, 1, 1, 4, 4, 4])  # weighted-with-replacement-style
    sampler = LengthBucketBatchSampler(lengths, table, sampler=upstream)

    e1 = list(sampler)
    e2 = list(sampler)
    assert upstream.calls == 2
    # Each epoch yields all drawn indices once (in some bucketing).
    assert sorted(i for b in e1 for i in b) == sorted(upstream.indices)
    assert sorted(i for b in e2 for i in b) == sorted(upstream.indices)
    _bucket_invariant(e1, lengths, table)
    _bucket_invariant(e2, lengths, table)


def test_wrapped_mode_preserves_duplicate_draws():
    """Weighted-with-replacement can yield the same idx multiple times; the
    sampler must keep all draws (including duplicates)."""
    lengths = [10] * 5
    table = {10: 3}
    upstream = [0, 0, 0, 0, 0]
    sampler = LengthBucketBatchSampler(lengths, table, sampler=upstream)
    drawn = [i for b in sampler for i in b]
    assert drawn.count(0) == 5


def test_len_uses_deterministic_buckets():
    lengths = [10] * 6 + [20] * 6
    table = {10: 6, 20: 2}
    s = LengthBucketBatchSampler(lengths, table, shuffle=False)
    assert len(s) == 1 + 3  # one bucket of 6 N=10's, then three buckets of 2 N=20's


def test_lookup_rounds_n_up_for_intermediate_values():
    """N=70 falls between sweep points 50 and 100; the cap at N=70 should be
    the cap at the next-larger sweep point (100), not at 50."""
    lengths = [70] * 5
    table = {50: 8, 100: 2}
    s = LengthBucketBatchSampler(lengths, table, shuffle=False)
    buckets = list(s)
    # All five N=70 examples should fit in 3 buckets of size ≤2 (cap at N≤100 is 2).
    assert all(len(b) <= 2 for b in buckets)
    assert sum(len(b) for b in buckets) == 5


def test_validates_max_b_table():
    with pytest.raises(ValueError):
        LengthBucketBatchSampler([10], {})
    with pytest.raises(ValueError):
        LengthBucketBatchSampler([10], {10: 0})


def test_compute_lengths_uses_sidecar_cache(tmp_path: Path):
    """First call decompresses chain_slot from npz files; second call reads
    only the sidecar JSON. Verified by deleting the npz files after the
    first call."""
    paths = []
    for i, n in enumerate([5, 10, 7]):
        p = tmp_path / f"{i}.npz"
        np.savez(p, chain_slot=np.zeros(n, dtype=np.int64))
        paths.append(p)
    ds = ExamplesDataset(paths)
    cache_path = tmp_path / "lengths.json"

    lengths1 = compute_lengths(ds, cache_path=cache_path)
    assert lengths1 == [5, 10, 7]
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text())
    assert len(cached) == 3

    # Delete the npz files; the cache should still answer.
    for p in paths:
        p.unlink()
    lengths2 = compute_lengths(ds, cache_path=cache_path)
    assert lengths2 == [5, 10, 7]


def test_compute_lengths_without_cache(tmp_path: Path):
    paths = []
    for i, n in enumerate([3, 8]):
        p = tmp_path / f"{i}.npz"
        np.savez(p, chain_slot=np.zeros(n, dtype=np.int64))
        paths.append(p)
    ds = ExamplesDataset(paths)
    assert compute_lengths(ds) == [3, 8]


def test_examples_dataset_length_method(tmp_path: Path):
    p = tmp_path / "0.npz"
    np.savez(p, chain_slot=np.arange(42, dtype=np.int64))
    ds = ExamplesDataset([p])
    assert ds.length(0) == 42


def test_ddp_each_rank_has_equal_bucket_count():
    """DDP requires every rank to yield the same number of buckets per epoch
    so gradient sync barriers don't deadlock."""
    lengths = [10] * 30 + [20] * 30 + [30] * 30
    table = {10: 6, 20: 3, 30: 1}
    full = LengthBucketBatchSampler(lengths, table)
    expected = len(full) // 4
    for rank in range(4):
        s = LengthBucketBatchSampler(lengths, table, num_replicas=4, rank=rank)
        assert len(s) == expected
        assert sum(1 for _ in s) == expected


def test_ddp_standalone_mode_disjoint_across_ranks():
    """In standalone mode, every example appears on exactly one rank per epoch.
    Aggregating buckets from all ranks must reconstruct a subset of the full
    deterministic bucket set with no duplicates."""
    lengths = list(range(10, 70))
    table = {30: 4, 50: 2, 70: 1}
    world_size = 3
    seen_buckets: list[tuple[int, ...]] = []
    for rank in range(world_size):
        s = LengthBucketBatchSampler(
            lengths, table, num_replicas=world_size, rank=rank, shuffle=False,
        )
        for batch in s:
            seen_buckets.append(tuple(batch))
    # Every bucket each rank yields is one of the deterministic buckets, and
    # no bucket appears on more than one rank in the same epoch.
    full = LengthBucketBatchSampler(lengths, table)
    full_buckets = {tuple(b) for b in full._buckets}
    for b in seen_buckets:
        assert b in full_buckets
    assert len(seen_buckets) == len(set(seen_buckets))


def test_ddp_wrapped_mode_pads_short_yields_to_target():
    """If the upstream draws produce fewer buckets than `buckets_per_rank`,
    the sampler pads with deterministic buckets so DDP sync stays balanced."""
    lengths = list(range(10, 50))
    table = {20: 4, 40: 2, 50: 1}
    full = LengthBucketBatchSampler(lengths, table)
    target = len(full) // 2

    # Upstream that yields a single index → at most 1 bucket from draws.
    s = LengthBucketBatchSampler(
        lengths, table, sampler=[0],
        num_replicas=2, rank=0, seed=0,
    )
    batches = list(s)
    assert len(batches) == target
    # First batch is from the upstream draw; rest are deterministic-bucket pads.
    assert batches[0] == [0]


def test_ddp_wrapped_mode_truncates_long_yields_to_target():
    """If upstream draws produce more buckets than buckets_per_rank, excess
    buckets are dropped (DDP needs a stable count, drop_last semantics)."""
    lengths = [10] * 20
    table = {10: 1}
    s = LengthBucketBatchSampler(
        lengths, table, sampler=list(range(20)),
        num_replicas=4, rank=0, seed=0,
    )
    target = len(s)
    batches = list(s)
    assert len(batches) == target
    assert target == 20 // 4


def test_validates_ddp_args():
    with pytest.raises(ValueError, match="invalid DDP"):
        LengthBucketBatchSampler([10], {10: 1}, num_replicas=0)
    with pytest.raises(ValueError, match="invalid DDP"):
        LengthBucketBatchSampler([10] * 4, {10: 1}, num_replicas=2, rank=2)


def test_validates_buckets_outnumber_ranks():
    """If world_size exceeds the bucket count, no balanced split exists and
    the sampler must refuse construction."""
    with pytest.raises(ValueError, match="only .* buckets across"):
        LengthBucketBatchSampler([10] * 3, {10: 1}, num_replicas=4)
