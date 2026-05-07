from __future__ import annotations

import pytest

from twistr.ml.datasets.val_split import unique_sequence_val_split


def test_no_helix_sequence_overlap_between_val_and_train():
    rel_paths = [f"p{i}" for i in range(20)]
    # 10 unique sequences, each appearing twice.
    helix_sequences = [f"S{i // 2}" for i in range(20)]
    val_idx, train_idx = unique_sequence_val_split(
        rel_paths, helix_sequences, val_count=4, seed=0,
    )
    val_seqs = {helix_sequences[i] for i in val_idx}
    train_seqs = {helix_sequences[i] for i in train_idx}
    assert val_seqs.isdisjoint(train_seqs)


def test_val_sequences_are_unique():
    helix_sequences = [f"S{i // 3}" for i in range(30)]   # 10 sequences × 3 examples
    rel_paths = [f"p{i}" for i in range(30)]
    val_idx, _ = unique_sequence_val_split(
        rel_paths, helix_sequences, val_count=5, seed=42,
    )
    val_seqs = [helix_sequences[i] for i in val_idx]
    assert len(set(val_seqs)) == len(val_seqs)


def test_dropped_count_matches_chosen_sequence_duplicates():
    # 6 sequences with 3 copies each = 18 examples.
    helix_sequences = [f"S{i // 3}" for i in range(18)]
    rel_paths = [f"p{i}" for i in range(18)]
    val_idx, train_idx = unique_sequence_val_split(
        rel_paths, helix_sequences, val_count=2, seed=7,
    )
    # 2 val sequences chosen, each had 3 copies → 1 in val + 2 dropped per sequence.
    # Total: 2 in val, 4 dropped, remaining 12 in train.
    assert len(val_idx) == 2
    assert len(train_idx) == 12
    n_dropped = 18 - len(val_idx) - len(train_idx)
    assert n_dropped == 4


def test_seed_is_deterministic():
    rel_paths = [f"p{i}" for i in range(50)]
    helix_sequences = [f"S{i}" for i in range(50)]
    a = unique_sequence_val_split(rel_paths, helix_sequences, 5, seed=123)
    b = unique_sequence_val_split(rel_paths, helix_sequences, 5, seed=123)
    assert a == b
    c = unique_sequence_val_split(rel_paths, helix_sequences, 5, seed=124)
    assert a != c


def test_val_count_capped_at_unique_sequence_count():
    helix_sequences = ["A", "A", "B", "B", "C", "C"]
    rel_paths = [f"p{i}" for i in range(6)]
    val_idx, train_idx = unique_sequence_val_split(
        rel_paths, helix_sequences, val_count=10, seed=0,
    )
    # Only 3 unique sequences; val cannot exceed 3 even when val_count=10.
    assert len(val_idx) == 3
    # All 3 sequences end up in val → train is empty.
    assert train_idx == []


def test_misaligned_inputs_raises():
    with pytest.raises(ValueError, match="must align"):
        unique_sequence_val_split(["a", "b"], ["S0"], val_count=1, seed=0)
