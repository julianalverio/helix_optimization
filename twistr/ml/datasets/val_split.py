from __future__ import annotations

import random


def unique_sequence_val_split(
    rel_paths: list[str],
    helix_sequences: list[str],
    val_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Carve a val set whose helix sequences are disjoint from train.

    Guarantees on the returned ``(val_indices, train_indices)`` over the
    input ``rel_paths`` / ``helix_sequences`` arrays:

      1. Every val example has a distinct ``helix_sequence`` — no two val
         examples share a helix sequence.
      2. No train example shares its ``helix_sequence`` with any val
         example.

    To keep both invariants simultaneously, examples whose helix sequence
    is *chosen* for val but which are not the single val representative
    of that sequence are **dropped** (excluded from both sides). Putting
    them in train would re-leak the held-out sequence; putting them in
    val would violate (1). The drop count is small in practice — on
    module 3 about 1000 chosen sequences shed ~1.5K duplicate examples
    from a 424K pool.

    Motivation: the val-time mutation-sensitivity metric only means
    something if the val helices are not literally present in train.
    Cluster-level generalisation is intentionally *not* a goal here;
    production training will see all PDB helices and inference is
    in-distribution.
    """
    if len(rel_paths) != len(helix_sequences):
        raise ValueError(
            f"rel_paths ({len(rel_paths)}) and helix_sequences "
            f"({len(helix_sequences)}) must align"
        )
    seq_to_first_idx: dict[str, int] = {}
    for i, s in enumerate(helix_sequences):
        seq_to_first_idx.setdefault(s, i)
    sequences = list(seq_to_first_idx.keys())
    rng = random.Random(seed)
    rng.shuffle(sequences)
    n_val = min(val_count, len(sequences))
    val_seqs = set(sequences[:n_val])
    val_indices = [seq_to_first_idx[s] for s in sequences[:n_val]]
    train_indices = [
        i for i, s in enumerate(helix_sequences) if s not in val_seqs
    ]
    return val_indices, train_indices
