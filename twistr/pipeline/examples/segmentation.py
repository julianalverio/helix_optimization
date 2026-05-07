from __future__ import annotations

import numpy as np

from .constants import SS8_H, SS8_NULL_SENTINEL, SS8_SMOOTHABLE


def smooth_ss8(ss8: np.ndarray) -> np.ndarray:
    """Rewrite 1-2 residue {G,I,T,S} runs embedded between H to H. Returns a copy."""
    out = ss8.copy()
    n = len(out)
    i = 0
    while i < n:
        if int(out[i]) in SS8_SMOOTHABLE:
            j = i
            while j < n and int(out[j]) in SS8_SMOOTHABLE:
                j += 1
            run_len = j - i
            if run_len <= 2 and i > 0 and j < n and int(out[i - 1]) == SS8_H and int(out[j]) == SS8_H:
                out[i:j] = SS8_H
            i = j
        else:
            i += 1
    return out


def find_helix_segments(smoothed_ss8: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive spans of contiguous H runs >= min_length."""
    spans: list[tuple[int, int]] = []
    n = len(smoothed_ss8)
    i = 0
    while i < n:
        if int(smoothed_ss8[i]) == SS8_H:
            j = i
            while j < n and int(smoothed_ss8[j]) == SS8_H:
                j += 1
            if j - i >= min_length:
                spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def merge_by_gap(is_contacting: np.ndarray, segment_start: int, segment_end: int,
                 max_gap: int) -> list[tuple[int, int]]:
    """Walk is_contacting inside [segment_start, segment_end]. Build True-runs.
    Merge adjacent True-runs when the gap of False-residues between them is <= max_gap.
    Returns inclusive (start, end) spans in absolute position."""
    runs: list[tuple[int, int]] = []
    i = segment_start
    while i <= segment_end:
        if bool(is_contacting[i]):
            j = i
            while j <= segment_end and bool(is_contacting[j]):
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not runs:
        return []

    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def filter_by_length(spans: list[tuple[int, int]], min_length: int) -> list[tuple[int, int]]:
    return [(s, e) for s, e in spans if (e - s + 1) >= min_length]


def geometric_helix_ss8(
    ca_coords: np.ndarray,
    ca_present: np.ndarray,
) -> np.ndarray:
    """Detect alpha-helical residues from CA geometry. Used as fallback when
    Module 2's DSSP codes are null. Returns an ss_8 array where 0 marks helix
    and 7 ('-') marks non-helix. `ca_coords` is (n, 3); `ca_present` is (n,) bool."""
    from .constants import (
        HELIX_CA_I_I3_MAX,
        HELIX_CA_I_I3_MIN,
        HELIX_CA_I_I4_MAX,
        HELIX_CA_I_I4_MIN,
    )

    n = ca_coords.shape[0]
    out = np.full(n, 7, dtype=np.int8)
    if n < 5:
        return out
    contributing = np.zeros(n, dtype=bool)
    for i in range(n - 3):
        if not (ca_present[i] and ca_present[i + 3]):
            continue
        d3 = float(np.linalg.norm(ca_coords[i] - ca_coords[i + 3]))
        if not (HELIX_CA_I_I3_MIN <= d3 <= HELIX_CA_I_I3_MAX):
            continue
        if i + 4 < n and ca_present[i + 4]:
            d4 = float(np.linalg.norm(ca_coords[i] - ca_coords[i + 4]))
            if not (HELIX_CA_I_I4_MIN <= d4 <= HELIX_CA_I_I4_MAX):
                continue
        for k in (i, i + 1, i + 2, i + 3):
            contributing[k] = True
    out[contributing] = SS8_H
    return out


def is_ss8_effectively_null(ss8: np.ndarray) -> bool:
    """Return True if all ss_8 values are the null sentinel — i.e., Module 2 produced
    no usable SS codes and we must fall back to geometric detection."""
    return bool(np.all(ss8 == SS8_NULL_SENTINEL))
