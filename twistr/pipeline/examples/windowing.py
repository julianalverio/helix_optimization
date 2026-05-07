from __future__ import annotations

import hashlib

import numpy as np


def stable_helix_seed(random_seed: int, pdb_id: str, assembly_id: int, helix_index: int) -> int:
    payload = f"{pdb_id}|{assembly_id}|{helix_index}".encode()
    digest = hashlib.sha256(payload).digest()[:8]
    return random_seed ^ int.from_bytes(digest, "big")


def tile_windows(length: int, seed: int, window_min: int, window_max: int) -> list[tuple[int, int]]:
    """Greedy left-to-right window tiling over [0, length-1].

    - length <= window_max: single window covering the full range.
    - length > window_max: sample each window length uniformly in [window_min, window_max],
      advance, continue until remaining <= window_max. Final window takes the remainder.
      If remainder < window_min, merge into previous window.
    Windows are returned as inclusive (start, end) pairs.
    """
    if length < window_min:
        return []
    if length <= window_max:
        return [(0, length - 1)]

    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    windows: list[tuple[int, int]] = []
    pos = 0
    remaining = length
    while remaining > window_max:
        w = int(rng.integers(window_min, window_max + 1))
        windows.append((pos, pos + w - 1))
        pos += w
        remaining -= w
    final_len = remaining
    if final_len < window_min and windows:
        prev_start, _prev_end = windows[-1]
        windows[-1] = (prev_start, length - 1)
    else:
        windows.append((pos, length - 1))
    return windows
