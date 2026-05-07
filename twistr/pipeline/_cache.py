"""Content-hash cache for the pipeline's slow stages.

Each cached output gets a sidecar with a SHA-1 of every input that affects
it (PDB bytes, chain selection, docker image tag, ScanNet mode, etc.). On
read we recompute the signature and compare; mismatch invalidates the cache.
A missing sidecar is trusted on first read and a sidecar is written so
future invalidations work.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def signature(*parts: str | bytes | Path) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, Path):
            h.update(p.read_bytes())
        elif isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def is_valid(sidecar: Path, expected: str) -> bool:
    if not sidecar.exists():
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(expected)
        return True
    return sidecar.read_text().strip() == expected


def mark(sidecar: Path, sig: str) -> None:
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(sig)
