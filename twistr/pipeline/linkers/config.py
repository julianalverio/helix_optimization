from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class LinkerWindow:
    """Where a helix is grafted into the framework chain.

    `cut_before` is the framework PDB author_seq_id of the last residue
    KEPT before the helix insertion. `cut_after` is the first framework
    residue KEPT after the insertion. Framework residues strictly between
    these (if any) are discarded.
    """
    cut_before: int
    cut_after: int


@dataclass(frozen=True)
class LinkerLengths:
    linker1: int      # F1 -> helix1
    linker2: int      # helix1 -> F2
    linker3: int      # F2 -> helix2
    linker4: int      # helix2 -> F3


@dataclass(frozen=True)
class LinkersConfig:
    framework_pdb: str
    helix1_pdb: str
    helix2_pdb: str

    window1: LinkerWindow      # for helix1 (between F1 and F2)
    window2: LinkerWindow      # for helix2 (between F2 and F3)

    lengths: LinkerLengths

    output_dir: str
    rosetta_python: str

    linker_aa_whitelist: str = "AGSDNTPQEKR"
    nstruct: int = 20
    num_trajectory: int = 20
    context_residues: int = 5
    seed: int = 0


def load_linkers_config(path: Path | str) -> LinkersConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(LinkersConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown linkers config keys: {sorted(unknown)}")
    raw["window1"] = LinkerWindow(**raw["window1"])
    raw["window2"] = LinkerWindow(**raw["window2"])
    raw["lengths"] = LinkerLengths(**raw["lengths"])
    return LinkersConfig(**raw)
