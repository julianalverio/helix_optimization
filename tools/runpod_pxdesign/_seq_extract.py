"""Extract a single chain's protein sequence from a PDB or CIF file.

Used by `build_msas.py` to derive the input sequence for each chain that
needs an MSA. We hash the target file + chain id for the cache key but
only the actual sequence is what the MMseqs2 server cares about.
"""
from __future__ import annotations

from pathlib import Path

import gemmi


def extract_chain_sequence(target_file: Path | str, chain_id: str) -> str:
    """Return the one-letter amino-acid sequence of `chain_id` in
    `target_file` (.pdb or .cif). Skips heteroatoms and non-standard
    residues that gemmi can't map to a single letter; raises if the
    chain is empty or absent."""
    structure = gemmi.read_structure(str(target_file))
    structure.setup_entities()
    if not structure:
        raise ValueError(f"no models in {target_file}")
    model = structure[0]
    for chain in model:
        if chain.name != chain_id:
            continue
        letters = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is None or not info.is_amino_acid():
                continue
            letters.append(info.one_letter_code.upper())
        if not letters:
            raise ValueError(
                f"chain {chain_id!r} in {target_file} has no amino-acid residues"
            )
        return "".join(letters)
    raise ValueError(
        f"chain {chain_id!r} not found in {target_file}; "
        f"available: {[c.name for c in model]}"
    )
