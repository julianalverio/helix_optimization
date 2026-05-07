import math
from pathlib import Path

import gemmi
import pytest

from twistr.pipeline.tensors.canonicalize import canonicalize_sidechains

FIXTURE = Path(__file__).parent / "fixtures" / "mmCIF" / "1BRS.cif.gz"


def _residue_atoms(res):
    return {atom.name: atom.pos for atom in res}


def _dihedral(positions, names):
    return math.degrees(gemmi.calculate_dihedral(*(positions[n] for n in names)))


def _load_fixture():
    return gemmi.read_structure(str(FIXTURE))


RULES = {
    "ARG": ("CD", "NE", "CZ", "NH1"),
    "ASP": ("CA", "CB", "CG", "OD1"),
    "GLU": ("CB", "CG", "CD", "OE1"),
    "LEU": ("CA", "CB", "CG", "CD1"),
    "PHE": ("CA", "CB", "CG", "CD1"),
    "TYR": ("CA", "CB", "CG", "CD1"),
    "VAL": ("N", "CA", "CB", "CG1"),
}


@pytest.mark.parametrize("res_name", sorted(RULES))
def test_canonicalize_puts_dihedral_in_canonical_range(res_name):
    structure = _load_fixture()
    canonicalize_sidechains(structure)
    atom_names = RULES[res_name]
    checked = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name != res_name:
                    continue
                pos = _residue_atoms(res)
                if not all(n in pos for n in atom_names):
                    continue
                angle = _dihedral(pos, atom_names)
                assert -90.0 <= angle <= 90.0, (
                    f"{res_name} {res.seqid.num} chi dihedral {angle:.1f} out of range"
                )
                checked += 1
    assert checked > 0, f"no {res_name} residues found with full atom set"


def test_canonicalize_is_idempotent():
    structure = _load_fixture()
    canonicalize_sidechains(structure)
    snapshot: list[tuple[str, int, str, tuple[float, float, float]]] = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    snapshot.append((chain.name, res.seqid.num, atom.name,
                                     (atom.pos.x, atom.pos.y, atom.pos.z)))
    canonicalize_sidechains(structure)
    second: list[tuple[str, int, str, tuple[float, float, float]]] = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    second.append((chain.name, res.seqid.num, atom.name,
                                   (atom.pos.x, atom.pos.y, atom.pos.z)))
    assert snapshot == second


def test_canonicalize_does_not_swap_his():
    structure = _load_fixture()
    before = {}
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name != "HIS":
                    continue
                for atom in res:
                    if atom.name in ("ND1", "NE2"):
                        before[(chain.name, res.seqid.num, atom.name)] = (
                            atom.pos.x, atom.pos.y, atom.pos.z,
                        )
    canonicalize_sidechains(structure)
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name != "HIS":
                    continue
                for atom in res:
                    if atom.name in ("ND1", "NE2"):
                        key = (chain.name, res.seqid.num, atom.name)
                        assert before[key] == (atom.pos.x, atom.pos.y, atom.pos.z)
