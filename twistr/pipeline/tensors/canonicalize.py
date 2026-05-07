from __future__ import annotations

import math

import gemmi

_RULES: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "ARG": (("CD", "NE", "CZ", "NH1"), (("NH1", "NH2"),)),
    "ASP": (("CA", "CB", "CG", "OD1"), (("OD1", "OD2"),)),
    "GLU": (("CB", "CG", "CD", "OE1"), (("OE1", "OE2"),)),
    "PHE": (("CA", "CB", "CG", "CD1"), (("CD1", "CD2"), ("CE1", "CE2"))),
    "TYR": (("CA", "CB", "CG", "CD1"), (("CD1", "CD2"), ("CE1", "CE2"))),
    "LEU": (("CA", "CB", "CG", "CD1"), (("CD1", "CD2"),)),
    "VAL": (("N", "CA", "CB", "CG1"), (("CG1", "CG2"),)),
}


def _dihedral_degrees(a: gemmi.Position, b: gemmi.Position, c: gemmi.Position, d: gemmi.Position) -> float:
    return math.degrees(gemmi.calculate_dihedral(a, b, c, d))


def _find_atom(res: gemmi.Residue, name: str) -> gemmi.Atom | None:
    for atom in res:
        if atom.name == name:
            return atom
    return None


def _swap_positions(atom_a: gemmi.Atom, atom_b: gemmi.Atom) -> None:
    tmp = atom_a.pos
    atom_a.pos = atom_b.pos
    atom_b.pos = tmp


def canonicalize_sidechains(structure: gemmi.Structure) -> None:
    for model in structure:
        for chain in model:
            for res in chain:
                rule = _RULES.get(res.name)
                if rule is None:
                    continue
                dihedral_atoms, swap_pairs = rule
                atoms = [_find_atom(res, name) for name in dihedral_atoms]
                if any(a is None for a in atoms):
                    continue
                angle = _dihedral_degrees(atoms[0].pos, atoms[1].pos, atoms[2].pos, atoms[3].pos)
                if -90.0 <= angle <= 90.0:
                    continue
                for a_name, b_name in swap_pairs:
                    a = _find_atom(res, a_name)
                    b = _find_atom(res, b_name)
                    if a is None or b is None:
                        continue
                    _swap_positions(a, b)
