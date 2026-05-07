from __future__ import annotations

import io
from pathlib import Path

import numpy as np

RESIDUE_TYPE_NAMES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)

RESIDUE_TYPE_INDEX: dict[str, int] = {name: i for i, name in enumerate(RESIDUE_TYPE_NAMES)}

_ATOM14 = {
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
}

ATOM14_SLOT_NAMES: tuple[tuple[str, ...], ...] = tuple(
    _ATOM14[name] + ("",) * (14 - len(_ATOM14[name])) for name in RESIDUE_TYPE_NAMES
)

ATOM14_SLOT_INDEX: dict[str, dict[str, int]] = {
    name: {atom: i for i, atom in enumerate(_ATOM14[name])} for name in RESIDUE_TYPE_NAMES
}

SS8_CODES: tuple[str, ...] = ("H", "G", "I", "E", "B", "T", "S", "-", "?")
SS3_CODES: tuple[str, ...] = ("H", "E", "C", "-")

SS8_NULL = 8
SS3_NULL = 3

DSSP_CHAR_TO_SS8: dict[str, int] = {
    "H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, " ": 7,
}

SS8_TO_SS3: tuple[int, ...] = (0, 0, 0, 1, 1, 2, 2, 2, 3)


def atom14_slot_names_array() -> np.ndarray:
    return np.array(ATOM14_SLOT_NAMES, dtype="<U4")


def write_constants_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez(
            f,
            residue_type_names=np.array(RESIDUE_TYPE_NAMES, dtype="<U3"),
            atom14_slot_names=atom14_slot_names_array(),
            ss_3_codes=np.array(SS3_CODES, dtype="<U1"),
            ss_8_codes=np.array(SS8_CODES, dtype="<U1"),
        )
    tmp.replace(path)


def constants_bytes() -> bytes:
    buf = io.BytesIO()
    np.savez(
        buf,
        residue_type_names=np.array(RESIDUE_TYPE_NAMES, dtype="<U3"),
        atom14_slot_names=atom14_slot_names_array(),
        ss_3_codes=np.array(SS3_CODES, dtype="<U1"),
        ss_8_codes=np.array(SS8_CODES, dtype="<U1"),
    )
    return buf.getvalue()
