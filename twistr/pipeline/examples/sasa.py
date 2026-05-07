from __future__ import annotations

import numpy as np

from ..tensors.constants import ATOM14_SLOT_NAMES, RESIDUE_TYPE_NAMES


def _format_atom_name(name: str) -> str:
    """PDB column 13-16 atom name formatting. Single-letter elements (C/N/O/S) get
    a leading space unless the name is 4 chars long."""
    if len(name) >= 4:
        return name[:4]
    return (" " + name).ljust(4)


def _residue_atom_records(
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    residue_type: np.ndarray,
    chain_index: int,
    residue_position: int,
) -> list[tuple[str, str, float, float, float]]:
    rtype = int(residue_type[chain_index, residue_position])
    if rtype < 0:
        return []
    slot_names = ATOM14_SLOT_NAMES[rtype]
    resname = RESIDUE_TYPE_NAMES[rtype]
    records: list[tuple[str, str, float, float, float]] = []
    slots = np.where(atom_mask[chain_index, residue_position] == 1)[0]
    for slot in slots:
        atom_name = slot_names[int(slot)]
        if not atom_name:
            continue
        x = float(coordinates[chain_index, residue_position, slot, 0])
        y = float(coordinates[chain_index, residue_position, slot, 1])
        z = float(coordinates[chain_index, residue_position, slot, 2])
        records.append((atom_name, resname, x, y, z))
    return records


def _chain_label(slot: int) -> str:
    if slot < 26:
        return chr(ord("A") + slot)
    if slot < 26 * 26:
        return chr(ord("A") + slot // 26) + chr(ord("A") + slot % 26)
    raise ValueError(f"chain_slot {slot} too large for label mapping")


def compute_partner_delta_sasa(
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    residue_type: np.ndarray,
    helix_chain_index: int,
    window_residue_positions: list[int],
    partner_chain_indices: list[int],
    chain_real_residues: dict[int, list[int]],
) -> tuple[dict[tuple[int, int], float], bool]:
    """Run freesasa on the full complex and on the complex minus helix-window residues.
    Returns (delta_sasa, sasa_used). delta_sasa maps (partner_chain_index, residue_position)
    to ΔSASA in Å². sasa_used=False signals a freesasa failure; caller should fall back to
    distance-only partner selection."""
    try:
        import freesasa
    except ImportError:
        return {}, False

    try:
        freesasa.setVerbosity(freesasa.silent)
    except Exception:
        pass

    chains_in_complex = [helix_chain_index] + list(partner_chain_indices)
    window_set = set(window_residue_positions)

    atom_specs: list[tuple[int, int, str, str, float, float, float]] = []
    for chain_idx in chains_in_complex:
        for res_pos in chain_real_residues[chain_idx]:
            for atom_name, resname, x, y, z in _residue_atom_records(
                coordinates, atom_mask, residue_type, chain_idx, res_pos,
            ):
                atom_specs.append((chain_idx, res_pos, atom_name, resname, x, y, z))

    def _run(skip_helix_window: bool) -> dict[tuple[int, int], float]:
        structure = freesasa.Structure()
        pairs: list[tuple[int, int]] = []
        for chain_idx, res_pos, atom_name, resname, x, y, z in atom_specs:
            if skip_helix_window and chain_idx == helix_chain_index and res_pos in window_set:
                continue
            structure.addAtom(
                _format_atom_name(atom_name),
                resname,
                str(res_pos),
                _chain_label(chains_in_complex.index(chain_idx)),
                x, y, z,
            )
            pairs.append((chain_idx, res_pos))
        if not pairs:
            return {}
        result = freesasa.calc(structure)
        per_atom = np.array([result.atomArea(i) for i in range(len(pairs))], dtype=np.float64)
        out: dict[tuple[int, int], float] = {}
        for (chain_idx, res_pos), area in zip(pairs, per_atom):
            key = (chain_idx, res_pos)
            out[key] = out.get(key, 0.0) + float(area)
        return out

    try:
        full_sasa = _run(skip_helix_window=False)
        strip_sasa = _run(skip_helix_window=True)
    except Exception:
        return {}, False

    delta: dict[tuple[int, int], float] = {}
    partner_set = set(partner_chain_indices)
    for key, full_val in full_sasa.items():
        chain_idx, _res_pos = key
        if chain_idx not in partner_set:
            continue
        delta[key] = strip_sasa.get(key, 0.0) - full_val
    return delta, True
