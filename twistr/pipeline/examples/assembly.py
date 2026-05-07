from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .constants import BACKBONE_SLOTS, RESIDUE_ONE_LETTER


@dataclass
class ExampleTensors:
    coordinates: np.ndarray
    atom_mask: np.ndarray
    residue_type: np.ndarray
    ss_3: np.ndarray
    ss_8: np.ndarray
    chain_slot: np.ndarray
    seqres_position: np.ndarray
    is_helix: np.ndarray
    is_interface_residue: np.ndarray
    chain_label: np.ndarray
    chain_module2_index: np.ndarray
    chain_role: np.ndarray


def expand_with_context(
    interface_positions: set[int],
    real_positions: list[int],
    context: int,
) -> tuple[set[int], set[int]]:
    """Expand interface_positions by ±context in tensor (real-residue) position.
    Returns (all_positions, interface_positions_only). `real_positions` is the sorted
    list of valid tensor positions for this chain; context expansion is clamped to it."""
    if not interface_positions:
        return set(), set()
    real_index = {p: i for i, p in enumerate(real_positions)}
    expanded: set[int] = set()
    for pos in interface_positions:
        if pos not in real_index:
            continue
        idx = real_index[pos]
        lo = max(0, idx - context)
        hi = min(len(real_positions) - 1, idx + context)
        for k in range(lo, hi + 1):
            expanded.add(real_positions[k])
    return expanded, set(interface_positions) & expanded


def _slice_per_residue(
    m2: dict,
    entries: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    n = len(entries)
    coords = np.zeros((n, 14, 3), dtype=np.float16)
    amask = np.full((n, 14), -1, dtype=np.int8)
    rtype = np.full(n, -1, dtype=np.int8)
    ss3 = np.zeros(n, dtype=np.int8)
    ss8 = np.zeros(n, dtype=np.int8)
    seqres = np.zeros(n, dtype=np.int32)
    for i, (c, r) in enumerate(entries):
        coords[i] = m2["coordinates"][c, r]
        amask[i] = m2["atom_mask"][c, r]
        rtype[i] = m2["residue_type"][c, r]
        ss3[i] = m2["ss_3"][c, r]
        ss8[i] = m2["ss_8"][c, r]
        seqres[i] = m2["residue_index"][c, r]
    return {
        "coordinates": coords, "atom_mask": amask, "residue_type": rtype,
        "ss_3": ss3, "ss_8": ss8, "seqres_position": seqres,
    }


def build_example_tensor(
    m2: dict,
    helix_chain_index: int,
    helix_positions: list[int],
    helix_is_contacting: list[bool],
    partner_chain_indices: list[int],
    partner_positions_by_chain: dict[int, list[int]],
    partner_is_interface_by_chain: dict[int, set[int]],
    protein_chain_names: np.ndarray,
) -> ExampleTensors:
    entries: list[tuple[int, int]] = [(helix_chain_index, p) for p in helix_positions]
    chain_slot_per: list[int] = [0] * len(helix_positions)
    is_helix_per: list[bool] = [True] * len(helix_positions)
    is_interface_per: list[bool] = list(helix_is_contacting)

    for slot_idx, chain_idx in enumerate(partner_chain_indices, start=1):
        positions = sorted(partner_positions_by_chain[chain_idx])
        direct = partner_is_interface_by_chain[chain_idx]
        for p in positions:
            entries.append((chain_idx, p))
            chain_slot_per.append(slot_idx)
            is_helix_per.append(False)
            is_interface_per.append(p in direct)

    sliced = _slice_per_residue(m2, entries)

    chain_order = [helix_chain_index] + list(partner_chain_indices)
    chain_label = np.array(
        [str(protein_chain_names[c]) for c in chain_order], dtype="<U8",
    )
    chain_module2_index = np.array(chain_order, dtype=np.int8)
    chain_role = np.array([0] + [1] * len(partner_chain_indices), dtype=np.int8)

    return ExampleTensors(
        coordinates=sliced["coordinates"],
        atom_mask=sliced["atom_mask"],
        residue_type=sliced["residue_type"],
        ss_3=sliced["ss_3"],
        ss_8=sliced["ss_8"],
        chain_slot=np.array(chain_slot_per, dtype=np.int8),
        seqres_position=sliced["seqres_position"],
        is_helix=np.array(is_helix_per, dtype=bool),
        is_interface_residue=np.array(is_interface_per, dtype=bool),
        chain_label=chain_label,
        chain_module2_index=chain_module2_index,
        chain_role=chain_role,
    )


def completeness_ok(atom_mask: np.ndarray, threshold: float) -> bool:
    n = atom_mask.shape[0]
    if n == 0:
        return False
    bb = atom_mask[:, list(BACKBONE_SLOTS)]
    present = int(np.sum(bb == 1))
    expected = n * len(BACKBONE_SLOTS)
    return (present / expected) >= threshold


def helix_sequence_from_types(residue_types: np.ndarray) -> str:
    chars: list[str] = []
    for t in residue_types:
        t_int = int(t)
        chars.append(RESIDUE_ONE_LETTER[t_int] if 0 <= t_int < len(RESIDUE_ONE_LETTER) else "X")
    return "".join(chars)


def serialize_example_npz(
    tensors: ExampleTensors,
    pdb_id: str,
    assembly_id: int,
    example_id: int,
    helix_seqres_start: int,
    helix_seqres_end: int,
    helix_sequence: str,
    n_helix_contacts: int,
    resolution: float | None,
    r_free: float | None,
    source_method: str | None,
    sasa_used: bool,
) -> bytes:
    payload = {
        "coordinates": tensors.coordinates,
        "atom_mask": tensors.atom_mask,
        "residue_type": tensors.residue_type,
        "ss_3": tensors.ss_3,
        "ss_8": tensors.ss_8,
        "chain_slot": tensors.chain_slot,
        "seqres_position": tensors.seqres_position,
        "is_helix": tensors.is_helix,
        "is_interface_residue": tensors.is_interface_residue,
        "chain_label": tensors.chain_label,
        "chain_module2_index": tensors.chain_module2_index,
        "chain_role": tensors.chain_role,
        "pdb_id": np.array(pdb_id.upper(), dtype="<U4"),
        "assembly_id": np.int32(assembly_id),
        "example_id": np.int32(example_id),
        "helix_seqres_start": np.int32(helix_seqres_start),
        "helix_seqres_end": np.int32(helix_seqres_end),
        "helix_sequence": np.array(helix_sequence, dtype=f"<U{max(1, len(helix_sequence))}"),
        "n_helix_contacts": np.int32(n_helix_contacts),
        "resolution": np.float32(resolution) if resolution is not None else np.float32(np.nan),
        "r_free": np.float32(r_free) if r_free is not None else np.float32(np.nan),
        "source_method": np.array(source_method or "", dtype=f"<U{max(1, len(source_method or ''))}"),
        "sasa_used": np.bool_(sasa_used),
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()
