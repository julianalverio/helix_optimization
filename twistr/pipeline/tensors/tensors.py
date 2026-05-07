from __future__ import annotations

import io

import gemmi
import numpy as np

from .constants import ATOM14_SLOT_INDEX, RESIDUE_TYPE_INDEX, SS3_NULL, SS8_NULL
from .dssp import SsCodes, SsKey


def _canonical_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [res for res in chain if res.name in RESIDUE_TYPE_INDEX]


def _empty_cofactor_block() -> dict[str, np.ndarray]:
    return {
        "cofactor_coords": np.zeros((0, 3), dtype=np.float16),
        "cofactor_atom_names": np.empty(0, dtype="<U4"),
        "cofactor_elements": np.empty(0, dtype="<U2"),
        "cofactor_residue_names": np.empty(0, dtype="<U3"),
        "cofactor_residue_indices": np.empty(0, dtype=np.int32),
        "cofactor_chain_names": np.empty(0, dtype="<U8"),
    }


def build_atom14(
    chains: list[gemmi.Chain],
    ss_map: dict[SsKey, SsCodes],
    cofactor_block: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray | int]:
    per_chain = [_canonical_residues(chain) for chain in chains]
    n_chains = len(chains)
    n_max = max((len(residues) for residues in per_chain), default=0)

    coordinates = np.zeros((n_chains, n_max, 14, 3), dtype=np.float16)
    atom_mask = np.full((n_chains, n_max, 14), -1, dtype=np.int8)
    residue_index = np.zeros((n_chains, n_max), dtype=np.int32)
    residue_type = np.full((n_chains, n_max), -1, dtype=np.int8)
    ss_3 = np.full((n_chains, n_max), SS3_NULL, dtype=np.int8)
    ss_8 = np.full((n_chains, n_max), SS8_NULL, dtype=np.int8)
    protein_chain_names = np.array([chain.name for chain in chains], dtype="<U8")

    for ci, (chain, residues) in enumerate(zip(chains, per_chain)):
        for ri, res in enumerate(residues):
            rtype_idx = RESIDUE_TYPE_INDEX[res.name]
            slot_lookup = ATOM14_SLOT_INDEX[res.name]
            seq_id = res.seqid.num
            residue_index[ci, ri] = seq_id
            residue_type[ci, ri] = rtype_idx
            atom_mask[ci, ri, :] = 0
            label_seq = res.label_seq if res.label_seq is not None else seq_id
            key = (res.subchain, label_seq)
            if key in ss_map:
                s3, s8 = ss_map[key]
                ss_3[ci, ri] = s3
                ss_8[ci, ri] = s8
            for atom in res:
                slot = slot_lookup.get(atom.name)
                if slot is None:
                    continue
                if atom.occ <= 0.0:
                    continue
                coordinates[ci, ri, slot, 0] = np.float16(atom.pos.x)
                coordinates[ci, ri, slot, 1] = np.float16(atom.pos.y)
                coordinates[ci, ri, slot, 2] = np.float16(atom.pos.z)
                atom_mask[ci, ri, slot] = 1

    out: dict[str, np.ndarray | int] = {
        "n_chains": int(n_chains),
        "n_max_residues": int(n_max),
        "residue_index": residue_index,
        "residue_type": residue_type,
        "ss_3": ss_3,
        "ss_8": ss_8,
        "coordinates": coordinates,
        "atom_mask": atom_mask,
        "protein_chain_names": protein_chain_names,
    }
    out.update(cofactor_block if cofactor_block else _empty_cofactor_block())
    return out


def serialize_npz(tensors: dict[str, np.ndarray | int]) -> bytes:
    buf = io.BytesIO()
    payload = {k: (np.int32(v) if isinstance(v, int) else v) for k, v in tensors.items()}
    np.savez_compressed(buf, **payload)
    return buf.getvalue()
