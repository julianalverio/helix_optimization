from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field

import gemmi
import numpy as np

from .canonicalize import canonicalize_sidechains
from .config import TensorsConfig, cofactor_set, solvent_set
from .constants import RESIDUE_TYPE_INDEX
from .dssp import run_dssp
from .tensors import build_atom14, serialize_npz

logger = logging.getLogger(__name__)

_NUCLEIC_POLYMER_TYPES = {
    gemmi.PolymerType.Dna,
    gemmi.PolymerType.Rna,
    gemmi.PolymerType.DnaRnaHybrid,
}


@dataclass
class EntryResult:
    pdb_id: str
    assembly_id: int
    processing_status: str
    drop_reason: str | None = None
    n_chains_processed: int | None = None
    n_substantive_chains: int | None = None
    tensor_bytes: bytes | None = None
    warnings: list[str] = field(default_factory=list)


def _load_structure(mmcif_bytes: bytes) -> gemmi.Structure:
    text = gzip.decompress(mmcif_bytes).decode("utf-8", errors="replace")
    doc = gemmi.cif.read_string(text)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    structure.setup_entities()
    return structure


def _residue_name(token: str) -> str:
    return token.split(",")[0].split(";")[0].strip("()")


def _early_drops(structure: gemmi.Structure, cfg: TensorsConfig) -> str | None:
    d_aa = set(cfg.d_amino_acid_codes)
    bad_mod = set(cfg.modified_residues_drop_entry)
    for entity in structure.entities:
        if entity.entity_type == gemmi.EntityType.Branched:
            return "contains_glycan"
        if entity.polymer_type in _NUCLEIC_POLYMER_TYPES:
            return "contains_nucleic_acid"
        if entity.polymer_type == gemmi.PolymerType.PeptideD:
            return "contains_d_amino_acid"
        for token in entity.full_sequence:
            name = _residue_name(token)
            if name in d_aa:
                return "contains_d_amino_acid"
            if name in bad_mod:
                return "contains_modified_residue"
    return None


def _expand_assembly(structure: gemmi.Structure, primary_assembly_id: str) -> bool:
    if not primary_assembly_id:
        return True
    try:
        structure.transform_to_assembly(
            str(primary_assembly_id),
            gemmi.HowToNameCopiedChain.Short,
        )
    except Exception as exc:
        logger.debug("assembly expansion failed: %s", exc)
        return False
    return True


def _restrict_to_plan_chains(structure: gemmi.Structure, plan) -> None:
    if plan is None:
        return
    wanted: set[str] = set()
    for entry in plan:
        a1 = entry.get("asym_id_1")
        a2 = entry.get("asym_id_2")
        if a1:
            wanted.add(str(a1))
        if a2:
            wanted.add(str(a2))
    if not wanted:
        return
    for model in structure:
        to_remove = [i for i, chain in enumerate(model) if chain.name not in wanted]
        for i in reversed(to_remove):
            del model[i]


def _strip_solvent_and_hydrogens(structure: gemmi.Structure, solvent_names: frozenset[str]) -> None:
    for model in structure:
        for chain in model:
            for res in chain:
                to_remove_atoms = [i for i, atom in enumerate(res) if atom.element.name in ("H", "D")]
                for i in reversed(to_remove_atoms):
                    del res[i]
            to_remove_res = [i for i, res in enumerate(chain) if res.name in solvent_names]
            for i in reversed(to_remove_res):
                del chain[i]


def _resolve_altlocs(structure: gemmi.Structure) -> None:
    for model in structure:
        for chain in model:
            for res in chain:
                groups: dict[str, float] = {}
                for atom in res:
                    if atom.altloc and atom.altloc not in (" ", "\0"):
                        groups[atom.altloc] = groups.get(atom.altloc, 0.0) + atom.occ
                if not groups:
                    continue
                chosen = min(groups.items(), key=lambda x: (-x[1], x[0]))[0]
                to_remove = [
                    i for i, atom in enumerate(res)
                    if atom.altloc and atom.altloc not in (" ", "\0") and atom.altloc != chosen
                ]
                for i in reversed(to_remove):
                    del res[i]
                for atom in res:
                    if atom.altloc == chosen:
                        atom.altloc = "\0"


def _convert_modified(structure: gemmi.Structure, convert_map: dict[str, dict]) -> None:
    for model in structure:
        for chain in model:
            for res in chain:
                rule = convert_map.get(res.name)
                if rule is None:
                    continue
                renames = rule.get("atom_renames", {}) or {}
                for atom in res:
                    if atom.name in renames:
                        atom.name = renames[atom.name]
                res.name = rule["parent"]
                res.het_flag = "A"


def _heavy_atom_coords(res: gemmi.Residue) -> np.ndarray:
    coords = [
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for atom in res
        if atom.element.name not in ("H", "D")
    ]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(coords, dtype=np.float64)


def _is_protein_residue(res: gemmi.Residue) -> bool:
    return res.name in RESIDUE_TYPE_INDEX or res.name == "UNK"


def _extract_cofactor_block(structure: gemmi.Structure, cofactors: frozenset[str]) -> dict[str, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    atom_names: list[str] = []
    elements: list[str] = []
    res_names: list[str] = []
    res_indices: list[int] = []
    chain_names: list[str] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name not in cofactors:
                    continue
                for atom in res:
                    if atom.element.name in ("H", "D"):
                        continue
                    if atom.occ <= 0.0:
                        continue
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    atom_names.append(atom.name)
                    elements.append(atom.element.name)
                    res_names.append(res.name)
                    res_indices.append(res.seqid.num)
                    chain_names.append(chain.name)
        break
    if not coords:
        return {
            "cofactor_coords": np.zeros((0, 3), dtype=np.float16),
            "cofactor_atom_names": np.empty(0, dtype="<U4"),
            "cofactor_elements": np.empty(0, dtype="<U2"),
            "cofactor_residue_names": np.empty(0, dtype="<U3"),
            "cofactor_residue_indices": np.empty(0, dtype=np.int32),
            "cofactor_chain_names": np.empty(0, dtype="<U8"),
        }
    return {
        "cofactor_coords": np.array(coords, dtype=np.float16),
        "cofactor_atom_names": np.array(atom_names, dtype="<U4"),
        "cofactor_elements": np.array(elements, dtype="<U2"),
        "cofactor_residue_names": np.array(res_names, dtype="<U3"),
        "cofactor_residue_indices": np.array(res_indices, dtype=np.int32),
        "cofactor_chain_names": np.array(chain_names, dtype="<U8"),
    }


def _handle_non_protein(structure: gemmi.Structure, cofactors: frozenset[str]) -> str | None:
    model = structure[0]
    protein: list[tuple[int, str, gemmi.Residue]] = []
    non_protein: list[tuple[int, gemmi.Residue]] = []
    for ci, chain in enumerate(model):
        for res in chain:
            if _is_protein_residue(res):
                protein.append((ci, chain.name, res))
            elif res.name in cofactors:
                continue
            else:
                non_protein.append((ci, res))

    if not non_protein:
        return None

    if len(protein) < 2:
        _strip_non_protein(model)
        return None

    res_coords_by_idx: dict[int, np.ndarray] = {}
    chain_coord_chunks: dict[int, list[np.ndarray]] = {}
    for idx, (ci, _, res) in enumerate(protein):
        c = _heavy_atom_coords(res)
        res_coords_by_idx[idx] = c
        if c.size:
            chain_coord_chunks.setdefault(ci, []).append(c)
    chain_coords: dict[int, np.ndarray] = {
        ci: np.vstack(chunks) for ci, chunks in chain_coord_chunks.items()
    }

    interface_coords_list: list[np.ndarray] = []
    for idx, (ci, _, _) in enumerate(protein):
        res_coords = res_coords_by_idx[idx]
        if res_coords.size == 0:
            continue
        at_interface = False
        for oi, other in chain_coords.items():
            if oi == ci:
                continue
            diffs = res_coords[:, None, :] - other[None, :, :]
            if np.any(np.sum(diffs * diffs, axis=-1) <= 25.0):
                at_interface = True
                break
        if at_interface:
            interface_coords_list.append(res_coords)

    if not interface_coords_list:
        _strip_non_protein(model)
        return None

    interface_coords = np.vstack(interface_coords_list)

    for ci, res in non_protein:
        c = _heavy_atom_coords(res)
        if c.size == 0:
            continue
        diffs = c[:, None, :] - interface_coords[None, :, :]
        dists_sq = np.sum(diffs * diffs, axis=-1)
        if np.any(dists_sq <= 25.0):
            return "non_protein_at_interface"

    _strip_non_protein(model)
    return None


def _strip_non_protein(model: gemmi.Model) -> None:
    for ci in range(len(model) - 1, -1, -1):
        chain = model[ci]
        to_remove = [i for i, res in enumerate(chain) if not _is_protein_residue(res)]
        for i in reversed(to_remove):
            del chain[i]
        if len(chain) == 0:
            del model[ci]


@dataclass
class _ChainStatus:
    chain: gemmi.Chain
    n_obs: int
    unk_frac: float
    length_ok: bool
    unk_ok: bool
    substantive: bool


def _chain_filters(structure: gemmi.Structure, cfg: TensorsConfig) -> tuple[str | None, list[_ChainStatus]]:
    model = structure[0]
    statuses: list[_ChainStatus] = []
    for chain in model:
        residues = [res for res in chain if res.name in RESIDUE_TYPE_INDEX or res.name == "UNK"]
        n_obs = len(residues)
        n_unk = sum(1 for res in residues if res.name == "UNK")
        unk_frac = (n_unk / n_obs) if n_obs else 0.0
        length_ok = n_obs >= cfg.min_observed_residues_per_chain
        unk_ok = unk_frac <= cfg.max_unk_fraction_per_chain
        statuses.append(_ChainStatus(
            chain=chain,
            n_obs=n_obs,
            unk_frac=unk_frac,
            length_ok=length_ok,
            unk_ok=unk_ok,
            substantive=length_ok and unk_ok,
        ))
    n_substantive = sum(1 for s in statuses if s.substantive)
    if n_substantive >= 2:
        return None, statuses
    non_substantive = [s for s in statuses if not s.substantive]
    if non_substantive and all(s.length_ok and not s.unk_ok for s in non_substantive):
        return "unk_dominated_structure", statuses
    return "insufficient_protein_chains_after_processing", statuses


def _entirely_ca_only(structure: gemmi.Structure) -> bool:
    for model in structure:
        for chain in model:
            for res in chain:
                if not _is_protein_residue(res):
                    continue
                for atom in res:
                    if atom.element.name in ("H", "D"):
                        continue
                    if atom.name != "CA":
                        return False
    return True


def process_entry(
    mmcif_bytes: bytes,
    pdb_id: str,
    assembly_id: int,
    m1_meta: dict,
    cfg: TensorsConfig,
) -> EntryResult:
    result = EntryResult(
        pdb_id=pdb_id.upper(),
        assembly_id=int(assembly_id),
        processing_status="dropped",
    )

    try:
        structure = _load_structure(mmcif_bytes)
    except Exception as exc:
        result.drop_reason = "unparseable_mmcif"
        result.warnings.append(f"{type(exc).__name__}: {exc}")
        return result

    drop = _early_drops(structure, cfg)
    if drop:
        result.drop_reason = drop
        return result

    primary_assembly_id = m1_meta.get("primary_assembly_id") or "1"
    large_assembly = bool(m1_meta.get("large_assembly"))
    if not _expand_assembly(structure, primary_assembly_id):
        result.drop_reason = "assembly_expansion_failed"
        return result

    if large_assembly:
        _restrict_to_plan_chains(structure, m1_meta.get("unique_interface_plan"))

    _strip_solvent_and_hydrogens(structure, solvent_set(cfg))
    _resolve_altlocs(structure)
    _convert_modified(structure, cfg.modified_residues_convert)
    canonicalize_sidechains(structure)

    cofactor_names = cofactor_set(cfg)
    cofactor_block = _extract_cofactor_block(structure, cofactor_names)
    drop = _handle_non_protein(structure, cofactor_names)
    if drop:
        result.drop_reason = drop
        return result

    drop, statuses = _chain_filters(structure, cfg)
    if drop:
        result.drop_reason = drop
        return result

    if _entirely_ca_only(structure):
        result.drop_reason = "ca_only_structure"
        return result

    substantive_chains = sorted(
        [s.chain for s in statuses if s.substantive],
        key=lambda c: c.name,
    )

    dssp_outcome = run_dssp(structure, cfg.dssp_executable)
    if not dssp_outcome.ok:
        result.drop_reason = "dssp_failed"
        result.warnings.append(f"dssp: {dssp_outcome.reason}")
        return result

    try:
        tensors = build_atom14(substantive_chains, dssp_outcome.ss_map, cofactor_block)
        result.tensor_bytes = serialize_npz(tensors)
    except Exception as exc:
        result.processing_status = "error"
        result.drop_reason = "processing_error"
        result.warnings.append(f"{type(exc).__name__}: {exc}")
        return result

    result.processing_status = "ok"
    result.drop_reason = None
    result.n_chains_processed = len(statuses)
    result.n_substantive_chains = len(substantive_chains)
    return result
