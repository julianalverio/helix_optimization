from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import pandas as pd

from . import paths
from .config import Config, config_hash

logger = logging.getLogger(__name__)

PROTEIN_POLYMER_TYPES = {gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD}
DNA_POLYMER_TYPES = {gemmi.PolymerType.Dna, gemmi.PolymerType.DnaRnaHybrid}
RNA_POLYMER_TYPES = {gemmi.PolymerType.Rna}

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
STANDARD_NUC = {"A", "C", "G", "T", "U", "DA", "DC", "DG", "DT", "DU"}


@dataclass
class ChainObservation:
    pdb_id: str
    chain_id: str
    entity_id: str | None
    chain_type: str
    seqres_length: int
    observed_length: int
    observed_fraction: float
    is_ca_only: bool
    canonical_sequence: str


@dataclass
class VerifyResult:
    pdb_id: str
    file_path: str
    sha256: str
    parse_ok: bool
    parse_error: str | None = None
    method: str | None = None
    resolution: float | None = None
    r_free: float | None = None
    n_atoms: int = 0
    n_chains: int = 0
    n_protein_chains: int = 0
    has_modified_residues: bool = False
    max_protein_observed_fraction: float | None = None
    min_protein_observed_fraction: float | None = None
    chains: list[ChainObservation] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _polymer_kind(entity: gemmi.Entity) -> str:
    if entity.polymer_type in PROTEIN_POLYMER_TYPES:
        return "protein"
    if entity.polymer_type in DNA_POLYMER_TYPES:
        return "dna"
    if entity.polymer_type in RNA_POLYMER_TYPES:
        return "rna"
    return "other"


def _is_ca_only(polymer: gemmi.ResidueSpan) -> bool:
    if len(polymer) == 0:
        return False
    for res in polymer:
        if len(res) == 1 and res[0].name == "CA":
            continue
        return False
    return True


def _canonical_sequence(full_sequence: list[str]) -> str:
    names = [item.split(",")[0].split(";")[0] for item in full_sequence]
    return gemmi.one_letter_code(names)


def parse_structure(path: Path) -> VerifyResult:
    pdb_id = path.stem.split(".")[0].upper()
    result = VerifyResult(pdb_id=pdb_id, file_path=path.name, sha256="", parse_ok=False)
    try:
        result.sha256 = _sha256_file(path)
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()
        result.parse_ok = True
        result.resolution = float(structure.resolution) if structure.resolution else None

        meta = structure.meta
        if meta.experiments:
            methods = [e.method for e in meta.experiments if e.method]
            if methods:
                result.method = methods[0]
        if meta.refinement:
            r_free = meta.refinement[0].r_free
            if r_free is not None and r_free == r_free:
                result.r_free = float(r_free)

        if len(structure) == 0:
            return result

        model = structure[0]
        atoms = 0
        has_mod = False
        protein_fractions: list[float] = []

        for chain in model:
            polymer = chain.get_polymer()
            entity = structure.get_entity_of(polymer)
            if entity is None:
                continue
            kind = _polymer_kind(entity)
            seqres_len = len(entity.full_sequence)
            observed_len = len(polymer)
            observed_fraction = (observed_len / seqres_len) if seqres_len else 0.0
            canonical = _canonical_sequence(entity.full_sequence) if kind == "protein" else ""
            ca_only = _is_ca_only(polymer)

            for res in polymer:
                atoms += len(res)
                if kind == "protein" and res.name not in STANDARD_AA:
                    has_mod = True
                elif kind in {"dna", "rna"} and res.name not in STANDARD_NUC:
                    has_mod = True

            if kind == "protein":
                result.n_protein_chains += 1
                protein_fractions.append(observed_fraction)

            result.n_chains += 1
            result.chains.append(
                ChainObservation(
                    pdb_id=pdb_id,
                    chain_id=chain.name,
                    entity_id=entity.name,
                    chain_type=kind,
                    seqres_length=seqres_len,
                    observed_length=observed_len,
                    observed_fraction=observed_fraction,
                    is_ca_only=ca_only,
                    canonical_sequence=canonical,
                )
            )

        result.n_atoms = atoms
        result.has_modified_residues = has_mod
        if protein_fractions:
            result.max_protein_observed_fraction = max(protein_fractions)
            result.min_protein_observed_fraction = min(protein_fractions)
        return result
    except Exception as exc:
        result.parse_ok = False
        result.parse_error = f"{type(exc).__name__}: {exc}"
        return result


def _verify_one(path_str: str) -> dict:
    result = parse_structure(Path(path_str))
    return {
        "entry": {
            "pdb_id": result.pdb_id,
            "file_path": result.file_path,
            "sha256": result.sha256,
            "parse_ok": result.parse_ok,
            "parse_error": result.parse_error,
            "method": result.method,
            "resolution": result.resolution,
            "r_free": result.r_free,
            "n_atoms": result.n_atoms,
            "n_chains": result.n_chains,
            "n_protein_chains": result.n_protein_chains,
            "has_modified_residues": result.has_modified_residues,
            "max_protein_observed_fraction": result.max_protein_observed_fraction,
            "min_protein_observed_fraction": result.min_protein_observed_fraction,
        },
        "chains": [c.__dict__ for c in result.chains],
    }


def run_phase_c(
    cfg: Config,
    data_root_path: Path,
    candidates_path: Path,
    workers: int | None = None,
) -> tuple[Path, Path]:
    candidates_df = pd.read_parquet(candidates_path)
    passing = candidates_df[candidates_df["passed_all_filters"]]

    file_paths: list[Path] = []
    for pdb_id in passing["pdb_id"]:
        p = paths.mmcif_abs_path(data_root_path, pdb_id)
        if p.exists():
            file_paths.append(p)

    workers = workers or os.cpu_count() or 4
    entry_rows: list[dict] = []
    chain_rows: list[dict] = []

    if file_paths:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_verify_one, str(p)): p for p in file_paths}
            for future in as_completed(futures):
                source_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("worker crash for %s: %s", source_path.name, exc)
                    pdb_id = source_path.stem.split(".")[0].upper()
                    entry_rows.append({
                        "pdb_id": pdb_id,
                        "file_path": source_path.name,
                        "sha256": "",
                        "parse_ok": False,
                        "parse_error": f"worker_exception:{type(exc).__name__}: {exc}",
                        "method": None,
                        "resolution": None,
                        "r_free": None,
                        "n_atoms": 0,
                        "n_chains": 0,
                        "n_protein_chains": 0,
                        "has_modified_residues": False,
                        "max_protein_observed_fraction": None,
                        "min_protein_observed_fraction": None,
                    })
                    continue
                if not result["entry"]["parse_ok"]:
                    logger.warning(
                        "parse failed for %s: %s",
                        result["entry"]["pdb_id"], result["entry"]["parse_error"],
                    )
                entry_rows.append(result["entry"])
                chain_rows.extend(result["chains"])

    manifests = paths.manifests_dir(data_root_path)
    entries_df = pd.DataFrame(entry_rows)
    chains_df = pd.DataFrame(chain_rows)

    entries_path = manifests / "verify_results.parquet"
    chains_path = manifests / "chain_sequences.parquet"

    tmp1 = entries_path.with_suffix(".parquet.tmp")
    entries_df.to_parquet(tmp1, index=False)
    tmp1.replace(entries_path)

    tmp2 = chains_path.with_suffix(".parquet.tmp")
    chains_df.to_parquet(tmp2, index=False)
    tmp2.replace(chains_path)

    return entries_path, chains_path


def apply_observed_fraction_filter(
    entries_df: pd.DataFrame, min_observed_fraction: float
) -> pd.DataFrame:
    keep = entries_df["max_protein_observed_fraction"].fillna(0.0) >= min_observed_fraction
    return entries_df[keep].copy()
