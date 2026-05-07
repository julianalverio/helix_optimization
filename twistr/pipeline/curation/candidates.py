from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ... import paths
from ...config import DEV_IDS, Config, config_hash
from . import interfaces, rcsb
from .obsolete import ObsoleteEntry, fetch_obsolete, parse_obsolete, resolve_redirect

logger = logging.getLogger(__name__)

FILTER_COLUMN_TO_NAME = {
    "passed_status_filter": "status",
    "passed_method_filter": "method",
    "passed_resolution_filter": "resolution",
    "passed_rfree_filter": "rfree",
    "passed_chains_filter": "chains",
    "passed_protein_chain_filter": "protein_chain",
    "passed_protein_length_filter": "protein_length",
    "passed_date_filter": "date",
    "passed_size_cap_filter": "size_cap",
}

PROTEIN_POLY_TYPES = {"polypeptide(L)", "polypeptide(D)"}
DNA_POLY_TYPES = {"polydeoxyribonucleotide"}
RNA_POLY_TYPES = {"polyribonucleotide"}
DISALLOWED_METHODS = {
    "SOLUTION NMR",
    "SOLID-STATE NMR",
    "NEUTRON DIFFRACTION",
    "FIBER DIFFRACTION",
    "SOLUTION SCATTERING",
    "THEORETICAL MODEL",
    "INTEGRATIVE",
    "POWDER DIFFRACTION",
}


@dataclass
class CandidateRow:
    pdb_id: str
    obsoleted_from: str | None
    status: str | None
    method: str | None
    multi_method: bool
    resolution: float | None
    r_free: float | None
    rfree_missing: bool
    deposition_date: str | None
    release_date: str | None
    n_polymer_entities: int | None
    n_instantiated_polymer_chains: int | None
    primary_assembly_id: str | None
    has_protein: bool
    has_dna: bool
    has_rna: bool
    has_ligands: bool
    max_protein_seqres_length: int | None
    min_protein_seqres_length: int | None
    has_short_peptide: bool
    total_assembly_residues: int | None
    large_assembly: bool
    title: str | None
    passed_status_filter: bool
    passed_method_filter: bool
    passed_resolution_filter: bool
    passed_rfree_filter: bool
    passed_chains_filter: bool
    passed_protein_chain_filter: bool
    passed_protein_length_filter: bool
    passed_date_filter: bool
    passed_size_cap_filter: bool
    passed_all_filters: bool
    phase_a_drop_reason: str | None


def resolve_candidate_ids(
    input_ids: list[str], obsolete_map: dict[str, ObsoleteEntry]
) -> tuple[list[str], dict[str, str], list[str]]:
    redirect_map: dict[str, str] = {}
    resolved: list[str] = []
    dropped_obsolete: list[str] = []
    seen: set[str] = set()
    for original in input_ids:
        original = original.upper()
        if original in obsolete_map:
            replacement = resolve_redirect(original, obsolete_map)
            if replacement is None:
                dropped_obsolete.append(original)
                continue
            if replacement != original:
                redirect_map[replacement] = original
            final = replacement
        else:
            final = original
        if final in seen:
            continue
        seen.add(final)
        resolved.append(final)
    return resolved, redirect_map, dropped_obsolete


def _pick_primary_assembly(assemblies: list[dict]) -> dict | None:
    if not assemblies:
        return None
    for asm in assemblies:
        gen_list = asm.get("pdbx_struct_assembly_gen") or []
        for gen in gen_list:
            if str(gen.get("assembly_id")) == "1":
                return asm
    return assemblies[0]


def _assembly_id(asm: dict) -> str | None:
    gen_list = asm.get("pdbx_struct_assembly_gen") or []
    if gen_list:
        return str(gen_list[0].get("assembly_id")) if gen_list[0].get("assembly_id") is not None else None
    return None


def _protein_chain_seqres_lengths(polymer_entities: list[dict]) -> list[int]:
    lengths: list[int] = []
    for pe in polymer_entities:
        ep = pe.get("entity_poly") or {}
        if ep.get("type") in PROTEIN_POLY_TYPES:
            length = ep.get("rcsb_sample_sequence_length")
            if length is not None:
                lengths.append(int(length))
    return lengths


def _has_polymer_of_types(polymer_entities: list[dict], types: set[str]) -> bool:
    for pe in polymer_entities:
        ep = pe.get("entity_poly") or {}
        if ep.get("type") in types:
            return True
    return False


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None


def build_candidate_row(
    entry: dict, obsoleted_from: str | None, cfg: Config
) -> CandidateRow:
    pdb_id = entry["rcsb_id"].upper()
    exptl = entry.get("exptl") or []
    methods = [m.get("method") for m in exptl if m.get("method")]
    method = methods[0] if methods else None
    multi_method = len(methods) > 1

    refine_list = entry.get("refine") or []
    refine = refine_list[0] if refine_list else {}
    r_free = refine.get("ls_R_factor_R_free")
    resolution = refine.get("ls_d_res_high")

    em_list = entry.get("em_3d_reconstruction") or []
    if resolution is None and em_list:
        em_res = [e.get("resolution") for e in em_list if e.get("resolution") is not None]
        if em_res:
            resolution = min(em_res)

    if resolution is None:
        entry_info = entry.get("rcsb_entry_info") or {}
        combined = entry_info.get("resolution_combined") or []
        if combined:
            resolution = combined[0]

    accession = entry.get("rcsb_accession_info") or {}
    status = accession.get("status_code")
    deposit = accession.get("deposit_date")
    release = accession.get("initial_release_date")

    entry_info = entry.get("rcsb_entry_info") or {}
    n_polymer_entities = entry_info.get("polymer_entity_count")
    has_ligands = (entry_info.get("nonpolymer_entity_count") or 0) > 0

    assemblies = entry.get("assemblies") or []
    primary = _pick_primary_assembly(assemblies)
    primary_assembly_id = _assembly_id(primary) if primary else None
    asm_info = (primary or {}).get("rcsb_assembly_info") or {}
    n_instantiated = asm_info.get("polymer_entity_instance_count")
    total_assembly_residues = entry_info.get("deposited_polymer_monomer_count")

    polymer_entities = entry.get("polymer_entities") or []
    has_protein = _has_polymer_of_types(polymer_entities, PROTEIN_POLY_TYPES)
    has_dna = _has_polymer_of_types(polymer_entities, DNA_POLY_TYPES)
    has_rna = _has_polymer_of_types(polymer_entities, RNA_POLY_TYPES)
    protein_lengths = _protein_chain_seqres_lengths(polymer_entities)
    max_protein = max(protein_lengths) if protein_lengths else None
    min_protein = min(protein_lengths) if protein_lengths else None
    has_short_peptide = any(
        (pe.get("entity_poly") or {}).get("rcsb_sample_sequence_length") is not None
        and (pe.get("entity_poly") or {}).get("rcsb_sample_sequence_length") < cfg.min_protein_chain_length
        for pe in polymer_entities
    )

    large_assembly = (
        n_instantiated is not None
        and n_instantiated >= cfg.large_assembly_chain_threshold
    )

    struct = entry.get("struct") or {}
    title = struct.get("title")

    passed_status = status in cfg.status_allowed if status is not None else False

    allowed = set(cfg.methods_allowed)
    if not methods:
        passed_method = False
    elif any(m in DISALLOWED_METHODS for m in methods):
        passed_method = False
    else:
        passed_method = any(m in allowed for m in methods)

    if method == "X-RAY DIFFRACTION":
        passed_resolution = resolution is not None and resolution <= cfg.resolution_max_xray
    elif method == "ELECTRON MICROSCOPY":
        passed_resolution = resolution is not None and resolution <= cfg.resolution_max_em
    else:
        passed_resolution = False

    rfree_missing = method == "X-RAY DIFFRACTION" and r_free is None
    if method == "X-RAY DIFFRACTION":
        if r_free is None:
            passed_rfree = cfg.r_free_missing_action == "keep_and_tag"
        else:
            passed_rfree = r_free <= cfg.r_free_max_xray
    else:
        passed_rfree = True

    passed_chains = (
        n_instantiated is not None
        and n_instantiated >= cfg.min_instantiated_polymer_chains
    )

    passed_protein_chain = has_protein if cfg.require_protein_chain else True

    passed_protein_length = (
        max_protein is not None and max_protein >= cfg.min_protein_chain_length
    )

    deposit_date = _parse_date(deposit)
    release_date = _parse_date(release)
    passed_date = True
    if cfg.deposition_date_min and (deposit_date is None or deposit_date < cfg.deposition_date_min):
        passed_date = False
    if cfg.deposition_date_max and (deposit_date is None or deposit_date > cfg.deposition_date_max):
        passed_date = False
    if cfg.release_date_min and (release_date is None or release_date < cfg.release_date_min):
        passed_date = False
    if cfg.release_date_max and (release_date is None or release_date > cfg.release_date_max):
        passed_date = False

    if cfg.hard_cap_total_residues is None:
        passed_size_cap = True
    else:
        passed_size_cap = (
            total_assembly_residues is not None
            and total_assembly_residues <= cfg.hard_cap_total_residues
        )

    filter_results = {
        "passed_status_filter": passed_status,
        "passed_method_filter": passed_method,
        "passed_resolution_filter": passed_resolution,
        "passed_rfree_filter": passed_rfree,
        "passed_chains_filter": passed_chains,
        "passed_protein_chain_filter": passed_protein_chain,
        "passed_protein_length_filter": passed_protein_length,
        "passed_date_filter": passed_date,
        "passed_size_cap_filter": passed_size_cap,
    }
    passed_all = all(filter_results.values())
    failed = [FILTER_COLUMN_TO_NAME[col] for col, ok in filter_results.items() if not ok]
    phase_a_drop_reason = f"filter:{','.join(failed)}" if failed else None

    return CandidateRow(
        pdb_id=pdb_id,
        obsoleted_from=obsoleted_from,
        status=status,
        method=method,
        multi_method=multi_method,
        resolution=float(resolution) if resolution is not None else None,
        r_free=float(r_free) if r_free is not None else None,
        rfree_missing=rfree_missing,
        deposition_date=deposit_date.isoformat() if deposit_date else None,
        release_date=release_date.isoformat() if release_date else None,
        n_polymer_entities=n_polymer_entities,
        n_instantiated_polymer_chains=n_instantiated,
        primary_assembly_id=primary_assembly_id,
        has_protein=has_protein,
        has_dna=has_dna,
        has_rna=has_rna,
        has_ligands=has_ligands,
        max_protein_seqres_length=max_protein,
        min_protein_seqres_length=min_protein,
        has_short_peptide=has_short_peptide,
        total_assembly_residues=total_assembly_residues,
        large_assembly=large_assembly,
        title=title,
        passed_status_filter=passed_status,
        passed_method_filter=passed_method,
        passed_resolution_filter=passed_resolution,
        passed_rfree_filter=passed_rfree,
        passed_chains_filter=passed_chains,
        passed_protein_chain_filter=passed_protein_chain,
        passed_protein_length_filter=passed_protein_length,
        passed_date_filter=passed_date,
        passed_size_cap_filter=passed_size_cap,
        passed_all_filters=passed_all,
        phase_a_drop_reason=phase_a_drop_reason,
    )


def _stub_row(pdb_id: str, drop_reason: str, obsoleted_from: str | None = None) -> CandidateRow:
    return CandidateRow(
        pdb_id=pdb_id,
        obsoleted_from=obsoleted_from,
        status=None,
        method=None,
        multi_method=False,
        resolution=None,
        r_free=None,
        rfree_missing=False,
        deposition_date=None,
        release_date=None,
        n_polymer_entities=None,
        n_instantiated_polymer_chains=None,
        primary_assembly_id=None,
        has_protein=False,
        has_dna=False,
        has_rna=False,
        has_ligands=False,
        max_protein_seqres_length=None,
        min_protein_seqres_length=None,
        has_short_peptide=False,
        total_assembly_residues=None,
        large_assembly=False,
        title=None,
        passed_status_filter=False,
        passed_method_filter=False,
        passed_resolution_filter=False,
        passed_rfree_filter=False,
        passed_chains_filter=False,
        passed_protein_chain_filter=False,
        passed_protein_length_filter=False,
        passed_date_filter=False,
        passed_size_cap_filter=False,
        passed_all_filters=False,
        phase_a_drop_reason=drop_reason,
    )


def select_input_ids(session, full_scale: bool) -> list[str]:
    if full_scale:
        return rcsb.fetch_all_released_ids(session)
    all_ids = rcsb.fetch_all_released_ids(session)
    rng = random.Random(42)
    sample = rng.sample(all_ids, min(100, len(all_ids)))
    merged = list(dict.fromkeys(DEV_IDS + sample))
    return merged


def run_phase_a(
    cfg: Config,
    data_root_path: Path,
    full_scale: bool,
    snapshot_date: datetime,
) -> Path:
    session = rcsb.build_session()
    input_ids = select_input_ids(session, full_scale=full_scale)
    return run_phase_a_on_ids(
        cfg, data_root_path, input_ids, snapshot_date, session=session
    )


def run_phase_a_on_ids(
    cfg: Config,
    data_root_path: Path,
    input_ids: list[str],
    snapshot_date: datetime,
    session=None,
    obsolete_map: dict[str, ObsoleteEntry] | None = None,
) -> Path:
    paths.ensure_dirs(data_root_path)
    manifests = paths.manifests_dir(data_root_path)
    aux = paths.aux_dir(data_root_path)
    cache = paths.cache_dir(data_root_path, config_hash(cfg), "phase_a")

    if session is None:
        session = rcsb.build_session()

    if obsolete_map is None:
        obsolete_path = aux / f"obsolete-{snapshot_date.date().isoformat()}.dat"
        if not obsolete_path.exists():
            fetch_obsolete(session, obsolete_path)
        obsolete_map = parse_obsolete(obsolete_path)

    resolved_ids, redirect_map, dropped_obsolete = resolve_candidate_ids(input_ids, obsolete_map)
    logger.info(
        "phase A inputs: %d selected, %d resolved, %d obsolete-dead-end",
        len(input_ids), len(resolved_ids), len(dropped_obsolete),
    )
    for pdb_id in dropped_obsolete:
        logger.warning("obsolete chain dead-ends: %s", pdb_id)

    metadata, failed_metadata_ids = rcsb.fetch_metadata(session, resolved_ids, cache_dir=cache)
    entries_by_id = {e["rcsb_id"].upper(): e for e in metadata}
    failed_metadata_set = set(failed_metadata_ids)
    missing_metadata = [
        pid for pid in resolved_ids
        if pid not in entries_by_id and pid not in failed_metadata_set
    ]
    for pdb_id in missing_metadata:
        logger.warning("metadata missing for %s (no row in GraphQL response)", pdb_id)

    rows: list[CandidateRow] = []
    for pdb_id in dropped_obsolete:
        rows.append(_stub_row(pdb_id, "obsolete_no_replacement"))
    for pdb_id in failed_metadata_set:
        rows.append(_stub_row(pdb_id, "metadata_error"))
    for pdb_id in missing_metadata:
        rows.append(_stub_row(pdb_id, "metadata_missing"))
    for pdb_id in resolved_ids:
        entry = entries_by_id.get(pdb_id)
        if entry is None:
            continue
        obsoleted_from = redirect_map.get(pdb_id)
        rows.append(build_candidate_row(entry, obsoleted_from, cfg))
    logger.info(
        "phase A filters: %d candidates built, %d passed all filters",
        len(entries_by_id), sum(1 for r in rows if r.passed_all_filters),
    )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["unique_interface_plan"] = None
    df["n_unique_interfaces"] = 0
    if not df.empty:
        large_mask = df["large_assembly"] & df["passed_all_filters"]
        for idx in df.index[large_mask]:
            pdb_id = df.at[idx, "pdb_id"]
            assembly_id = df.at[idx, "primary_assembly_id"] or "1"
            try:
                plan = interfaces.fetch_unique_interfaces(session, pdb_id, str(assembly_id))
            except interfaces.InterfaceFetchError as exc:
                logger.warning("interface plan failed for %s: %s", pdb_id, exc)
                df.at[idx, "passed_all_filters"] = False
                df.at[idx, "phase_a_drop_reason"] = "interface_fetch_error"
                continue
            records = interfaces.plan_to_records(plan)
            df.at[idx, "unique_interface_plan"] = records
            df.at[idx, "n_unique_interfaces"] = len(records)

    out_path = manifests / "candidates.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    return out_path
