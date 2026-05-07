from __future__ import annotations

import io
import traceback
from dataclasses import dataclass, field

import numpy as np

from .assembly import (
    ExampleTensors,
    build_example_tensor,
    completeness_ok,
    expand_with_context,
    helix_sequence_from_types,
    serialize_example_npz,
)
from .config import ExamplesConfig
from .contacts import (
    build_spatial_index,
    distance_interface_partners,
    mark_contacting_residues,
    partner_chains_for_window,
)
from .sasa import compute_partner_delta_sasa
from .segmentation import (
    filter_by_length,
    find_helix_segments,
    geometric_helix_ss8,
    is_ss8_effectively_null,
    merge_by_gap,
    smooth_ss8,
)
from .windowing import stable_helix_seed, tile_windows


@dataclass
class ExtractedExample:
    example_id: int
    tensor_bytes: bytes
    helix_seqres_start: int
    helix_seqres_end: int
    helix_length: int
    n_helix_residues: int
    n_partner_residues: int
    n_partner_chains: int
    n_helix_contacts: int
    n_partner_interface_residues: int
    n_residues_total: int
    helix_sequence: str
    sasa_used: bool


@dataclass
class ExampleResult:
    pdb_id: str
    assembly_id: int
    processing_status: str
    drop_reason: str | None = None
    examples: list[ExtractedExample] = field(default_factory=list)
    n_helix_segments: int | None = None
    n_interacting_helices: int | None = None
    n_windows_before_filter: int | None = None
    n_examples_emitted: int | None = None
    warnings: list[str] = field(default_factory=list)


def _load_module2_tensor(module2_npz_bytes: bytes) -> dict:
    data = np.load(io.BytesIO(module2_npz_bytes))
    return {
        "n_chains": int(data["n_chains"]),
        "n_max_residues": int(data["n_max_residues"]),
        "residue_index": data["residue_index"],
        "residue_type": data["residue_type"],
        "ss_3": data["ss_3"],
        "ss_8": data["ss_8"],
        "coordinates": data["coordinates"].astype(np.float16),
        "atom_mask": data["atom_mask"],
        "protein_chain_names": data["protein_chain_names"],
    }


def _chain_real_positions(atom_mask: np.ndarray, chain_index: int) -> list[int]:
    real = np.where((atom_mask[chain_index] != -1).any(axis=-1))[0]
    return real.tolist()


def process_entry(
    module2_npz_bytes: bytes,
    pdb_id: str,
    assembly_id: int,
    m2_meta: dict,
    cfg: ExamplesConfig,
) -> ExampleResult:
    pdb_id = pdb_id.upper()
    try:
        m2 = _load_module2_tensor(module2_npz_bytes)
    except Exception:
        return ExampleResult(pdb_id, assembly_id, "dropped",
                             drop_reason="unparseable_module2_output")

    try:
        return _process(m2, pdb_id, assembly_id, m2_meta, cfg)
    except Exception:
        tb = traceback.format_exc()
        return ExampleResult(pdb_id, assembly_id, "error",
                             drop_reason="processing_error",
                             warnings=[tb])


def _process(m2: dict, pdb_id: str, assembly_id: int, m2_meta: dict,
             cfg: ExamplesConfig) -> ExampleResult:
    n_chains = m2["n_chains"]
    if n_chains == 0:
        return ExampleResult(pdb_id, assembly_id, "dropped",
                             drop_reason="no_helix_segments",
                             n_helix_segments=0, n_interacting_helices=0,
                             n_windows_before_filter=0, n_examples_emitted=0)

    chain_real: dict[int, list[int]] = {
        c: _chain_real_positions(m2["atom_mask"], c) for c in range(n_chains)
    }

    ss8_is_null = is_ss8_effectively_null(m2["ss_8"])
    warnings: list[str] = []
    if ss8_is_null:
        warnings.append("module2_ss_codes_null_using_geometric_fallback")
    all_segments: list[tuple[int, int, int]] = []
    for c in range(n_chains):
        real = chain_real[c]
        if not real:
            continue
        last = real[-1]
        if ss8_is_null:
            ca_coords = m2["coordinates"][c, : last + 1, 1, :].astype(np.float32)
            ca_present = (m2["atom_mask"][c, : last + 1, 1] == 1)
            ss8_chain = geometric_helix_ss8(ca_coords, ca_present)
        else:
            ss8_chain = m2["ss_8"][c, : last + 1]
        smoothed = smooth_ss8(ss8_chain) if cfg.dssp_smoothing else ss8_chain.copy()
        segments = find_helix_segments(smoothed, cfg.min_helix_segment_length)
        for s, e in segments:
            all_segments.append((c, s, e))

    if not all_segments:
        return ExampleResult(pdb_id, assembly_id, "dropped",
                             drop_reason="no_helix_segments",
                             n_helix_segments=0, n_interacting_helices=0,
                             n_windows_before_filter=0, n_examples_emitted=0,
                             warnings=warnings)

    index = build_spatial_index(m2["coordinates"], m2["atom_mask"])

    interacting_helices: list[tuple[int, int, int]] = []
    contact_by_segment: dict[tuple[int, int, int], np.ndarray] = {}
    for c, s, e in all_segments:
        is_contacting = mark_contacting_residues(
            index, m2["coordinates"], m2["atom_mask"],
            c, s, e, cfg.contact_distance_heavy_atom,
        )
        full = np.zeros(e - s + 1, dtype=bool)
        full[:] = is_contacting
        contact_by_segment[(c, s, e)] = full
        full_chain_len = max(s, e) + 1
        padded = np.zeros(full_chain_len + 1, dtype=bool)
        padded[s : e + 1] = full
        merged = merge_by_gap(padded, s, e, cfg.max_helix_gap_residues)
        for ms, me in merged:
            interacting_helices.append((c, ms, me))

    if not interacting_helices:
        return ExampleResult(pdb_id, assembly_id, "dropped",
                             drop_reason="no_interacting_helices",
                             n_helix_segments=len(all_segments),
                             n_interacting_helices=0,
                             n_windows_before_filter=0, n_examples_emitted=0,
                             warnings=warnings)

    interacting_helices.sort(key=lambda t: (t[0], t[1]))
    length_filtered = [h for h in interacting_helices
                       if (h[2] - h[1] + 1) >= cfg.window_length_min]

    examples: list[ExtractedExample] = []
    example_id = 0
    windows_before_filter = 0

    for helix_index, (c, hs, he) in enumerate(length_filtered):
        helix_positions_all = list(range(hs, he + 1))
        helix_len = len(helix_positions_all)
        seed = stable_helix_seed(cfg.random_seed, pdb_id, assembly_id, helix_index)
        window_spans = tile_windows(helix_len, seed, cfg.window_length_min, cfg.window_length_max)
        windows_before_filter += len(window_spans)

        is_contacting_full = np.zeros(helix_len, dtype=bool)
        for c_seg, s_seg, e_seg in all_segments:
            if c_seg != c:
                continue
            if s_seg > he or e_seg < hs:
                continue
            seg_contact = contact_by_segment[(c_seg, s_seg, e_seg)]
            overlap_start = max(hs, s_seg)
            overlap_end = min(he, e_seg)
            for pos in range(overlap_start, overlap_end + 1):
                is_contacting_full[pos - hs] = is_contacting_full[pos - hs] or bool(
                    seg_contact[pos - s_seg]
                )

        for w_start, w_end in window_spans:
            abs_start = hs + w_start
            abs_end = hs + w_end
            window_positions = list(range(abs_start, abs_end + 1))
            window_is_contacting = [bool(is_contacting_full[p - hs]) for p in window_positions]
            n_contacts = sum(window_is_contacting)
            if n_contacts < cfg.min_contacts_per_window:
                continue

            partner_chains = partner_chains_for_window(
                index, m2["coordinates"], m2["atom_mask"],
                c, window_positions, cfg.contact_distance_heavy_atom,
            )
            if not partner_chains:
                continue

            partner_distance: dict[int, set[int]] = {}
            for pc in partner_chains:
                partner_distance[pc] = distance_interface_partners(
                    index, m2["coordinates"], m2["atom_mask"],
                    c, window_positions, pc, cfg.contact_distance_heavy_atom,
                )

            sasa_used = False
            partner_interface: dict[int, set[int]] = {
                pc: set(partner_distance[pc]) for pc in partner_chains
            }
            if cfg.partner_use_sasa:
                delta_sasa, sasa_used = compute_partner_delta_sasa(
                    m2["coordinates"], m2["atom_mask"], m2["residue_type"],
                    c, window_positions, partner_chains, chain_real,
                )
                if sasa_used:
                    for (pc, res_pos), d in delta_sasa.items():
                        if pc in partner_interface and d >= cfg.partner_sasa_threshold:
                            partner_interface[pc].add(res_pos)

            partner_positions: dict[int, list[int]] = {}
            partner_is_interface: dict[int, set[int]] = {}
            for pc in partner_chains:
                expanded, interface_subset = expand_with_context(
                    partner_interface[pc], chain_real[pc], cfg.partner_sequence_context,
                )
                if not expanded:
                    continue
                partner_positions[pc] = sorted(expanded)
                partner_is_interface[pc] = interface_subset

            actual_partner_chains = [pc for pc in partner_chains if pc in partner_positions]
            if not actual_partner_chains:
                continue

            tensors = build_example_tensor(
                m2, c, window_positions, window_is_contacting,
                actual_partner_chains, partner_positions, partner_is_interface,
                m2["protein_chain_names"],
            )
            if not completeness_ok(tensors.atom_mask, cfg.min_backbone_atom_completeness):
                continue

            helix_seqres = [int(m2["residue_index"][c, p]) for p in window_positions]
            helix_rtypes = np.array([int(m2["residue_type"][c, p]) for p in window_positions])
            helix_seq = helix_sequence_from_types(helix_rtypes)

            n_helix_residues = len(window_positions)
            n_partner_residues = sum(len(partner_positions[pc]) for pc in actual_partner_chains)
            n_partner_interface = sum(len(partner_is_interface[pc]) for pc in actual_partner_chains)

            tensor_bytes = serialize_example_npz(
                tensors,
                pdb_id=pdb_id,
                assembly_id=assembly_id,
                example_id=example_id,
                helix_seqres_start=helix_seqres[0],
                helix_seqres_end=helix_seqres[-1],
                helix_sequence=helix_seq,
                n_helix_contacts=n_contacts,
                resolution=m2_meta.get("resolution"),
                r_free=m2_meta.get("r_free"),
                source_method=m2_meta.get("method"),
                sasa_used=sasa_used,
            )

            examples.append(ExtractedExample(
                example_id=example_id,
                tensor_bytes=tensor_bytes,
                helix_seqres_start=helix_seqres[0],
                helix_seqres_end=helix_seqres[-1],
                helix_length=helix_seqres[-1] - helix_seqres[0] + 1,
                n_helix_residues=n_helix_residues,
                n_partner_residues=n_partner_residues,
                n_partner_chains=len(actual_partner_chains),
                n_helix_contacts=n_contacts,
                n_partner_interface_residues=n_partner_interface,
                n_residues_total=n_helix_residues + n_partner_residues,
                helix_sequence=helix_seq,
                sasa_used=sasa_used,
            ))
            example_id += 1

    if not examples:
        return ExampleResult(
            pdb_id, assembly_id, "dropped",
            drop_reason="no_surviving_windows",
            n_helix_segments=len(all_segments),
            n_interacting_helices=len(interacting_helices),
            n_windows_before_filter=windows_before_filter,
            n_examples_emitted=0,
            warnings=warnings,
        )

    return ExampleResult(
        pdb_id, assembly_id, "ok",
        drop_reason=None,
        examples=examples,
        n_helix_segments=len(all_segments),
        n_interacting_helices=len(interacting_helices),
        n_windows_before_filter=windows_before_filter,
        n_examples_emitted=len(examples),
        warnings=warnings,
    )
