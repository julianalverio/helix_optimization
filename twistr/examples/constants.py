from __future__ import annotations

from ..tensors.constants import RESIDUE_TYPE_NAMES

RESIDUE_ONE_LETTER: tuple[str, ...] = (
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
)

assert len(RESIDUE_ONE_LETTER) == len(RESIDUE_TYPE_NAMES)

BACKBONE_SLOTS: tuple[int, ...] = (0, 1, 2, 3)

SS8_H = 0
SS8_G = 1
SS8_I = 2
SS8_T = 5
SS8_S = 6
SS8_NULL_SENTINEL = 8
SS8_SMOOTHABLE: frozenset[int] = frozenset({SS8_G, SS8_I, SS8_T, SS8_S})

# CA-based geometric helix detection. Alpha helix CA(i)-CA(i+3) ≈ 5.0-5.8 Å,
# CA(i)-CA(i+4) ≈ 6.0-7.5 Å. Thresholds are deliberately generous so we tolerate
# minor distortion.
HELIX_CA_I_I3_MIN = 4.7
HELIX_CA_I_I3_MAX = 6.3
HELIX_CA_I_I4_MIN = 5.4
HELIX_CA_I_I4_MAX = 7.8

CANONICAL_DROP_REASONS = {
    "unparseable_module2_output",
    "no_helix_segments",
    "no_interacting_helices",
    "no_surviving_windows",
    "processing_error",
    "batch_retry_exhausted",
}

EXAMPLE_NPZ_KEYS = {
    "coordinates", "atom_mask", "residue_type", "ss_3", "ss_8",
    "chain_slot", "seqres_position", "is_helix", "is_interface_residue",
    "chain_label", "chain_module2_index", "chain_role",
    "pdb_id", "assembly_id", "example_id",
    "helix_seqres_start", "helix_seqres_end", "helix_sequence",
    "n_helix_contacts", "resolution", "r_free", "source_method", "sasa_used",
}

EXAMPLE_MANIFEST_COLUMNS = [
    "example_id_full", "pdb_id", "assembly_id", "example_id",
    "helix_seqres_start", "helix_seqres_end", "helix_length",
    "n_helix_residues", "n_partner_residues", "n_partner_chains",
    "n_helix_contacts", "n_partner_interface_residues", "n_residues_total",
    "helix_sequence", "resolution", "r_free", "source_method", "sasa_used",
    "path_example", "pipeline_version", "config_hash", "processing_date",
]

ENTRY_STATUS_COLUMNS = [
    "pdb_id", "assembly_id", "processing_status", "drop_reason",
    "n_helix_segments", "n_interacting_helices", "n_windows_before_filter",
    "n_examples_emitted", "processing_date",
]
