from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExamplesConfig:
    tensors_output_dir: str
    tensors_manifest_path: str
    output_dir: str

    min_helix_segment_length: int = 6
    dssp_smoothing: bool = True
    contact_distance_heavy_atom: float = 5.0
    max_helix_gap_residues: int = 7
    window_length_min: int = 8
    window_length_max: int = 15
    min_contacts_per_window: int = 2
    partner_use_sasa: bool = True
    partner_sasa_threshold: float = 1.0
    partner_sequence_context: int = 2
    min_backbone_atom_completeness: float = 0.8
    random_seed: int = 42

    modal_workers: int = 200
    modal_cpu_per_worker: int = 1
    modal_memory_mb: int = 2048
    modal_batch_size: int = 10
    modal_timeout_seconds: int = 600
    modal_max_retries: int = 3
    modal_retry_backoff_seconds: int = 30

    test_mode: bool = False
    test_n_entries: int = 100
    test_modal_workers: int = 5
    test_modal_batch_size: int = 5
    test_output_subdir: str = "test_run"


_HASH_FIELDS = {
    "min_helix_segment_length",
    "dssp_smoothing",
    "contact_distance_heavy_atom",
    "max_helix_gap_residues",
    "window_length_min",
    "window_length_max",
    "min_contacts_per_window",
    "partner_use_sasa",
    "partner_sasa_threshold",
    "partner_sequence_context",
    "min_backbone_atom_completeness",
    "random_seed",
}


def load_examples_config(path: Path | str) -> ExamplesConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(ExamplesConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown examples config keys: {sorted(unknown)}")
    required = {"tensors_output_dir", "tensors_manifest_path", "output_dir"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing required examples config keys: {sorted(missing)}")
    return ExamplesConfig(**raw)


def examples_config_hash(cfg: ExamplesConfig) -> str:
    payload = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name in _HASH_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
