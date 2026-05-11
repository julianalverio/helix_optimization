from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class TensorsConfig:
    module1_manifest_path: str
    local_mmcif_base_path: str
    output_dir: str

    solvent_residues: dict[str, list[str]]
    modified_residues_convert: dict[str, dict]
    modified_residues_drop_entry: list[str]
    d_amino_acid_codes: list[str]
    allowed_cofactors: dict[str, list[str]] = field(default_factory=dict)

    min_observed_residues_per_chain: int = 20
    max_unk_fraction_per_chain: float = 0.5
    dssp_executable: str = "mkdssp"

    modal_workers: int = 100
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
    "solvent_residues",
    "modified_residues_convert",
    "modified_residues_drop_entry",
    "d_amino_acid_codes",
    "allowed_cofactors",
    "min_observed_residues_per_chain",
    "max_unk_fraction_per_chain",
    "dssp_executable",
}


def load_tensors_config(path: Path | str) -> TensorsConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(TensorsConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown tensors config keys: {sorted(unknown)}")
    required = {"module1_manifest_path", "local_mmcif_base_path", "output_dir",
                "solvent_residues", "modified_residues_convert",
                "modified_residues_drop_entry", "d_amino_acid_codes"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing required tensors config keys: {sorted(missing)}")
    return TensorsConfig(**raw)


def tensors_config_hash(cfg: TensorsConfig) -> str:
    payload = {}
    for f in fields(cfg):
        if f.name not in _HASH_FIELDS:
            continue
        payload[f.name] = getattr(cfg, f.name)
    encoded = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def solvent_set(cfg: TensorsConfig) -> frozenset[str]:
    out: set[str] = set()
    for group in cfg.solvent_residues.values():
        out.update(group)
    return frozenset(out)


def cofactor_set(cfg: TensorsConfig) -> frozenset[str]:
    out: set[str] = set()
    for group in cfg.allowed_cofactors.values():
        out.update(group)
    return frozenset(out)
