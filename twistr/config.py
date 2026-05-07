from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

PIPELINE_VERSION = "0.1.0"

DEV_IDS = [
    "1HH6",
    "1BRS",
    "2REB",
    "1TIM",
    "6VXX",
    "1AON",
    "3J3Q",
    "1A3N",
    "1G03",
    "1RWH",
]


@dataclass(frozen=True)
class Config:
    methods_allowed: tuple[str, ...] = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")
    resolution_max_xray: float = 3.5
    resolution_max_em: float = 3.5
    r_free_max_xray: float = 0.30
    r_free_missing_action: str = "keep_and_tag"
    status_allowed: tuple[str, ...] = ("REL",)
    min_protein_chain_length: int = 20
    min_instantiated_polymer_chains: int = 2
    require_protein_chain: bool = True
    min_observed_fraction: float = 0.5
    large_assembly_chain_threshold: int = 20
    hard_cap_total_residues: int | None = None
    deposition_date_min: date | None = None
    deposition_date_max: date | None = None
    release_date_min: date | None = None
    release_date_max: date | None = None

    rsync_primary: str = "rsync.rcsb.org::ftp_data/structures/divided/mmCIF/"
    rsync_primary_port: int = 33444
    rsync_fallback: str = "rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/"
    rsync_fallback_port: int = 873

    data_root: str = "data"


_FILTER_FIELDS = {
    "methods_allowed",
    "resolution_max_xray",
    "resolution_max_em",
    "r_free_max_xray",
    "r_free_missing_action",
    "status_allowed",
    "min_protein_chain_length",
    "min_instantiated_polymer_chains",
    "require_protein_chain",
    "min_observed_fraction",
    "large_assembly_chain_threshold",
    "hard_cap_total_residues",
    "deposition_date_min",
    "deposition_date_max",
    "release_date_min",
    "release_date_max",
}


def load_config(path: Path | str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("methods_allowed", "status_allowed"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    for key in ("deposition_date_min", "deposition_date_max", "release_date_min", "release_date_max"):
        if raw.get(key):
            raw[key] = date.fromisoformat(raw[key])
    return Config(**raw)


def config_hash(cfg: Config) -> str:
    payload = {}
    for f in fields(cfg):
        if f.name not in _FILTER_FIELDS:
            continue
        value = getattr(cfg, f.name)
        if isinstance(value, date):
            value = value.isoformat()
        elif isinstance(value, tuple):
            value = list(value)
        payload[f.name] = value
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def snapshot_now() -> datetime:
    return datetime.now(timezone.utc)


def config_as_dict(cfg: Config) -> dict:
    out = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, date):
            out[k] = v.isoformat()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out
