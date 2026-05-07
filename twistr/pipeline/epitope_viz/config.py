from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class EpitopeVizConfig:
    patches_parquet: str
    pdb_dir: str
    output_dir: str

    chain_color: str = "gray70"
    background_color: str = "black"
    sticks_colors: dict[str, str] = field(
        default_factory=lambda: {
            "hydrophobic": "yellow",
            "negative": "blue",
            "positive": "red",
            "polar": "green",
        }
    )
    # Hotspot residues (PPI-hotspotID hits) override the 4-class color above
    # so they pop visually. Default = PyMOL's "hotpink".
    hotspot_color: str = "hotpink"
    zoom_padding_a: float = 5.0


def load_epitope_viz_config(path: Path | str) -> EpitopeVizConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(EpitopeVizConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown epitope-viz config keys: {sorted(unknown)}")
    return EpitopeVizConfig(**raw)
