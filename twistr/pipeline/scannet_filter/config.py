from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanNetConfig:
    masif_parquet: str
    pdb_dir: str
    output_path: str
    work_dir: str

    scannet_image: str = "jertubiana/scannet"
    scannet_platform: str = "linux/amd64"
    scannet_mode: str = "epitope"          # interface | epitope | idp
    scannet_assembly: bool = True

    residue_positive_threshold: float = 0.5
    patch_min_mean_score: float = 0.4
    patch_min_positive_fraction: float = 0.25
    patch_min_max_score: float = 0.65
    patch_min_residues: int = 5
