"""Stage-4 driver: read the final patches parquet, emit one .pml per patch.

Renders every row in `patches_final.parquet`. Upstream stages now annotate
patches (with ScanNet stats and PPI-hotspot counts) for ranking rather than
rejecting them, so this stage treats every row as something the user wants
to inspect; per-patch hotspot residues, when present, are recolored with
`cfg.hotspot_color` so the hotspot subset stands out from the 4-class
biochemical coloring of the rest of the patch.

Writes one `.pml` per patch into `<output_dir>/`, plus a single decompressed
`.pdb` per source structure under `<output_dir>/pdb/` so each script is
self-contained and references an absolute path."""
from __future__ import annotations

import gzip
import logging
from pathlib import Path

import gemmi
import pandas as pd

from .config import EpitopeVizConfig
from .pymol_writer import build_pml

logger = logging.getLogger(__name__)


def _stage_pdb(pdb_id: str, pdb_dir: Path, pdb_out_dir: Path) -> Path:
    """Decompress `<pdb_dir>/<2-char>/<pdb_id>.cif.gz` → `<pdb_out_dir>/<pdb_id>.pdb`."""
    pid = pdb_id.lower()
    cif_gz = pdb_dir / pid[1:3] / f"{pid}.cif.gz"
    if not cif_gz.exists():
        raise FileNotFoundError(str(cif_gz))
    out_path = pdb_out_dir / f"{pid}.pdb"
    if out_path.exists():
        return out_path
    with gzip.open(cif_gz, "rt") as f:
        text = f.read()
    structure = gemmi.read_structure_string(text, format=gemmi.CoorFormat.Mmcif)
    structure.setup_entities()
    structure.write_pdb(str(out_path))
    return out_path


def _build_aa_lookup(pdb_path: Path) -> dict[tuple[str, int, str], str]:
    """{(chain, seq, icode) → one-letter AA code} for every protein residue."""
    structure = gemmi.read_structure(str(pdb_path))
    out: dict[tuple[str, int, str], str] = {}
    for model in structure:
        for chain in model:
            for res in chain:
                info = gemmi.find_tabulated_residue(res.name)
                if info is None or not info.is_amino_acid():
                    continue
                code = info.one_letter_code.upper()
                key = (chain.name, res.seqid.num, res.seqid.icode.strip())
                out[key] = code
        break  # only first model
    return out


def run_epitope_viz(cfg: EpitopeVizConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_out_dir = out_dir / "pdb"
    pdb_out_dir.mkdir(exist_ok=True)
    error_log = out_dir / "errors.log"

    df = pd.read_parquet(cfg.patches_parquet)
    if df.empty:
        logger.info("patches parquet is empty; nothing to render")
        error_log.write_text("")
        return out_dir
    logger.info("Loaded %d patches across %d PDBs",
                len(df), df["pdb_id"].nunique())

    n_written = 0
    with error_log.open("w") as ef:
        for pdb_id, pdb_rows in df.groupby("pdb_id", sort=False):
            try:
                pdb_path = _stage_pdb(pdb_id, Path(cfg.pdb_dir), pdb_out_dir)
                aa_lookup = _build_aa_lookup(pdb_path)
            except Exception as e:
                logger.exception("%s: failed to stage PDB", pdb_id)
                ef.write(f"{pdb_id}\t{type(e).__name__}\t{e}\n")
                ef.flush()
                continue

            for _, row in pdb_rows.iterrows():
                patch_id = str(row["patch_id"])
                try:
                    hotspots = (list(row["hotspot_residue_ids"])
                                if "hotspot_residue_ids" in pdb_rows.columns
                                else None)
                    pml = build_pml(
                        pdb_id=pdb_id,
                        patch_id=patch_id,
                        patch_residue_ids=list(row["residue_ids"]),
                        pdb_path=pdb_path,
                        aa_lookup=aa_lookup,
                        cfg=cfg,
                        hotspot_residue_ids=hotspots,
                    )
                    (out_dir / f"{pdb_id}_{patch_id}.pml").write_text(pml)
                    n_written += 1
                except Exception as e:
                    logger.exception("%s/%s: failed to build pml", pdb_id, patch_id)
                    ef.write(f"{pdb_id}\t{patch_id}\t{type(e).__name__}\t{e}\n")
                    ef.flush()

    logger.info("Wrote %d .pml files to %s", n_written, out_dir)
    return out_dir
