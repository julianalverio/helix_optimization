"""Unit tests for the epitope-viz PyMOL writer + driver."""
from __future__ import annotations

import gzip
from pathlib import Path

import gemmi
import pandas as pd

from twistr.pipeline.epitope_viz.aa_classes import classify
from twistr.pipeline.epitope_viz.config import EpitopeVizConfig
from twistr.pipeline.epitope_viz.driver import run_epitope_viz
from twistr.pipeline.epitope_viz.pymol_writer import build_pml


def _write_minimal_cif_gz(pdb_dir: Path, pdb_id: str) -> None:
    """Write a 2-residue (LEU at A/1, LYS at A/2) mmCIF.gz under
    `<pdb_dir>/<2-char>/<pdb_id>.cif.gz` — the layout the driver expects."""
    pid = pdb_id.lower()
    structure = gemmi.Structure()
    structure.name = pid.upper()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for seq, resname, pos in [(1, "LEU", (0.0, 0.0, 0.0)),
                              (2, "LYS", (3.8, 0.0, 0.0))]:
        res = gemmi.Residue()
        res.name = resname
        res.seqid = gemmi.SeqId(seq, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*pos)
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()

    target_dir = pdb_dir / pid[1:3]
    target_dir.mkdir(parents=True, exist_ok=True)
    cif_text = structure.make_mmcif_document().as_string()
    with gzip.open(target_dir / f"{pid}.cif.gz", "wt") as f:
        f.write(cif_text)


# ---------------------- aa_classes.classify ----------------------

def test_classify_canonical_aas():
    for aa in "AVLIMFWYPCG":
        assert classify(aa) == "hydrophobic", aa
    for aa in "DE":
        assert classify(aa) == "negative", aa
    for aa in "KRH":
        assert classify(aa) == "positive", aa
    for aa in "STNQ":
        assert classify(aa) == "polar", aa


def test_classify_non_canonical_returns_none():
    assert classify("X") is None
    assert classify("U") is None  # selenocysteine
    assert classify("?") is None
    assert classify("") is None


def test_classify_case_insensitive():
    assert classify("a") == "hydrophobic"
    assert classify("e") == "negative"


# ---------------------- pymol_writer.build_pml ----------------------

def _cfg(**overrides) -> EpitopeVizConfig:
    base = dict(
        patches_parquet="ignored",
        pdb_dir="ignored",
        output_dir="ignored",
    )
    base.update(overrides)
    return EpitopeVizConfig(**base)


def test_build_pml_groups_by_chain_and_category(tmp_path: Path):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")
    aa_lookup = {
        ("A", 12, ""): "L",  # hydrophobic
        ("A", 15, ""): "F",  # hydrophobic
        ("A", 22, ""): "D",  # negative
        ("A", 30, ""): "K",  # positive
        ("B", 5, ""): "S",   # polar
        ("B", 8, ""): "N",   # polar
    }
    pml = build_pml(
        pdb_id="1abc", patch_id="patch_0",
        patch_residue_ids=["A/12", "A/15", "A/22", "A/30", "B/5", "B/8"],
        pdb_path=pdb_path, aa_lookup=aa_lookup, cfg=_cfg(),
    )

    assert pml.startswith("reinitialize\n")
    assert "bg_color black" in pml
    assert f"load {pdb_path.absolute()}, 1abc" in pml
    assert "show cartoon" in pml
    assert "color gray70, 1abc" in pml

    assert "select patch_A_hydrophobic, 1abc and chain A and resi 12+15" in pml
    assert "color yellow, patch_A_hydrophobic" in pml
    assert "select patch_A_negative, 1abc and chain A and resi 22" in pml
    assert "color blue, patch_A_negative" in pml
    assert "select patch_A_positive, 1abc and chain A and resi 30" in pml
    assert "color red, patch_A_positive" in pml
    assert "select patch_B_polar, 1abc and chain B and resi 5+8" in pml
    assert "color green, patch_B_polar" in pml

    assert "orient patch_all" in pml
    assert "zoom patch_all, 5.0" in pml


def test_build_pml_skips_unknown_and_noncanonical(tmp_path: Path, caplog):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")
    aa_lookup = {
        ("A", 1, ""): "A",     # hydrophobic
        ("A", 2, ""): "X",     # non-canonical
        # A/3 is intentionally missing from the lookup
    }
    with caplog.at_level("WARNING"):
        pml = build_pml(
            pdb_id="1abc", patch_id="p",
            patch_residue_ids=["A/1", "A/2", "A/3"],
            pdb_path=pdb_path, aa_lookup=aa_lookup, cfg=_cfg(),
        )

    assert "patch_A_hydrophobic" in pml
    assert "non-canonical" in caplog.text
    assert "not found in PDB" in caplog.text


def test_build_pml_empty_patch_emits_cartoon_only(tmp_path: Path):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")
    pml = build_pml(
        pdb_id="1abc", patch_id="p",
        patch_residue_ids=[], pdb_path=pdb_path,
        aa_lookup={}, cfg=_cfg(),
    )
    assert "show cartoon" in pml
    assert "show sticks" not in pml
    assert "orient" not in pml
    assert "zoom" not in pml


def test_build_pml_recolors_hotspots_with_hotspot_color(tmp_path: Path):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")
    aa_lookup = {
        ("A", 12, ""): "L",  # hydrophobic
        ("A", 22, ""): "D",  # negative — and a hotspot
        ("A", 30, ""): "K",  # positive — and a hotspot
    }
    pml = build_pml(
        pdb_id="1abc", patch_id="p",
        patch_residue_ids=["A/12", "A/22", "A/30"],
        pdb_path=pdb_path, aa_lookup=aa_lookup, cfg=_cfg(),
        hotspot_residue_ids=["A/22", "A/30"],
    )
    # 4-class buckets are still rendered.
    assert "color yellow, patch_A_hydrophobic" in pml
    assert "color blue, patch_A_negative" in pml
    assert "color red, patch_A_positive" in pml
    # Hotspot recolor block.
    assert "select patch_A_hotspot, 1abc and chain A and resi 22+30" in pml
    assert "color hotpink, patch_A_hotspot" in pml
    # Hotspot recolor lines come AFTER the per-class blocks so hotpink wins.
    assert pml.index("color hotpink") > pml.index("color blue")
    assert pml.index("color hotpink") > pml.index("color red")
    # patch_all selection includes the hotspot selection too.
    assert "patch_A_hotspot" in pml.split("select patch_all,")[1].split("\n")[0]


def test_build_pml_handles_icode(tmp_path: Path):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")
    aa_lookup = {("A", 27, "A"): "K"}
    pml = build_pml(
        pdb_id="1abc", patch_id="p",
        patch_residue_ids=["A/27A"],
        pdb_path=pdb_path, aa_lookup=aa_lookup, cfg=_cfg(),
    )
    assert "resi 27A" in pml


# ---------------------- driver.run_epitope_viz ----------------------

def test_driver_writes_one_pml_per_row_and_logs_missing_pdb(tmp_path: Path):
    parquet_path = tmp_path / "patches_final.parquet"
    out_dir = tmp_path / "out"
    pdb_dir = tmp_path / "pdb"
    _write_minimal_cif_gz(pdb_dir, "1abc")

    pd.DataFrame([
        {"pdb_id": "1abc", "patch_id": "p0", "residue_ids": ["A/1", "A/2"]},
        {"pdb_id": "1abc", "patch_id": "p1", "residue_ids": ["A/1"]},
        {"pdb_id": "9zzz", "patch_id": "p0", "residue_ids": ["A/1"]},
    ]).to_parquet(parquet_path, index=False)

    cfg = EpitopeVizConfig(
        patches_parquet=str(parquet_path),
        pdb_dir=str(pdb_dir),
        output_dir=str(out_dir),
    )
    assert run_epitope_viz(cfg) == out_dir

    assert (out_dir / "1abc_p0.pml").exists()
    assert (out_dir / "1abc_p1.pml").exists()
    assert not (out_dir / "9zzz_p0.pml").exists()

    errors = (out_dir / "errors.log").read_text()
    assert "9zzz" in errors
    assert "FileNotFoundError" in errors

    p0_text = (out_dir / "1abc_p0.pml").read_text()
    assert "resi 1" in p0_text
    assert "resi 2" in p0_text
    assert "patch_A_hydrophobic" in p0_text  # LEU
    assert "patch_A_positive" in p0_text     # LYS


def test_driver_handles_empty_parquet(tmp_path: Path):
    parquet_path = tmp_path / "patches_final.parquet"
    pd.DataFrame({"pdb_id": pd.Series(dtype="object"),
                  "patch_id": pd.Series(dtype="object"),
                  "residue_ids": pd.Series(dtype="object")}).to_parquet(parquet_path)

    cfg = EpitopeVizConfig(
        patches_parquet=str(parquet_path),
        pdb_dir=str(tmp_path / "pdb"),
        output_dir=str(tmp_path / "out"),
    )
    out = run_epitope_viz(cfg)
    assert out.exists()
    assert (out / "errors.log").exists()
    assert not list(out.glob("*.pml"))
