import io
import shutil
from pathlib import Path

import numpy as np
import pytest

from twistr.pipeline.tensors.config import TensorsConfig
from twistr.pipeline.tensors.constants import RESIDUE_TYPE_INDEX
from twistr.pipeline.tensors.pipeline import process_entry
from twistr.pipeline.tensors.tensors import build_atom14, serialize_npz

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mmCIF"
HAS_DSSP = shutil.which("mkdssp") is not None


def _default_config(tmp_path: Path) -> TensorsConfig:
    return TensorsConfig(
        module1_manifest_path=str(tmp_path / "m1.parquet"),
        local_mmcif_base_path=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        solvent_residues={
            "waters": ["HOH", "DOD", "H2O", "WAT"],
            "cryoprotectants": ["GOL", "EDO", "PEG"],
            "buffers": ["SO4", "PO4"],
            "reductants": ["BME", "DTT"],
            "other_artifacts": [],
        },
        modified_residues_convert={
            "MSE": {"parent": "MET", "atom_renames": {"SE": "SD"}},
            "SEC": {"parent": "CYS", "atom_renames": {"SE": "SG"}},
        },
        modified_residues_drop_entry=["SEP", "TPO", "PTR", "HYP"],
        d_amino_acid_codes=["DAL", "DLE"],
    )


def _load_fixture_bytes(name: str) -> bytes:
    return (FIXTURE_DIR / f"{name}.cif.gz").read_bytes()


def test_build_atom14_empty_chains(tmp_path):
    tensors = build_atom14([], ss_map={})
    assert tensors["n_chains"] == 0
    assert tensors["n_max_residues"] == 0
    assert tensors["coordinates"].shape == (0, 0, 14, 3)
    assert tensors["atom_mask"].shape == (0, 0, 14)
    assert tensors["coordinates"].dtype == np.float16
    assert tensors["atom_mask"].dtype == np.int8


def test_serialize_npz_roundtrip(tmp_path):
    tensors = build_atom14([], ss_map={})
    blob = serialize_npz(tensors)
    data = np.load(io.BytesIO(blob))
    assert int(data["n_chains"]) == 0
    assert set(data.files) == {
        "n_chains", "n_max_residues", "residue_index", "residue_type",
        "ss_3", "ss_8", "coordinates", "atom_mask", "protein_chain_names",
        "cofactor_coords", "cofactor_atom_names", "cofactor_elements",
        "cofactor_residue_names", "cofactor_residue_indices", "cofactor_chain_names",
    }
    assert data["cofactor_coords"].shape == (0, 3)
    assert data["cofactor_coords"].dtype == np.float16


def test_process_entry_1brs(tmp_path):
    cfg = _default_config(tmp_path)
    result = process_entry(
        _load_fixture_bytes("1BRS"),
        pdb_id="1BRS",
        assembly_id=1,
        m1_meta={
            "primary_assembly_id": "1",
            "large_assembly": False,
            "unique_interface_plan": None,
        },
        cfg=cfg,
    )
    assert result.pdb_id == "1BRS"
    assert result.assembly_id == 1
    if HAS_DSSP:
        assert result.processing_status == "ok", (result.drop_reason, result.warnings)
        assert result.tensor_bytes is not None
        data = np.load(io.BytesIO(result.tensor_bytes))
        coords = data["coordinates"]
        atom_mask = data["atom_mask"]
        assert coords.dtype == np.float16
        assert atom_mask.dtype == np.int8
        assert coords.shape[-2:] == (14, 3)
        assert set(np.unique(atom_mask).tolist()).issubset({-1, 0, 1})
    else:
        assert result.drop_reason == "dssp_failed", (result.processing_status, result.drop_reason)


@pytest.mark.skipif(not HAS_DSSP, reason="mkdssp v4 not installed locally")
def test_process_entry_1a3n_deterministic(tmp_path):
    cfg = _default_config(tmp_path)
    m1_meta = {
        "primary_assembly_id": "1",
        "large_assembly": False,
        "unique_interface_plan": None,
    }
    a = process_entry(_load_fixture_bytes("1A3N"), "1A3N", 1, m1_meta, cfg)
    b = process_entry(_load_fixture_bytes("1A3N"), "1A3N", 1, m1_meta, cfg)
    assert a.processing_status == "ok"
    assert b.processing_status == "ok"
    assert a.tensor_bytes == b.tensor_bytes


def test_process_entry_unparseable(tmp_path):
    cfg = _default_config(tmp_path)
    import gzip
    bad = gzip.compress(b"not valid mmcif at all")
    result = process_entry(bad, "XXXX", 1, {
        "primary_assembly_id": "1", "large_assembly": False, "unique_interface_plan": None,
    }, cfg)
    assert result.processing_status == "dropped"
    assert result.drop_reason == "unparseable_mmcif"


def test_build_atom14_ss_lookup_matches_written_label_keys(tmp_path):
    """Regression test for the bug where SS codes ended up all-null because
    build_atom14 keyed the lookup by (chain.name, seqid.num) while DSSP returns
    keys based on what gemmi WROTE to mmCIF — namely (label_asym_id, label_seq_id),
    which is (res.subchain, res.label_seq).

    This test fakes a DSSP ss_map with the SAME keys gemmi would write to mmCIF
    and asserts build_atom14 actually places the SS codes in the tensor."""
    import gzip
    import gemmi
    from twistr.pipeline.tensors.constants import RESIDUE_TYPE_INDEX

    text = gzip.decompress(_load_fixture_bytes("1BRS")).decode("utf-8", errors="replace")
    doc = gemmi.cif.read_string(text)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    structure.setup_entities()
    structure.transform_to_assembly("1", gemmi.HowToNameCopiedChain.Short)

    # Pick the first protein chain and two of its canonical residues. Build the
    # ss_map using exactly what gemmi will write to label_asym_id / label_seq_id.
    chains = []
    for model in structure:
        for chain in model:
            canonical = [r for r in chain if r.name in RESIDUE_TYPE_INDEX]
            if len(canonical) >= 2:
                chains.append(chain)
                if len(chains) >= 1:
                    break
        if chains:
            break
    assert chains, "fixture should have at least one canonical-residue chain"
    chain = chains[0]
    canonical = [r for r in chain if r.name in RESIDUE_TYPE_INDEX]

    expected_keys = []
    ss_map = {}
    for r in canonical[:2]:
        label_seq = r.label_seq if r.label_seq is not None else r.seqid.num
        key = (r.subchain, label_seq)
        expected_keys.append(key)
        ss_map[key] = (0, 0)  # ss3=H, ss8=H

    # Sanity: the keys must NOT collide with what the buggy code would have used.
    bad_keys = [(chain.name, r.seqid.num) for r in canonical[:2]]
    assert set(expected_keys).isdisjoint(set(bad_keys)), (
        "fixture chosen poorly: subchain happens to equal chain.name and label_seq "
        "happens to equal seqid.num, so this test wouldn't distinguish the fix"
    )

    tensors = build_atom14([chain], ss_map=ss_map)
    ss_3 = tensors["ss_3"]
    ss_8 = tensors["ss_8"]

    # The first two canonical residues should pick up our fake SS codes (H/H).
    assert int(ss_3[0, 0]) == 0, f"first residue ss_3 should be H (0), got {int(ss_3[0, 0])}"
    assert int(ss_8[0, 0]) == 0, f"first residue ss_8 should be H (0), got {int(ss_8[0, 0])}"
    assert int(ss_3[0, 1]) == 0
    assert int(ss_8[0, 1]) == 0


@pytest.mark.skipif(not HAS_DSSP, reason="mkdssp v4 not installed locally")
def test_process_entry_1brs_has_non_null_ss_codes(tmp_path):
    """End-to-end regression: when DSSP is available, an ok-status entry must
    have at least one non-null SS code. This catches lookup-key mismatches that
    silently produce all-null SS arrays."""
    cfg = _default_config(tmp_path)
    result = process_entry(
        _load_fixture_bytes("1BRS"), "1BRS", 1,
        {"primary_assembly_id": "1", "large_assembly": False, "unique_interface_plan": None},
        cfg,
    )
    assert result.processing_status == "ok", (result.drop_reason, result.warnings)
    data = np.load(io.BytesIO(result.tensor_bytes))
    am = data["atom_mask"]
    real = (am != -1).any(axis=-1)
    ss8_real = data["ss_8"][real]
    assert int(np.sum(ss8_real != 8)) > 0, "all SS codes are null — lookup-key regression"
