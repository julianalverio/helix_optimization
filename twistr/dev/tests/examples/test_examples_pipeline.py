import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from twistr.examples.config import ExamplesConfig
from twistr.examples.constants import CANONICAL_DROP_REASONS, EXAMPLE_NPZ_KEYS
from twistr.examples.pipeline import process_entry

M2_TENSOR = Path(__file__).resolve().parents[3] / "runtime" / "data" / "tensors" / "tensors" / "br" / "1brs_1.npz"

freesasa = pytest.importorskip("freesasa")

pytestmark = pytest.mark.skipif(
    not M2_TENSOR.exists(),
    reason=f"Module 2 tensor {M2_TENSOR} not on disk",
)


def _cfg(tmp_path: Path) -> ExamplesConfig:
    return ExamplesConfig(
        tensors_output_dir=str(tmp_path),
        tensors_manifest_path=str(tmp_path / "m2.parquet"),
        output_dir=str(tmp_path / "out"),
    )


def _load_m2_bytes() -> bytes:
    return M2_TENSOR.read_bytes()


def test_process_entry_1brs_produces_examples_or_canonical_drop(tmp_path):
    cfg = _cfg(tmp_path)
    result = process_entry(
        _load_m2_bytes(),
        pdb_id="1BRS",
        assembly_id=1,
        m2_meta={"resolution": 2.0, "r_free": 0.25, "method": "X-RAY DIFFRACTION"},
        cfg=cfg,
    )
    assert result.pdb_id == "1BRS"
    assert result.assembly_id == 1
    if result.processing_status == "ok":
        assert len(result.examples) >= 1
    else:
        assert result.drop_reason in CANONICAL_DROP_REASONS


def test_process_entry_1brs_invariants(tmp_path):
    cfg = _cfg(tmp_path)
    result = process_entry(
        _load_m2_bytes(), "1BRS", 1,
        {"resolution": 2.0, "r_free": 0.25, "method": "X-RAY DIFFRACTION"}, cfg,
    )
    if result.processing_status != "ok":
        pytest.skip(f"entry dropped with {result.drop_reason}")

    for ex in result.examples:
        data = np.load(io.BytesIO(ex.tensor_bytes))
        assert EXAMPLE_NPZ_KEYS.issubset(set(data.files)), (
            "missing keys: "
            f"{EXAMPLE_NPZ_KEYS - set(data.files)}"
        )
        coords = data["coordinates"]
        amask = data["atom_mask"]
        is_helix = data["is_helix"].astype(bool)
        chain_slot = data["chain_slot"]

        assert coords.dtype == np.float16
        assert coords.ndim == 3 and coords.shape[-2:] == (14, 3)
        assert amask.dtype == np.int8
        assert set(np.unique(amask).tolist()).issubset({-1, 0, 1})

        n = coords.shape[0]
        assert is_helix.shape == (n,)
        assert is_helix[0]
        n_helix = int(is_helix.sum())
        assert is_helix[:n_helix].all()
        assert not is_helix[n_helix:].any()

        assert int(chain_slot[0]) == 0
        assert np.all(np.diff(chain_slot) >= 0)

        bb = amask[:, [0, 1, 2, 3]]
        completeness = float(np.sum(bb == 1)) / (n * 4)
        assert completeness >= cfg.min_backbone_atom_completeness


def test_process_entry_1brs_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    meta = {"resolution": 2.0, "r_free": 0.25, "method": "X-RAY DIFFRACTION"}
    a = process_entry(_load_m2_bytes(), "1BRS", 1, meta, cfg)
    b = process_entry(_load_m2_bytes(), "1BRS", 1, meta, cfg)
    assert a.processing_status == b.processing_status
    assert len(a.examples) == len(b.examples)
    for ex_a, ex_b in zip(a.examples, b.examples):
        ha = hashlib.sha256(ex_a.tensor_bytes).hexdigest()
        hb = hashlib.sha256(ex_b.tensor_bytes).hexdigest()
        assert ha == hb


def test_process_entry_unparseable(tmp_path):
    cfg = _cfg(tmp_path)
    result = process_entry(b"not a valid npz", "XXXX", 1, {}, cfg)
    assert result.processing_status == "dropped"
    assert result.drop_reason == "unparseable_module2_output"
