from __future__ import annotations

import re
import subprocess

import modal

from .pipeline import process_entry

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("dssp", channels=["bioconda", "conda-forge"])
    .pip_install("gemmi", "numpy", "pandas", "pyarrow")
    .add_local_python_source("twistr")
)

app = modal.App("twistr-tensors", image=image)

_dssp_verified = False


def _ensure_dssp() -> None:
    global _dssp_verified
    if _dssp_verified:
        return
    result = subprocess.run(
        ["mkdssp", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = (result.stdout + result.stderr).strip()
    m = re.search(r"\b(\d+)\.\d+", out)
    if not m or int(m.group(1)) < 4:
        raise RuntimeError(f"mkdssp v4.x required; version string: {out!r}")
    _dssp_verified = True


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=600,
    max_containers=100,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30.0),
)
def process_batch(batch: list[dict]) -> list[dict]:
    _ensure_dssp()
    from .config import TensorsConfig

    results: list[dict] = []
    for item in batch:
        cfg_raw = item["cfg"]
        cfg = TensorsConfig(**cfg_raw)
        outcome = process_entry(
            item["mmcif_bytes"],
            item["pdb_id"],
            item["assembly_id"],
            item["m1_meta"],
            cfg,
        )
        results.append({
            "pdb_id": outcome.pdb_id,
            "assembly_id": outcome.assembly_id,
            "processing_status": outcome.processing_status,
            "drop_reason": outcome.drop_reason,
            "n_chains_processed": outcome.n_chains_processed,
            "n_substantive_chains": outcome.n_substantive_chains,
            "tensor_bytes": outcome.tensor_bytes,
            "warnings": outcome.warnings,
        })
    return results
