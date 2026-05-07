from __future__ import annotations

import modal

from .pipeline import process_entry

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "freesasa", "gemmi", "scipy")
    .add_local_python_source("twistr")
)

app = modal.App("twistr-examples", image=image)

_deps_verified = False


def _ensure_deps() -> None:
    global _deps_verified
    if _deps_verified:
        return
    import freesasa  # noqa: F401
    import gemmi  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    _deps_verified = True


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=600,
    max_containers=100,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30.0),
)
def process_batch(batch: list[dict]) -> list[dict]:
    _ensure_deps()
    from .config import ExamplesConfig

    results: list[dict] = []
    for item in batch:
        cfg = ExamplesConfig(**item["cfg"])
        outcome = process_entry(
            item["module2_npz_bytes"],
            item["pdb_id"],
            item["assembly_id"],
            item["m2_meta"],
            cfg,
        )
        results.append({
            "pdb_id": outcome.pdb_id,
            "assembly_id": outcome.assembly_id,
            "processing_status": outcome.processing_status,
            "drop_reason": outcome.drop_reason,
            "n_helix_segments": outcome.n_helix_segments,
            "n_interacting_helices": outcome.n_interacting_helices,
            "n_windows_before_filter": outcome.n_windows_before_filter,
            "n_examples_emitted": outcome.n_examples_emitted,
            "examples": [
                {
                    "example_id": ex.example_id,
                    "tensor_bytes": ex.tensor_bytes,
                    "helix_seqres_start": ex.helix_seqres_start,
                    "helix_seqres_end": ex.helix_seqres_end,
                    "helix_length": ex.helix_length,
                    "n_helix_residues": ex.n_helix_residues,
                    "n_partner_residues": ex.n_partner_residues,
                    "n_partner_chains": ex.n_partner_chains,
                    "n_helix_contacts": ex.n_helix_contacts,
                    "n_partner_interface_residues": ex.n_partner_interface_residues,
                    "n_residues_total": ex.n_residues_total,
                    "helix_sequence": ex.helix_sequence,
                    "sasa_used": ex.sasa_used,
                }
                for ex in outcome.examples
            ],
            "warnings": outcome.warnings,
        })
    return results
