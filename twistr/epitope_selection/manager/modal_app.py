"""Modal app: a thin RPC for the one stage that genuinely needs Modal —
PPI-hotspotID's `critires.sh`, which depends on AmberTools / freesasa / DSSP /
AutoGluon.

All other post-MaSIF processing (residue-graph patches, ScanNet filter,
hotspot filter, visualization) is pure Python and runs on the host so the
host can persist each stage's parquet *between* stages.
"""
from __future__ import annotations

import modal

from .modal_image import pipeline_image

app = modal.App("twistr-epitope-pipeline")


@app.function(
    image=pipeline_image,
    cpu=2.0,
    memory=4096,
    timeout=900,
    retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=30.0),
)
def run_critires(pdb_id: str, cleaned_pdb_bytes: bytes) -> dict:
    """Run PPI-hotspotID's critires.sh on a pre-cleaned PDB.

    Returns {str(ResidueId): score} for every residue critires.sh emitted in
    `results.txt`. The host stringifies the keys so the response is JSON-safe.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from twistr.epitope_selection.hotspot_filter.hotspot_runner import parse_results_txt

    work = Path(tempfile.mkdtemp(prefix=f"twistr_{pdb_id}_"))
    try:
        (work / "input.pdb").write_bytes(cleaned_pdb_bytes)
        result = subprocess.run(
            ["bash", "/ppi/critires.sh"],
            cwd=work, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PPI-hotspot critires.sh failed: "
                f"stderr={result.stderr[-2000:]}\nstdout={result.stdout[-1000:]}"
            )
        scores = parse_results_txt(work / "results.txt")
        return {str(rid): score for rid, score in scores.items()}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
