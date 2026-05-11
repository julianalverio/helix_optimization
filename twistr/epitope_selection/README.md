# `twistr.epitope_selection` — Multi-method epitope discovery on the antigen

Picks which patches of an antigen surface are worth designing a binder
*against*. Runs upstream of helix design — its output (ranked epitope
patches with anchor residues, MaSIF / ScanNet / hotspot scores) is what
the helix-design step (PXDesign + Boltz, external) takes as a target.
The lead-optimization ML model in `twistr.ml` then refines binders
against the chosen epitope.

## Why a multi-method approach

No single binding-site predictor is reliable on the targets we care
about. Each of the three methods we run has independent failure modes:

- **MaSIF** (Gainza et al. 2020) learns from molecular-surface geometry
  and chemical features. Strong on hydrophobic patches, less reliable
  on epitopes dominated by long-range electrostatics.
- **ScanNet** (Tubiana et al. 2022) learns from graph-structured
  residue-level features. Strong on residues with conserved binding
  motifs, less reliable on cryptic or allosteric sites.
- **PPI-hotspot ID** (uses critires + AmberTools + AutoGluon) does
  per-residue hotspot identification by alanine-scan energetics on the
  bound complex. Strong on residues that contribute most ΔΔG to a
  *known* interface, less useful when no co-complex exists.

The pipeline runs all three and combines them as a **ranking signal,
not a gating filter**. The shipped thresholds for ScanNet and hotspot
acceptance are zero — every candidate patch flows through, with
per-method scores attached so downstream selection logic can weight
them however it wants. Hotspot's legacy reject-path classifier is
preserved but informational only.

## Stage pipeline

```
mmCIF → MaSIF → patch extraction → ScanNet rescoring → PPI-hotspot → PyMOL viz
        (Docker)                    (Docker)            (Modal)        (local)
```

Stages are selected by the YAML's `stages: [...]` list — the pipeline
can be run incrementally (just MaSIF, just hotspot on existing patches,
etc.) without recomputing earlier stages. Inter-stage I/O is parquet;
content-hashed caching at the hotspot stage avoids re-running expensive
critires runs on unchanged inputs.

## Selected technical details

- **Residue-graph patch extraction (not mesh patches).** MaSIF natively
  produces mesh-vertex scores. The standard mesh-patch aggregation is
  sensitive to mesh quality and doesn't map cleanly to residue-level
  decisions. We instead build a **residue graph** with two edge types
  — helix-face neighbors (`i ± 4`) and spatial-sidechain neighbors
  (≤ 5 Å Cβ-Cβ) — and extract connected components. Per-residue scores
  come from **top-quartile averaging** of the nearest mesh vertices.
  Patch admission gates on anchor-residue density and patch mean / max
  scores; "anchors" are residues passing the strictest score threshold
  inside the patch.
- **Core / halo node sets.** Two node sets feed the graph CC: a strict
  **core** (helix-class SS3 ∧ rSASA ≥ 0.3 ∧ MaSIF ≥ 0.55) and a more
  permissive **halo** (helix-neighbor ∧ rSASA ≥ 0.1 ∧ MaSIF ≥ 0.50).
  This is a deliberate trade-off — strict core for the binding anchor,
  permissive halo for the surrounding context.
- **Docker for MaSIF and ScanNet.** Both upstream models ship with
  incompatible Python + CUDA + system-library requirements. We invoke
  them through Docker containers (locally) rather than try to make one
  venv host both. Output is per-vertex (MaSIF) / per-residue (ScanNet)
  parquet files, easy to consume from the rest of the pipeline.
- **Modal for PPI-hotspot.** critires.sh is a long-running AmberTools +
  AutoGluon job. Modal handles fan-out for batch runs; results are
  cached by `sha256(structure + critires version)` so a re-run on the
  same target reuses prior work.
- **PyMOL visualization with biochemical-class coloring.** The final
  stage renders one `.pml` per accepted patch, with sticks colored by
  4-class biochemical category (hydrophobic / negative / positive /
  polar) and an override color for hotspot residues. Output goes
  directly into design-review slides.

## File index

```
epitopes/        — MaSIF runner + patch extraction
scannet_filter/  — ScanNet runner + per-patch stat annotation
hotspot_filter/  — PPI-hotspot Modal orchestration + clustering
epitope_viz/     — PyMOL .pml script generation
stages/          — per-stage batch orchestrators (masif, scannet, hotspot, viz)
manager/         — entry point + stage selector + Modal app definition
```

| File | Role |
|---|---|
| `epitopes/masif_runner.py` | Docker invocation for MaSIF site-binding; vertex / face / score tensor extraction. |
| `epitopes/patches.py` | Residue-graph construction, top-quartile score aggregation, helix-face + spatial CC patch extraction. |
| `epitopes/filter.py` | DSSP parsing; residue record construction; core / halo membership via rSASA + DSSP. |
| `epitopes/config.py` | MaSIF + DSSP + graph parameters (helix-face offsets, score aggregation, thresholds). |
| `scannet_filter/scannet_runner.py` | Docker runner for ScanNet per-residue scoring. |
| `scannet_filter/filter.py` | Reads per-residue ScanNet scores; stamps mean / max / positive-fraction stats onto patches. |
| `hotspot_filter/hotspot_runner.py` | Modal orchestration for critires.sh execution. |
| `hotspot_filter/filter.py` | PPI-hotspot clustering; legacy accept-path classifier (informational). |
| `epitope_viz/driver.py` | Reads final patches parquet; builds per-patch PyMOL scripts. |
| `epitope_viz/pymol_writer.py` | Renders `.pml` text with 4-class biochemical coloring + hotspot override. |
| `epitope_viz/aa_classes.py` | Amino-acid → biochemical class lookup. |
| `stages/context.py` | `PipelineContext` — shared config + in-memory row accumulators to avoid disk round-trips. |
| `stages/common.py` | mmCIF decompression + gemmi sanitization (alt-confs, hydrogens, waters, ligands), backbone-completeness filter, DSSP invocation, parquet I/O with fsync. |
| `stages/masif.py` | Batch orchestrator: per-PDB MaSIF → patch extraction → parquet. |
| `stages/scannet.py` | Batch orchestrator: per-PDB ScanNet → stat annotation → parquet append. |
| `stages/hotspot.py` | Batch orchestrator: Modal critires → hotspot clustering → final parquet. |
| `stages/viz.py` | Thin wrapper invoking the PyMOL writer per final patch. |
| `manager/manager.py` | Entry point: stage selection, execution order, error logging, skip tracking. |
| `manager/modal_app.py` | Modal SDK setup; `app.run()` context for remote critires invocation. |
| `manager/modal_image.py` | Modal custom Docker image — AmberTools + critires.sh installation. |
| `_cache.py` | Content-hash signature + validation for hotspot result caching. |
