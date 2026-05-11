# `twistr.examples` — ML training-example assembly

Third stage of the pipeline. Reads atom14 tensors, detects α-helix
segments, identifies each helix's spatial partner chain(s) via heavy-atom
contact, samples variable-length windows along the helix with
deterministic stable seeding, and writes per-example NPZ files that the
ML stack's `ExamplesDataset` consumes directly. Output feeds `ml/`.

## Helix detection

Two-tier strategy because DSSP isn't always available or reliable:

1. **DSSP-based.** Smooth the SS8 string by filling 1- to 2-residue
   runs of `{G, I, T, S}` between H-class neighbors (these are short
   3₁₀ / π-helix / turn excursions inside what's morphologically a
   regular α-helix — DSSP's strict per-residue classification fragments
   them, but we want contiguous spans). Return inclusive (start, end)
   spans of length ≥ `min_length`.
2. **Geometric fallback.** When DSSP is unavailable or marks a region
   `?`, check CA-CA distance windows for the canonical α-helix
   signature: `d(i, i+3) ≈ 5.4 Å` and `d(i, i+4) ≈ 6.2 Å`. This
   catches helices that DSSP missed (often at chain termini or in
   low-quality density) without false positives in turn / coil regions.

The `is_helix` and `is_interface_residue` bool arrays that flow through
the entire ML stack — gating the dihedral loss, the backbone-continuity
loss, the packing-neighbor loss, the coord-MSE inclusion masks — start
here.

## Partner contact via cKDTree

For each detected helix, an `scipy.spatial.cKDTree` is built over
**every heavy atom of every other chain** in the assembly. A residue is
marked an interface partner if at least one of its heavy atoms lies
within the contact radius (default 5.5 Å) of any helix heavy atom.
Residues are then merged into contact segments across small gaps
(controllable `max_gap`), filtered by minimum / maximum segment length,
and contiguous partner crops are emitted alongside the helix.

The partner crop is intentionally **not** the full partner chain — it's
the contact-radius window around the interface, expanded by
`context_residues` flanking residues on each side. This bounds N per
example (the helix design problem doesn't need to model the entire
antigen) while preserving enough local context for the ML stack's
backbone-continuity loss to operate.

## Stable-seed windowing

A single helix often spans more residues than fit in one training
example (the calibration table caps N at ~200 for memory reasons).
Windows are tiled greedily left-to-right with sizes drawn from
`[window_min, window_max]`. Crucially, the per-helix window-size draw
uses a **stable seed** derived as
`SHA256(pdb_id ‖ assembly_id ‖ helix_index) XOR random_seed`. The same
helix sampled across pipeline reruns produces the same windowing —
making the dataset reproducible without snapshotting NPZ files. The
final window is merged into its predecessor if the remainder is
shorter than `window_min`.

## SASA-based interface filtering (optional)

Solvent-accessible surface area can be computed per residue and a
per-partner δ-SASA (free vs. complex) used to gate the
`is_interface_residue` flag more tightly than raw contact. Toggled by
config; off by default for the current production runs because the raw
contact radius proves a stronger signal at lower compute cost.

## File index

| File | Role |
|---|---|
| `assembly.py` | Helix extraction, `ExampleTensors` slicing, multi-chain example packing. |
| `contacts.py` | cKDTree spatial indexing; residue-level contact marking. |
| `windowing.py` | Stable-seed hashing; greedy variable-size window tiling along a helix. |
| `segmentation.py` | DSSP SS8 smoothing, helix span finding, gap-merging, geometric CA-CA fallback. |
| `sasa.py` | Per-residue SASA + δ-SASA (free vs. complex) computation. |
| `pipeline.py` | Per-entry orchestration: load tensor → detect helices → find partners → window → write NPZ. |
| `driver.py` | Batch / parallel processing over the tensors-stage manifest. |
| `constants.py` | Window sizes, contact radius, SASA parameters, min / max segment lengths. |
| `config.py` | Examples config dataclass. |
| `modal_app.py` | Modal app definition for fan-out parallel example assembly. |
| `report.py` | Per-batch summary: helix count, average length, partner-chain distribution. |
| `finalize_manifest.py` | Recovery script — rebuild the manifest from on-disk NPZs when a Modal run disconnects mid-stream. |
