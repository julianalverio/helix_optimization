# `twistr.tensors` — mmCIF → per-residue Atom14 tensors

Second stage of the pipeline. Reads curated mmCIF files, runs DSSP for
secondary-structure assignment, canonicalizes sidechains for residues
with rotamer-symmetric atom-naming ambiguity, packs heavy-atom
coordinates into the fixed **atom14** layout (14 atom slots per residue,
indexed by residue type), and serializes per-PDB outputs as compressed
NPZ. Output feeds `examples/`.

## The atom14 layout

Variable-size sidechains (GLY has 4 heavy atoms; TRP has 14) are packed
into a single fixed `(N_residues, 14, 3)` coordinate tensor with a
companion `(N_residues, 14)` int8 `atom_mask`. Slot indexing per residue
type is fixed (`N, CA, C, O` at slots 0–3; sidechain atoms at slots 4+ in
a residue-specific order). Missing-slot positions are zero in the
coordinate tensor and `0` in the mask; `1` marks atom present; `-1`
marks the residue itself not present (introduced at batch-collate time
for length padding). This fixed-shape layout is what makes the
downstream ML stack's dynamic-batching memory math possible — every
residue is one fixed-cost lookup regardless of amino acid identity.

## Sidechain canonicalization

Several residue types have rotamer-symmetric naming conventions where
the same physical conformation can appear in the PDB under two atom
labelings:

| Residue | Symmetry |
|---|---|
| ARG | NH1 ↔ NH2 (guanidinium 180° flip) |
| ASP | OD1 ↔ OD2 |
| GLU | OE1 ↔ OE2 |
| PHE | CD1 ↔ CD2 and CE1 ↔ CE2 (ring 180° flip) |
| TYR | CD1 ↔ CD2 and CE1 ↔ CE2 |
| LEU | CD1 ↔ CD2 |
| VAL | CG1 ↔ CG2 |

Both labelings appear in the PDB depending on the depositor. The
canonicalization pass detects the ambiguous case by dihedral sign (e.g.
χ₂ for PHE / TYR) and swaps atom labels to the canonical orientation
when the threshold (typically ±90°) is exceeded. Downstream ML losses
that don't natively handle the symmetry (a direct chi-regression loss
would, for example) see a deterministic labeling rather than a mixture.

## DSSP integration

`dssp.py` is a thin subprocess wrapper around the `mkdssp` binary
(invoked with mmCIF in / out, 180 s timeout). Output is parsed into the
8-class SS8 alphabet (`H G I E B T S - ?`) and stored alongside the
coordinate tensor. A 3-class SS3 view (`H E C -`) is derived for
downstream consumers. The wrapper degrades gracefully if `mkdssp` is
absent — secondary structure is filled with the `?` unknown class
rather than failing the whole pipeline run.

## Numeric format

Coordinates are stored as **float16** — 2× memory savings vs fp32 with
no measurable accuracy loss at heavy-atom precision (interatomic
distances live in the ~1–10 Å regime and fp16's ~0.001 Å resolution is
far below any geometric tolerance the downstream losses care about).
Masks are int8. The ML stack upcasts to fp32 at load time.

## Cofactor tracking

Non-protein ligands (cofactors, small molecules) are stored in a
separate set of arrays: coordinates, elements, residue names, and a
chain-mapping table. This isn't currently used by the ML stack — the
helix design problem ignores cofactors — but it's preserved for later
downstream contact-detection work without re-running the curation phase.

## File index

| File | Role |
|---|---|
| `tensors.py` | Atom14 packing — coordinate / mask / residue-type array assembly. |
| `dssp.py` | `mkdssp` subprocess wrapper + SS8 → SS3 translation. |
| `canonicalize.py` | Rotamer-symmetric sidechain labeling normalization (ARG / ASP / GLU / PHE / TYR / LEU / VAL). |
| `pipeline.py` | Per-entry orchestration: load mmCIF → canonicalize → DSSP → pack → serialize. |
| `driver.py` | Batch iteration over the curation manifest with manifest-out writing. |
| `constants.py` | atom14 slot indices, SS3 / SS8 alphabets, residue type indices. |
| `config.py` | Tensors config dataclass (DSSP timeout, canonicalization thresholds, output dir). |
| `report.py` | Per-batch summary statistics (residue counts, missing-DSSP rate, canonicalization fire-rate). |
| `modal_app.py` | Modal app definition for fan-out parallel processing of the curation manifest. |
