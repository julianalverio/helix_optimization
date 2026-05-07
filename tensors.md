# Module 2: atom14 Tensor Pipeline

Module 2 consumes the Module 1 manifest (`data/manifests/module1_manifest.parquet`, 128,969 entries) plus the local mmCIF archive at `data/pdb/{mid2}/{pdb_id}.cif.gz`. It produces per-entry atom14 tensor `.npz` files under `data/module2/tensors/{mid2}/{pdb}_{asm}.npz` and a `module2_manifest.parquet`, for Module 3's helix-mediated interface extraction.

## Execution model

The local driver reads each mmCIF's gzipped bytes from disk and ships them **inline** in batch payloads to a Modal function — no S3, no Modal Volume. Workers decompress in memory, run the 14-step pipeline, and stream `.npz` bytes back. Each worker runs `mkdssp --version` once at import and asserts `4.x` before processing.

## Per-entry pipeline (14 steps)

Entry point: `process_entry(mmcif_bytes, pdb_id, assembly_id, m1_meta, cfg)` in `twistr/module2/pipeline.py`.

**Step 1 — Load.** `gzip.decompress` → `gemmi.cif.read_string` → `gemmi.make_structure_from_block` → `structure.setup_entities()`. Parse errors → `drop_reason = "unparseable_mmcif"`.

**Step 2 — Content drops.** Scan `structure.entities`:
- `EntityType.Branched` → `contains_glycan`
- `PolymerType` in {Dna, Rna, DnaRnaHybrid} → `contains_nucleic_acid`
- `PolymerType.PeptideD` → `contains_d_amino_acid`
- Per-residue token scan across every entity's `full_sequence`:
  - residue in `d_amino_acid_codes` → `contains_d_amino_acid`
  - residue in `modified_residues_drop_entry` → `contains_modified_residue`

**Step 3 — Assembly expansion.** `structure.transform_to_assembly(primary_assembly_id, HowToNameCopiedChain.Short)` from `m1_meta`. Failures → `assembly_expansion_failed`. If `large_assembly` is set, restrict chains to those named in `unique_interface_plan.asym_id_{1,2}`.

**Step 4 — Strip solvent & hydrogens.** Delete all H/D atoms. Delete residues whose name is in the solvent set (waters, cryoprotectants, buffers, reductants, phasing heavy atoms, other artifacts — flattened from `solvent_residues`). This runs *before* the interface check so artifacts don't falsely trigger ligand drops.

**Step 5 — Altloc resolution.** Per residue, sum occupancy per altloc label. Pick the altloc with max summed occupancy (alphabetical tie-break). Delete atoms with other altlocs; re-tag survivors with altloc `'\0'`. Ignores `' '` and `'\0'` as non-altloc sentinels.

**Step 6 — MSE/SEC conversion.** For residues matching `modified_residues_convert`, apply atom renames (e.g. `SE → SD` for MSE) and rename the residue to its parent canonical AA (e.g. `MSE → MET`). Set `het_flag = "A"`.

**Step 7 — Canonicalize sidechains** (`canonicalize.py`). For Arg/Asp/Glu/Phe/Tyr/Leu/Val, compute the relevant χ dihedral; if it lies outside `[-90°, 90°]`, swap the symmetric sidechain atom pair into canonical orientation. His is left alone. Idempotent.

**Step 8 — Cofactor extraction + non-protein interface check.**
- First: `_extract_cofactor_block` snapshots every allowed-cofactor residue (hemes, flavins, nicotinamides, pyridoxal, nucleotides, coenzymes, iron-sulfur clusters, metal ions, chlorophylls) as flat arrays: `cofactor_coords`, `cofactor_atom_names`, `cofactor_elements`, `cofactor_residue_names`, `cofactor_residue_indices`, `cofactor_chain_names`. Stored in the output `.npz` for downstream complex reconstruction.
- Then: identify protein residues (canonical 20 + UNK). Skip cofactors (already extracted). For all other non-protein residues, compute an ad-hoc interface set: protein residues with any heavy atom within 5 Å (squared-distance ≤ 25) of a heavy atom on a *different* protein chain. If any remaining non-protein heavy atom lies within 5 Å of the interface set → `non_protein_at_interface`. Otherwise delete all non-protein residues (and empty chains) in place. The interface set is transient; not stored.

**Step 9 — Chain filters.** Per chain, count canonical + UNK residues and UNK fraction:
- `length_ok = n_obs >= min_observed_residues_per_chain` (default 20)
- `unk_ok = unk_frac <= max_unk_fraction_per_chain` (default 0.5)
- `substantive = length_ok and unk_ok`

If fewer than 2 chains are substantive:
- If all non-substantive chains failed *only* on UNK fraction → `unk_dominated_structure`
- Otherwise → `insufficient_protein_chains_after_processing`

**Step 10 — CA-only check.** If every heavy atom across all protein residues is named `CA` → `ca_only_structure`.

**Step 11 — Chain ordering.** Substantive chains sorted alphabetically by `chain.name` — gives a deterministic chain axis.

**Step 12 — DSSP** (`dssp.py`). Write the structure to a temp mmCIF, run `mkdssp --output-format mmcif`, parse with `gemmi.cif`. Returns an `SsKey → (ss3, ss8)` map keyed by `(chain_name, residue_seq_id)`. Partial failure (some residues unassigned) keeps the entry — unassigned slots carry null codes (`ss3=3`, `ss8=8`). Total failure (non-zero exit, unparseable output, zero residues assigned) → `dssp_failed`.

**Step 13 — Build atom14 tensors** (`tensors.py::build_atom14`). For the sorted substantive chains, produce:

| Key | Shape | Dtype | Contents |
|---|---|---|---|
| `n_chains` | scalar | int32 | number of chains |
| `n_max_residues` | scalar | int32 | longest chain's canonical residue count |
| `protein_chain_names` | `(n_chains,)` | `<U8` | chain labels |
| `residue_index` | `(n_chains, n_max)` | int32 | PDB seqid per residue (0 in padding) |
| `residue_type` | `(n_chains, n_max)` | int8 | AF2 residue-type index 0–19, `-1` in padding |
| `ss_3` / `ss_8` | `(n_chains, n_max)` | int8 | DSSP codes, null sentinel in padding/unassigned |
| `coordinates` | `(n_chains, n_max, 14, 3)` | float16 | per atom-slot xyz; 0 where no atom |
| `atom_mask` | `(n_chains, n_max, 14)` | int8 | `-1` = padding residue/slot; `0` = valid residue but this slot not in its canonical atom set, or atom missing/zero-occupancy; `1` = real atom present |
| `cofactor_*` (6 arrays) | `(n_atoms, …)` | see above | flat cofactor heavy atoms, parallel arrays |

Only canonical 20 residues end up in the protein arrays; UNK is counted for chain-size tests but excluded from `residue_type`.

**Step 14 — Serialize.** `np.savez_compressed(BytesIO, **tensors)` → bytes. Integer scalars are cast to `np.int32` so `np.load` returns 0-d arrays.

## Drop reasons (canonical set)

`contains_glycan`, `contains_nucleic_acid`, `contains_d_amino_acid`, `contains_modified_residue`, `non_protein_at_interface`, `ca_only_structure`, `insufficient_protein_chains_after_processing`, `unk_dominated_structure`, `unparseable_mmcif`, `assembly_expansion_failed`, `dssp_failed`, `processing_error`, `batch_retry_exhausted`.

## Driver (`driver.py`)

`run_module2(cfg_path, test_mode, force)`:

1. Load config, make output dir (`test_run/` subdir in test mode), install file log handler.
2. Write `constants.npz` + `config_used.yaml` snapshot once.
3. Read Module 1 manifest. Test mode clips to `test_n_entries` (default 100).
4. Planning pass: for each row, if the tensor already exists and not `--force`, record as pre-existing ok; if the mmCIF is missing on disk, emit a `processing_error` row; otherwise queue.
5. Chunk into batches of `modal_batch_size` (default 10). Build payloads of `{pdb_id, assembly_id, mmcif_bytes, m1_meta, cfg}` dicts.
6. `with app.run(): process_batch.map(payloads, return_exceptions=True)` — streaming fan-out capped by `modal_workers` (default 100). For each completed batch:
   - Exception → every entry in the batch becomes `batch_retry_exhausted` after Modal's 3 retries.
   - Success → write each `.npz` atomically via `{path}.npz.tmp` → `rename`, log warnings, append a manifest row.
7. After all batches, assemble the manifest DataFrame, coerce dtypes (int8/Int32/float32/date/Timestamp per spec), atomically write `module2_manifest.parquet`.
8. `build_summary_report` → `summary_report.md`. In test mode, also `build_test_summary` → `test_summary.md` with 5 pass/fail assertions.

## Modal configuration (`modal_app.py`)

- Image: `modal.Image.micromamba(python_version="3.11").micromamba_install("dssp", channels=["bioconda","conda-forge"]).pip_install("gemmi","numpy","pandas","pyarrow").add_local_python_source("twistr")`
- `@app.function(cpu=1.0, memory=2048, timeout=600, retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30))`
- Worker init: `_ensure_dssp()` runs `mkdssp --version` once per worker, asserts the version string begins with `4.`.

## Output layout

```
data/module2/
  tensors/{mid2}/{pdb}_{asm}.npz      # per-entry atom14 tensors
  module2_manifest.parquet            # per-entry row: status, drop_reason, path, provenance
  summary_report.md                   # ok/drop counts, drop reasons, shape stats
  config_used.yaml                    # frozen config snapshot
  constants.npz                       # residue-type + atom14-slot ordering
  module2.log                         # per-batch progress + warnings
  test_run/                           # same tree for --test-mode
```

## Manifest columns

`pdb_id`, `assembly_id`, `processing_status` (`ok` / `dropped` / `error`), `drop_reason`, `method`, `resolution`, `r_free`, `deposition_date`, `release_date`, `n_chains_processed`, `n_substantive_chains`, `path_tensor`, `pipeline_version` (git SHA, falls back to `PIPELINE_VERSION`), `config_hash` (16-char SHA256 of the hashable config fields), `processing_date` (UTC ISO-8601).

## Production outcome

128,969 entries → 58,747 `ok` (45.6%), 70,052 `dropped`, 170 `batch_retry_exhausted` (0.13%). Top drops: `non_protein_at_interface` (30.3%), `contains_nucleic_acid` (9.6%), `contains_modified_residue` (5.8%), `contains_glycan` (5.6%). Wall time ~140 min at 100 workers.
