# Module 2 Implementation Plan (for AI Agent Execution)

## Goal

Build Module 2 of the PDB protein-protein interaction pipeline. Takes the filtered mmCIF archive and manifest from Module 1, processes entries containing only canonical L-amino acid protein chains, and produces per-entry tensor files for downstream helix-mediated interface extraction in Module 3. Execution runs on Modal with mmCIF files uploaded directly in each invocation and results streamed back to local disk.

## Design Principles

- **Aggressive content filtering.** Drop entries containing anything other than standard L-amino acid polymers (no glycans, modified residues, D-amino acids, nucleic acids).
- **Tag but don't drop atoms.** Annotate atom presence via `atom_mask`; don't delete atoms based on completeness.
- **Reproducible.** Pin pipeline version, config, dependency versions in manifest metadata.
- **Idempotent.** Re-running on an already-processed entry skips by default; `--force` to overwrite.
- **Deterministic.** Chain ordering within each entry is deterministic (sort by `label_asym_id`).
- **Minimal logging.** Single global log; errors only. No per-entry log files.
- **Configurable via YAML.** All thresholds and Modal parameters exposed.
- **Test mode first.** Pipeline includes a test mode for end-to-end validation on a small subset before full runs.
- **Direct upload, direct stream-back.** No intermediate cloud storage. mmCIFs are passed as bytes in Modal invocations; results are returned directly and written to local disk as they arrive.

## Prerequisites and Environment Setup

Before any code is written or run, the agent must confirm the following are ready. Each item is blocking — do not proceed to production runs without verification.

### Modal account and authentication

- Create a Modal account at https://modal.com if not already active.
- Install the Modal Python client: `pip install modal`.
- Authenticate the CLI: `modal token new`.
- Verify a workspace is available and a billing method is on file.
- Default quotas for individual accounts are typically around 100 concurrent containers, which matches `modal_workers` = 100. Production runs don't need a quota increase at this scale; if you later raise `modal_workers` beyond the default quota, file a Modal support request ahead of time.

### Modal container image definition

The Modal function runs inside a container defined in Python. Use the following image specification (conda-based, as it gives reliable access to `mkdssp` v4):

```python
import modal

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("dssp", channels=["bioconda", "conda-forge"])
    .pip_install("gemmi", "numpy", "pandas", "pyarrow")
)
```

Alternative (Debian-based) if micromamba has issues:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("dssp")
    .pip_install("gemmi", "numpy", "pandas", "pyarrow")
)
```

### Worker initialization checks (run once per worker startup)

At the start of each Modal worker's lifetime, before processing any entries, run these checks. If any fail, the worker must fail loudly (raise an exception) so the issue is visible via Modal's error reporting.

- Run `subprocess.run(["mkdssp", "--version"], capture_output=True, check=True)`.
- Assert that the returned version string begins with `4.`. If not, raise with a clear message naming the observed version.
- This ensures every worker has a correctly installed DSSP v4 binary before processing begins.

### Modal function decorator

The per-batch processing function should be declared with:

```python
@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=600,   # 10 min per batch (batches are small)
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=30.0,
    ),
)
def process_batch(batch_payload: dict) -> list[dict]:
    # batch_payload contains a list of (pdb_id, assembly_id, mmcif_bytes, module1_metadata) tuples
    # process each entry, return list of results (each with tensor arrays + manifest row)
    ...
```

No cloud-storage secrets are needed — mmCIFs are uploaded directly in the invocation.

### Local machine requirements

- Sufficient disk space for Module 1 output (mmCIF archive, ~50 GB), Module 2 output (~15 GB), and intermediate buffers.
- Stable internet connection for the duration of the full run (~1-2 hours depending on bandwidth).
- Local entrypoint must stay running continuously during the pipeline. Use `tmux`, `screen`, or a systemd service so it doesn't terminate if the SSH session drops.
- Upload bandwidth is a hard ceiling on ramp-up speed. Estimate: ~15 GB of mmCIF upload total during the run. On a 100 Mbps link that's ~20 minutes of upload time across the full pipeline; on gigabit, ~2 minutes. Plan accordingly.

### Warmup test

Before running the full pipeline, run test mode (`test_mode: true`) on ~100 entries. Verify:

- Test entries complete successfully end-to-end.
- All assertions in `test_summary.md` pass.
- Sample `.npz` files load correctly with expected shapes and dtypes.

Only proceed to the full production run after test mode passes cleanly.

## Configuration (YAML)

| Parameter | Default | Notes |
|---|---|---|
| `solvent_residues` | see below | Stripped without exception |
| `modified_residues_convert` | `{MSE, SEC}` | Converted to parent amino acids |
| `modified_residues_drop_entry` | see below | Presence triggers entry drop |
| `d_amino_acid_codes` | see below | Presence triggers entry drop |
| `min_observed_residues_per_chain` | 20 | For substantive chain filter |
| `max_unk_fraction_per_chain` | 0.5 | Chains with >50% UNK excluded from substantive counting |
| `dssp_executable` | `mkdssp` | Path to DSSP v4 binary |
| `module1_manifest_path` | required | Module 1 output manifest path (local) |
| `local_mmcif_base_path` | required | Local path to mmCIF archive from Module 1 |
| `output_dir` | required | Local directory for streamed results |
| `modal_workers` | 100 | Concurrent Modal workers (fits default individual-account quota) |
| `modal_cpu_per_worker` | 1 | CPUs per worker |
| `modal_memory_mb` | 2048 | Memory per worker (MB) |
| `modal_batch_size` | 10 | Entries per Modal invocation (kept small to stay under payload limits when uploading mmCIFs directly) |
| `modal_timeout_seconds` | 600 | Timeout per batched invocation |
| `modal_max_retries` | 3 | Retry count for failed invocations |
| `modal_retry_backoff_seconds` | 30 | Initial backoff; exponential |
| `test_mode` | false | Enable test mode |
| `test_n_entries` | 100 | Entries to process in test mode |
| `test_modal_workers` | 5 | Concurrent workers in test mode |
| `test_modal_batch_size` | 5 | Entries per invocation in test mode |
| `test_output_subdir` | `test_run` | Subdirectory under output_dir for test results |

### Solvent / buffer / additive list

```yaml
solvent_residues:
  waters: [HOH, DOD, H2O, WAT]
  cryoprotectants: [GOL, EDO, PEG, PG4, PGE, P6G, 1PE, 2PE, MPD]
  buffers: [SO4, PO4, ACT, CIT, TRS, DMS, IMD, FMT, EPE, MES, BTB, TAM]
  reductants: [BME, DTT, EDT]
  other_artifacts: [IPA, MLI, TLA, FLC]
```

### Modified residues to convert (MSE and SEC only)

```yaml
modified_residues_convert:
  MSE: { parent: MET, atom_renames: { SE: SD } }
  SEC: { parent: CYS, atom_renames: { SE: SG } }
```

### Modified residues that trigger entry drop

```yaml
modified_residues_drop_entry:
  - SEP
  - TPO
  - PTR
  - HYP
  - CSO
  - CSD
  - CSS
  - CSX
  - KCX
  - LLP
  - MLY
  - M3L
  - ALY
  - PYL
  - FME
```

### D-amino acid codes

```yaml
d_amino_acid_codes:
  [DAL, DLE, DTR, DVA, DCY, DAS, DGL, DHI, DIL, DPN, DLY, DSN, DTH, DTY,
   DAR, DGN, DPR, DSE]
```

## Modal Execution Architecture

### Deployment

- Module 2 entrypoint runs on the local machine: reads Module 1 manifest, reads mmCIF bytes from local disk, splits entries into batches, submits to Modal, aggregates streamed results.
- Modal function is a batched processor: receives mmCIF bytes directly in the invocation payload along with Module 1 metadata, processes each entry in sequence, and returns results.
- Results stream back to local disk as each batched invocation completes.
- No intermediate cloud storage. mmCIFs go up as bytes; tensors come back as bytes.

### Per-batch data flow

1. Local entrypoint reads Module 1 manifest.
2. Local entrypoint groups entry IDs into batches of `modal_batch_size` (default 10; `test_modal_batch_size` in test mode).
3. For each batch:
   - Local entrypoint reads the gzipped mmCIF bytes for each entry from `local_mmcif_base_path`.
   - Builds a batch payload: list of dicts, each containing `pdb_id`, `assembly_id`, `mmcif_bytes` (gzipped), and the Module 1 metadata row needed for processing (specifically `large_assembly`, `unique_interface_plan`, `primary_assembly_id`).
   - Submits the batch to Modal via async dispatch (non-blocking).
4. Modal function receives the batch payload, decompresses and parses each mmCIF from bytes, processes each entry, and returns a list of result dicts. Each result contains either:
   - `status = "ok"`: tensor arrays serialized as bytes (from `numpy.savez_compressed` applied to an in-memory buffer), plus the manifest row.
   - `status = "dropped"` or `"error"`: drop reason, plus the manifest row.
5. Local entrypoint receives completed batch results via async iteration. For each result:
   - If `ok`: write the tensor bytes to `{output_dir}/tensors/{mid2}/{pdb_id}_{assembly_id}.npz`.
   - Append the manifest row to an in-memory buffer.
   - Write a progress log line every 1000 successfully processed entries.
6. At pipeline end, local entrypoint writes the consolidated manifest parquet and summary report.

### Payload size considerations

- Gzipped mmCIFs average ~100-500 KB each. At `modal_batch_size = 10`, each batch upload is ~1-5 MB, comfortably under Modal's RPC payload limits.
- Result bytes per entry (the `.npz` tensor file) average ~100 KB. Batch result payloads return ~1 MB each.
- If individual mmCIFs exceed ~5 MB (very large assemblies), consider reducing `modal_batch_size` or processing large entries individually. In practice, the largest gzipped mmCIFs are under 3 MB.

### Retries

- Failed invocations are retried up to `modal_max_retries` times with exponential backoff starting at `modal_retry_backoff_seconds`.
- Individual entry failures within a batch (unparseable mmCIF, DSSP failure, etc.) are logged and recorded in the manifest as `drop_reason` values. They do NOT trigger retries.
- Batch-level failures (timeout, OOM, Modal infrastructure errors) trigger retries of the entire batch.
- After all retries exhausted, the batch's entries are recorded with `processing_status = "error"` and `drop_reason = "batch_retry_exhausted"`.

### Non-blocking execution

- Local entrypoint uses Modal's async API to dispatch all batches without waiting on any single batch.
- A local coroutine iterates completed results and writes them to disk as they arrive.
- The pipeline never blocks on a slow entry; slow batches don't delay fast ones.
- Upload bandwidth is the limiting factor on initial ramp-up. Dispatch with appropriate concurrency so the local machine doesn't saturate its connection.

### Test mode

When `test_mode: true`:

- Process only the first `test_n_entries` entries from the Module 1 manifest.
- Use `test_modal_workers` and `test_modal_batch_size` for Modal configuration.
- Write outputs to `{output_dir}/{test_output_subdir}/` instead of `{output_dir}/`.
- All pipeline logic runs as in production — same filters, DSSP, tensor assembly. Only scale and output location differ.
- At pipeline end, emit a `test_summary.md` in the test output directory with pass/fail assertions:
  - At least one entry successfully processed.
  - All `drop_reason` values are from the canonical list.
  - Manifest has all expected columns with correct dtypes.
  - At least one tensor `.npz` file is loadable via `np.load` and contains expected array keys.
  - Loaded tensor arrays have expected shapes and dtypes (e.g., `coordinates` is float16 with shape `(n_chains, N_max, 14, 3)`; `atom_mask` values are in {-1, 0, 1}).
- Exit with a clear success/failure message. Do not proceed to production runs unless the test passes.

Recommended workflow: run test mode first, inspect results, then run production.

## Pipeline Steps (per entry, run inside Modal worker)

### Step 1. Load entry

- Decompress mmCIF bytes (gzip) received in the batch payload.
- Parse with `gemmi.read_structure` (from buffer, not from file).
- Catch `gemmi` parse errors. If malformed: record entry with `drop_reason = "unparseable_mmcif"` and skip remaining steps.
- Use the Module 1 metadata passed in the payload for `large_assembly`, `unique_interface_plan`, `primary_assembly_id`.

### Step 2. Early content-based drop filters

Apply before any processing; drop entry immediately on trigger.

- **Glycans / branched entities.** Any entity with `_entity.type = "branched"` → `drop_reason = "contains_glycan"`.
- **Nucleic acids.** Any polymer entity with `_entity_poly.type` in `{polydeoxyribonucleotide, polyribonucleotide, polydeoxyribonucleotide/polyribonucleotide hybrid}` → `drop_reason = "contains_nucleic_acid"`.
- **D-amino acids.** Any polymer entity with `_entity_poly.type = "polypeptide(D)"`, or any residue code in `d_amino_acid_codes` → `drop_reason = "contains_d_amino_acid"`.
- **Modified residues.** Any residue code in `modified_residues_drop_entry` → `drop_reason = "contains_modified_residue"`.

### Step 3. Expand biological assembly

- If `large_assembly == False`: expand the full primary biological assembly.
- If `large_assembly == True`: expand only chains referenced in `unique_interface_plan`.
- Catch expansion errors: if expansion fails, record entry with `drop_reason = "assembly_expansion_failed"`.
- Use `label_asym_id` as the canonical chain identifier throughout.

### Step 4. Strip solvent and hydrogens

- Remove atoms with residue names in any list under `solvent_residues`.
- Remove all atoms with element `H` or `D`.

### Step 5. Resolve altlocs

For each residue with multiple altlocs:

- Compute total occupancy per altloc group.
- Select the altloc with highest total occupancy; tie-break alphabetically.
- Strip altloc labels.
- Keep zero-occupancy atoms (they get `atom_mask = 0` in Step 13).

### Step 6. Convert MSE → MET and SEC → CYS

- MSE: rename atom `SE` to `SD`; set residue name to `MET`.
- SEC: rename atom `SE` to `SG`; set residue name to `CYS`.

### Step 7. Canonicalize sidechain atom coordinates

For residues with symmetric or ambiguously-labeled sidechain atoms, swap coordinates (not labels) to conform to IUPAC conventions. This makes the dataset geometrically consistent without requiring downstream awareness.

Per-residue rules:
- **Arg (NH1, NH2):** swap coordinates if CD–NE–CZ–NH1 dihedral doesn't match IUPAC convention.
- **Asp (OD1, OD2):** swap based on chi2.
- **Glu (OE1, OE2):** swap based on chi3.
- **Phe (CD1↔CD2 and CE1↔CE2 as paired swap):** swap so chi2 ∈ [-90°, 90°]; both pairs swap together.
- **Tyr:** same paired swap as Phe.
- **Leu (CD1, CD2):** swap based on chi2.
- **Val (CG1, CG2):** swap based on chi1.
- **His:** do not swap ND1/NE2; no action or logging.

Test against a small set of reference structures after implementation.

### Step 8. Handle remaining non-protein entities (ions, small molecules)

After Steps 2 and 4, any remaining non-protein atoms are ions or small molecules not in the solvent list.

This step uses an ad-hoc, transient contact scan — no reusable contact matrix is built or stored. For each non-protein group:

- Ad-hoc interface-residue identification (not stored): find protein residues that have any heavy atom within 5 Å of a heavy atom on a different protein chain. These are the "interface residues" for this check only.
- Ad-hoc proximity check: determine whether any heavy atom of the non-protein group is within 5 Å of any interface residue identified above.
- If yes: drop entry with `drop_reason = "non_protein_at_interface"`.
- If no: strip the non-protein group from the structure; keep proteins.

The interface-residue identification is computed only for this step's purposes and discarded afterward. No contact data is propagated to later steps or to the output.

### Step 9. Final chain and entry filters

- **UNK filter.** For each chain, compute fraction of residues with code UNK. Chains with `UNK fraction > max_unk_fraction_per_chain` are excluded from substantive-chain counting.
- **Substantive chain count.** A chain is substantive if `n_residues_observed ≥ min_observed_residues_per_chain` AND UNK fraction ≤ `max_unk_fraction_per_chain`.
- If fewer than 2 substantive chains:
  - If all candidate chains failed specifically on UNK dominance: `drop_reason = "unk_dominated_structure"`.
  - Otherwise: `drop_reason = "insufficient_protein_chains_after_processing"`.

### Step 10. CA-only check

- Check if any chain has any non-CA heavy atom.
- If no chain has any non-CA heavy atom: drop entry with `drop_reason = "ca_only_structure"`.

### Step 11. Determine chain ordering

Sort retained chains alphabetically by `label_asym_id`. This is the canonical chain order for all tensor arrays. Deterministic across runs.

### Step 12. Run DSSP

- Write processed structure to a temporary mmCIF inside the worker's local filesystem (e.g., `/tmp`).
- Run `mkdssp` v4 via subprocess.

Handle the outcome as follows:

- **DSSP succeeds and assigns SS to at least one residue:** parse output. Per residue:
  - `ss_8` (int8): 0=H, 1=G, 2=I, 3=E, 4=B, 5=T, 6=S, 7=coil, 8=null.
  - `ss_3` (int8): 0=H (from H/G/I), 1=E (from E/B), 2=C (from T/S/coil), 3=null.
  - Residues DSSP couldn't assign (e.g., incomplete backbone) get the null codes (`ss_8 = 8`, `ss_3 = 3`). Partial DSSP failures do NOT cause entry drop; continue processing.
- **DSSP fails altogether:** the subprocess exits non-zero, produces unparseable output, OR successfully runs but assigns a non-null SS to zero residues. In this case, drop the entry with `drop_reason = "dssp_failed"`.

### Step 13. Build atom14 tensors

For each retained protein chain:

- **Residue type:** int8 0-19 per AF2 / OpenFold canonical ordering.
- **Atom14 slot ordering:** AF2 / OpenFold canonical 14-slot ordering per residue type.
- **Coordinates tensor (`coordinates`):** shape `(n_chains, N_max, 14, 3)` float16.
- **Atom mask tensor (`atom_mask`):** shape `(n_chains, N_max, 14)` int8, tri-state.

Rules for assigning each atom_mask value:

- If the residue position is padding (chain shorter than `N_max` at that position): all 14 slots for that residue are `-1`. Padding takes precedence over every other rule.
- For non-padded residues, for each of the 14 slots:
  - Slot is not part of this residue type's canonical atom set (e.g., slot 13 for alanine, which has only 5 heavy atoms): value is `0`.
  - Slot is part of the residue's canonical atom set but the atom is missing from the deposited structure: value is `0`.
  - Slot is part of the residue's canonical atom set but the atom has zero occupancy (after altloc resolution): value is `0`.
  - Slot is part of the residue's canonical atom set and the atom is present with non-zero occupancy: value is `1`.

Summary: `-1` only means "this residue doesn't exist here (padding)." `0` means "no usable atom at this slot for any reason." `1` means "real atom present."

### Step 14. Serialize tensor output

In the worker, write the tensor arrays via `numpy.savez_compressed` to an in-memory `BytesIO` buffer. Return the serialized bytes as part of the result dict. The local entrypoint writes them to disk at `{output_dir}/tensors/{mid2}/{pdb_id}_{assembly_id}.npz`.

Contents:

**Entry-level scalars:**
- `n_chains`: int
- `n_max_residues`: int

**Residue-level arrays — shape `(n_chains, N_max)`:**
- `residue_index`: int32
- `residue_type`: int8
- `ss_3`: int8
- `ss_8`: int8

**Atom-level arrays:**
- `coordinates`: `(n_chains, N_max, 14, 3)` float16
- `atom_mask`: `(n_chains, N_max, 14)` int8

PDB ID and assembly ID are encoded in filename; no such fields inside the file.

## Reference Lookups (Global)

Path: `{output_dir}/constants.npz`. Written once by the local entrypoint at pipeline start; does not go through Modal.

Contents:
- `residue_type_names`: `(20,)` string — AF2 canonical ordering.
- `atom14_slot_names`: `(20, 14)` string — IUPAC atom name per slot per residue type.
- `ss_3_codes`: `(4,)` string — `H`, `E`, `C`, `-`.
- `ss_8_codes`: `(9,)` string — `H`, `G`, `I`, `E`, `B`, `T`, `S`, `-`, `?`.

## Module 2 Manifest

Single parquet file at `{output_dir}/module2_manifest.parquet`. Written via pandas by the local entrypoint after all batches complete. One row per entry processed.

Columns:
- `pdb_id`: string (uppercase, 4 chars)
- `assembly_id`: int8
- `processing_status`: string — `"ok"`, `"dropped"`, or `"error"`
- `drop_reason`: string (nullable)
- `method`: string (from Module 1)
- `resolution`: float32 (nullable)
- `r_free`: float32 (nullable)
- `deposition_date`: date
- `release_date`: date
- `n_chains_processed`: int32 (null if not `ok`)
- `n_substantive_chains`: int32 (null if not `ok`)
- `path_tensor`: string (null if not `ok`)
- `pipeline_version`: string (git SHA)
- `config_hash`: string
- `processing_date`: timestamp

Manifest-level metadata (parquet metadata block):
- Pipeline version (git SHA), full config YAML, dependency versions (gemmi, mkdssp, numpy, pandas), Module 1 manifest path used, processing date range.

## Canonical Drop Reasons

- `contains_glycan`
- `contains_nucleic_acid`
- `contains_d_amino_acid`
- `contains_modified_residue`
- `non_protein_at_interface`
- `ca_only_structure`
- `insufficient_protein_chains_after_processing`
- `unk_dominated_structure`
- `unparseable_mmcif`
- `assembly_expansion_failed`
- `dssp_failed`
- `processing_error`
- `batch_retry_exhausted`

## Logging

Single consolidated global log at `{output_dir}/module2.log`. Written by the local entrypoint using standard Python logging.

Log content:
- Pipeline start and end.
- Per-batch dispatch and completion events.
- Errors with full tracebacks (both local and worker-reported).
- Progress markers every 1000 successfully processed entries.
- DSSP whole-entry failures (one log line per occurrence, triggered by the `dssp_failed` drop reason).

Explicitly NOT logged:
- Per-entry success messages.
- Altloc choices or sidechain canonicalization swaps.
- Histidine connectivity checks.
- Per-residue DSSP partial failures (those are silently reflected as null SS codes in the output).

No per-entry JSONL logs.

Modal workers include any warnings or errors in their returned result dicts; the local entrypoint transcribes these to the global log.

## Summary Report

Path: `{output_dir}/summary_report.md`. Written at pipeline end.

Contents:
- Total entries from Module 1 manifest.
- Entries processed successfully.
- Entries dropped, broken down by `drop_reason` (counts and percentages).
- Entries with infrastructure errors.
- Distribution of `n_substantive_chains` for successful entries.
- Distribution of `n_max_residues` for successful entries.
- Total wall time and Modal compute cost summary.

## Recommended Python Stack

- **mmCIF parsing and assembly expansion:** `gemmi` (parsing from in-memory bytes; no file I/O required in workers)
- **DSSP:** external `mkdssp` v4, via subprocess (installed in Modal container via micromamba)
- **Tensor construction and I/O:** `numpy`; save via `numpy.savez_compressed` to `BytesIO` in workers, written to disk by local entrypoint
- **Manifest:** `pandas` with `pyarrow` for parquet I/O
- **Modal execution:** `modal` (latest)
- **Logging:** standard library `logging`

## Output Directory Layout

```
{output_dir}/
  module2_manifest.parquet
  summary_report.md
  module2.log
  config_used.yaml
  constants.npz
  tensors/
    ab/
      1abc_1.npz
      ...
  test_run/                   # present only if test mode was used
    module2_manifest.parquet
    summary_report.md
    test_summary.md
    module2.log
    constants.npz
    tensors/
      ab/
        1abc_1.npz
        ...
```

## Implementation Notes

- Each Modal worker receives mmCIF bytes and Module 1 metadata directly in the batch payload. No file system or cloud access from workers.
- Each worker validates DSSP v4 is installed at startup (see Worker initialization checks).
- Output from each batch is streamed back to the local entrypoint as the batch completes, and written to the local filesystem.
- Chain ordering within each entry is deterministic (sort by `label_asym_id`).
- Local file I/O for outputs is atomic: write to `.tmp`, rename on completion.
- Re-running is idempotent: entries with existing output files are skipped unless `--force`.
- `label_asym_id` is the canonical chain identifier; internal chain index (0-based) matches sorted order.
- AF2 / OpenFold canonical atom14 ordering and residue-type indexing are hardcoded in a constants module and also written to `constants.npz` at pipeline start by the local entrypoint.
- Catch `gemmi` parse errors, DSSP subprocess failures, and assembly expansion errors inside workers; record appropriate drop reasons.
- Keep `modal_batch_size = 10` by default to stay well under Modal's RPC payload limits when uploading mmCIFs directly. Larger batch sizes are possible if mmCIFs are small; monitor for payload errors in test mode.
- Run `test_mode: true` first on a small subset (~100 entries) and verify all `test_summary.md` assertions pass before launching the production run.
- Local entrypoint must stay running for the full duration of the pipeline. Use `tmux`, `screen`, or equivalent to prevent accidental termination.