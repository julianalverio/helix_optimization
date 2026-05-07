# Module 3 Implementation Plan (for AI Agent Execution)

## Goal

Build Module 3 of the PDB protein-protein interaction pipeline. Takes Module 2's processed tensor outputs and produces training examples of helix-mediated interchain interfaces. Each training example captures a single helix window from one chain plus the residues on other chains ("partners") that interact with it, in a unified tensor format suitable for training a generative model that produces small helical binders (2.5-3.5 turns).

Execution runs on Modal with Module 2 `.npz` tensors uploaded directly in each invocation and extracted training examples streamed back to local disk.

## Design Principles

- **One training example = one helix window.** Each example captures an 8-15 residue alpha-helical segment that contacts at least one other chain, plus the partner residues it interacts with.
- **Unified residue tensor.** Helix and partner residues stored in a single tensor, with helix residues first (indices 0 to n_helix_residues - 1) followed by partner residues. Per-residue annotations (chain_slot, is_helix, is_interface_residue, etc.) distinguish them.
- **Tiled non-overlapping windows.** A long interacting helix produces multiple training examples; adjacent windows are correlated (accepted as mild leakage, handled downstream via splits).
- **Include all contacted chains as partners.** A helix contacting multiple chains produces one example with all contacted chains merged as the partner context.
- **Quality-filter but don't over-filter.** Minimum atom completeness per example. No per-example dedup.
- **Reproducible.** Seeded RNG; all config, dependency versions, and provenance captured in manifests.
- **Idempotent.** Re-running on an already-processed entry skips by default; `--force` to overwrite.
- **Direct upload, direct stream-back.** No intermediate cloud storage. Module 2 `.npz` tensors are passed as bytes in Modal invocations; extracted examples are returned and written to local disk.

## Prerequisites and Environment Setup

### Modal account and authentication

- Existing Modal account (same one used for Module 2).
- Existing authentication via `modal token new`.
- Concurrency quota of 200+ workers (Module 3 is lighter than Module 2; 1000 workers not needed).

### Modal container image definition

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "freesasa", "gemmi", "scipy")
)
```

No DSSP needed — Module 3 uses Module 2's pre-computed secondary structure.

### Worker initialization checks

At the start of each Modal worker, before processing any entries:

- Verify `freesasa`, `numpy`, `pandas`, `gemmi`, `scipy` import cleanly.
- If any fail, raise with a clear message.

### Modal function decorator

```python
@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=600,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=30.0,
    ),
)
def process_batch(batch_payload: dict) -> list[dict]:
    # batch_payload contains a list of (pdb_id, assembly_id, module2_npz_bytes, module2_metadata) tuples
    # for each entry, extract helix-mediated interface training examples
    # return list of results (each with extracted examples as serialized bytes + manifest rows)
    ...
```

### Local machine requirements

- Sufficient disk space for Module 2 output (~15 GB) and Module 3 output (~30-60 GB depending on example count).
- Stable internet connection for pipeline duration.
- Local entrypoint must stay running throughout. Use `tmux` or `screen`.

### Warmup test

Before running the full pipeline, run test mode (`test_mode: true`) on ~100 entries. Verify:

- Test entries complete successfully end-to-end.
- All assertions in `test_summary.md` pass.
- Sample `.npz` example files load correctly with expected shapes and dtypes.

Only proceed to the full production run after test mode passes cleanly.

## Configuration (YAML)

| Parameter | Default | Notes |
|---|---|---|
| `module2_output_dir` | required | Path to Module 2's output directory containing tensors and manifest |
| `module2_manifest_path` | required | Path to Module 2's manifest parquet |
| `output_dir` | required | Local directory for streamed Module 3 results |
| `min_helix_segment_length` | 6 | Minimum contiguous (smoothed) alpha-helix length to consider |
| `dssp_smoothing` | true | Smooth 1-2 residue G/I/T/S interruptions within H stretches for segmentation |
| `contact_distance_heavy_atom` | 5.0 | Å; heavy-atom distance for ALL contact-related decisions |
| `max_helix_gap_residues` | 7 | Max non-contacting helix residues that still merge segments |
| `window_length_min` | 8 | Minimum window length (residues) |
| `window_length_max` | 15 | Maximum window length (residues) |
| `min_contacts_per_window` | 2 | Minimum number of contacting helix residues per window |
| `partner_use_sasa` | true | Include SASA-decreasing residues in partner set (union with distance-based) |
| `partner_sasa_threshold` | 1.0 | Å²; minimum ΔSASA to count as SASA-decreasing |
| `partner_sequence_context` | 2 | Residues on each side of interface residues |
| `min_backbone_atom_completeness` | 0.8 | Fraction of N/CA/C/O present across helix + partners |
| `random_seed` | 42 | Seed for window-length sampling |
| `modal_workers` | 200 | Concurrent Modal workers |
| `modal_cpu_per_worker` | 1 | CPUs per worker |
| `modal_memory_mb` | 2048 | Memory per worker (MB) |
| `modal_batch_size` | 10 | Entries per Modal invocation |
| `modal_timeout_seconds` | 600 | Timeout per batched invocation |
| `modal_max_retries` | 3 | Retry count for failed invocations |
| `modal_retry_backoff_seconds` | 30 | Initial backoff; exponential |
| `test_mode` | false | Enable test mode |
| `test_n_entries` | 100 | Entries to process in test mode |
| `test_modal_workers` | 5 | Concurrent workers in test mode |
| `test_modal_batch_size` | 5 | Entries per invocation in test mode |
| `test_output_subdir` | `test_run` | Subdirectory under output_dir for test results |

## Modal Execution Architecture

### Deployment

- Module 3 entrypoint runs on the local machine: reads Module 2 manifest, groups entry IDs into batches, uploads Module 2 tensor bytes to Modal, receives extracted examples back, writes to local disk.
- Modal function processes each entry in a batch, extracting all helix-mediated interface training examples for that entry.
- No intermediate cloud storage.

### Per-batch data flow

1. Local entrypoint reads Module 2's manifest.
2. Local entrypoint groups entry IDs into batches of `modal_batch_size` (default 10).
3. For each batch:
   - Reads the Module 2 `.npz` bytes for each entry from `module2_output_dir/tensors/{mid2}/{pdb_id}_{assembly_id}.npz`.
   - Builds a batch payload: list of dicts, each containing `pdb_id`, `assembly_id`, `module2_npz_bytes`, and pass-through Module 2 metadata (`resolution`, `r_free`, `method`, `deposition_date`, `release_date`).
   - Submits batch to Modal via async dispatch (non-blocking).
4. Modal worker receives the batch, loads each entry's `.npz` from bytes, extracts all training examples, returns a list of result dicts. Each result contains:
   - `pdb_id`, `assembly_id`, `processing_status` (`"ok"`, `"no_examples"`, `"dropped"`, or `"error"`)
   - `drop_reason` if applicable
   - List of extracted examples, each serialized as `.npz` bytes (from `numpy.savez_compressed` to BytesIO)
   - Per-example metadata rows for aggregation into the Module 3 manifest
   - Any warnings (SASA failures etc.)
5. Local entrypoint receives completed batch results via async iteration. For each result:
   - For each extracted example: write bytes to `{output_dir}/examples/{mid2}/{pdb_id}_{assembly_id}_{example_id}.npz`.
   - Write a marker file `{output_dir}/markers/{mid2}/{pdb_id}_{assembly_id}.marker` (empty file) atomically after all examples for an entry have been written. The marker indicates "this entry has been fully processed."
   - Append per-example rows to an in-memory buffer.
   - Log progress every 1000 successfully extracted examples.
6. At pipeline end, local entrypoint writes the consolidated Module 3 manifest parquet, per-entry status parquet, and summary report.

### Idempotency

Before processing each entry:
- Check if `{output_dir}/markers/{mid2}/{pdb_id}_{assembly_id}.marker` exists.
- If yes, and `--force` is not set, skip this entry.
- If no, process the entry. After all examples are written successfully, write the marker.
- Marker writing is atomic (write to `.tmp`, then rename).
- Entries that produced zero examples still get markers written (so they don't get re-attempted).

### Retries

- Failed Modal invocations retried up to `modal_max_retries` times with exponential backoff.
- Individual entry failures within a batch (corrupt Module 2 `.npz`, processing exceptions, etc.) are logged and recorded. They do NOT trigger retries.
- Batch-level failures (timeout, OOM, Modal infrastructure errors) trigger retries of the entire batch.
- After all retries exhausted, batch's entries get `processing_status = "error"` and `drop_reason = "batch_retry_exhausted"`.

### Non-blocking execution

- Local entrypoint uses Modal's async API to dispatch all batches without waiting.
- Coroutine iterates completed results and writes to disk as they arrive.

### Test mode

When `test_mode: true`:

- Process only the first `test_n_entries` entries from Module 2's manifest.
- Use `test_modal_workers` and `test_modal_batch_size`.
- Write outputs to `{output_dir}/{test_output_subdir}/`.
- At pipeline end, emit `test_summary.md` with pass/fail assertions:
  - At least one example successfully extracted.
  - All `drop_reason` values are from the canonical list.
  - Manifest has all expected columns with correct dtypes.
  - At least one example `.npz` file is loadable and contains all expected array keys (see Output Schema).
  - Loaded example tensor shapes and dtypes are correct (e.g., `coordinates` is float16 with shape `(n_residues, 14, 3)`).
  - For each loaded example: residues with `is_helix == True` come BEFORE residues with `is_helix == False` in the tensor (helix-first ordering invariant).
  - For each loaded example: `chain_slot[0] == 0` (first residue is helix chain).
- Exit with clear success/failure message. Do not proceed to production unless test passes.

## Pipeline Steps (per entry, run inside Modal worker)

### Step 1. Load entry

- Decompress Module 2 `.npz` bytes via `np.load` on a BytesIO buffer.
- Extract arrays: `n_chains`, `n_max_residues`, `residue_index`, `residue_type`, `ss_3`, `ss_8`, `coordinates`, `atom_mask`.
- Catch load errors. If unparseable: record with `drop_reason = "unparseable_module2_output"` and skip.

### Step 2. Extract per-chain working data

For each chain (0 to n_chains-1):

- Identify real residues: those where `(atom_mask[chain, residue, :] != -1).any()`.
- Extract per-chain working data: residue indices into the Module 2 tensor's residue dimension, SEQRES positions (from `residue_index`), residue types, ss_3 values, ss_8 values, coordinates, atom_masks.

### Step 3. DSSP smoothing per chain

For each chain, compute a smoothed ss_8 array used for helix segmentation only:

- Walk the chain's real residues' ss_8 array.
- Identify runs of 1-2 consecutive residues with ss_8 in {1 (G), 2 (I), 5 (T), 6 (S)}.
- A run is "embedded" if the residue immediately before and the residue immediately after are both H (ss_8 == 0).
- For embedded runs, rewrite ss_8 to 0 (H) in the smoothed array.
- The original ss_8 is preserved unchanged in the output; only the smoothed version is used for downstream segmentation.

### Step 4. Identify helix segments

For each chain:

- Find contiguous runs of smoothed ss_8 == 0 of length ≥ `min_helix_segment_length` (default 6).
- Each run is a "helix segment" identified by (chain_index, start_position_in_chain, end_position_in_chain), where positions are 0-based into the chain's real-residue list.

If no helix segments exist in any chain across the entry: record entry with `drop_reason = "no_helix_segments"` and skip remaining steps.

### Step 5. Build entry-wide spatial index

Build a single KD-tree containing all heavy atoms across all chains in the entry.

- For each atom in `coordinates` where `atom_mask == 1`: add to the tree with metadata (chain_index, residue_position_in_chain, atom_slot).
- Use `scipy.spatial.KDTree` or equivalent.

This index is built once per entry and queried with chain-filtering for each helix segment.

### Step 6. Identify helix residues with inter-chain contacts

For each helix segment on chain c:

- For each residue in the segment, query the spatial index for atoms within `contact_distance_heavy_atom` (default 5.0 Å) of any of this residue's heavy atoms.
- Filter the query results to exclude atoms on chain c (intra-chain contacts don't count).
- Mark the residue `is_contacting = True` if any non-c atom is within range; False otherwise.

### Step 7. Merge contact-interrupted sub-segments (gap rule)

For each helix segment:

- Look at the sequence of `is_contacting` flags along the segment.
- Identify "interaction regions" = maximal runs of `is_contacting == True`.
- Between adjacent interaction regions, count the number of consecutive `is_contacting == False` helix residues (the "gap").
- Merge adjacent interaction regions if their gap is ≤ `max_helix_gap_residues` (default 7).
- The merged region (spanning the first contact to the last contact of merged regions, inclusive of intervening residues) is an "interacting helix."
- If no residues in a segment contact another chain, no interacting helix is produced from that segment.

If no interacting helices exist in the entry: record with `drop_reason = "no_interacting_helices"` and skip remaining steps.

### Step 8. Filter interacting helices by length

- Discard interacting helices shorter than `window_length_min` (default 8 residues).
- These are too short to produce valid windows.

### Step 9. Compute window tiling per interacting helix

For each interacting helix of length L, with seed `random_seed + hash((pdb_id, assembly_id, helix_index))`:

- If L ≤ `window_length_max` (default 15): single window covering the full interacting helix.
- If L > `window_length_max`: tile greedily left to right.
  - Sample window length uniformly from [`window_length_min`, `window_length_max`] using the seeded RNG.
  - Advance by that length; sample again.
  - Continue until the remaining helix length ≤ `window_length_max`. Take the remaining residues as the final window.
  - If the final window has length < `window_length_min`: merge it into the previous window. The merged previous window may exceed `window_length_max` by up to (window_length_max - window_length_min - 1) residues. This is acceptable and rare.

`helix_index` is determined by deterministic iteration: chains in chain_index order (Module 2's alphabetical-by-label_asym_id order); within each chain, helix segments in position order; within the entry, sequential index 0, 1, 2, ...

### Step 10. For each window, check contact count

- Count helix residues within the window where `is_contacting == True`.
- If count < `min_contacts_per_window` (default 2): discard window.

### Step 11. For each surviving window, identify partner chains

For each window:

- Identify the set of chains contacted by any window residue (per the spatial index query from Step 6, restricted to the window).
- Exclude the helix's own chain. The remaining set is the partner chains for this window.

### Step 12. For each partner chain, compute SASA-based residue selection

For each window with N partner chains:

**Build the atom set for SASA computation.**

The "complex" includes all heavy atoms from:
- The helix chain (all residues, not just the helix window).
- All partner chains (full chains).

The "complex minus helix window" is identical except the heavy atoms of helix window residues are removed. Other residues on the helix chain remain in both computations.

**Run two SASA computations using freesasa:**

1. SASA of the full complex.
2. SASA of complex minus helix window residues.

For each atom in the input, freesasa returns per-atom SASA. Aggregate per residue by summing atom SASAs belonging to that residue.

**Compute ΔSASA per residue:**

For each residue in the partner chains:

```
ΔSASA = SASA_minus_helix_residue - SASA_full_complex_residue
```

Positive ΔSASA means the residue is more exposed when the helix is removed (i.e., the helix was burying it).

**Mark partner residues as SASA-decreasing if ΔSASA ≥ `partner_sasa_threshold` (default 1.0 Å²).**

**Atom-to-freesasa mapping:** for each atom with `atom_mask == 1`, use the atom's slot and the residue's parent residue type to look up the IUPAC atom name from `atom14_slot_names`. Determine element from atom name. Use freesasa's default radii (Lee-Rupley, probe radius 1.4 Å). Tag each atom in the freesasa input with (chain_slot, residue_position) for per-residue aggregation after computation.

**SASA failure handling:** if freesasa raises an error (rare; typically on degenerate inputs), fall back to distance-only partner selection (skip the SASA criterion). Log as warning per example. Set `sasa_used = False` in metadata.

### Step 13. For each partner chain, compute distance-based residue selection

For each partner chain:

- For each partner chain residue, query the spatial index for atoms within `contact_distance_heavy_atom` (default 5.0 Å) of any helix window atom.
- Restrict query to atoms on this partner chain.
- Mark residue as distance-based interface if any of its atoms is within the cutoff.

### Step 14. Combine partner residue selections

For each partner chain:

- Take the union of distance-based interface residues and SASA-decreasing residues. These are the "interface residues" for this partner chain (per-residue `is_interface_residue == True`).
- Expand the interface residue set by ±`partner_sequence_context` (default 2) sequence positions on each side. Clamp to the partner chain's real-residue range.
- Residues added by the sequence context expansion (not already interface residues) are partner residues with `is_interface_residue == False`.

### Step 15. Assemble the unified residue tensor

**Critical ordering invariants:**

- Indices [0, n_helix_residues) are helix window residues. `is_helix == True`.
- Indices [n_helix_residues, n_residues) are partner residues. `is_helix == False`.
- Within the helix region: residues in SEQRES position ascending order.
- Within the partner region: ordered by chain_slot ascending (1, 2, 3, ...). Within each chain_slot, residues in SEQRES position ascending order.

**chain_slot semantics (document explicitly in code and metadata):**

- `chain_slot == 0`: ALWAYS the helix chain.
- `chain_slot >= 1`: partner chains, indexed in ascending order of their Module 2 chain_index (which is alphabetical by label_asym_id).
- This is distinct from Module 2's chain_index, which refers to a chain's position in the Module 2 tensor. Mapping back to Module 2 is via `chain_module2_index`. Mapping to source mmCIF is via `chain_label`.

**Build per-residue arrays:**

| Field | Shape | Type | Source |
|---|---|---|---|
| `coordinates` | `(n_residues, 14, 3)` | float16 | sliced from Module 2 |
| `atom_mask` | `(n_residues, 14)` | int8 | sliced; tri-state preserved |
| `residue_type` | `(n_residues,)` | int8 | sliced |
| `ss_3` | `(n_residues,)` | int8 | sliced (ORIGINAL, not smoothed) |
| `ss_8` | `(n_residues,)` | int8 | sliced (ORIGINAL, not smoothed) |
| `chain_slot` | `(n_residues,)` | int8 | computed (0 for helix, 1+ for partners) |
| `seqres_position` | `(n_residues,)` | int32 | from Module 2's `residue_index` |
| `is_helix` | `(n_residues,)` | bool | True for [0, n_helix_residues), False otherwise |
| `is_interface_residue` | `(n_residues,)` | bool | unified field (see semantics below) |

**`is_interface_residue` semantics:**

- For helix residues (`is_helix == True`): True if the residue contacts any partner chain (i.e., `is_contacting == True` from Step 6, restricted to the helix window).
- For partner residues (`is_helix == False`): True if the residue is in the interface residue set from Step 14 (distance ∪ SASA), False if included only as sequence context.

Downstream four-way classification:
- `is_helix=True, is_interface_residue=True`: helix residue making contact with partners.
- `is_helix=True, is_interface_residue=False`: helix residue in window but not contacting (still structurally part of the window; included for helix integrity).
- `is_helix=False, is_interface_residue=True`: partner residue, direct interface contact.
- `is_helix=False, is_interface_residue=False`: partner residue, sequence context only.

### Step 16. Build per-chain metadata

For each chain present in the example (helix chain + partner chains):

| Field | Shape | Type |
|---|---|---|
| `chain_label` | `(n_chains_in_example,)` | string array (variable-length) |
| `chain_module2_index` | `(n_chains_in_example,)` | int8 |
| `chain_role` | `(n_chains_in_example,)` | int8 (0 = helix, 1 = partner) |

Index 0 in these arrays is the helix chain; indices 1+ are partner chains in chain_slot order.

`chain_label` for each chain is the source mmCIF's label_asym_id. Module 2 doesn't store this directly in the tensor, so during Module 3 processing, the agent must either:
- Re-load the source mmCIF to recover label_asym_ids, OR
- Have Module 2 amended to include chain_labels in its `.npz` output.

**Recommendation:** if Module 2's output already includes chain labels (verify this), use them. If not, the simplest path is to leave `chain_label` blank for now (empty strings) and add it later when Module 2 is updated. Document the limitation.

### Step 17. Quality filter

Compute backbone atom completeness for the example:

- Backbone atoms: N (atom14 slot 0), CA (slot 1), C (slot 2), O (slot 4) per AF2 atom14 ordering.
- Count atoms with `atom_mask == 1` at these slots across all residues in the example.
- Count expected backbone atoms (4 per residue × n_residues).
- Compute fraction = present / expected.
- If fraction < `min_backbone_atom_completeness` (default 0.8): discard example.

### Step 18. Assign example ID and serialize

- Assign `example_id` integer (0-indexed within entry, deterministic per iteration order).
- Serialize all per-residue arrays, per-chain arrays, and scalar metadata to `.npz` bytes via `numpy.savez_compressed` to a BytesIO buffer.
- Return the bytes to the local entrypoint.

### Step 19. Local-side: write to disk

Local entrypoint receives results. For each example:
- Write bytes to `{output_dir}/examples/{mid2}/{pdb_id}_{assembly_id}_{example_id}.npz`.
- Sharded by middle two characters of PDB ID, matching Module 2's convention.

After all examples for an entry are written:
- Write marker file `{output_dir}/markers/{mid2}/{pdb_id}_{assembly_id}.marker` (atomic: tmp + rename).

## Output Schema (per `.npz`)

### Per-residue arrays — shape `(n_residues,)` or `(n_residues, ...)`

Helix residues at indices [0, n_helix_residues), partner residues at indices [n_helix_residues, n_residues).

| Field | Shape | Type | Notes |
|---|---|---|---|
| `coordinates` | `(n_residues, 14, 3)` | float16 | atom14 layout |
| `atom_mask` | `(n_residues, 14)` | int8 | tri-state {-1, 0, 1} |
| `residue_type` | `(n_residues,)` | int8 | 0-19, AF2 ordering |
| `ss_3` | `(n_residues,)` | int8 | original DSSP, unsmoothed |
| `ss_8` | `(n_residues,)` | int8 | original DSSP, unsmoothed |
| `chain_slot` | `(n_residues,)` | int8 | 0 = helix, 1+ = partners |
| `seqres_position` | `(n_residues,)` | int32 | SEQRES position in source chain |
| `is_helix` | `(n_residues,)` | bool | True for helix, False for partners |
| `is_interface_residue` | `(n_residues,)` | bool | True for residues making/receiving contact (see Step 15 for semantics) |

### Per-chain arrays — shape `(n_chains_in_example,)`

| Field | Shape | Type | Notes |
|---|---|---|---|
| `chain_label` | `(n_chains_in_example,)` | string array | label_asym_id from source mmCIF |
| `chain_module2_index` | `(n_chains_in_example,)` | int8 | chain index in Module 2 tensor |
| `chain_role` | `(n_chains_in_example,)` | int8 | 0 = helix chain, 1 = partner chain |

### Scalar metadata

| Field | Type | Notes |
|---|---|---|
| `pdb_id` | string | 4 chars, uppercase |
| `assembly_id` | int | |
| `example_id` | int | 0-indexed within entry |
| `helix_seqres_start` | int32 | SEQRES start of helix window |
| `helix_seqres_end` | int32 | SEQRES end of helix window |
| `helix_sequence` | string | one-letter AA codes for helix residues |
| `n_helix_contacts` | int | count of helix residues with is_interface_residue == True |
| `resolution` | float32 nullable | from Module 2 |
| `r_free` | float32 nullable | from Module 2 |
| `source_method` | string | from Module 2 |
| `sasa_used` | bool | True if SASA was successfully computed; False if fallback to distance-only |

## Module 3 Manifest

Single parquet file at `{output_dir}/module3_manifest.parquet`. Written via pandas by local entrypoint at pipeline end. One row per training example produced.

Columns:

- `example_id_full`: string — `{pdb_id}_{assembly_id}_{example_id}`
- `pdb_id`: string
- `assembly_id`: int8
- `example_id`: int32
- `helix_seqres_start`: int32
- `helix_seqres_end`: int32
- `helix_length`: int16 (= helix_seqres_end - helix_seqres_start + 1)
- `n_helix_residues`: int16
- `n_partner_residues`: int16
- `n_partner_chains`: int8
- `n_helix_contacts`: int16
- `n_partner_interface_residues`: int16 (count of partner residues with is_interface_residue == True)
- `n_residues_total`: int32
- `helix_sequence`: string
- `resolution`: float32 (nullable)
- `r_free`: float32 (nullable)
- `source_method`: string
- `sasa_used`: bool
- `path_example`: string — relative path to example `.npz`
- `pipeline_version`: string (git SHA)
- `config_hash`: string
- `processing_date`: timestamp

Manifest-level metadata (parquet metadata block):
- Pipeline version, full config YAML, dependency versions (numpy, pandas, freesasa, gemmi, scipy), Module 2 manifest path used, processing date range.

## Per-Entry Status Manifest

Separate parquet file at `{output_dir}/module3_entry_status.parquet`. One row per Module 2 entry processed by Module 3. Tracks which entries succeeded, which produced zero examples, which failed.

Columns:

- `pdb_id`: string
- `assembly_id`: int8
- `processing_status`: string — `"ok"`, `"no_examples"`, `"dropped"`, `"error"`
- `drop_reason`: string (nullable)
- `n_helix_segments`: int32 (null if error)
- `n_interacting_helices`: int32 (null if error)
- `n_windows_before_filter`: int32 (null if error)
- `n_examples_emitted`: int32 (null if error)
- `processing_date`: timestamp

## Canonical Drop Reasons

- `unparseable_module2_output` — Module 2 `.npz` failed to load.
- `no_helix_segments` — entry has no alpha-helix segments of sufficient length (informational; not an error).
- `no_interacting_helices` — entry has helix segments but none contact another chain (informational).
- `no_surviving_windows` — all windows failed the contact-count or completeness filter.
- `processing_error` — uncaught exception with traceback in global log.
- `batch_retry_exhausted` — Modal batch failed after all retries.

## Logging

Single consolidated global log at `{output_dir}/module3.log`. Standard Python logging.

Log content:
- Pipeline start and end.
- Per-batch dispatch and completion events.
- Errors with full tracebacks.
- Progress markers every 1000 successfully extracted examples.
- SASA computation failures (one log line per occurrence, reported by workers).

Explicitly NOT logged:
- Per-entry routine success messages.
- Per-example extraction messages.
- Windowing decisions.

No per-entry JSONL logs.

## Summary Report

Path: `{output_dir}/summary_report.md`. Written at pipeline end.

Contents:
- Total entries from Module 2 manifest.
- Entries successfully processed.
- Entries producing zero examples (broken down by reason: no helices, no interacting helices, no surviving windows).
- Entries with infrastructure errors.
- Total training examples produced.
- Distribution of `helix_length` (8 to ~22 with the merged-final-window edge case).
- Distribution of `n_helix_contacts` per example.
- Distribution of `n_partner_interface_residues` per example.
- Distribution of `n_partner_chains` per example.
- Count of examples with SASA computation failures (fallback to distance-only).
- Wall time and Modal compute cost summary.

## Recommended Python Stack

- **Tensor loading and manipulation:** `numpy`
- **SASA computation:** `freesasa` (Python bindings)
- **Spatial indexing for contacts:** `scipy.spatial.KDTree`
- **Manifest:** `pandas` with `pyarrow` for parquet I/O
- **Modal execution:** `modal`
- **Logging:** standard library `logging`

## Output Directory Layout

```
{output_dir}/
  module3_manifest.parquet            # one row per training example
  module3_entry_status.parquet        # one row per Module 2 entry processed
  summary_report.md
  module3.log
  config_used.yaml
  examples/
    ab/
      1abc_1_0.npz
      1abc_1_1.npz
      1abc_1_2.npz
      ...
  markers/
    ab/
      1abc_1.marker                   # empty file; presence = entry processed
      ...
  test_run/                           # present only if test mode was used
    module3_manifest.parquet
    module3_entry_status.parquet
    summary_report.md
    test_summary.md
    module3.log
    examples/
      ab/
        1abc_1_0.npz
        ...
    markers/
      ab/
        1abc_1.marker
        ...
```

Subdirectories named by middle two characters of PDB ID.

## Determinism Invariants

The following must be reproducible across runs with the same `random_seed`:

- Within an entry, helix iteration order is: chains in Module 2 chain_index order (alphabetical by label_asym_id); helices within a chain in SEQRES start position order. `helix_index` is the sequential global index within the entry.
- Window-length sampling for each helix uses RNG seeded by `random_seed + hash((pdb_id, assembly_id, helix_index))`.
- Within a partner region, residues are ordered by chain_slot ascending, then by SEQRES position ascending.
- `example_id` is assigned 0, 1, 2, ... in the order examples are produced (helix-by-helix, window-by-window).

Document these invariants in the code. Verify in test mode.

## Implementation Notes

- Each Modal worker receives Module 2 `.npz` bytes and metadata directly in the batch payload. No file system or cloud access from workers.
- Extracted examples are returned as serialized bytes and written to local disk by the entrypoint.
- File I/O is atomic: write to `.tmp`, rename on completion.
- Re-running is idempotent via marker files. Check for marker before processing; skip if present unless `--force`.
- Use AF2 canonical atom14 ordering throughout. Backbone atoms are at slots 0 (N), 1 (CA), 2 (C), 4 (O). Reference Module 2's `constants.npz` for slot definitions.
- For SASA, map each atom (atom_mask == 1) to a freesasa input using element-based default radii. Tag each freesasa input atom with (chain_slot, residue_index_in_example) for per-residue aggregation.
- The helix's own chain is NEVER a partner chain, even if other residues on the helix's chain contact partner chains (homomeric or self-contacting cases). This keeps chain_role assignment unambiguous: exactly one chain has role 0 (the helix chain), all others contacted by the helix have role 1 (partner chains). Intra-chain contacts from the helix are not represented in the example.
- Chain `chain_label` strings should be retrieved from Module 2's output if available. If not directly available, document this as a known limitation for the first version; can be addressed later by amending Module 2 or by re-loading source mmCIFs.
- Catch all worker-side errors per entry. Record as `processing_error` with traceback; continue with remaining entries in the batch.
- Run `test_mode: true` first on ~100 entries and verify all `test_summary.md` assertions pass before launching the production run.
- Local entrypoint must stay running for the full duration. Use `tmux` or `screen`.