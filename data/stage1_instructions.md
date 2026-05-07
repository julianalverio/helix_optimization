# Module 1 Implementation Plan (for AI Agent Execution)

## Goal

Build Module 1 of a PDB protein-protein interaction pipeline. This module produces a filtered local archive of mmCIF structure files plus a parquet manifest describing what was downloaded, filtered, and why. This manifest will be the input to Module 2.

## Design Principles

- **Idempotent.** Every step can be re-run without redoing work already done. Re-running the full pipeline on the same snapshot should produce the same outputs.
- **Reproducible.** Every filter decision, threshold, software version, and data snapshot date is recorded in the manifest.
- **Permissive at the filter stage, rich at the manifest stage.** Capture many attributes as *tags* on the manifest; apply only the essential filters here. Tighten thresholds later by querying the manifest, not by re-downloading.
- **Metadata before bytes.** Build the candidate list from the RCSB API first; only download mmCIF files for entries that pass preliminary filters.
- **Two-manifest output.** One manifest records every entry considered with per-filter pass/fail flags (for auditing). A second records the subset that passed all filters and was successfully downloaded (for Module 2 consumption).

## Configuration

All thresholds below must be exposed as configuration (YAML or Python dataclass). The values given are **defaults**. The agent must record the effective config in the manifest metadata.

| Parameter | Default | Notes |
|---|---|---|
| `methods_allowed` | `["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]` | NMR excluded |
| `resolution_max_xray` | 3.5 Å | |
| `resolution_max_em` | 3.5 Å | |
| `r_free_max_xray` | 0.30 | Not applied to EM |
| `r_free_missing_action` | `"keep_and_tag"` | Alternative: `"drop"` |
| `status_allowed` | `["REL"]` | Excludes OBS, WDRN, HOLD, HPUB |
| `min_protein_chain_length` | 20 | Residues (SEQRES) |
| `min_instantiated_polymer_chains` | 2 | See Phase A Step 6 |
| `require_protein_chain` | `true` | |
| `min_observed_fraction` | 0.5 | Fraction of SEQRES residues with coordinates |
| `large_assembly_chain_threshold` | 20 | Trigger special handling above this |
| `hard_cap_total_residues` | None | Default off; set to e.g. 50000 to exclude ribosomes outright |
| `deposition_date_min` / `_max` | None | Set when building temporal benchmarks |
| `release_date_min` / `_max` | None | |
| `snapshot_date` | Today | Record exact UTC date of run |

## Phase A — Candidate List from RCSB Metadata

Do all of this **before any file download**. Use the RCSB Search API v2 and Data API.

### Step 1. Fetch the full current PDB entry list
- Query RCSB Search API for all released entries.
- Record the snapshot date (UTC) in manifest metadata.

### Step 2. Fetch and apply the obsolete list
- Download `https://files.wwpdb.org/pub/pdb/data/status/obsolete.dat`.
- Parse into a table: `obsoleted_id`, `replacement_id`, `obsoletion_date`.
- Store this file versioned alongside the manifest (the obsolete list itself changes over time).
- Drop entries with status `OBS`, `WDRN`, `HOLD`, or `HPUB` from the candidate list.
- For any entries in your candidate list that were obsoleted with a replacement: follow the chain to the current replacement and replace; drop if the replacement is also obsolete.
- Store the `obsolete_id → replacement_id` mapping as a sidecar artifact.

### Step 3. Filter by experimental method
- Keep only entries where `exptl.method` is in `methods_allowed`.
- **NMR is excluded entirely in this pipeline.** Do not attempt to handle NMR ensembles.
- Multi-method entries: keep if *any* of the methods is in `methods_allowed` and none are disallowed; tag with `multi_method=True`.
- Drop neutron diffraction, fiber diffraction, solid-state NMR, solution scattering, theoretical models, and integrative/hybrid entries (PDB-IHM).

### Step 4. Filter by resolution
- X-ray: `refine.ls_d_res_high ≤ resolution_max_xray`.
- Cryo-EM: reported reconstruction resolution ≤ `resolution_max_em`. Field is in `em_3d_reconstruction.resolution` or the summary resolution.
- Drop entries with missing resolution when resolution is expected for the method.

### Step 5. Filter by R-free (X-ray only)
- X-ray: `refine.ls_R_factor_R_free ≤ r_free_max_xray`.
- Missing R-free on X-ray: set `rfree_missing=True` and apply `r_free_missing_action` (default: keep and tag).
- Skip this step for cryo-EM entries — R-free is not defined.

### Step 6. Filter by instantiated polymer chains
**This replaces a naive "≥2 polymer entities" filter.** Homo-oligomers have only one unique entity but multiple chain instances and must not be dropped.

- Use the field `rcsb_assembly_info.polymer_entity_instance_count` from the RCSB Data API (attached to each assembly). This counts **chain instances** after assembly expansion, not unique entities.
- For each entry, pick the primary biological assembly (assembly_id = 1 by default, or the first author-defined assembly).
- Require `polymer_entity_instance_count ≥ min_instantiated_polymer_chains` (default 2).
- Require at least one polymer entity of type `polypeptide(L)` (i.e., protein; `polypeptide(D)` also acceptable but rare).
- Tag DNA/RNA presence rather than filtering.

Note for the agent: do **not** use `rcsb_entry_info.polymer_entity_count` as the filter — that counts unique entities, which would incorrectly drop homo-oligomers. Use `polymer_entity_instance_count` from the assembly-level info.

### Step 7. Filter by protein chain length
- At least one protein polymer entity with SEQRES length ≥ `min_protein_chain_length` (default 20).
- Tag `has_short_peptide=True` if any polymer is shorter — useful for antibody-peptide work later.

### Step 8. Optional date filters
- Apply `deposition_date_min/max` and `release_date_min/max` if configured.
- Default: no filtering; dates recorded as manifest columns for later temporal splits.

### Step 9. Optional size cap
- If `hard_cap_total_residues` is set, drop entries whose assembly total residue count exceeds it.
- Default: off. Large structures are handled specially in Phase B (see Step 11.5) rather than dropped.

### Step 10. Write the candidate manifest
- Write to `candidates.parquet`.
- **Use parquet, not CSV.** Parquet preserves types, compresses well, and supports columnar reads. Keep a small CSV sidecar if human-readable diffs are wanted, but parquet is the source of truth.
- One row per PDB ID. Include per-filter pass/fail boolean columns (`passed_method_filter`, `passed_resolution_filter`, etc.) and an aggregate `passed_all_filters` column.
- Include attributes captured regardless of filtering: method, resolution, R-free, status, instantiated chain count, entity composition, DNA/RNA presence, deposition/release dates, R-free missingness tag, title, and any relevant flags.
- Add manifest-level metadata: snapshot date, config hash, pipeline git SHA, obsolete-list version.

## Phase B — Download

### Step 11. Rsync the mmCIF files for candidate IDs
Use rsync against the RCSB mirror. Rsync is idempotent by design: re-running skips files already present and unchanged.

**Recommended command:**
```bash
rsync -rlpt -v -z --partial --partial-dir=.rsync-partial \
      --port=33444 \
      --files-from=candidate_paths.txt \
      rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
      /local/path/mmCIF/
```

Flag notes for the agent:
- `-r -l -p -t`: recursive, preserve symlinks/perms/mtimes. **`-t` is critical for idempotency** — rsync's quick-check relies on mtime + size to skip unchanged files.
- `-z`: compress in transit.
- `--partial --partial-dir`: resume interrupted transfers.
- `--port=33444`: RCSB's rsync port (not the default 873).
- `--files-from`: path to a plain-text file listing relative paths like `ab/1abc.cif.gz`, built from the Phase A candidate list. This limits the download to only candidates.
- **Do not use `--delete`** on the first build. Enable it only for a scheduled sync where intentionally removing obsoleted local files is desired.
- Alternate mirrors if RCSB is slow: PDBe (`rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/`) or PDBj.

Store files gzipped in the two-letter subdirectory layout (`ab/1abc.cif.gz`) that the mirror already uses. Do not flatten.

### Step 11.5. Special handling for very large assemblies
For entries flagged as large (assembly chain count `≥ large_assembly_chain_threshold`, default 20), record a processing plan in the manifest so Module 2 can avoid expanding the full assembly. **Do not skip these entries.** For your use case (interfaces only), unique interfaces are what matter, not copies; a viral capsid has only a handful of unique interface types regardless of having 60 or 180 chains.

For each large entry, during Phase A (metadata stage):
1. Enumerate assembly operators from `pdbx_struct_assembly_gen.oper_expression` via the Data API.
2. Compute the set of **unique pairwise entity interfaces** in the assembly. Two interfaces are equivalent if they involve the same ordered entity pair under symmetry-equivalent operators. Use RCSB's pre-computed interface data if available (`rcsb_interface_partner` / interface summary endpoints); otherwise compute from operator geometry.
3. For each unique interface type, record a minimal operator pair that instantiates it — the two operators needed to generate the two partner chains.
4. Store this as a structured column `unique_interface_plan` in the candidates manifest: a list of records, each containing `entity_id_1`, `entity_id_2`, `operator_1`, `operator_2`, and a stable interface-type hash.
5. Tag the entry with `large_assembly=True` and `n_unique_interfaces`.

Module 2 will consume this plan and expand only the chains needed for each unique interface, rather than building the full assembly.

For entries below the threshold, `unique_interface_plan` is null and Module 2 builds the full assembly normally.

Helpful APIs and tools:
- RCSB Data API: `rcsb_assembly_info` and interface endpoints.
- `gemmi`: can parse operator expressions and apply them selectively without materializing the full assembly.
- PDBePISA pre-computed interface tables are an alternative source of ground truth for unique interfaces and can be cross-checked against.

Record which source was used for interface enumeration in the manifest metadata.

### Step 12. Download auxiliary data
Alongside the mmCIF files, fetch and version:
- `obsolete.dat` (already in Step 2; retain).
- SIFTS mapping files: at minimum `pdb_chain_uniprot.tsv.gz` and `pdb_chain_taxonomy.tsv.gz` from `ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/`.
- RCSB validation report XMLs for each candidate entry (one per entry): useful for quality metrics beyond R-free. URL pattern: `https://files.rcsb.org/pub/pdb/validation_reports/<mid2>/<pdb_id>/<pdb_id>_validation.xml.gz`.
- Record the fetch date of each auxiliary dataset.

## Phase C — Post-Download Verification and Finalization

### Step 13. Verify file integrity
- For each candidate, confirm the mmCIF file exists and is parseable.
- Compute SHA256 of each downloaded file; store in manifest.
- Drop and log any corrupt or unparseable files with a `drop_reason`.

### Step 14. Extract per-entry and per-chain details from the file
Open each mmCIF and record:
- Confirm method, resolution, R-free match API values; flag mismatches.
- Per-chain: SEQRES sequence, observed sequence, `observed_fraction = len(observed_residues) / len(seqres_residues)`, chain type (protein / DNA / RNA / other), modified residue counts, CA-only flag.
- Per-entry: atom count, assembly count, presence of author-defined assembly, source organism.

### Step 15. Filter out mostly-unresolved structures
Using the per-chain `observed_fraction` computed in Step 14:
- Drop entries where **no protein chain** has `observed_fraction ≥ min_observed_fraction` (default 0.5).
- Rationale: entries with <50% of SEQRES residues actually modeled have so many gaps that interface analysis becomes unreliable. Loops — the target of this dataset — are exactly what disappears when resolution is poor or density is weak.
- Record per-chain `observed_fraction` in the final manifest regardless, so Module 3 can apply its own stricter thresholds if desired.

### Step 16. Extract validation-report quality metrics (optional but recommended)
From the validation report XML, record:
- `clashscore`
- `percent_ramachandran_outliers`
- `percent_rotamer_outliers`
- `percent_RSRZ_outliers` (X-ray only)
- `Q_score` (cryo-EM only, if present)

These are not used as filters by default but are stored for downstream use. They are better quality indicators than R-free alone.

### Step 17. Compute and store per-chain canonical sequences
- One-letter canonical sequences for every protein chain, keyed by `(pdb_id, chain_id)`.
- Store in a separate parquet file `chain_sequences.parquet` (long format: one row per chain).
- This feeds downstream sequence-identity clustering (MMseqs2) without re-parsing mmCIF.

### Step 18. Write the final Module 1 manifest
Write to `module1_manifest.parquet`. One row per successfully downloaded, verified, non-filtered-out entry.

Required columns:
- **Identity:** `pdb_id`, `file_path`, `sha256`
- **Experimental:** `method`, `resolution`, `r_free`, `rfree_missing`, `multi_method`
- **Quality (optional):** `clashscore`, `ramachandran_outlier_pct`, `rotamer_outlier_pct`, `rsrz_outlier_pct`, `q_score`
- **Composition:** `n_polymer_entities`, `n_instantiated_polymer_chains`, `n_protein_chains`, `has_dna`, `has_rna`, `has_ligands`, `has_modified_residues`, `has_short_peptide`
- **Dates:** `deposition_date`, `release_date`
- **Assembly:** `primary_assembly_id`, `large_assembly`, `n_unique_interfaces`, `unique_interface_plan` (nullable struct/list column)
- **Observed coverage:** `max_protein_observed_fraction`, `min_protein_observed_fraction`
- **Status:** `status`, `obsoleted_to` (nullable; populated if this entry was reached via obsolete redirection)
- **Provenance:** `pipeline_version`, `config_hash`, `snapshot_date`

Manifest-level metadata (written to parquet metadata block or sidecar JSON):
- snapshot date, config YAML, git SHA of pipeline code, versions of key dependencies (gemmi, biotite, etc.), obsolete.dat version, SIFTS version.

### Step 19. Write the audit manifest
Write `candidates_audit.parquet` — one row per PDB ID **considered** (before filtering), with per-filter pass/fail columns. This is what you query when someone asks "why was 1ABC not in the dataset?"

### Step 20. Emit a summary report
Plain-text or markdown report with:
- Total PDB entries on snapshot date.
- Entries dropped at each filter step (counts and percentages).
- Final dataset size.
- Distribution of method, resolution, R-free, chain count, large_assembly flag.
- Any entries flagged for manual review (R-free missing, resolution mismatch between API and file, failed parses, etc.).

## Integration Tests

Before declaring the pipeline done, verify against these canonical cases. Commit them as automated tests.

| PDB ID | Expected outcome | Why |
|---|---|---|
| `1HH6` | pass | Antibody-antigen, standard X-ray |
| `1BRS` | pass | Barnase-barstar, classic heteromer |
| `2REB` | evaluate | Homodimer — must pass despite single entity |
| `6VXX` | pass with `large_assembly=True` | SARS-CoV-2 spike trimer |
| `1AON` | pass with `large_assembly=True` | GroEL-GroES, 14+7 chains |
| `3J3Q` | pass with `large_assembly=True` | HIV capsid — verify unique-interface planning |
| any obsoleted ID | redirected to replacement or dropped | Verify obsolete logic |
| `1A3N` | pass | Hemoglobin, 2 entities × 2 chains each |
| any NMR-only ID | dropped | Verify NMR exclusion |
| any structure with resolution >5 Å | dropped | Verify resolution filter |

## Recommended Python Stack

- **RCSB API calls:** plain `requests` or the `rcsb-api` package.
- **mmCIF metadata parsing (Phase A/C):** `gemmi` (fastest mmCIF parser; first-class mmCIF support; excellent assembly handling; handles gzip directly). Use `pdbecif` as an alternative for category-only metadata reads.
- **Structure manipulation and assembly expansion:** `gemmi` for heavy lifting.
- **Manifests:** `pandas` or `polars` with `pyarrow` for parquet I/O.
- **Do not use `Bio.PDB`** as the primary parser — it is slow and clunky at PDB scale.