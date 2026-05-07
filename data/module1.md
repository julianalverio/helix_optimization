# Module 1 — How It Works

Module 1 turns the live PDB into a filtered local archive of mmCIF files plus a parquet manifest that Module 2 consumes. This document walks the implementation end-to-end and describes the test suite that guards the risky parts.

## What Module 1 produces

After a successful run, `data/manifests/` contains:

| File | Rows | Purpose |
|---|---|---|
| `candidates.parquet` | one per considered PDB ID | Phase A output — per-filter pass/fail booleans and `phase_a_drop_reason` for every input ID, including stub rows for obsolete-chain dead-ends, metadata-missing IDs, and interface-fetch failures |
| `verify_results.parquet` | one per file attempted | Phase C output — `parse_ok`, `parse_error`, SHA256, method/resolution/r_free read from the file, observed-fraction aggregates |
| `chain_sequences.parquet` | one per `(pdb_id, chain_id)` | Long-form SEQRES + observed chain table with canonical one-letter sequences |
| `module1_manifest.parquet` | one per record reaching the final dataset | The clean artifact for Module 2 — entries with `drop_reason IS NULL` |
| `candidates_audit.parquet` | one per considered PDB ID | The union of Phase A + Phase B + Phase C with a single `drop_reason` column. This is the "why isn't X in the dataset?" lookup |
| `module1_manifest.meta.json` | — | snapshot date, pipeline version, config hash, full config dump |
| `report.md` | — | Human-readable summary: total considered, drops by reason, per-filter breakdown, final dataset distributions |

`data/pdb/` holds the rsync mirror in the standard two-letter layout (`pdb/ab/1abc.cif.gz`). `data/aux/` holds `obsolete-<date>.dat`.

## The pipeline

The CLI (`twistr/cli.py`) exposes subcommands `fetch-candidates`, `download`, `verify`, `report`, and `run-all`. Each subcommand reads and writes parquet artifacts, so they compose cleanly and can be run independently during development.

### Phase A — Metadata and filtering (`twistr/candidates.py`)

Phase A never touches mmCIF bytes. It builds the candidate manifest purely from RCSB metadata.

1. **Input IDs.**
   - Dev subset (default): `DEV_IDS` (9 canonical IDs) + 100 random IDs from the full RCSB Search API list. See `select_input_ids`.
   - Full scale (`--full-scale`): every released experimental entry from the Search API.
   - Alternative: `run_phase_a_on_ids(cfg, root, input_ids, snapshot)` takes an arbitrary list and skips the Search API call. Used by tests and ad-hoc runs.

2. **Obsolete resolution** (`resolve_candidate_ids`). For each input ID, follow the `obsolete.dat` chain. Three outcomes:
   - Not obsolete → keep as-is.
   - Obsolete with a reachable replacement → swap to the replacement, record the original in `obsoleted_from`.
   - Obsolete with dead-end chain (no replacement, or cycle) → stub row with `phase_a_drop_reason = "obsolete_no_replacement"` and a WARNING logged.

3. **Metadata batch fetch** (`rcsb.fetch_metadata`). GraphQL at `data.rcsb.org/graphql`, 100 IDs per batch. One shared `requests.Session` with retry/backoff. Each batch response is cached as NDJSON at `data/manifests/.cache/<config_hash>/phase_a/<batch_sha>.jsonl` — re-runs skip cached batches. On batch error, IDs are added to `failed_metadata_ids` (stub row with `drop_reason = "metadata_error"`), WARNING logged, pipeline continues.

4. **Build candidate rows** (`build_candidate_row`). One row per successfully-fetched entry. Computes the nine per-filter booleans:
   - `passed_status_filter` — `rcsb_accession_info.status_code` is in `status_allowed` (default `["REL"]`).
   - `passed_method_filter` — method is in `methods_allowed`; any disallowed method (NMR variants, theoretical, integrative, etc.) auto-fails.
   - `passed_resolution_filter` — X-ray uses `refine.ls_d_res_high ≤ 3.5`, EM uses `em_3d_reconstruction.resolution ≤ 3.5`, falls back to `rcsb_entry_info.resolution_combined`. EM entries without refine must still get a resolution from the EM branch of the cascade.
   - `passed_rfree_filter` — X-ray only. `r_free ≤ 0.30`, or missing-and-tagged if `r_free_missing_action = "keep_and_tag"`. Non-X-ray entries auto-pass.
   - `passed_chains_filter` — `rcsb_assembly_info.polymer_entity_instance_count ≥ 2`. **Not** `polymer_entity_count`; the latter drops every homodimer.
   - `passed_protein_chain_filter` / `passed_protein_length_filter` — at least one protein entity, max SEQRES length ≥ 20.
   - `passed_date_filter`, `passed_size_cap_filter` — optional; off by default.

   `phase_a_drop_reason` is `filter:<a>,<b>,...` (comma-joined failing filter names in declaration order) or `None`.

5. **Missing-metadata detection.** Any resolved ID that is not in `entries_by_id` and not in `failed_metadata_ids` gets a stub row with `phase_a_drop_reason = "metadata_missing"` and a WARNING.

6. **Interface planning** (`twistr/interfaces.py`). Only runs for rows with `large_assembly = True` (chain count ≥ 20). `fetch_unique_interfaces`:
   - Pulls the assembly summary from `/rest/v1/core/assembly/{id}/{asm}` — a failure here raises `InterfaceFetchError`, the record keeps its row but gets `phase_a_drop_reason = "interface_fetch_error"` and is excluded from the final manifest.
   - For each listed interface, fetches `/rest/v1/core/interface/{id}/{asm}/{iface}`. A 5xx on one interface drops that interface and continues (WARNING per failure).
   - Deduplicates via `_dedupe_key` — `(sorted(entity_pair), area_bucket_of_50_Å², residue_bucket_of_3)`. The RCSB REST API does not expose an `interface_cluster_id` field, so the heuristic errs toward over-reporting rather than collapsing distinct contacts.

Phase A writes `candidates.parquet` atomically (tmp-then-rename).

### Phase B — Rsync (`twistr/download.py`)

1. Filter `candidates.parquet` to rows where `passed_all_filters=True`.
2. Write `data/manifests/candidate_paths.txt` — one relative path per row, `{ab}/{pdb_id}.cif.gz`.
3. Invoke rsync via `subprocess.run` with the spec's flags (`-rlpt -v -z --partial --port=33444 --files-from=…`). `-t` is kept to preserve mtime, which is how rsync's quick-check skips already-current files.
4. On primary failure, log WARNING and retry against the PDBe fallback mirror. Both failing → ERROR, pipeline continues. Missing files are caught by Phase C/manifest as `download_missing`, not by raising.

Rsync is self-idempotent (mtime + size), so re-runs are cheap.

### Phase C — Verify (`twistr/verify.py`)

For every `passed_all_filters=True` candidate whose file is on disk:

1. `ProcessPoolExecutor` fans the files out across all CPU cores.
2. In each worker, `parse_structure` does SHA256, `gemmi.read_structure` (handles `.cif.gz` natively), `structure.setup_entities()`, then per-chain extraction: SEQRES length, observed residue count, `observed_fraction = observed / seqres`, chain type from `entity.polymer_type`, CA-only detection, canonical one-letter sequence (with `ALA;SER` / `ALA,SER` heterogeneity markers stripped).
3. The entire body is wrapped in `try/except` — any exception is captured in `parse_error` and returned as a result with `parse_ok=False`. Worker-level crashes are caught at `future.result()` and produce a stub `{parse_ok: False, parse_error: "worker_exception:<Type>"}` row so the audit covers every attempted file.

Outputs: `verify_results.parquet` and `chain_sequences.parquet`.

### Final manifest (`twistr/manifest.py`)

`build_final_manifest` merges candidates + verify_results and computes `drop_reason` per row using this precedence:

```
phase_a_drop_reason   (obsolete_no_replacement | metadata_missing | metadata_error |
                       filter:<a>,<b>,… | interface_fetch_error)
      ↓
download_missing      (passed_all_filters=True but file not on disk)
      ↓
parse_error:<Type>    (gemmi / worker crash)
      ↓
filter:observed_fraction  (max_protein_observed_fraction < 0.5)
      ↓
None                  → row survives into module1_manifest.parquet
```

A subtle pitfall: `phase_a_drop_reason` becomes a `float(NaN)` (not `None`) after the pandas merge for passing rows, and `bool(NaN)` is truthy. The implementation uses `isinstance(phase_a, str)` to guard against this — regression-tested in `test_drop_reason.py`.

Writes `candidates_audit.parquet` (all rows + `drop_reason`), `module1_manifest.parquet` (just the survivors), and a JSON sidecar with config hash + snapshot date.

### Report (`twistr/report.py`)

`report.md`: counts by `drop_reason`, per-filter breakdown (the raw booleans), final dataset distributions (methods, resolution, chain count, large-assembly count), and a manual-review list of entries that passed with caveats (r_free missing, parse error, observed_fraction low).

## Configuration and provenance

- `config.yaml` at the repo root is loaded into a frozen `Config` dataclass (`twistr/config.py`).
- `config_hash(cfg)` is the first 16 hex chars of SHA256 over the filter-affecting fields only (not `snapshot_date`). This hash appears in manifest metadata and in cache-dir names, so changing a threshold cleanly invalidates caches while unchanged reruns hit them.
- Snapshot date is captured once in `cli.main()` via `snapshot_now()` and threaded through every phase — no repeated `datetime.now()` calls.
- `module1_manifest.meta.json` records snapshot date, pipeline version, config hash, and the full config dict.

## Logging

Configured in `cli.main()` via `logging.basicConfig`. Default INFO; `--verbose` bumps to DEBUG. Each module uses `logger = logging.getLogger(__name__)`.

- **INFO** — phase boundaries and per-phase counts.
- **WARNING** — per-record non-fatal failures: GraphQL batch errors, missing metadata, interface fetch failures, rsync exit-code issues, parse errors, worker crashes. Filter-triggered drops are **not** logged — they're normal, and would flood the log.
- **ERROR** — phase-level aborts (both rsync mirrors unreachable, etc.).

## Running it

```bash
.venv/bin/twistr run-all                       # dev subset (default)
.venv/bin/twistr run-all --full-scale          # full PDB
.venv/bin/twistr fetch-candidates              # Phase A only
.venv/bin/twistr verify --workers 8            # Phase C only
```

Re-running is near-instant when nothing changed (GraphQL cache hits, rsync mtime-skips).

---

# The Test Suite

Twelve offline tests. Zero mocking — fixtures are either synthetic data shaped like GraphQL responses, or real captured API responses committed under `tests/fixtures/`. The pre-existing `--run-network` integration tests (under `tests/test_*.py`) still exist and hit live RCSB; they're skipped by default.

```
tests/
├── conftest.py                    # --run-network flag for the network suite
├── fixtures/
│   ├── mmCIF/
│   │   ├── 1BRS.cif.gz            # for gemmi parse tests
│   │   └── 1A3N.cif.gz            # committed but currently unused by the unit suite
│   ├── rcsb_graphql/
│   │   └── bd18030be79b81e9.jsonl # GraphQL responses for the 6 Phase A test IDs
│   └── rcsb_interfaces/
│       ├── 1AON_1_assembly.json
│       └── 1AON_1_{1..42}.json    # the 42 raw interface responses
├── unit/
│   ├── test_obsolete.py
│   ├── test_build_candidate_row.py
│   └── test_drop_reason.py
└── integration/
    ├── test_verify_parse.py
    ├── test_interfaces_dedupe.py
    └── test_phase_a_offline.py
```

## Why these tests

The 12 tests target the areas of Module 1 where a bug would silently corrupt data for Module 2 — not where it would crash. Rsync correctness, SHA256, parquet I/O, YAML loading, and the filter comparators themselves are all library-level or trivial and omitted.

### Unit tests (pure, synthetic data)

**`tests/unit/test_obsolete.py` — 2 tests**
- `test_parse_fixed_columns` — verifies `parse_obsolete` handles the fixed-column `obsolete.dat` format, including lines with multiple replacement IDs and lines with no replacement. A naive whitespace-split parser would attach the next line's content to an empty-replacement record.
- `test_resolve_redirect_chain_and_cycle` — verifies `resolve_redirect` follows multi-hop chains to the terminal replacement, returns `None` for dead-ends, passes through non-obsolete IDs unchanged, and breaks cycles rather than looping forever.

**`tests/unit/test_build_candidate_row.py` — 4 tests** (synthetic GraphQL entry dicts assembled by `_make_entry`)
- `test_homomer_passes_chain_filter` — 1 unique entity, 2 chain instances. Traps the `polymer_entity_count` vs `polymer_entity_instance_count` bug (every homodimer silently dropped).
- `test_em_resolution_cascade` — EM entry with `refine=None` and `em_3d_reconstruction.resolution=2.8` must resolve to 2.8 and pass both the resolution and r_free filters. Traps the "EM entries get resolution=None because we only looked at refine" bug.
- `test_xray_rfree_missing_keeps_and_tags` — X-ray entry with `r_free=None` must pass under the default `keep_and_tag` policy with `rfree_missing=True`. Traps a reversed comparator or a dropped default case.
- `test_multi_filter_fail_produces_ordered_drop_reason` — an entry failing method, resolution, and chains must produce `phase_a_drop_reason = "filter:method,resolution,chains"` in that exact order. Traps dict-iteration-order regressions and non-deterministic `drop_reason` strings.

**`tests/unit/test_drop_reason.py` — 2 tests** (synthetic `pd.Series` rows)
- `test_precedence_order` — feeds `_compute_drop_reason` rows that match each branch of the precedence chain (phase_a wins over parse error; download_missing wins over parse error; parse_error extracts just the exception type; clean row returns `None`).
- `test_nan_phase_a_drop_reason_does_not_trigger_phase_a_branch` — regression guard against a real bug hit in development: `bool(float('nan'))` is `True`, so a truthy check on a pandas-merged `phase_a_drop_reason` column returns NaN for rows that actually passed. The implementation uses `isinstance(str)`; this test breaks if that's ever reverted.

### Integration tests (real fixtures, no network)

**`tests/integration/test_verify_parse.py` — 2 tests**
- `test_parse_1brs_observed_fraction` — full gemmi parse of `1BRS.cif.gz`. Asserts `parse_ok`, method, chain count, per-chain `observed_fraction` bounds, canonical sequence non-empty, and `len(canonical_sequence) == seqres_length`. Covers chain iteration, entity lookup, SEQRES `full_sequence` extraction, and the heterogeneity-stripped one-letter conversion — all brittle surfaces in custom gemmi code.
- `test_parse_corrupt_file_returns_parse_error` — writes a gzipped `"not a real mmcif"` to a temp file. Must return `parse_ok=False` with `parse_error` populated, **without raising**. Guards the error-isolation contract that `build_final_manifest` depends on.

**`tests/integration/test_interfaces_dedupe.py` — 1 test**
- `test_1aon_unique_interfaces_dedupe` — loads the committed 42 interface + 1 assembly RCSB responses for 1AON; calls `interfaces.dedupe_from_responses` (the pure-logic entry point) and asserts:
  - plan size between 3 and 20 (raw 42 must be deduped, but not below the three real entity-pair types);
  - plan size strictly less than raw count (otherwise the dedupe check is no-op);
  - all three entity pairs `(1,1)`, `(1,2)`, `(2,2)` are represented (GroEL-GroEL, GroEL-GroES, GroES-GroES — a dedupe key too coarse would collapse one of these);
  - no duplicate dedupe keys in the final plan (the `seen` set must be functioning).

**`tests/integration/test_phase_a_offline.py` — 1 test**
- `test_phase_a_offline_end_to_end` — copies the committed GraphQL JSONL into a fresh tmp data root at the correct cache path, calls `run_phase_a_on_ids` with `obsolete_map={}` (so no network), reads `candidates.parquet`, and asserts five specific integration-level behaviors:
  - **1TIM** — passes with 1 entity and 2 chain instances (homodimer regression case).
  - **1G03** — NMR, no resolution, 1 chain → `drop_reason == "filter:method,resolution,chains"` exactly.
  - **1HH6** — `r_free == 0.338` → `drop_reason == "filter:rfree"` (traps a reversed comparator).
  - **6VXX** — EM, resolution picked up from `em_3d_reconstruction` (not None), passes.
  - **1BRS** — X-ray with `r_free=None`, `rfree_missing=True`, passes under `keep_and_tag`.

  This test exercises the full Phase A integration path — resolve, metadata, row-building, filter, drop_reason, stub rows — with zero external dependencies.

## Running the suite

```bash
.venv/bin/python -m pytest tests/unit tests/integration          # 12 tests, offline, ~seconds
.venv/bin/python -m pytest tests/ --run-network                  # 12 offline + 22 live RCSB
.venv/bin/python -m pytest tests/unit tests/integration --collect-only   # discover without running
```

## What the suite intentionally does not cover

- **rsync mechanics.** Library code; failure modes are visible via `download_missing` rows in the audit.
- **RCSB response shape.** Not our bug to find.
- **SHA256, parquet I/O, YAML parsing.** Stdlib / pyarrow / PyYAML.
- **The filter threshold comparators themselves.** `resolution ≤ 3.5` doesn't need a test; any regression is visible in the report.
- **Full Phase A including `fetch_all_released_ids`.** The `--run-network` integration suite covers this; re-testing it offline adds nothing.

The pre-existing `tests/test_*.py` network suite covers the live-API integration cases (including bogus-ID handling, real obsolete.dat fetch, multi-ID GraphQL round-trip). Together with the 12 offline tests, Module 1 has pragmatic coverage of the surfaces where it can silently go wrong.
