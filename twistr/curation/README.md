# `twistr.curation` — PDB ingestion with a multi-filter audit trail

First stage of the pipeline. Fetches PDB entry metadata from RCSB,
selects candidates passing a chain of structural and biological filters,
downloads the corresponding mmCIF files, validates them structurally,
and emits a final manifest tagged with `PIPELINE_VERSION` and a
content-hashed config signature. Output feeds `tensors/`.

## Phase structure

Curation runs in three phases, individually invokable via CLI:

1. **Phase A — `fetch-candidates`.** Query RCSB GraphQL + REST endpoints
   for assembly / interface metadata. Apply nine filter gates (method,
   resolution ≤ 3.5 Å X-ray / EM, R-free, chain topology, polymer
   entity count, protein length window, deposition date bounds, total
   size cap, observation fraction). Emit a candidate parquet **with
   per-entry drop-reason tagging** — every excluded entry carries the
   first gate it failed, so a downstream "why did this PDB not make it
   in?" question is one parquet read away.
2. **Phase B — `download`.** Chunked-rsync over RCSB mirrors with
   configurable primary / fallback hosts and a 300 s per-chunk timeout.
   Handles obsolete entries through RCSB's PDB-status API and resolves
   superseded entries through their successor IDs.
3. **Phase C — `verify`.** Parse every downloaded mmCIF with gemmi.
   Classify polymer chains by `gemmi.PolymerType` (protein vs DNA /
   RNA), count chains by type, validate per-residue observation
   completeness, and emit SHA256 hashes for downstream integrity
   checks.

The `run-all` command chains all three phases and then writes the final
manifest + audit report.

## Selected technical details

- **GraphQL + REST hybrid query.** RCSB's GraphQL endpoint is the
  efficient way to pull structured metadata in bulk; some interface
  fields are REST-only. The candidate pipeline uses GraphQL for primary
  selection and REST for the supplementary fields, with exponential
  backoff at both layers.
- **Drop-reason tagging.** Each filter gate stamps an enum identifier
  onto failing entries before the chain continues. The final audit
  report aggregates by drop-reason — useful for tuning thresholds and
  diagnosing dataset shifts when RCSB's metadata schema changes.
- **gemmi-based polymer classification.** Counting protein chains is
  not as trivial as it sounds: many depositions contain DNA / RNA /
  peptide / ligand entities that the candidate metadata doesn't cleanly
  distinguish. gemmi's `PolymerType` resolution applied at the chain
  level gives unambiguous counts.
- **Content-hashed config signature.** The final manifest carries
  `(PIPELINE_VERSION, sha256(curation_config))` so a downstream
  `tensors/` run can refuse to consume a manifest produced by a
  configuration that doesn't match its own.
- **Obsolete-entry handling.** RCSB's `obsolete` API returns the
  successor entry for superseded depositions; the candidate pipeline
  follows the chain rather than silently dropping the obsoleted ID,
  preserving sequence coverage when authors re-deposit at higher
  resolution.

## File index

| File | Role |
|---|---|
| `candidates.py` | Phase A — RCSB query + 9-gate filter chain with drop-reason tagging. |
| `download.py` | Phase B — chunked rsync orchestration with primary / fallback mirrors. |
| `verify.py` | Phase C — gemmi parsing, polymer-type classification, observation-fraction validation, SHA256 hashing. |
| `manifest.py` | Final manifest assembly with `PIPELINE_VERSION` + config hash. |
| `report.py` | Drop-reason audit summary across all phases. |
| `rcsb.py` | RCSB GraphQL + REST client with exponential backoff. |
| `interfaces.py` | Assembly / interface deduplication (one canonical entry per biological assembly). |
| `obsolete.py` | RCSB obsolete-API client; successor resolution for superseded entries. |
| `config.py` | Curation config dataclass (filter thresholds, date bounds, mirror URLs). |
