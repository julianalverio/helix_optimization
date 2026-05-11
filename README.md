# `twistr` — Computational pipeline for α-helix-mediated binder design

End-to-end Python package for designing α-helix-mediated protein binders
targeting GI indications at a stealth biotech. The pipeline is structured
as discrete, individually-callable stages — PDB curation, tensor
extraction, ML training-example assembly, epitope selection on the target
antigen, lead-optimization structure prediction, and Rosetta-based linker
design — composed so each stage's output is the input to the next.

## Design context

The deployed system is an **agentic loop**. Initial helix candidates are
generated externally (PXDesign + Boltz against the epitope this pipeline
identifies). An agent then iteratively mutates each candidate, scoring
every proposal against three downstream metrics — **shape complementarity
(SC), electrostatic complementarity (EC), and buried surface area (BSA)**
— computed from this pipeline's structure prediction (in `ml/`). The
lead-optimization model is the per-mutation predictor in the agent's
inner loop; the rest of the package builds the training data,
infrastructure, and surrounding design tooling that makes that predictor
useful.

## Pipeline stages

```
┌───────────────────────────────────────────────────────────────────────┐
│  PDB           atom14           training         lead-opt   linker    │
│  curation  →   tensors      →   examples    →    ML model →  design   │
│  (RCSB)        (gemmi+DSSP)     (helix+      ↘   ↗                    │
│                                  partner)   agent loop:               │
│                                              SC / EC / BSA            │
│                                              scoring                  │
│              ┌─────────────────────────────┐                          │
│              │ epitope_selection runs in   │                          │
│              │ parallel — chooses where    │                          │
│              │ on the antigen to bind:     │                          │
│              │ MaSIF → ScanNet → hotspot   │                          │
│              └─────────────────────────────┘                          │
└───────────────────────────────────────────────────────────────────────┘
```

## CLI surface

`twistr/cli.py` exposes the pipeline as subcommands:

| Command | Stage |
|---|---|
| `fetch-candidates` | Curation phase A — RCSB GraphQL/REST → candidate manifest. |
| `download` | Curation phase B — chunked rsync of selected mmCIFs. |
| `verify` | Curation phase C — gemmi parsing, structural validation. |
| `report` | Curation final — manifest + drop-reason audit report. |
| `run-all` | Curation phases A→C → manifest → report in one invocation. |
| `tensors` | Module 2 — atom14 tensor extraction with DSSP + canonicalization. |
| `examples` | Module 3 — per-helix training examples (helix + partner crop). |
| `linkers` | Rosetta Remodel linker design between two helices and a framework. |
| `epitope-selection-run` | MaSIF → ScanNet → PPI-hotspot → PyMOL viz, stages selected by YAML. |

The ML training run itself lives in `twistr/ml/training/train.py` and is
launched separately (cloud orchestration in the repo-root `dev/tools/runpod/`
directory, not exposed through this CLI).

## Sub-packages

| Path | Role |
|---|---|
| `curation/` | RCSB ingestion: GraphQL/REST candidate selection, chunked rsync, gemmi-based structural validation, multi-filter audit trail. |
| `tensors/` | mmCIF → atom14 tensor packing with DSSP-derived secondary structure and sidechain canonicalization for rotamer-symmetric residues. |
| `examples/` | Per-helix training-example assembly: helix detection, cKDTree partner-contact indexing, deterministic stable-seed windowing, optional SASA filtering. |
| `epitope_selection/` | MaSIF (Docker) → ScanNet (Docker) → PPI-hotspot (Modal) → PyMOL visualization. Content-hashed caching at each stage. |
| `ml/` | Lead-optimization model: Pairformer trunk + frame / torsion / interaction-matrix heads, multi-objective loss with handoff-scheduled physics priors. **See `ml/README.md` for the full architecture.** |
| `linkers/` | Rosetta Remodel-based linker design between two helices and a framework, run in an isolated PyRosetta subprocess to avoid dependency conflicts. |
| `external/` | Vendored git submodules: Protenix (Pairformer source), AlphaFold (atom14 rigid-group constants), MaSIF, ScanNet, PXDesign, ProteinMPNN, FAFE. Canonical constants extracted via AST at module load — no copy-paste, no transitive dependency blow-up. |

## Design principles worth flagging

- **Stages are first-class.** Each pipeline stage is a separate CLI
  subcommand with its own YAML config, idempotent on its inputs, with
  drop-reason audit trails where applicable. Failures at one stage don't
  invalidate earlier outputs.
- **Vendored externals, AST-extracted constants.** Upstream packages
  (Protenix, AlphaFold) drag in heavy dependency trees (rdkit, JAX). We
  extract their canonical constants — chi-angle atom tuples, rigid-group
  geometry — via `ast.parse` at module load. The effect is identical to
  a direct import but doesn't pull JAX into our hot path, and the
  upstream source is the single source of truth (never copy-pasted).
- **Container-wrapped external models.** MaSIF, ScanNet, and PPI-hotspot
  each ship with incompatible Python / CUDA / system-library
  requirements. We invoke them through Docker (locally) or Modal
  (remotely) rather than wrestle a single venv to host all of them.
- **Content-hashed caching.** Long-running stages (PPI-hotspot via
  Modal, batch calibration on GPU) cache by content hash of their
  inputs plus the relevant configuration fields — *not* by loss
  weights, learning rate, or anything else that doesn't affect the
  cached computation. Hyperparameter sweeps reuse the cache.

## Repository layout

```
twistr/             # the package (this directory)
  cli.py            # argparse CLI surface
  curation/  tensors/  examples/  ml/  epitope_selection/  linkers/  external/
configs/            # YAML config files (one per stage)
tools/              # repo-root tools — RunPod orchestration, smoke tests, dunbrack fit
data/               # curated PDB outputs (mmCIF, atom14 npz, training examples)
tests/              # pytest suite
```
