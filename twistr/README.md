# `twistr/` — repo layout

Navigation guide for people inside the repo. For the project's *what* and
*why*, see [`../README.md`](../README.md).

## Layout

```
twistr/
├── cli.py                 # twistr-CLI dispatcher (curation, tensors, examples,
│                          # linkers, epitope-selection-run subcommands)
│
├── curation/              # Phase A–D PDB curation: RCSB candidates →
│                          # rsync mmCIF → per-entry verification →
│                          # final manifest. Houses the curation Config
│                          # dataclass + path helpers used by tensors/ and
│                          # examples/.  ↳ README inside.
│
├── tensors/               # Per-PDB structure → atom14 .npz tensors
│                          # (gemmi parse, altloc/MSE/canonicalization,
│                          # cofactor extraction, DSSP, chain filters).
│                          # Was "module 2".  ↳ README inside.
│
├── examples/              # Per-tensor → helix-mediated training-example
│                          # .npz files (segmentation, windowing, partner
│                          # selection via distance ∪ ΔSASA, manifest).
│                          # Was "module 3".  ↳ README inside.
│
├── epitope_selection/     # MaSIF → ScanNet → PPI-hotspot → PyMOL viz
│                          # orchestrator. Picks the surface region on a
│                          # target antigen that downstream binder design
│                          # will engage.  ↳ README inside.
│
├── linkers/               # Rosetta-Remodel-driven linker design between
│                          # two helices anchored to a framework chain.
│                          # ↳ README inside.
│
├── pipeline/              # ML training pipeline: PyTorch-Lightning
│                          # datamodule + features + losses + models +
│                          # training loop. Consumes examples/ output;
│                          # produces the per-mutation lead-optimization
│                          # predictor.  ↳ README inside.
│
├── agent/                 # Agentic loop that mutates candidate binders
│                          # and scores each proposal against SC / EC /
│                          # BSA via the pipeline/ predictor. Active
│                          # development.  ↳ README inside.
│
├── external/              # Git submodules: PXDesign, Protenix, FAFE,
│                          # alphafold, ProteinMPNN, ScanNet, masif.
│                          # Vendored upstream — not modified here.
│
├── runtime/               # User-edited operational artifacts.
│   ├── configs/           # One YAML per pipeline stage (curation,
│   │                      # tensors, examples, linkers, ml, pxdesign,
│   │                      # epitopes, epitope_viz, hotspot, scannet,
│   │                      # epitope_selection, agent).
│   ├── data/              # PDB cache, curation manifests, tensor and
│   │                      # example .npz trees, epitope-pipeline
│   │                      # parquets, dunbrack rotamer dataset, target
│   │                      # PDB-id lists. Mostly gitignored.
│   └── outputs/           # All runtime artifacts (PXDesign, BoltzGen,
│                          # MPNN, refold, smoke, scoring, rankings,
│                          # cropped/grafted helices, pymol views,
│                          # logs, wandb). Gitignored.
│
└── dev/                   # Developer tooling (excluded from wheel build).
    ├── tests/             # pytest suite, mirrors the package layout:
    │                      # curation/ tensors/ examples/ linkers/
    │                      # pipeline/ epitope_selection/ + fixtures/.
    └── tools/
        ├── runpod/        # Pod-targeted launchers + bootstrap scripts:
        │                  # boltzgen/ pxdesign/ train/ smoke_test/.
        └── local/         # Local-only utilities: dunbrack/ (rotamer
                           # library fitter) oom_probe/.
```

## Conventions

- **Working directory: `twistr/`**. CLI invocations and path strings in
  YAMLs/code are written relative to here. From inside `twistr/`,
  `runtime/configs/curation.yaml` and `runtime/data/pdb/` resolve as
  written.
- **Each pipeline subpackage owns one config**: `curation/` ↔
  `runtime/configs/curation.yaml`, `tensors/` ↔ `runtime/configs/tensors.yaml`,
  and so on. `epitope_selection.yaml` is the meta-config for the
  MaSIF→ScanNet→hotspot→viz orchestrator.
- **Per-subpackage READMEs are the source of truth** for stage internals
  (algorithms, output schemas, drop-reason taxonomies, Modal config). This
  file just tells you *where* — open the subpackage README for *what* and
  *how*.
- **Module paths mirror the directory tree.** Launchers under
  `dev/tools/runpod/boltzgen/` import as `twistr.dev.tools.runpod.boltzgen.X`.

## Quick CLI

From inside `twistr/`:

```
python -m twistr.cli fetch-candidates       # Phase A: RCSB candidate set
python -m twistr.cli download               # Phase B: rsync mmCIF
python -m twistr.cli verify                 # Phase C: per-entry verification
python -m twistr.cli report                 # Phase D: final manifest + report
python -m twistr.cli tensors                # build atom14 tensors
python -m twistr.cli examples               # build training examples
python -m twistr.cli linkers                # design linkers via Rosetta
python -m twistr.cli epitope-selection-run  # MaSIF → ScanNet → hotspot → viz
```

Each subcommand reads its YAML from `runtime/configs/`; pass `--<subcommand>-config <path>` to override.
