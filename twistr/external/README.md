# `twistr.external` — Vendored upstream submodules

Third-party code we depend on, vendored as git submodules rather than
installed from PyPI. The general pattern is: take the canonical
implementation from the original authors, pin it at a specific commit,
and either (a) extract its constants via AST at module-load time, or
(b) invoke it through Docker / Modal in isolation. Both strategies
avoid pulling the upstream's full dependency tree (rdkit / JAX / CUDA
variants) into our hot path while still treating the upstream as the
single source of truth.

## Submodule inventory

| Submodule | Upstream | How we use it |
|---|---|---|
| `Protenix/` | ByteDance's open-source AF3 implementation. | Source of the Pairformer trunk (`twistr/ml/models/pairformer.py` is a clean transcription of `Protenix/protenix/model/modules/pairformer.py` with `Source:` line comments). `twistr/ml/features/chi_angles.py` extracts the canonical `_CHI_ANGLES_ATOMS` table from the submodule's `constants.py` via `ast.parse` at import time — no `import protenix` needed (avoids rdkit). |
| `alphafold/` | DeepMind's AlphaFold 2 reference implementation. | `twistr/ml/models/sidechain.py` extracts the atom-14 rigid-group constants (`restype_rigid_group_default_frame`, `restype_atom14_rigid_group_positions`) from `alphafold/common/residue_constants.py` via AST. The actual `alphafold` package is never imported — that would drag in JAX, which we don't need at inference. |
| `PXDesign/` | Wraps Protenix for the targeted-mutation inference regime. | Used for the **initial helix-candidate generation** step (upstream of this lead-optimization model). Runs at design-pipeline-start time on a separate stack; the output (initial helix sequences + structures) is what the agent loop then iterates on. RunPod orchestration lives in `tools/runpod/pxdesign/`. |
| `ProteinMPNN/` | The Baker lab's sequence-from-structure design model. | Sequence-design utility on candidate backbones. Used by the agentic loop for proposal generation (mutate-and-test), parallel to the pure-mutation path. |
| `ScanNet/` | Tubiana et al. 2022 graph-based binding-site predictor. | One of the three epitope-scoring methods in `twistr/epitope_selection/`. Invoked through Docker (`epitope_selection/scannet_filter/scannet_runner.py`) — ScanNet's TF/Keras stack is incompatible with our PyTorch venv. Outputs per-residue binding-probability scores. |
| `masif/` + `masif-dockerfile/` | Gainza et al. 2020 molecular-surface interaction-field predictor. | The first epitope-scoring method. Invoked through Docker (`epitope_selection/epitopes/masif_runner.py`). Outputs per-vertex contactability scores on the molecular surface; our `patches.py` re-aggregates to residues by top-quartile nearest-vertex averaging. The companion `masif-dockerfile/` is the container definition. |
| `FAFE/` | Function-Annotation Field Energy — auxiliary scoring. | Currently vendored; integration with the main pipeline is exploratory. |

## Integration patterns

Three patterns recur:

1. **AST extraction.** Used for `Protenix` constants and `alphafold`
   residue constants. Read the upstream `.py` source as text, parse to
   AST, walk for the named module-level assignments, evaluate the
   right-hand sides with `ast.literal_eval`. Effect: we get the
   canonical Python value (dict / tuple / list of strings) at module
   load time without ever executing `import alphafold` or
   `import protenix`. The upstream value is the single source of truth
   and stays in sync automatically with the pinned commit.
2. **Clean transcription with `Source:` comments.** For `Protenix`'s
   Pairformer. The upstream module imports the full Protenix dependency
   tree (rdkit, optree, an `inplace_safe` chunking layer, JIT-compiled
   CUDA LayerNorm). At our problem size none of that is needed, so
   `twistr/ml/models/pairformer.py` reimplements the same Algorithm-17
   blocks class-for-class, with a `Source: protenix/.../pairformer.py
   :L<start>-L<end>` comment on each block so upstream changes remain
   traceable.
3. **Container invocation.** For `masif`, `ScanNet`, and (via Modal)
   `PPI-hotspot`. Each upstream model has incompatible Python / CUDA /
   system-library requirements; we invoke them through Docker (locally)
   or Modal (remotely) and exchange data through parquet / JSON files
   on the host filesystem.

## Why not just pip-install everything

A counterfactual where all the above were `pip install`-able would
require: rdkit (massive, breaks on many platforms), JAX with matching
CUDA, TF 1.x for ScanNet, TF 2.x for MaSIF, PyRosetta (license-gated +
own Python version requirement), AmberTools (Conda-only). The vendored
+ AST-extracted + containerized approach lets us pin upstream commits
to specific known-good states and keep the hot path's import graph
narrow enough to load in <1 second on a fresh process.
