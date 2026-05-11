# `twistr.linkers` — Rosetta Remodel-based linker design

Designs four peptide linkers connecting two designed α-helices to a fixed
framework chain. Downstream of helix design and lead optimization; the
final construct is a single chain `[framework]–[linker 1]–[helix
A]–[linker 2]–[framework continuation]–[linker 3]–[helix B]–[linker 4]–
[framework continuation]` ready for experimental construct synthesis.

## Why Rosetta Remodel and not an ML approach

Linker design is the only stage in this pipeline that doesn't run on
ML. The reasoning:

- Linker conformational space is small and well-characterized — short
  loops between fixed anchors with constrained chain-closure geometry.
- Rosetta Remodel's KIC (kinematic-closure) sampling has been the
  field-standard tool for this for over a decade. It generates
  physically realizable backbones, samples from the Dunbrack rotamer
  library, and enforces chain closure to angstrom-level precision.
- An end-to-end ML approach would have to learn what Remodel + KIC
  already gets right, against a much smaller training set than the rest
  of this pipeline targets. Not worth building.

The choice is **use Rosetta where it's already excellent; use ML where
it isn't**.

## Pipeline

```
3 input PDBs (framework + helix A + helix B, pre-aligned)
   ↓ pose_builder.py
4 sub-poses (each: 2 anchors + N Ala placeholders)
   ↓ blueprint.py
4 Rosetta Remodel blueprints (DSL: '.' = fixed, 'L PIKAA <whitelist>' = design)
   ↓ remodel_runner.py (subprocess into isolated PyRosetta venv)
4 × nstruct × num_trajectory design attempts
   ↓ scoring + best-of-nstruct
4 chosen linker designs
   ↓ driver.py
final.pdb (full construct) + designs.parquet (all attempts scored)
```

## Selected technical details

- **Isolated PyRosetta subprocess.** Rosetta's Python bindings have a
  dependency tree (Python version, system libraries) that conflicts
  with the main twistr venv. `remodel_runner.py` spawns a subprocess
  using a separate `.venv-rosetta` interpreter, marshals the job
  specification as JSON, and collects per-attempt scores via
  `scores.json`. The worker script `_remodel_script.py` has **zero
  twistr imports** — strictly stdlib + PyRosetta — so the dependency
  walls don't leak.
- **Sub-pose extraction with gemmi.** Each of the four linkers needs
  its own input pose: upstream anchor (last few residues before the
  cut point on the framework / previous helix) + linker placeholders +
  downstream anchor. `pose_builder.py` does the splicing through gemmi:
  segment the framework chain at configured cut points, extract helix
  residue ranges, clone atoms with residue renumbering.
- **Ala placeholder geometry.** Linker residues need *something*
  occupying the gap between the anchors so Remodel has an initial
  pose. We use minimal Ala residues (N / CA / C / O / CB only) placed
  at canonical bond distances. Real sidechain geometry is irrelevant
  because Remodel rebuilds the entire linker backbone via fragment
  insertion + KIC closure; the placeholders are just topology.
- **Extended-chain initial CA positions.** Linker placeholders are CA-
  spaced at 3.80 Å in an extended-chain conformation walking from
  upstream to downstream anchor. This is the geometric initialization
  that gives KIC the most slack for chain closure — folded
  conformations would have less room for the minimizer to find a
  valid solution.
- **Blueprint DSL.** Rosetta Remodel takes a per-residue text file:
  one residue per line, with a flag indicating fixed (`.`) or rebuild
  (`L`) plus an optional `PIKAA <whitelist>` clause restricting the
  amino-acid choices at that position. We constrain linker positions
  to a loop-friendly whitelist (default `AGSDNTPQEKR`) — excludes
  large hydrophobics (no L, I, V, F, W, Y), prolines (Remodel handles
  P specially), and cysteines (avoid spurious disulfides).
- **Chain-break acceptance.** Designs are accepted only if all CA-CA
  bond deviations stay within 0.5 Å of the 3.80 Å ideal — eliminates
  KIC solutions that found local energy minima at the cost of broken
  chain geometry.
- **Best-of-nstruct selection.** Per linker, `nstruct=20`
  conformational attempts × `num_trajectory=20` centroid trajectories
  by default. Selection is by Rosetta total score among chain-break-
  passing designs; ties broken by lowest deviation from the ideal CA-
  CA spacing.

## File index

| File | Role |
|---|---|
| `config.py` | `LinkersConfig` — input PDB paths, `LinkerWindow` cut points per linker, `LinkerLengths` per linker, output dir, Rosetta venv path, nstruct / num_trajectory / context_residues / seed, AA whitelist. |
| `pose_builder.py` | Framework segmentation; gemmi atomic cloning with renumbering; `SubposeLayout` dataclass; extended-chain CA placement; Ala placeholder construction; `build_all_subposes` factory. |
| `blueprint.py` | Generates Remodel blueprint text — `.` for anchors, `L PIKAA <whitelist>` for linker residues. |
| `remodel_runner.py` | Subprocess orchestration: spawn isolated PyRosetta interpreter, marshal job JSON, collect scored designs. |
| `_remodel_script.py` | PyRosetta worker — zero twistr imports. Applies Remodel with blueprint, runs nstruct × num_trajectory attempts, validates chain breaks, writes per-design PDB + scores.json. |
| `driver.py` | End-to-end: build sub-poses → per-linker (blueprint → remodel → best-of-nstruct pick) → assemble full pose → write `final.pdb` + `designs.parquet`. |
