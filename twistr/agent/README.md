# `twistr.agent` — Claude Opus 4.7 lead-optimization agent

Closes the outer loop of the binder-design pipeline. Given a target
structure with two α-helices bound (output of the upstream
PXDesign + Boltz design step) and a trained lead-optimization model
checkpoint (from `twistr.pipeline`), an LLM-driven agent iteratively
proposes single-residue point mutations to either helix, scores each
proposal against shape complementarity, electrostatic complementarity,
and buried surface area on the model's predicted heavy-atom geometry,
and maintains a Pareto frontier of non-dominated designs. Per-design
PDBs are written to disk for every Pareto-admitted candidate so the
final deliverable is *structures*, not just sequences and scores.

## Where it sits in the pipeline

```
curation ─► tensors ─► examples ─► ML training (one-time)
                                    ↓
                              trained checkpoint
                                    ↓
PXDesign + Boltz ─► AGENT (this dir) ─► PDBs ready for AF3 callback
   (initial      (outer loop:                +  wet-lab synthesis
   proposals)    iterative mutation +
                 multi-objective scoring)
```

The ML model in `twistr.pipeline` is the inner-loop predictor. This
directory is the outer-loop optimizer that drives it. Every proposed
mutation triggers one forward pass.

## Why an LLM agent and not a classical optimizer

The search space is categorical and combinatorial (20 amino acids ×
~50 mutable positions, with multi-mutation lineages), the objectives
are non-smooth, and the inner-loop predictor is expensive enough
(~1 s per forward pass on an A100) that the budget per design session
is ~200 evaluations, not 20,000. That rules out gradient-based
methods, makes Bayesian optimization awkward (assumes continuous
objectives), and makes random sampling wasteful.

A genetic algorithm or Rosetta-style MCMC over residue identities
would also work — but neither brings prior knowledge of *which* amino-
acid substitutions are likely to pay off. An LLM agent brings native
biochemistry into the proposer: "this position pairs across the
interface with target E156 — introduce K or R to make a salt bridge"
is a move a Bayesian-optimization framework can't make without being
re-taught.

The trade-offs we accept in exchange: more expensive per iteration
(each tool call is a Claude turn), no provable optimality guarantees,
quality depends on the LLM's protein-biophysics knowledge. For the
lead-opt regime — small, in-distribution edits to known-binding
helices — these are acceptable.

## Loop structure

```
        ┌─────────────────────────────────────────────────────┐
        │  Claude Opus 4.7                                    │
        │   - reads current state + Pareto frontier           │
        │   - returns tool_use: propose_mutation(...)         │
        └────────────────────┬────────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │  Designer.apply_mutation                            │
        │  → batch["residue_type"][n] := new_aa               │
        │  → atom_mask row rebuilt for new residue's slots    │
        └────────────────────┬────────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │  Designer.predict  (one forward pass)               │
        │  → atom-14 coords (Å) + (B, N, N, 6) IM probs       │
        └────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────────────────┐
        ▼                    ▼                                ▼
  shape_compl       electrostatic_compl            buried_surface_area
  (Lawrence-        (residue-charge Coulomb,       (freesasa,
   Colman proxy)     tanh-normalized)               probe = 1.4 Å)
        │                    │                                │
        └────────────────────┼────────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │  ParetoFrontier.add(design)                         │
        │  → tool_result: SC, EC, BSA,                        │
        │     admitted_to_pareto, remaining_iterations        │
        │  → if admitted: write_pdb(design_<id>.pdb)          │
        └────────────────────┬────────────────────────────────┘
                             ▼
                       (loop continues)
```

The conversation is multi-turn. Claude can call any of five tools —
`propose_mutation`, `query_residue`, `report_pareto`,
`revert_to_design`, `finish` — accumulate results in its context, and
reason about which mutation to try next.

## Three-objective optimization

All three metrics operate directly on the predicted atom-14 tensors;
no PDBs are written during the inner loop.

| Metric | Definition | Implementation |
|---|---|---|
| **SC** | Lawrence & Colman (1993) shape-complementarity proxy. For each interface Cβ, find the nearest cross-chain interface Cβ; take the Gaussian-weighted median of `-dot(helix_dir, target_dir)` where `*_dir = Cβ − Cα`. Glycine excluded (no Cβ). Range [-1, +1]. | `metrics/shape_complementarity.py` |
| **EC** | Residue-charge Coulomb sum across interface charged residues (D / E = −1; K / R = +1; H = +½). Sign flipped so opposite-charge pairs increase the metric; normalized over contributing pairs; tanh-squashed to [-1, +1]. A McCoy et al. (1997) PB-based formulation would be more rigorous but is impractical at agent-loop frequency. | `metrics/electrostatic_complementarity.py` |
| **BSA** | freesasa Lee-Richards SASA at probe = 1.4 Å. `(SASA_helix_alone + SASA_target_alone − SASA_complex) / 2`. freesasa structures are built via `addAtom` directly from the atom-14 tensors — no disk round-trip. | `metrics/buried_surface_area.py` |

A design dominates another iff it is ≥ on all three objectives and >
on at least one. The frontier is recomputed in O(|frontier|) per
insertion, with NaN treated as the worst possible value — so a metric
that is undefined for a given complex (e.g. EC on a purely
hydrophobic interface) neither dominates nor is dominated arbitrarily.

**Why Pareto and not a weighted sum.** SC, EC, and BSA pull in
slightly different directions (e.g. introducing a charged sidechain
helps EC but can disrupt hydrophobic packing → drops SC). A scalarized
weighted sum requires picking weights up front; the Pareto formulation
defers the trade-off to the wet lab, which gets a small set of
non-dominated candidates rather than one winner that depended on the
choice of weights.

## Selected technical decisions

- **Stateful Designer, not stateless re-parse.** Mutations apply in
  place to a cached batch tensor. `revert_to_design` replays the
  inverse of the current mutation history then re-applies the target
  design's lineage, so backtracking is O(|history|) rather than
  O(re-parse + re-build features). The model is loaded once via
  `ExamplesModule.load_from_checkpoint`; hyperparameters travel inside
  the checkpoint, so the agent YAML doesn't repeat ML-model knobs.

- **atom-14 mask rebuilt on every mutation.** A residue's atom-14
  layout depends on its identity (ALA 5 slots, LEU 8, TRP 14). On
  `apply_mutation` the (14,) mask row is rebuilt from
  `ATOM14_SLOT_NAMES[new_idx]` — both *enabling* slots the new residue
  owns and *zeroing* slots it doesn't. Coords for newly-enabled slots
  start at zero but are re-synthesised by `apply_torsions_to_atom14`
  from the predicted frame + χ angles, so the zero initialisation
  never reaches a metric. (An earlier version zeroed only beyond
  `new_slot_count` and silently left previously-empty slots disabled,
  so adding atoms via mutation was invisible to downstream features —
  a real bug I caught during the audit.)

- **Per-design PDB output.** Every Pareto-admitted design is written
  to `runtime/outputs/agent/designs/design_<id>.pdb` with the original
  chain IDs and PDB residue numbers preserved, so designs open in
  PyMOL alongside the wild-type input for direct comparison. This is
  what makes the Pareto frontier a *deliverable*: sequences plus
  scores aren't enough for an experimental review.

- **Conditioning IM is plumbed but not exposed as a lever yet.** The
  model takes a noised (B, N, N, 8) conditioning IM in addition to
  coordinates; the agent currently only mutates `residue_type`. The
  conditioning-IM channel is the natural place to add a second tool
  (`adjust_contact_prior(...)`) once the sequence-only loop is
  validated end-to-end.

- **Five tools, not a function-call free-for-all.**
  `propose_mutation` is the workhorse. `query_residue` lets the agent
  inspect without consuming budget. `report_pareto` lets it re-anchor
  on the frontier. `revert_to_design` lets it escape local optima.
  `finish` lets it terminate early when the frontier has stabilized.
  The minimal surface keeps agent behaviour tractable.

- **Argument errors don't consume budget.** Invalid chain IDs or
  out-of-range positions return an error without incrementing the
  iteration counter; the agent gets the error message back in the
  tool result and can retry. Every tool result also carries
  `remaining_iterations` so the agent can pace itself.

## Outputs

Per session, under `runtime/outputs/agent/`:

- `designs/design_<id>.pdb` — one PDB per Pareto-admitted design,
  including the wild-type baseline at id 0. Original chain IDs and
  PDB residue numbers preserved; predicted heavy-atom coordinates in Å.
- `pareto_current.jsonl` — flushed every `save_intermediate_every`
  iterations. Each line is one `Design`: id, parent_id, mutation
  lineage from WT, helix sequences in 1-letter code, SC / EC / BSA,
  iteration number, agent-supplied rationale string.
- `pareto_final_<unix>.jsonl` — written at clean termination.

The combination is sufficient for an experimental scientist to filter
candidates, render them in PyMOL, and decide which to synthesize.

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run --extra agent python -m twistr.agent.driver --config runtime/configs/agent.yaml
```

`runtime/configs/agent.yaml` carries the target PDB path, helix and
target chain IDs, checkpoint path, search budget, and metric
hyperparameters. Unknown keys are rejected at load time so config
drift fails fast.

## File index

| File | Role |
|---|---|
| `config.py` | `AgentConfig` dataclass + strict YAML loader. |
| `designer.py` | Multi-chain PDB → batch dict (atom-14 packing + cKDTree interface detection); checkpoint load; mutation apply / revert with atom-mask rebuild; forward pass; per-design PDB writer. |
| `pareto.py` | `Design` dataclass; `ParetoFrontier` with non-dominated insertion, NaN-as-worst handling, JSONL persistence. |
| `prompts.py` | `SYSTEM_PROMPT` covering role, tools, reference scales for SC / EC / BSA, substitution heuristics, loop-pacing guidance; `initial_user_prompt` seeding the conversation with the WT baseline. |
| `tools.py` | Anthropic-format tool schema for the five exposed actions. |
| `claude_client.py` | Thin wrapper around `anthropic.Anthropic().messages.create`. |
| `driver.py` | The agent loop: Claude turn → tool execution → tool_result → next turn. Persists per-design PDBs and Pareto JSONL. CLI entry. |
| `metrics/shape_complementarity.py` | Cβ-direction Lawrence-Colman proxy. |
| `metrics/electrostatic_complementarity.py` | Residue-charge Coulomb sum, tanh-normalized. |
| `metrics/buried_surface_area.py` | freesasa SASA, direct from atom-14 tensors. |
