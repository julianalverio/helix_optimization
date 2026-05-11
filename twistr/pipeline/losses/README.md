# `twistr.pipeline.losses` — supervision that targets the agent's metrics

The agent's downstream scoring depends on three geometric quantities —
**shape complementarity, electrostatic complementarity, buried surface
area** — that are computed from predicted heavy-atom positions and
contact patterns *at the interface*, not from overall fold accuracy. The
loss design follows directly from this: a structural anchor, a
classification objective on the interaction matrix, and nine physical
priors that supervise the geometric primitives those three metrics
depend on.

## Agent-metric coverage

| Agent metric | Primary losses | Mechanism |
|---|---|---|
| Buried surface area | `coord_mse_annealed`, `packing` | Anchored sidechain positions + explicit neighbor-density reward. |
| Shape complementarity | `packing`, `vdw`, aromatic stacking sub-types (`parallel_displaced`, `sandwich`, `t_shaped`), `clash` | Heavy-atom geometry in correct VDW / stacking bands; clash penalty prevents inter-penetration. |
| Electrostatic complementarity | `hbond` geometric, `interaction_bce` (h-bond channel) | Donor-acceptor distance + angle geometry; supervised contact-pattern prediction. |
| Rotamer plausibility (prerequisite for all three) | `dunbrack` (joint vMM) | Predicted χ angles stay on the empirical rotamer manifold. |
| Local backbone integrity | `helix_dihedral`, `backbone_continuity` | Helix φ/ψ near canonical; consecutive same-chain C(i)–N(i+1) bond geometry. |

## Structural anchor

`coord_mse.py` — per-atom MSE in Å² with caller-supplied inclusion masks.
No Kabsch alignment: predictions and GT share the per-example random-
rotation augmentation from `ExamplesDataset`, so they live in the same
frame. Two callers in `lightning_module._losses_from_out`:

- **Antigen backbone** (slots N, CA, C, O of every non-helix real
  residue) — constant weight throughout training. The partner backbone
  is fixed input, so we always want it pinned to GT.
- **Interface helix all-atoms ∪ antigen interface-residue sidechains**
  — weight on a linear-anneal schedule. Early training imitates the GT
  design closely; later training is free to find any band-/clash-/
  dihedral-valid configuration. The annealed mask is gated on
  `is_interface_residue` on both sides: non-interacting residues
  contribute to neither numerator nor denominator, so the metric is
  computed where it matters for SC / BSA.

## Classification

`interaction_bce.py` — per-channel binary cross-entropy with logits on
the 6-channel interaction matrix. **Multi-label, not multi-class**: a
pair can have VDW = 1 and h-bond = 1 simultaneously; the channels are
independent indicators. Per-channel label smoothing is configurable.

## Geometric flat-bottom losses

`interaction_geometry.py` — two-sided losses for VDW, h-bond, and the three
π-stacking sub-types. Zero inside the geometric band, linear in physical
units (Å, cosine) outside it. The two-sided structure handles GT = 1
(must form: penalty for distance / angle outside the band) and GT = 0
(must not form: penalty for sitting inside the band) with one loss
function per channel. Reduced as **minimum violation over atom-level
alternatives** so multi-atom-pair geometries (e.g., a residue pair
with multiple donor/acceptor combinations) need only one satisfying
configuration per pair.

Band parameters are the same constants used by the IM detector in
`features/interaction_matrix.py` — labels and losses cannot drift apart by
construction.

## Steric clash + backbone integrity

`steric_clash.py` — AF2-style heavy-atom clash penalty (Jumper et al.
2021, Suppl. Alg. 28). One-sided: only fires on overlap.

`backbone_continuity.py` — AF2 between-residue bond geometry (Suppl.
Sec. 1.9.11, eq. 44–45) on C(i)–N(i+1) bond length and adjacent bond
angles. **Helix-only gated**: the antigen comes in as a contact-radius
crop with arbitrary chain breaks at the boundaries, so peptide-bond
geometry across a crop break is meaningless. Gating both residues on
`is_helix` was a real bug fix — antigen pairs at crop boundaries were
contributing nonsense terms before.

`helix_dihedral.py` — canonical α-helix φ/ψ target on `is_helix`
residues.

## Dunbrack rotamer prior (joint vMM)

`dunbrack.py` — `-log p(χ₁..χ_n | residue, ss_class)` under a joint
von-Mises mixture in n_χ dimensions. Two libraries indexed by secondary-
structure class ('helix' fit on SS = H rows; 'general' fit on all
rows); the loss dispatches by `is_helix`. The Dunbrack term is logged
split — interacting vs non-interacting residues — as a diagnostic for
whether rotamer plausibility is being satisfied at hot-spot positions
specifically.

Two design decisions worth flagging:

- **Joint, not factorized.** Standard practice fits one 1D vMM per chi
  axis. Real rotamers have strong joint structure — Leu's χ₁ = −60°
  goes with χ₂ = 180° together, almost never with other χ₂s.
  Factorizing `p(χ)` as a product of marginals overcounts the rotamer
  space and gives the model an implausibly easy plausibility signal.
  We fit one joint n_χ-dimensional vMM per (residue, ss_class), with
  K = min(3^n_χ, 36) components initialized at the canonical-rotamer
  Cartesian product `{−60°, +60°, +180°}^n_χ`.
- **π-periodicity handled robustly.** ASP χ₂, GLU χ₃, PHE χ₂, TYR χ₂
  have 2-fold rotational symmetry — the +90° and −90° conformations
  are physically identical. The PDB contains both depending on which
  atom got labeled CD1 vs CD2 by the depositor. We don't trust the
  data to be symmetric on average: the offline fitter explicitly
  augments every π-periodic χ with its χ + π copy before EM, and the
  loss evaluates the joint density at the original χ AND at χ flipped
  by π on the periodic axes, averaging the two. Result: the loss is
  *exactly* invariant under χ_c → χ_c + π regardless of any residual
  EM convergence asymmetry.

Dataset quality filter: drop rows with `RSPERC < 25` (the Dunbrack
group's own recommended developer cutoff). For ASN/GLN/HIS, also drop
rows where the FlpConfid amide-flip determination is not "clear".

## Packing-neighbor prior (designed for SC / BSA)

`packing.py` — heavy-atom neighbor-count prior that explicitly targets
the agent's SC and BSA metrics. For each interface residue's **packing
atoms** — aliphatic stub carbons (CB / CG / CD on hydrophobic residues;
just CB on polar residues whose CG would be a carbonyl or carboxylate
carbon) ∪ aromatic ring atoms — counts heavy-atom neighbors in the VDW
band [3.3, 5.5] Å on residues with supervised positions, and penalizes
under-packing via `relu(n_target − count)` with n_target = 4.

Neighbor scope ("Option III"): helix all-atoms ∪ antigen backbone ∪
antigen interface-residue sidechain. Excludes antigen non-interface
sidechain atoms — those positions are not anchored by any other loss,
and including them would let the model fake packing against
unconstrained drifted sidechains. Self-residue atoms excluded so a
residue's CB doesn't count its own CG as a neighbor.

One-sided `relu`: only under-packing is penalized; clash handles the
lower bound. Reduction: per-atom mean → per-example mean over examples
with ≥1 from-atom → batch mean (same shape as `vdw_interaction_loss`).

## Loss-handoff schedule

Per-loss weights are routed through a shared `_ramped_weight` helper:
`weight(step) = lerp(start, end, min(step / ramp_steps, 1))`, with
validation pinned to `max(start, end)` so val totals stay comparable
across the run regardless of where training is on its schedule.

Coordinate MSE dominates the gradient at step 0 (real protein geometry
trivially satisfies all the band losses, so MSE does most of the work
implicitly). As MSE anneals down, the geometric priors are ramped up in
lockstep. This is **not cosmetic infrastructure**: an earlier training
run with annealing-down on MSE but no compensating ramp-up on the
geometric priors showed a sustained step-up in VDW and aromatic losses
right around the annealing breakpoint — the structural anchor releasing
without geometric priors compensating let predicted heavy-atom geometry
drift out of the VDW bands. The handoff infrastructure fixes this
class of artifact.

## File index

| File | Role |
|---|---|
| `coord_mse.py` | Per-atom MSE with caller-supplied inclusion mask. |
| `interaction_bce.py` | 6-channel BCE with logits + per-channel label smoothing. |
| `interaction_geometry.py` | Flat-bottom geometric losses for VDW, h-bond, π-stacking sub-types. |
| `steric_clash.py` | AF2 heavy-atom clash. |
| `backbone_continuity.py` | AF2 C(i)–N(i+1) bond geometry, helix-only gated. |
| `helix_dihedral.py` | Canonical α-helix φ/ψ target. |
| `packing.py` | Heavy-atom neighbor-count prior on interface stub & ring atoms. |
| `dunbrack.py` | Joint vMM rotamer prior, π-periodic-robust evaluation. |
| `_dunbrack_library.npz` | Offline-fitted rotamer library (μ, κ, π) per (ss_class, residue). |
