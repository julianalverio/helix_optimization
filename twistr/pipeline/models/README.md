# `twistr.pipeline.models` — model architecture & Lightning wrapper

End-to-end network plus the LightningModule that drives training. The
forward pass is **one-shot** — no recycling, no diffusion — by design:
the agent calls this model on every proposed point mutation, and an
8× recycle or a 50-step diffusion sample is not practical at agent-loop
frequency.

## Forward pass

```
features  →  InputEmbedder  →  Pairformer trunk  →  three heads
                (s, z)            (s, z)              { (R, t),
                                                        χ-sincos,
                                                        IM logits }
                                                          ↓
                                              apply_torsions_to_atom14
                                                          ↓
                                                  atom-14 coordinates
```

## Input embedder (with a load-bearing AF3 detail)

Per-residue features (residue-type one-hot, chi masks, per-residue
conditioning fields) project to the single representation `s` of
dimension `c_s`. Pair features (the 8-channel noised conditioning IM +
relative-position one-hot in AF-Multimer's same-chain / different-chain
encoding) project to the pair representation `z` of dimension `c_z`.

The pair init then **adds** row and column projections of `s` into `z`,
matching AlphaFold 3's `InputFeatureEmbedder` (Algorithm 2). This
coupling is non-obvious but load-bearing for the agent's primary use
case: without it, `z` is a pure function of the conditioning IM and the
relative-position one-hot. Because the Pairformer trunk has no s→z
update path of its own, residue-type information never reaches `z` for
the rest of the forward pass. We caught this experimentally — the
validation mutation-sensitivity diagnostic was identically zero, so the
predicted interaction matrix was insensitive to residue substitutions,
which would make the model useless for the agent. The row and column
projections are biased separately so the s-to-z contribution is
asymmetric at init and symmetry breaks immediately.

## Pairformer trunk

Clean transcription of Protenix's open-source AF3 implementation
following AlphaFold 3 Algorithm 17. Each block carries a `Source:`
comment pointing at the original Protenix line span for traceability.

Compared to the source, the transcription drops optimization branches
that buy nothing at our scale: cuequivariance, DeepSpeed, fused-CUDA
LayerNorm, gradient checkpointing, the chunking infrastructure,
`inplace_safe` paths. Reason for transcribing rather than importing:
Protenix's `pairformer.py` transitively pulls in rdkit-tainted
constants modules, an optree-using utils module, and a JIT-compiled
CUDA LayerNorm — none of which we need at N ≤ 200 and no MSA / no
templates.

Block contents are standard AF3:

- Triangle multiplications outgoing (Alg. 11) and incoming (Alg. 12)
- Triangle attentions starting (Alg. 13) and ending (Alg. 14)
- Pair-biased single attention (Alg. 24 with `has_s = False`)
- Gated SiLU transitions (n = 2 by default)
- Row-wise and column-wise dropout sharing masks along the spatial
  dimension (matches Protenix's `DropoutRowwise` / `DropoutColumnwise`)

## Output heads

**Frame head.** Emits 9 values per residue: 6 for the continuous 6D
rotation parameterization (Zhou et al. 2019, Gram-Schmidt to a 3×3
matrix) plus 3 for a translation delta. Both R and t are **residuals on
the per-residue conditioning frame**:

```
R = conditioning_R @ delta_R
t = conditioning_translation + delta_t
```

The output bias is initialized so `delta_R = I` (the Gram-Schmidt of
`(1, 0, 0, 0, 1, 0)` is the identity matrix) and `delta_t = 0`; the
weight matrix is zero-initialized. At step 0, partner residues with
valid frames land exactly at their GT backbone pose, and helix /
missing-backbone residues land at the canonical-layout pose at the
origin (R = I, t = 0). This is critical for the loss-handoff strategy:
with R, t correct on the partner side at init, the geometric losses
see real protein geometry there from step 0 and only have to teach the
helix side from scratch.

**Torsion head.** Seven (sin, cos) pairs per residue in AF2 rigid-group
ordering: [ω, φ, ψ, χ₁..χ₄]. Each pair is L2-normalized so it lies on
the unit circle. atom-14 placement uses χ₁..χ₄ (sidechain) and ψ
(carbonyl O via AF2's psi rigid group); φ and ω have no atom-14 atoms
in their groups (hydrogens only) and are emitted for future backbone-
torsion losses.

**Interaction-matrix head.** Symmetrizes `z` (`0.5 · (z + z.transpose)`),
layer-norms, projects to 6 logits per pair. The output bias is
hard-coded to the **logit of each channel's prior positive rate**:
`(-3, -3, -4, -4, -4, +3)` for vdw / hbond / pd / sandwich / t_shaped /
none. Real interfaces are sparse — most pairs are "none", and per-
channel positive rates for VDW, h-bond, and the three stacking types
are O(10⁻²) or lower. Starting at sigmoid(0) = 0.5 would force the head
to spend its first ~1k steps learning the per-channel prior; hard-coded
biases get there at step 0.

## Atom-14 placement

`sidechain.py` composes AF2's canonical "atom-14 layout at chi=0" with
the predicted (R, t, χ₁..χ₄). AF2's group constants
(`restype_rigid_group_default_frame`,
`restype_atom14_rigid_group_positions`) are extracted via AST from the
AlphaFold submodule's `residue_constants.py` — same pattern as
`features/chi_angles.py`. We never `import alphafold` (which would
pull in JAX) and the canonical values are never copy-pasted into this
repo.

Known follow-up: AF2's `chi_pi_periodic` (ASP χ₂, GLU χ₃, PHE χ₂,
TYR χ₂, HIS χ₂, plus ARG's NH1↔NH2 atom-level swap) is not handled in
sidechain placement. Current losses either reduce by min/max over atom
alternatives (VDW, h-bond, aromatic stacking) or are explicitly
symmetrized under the π flip (Dunbrack), so they are naturally
invariant. A direct chi-regression or per-atom FAPE loss would need to
add it.

## LightningModule

`ExamplesModule` wraps the model for Lightning. Beyond the standard
train/val step + multi-loss reduction, two points worth flagging:

- **LR warmup.** Linear ramp from 0 to `learning_rate` over
  `lr_warmup_steps`. Several last-layer weights are zero-init (frame
  head, IM head projection), so the first few hundred steps see very
  asymmetric gradient flow; without warmup, Adam at full lr diverges.
- **Mutation-sensitivity diagnostic.** For each val example, sample K
  random single-residue helix substitutions, hold every other input
  fixed (including the conditioning IM, so the metric isolates the
  model's sensitivity to the residue-type input rather than to a
  cascading recomputation of the conditioning), and measure mean
  |Δp_predicted_IM| relative to wild-type. Split into "local" — pairs
  where at least one residue is on the same chain as the mutation site
  AND within a configurable index radius — and "far" (all other real
  pairs). Same-chain gating is important because the batch concatenates
  helix and antigen along N; raw index distance crosses the chain
  boundary otherwise. `local > far` is the signal we want — sequence
  sensitivity that respects locality. This diagnostic surfaced the
  missing s→z coupling described above.

## File index

| File | Role |
|---|---|
| `architecture.py` | `HelixDesignModel` end-to-end; `InputEmbedder`; `InteractionMatrixHead`; `TorsionHead`. |
| `pairformer.py` | AF3 Algorithm 17 trunk (transitions, triangle mults, triangle attentions, pair-biased single attention, dropout). |
| `output_head.py` | `FrameOutputHead` — 6D rotation + translation, residual on conditioning, identity-init. |
| `rotation.py` | 6D ↔ 3×3 conversion (Zhou et al. 2019); `frame_from_three_points` in AF2 convention. |
| `sidechain.py` | atom-14 placement from (R, t, χ_sincos) using AST-extracted AF2 rigid-group constants. |
| `lightning_module.py` | `ExamplesModule` — train/val step, multi-loss reduction, LR warmup, grad-norm logging, mutation-sensitivity diagnostic. |
