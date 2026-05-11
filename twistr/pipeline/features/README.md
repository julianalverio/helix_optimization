# `twistr.pipeline.features` — input features & supervision targets

Builds the tensors the model consumes and the supervision targets the
loss expects. Two non-obvious decisions sit here: how the model is
*conditioned* on the partner structure plus an agent-supplied contact
prior, and how a single differentiable detector serves both as label
generator and as supervision target so the two never drift.

## Conditioning the predictor

The model is given the partner (antigen) structure as coordinates and
the contact pattern as a soft, noised version of the interaction matrix.
The agent therefore has two distinct levers: it can mutate the helix
sequence (model responds with new geometry + new IM) and it can adjust
the contact prior ("I want a hot-spot here, an aromatic stack there").
Both inputs flow through the same forward pass.

Partner-residue conditioning carries the residue-type one-hot, the CA
translation, the backbone frame in 6D parameterization (Zhou et al.
2019 — continuous everywhere, no singularities under coord-space
supervision), and per-residue χ angles as sin/cos pairs. Each numeric
channel is paired with an explicit `*_validity` flag so partner residues
with missing backbone or sidechain atoms can contribute residue-type
information without polluting the geometric channels with zero
sentinels.

Helix residues are not conditioned on coordinates — the frame head's
residual-on-conditioning starts at the identity rotation + zero
translation for helix residues, so they begin at the canonical-layout
pose at the origin and are pushed into place by the structural losses.

## Interaction-matrix detector as single source of truth

`clean_interaction_matrix(batch)` is a differentiable 6-channel soft
detector over residue pairs:

| Channel | Definition |
|---|---|
| 0 — VDW | Sidechain VDW band over closest heavy-atom pair (backbone excluded so sequence-adjacent residues don't fire from N-O proximity). |
| 1 — h-bond | Distance + X-D-A angle + D-A-Y angle bands, symmetrized over donor↔acceptor direction. |
| 2 — parallel-displaced π | Ring-normal alignment + centroid distance band + parallel-axis offset band. |
| 3 — sandwich π | Same family, tighter distance + offset bands. |
| 4 — T-shaped π | Perpendicular normal + centroid distance band. |
| 5 — none | `1 − max(channels 0..4)`. |

Output is `(B, N, N, 6)`, diagonal forced one-hot "none", whole tensor
backward-differentiable. The same tensor (thresholded at 0.5) serves as:

- the BCE target for the interaction-matrix head,
- the channel selector for the geometric losses (each channel's
  flat-bottom loss fires only on pairs labeled with that channel),
- the base for the conditioning noise pipeline.

Treating it as the single source of truth means BCE labels, geometric-
loss targets, and conditioning input semantics never drift relative to
each other — a category of bug we explicitly designed out.

## Conditioning-IM noise

The model's pair input is the clean target with per-cell stochastic
relaxation:

1. **Per-example bit flips.** Rate `u · max_*_flip_rate`, `u ~ U(0,1)`,
   with separate 0→1 and 1→0 rates so the noise model has independent
   control over false-positive vs false-negative agent prescriptions.
2. **Per-cell Beta sampling** to convert binary to soft probabilities,
   with `Beta(μ·ν, (1−μ)·ν)` parameterization and separate (μ, ν) for
   positive and negative cells. The model trains against partially-
   confident contact priors — which is what the agent provides at
   inference, where some contacts are high-confidence prescriptions and
   others are low-confidence suggestions.
3. **Two auxiliary channels** appended to the 6-channel matrix:
   `augmentation_mask` ("the user did not specify this entry") and
   `padding_mask` ("either residue is padding"). Final shape (B, N, N, 8).

## Chi-angle constants — AST extraction

AF2's `_CHI_ANGLES_ATOMS` table (atom-name tuples per residue defining
each chi dihedral) is extracted from the Protenix submodule's source
via AST at module load. Same pattern in `models/sidechain.py` for AF2's
rigid-group constants. Two reasons: (i) we don't pull in the dependency
trees of upstream packages (rdkit, JAX), and (ii) it's unambiguous about
single source of truth — canonical AF2 / Protenix values are extracted
on every load, never copy-pasted into this repo.

`chi_mask(residue_type)` returns the (B, N, 4) bool of which chi indices
actually exist for each residue type (SER has only χ₁; aliphatics have
0–4). All chi-dependent losses gate on this mask.

## File index

| File | Role |
|---|---|
| `builder.py` | `build_features` — single entry point assembling the feature dict. |
| `conditioning.py` | Partner-residue conditioning: mask, translation, 6D frame, chi sin/cos, per-residue validity flags. |
| `interaction_matrix.py` | Differentiable 6-channel IM detector + band constants; conditioning-noise pipeline; atom-14 tables (VDW radii, h-bond donor/acceptor pairs, aromatic ring atoms, packing-atom set). |
| `chi_angles.py` | AST-extracted chi atom tuples + differentiable `compute_chi_angles` / `chi_sincos` / `chi_mask`. |
| `residue_type.py` | 20-class one-hot encoding. |
