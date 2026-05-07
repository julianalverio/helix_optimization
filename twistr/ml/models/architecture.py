"""Full model: input embedder → Pairformer trunk → output heads.

Inputs come from `twistr.ml.features.builder.build_features` — see that file
for the per-residue and pair feature dimensions. Outputs are a dict with:
  - "interaction_matrix": (B, N, N, 6) per-channel logits (apply sigmoid for
    independent probabilities; the BCE loss consumes logits directly via
    `binary_cross_entropy_with_logits` for numerical stability)
  - "rotation": (B, N, 3, 3) backbone frame R = conditioning_R @ delta_R,
    where conditioning_R = rotation_from_6d(conditioning_frame_6d) for
    partner residues with a valid GT frame and the 3x3 identity otherwise
  - "translation": (B, N, 3) CA position = conditioning_translation + delta_t
  - "torsion_sincos": (B, N, 7, 2) sin/cos pairs for AF2's seven torsion
    angles per residue [omega, phi, psi, chi_1, chi_2, chi_3, chi_4],
    L2-normalized per (residue, torsion)

The trunk is a Pairformer stack with hyperparameters configured via MLConfig.
Defaults are scaled for our regime: helix-mediated interfaces with N≤200 and
a lead-optimization task that's closer to denoising than de-novo prediction."""
from __future__ import annotations

import torch
import torch.nn as nn

from twistr.ml.config import MLConfig
from twistr.ml.features.residue_type import NUM_RESIDUE_TYPES

from .output_head import FrameOutputHead
from .pairformer import PairformerStack


class InputEmbedder(nn.Module):
    """Concatenates per-residue and pair features and projects to the
    Pairformer's (s, z) channel dims."""

    def __init__(self, c_s: int, c_z: int, relpos_max_offset: int = 32):
        super().__init__()
        self.relpos_max_offset = relpos_max_offset
        # 2k+1 same-chain buckets (clamped Δresidue) + 1 different-chain bucket (AF-Multimer-style).
        self.relpos_buckets = 2 * relpos_max_offset + 2

        # Per-residue: 20 (residue_type) + 4 (chi_mask) + 1 (cond_mask) +
        # 3 (cond_translation) + 1 (cond_translation_validity) +
        # 6 (cond_frame_6d) + 1 (cond_frame_6d_validity) +
        # 8 (cond_chi_sincos flat) + 4 (cond_chi_validity) = 48
        single_in = NUM_RESIDUE_TYPES + 4 + 1 + 3 + 1 + 6 + 1 + 8 + 4
        self.single_proj = nn.Linear(single_in, c_s)

        # Pair: 8 (6 interaction channels + augmentation_mask + padding_mask) + relpos_buckets
        pair_in = 8 + self.relpos_buckets
        self.pair_proj = nn.Linear(pair_in, c_z)

        # AF3 InputFeatureEmbedder Algorithm 2: pair init reads s via
        # z_ij ← z_ij + W_row · s_i + W_col · s_j. Without this, z is a pure
        # function of the conditioning IM and relpos — the Pairformer trunk
        # has no s→z update, so residue-type information never reaches z and
        # the interaction-matrix head's predictions are bit-invariant to
        # sequence changes (mutation sensitivity = 0 forever). The two
        # projections are biased separately so symmetry is broken at init.
        self.s_to_z_row = nn.Linear(c_s, c_z, bias=False)
        self.s_to_z_col = nn.Linear(c_s, c_z, bias=False)

    def _relative_position_one_hot(self, chain_slot: torch.Tensor) -> torch.Tensor:
        is_partner = chain_slot != 0
        same_chain = is_partner[:, :, None] == is_partner[:, None, :]
        idx = torch.arange(chain_slot.shape[1], device=chain_slot.device)
        delta = (idx[:, None] - idx[None, :]).clamp(
            -self.relpos_max_offset, self.relpos_max_offset
        ) + self.relpos_max_offset
        delta = delta.unsqueeze(0).expand_as(same_chain)
        off_chain = torch.full_like(delta, 2 * self.relpos_max_offset + 1)
        buckets = torch.where(same_chain, delta, off_chain)
        return torch.nn.functional.one_hot(buckets, num_classes=self.relpos_buckets).float()

    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = features["residue_type_one_hot"].shape[:2]

        single = torch.cat([
            features["residue_type_one_hot"],
            features["chi_mask"].float(),
            features["conditioning_mask"].float().unsqueeze(-1),
            features["conditioning_translation"],
            features["conditioning_translation_validity"].float().unsqueeze(-1),
            features["conditioning_frame_6d"],
            features["conditioning_frame_6d_validity"].float().unsqueeze(-1),
            features["conditioning_chi_sincos"].reshape(B, N, 8),
            features["conditioning_chi_validity"].float(),
        ], dim=-1)
        s = self.single_proj(single)

        relpos = self._relative_position_one_hot(features["chain_slot"])
        pair = torch.cat([features["conditioning_interaction_matrix"], relpos], dim=-1)
        z = self.pair_proj(pair)
        z = z + self.s_to_z_row(s).unsqueeze(2) + self.s_to_z_col(s).unsqueeze(1)
        return s, z


class InteractionMatrixHead(nn.Module):
    """Symmetrize z, project to 6 per-channel **logits**. Channels are
    independent binary classifiers (multi-label, not multi-class) — a pair
    forming a hydrogen bond often also fires VDW, so the channels do not sum
    to 1. Channel order matches `clean_interaction_matrix`: [vdw, hbond,
    parallel_displaced, sandwich, t_shaped, none]. Train with
    `binary_cross_entropy_with_logits` per channel; consumers needing
    probabilities should sigmoid the output themselves."""

    # Per-channel bias init = logit of the channel's prior positive rate. Real
    # interfaces are sparse: most pairs are negative for every contact channel
    # and almost all pairs match `none`. Starting at sigmoid(0)=0.5 forces the
    # head to spend its first ~1k steps just learning the prior, so we hard-
    # code it instead. Order: [vdw, hbond, pd, sandwich, t_shaped, none].
    INIT_BIAS = (-3.0, -3.0, -4.0, -4.0, -4.0, 3.0)

    def __init__(self, c_z: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_z)
        self.proj = nn.Linear(c_z, 6)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.copy_(torch.tensor(self.INIT_BIAS))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_sym = 0.5 * (z + z.transpose(-2, -3))
        return self.proj(self.layer_norm(z_sym))


class TorsionHead(nn.Module):
    """s → (N, 7, 2) sin/cos pairs for AF2's seven torsion angles per residue.
    Slot order matches AF2's `restype_rigid_group_default_frame` group ordering:
    [omega, phi, psi, chi_1, chi_2, chi_3, chi_4]. Each (sin, cos) pair is
    L2-normalized to live on the unit circle.

    Atom14 placement uses chi_1..chi_4 (sidechain) and psi (carbonyl O via the
    psi rigid group). Phi and omega have no atom14 atoms in their groups
    (only hydrogens, which we don't model) and are emitted for downstream
    consumers / future losses on backbone torsions."""

    def __init__(self, c_s: int, n_torsions: int = 7):
        super().__init__()
        self.n_torsions = n_torsions
        self.layer_norm = nn.LayerNorm(c_s)
        self.proj = nn.Linear(c_s, 2 * n_torsions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        out = self.proj(self.layer_norm(s)).reshape(*s.shape[:-1], self.n_torsions, 2)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class HelixDesignModel(nn.Module):
    """End-to-end model. Predicts the interaction matrix and per-residue
    structure (CA translation, 6D rotation, 7 torsion angles) in one shot."""

    def __init__(self, cfg: MLConfig):
        super().__init__()
        self.cfg = cfg
        self.input_embedder = InputEmbedder(
            c_s=cfg.c_s, c_z=cfg.c_z, relpos_max_offset=cfg.relpos_max_offset,
        )
        self.trunk = PairformerStack(
            n_blocks=cfg.pairformer_blocks,
            c_s=cfg.c_s, c_z=cfg.c_z,
            n_heads_single=cfg.n_heads_single, n_heads_pair=cfg.n_heads_pair,
            c_hidden_mul=cfg.c_hidden_mul, c_hidden_pair_att=cfg.c_hidden_pair_att,
            transition_n=cfg.transition_n, dropout=cfg.pairformer_dropout,
        )
        self.frame_head = FrameOutputHead(cfg.c_s)
        self.torsion_head = TorsionHead(cfg.c_s)
        self.interaction_head = InteractionMatrixHead(cfg.c_z)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        s, z = self.input_embedder(features)
        padding_mask = features["padding_mask"]                                  # (B, N) bool
        pair_mask = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)      # (B, N, N) bool
        s, z = self.trunk(s, z, pair_mask)
        R, t = self.frame_head(
            s,
            features["conditioning_translation"],
            features["conditioning_frame_6d"],
            features["conditioning_frame_6d_validity"],
        )
        return {
            "interaction_matrix": self.interaction_head(z),
            "rotation": R,
            "translation": t,
            "torsion_sincos": self.torsion_head(s),
        }
