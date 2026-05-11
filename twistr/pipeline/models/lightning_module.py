from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import grad_norm

from twistr.pipeline.config import MLConfig
from twistr.pipeline.constants import COORD_SCALE_ANGSTROMS
from twistr.pipeline.features.builder import build_features
from twistr.pipeline.features.chi_angles import chi_mask
from twistr.pipeline.features.residue_type import one_hot_residue_type
from twistr.pipeline.losses.backbone_continuity import backbone_continuity_loss
from twistr.pipeline.losses.coord_mse import coord_mse_loss
from twistr.pipeline.losses.dunbrack import dunbrack_rotamer_loss
from twistr.pipeline.losses.helix_dihedral import helix_dihedral_loss
from twistr.pipeline.losses.interaction_bce import interaction_bce_loss
from twistr.pipeline.losses.interaction_geometry import interaction_geometry_losses
from twistr.pipeline.losses.packing import packing_neighbor_loss
from twistr.pipeline.losses.steric_clash import steric_clash_loss

from .architecture import HelixDesignModel
from .sidechain import apply_torsions_to_atom14


class ExamplesModule(pl.LightningModule):
    """LightningModule wrapping the HelixDesignModel. Builds features,
    runs the Pairformer-based model, and produces a dict with the predicted
    interaction matrix and per-residue structure."""

    def __init__(self, cfg: MLConfig | None = None):
        super().__init__()
        self.cfg = cfg or MLConfig()
        self.save_hyperparameters({"cfg": self.cfg.__dict__})
        self.model = HelixDesignModel(self.cfg)

    def _ramped_weight(
        self, start: float, end: float, steps: int, training: bool,
    ) -> float:
        """Linear ramp from `start` at step 0 to `end` at step `steps`, then
        constant at `end`. Validation is pinned to `max(start, end)` (the
        strongest weight in the schedule) so val totals stay comparable
        across the whole run. `steps <= 0` means no ramp — returns `end`."""
        if not training:
            return max(start, end)
        if steps <= 0:
            return end
        progress = min(self.global_step / steps, 1.0)
        return start + progress * (end - start)

    def _compute_losses(
        self, batch: dict[str, torch.Tensor], training: bool,
    ) -> dict[str, torch.Tensor]:
        features = build_features(batch, self.cfg)
        out = self.model(features)
        return self._losses_from_out(batch, features, out, training)

    def _losses_from_out(
        self,
        batch: dict[str, torch.Tensor],
        features: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        atoms_atom14 = apply_torsions_to_atom14(
            out["rotation"], out["translation"], out["torsion_sincos"], batch["residue_type"],
        )
        loss_helix = self.cfg.helix_dihedral_weight * helix_dihedral_loss(
            atoms_atom14, batch["is_helix"], batch["padding_mask"],
        )
        loss_bce = self.cfg.interaction_bce_weight * interaction_bce_loss(
            out["interaction_matrix"],
            features["target_interaction_matrix"],
            batch["padding_mask"],
            label_smoothing=self.cfg.interaction_label_smoothing,
        )
        geom = interaction_geometry_losses(
            atoms_atom14,
            batch["residue_type"],
            batch["atom_mask"],
            features["target_interaction_matrix"],
            batch["padding_mask"],
        )
        loss_clash = self.cfg.clash_loss_weight * steric_clash_loss(
            atoms_atom14 * COORD_SCALE_ANGSTROMS,
            batch["atom_mask"],
            batch["residue_type"],
        )
        loss_backbone_continuity = self.cfg.backbone_continuity_weight * backbone_continuity_loss(
            atoms_atom14 * COORD_SCALE_ANGSTROMS,
            batch["atom_mask"],
            batch["residue_type"],
            batch["chain_slot"],
            batch["is_helix"],
            batch["padding_mask"],
        )
        is_helix = batch["is_helix"].bool()
        is_interface = batch["is_interface_residue"].bool()
        real = batch["padding_mask"]
        chi_mask_t = chi_mask(batch["residue_type"])
        loss_dunbrack_interacting = self.cfg.dunbrack_weight * dunbrack_rotamer_loss(
            out["torsion_sincos"],
            batch["residue_type"],
            batch["is_helix"],
            chi_mask_t,
            real & is_interface,
        )
        loss_dunbrack_non_interacting = self.cfg.dunbrack_weight * dunbrack_rotamer_loss(
            out["torsion_sincos"],
            batch["residue_type"],
            batch["is_helix"],
            chi_mask_t,
            real & ~is_interface,
        )
        device = atoms_atom14.device
        backbone_slots = torch.zeros(14, dtype=torch.bool, device=device)
        backbone_slots[:4] = True                                                    # N, CA, C, O
        sidechain_slots = ~backbone_slots                                            # 4..13
        # Antigen backbone: every non-helix real residue, slots 0-3.
        antigen_backbone_inclusion = (
            ((~is_helix) & real).unsqueeze(-1) & backbone_slots
        )                                                                            # (B, N, 14)
        # Annealed: every interface helix real residue (all 14 slots) ∪ every
        # non-helix interface-residue real residue (sidechain slots only).
        helix_inclusion = (is_helix & is_interface & real).unsqueeze(-1)             # (B, N, 1) → broadcasts to all 14
        non_helix_iface_sidechain_inclusion = (
            ((~is_helix) & is_interface & real).unsqueeze(-1) & sidechain_slots
        )
        annealed_inclusion = (
            helix_inclusion.expand(-1, -1, 14) | non_helix_iface_sidechain_inclusion
        )
        coord_mse_antigen_backbone = coord_mse_loss(
            atoms_atom14, batch["coordinates"], batch["atom_mask"], antigen_backbone_inclusion,
        )
        coord_mse_annealed = coord_mse_loss(
            atoms_atom14, batch["coordinates"], batch["atom_mask"], annealed_inclusion,
        )
        loss_packing = packing_neighbor_loss(
            atoms_atom14 * COORD_SCALE_ANGSTROMS,
            batch["residue_type"],
            batch["atom_mask"],
            batch["is_interface_residue"],
            batch["is_helix"],
            batch["padding_mask"],
            n_target=self.cfg.packing_loss_n_target,
            d_lo=self.cfg.packing_loss_d_lo,
            d_hi=self.cfg.packing_loss_d_hi,
            tau=self.cfg.packing_loss_tau,
        )
        ramp_steps = self.cfg.geometric_handoff_ramp_steps
        vdw_w = self._ramped_weight(
            self.cfg.vdw_loss_start_weight, self.cfg.vdw_loss_weight, ramp_steps, training,
        )
        hbond_w = self._ramped_weight(
            self.cfg.hbond_loss_start_weight, self.cfg.hbond_loss_weight, ramp_steps, training,
        )
        pd_w = self._ramped_weight(
            self.cfg.parallel_displaced_loss_start_weight,
            self.cfg.parallel_displaced_loss_weight, ramp_steps, training,
        )
        sandwich_w = self._ramped_weight(
            self.cfg.sandwich_loss_start_weight, self.cfg.sandwich_loss_weight, ramp_steps, training,
        )
        t_shaped_w = self._ramped_weight(
            self.cfg.t_shaped_loss_start_weight, self.cfg.t_shaped_loss_weight, ramp_steps, training,
        )
        mse_annealed_w = self._ramped_weight(
            self.cfg.coord_mse_annealed_start_weight,
            self.cfg.coord_mse_annealed_floor_weight,
            self.cfg.coord_mse_annealed_steps, training,
        )
        packing_w = self._ramped_weight(
            self.cfg.packing_loss_start_weight,
            self.cfg.packing_loss_weight, ramp_steps, training,
        )
        return {
            "helix": loss_helix,
            "bce": loss_bce,
            "vdw": vdw_w * geom["vdw"],
            "hbond": hbond_w * geom["hbond"],
            "parallel_displaced": pd_w * geom["parallel_displaced"],
            "sandwich": sandwich_w * geom["sandwich"],
            "t_shaped": t_shaped_w * geom["t_shaped"],
            "clash": loss_clash,
            "backbone_continuity": loss_backbone_continuity,
            "dunbrack_interacting": loss_dunbrack_interacting,
            "dunbrack_non_interacting": loss_dunbrack_non_interacting,
            "coord_mse_antigen_backbone": self.cfg.coord_mse_antigen_backbone_weight * coord_mse_antigen_backbone,
            "coord_mse_annealed": mse_annealed_w * coord_mse_annealed,
            "packing": packing_w * loss_packing,
        }

    def _log_losses(self, losses: dict[str, torch.Tensor], prefix: str, batch_size: int) -> torch.Tensor:
        on_step = prefix == "train"
        loss_total = sum(losses.values())
        for name, value in losses.items():
            self.log(f"{prefix}/loss_{name}", value, on_step=on_step, on_epoch=not on_step, batch_size=batch_size)
        self.log(f"{prefix}/loss_total", loss_total, on_step=on_step, on_epoch=not on_step, batch_size=batch_size)
        return loss_total

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        losses = self._compute_losses(batch, training=True)
        loss_total = self._log_losses(losses, "train", batch["is_helix"].shape[0])
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss_total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = build_features(batch, self.cfg)
        out = self.model(features)
        losses = self._losses_from_out(batch, features, out, training=False)
        bs = batch["is_helix"].shape[0]
        loss_total = self._log_losses(losses, "val", bs)
        metrics = self._mutation_sensitivity_metrics(
            batch, features, out["interaction_matrix"], batch_idx,
        )
        for name, value in metrics.items():
            self.log(
                f"mutation_sensitivity/{name}", value,
                on_step=False, on_epoch=True, batch_size=bs,
            )
        return loss_total

    def _mutation_sensitivity_metrics(
        self,
        batch: dict[str, torch.Tensor],
        wt_features: dict[str, torch.Tensor],
        wt_logits: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """For each example, sample `cfg.mutation_metric_k` random single-residue
        helix substitutions and measure how much the predicted interaction-
        matrix probabilities shift relative to WT. Three numbers per val step:

          im_total:  mean |Δp| over real off-diagonal pairs × 6 channels.
          im_local:  same, restricted to pairs where at least one residue is
                     on the same chain as the mutation site AND within
                     `cfg.mutation_metric_locality_radius` residues of it
                     (by index). Same-chain gating matters because the
                     batch concatenates helix and antigen along N — raw
                     index distance crosses the chain boundary otherwise.
          im_far:    same, restricted to pairs where neither residue
                     satisfies the local condition.

        The WT features (in particular `conditioning_interaction_matrix`) are
        held fixed across mutants — only `residue_type_one_hot` and `chi_mask`
        are overridden — so the metric isolates the model's sensitivity to the
        residue-type input rather than to the cascading effect of recomputing
        the conditioning IM detector under the mutated identity.
        """
        K = self.cfg.mutation_metric_k
        radius = self.cfg.mutation_metric_locality_radius
        residue_type = batch["residue_type"]
        B, N = residue_type.shape
        device = residue_type.device

        wt_probs = torch.sigmoid(wt_logits)                                       # (B, N, N, 6)

        padding_mask = batch["padding_mask"]
        eye = torch.eye(N, dtype=torch.bool, device=device)
        real_pair = (
            padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2) & ~eye
        )                                                                          # (B, N, N)

        chain_slot = batch["chain_slot"]                                           # (B, N) long
        candidates = batch["is_helix"].bool() & padding_mask                       # (B, N)
        has_helix = candidates.any(dim=-1)                                         # (B,)

        # Deterministic CPU generator. global_step ticks across val passes;
        # batch_idx varies within a pass; k indexes the K mutations.
        seed = int(self.global_step) * 1_000_003 + int(batch_idx)
        g = torch.Generator(device="cpu").manual_seed(seed)

        res_idx = torch.arange(N, device=device)

        sum_total = torch.zeros((), device=device)
        sum_local = torch.zeros((), device=device)
        sum_far = torch.zeros((), device=device)
        cnt_total = torch.zeros((), device=device)
        cnt_local = torch.zeros((), device=device)
        cnt_far = torch.zeros((), device=device)

        for _ in range(K):
            mut_residue_type = residue_type.clone()
            positions = torch.zeros(B, dtype=torch.long, device=device)
            valid_mut = torch.zeros(B, dtype=torch.bool, device=device)
            for b in range(B):
                if not bool(has_helix[b]):
                    continue
                cand = torch.nonzero(candidates[b], as_tuple=False).flatten()
                pos = int(cand[torch.randint(cand.numel(), (1,), generator=g).item()].item())
                current_aa = int(residue_type[b, pos].item())
                new_aa = int(torch.randint(20, (1,), generator=g).item())
                while new_aa == current_aa:
                    new_aa = int(torch.randint(20, (1,), generator=g).item())
                mut_residue_type[b, pos] = new_aa
                positions[b] = pos
                valid_mut[b] = True

            if not bool(valid_mut.any()):
                continue

            mut_features = dict(wt_features)
            mut_features["residue_type_one_hot"] = one_hot_residue_type(mut_residue_type)
            mut_features["chi_mask"] = chi_mask(mut_residue_type)

            mut_logits = self.model(mut_features)["interaction_matrix"]
            mut_probs = torch.sigmoid(mut_logits)

            diff = (mut_probs - wt_probs).abs().sum(dim=-1)                        # (B, N, N), summed over 6 channels

            mut_chain = chain_slot.gather(1, positions.unsqueeze(1)).squeeze(1)    # (B,)
            same_chain = chain_slot == mut_chain.unsqueeze(1)                      # (B, N)
            i_dist = (res_idx.unsqueeze(0) - positions.unsqueeze(1)).abs()         # (B, N)
            near = same_chain & (i_dist <= radius)                                 # (B, N)
            near_pair = near.unsqueeze(-1) | near.unsqueeze(-2)                    # (B, N, N)

            valid_e = valid_mut.view(B, 1, 1)
            cell_total = real_pair & valid_e
            cell_local = cell_total & near_pair
            cell_far = cell_total & ~near_pair

            sum_total = sum_total + (diff * cell_total.float()).sum()
            sum_local = sum_local + (diff * cell_local.float()).sum()
            sum_far = sum_far + (diff * cell_far.float()).sum()
            cnt_total = cnt_total + cell_total.sum().float() * 6
            cnt_local = cnt_local + cell_local.sum().float() * 6
            cnt_far = cnt_far + cell_far.sum().float() * 6

        return {
            "im_total": sum_total / cnt_total.clamp_min(1.0),
            "im_local": sum_local / cnt_local.clamp_min(1.0),
            "im_far": sum_far / cnt_far.clamp_min(1.0),
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        norms = grad_norm(self.model, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"], on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        warmup = max(self.cfg.lr_warmup_steps, 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
