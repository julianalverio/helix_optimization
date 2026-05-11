from __future__ import annotations

import torch
import torch.nn as nn

from .rotation import rotation_from_6d


class FrameOutputHead(nn.Module):
    """Predicts per-residue rigid frames (R, t). The 9 output floats split into
    a 6D rotation parameterization (Zhou et al. 2019) and a 3D translation
    delta; Gram-Schmidt converts the 6D part to a 3x3 delta rotation matrix.

    Both R and t are parameterized as residuals on the per-residue conditioning:

      R = conditioning_R @ delta_R
      t = conditioning_translation + delta_t

    where `conditioning_R = rotation_from_6d(conditioning_frame_6d)` for
    partner residues with a valid (N, CA, C) frame and the 3x3 identity for
    helix residues / partner residues with missing backbone atoms. At init
    delta_R = I and delta_t = 0, so partner residues land at their GT
    backbone pose immediately and the helix starts at the canonical-layout
    pose at the origin (R = I, t = 0) and is learned from scratch via the
    coord / geom losses. Sidechain torsions are NOT residualized — chi
    angles are predicted independently by `TorsionHead` for every residue,
    helix and partner alike."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 9)
        # Init so the head emits identity delta-rotation + zero delta-translation
        # for every residue: zero the weights, bias the 6D delta-rotation part
        # to (1,0,0,0,1,0) — Gram-Schmidt of that is the identity matrix — and
        # zero the delta-translation bias. With the conditioning residuals
        # composed in `forward`, partner residues start at their GT backbone pose.
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.zero_()
            self.proj.bias[0] = 1.0
            self.proj.bias[4] = 1.0

    def forward(
        self,
        hidden: torch.Tensor,
        conditioning_translation: torch.Tensor,
        conditioning_frame_6d: torch.Tensor,
        conditioning_frame_6d_validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(self.layer_norm(hidden))
        rot_6d = out[..., :6]
        delta_t = out[..., 6:]
        delta_R = rotation_from_6d(rot_6d)

        # For invalid residues (helix; partner with missing N/CA/C),
        # conditioning_frame_6d is the zero vector. Substitute the 6D identity
        # so `rotation_from_6d` yields I instead of the degenerate zero matrix
        # (Gram-Schmidt of all-zero collapses through the clamp_min) — that
        # also keeps gradients well-conditioned through the unused branch.
        identity_6d = conditioning_frame_6d.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        safe_6d = torch.where(
            conditioning_frame_6d_validity.unsqueeze(-1),
            conditioning_frame_6d,
            identity_6d,
        )
        base_R = rotation_from_6d(safe_6d)
        return base_R @ delta_R, conditioning_translation + delta_t
