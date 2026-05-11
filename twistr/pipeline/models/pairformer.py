"""Pairformer trunk — clean transcription of Protenix's AF3-style Pairformer.

Why we transcribe rather than import: Protenix's pairformer.py reaches
transitively into the rdkit-tainted constants module, an optree-using utils
module, and a JIT-compiled CUDA LayerNorm — none of which we need at our
scale (small N, no MSA, no templates, no chunking). The transcription below
mirrors Protenix's classes 1-to-1 in name and forward structure but drops
the optimization-mode branches (cuequivariance, deepspeed, fused kernels,
inplace_safe paths, gradient checkpointing). Each block carries a
`Source:` line pointing at the Protenix file + line span it was transcribed
from so updates stay traceable.

The block layout follows AlphaFold 3 Algorithm 17. Triangle multiplications
are Algorithms 11 (outgoing) and 12 (incoming); Triangle attentions are
Algorithm 13 (starting) and 14 (ending); pair-biased single attention is
Algorithm 24 (with `has_s=False`); transitions are gated SiLU FFNs."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    """Gated SiLU FFN. Source: protenix/model/modules/primitives.py:166-205."""

    def __init__(self, c_in: int, n: int = 2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)
        self.linear_a = nn.Linear(c_in, n * c_in, bias=False)
        self.linear_b = nn.Linear(c_in, n * c_in, bias=False)
        self.linear_out = nn.Linear(n * c_in, c_in, bias=False)
        nn.init.zeros_(self.linear_out.weight)  # zero-init last linear (AF3 convention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.linear_out(F.silu(self.linear_a(x)) * self.linear_b(x))


class DropoutRowwise(nn.Module):
    """Dropout where the mask is shared along the second-to-last spatial dim
    (i.e. the same (j, c) is dropped/kept across every row i). Matches
    Protenix's `DropoutRowwise` (batch_dim=-3). Identity in eval mode."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        shape = list(x.shape)
        shape[-3] = 1
        mask = x.new_ones(shape)
        mask = F.dropout(mask, p=self.p, training=True)
        return x * mask


class TriangleMultiplication(nn.Module):
    """Algorithms 11 & 12. Source: protenix/model/triangular/triangular.py:172-570
    (the `triangle_multiplicative == "torch"` path, with the fp16 std-norm
    branch dropped)."""

    def __init__(self, c_z: int, c_hidden: int, outgoing: bool):
        super().__init__()
        self.outgoing = outgoing
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_hidden)
        self.linear_a_p = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = nn.Linear(c_z, c_hidden, bias=False)
        self.linear_z = nn.Linear(c_hidden, c_z, bias=False)
        self.linear_g = nn.Linear(c_z, c_z, bias=False)
        nn.init.zeros_(self.linear_z.weight)
        nn.init.zeros_(self.linear_g.weight)

    def forward(self, z: torch.Tensor, pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pair_mask is not None:
            mask = pair_mask.unsqueeze(-1).to(z.dtype)
        else:
            mask = z.new_ones(*z.shape[:-1], 1)
        z_n = self.layer_norm_in(z)
        a = mask * torch.sigmoid(self.linear_a_g(z_n)) * self.linear_a_p(z_n)
        b = mask * torch.sigmoid(self.linear_b_g(z_n)) * self.linear_b_p(z_n)
        if self.outgoing:
            x = torch.einsum("...ikc,...jkc->...ijc", a, b)                     # x_ij = Σ_k a_ik b_jk
        else:
            x = torch.einsum("...kic,...kjc->...ijc", a, b)                     # x_ij = Σ_k a_ki b_kj
        x = self.linear_z(self.layer_norm_out(x))
        g = torch.sigmoid(self.linear_g(z_n))
        return x * g


class TriangleAttention(nn.Module):
    """Algorithms 13 & 14. Source: protenix/model/triangular/triangular.py:589-715
    (torch path). Query at pair (i, j) attends over k along row i, biased by
    a (j, k)-derived term. `starting=False` → transpose first."""

    def __init__(self, c_z: int, c_hidden: int, n_heads: int, starting: bool):
        super().__init__()
        self.starting = starting
        self.n_heads = n_heads
        self.c_hidden = c_hidden
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, n_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_z, n_heads * c_hidden, bias=False)
        self.linear_v = nn.Linear(c_z, n_heads * c_hidden, bias=False)
        self.linear_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_g = nn.Linear(c_z, n_heads * c_hidden, bias=False)
        self.linear_o = nn.Linear(n_heads * c_hidden, c_z)
        nn.init.zeros_(self.linear_o.weight)
        nn.init.zeros_(self.linear_o.bias)

    def forward(self, z: torch.Tensor, pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.starting:
            z = z.transpose(-2, -3)
            if pair_mask is not None:
                pair_mask = pair_mask.transpose(-1, -2)
        z_n = self.layer_norm(z)
        *batch, N, M, _ = z_n.shape
        H, D = self.n_heads, self.c_hidden

        q = self.linear_q(z_n).view(*batch, N, M, H, D)
        k = self.linear_k(z_n).view(*batch, N, M, H, D)
        v = self.linear_v(z_n).view(*batch, N, M, H, D)
        bias = self.linear_bias(z_n)                                            # (..., N, M, H)

        # Pre-softmax logits: (..., N, M_q, M_k, H)  with M_q = M_k = M.
        scale = D ** -0.5
        logits = torch.einsum("...injhd,...inkhd->...injkh", q, k) * scale
        logits = logits + bias.unsqueeze(-4)                                    # broadcast bias[(j, k)] across i
        if pair_mask is not None:
            mask = pair_mask.unsqueeze(-2).unsqueeze(-1)                        # (..., N, 1, M, 1)
            logits = logits.masked_fill(~mask.bool(), -1e9)
        attn = logits.softmax(dim=-2)
        out = torch.einsum("...injkh,...inkhd->...injhd", attn, v)
        out = out.reshape(*batch, N, M, H * D)
        out = out * torch.sigmoid(self.linear_g(z_n))
        out = self.linear_o(out)
        if not self.starting:
            out = out.transpose(-2, -3)
        return out


class PairBiasAttention(nn.Module):
    """Single-rep attention biased by the pair representation. Source:
    protenix/model/modules/transformer.py:40-254 (AttentionPairBias with
    has_s=False, standard_multihead_attention path)."""

    def __init__(self, c_s: int, c_z: int, n_heads: int):
        super().__init__()
        assert c_s % n_heads == 0
        self.n_heads = n_heads
        self.c_head = c_s // n_heads
        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_qkv = nn.Linear(c_s, 3 * c_s)
        self.linear_g = nn.Linear(c_s, c_s, bias=False)
        self.linear_z_bias = nn.Linear(c_z, n_heads, bias=False)
        self.linear_o = nn.Linear(c_s, c_s)
        nn.init.zeros_(self.linear_o.weight)
        nn.init.zeros_(self.linear_o.bias)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_n = self.layer_norm_s(s)
        *batch, N, _ = s_n.shape
        H, D = self.n_heads, self.c_head
        qkv = self.linear_qkv(s_n).view(*batch, N, 3, H, D)
        q, k, v = qkv.unbind(dim=-3)                                            # each (..., N, H, D)

        bias = self.linear_z_bias(self.layer_norm_z(z))                         # (..., N, N, H)
        bias = bias.permute(*range(bias.dim() - 3), -1, -3, -2)                 # (..., H, N, N)

        scale = D ** -0.5
        # Use SDPA when available (mask + bias).
        attn_mask = bias
        if pair_mask is not None:
            pm = pair_mask.unsqueeze(-3)                                        # (..., 1, N, N)
            attn_mask = bias.masked_fill(~pm.bool(), -1e9)
        # SDPA expects q/k/v as (..., H, L, D).
        q_h = q.transpose(-2, -3)
        k_h = k.transpose(-2, -3)
        v_h = v.transpose(-2, -3)
        out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask, scale=scale)
        out = out.transpose(-2, -3).reshape(*batch, N, H * D)
        out = out * torch.sigmoid(self.linear_g(s_n))
        return self.linear_o(out)


class PairformerBlock(nn.Module):
    """One Pairformer block. AF3 Algorithm 17 lines 2-8. Source:
    protenix/model/modules/pairformer.py:42-224 (we keep the `inplace_safe=False`
    path; rowwise-dropout-and-add is broken into explicit `+= dropout(...)`
    for readability)."""

    def __init__(
        self,
        c_s: int = 128,
        c_z: int = 64,
        n_heads_single: int = 4,
        n_heads_pair: int = 4,
        c_hidden_mul: int = 32,
        c_hidden_pair_att: int = 16,
        transition_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(c_z, c_hidden_mul, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(c_z, c_hidden_mul, outgoing=False)
        self.tri_att_start = TriangleAttention(c_z, c_hidden_pair_att, n_heads_pair, starting=True)
        self.tri_att_end = TriangleAttention(c_z, c_hidden_pair_att, n_heads_pair, starting=False)
        self.dropout_row = DropoutRowwise(dropout)
        self.pair_transition = Transition(c_z, transition_n)
        self.attention_pair_bias = PairBiasAttention(c_s, c_z, n_heads_single)
        self.single_transition = Transition(c_s, transition_n)

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = z + self.dropout_row(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout_row(self.tri_mul_in(z, pair_mask))
        z = z + self.dropout_row(self.tri_att_start(z, pair_mask))
        z = z + self.dropout_row(self.tri_att_end(z, pair_mask))
        z = z + self.pair_transition(z)
        s = s + self.attention_pair_bias(s, z, pair_mask)
        s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    """Stack of identical Pairformer blocks. Source:
    protenix/model/modules/pairformer.py:227-340 (we drop the
    `checkpoint_blocks` wrapper — gradient checkpointing is a non-default
    optimization we don't need at our scale)."""

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 128,
        c_z: int = 64,
        n_heads_single: int = 4,
        n_heads_pair: int = 4,
        c_hidden_mul: int = 32,
        c_hidden_pair_att: int = 16,
        transition_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            PairformerBlock(
                c_s=c_s, c_z=c_z,
                n_heads_single=n_heads_single, n_heads_pair=n_heads_pair,
                c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att,
                transition_n=transition_n, dropout=dropout,
            )
            for _ in range(n_blocks)
        )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, pair_mask)
        return s, z
