"""Dunbrack 2010 backbone-dependent rotamer-plausibility loss.

Scores predicted χ angles against an empirical **joint** mixture-of-von-
Mises density fitted from the Shapovalov & Dunbrack 2011 dataset.

Two libraries indexed by `ss_class`:

  * 'helix'   — fitted on rows where the central residue's SS == 'H'
  * 'general' — fitted on every row regardless of SS

For each (ss_class, residue_type), we use one joint vMM in n_χ
dimensions (n_χ ranges 1–4 across residue types). Each component has
its own n_χ-D mean μ_k and a single isotropic concentration κ_k:

    p_k(χ_1, …, χ_n) = exp(κ_k · Σ_i cos(χ_i − μ_k_i)) / (2π · I_0(κ_k))^n

The joint formulation captures χ–χ correlations that the previous
factorised version missed. Library produced offline by
`tools/dunbrack/fit_rotamer_library.py` and persisted to
`twistr/ml/losses/_dunbrack_library.npz`.

For a given predicted residue, dispatch to the helix library if
`is_helix[i]` is True, else to the general library — per the project's
helix-vs-antigen split. Antigen residues use the general library.

Robust handling of:

  * **π-periodic χs** (ASP χ2, GLU χ3, PHE χ2, TYR χ2): two-layer
    handling so the loss is *exactly* invariant under χ → χ+π for
    those axes regardless of dataset bias.
      1. Offline-fitter augmentation: each row with a π-periodic χ
         value is duplicated with χ+π before EM, so the fitted
         mixture is approximately symmetric.
      2. Eval-time symmetrisation: for any residue type with a
         π-periodic χ_c, the loss evaluates the joint density at the
         original χ AND at χ with χ_c+π, then averages. This removes
         the residual asymmetry from EM convergence noise and
         guarantees `loss(χ) ≡ loss(χ_c → χ_c+π)`.
  * **invalid χ slots** (e.g., SER has only χ1): residue-specific
    n_χ controls how many χ axes contribute to the cosine sum and
    the normalising constant. The library carries `n_chi[r]` for each
    residue.
  * **unused mixture components** (`k ≥ K_used`): library carries
    `log_pi = -inf` there, but to avoid `0 × ∞ → NaN` in the backward
    pass we substitute a finite uniform `log_pi` for invalid
    components in the forward; the components are then masked out of
    the logsumexp via the `valid_k` mask described below.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

# In-package copy is the source of truth at training time (data/ is
# excluded from the launcher's source upload). The data/ location remains
# a fallback for local dev runs.
_PACKAGE_LIBRARY = Path(__file__).with_name("_dunbrack_library.npz")
_DATA_LIBRARY = (
    Path(__file__).resolve().parents[3]
    / "data" / "dunbrack" / "_dunbrack_library.npz"
)
_LIBRARY_PATH = _PACKAGE_LIBRARY if _PACKAGE_LIBRARY.exists() else _DATA_LIBRARY

_HELIX_INDEX = 0
_GENERAL_INDEX = 1
_N_CHI_MAX = 4

# Lazy load — keyed by (device, dtype) so first call on a GPU dispatches buffers there once.
_CACHE: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}


def _load_library(device: torch.device, dtype: torch.dtype):
    key = (str(device), dtype)
    if key in _CACHE:
        return _CACHE[key]
    if not _LIBRARY_PATH.exists():
        raise RuntimeError(
            f"Dunbrack rotamer library not found at {_LIBRARY_PATH}. "
            "Build it once with: "
            "uv run python tools/dunbrack/fit_rotamer_library.py"
        )
    raw = np.load(_LIBRARY_PATH)
    mu = torch.from_numpy(raw["mu"]).to(device=device, dtype=dtype)             # (n_ss, 20, K, n_chi_max)
    kappa = torch.from_numpy(raw["kappa"]).to(device=device, dtype=dtype)       # (n_ss, 20, K)
    log_pi = torch.from_numpy(raw["log_pi"]).to(device=device, dtype=dtype)     # (n_ss, 20, K)
    n_chi = torch.from_numpy(raw["n_chi"]).to(device=device, dtype=torch.long)  # (20,)
    K_used = torch.from_numpy(raw["K_used"]).to(device=device, dtype=torch.long)# (n_ss, 20)
    has_data = torch.from_numpy(raw["has_data"]).to(device=device, dtype=torch.bool)  # (n_ss, 20)
    chi_pi_periodic = torch.from_numpy(raw["chi_pi_periodic"]).to(             # (20, n_chi_max)
        device=device, dtype=torch.bool,
    )
    K = mu.shape[2]
    # `valid_k[s, r, k]`: is component `k` actually fit (k < K_used[s, r])?
    valid_k = torch.arange(K, device=device).view(1, 1, K) < K_used.unsqueeze(-1)
    # log Z_k = n_chi · log(2π · I_0(κ_k)) — but n_chi varies per residue, so
    # store the per-component "unit" log normaliser log(2π · I_0(κ_k)) and
    # multiply by n_chi at evaluation time.
    from scipy.special import i0
    unit_log_norm = np.log(
        2 * np.pi * i0(raw["kappa"].astype(np.float64)) + 1e-300,
    )
    unit_log_norm_t = torch.from_numpy(unit_log_norm).to(device=device, dtype=dtype)
    _CACHE[key] = {
        "mu": mu, "kappa": kappa, "log_pi": log_pi,
        "n_chi": n_chi, "valid_k": valid_k,
        "unit_log_norm": unit_log_norm_t, "has_data": has_data,
        "chi_pi_periodic": chi_pi_periodic,
    }
    return _CACHE[key]


def _joint_log_p(
    chi: torch.Tensor,                # (B, N, 4)
    mu_sel: torch.Tensor,             # (B, N, K, 4)
    kappa_sel: torch.Tensor,          # (B, N, K)
    n_chi_per_res: torch.Tensor,      # (B, N)
    chi_axis_mask: torch.Tensor,      # (B, N, 4) float
    unit_log_norm_sel: torch.Tensor,  # (B, N, K)
    safe_log_pi: torch.Tensor,        # (B, N, K)
) -> torch.Tensor:
    """Joint vMM log-density for one (un-symmetrised) χ tensor."""
    cos_diff = torch.cos(chi.unsqueeze(-2) - mu_sel)                                # (B, N, K, 4)
    cos_sum = (cos_diff * chi_axis_mask.unsqueeze(-2)).sum(dim=-1)                  # (B, N, K)
    log_components = (
        kappa_sel * cos_sum
        - n_chi_per_res.unsqueeze(-1).to(chi.dtype) * unit_log_norm_sel
    )                                                                                # (B, N, K)
    return torch.logsumexp(log_components + safe_log_pi, dim=-1)                    # (B, N)


def dunbrack_rotamer_loss(
    torsion_sincos: torch.Tensor,    # (B, N, 7, 2) — model output for ω/φ/ψ/χ1..χ4
    residue_type: torch.Tensor,      # (B, N) long, RESIDUE_TYPE_NAMES order
    is_helix: torch.Tensor,          # (B, N) bool
    chi_mask: torch.Tensor,          # (B, N, 4) — 1 where the residue HAS that χ
    padding_mask: torch.Tensor,      # (B, N) bool, True = real residue
) -> torch.Tensor:
    """Mean-over-real-residues of `-log p(χ | residue, ss_class)`. The
    library's per-residue `n_chi` controls how many χ axes enter the
    joint vMM density."""
    lib = _load_library(torsion_sincos.device, torsion_sincos.dtype)
    mu = lib["mu"]                                                                 # (n_ss, 20, K, n_chi_max)
    kappa = lib["kappa"]                                                           # (n_ss, 20, K)
    log_pi = lib["log_pi"]                                                         # (n_ss, 20, K)
    n_chi_table = lib["n_chi"]                                                     # (20,)
    valid_k = lib["valid_k"]                                                       # (n_ss, 20, K)
    unit_log_norm = lib["unit_log_norm"]                                           # (n_ss, 20, K)
    has_data = lib["has_data"]                                                     # (n_ss, 20)
    chi_pi_periodic = lib["chi_pi_periodic"]                                       # (20, n_chi_max)

    # Predicted χ angles. Slots 3..6 = χ1..χ4 (per architecture.py:12-15).
    chi = torch.atan2(
        torsion_sincos[..., 3:7, 0],
        torsion_sincos[..., 3:7, 1],
    )                                                                              # (B, N, 4)

    # ss_class dispatch.
    ss_idx = torch.where(is_helix.bool(), _HELIX_INDEX, _GENERAL_INDEX)            # (B, N)

    # Gather library entries per (ss, residue) → (B, N, K, n_chi_max), etc.
    mu_sel = mu[ss_idx, residue_type]
    kappa_sel = kappa[ss_idx, residue_type]
    log_pi_sel = log_pi[ss_idx, residue_type]
    valid_k_sel = valid_k[ss_idx, residue_type]
    unit_log_norm_sel = unit_log_norm[ss_idx, residue_type]
    has_data_sel = has_data[ss_idx, residue_type]                                  # (B, N)
    n_chi_per_res = n_chi_table[residue_type]                                      # (B, N)
    pi_periodic_per_res = chi_pi_periodic[residue_type]                            # (B, N, n_chi_max)

    # Per-(B, N) mask over χ axes: True for axis c iff c < n_chi_per_res[b, n].
    arange = torch.arange(_N_CHI_MAX, device=chi.device)                           # (n_chi_max,)
    chi_axis_mask = (arange.view(1, 1, -1) < n_chi_per_res.unsqueeze(-1)).to(chi.dtype)  # (B, N, n_chi_max)

    # Substitute uniform log_pi for invalid components / residues so logsumexp
    # produces finite values; mask out via the residue-level `valid` below.
    K = mu_sel.shape[-2]
    valid_residue = has_data_sel & padding_mask.bool()                             # (B, N)
    safe_log_pi = torch.where(
        valid_k_sel & valid_residue.unsqueeze(-1),
        log_pi_sel,
        torch.full_like(log_pi_sel, -math.log(K)),
    )                                                                              # (B, N, K)

    # Eval-time π-periodic symmetrisation. AF2's `chi_pi_periodic` marks at
    # most one χ axis per residue; we enumerate the 2^n_periodic χ-flip
    # combinations (≤2 in practice) and average their densities. This makes
    # the loss exactly invariant under χ_c → χ_c+π for π-periodic axes,
    # independent of any residual EM convergence asymmetry in the fit.
    log_p_terms = [
        _joint_log_p(chi, mu_sel, kappa_sel, n_chi_per_res, chi_axis_mask,
                    unit_log_norm_sel, safe_log_pi),
    ]
    # Per-residue: which χ axes are π-periodic? Build flipped versions where
    # any periodic axis is flipped by +π, but only on residues that actually
    # have that axis π-periodic. Residues without π-periodic axes get the
    # same loss in both terms (the average is then just the original).
    flip = (pi_periodic_per_res.to(chi.dtype) * math.pi)                           # (B, N, n_chi_max)
    if flip.abs().sum() > 0:
        chi_flipped = chi + flip
        log_p_terms.append(
            _joint_log_p(chi_flipped, mu_sel, kappa_sel, n_chi_per_res,
                        chi_axis_mask, unit_log_norm_sel, safe_log_pi),
        )
        log_p = torch.logsumexp(torch.stack(log_p_terms, dim=0), dim=0) - math.log(2)
    else:
        log_p = log_p_terms[0]
    nll = -log_p

    # Per-example mean over residues with library coverage, then mean over
    # examples that had ≥1 such residue.
    valid_f = valid_residue.to(nll.dtype)
    res_count = valid_f.sum(dim=-1)                                                # (B,)
    per_example = (nll * valid_f).sum(dim=-1) / res_count.clamp_min(1.0)
    has_signal = (res_count > 0).to(per_example.dtype)
    return (per_example * has_signal).sum() / has_signal.sum().clamp_min(1.0)
