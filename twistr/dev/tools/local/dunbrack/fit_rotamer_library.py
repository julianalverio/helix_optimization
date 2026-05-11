"""Fit a per-(residue, ss_class) **joint** mixture of von Mises rotamer
library from the Shapovalov & Dunbrack 2011 backbone-dependent dataset.

Differences from the previous (factorised) version:

  * **Joint** distribution over (χ_1, …, χ_n) rather than independent
    1D vMMs per χ. Captures χ–χ correlations (e.g., LEU's χ1=−60°
    coexists with χ2=180°, almost never with other χ2s).
  * **Quality cutoff**: drop rows with `RSPERC < 25` per the dataset
    README's developer recommendation; for ASN/GLN/HIS additionally
    require `FLP_CONFID == "clear"` (Asn/Gln/His amide-flip confidence).
  * **π-periodic augmentation**: AF2 marks ASP χ2, GLU χ3, PHE χ2,
    TYR χ2 as 2-fold rotationally symmetric in
    `alphafold/common/residue_constants.py:chi_pi_periodic`. For these
    χs we emit an additional sample with χ + π so the fit is
    guaranteed symmetric regardless of which heavy-atom labelling each
    PDB structure happens to have used.

Per-residue K (number of mixture components):
  K = min(3 ** n_chi, 36).
The 3-per-χ canonical rotamer count (g+, g−, t) is the natural upper
bound; we cap at 36 so even ARG (n_chi=4 → 81) stays tractable. Init at
the canonical Cartesian product; EM refines or collapses unsupported
components to small mixture weight.

Output: `twistr/pipeline/losses/_dunbrack_library.npz` with arrays
    mu        (n_ss=2, 20, K_max=36, 4)  μ_k_i in radians (radians; 0 in unused χ slots)
    kappa     (n_ss=2, 20, K_max=36)     concentration κ_k (1.0 in unused k slots)
    log_pi    (n_ss=2, 20, K_max=36)     log mixture weight (-inf in unused k slots)
    n_chi     (20,)                      number of χ angles per residue type
    K_used    (n_ss=2, 20)               number of components actually fit
    has_data  (n_ss=2, 20)               True iff fit succeeded
    ss_classes
    residue_names

ss_class order: ('helix', 'general'), matching SS_CLASSES below.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
from scipy.special import i0

from twistr.tensors.constants import RESIDUE_TYPE_NAMES

SS_CLASSES = ("helix", "general")
N_CHI_MAX = 4
K_MAX = 36
MIN_SAMPLES = 50

_DUNBRACK_TO_OUR = {
    "ALA": "ALA", "ARG": "ARG", "ASN": "ASN", "ASP": "ASP",
    "CYS": "CYS", "CYD": "CYS", "CYH": "CYS",
    "GLN": "GLN", "GLU": "GLU", "GLY": "GLY",
    "HIS": "HIS", "ILE": "ILE", "LEU": "LEU", "LYS": "LYS",
    "MET": "MET", "PHE": "PHE",
    "PRO": "PRO", "CPR": "PRO", "TPR": "PRO",
    "SER": "SER", "THR": "THR", "TRP": "TRP", "TYR": "TYR", "VAL": "VAL",
}

_AMIDE_FLIP_RESIDUES = {"ASN", "GLN", "HIS"}

# AF2's chi_pi_periodic, AST-extracted from the alphafold submodule.
_AF2_RESIDUE_CONSTANTS_PATH = (
    Path(__file__).resolve().parents[5]
    / "twistr" / "external" / "alphafold" / "alphafold" / "common"
    / "residue_constants.py"
)


def _ast_extract(name: str):
    tree = ast.parse(_AF2_RESIDUE_CONSTANTS_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found in {_AF2_RESIDUE_CONSTANTS_PATH}")


# AF2 restypes in 3-letter form, in the order the chi_pi_periodic /
# chi_angles_mask tables are indexed by. AF2's `restypes` is a list of
# 1-letter codes; map to 3-letter via `restype_1to3`.
_AF2_RESTYPES = _ast_extract("restypes")
_AF2_RESTYPE_1TO3 = _ast_extract("restype_1to3")
_AF2_CHI_ANGLES_MASK = _ast_extract("chi_angles_mask")
_AF2_CHI_PI_PERIODIC = _ast_extract("chi_pi_periodic")


def _build_residue_tables():
    """Per-residue n_chi and (4,)-shape pi-periodic mask, indexed by
    RESIDUE_TYPE_NAMES order."""
    n_chi = np.zeros(len(RESIDUE_TYPE_NAMES), dtype=np.int32)
    pi_periodic = np.zeros((len(RESIDUE_TYPE_NAMES), N_CHI_MAX), dtype=bool)
    for i, name in enumerate(RESIDUE_TYPE_NAMES):
        # AF2's tables are indexed by 1-letter code position in `restypes`.
        af2_idx = next(
            j for j, c in enumerate(_AF2_RESTYPES)
            if _AF2_RESTYPE_1TO3[c] == name
        )
        mask = _AF2_CHI_ANGLES_MASK[af2_idx]
        n_chi[i] = int(sum(mask))
        for c in range(N_CHI_MAX):
            pi_periodic[i, c] = bool(_AF2_CHI_PI_PERIODIC[af2_idx][c])
    return n_chi, pi_periodic


def _wrap_pi(theta_rad: np.ndarray) -> np.ndarray:
    return ((theta_rad + np.pi) % (2 * np.pi)) - np.pi


def parse_dataset(path: Path):
    """Yield (residue_3letter, ss_central_char, [chi1..chi4_rad_or_nan])
    per row that survives the quality filter."""
    n_total = 0
    n_kept = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 16:
                continue
            n_total += 1
            res = _DUNBRACK_TO_OUR.get(parts[0])
            if res is None:
                continue
            try:
                rsperc = float(parts[14])
            except ValueError:
                continue
            if rsperc < 25.0:
                continue
            if res in _AMIDE_FLIP_RESIDUES and parts[17].strip() != "clear":
                continue
            ss = parts[15]
            ss_central = ss[1] if len(ss) >= 2 else "?"
            chis = []
            for chi_str in parts[8:12]:
                if chi_str.lower() == "nan":
                    chis.append(np.nan)
                else:
                    chis.append(_wrap_pi(np.radians(float(chi_str))))
            n_kept += 1
            yield res, ss_central, chis
    print(f"quality-filtered: {n_kept:,} / {n_total:,} rows kept "
          f"({100 * n_kept / max(n_total, 1):.1f}%)")


def _augment_pi_periodic(samples: np.ndarray, periodic_mask: np.ndarray) -> np.ndarray:
    """For each row in `samples` (shape (M, n_chi)), for every χ that's
    π-periodic emit an additional row with χ_i + π (other χs unchanged
    so χ–χ correlations are preserved on both copies). Both rows
    survive in the output."""
    if not periodic_mask.any():
        return samples
    out = [samples]
    for c in np.where(periodic_mask)[0]:
        flipped = samples.copy()
        flipped[:, c] = _wrap_pi(flipped[:, c] + np.pi)
        out.append(flipped)
    return np.concatenate(out, axis=0)


def _canonical_init_centres(K: int, n_chi: int) -> np.ndarray:
    """Initial μ for K components in an n-D circular space. Generates
    the Cartesian product of {-60°, +60°, 180°} per χ, then keeps the
    first K (or pads with random if K > 3^n)."""
    base = np.deg2rad(np.array([-60.0, 60.0, 180.0]))
    grid = np.array(np.meshgrid(*([base] * n_chi), indexing="ij")).reshape(n_chi, -1).T
    if grid.shape[0] >= K:
        return grid[:K]
    rng = np.random.default_rng(0)
    extra = (rng.random((K - grid.shape[0], n_chi)) - 0.5) * 2 * np.pi
    return np.concatenate([grid, extra], axis=0)


def fit_joint_vmm_em(
    samples: np.ndarray,                # (M, n_chi)
    K: int,
    max_iter: int = 100,
    tol: float = 1e-5,
):
    """EM for a K-component joint von Mises mixture in n_chi dimensions
    with isotropic concentration κ_k shared across the n_chi axes. Each
    component has its own n_chi-D mean μ_k. Returns (μ, κ, π) of shape
    (K, n_chi), (K,), (K,)."""
    M, n_chi = samples.shape
    if M < MIN_SAMPLES:
        return None

    mu = _canonical_init_centres(K, n_chi)
    kappa = np.full(K, 2.0)
    pi = np.full(K, 1.0 / K)

    log_2pi = np.log(2 * np.pi)
    prev_ll = -np.inf
    for _ in range(max_iter):
        # log Z_k = n_chi · (log(2π) + log I_0(κ_k))
        log_norm = n_chi * (log_2pi + np.log(i0(kappa) + 1e-300))            # (K,)
        diff = samples[:, None, :] - mu[None, :, :]                          # (M, K, n_chi)
        cos_sum = np.cos(diff).sum(axis=-1)                                  # (M, K)
        log_p = kappa * cos_sum - log_norm                                    # (M, K)
        log_joint = log_p + np.log(pi + 1e-12)
        log_total = np.logaddexp.reduce(log_joint, axis=-1)                  # (M,)
        ll = log_total.sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
        resp = np.exp(log_joint - log_total[:, None])                        # (M, K)

        Nk = resp.sum(axis=0)                                                # (K,)
        # Per-axis circular mean of the responsibilities.
        sin_sum = (resp[:, :, None] * np.sin(samples)[:, None, :]).sum(axis=0)
        cos_sum_axis = (resp[:, :, None] * np.cos(samples)[:, None, :]).sum(axis=0)
        mu = np.arctan2(sin_sum, cos_sum_axis)                               # (K, n_chi)
        # Isotropic kappa from the average resultant length across axes.
        R_axis = np.sqrt(sin_sum**2 + cos_sum_axis**2) / np.maximum(Nk[:, None], 1e-9)
        R = R_axis.mean(axis=-1)                                             # (K,)
        kappa = R * (2 - R**2) / np.maximum(1 - R**2, 1e-9)
        kappa = np.clip(kappa, 1e-3, 1e3)
        pi = np.maximum(Nk / Nk.sum(), 1e-12)

    return mu, kappa, pi


def main(args: argparse.Namespace) -> None:
    n_chi_table, pi_periodic_table = _build_residue_tables()
    n_ss = len(SS_CLASSES)
    n_res = len(RESIDUE_TYPE_NAMES)

    res_idx_lookup = {r: i for i, r in enumerate(RESIDUE_TYPE_NAMES)}
    helix_idx = SS_CLASSES.index("helix")
    general_idx = SS_CLASSES.index("general")

    # Buckets per (ss_class, residue): list of chi tuples (length n_chi[r]).
    buckets: dict[tuple[int, int], list[list[float]]] = {
        (s, r): [] for s in range(n_ss) for r in range(n_res)
    }

    for res, ss_central, chis in parse_dataset(Path(args.dataset)):
        ridx = res_idx_lookup.get(res)
        if ridx is None:
            continue
        n_c = int(n_chi_table[ridx])
        if n_c == 0:
            continue
        sample = chis[:n_c]
        if any(np.isnan(x) for x in sample):
            continue
        is_helix = ss_central == "H"
        buckets[(general_idx, ridx)].append(sample)
        if is_helix:
            buckets[(helix_idx, ridx)].append(sample)

    mu_out = np.zeros((n_ss, n_res, K_MAX, N_CHI_MAX), dtype=np.float32)
    kappa_out = np.ones((n_ss, n_res, K_MAX), dtype=np.float32)
    log_pi_out = np.full((n_ss, n_res, K_MAX), -np.inf, dtype=np.float32)
    has_data = np.zeros((n_ss, n_res), dtype=bool)
    K_used = np.zeros((n_ss, n_res), dtype=np.int32)

    for (s, r), rows in buckets.items():
        if len(rows) < MIN_SAMPLES:
            continue
        n_c = int(n_chi_table[r])
        samples = np.asarray(rows, dtype=np.float64)                          # (M, n_c)
        # π-periodic augmentation BEFORE the fit so the resulting vMM is
        # guaranteed-symmetric regardless of dataset bias.
        samples = _augment_pi_periodic(samples, pi_periodic_table[r, :n_c])
        K = min(3 ** n_c, K_MAX)
        result = fit_joint_vmm_em(samples, K=K)
        if result is None:
            continue
        mu, kappa, pi = result
        mu_out[s, r, :K, :n_c] = mu.astype(np.float32)
        kappa_out[s, r, :K] = kappa.astype(np.float32)
        log_pi_out[s, r, :K] = np.log(pi + 1e-12).astype(np.float32)
        has_data[s, r] = True
        K_used[s, r] = K
        # Print top-3 components by mixture weight for sanity.
        top = np.argsort(-pi)[:3]
        top_summary = ", ".join(
            f"k{k} π={pi[k]:.3f} μ_deg=[{','.join(f'{np.degrees(mu[k,c]):+5.0f}' for c in range(n_c))}] κ={kappa[k]:.1f}"
            for k in top
        )
        print(f"  {SS_CLASSES[s]:>7s} {RESIDUE_TYPE_NAMES[r]} (n={len(rows):>6d}, K={K}): {top_summary}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mu=mu_out, kappa=kappa_out, log_pi=log_pi_out,
        n_chi=n_chi_table, K_used=K_used, has_data=has_data,
        # Per-residue π-periodicity mask (ASP χ2, GLU χ3, PHE χ2, TYR χ2 — see
        # AF2 chi_pi_periodic). Used by the loss for evaluation-time
        # symmetrisation that exact-invariants χ → χ+π for those axes.
        chi_pi_periodic=pi_periodic_table,
        ss_classes=np.array(SS_CLASSES),
        residue_names=np.array(RESIDUE_TYPE_NAMES),
        K_max=np.int32(K_MAX),
    )
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="runtime/data/dunbrack/DatasetForBBDepRL2010.txt")
    ap.add_argument(
        "--output",
        default="twistr/pipeline/losses/_dunbrack_library.npz",
        help="In-package by default so training pods get the library via "
             "the source upload (data/ is excluded).",
    )
    main(ap.parse_args())
