"""Cluster training-example helix sequences by length-normalized Levenshtein
distance and write a per-example sidecar parquet with cluster ids and inverse-
cluster-size sampling weights.

Clustering is complete-linkage hierarchical at an automatically selected
threshold (silhouette over a τ-grid of normalized edit distances). Because the
unique-sequence count is far above the in-memory full-pairwise threshold, we
first build a sparse candidate graph (pairs with d_norm ≤ τ_max via chunked
rapidfuzz cdist), then compute connected components and run dense complete-
linkage clustering inside each component. Pairs absent from the candidate
graph have d > τ_max and never merge under complete linkage at any τ ≤ τ_max,
so this is exact, not an approximation.

Run: `python -m twistr.pipeline.datasets.cluster_helices`.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.cluster import hierarchy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("runtime/data/examples/module3_manifest.parquet")
DEFAULT_OUTPUT = Path("runtime/data/examples/helix_clusters.parquet")
DEFAULT_TAU_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
MAX_DENSE_COMPONENT = 30_000


def _build_candidate_graph(
    seqs: np.ndarray,
    lens: np.ndarray,
    tau_max: float,
    chunk: int = 1024,
    workers: int = -1,
) -> coo_matrix:
    """Sparse upper-triangular raw-Levenshtein matrix of pairs with
    d_norm ≤ tau_max."""
    n = len(seqs)
    raw_cutoff = int(np.ceil(tau_max * lens.max()))
    seqs_list = seqs.tolist()
    rows_all, cols_all, data_all = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        D = process.cdist(
            seqs_list[start:stop], seqs_list,
            scorer=Levenshtein.distance,
            dtype=np.uint8,
            score_cutoff=raw_cutoff,
            workers=workers,
        )
        # Restrict to upper triangle: column index strictly greater than the
        # absolute query index.
        rs, cs = np.where(D <= raw_cutoff)
        i_abs = rs + start
        upper = cs > i_abs
        rs, cs, i_abs = rs[upper], cs[upper], i_abs[upper]
        if rs.size == 0:
            continue
        ds = D[rs, cs].astype(np.float32)
        lmax = np.maximum(lens[i_abs], lens[cs]).astype(np.float32)
        keep = (ds / lmax) <= tau_max
        rows_all.append(i_abs[keep])
        cols_all.append(cs[keep])
        data_all.append(ds[keep])
        if (start // chunk) % 10 == 0:
            logger.info(
                "    chunk %d/%d: running pairs=%d",
                start // chunk + 1, (n + chunk - 1) // chunk,
                sum(len(a) for a in rows_all),
            )
    if not rows_all:
        return coo_matrix((n, n), dtype=np.float32)
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)
    return coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)


def _component_dense_distance(
    seqs_sub: list[str], lens_sub: np.ndarray
) -> np.ndarray:
    """Symmetric, zero-diagonal length-normalized distance matrix for a small
    component."""
    D_raw = process.cdist(
        seqs_sub, seqs_sub, scorer=Levenshtein.distance, dtype=np.uint16
    )
    Lmax = np.maximum(lens_sub[:, None], lens_sub[None, :]).astype(np.float32)
    D = D_raw.astype(np.float32) / Lmax
    np.fill_diagonal(D, 0.0)
    return (D + D.T) * 0.5


def _silhouette_component(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette within one component. Returns 0 if a single cluster."""
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            continue
        a = float(D[i, own].mean())
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            d = float(D[i, other].mean())
            if d < b:
                b = d
        denom = max(a, b)
        s[i] = (b - a) / denom if denom > 0 else 0.0
    return float(s.mean())


def cluster_helices(
    manifest_path: Path, output_path: Path, tau_grid: tuple[float, ...]
) -> None:
    logger.info("loading %s", manifest_path)
    df = pd.read_parquet(
        manifest_path, columns=["example_id_full", "path_example", "helix_sequence"]
    )
    logger.info("  %d examples", len(df))

    counts = df.groupby("helix_sequence").size().sort_values(ascending=False)
    seqs = np.asarray(counts.index, dtype=object)
    seq_n_examples = counts.values.astype(np.int64)
    lens = np.asarray([len(s) for s in seqs], dtype=np.int32)
    n_uniq = len(seqs)
    logger.info(
        "  %d unique helix_sequences (lengths %d..%d, mean %.1f)",
        n_uniq, lens.min(), lens.max(), lens.mean(),
    )

    tau_max = max(tau_grid)
    logger.info("building candidate-pair graph (tau_max=%.2f)", tau_max)
    t0 = time.perf_counter()
    coo = _build_candidate_graph(seqs, lens, tau_max)
    logger.info("  %d candidate pairs in %.1fs", coo.nnz, time.perf_counter() - t0)

    adj = csr_matrix(
        (np.ones(coo.nnz, dtype=np.int8), (coo.row, coo.col)),
        shape=(n_uniq, n_uniq),
    )
    n_comp, comp = connected_components(adj, directed=False)
    comp_sizes = np.bincount(comp, minlength=n_comp)
    largest = int(comp_sizes.max())
    logger.info(
        "  %d connected components; size: max=%d, top10=%s, singletons=%d",
        n_comp, largest, np.sort(comp_sizes)[::-1][:10].tolist(),
        int((comp_sizes == 1).sum()),
    )
    if largest > MAX_DENSE_COMPONENT:
        raise RuntimeError(
            f"largest component has {largest} sequences (limit "
            f"{MAX_DENSE_COMPONENT}); reduce tau_max in --tau-grid or use a "
            f"different algorithm"
        )

    per_tau_labels = [np.zeros(n_uniq, dtype=np.int64) for _ in tau_grid]
    next_cid = [1] * len(tau_grid)
    per_tau_silhouette = [
        [] for _ in tau_grid
    ]  # list of (n_points, silhouette) for components with ≥2 points

    logger.info(
        "clustering each component at τ ∈ %s", [f"{t:.2f}" for t in tau_grid],
    )
    for c_id in range(n_comp):
        idx = np.where(comp == c_id)[0]
        m = len(idx)
        if m == 1:
            i = idx[0]
            for ti in range(len(tau_grid)):
                per_tau_labels[ti][i] = next_cid[ti]
                next_cid[ti] += 1
            continue
        sub_seqs = seqs[idx].tolist()
        sub_lens = lens[idx]
        D_sub = _component_dense_distance(sub_seqs, sub_lens)
        Z = hierarchy.linkage(squareform(D_sub, checks=False), method="complete")
        for ti, tau in enumerate(tau_grid):
            local = hierarchy.fcluster(Z, t=tau, criterion="distance")
            offset = next_cid[ti] - 1
            per_tau_labels[ti][idx] = local + offset
            next_cid[ti] += int(local.max())
            sil = _silhouette_component(D_sub, local)
            per_tau_silhouette[ti].append((m, sil))

    logger.info("threshold sweep:")
    best_ti, best_score = 0, -np.inf
    for ti, tau in enumerate(tau_grid):
        contribs = [
            (sz, sil) for sz, sil in per_tau_silhouette[ti] if sz > 1
        ]
        if not contribs:
            score = 0.0
        else:
            sizes = np.array([sz for sz, _ in contribs], dtype=np.float64)
            sils = np.array([sil for _, sil in contribs], dtype=np.float64)
            score = float((sizes * sils).sum() / sizes.sum())
        n_clusters = next_cid[ti] - 1
        if score > best_score:
            best_score = score
            best_ti = ti
        logger.info(
            "  τ=%.2f: n_clusters=%d, silhouette=%+.4f", tau, n_clusters, score,
        )
    chosen_tau = tau_grid[best_ti]
    logger.info(
        "chosen τ=%.2f (silhouette=%+.4f, %d clusters)",
        chosen_tau, best_score, next_cid[best_ti] - 1,
    )
    chosen_labels = per_tau_labels[best_ti]

    seq_to_cluster = pd.Series(chosen_labels, index=seqs)
    seq_to_count = pd.Series(seq_n_examples, index=seqs)
    cluster_to_n = (
        pd.DataFrame({"cid": chosen_labels, "n": seq_n_examples})
        .groupby("cid")["n"].sum()
    )

    out = pd.DataFrame(
        {
            "example_id_full": df["example_id_full"].values,
            "path_example": df["path_example"].values,
            "helix_sequence": df["helix_sequence"].values,
        }
    )
    out["cluster_id"] = out["helix_sequence"].map(seq_to_cluster).astype(np.int64)
    out["n_examples_in_cluster"] = (
        out["cluster_id"].map(cluster_to_n).astype(np.int64)
    )
    out["weight"] = (1.0 / out["n_examples_in_cluster"]).astype(np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    logger.info("wrote %s (%d rows)", output_path, len(out))

    sizes = cluster_to_n.values.astype(np.int64)
    pct = np.percentile(sizes, [50, 75, 90, 99])
    logger.info(
        "cluster size: min=%d, max=%d, mean=%.1f, p50=%.0f, p75=%.0f, "
        "p90=%.0f, p99=%.0f, singletons=%d",
        sizes.min(), sizes.max(), sizes.mean(),
        pct[0], pct[1], pct[2], pct[3], int((sizes == 1).sum()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--tau-grid",
        type=str,
        default=",".join(f"{x:.2f}" for x in DEFAULT_TAU_GRID),
        help="Comma-separated normalized-edit-distance thresholds to sweep.",
    )
    args = p.parse_args()
    tau_grid = tuple(sorted(float(x) for x in args.tau_grid.split(",")))
    cluster_helices(args.manifest, args.output, tau_grid)


if __name__ == "__main__":
    main()
