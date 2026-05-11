"""Wrapper around MaSIF-site running inside the official Docker image.

Pipeline per PDB:
  1. Stage the input PDB into a per-call scratch dir mounted as `/work`.
  2. In the container:
     - data_prepare_one.sh --file /work/<pdbid>.pdb <PDBID>_<chains>
     - predict_site.sh nn_models.all_feat_3l.custom_params <PDBID>_<chains>
     - copy the .ply mesh + .npy scores into /work/out/
  3. Parse the ASCII PLY (vertices + faces) and the .npy (per-vertex iface scores).

Returns (vertices, faces, scores). Vertex coordinates are in PDB Å in the same
frame as the input PDB. Scores are MaSIF's per-vertex iface_pred ∈ [0, 1].
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .._cache import is_valid, mark, signature


_CONTAINER_SCRIPT = """\
set -euo pipefail
cd /masif/data/masif_site
./data_prepare_one.sh --file /work/{pdbid_lower}.pdb {pdbid_upper}_{chains}
./predict_site.sh {pdbid_upper}_{chains}
mkdir -p /work/out
cp data_preparation/01-benchmark_surfaces/{pdbid_upper}_{chains}.ply /work/out/
cp output/all_feat_3l/pred_data/pred_{pdbid_upper}_{chains}.npy /work/out/
"""


def run_masif_site(
    pdb_path: Path,
    pdb_id: str,
    chains: str,
    scratch_dir: Path,
    image: str,
    platform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MaSIF-site in Docker on `pdb_path` for the chain set `chains`
    (concatenated single-letter chain IDs, e.g. "AB"). Returns (vertices,
    faces, scores) parsed from the container outputs."""
    pdbid_lower = pdb_id.lower()
    pdbid_upper = pdb_id.upper()
    scratch_dir = scratch_dir.resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir = scratch_dir / "out"
    out_dir.mkdir(exist_ok=True)

    # Cache hit: outputs exist AND the cache sidecar matches the signature
    # of the current input PDB + chain selection + docker image tag. Any
    # change invalidates the cache.
    cached_ply = out_dir / f"{pdbid_upper}_{chains}.ply"
    cached_npy = out_dir / f"pred_{pdbid_upper}_{chains}.npy"
    sidecar = out_dir / ".cache_sig"
    sig = signature(pdb_path, chains, image)
    if cached_ply.exists() and cached_npy.exists() and is_valid(sidecar, sig):
        return _load_masif_outputs(cached_ply, cached_npy, pdb_id, chains)

    shutil.copy(pdb_path, scratch_dir / f"{pdbid_lower}.pdb")

    script = _CONTAINER_SCRIPT.format(
        pdbid_lower=pdbid_lower, pdbid_upper=pdbid_upper, chains=chains,
    )
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--platform", platform,
            "-v", f"{scratch_dir}:/work",
            image,
            "bash", "-lc", script,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MaSIF docker failed for {pdb_id}_{chains}: exit={result.returncode}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}\n"
            f"--- stdout ---\n{result.stdout[-2000:]}"
        )

    mark(sidecar, sig)
    return _load_masif_outputs(cached_ply, cached_npy, pdb_id, chains)


def _load_masif_outputs(
    ply_path: Path, npy_path: Path, pdb_id: str, chains: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces = read_ascii_ply(ply_path)
    scores_obj = np.load(npy_path, allow_pickle=True)
    # masif_site_predict.py saves `scores` from run_masif_site, which is a
    # length-1 array-of-arrays whose [0] entry is the per-vertex prediction.
    scores = np.asarray(scores_obj[0], dtype=np.float64).reshape(-1)
    if scores.shape[0] != vertices.shape[0]:
        raise RuntimeError(
            f"MaSIF score/vertex count mismatch for {pdb_id}_{chains}: "
            f"{scores.shape[0]} scores vs {vertices.shape[0]} vertices"
        )
    return vertices, faces, scores


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices (V, 3) float64 and triangle faces (F, 3) int64 from an
    ASCII PLY file. Ignores all extra vertex/face attributes."""
    n_vertex = 0
    n_face = 0
    vertex_props: list[str] = []
    in_header = True

    with path.open() as f:
        # Header.
        while in_header:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF in PLY header: {path}")
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format" and tokens[1] != "ascii":
                raise ValueError(f"non-ASCII PLY not supported: {path}")
            if tokens[0] == "element":
                if tokens[1] == "vertex":
                    n_vertex = int(tokens[2])
                    current_element = "vertex"
                elif tokens[1] == "face":
                    n_face = int(tokens[2])
                    current_element = "face"
                else:
                    current_element = tokens[1]
            elif tokens[0] == "property" and current_element == "vertex":
                vertex_props.append(tokens[-1])
            elif tokens[0] == "end_header":
                in_header = False

        if vertex_props[:3] != ["x", "y", "z"]:
            raise ValueError(f"PLY vertex header missing leading x/y/z: {vertex_props}")

        vertices = np.empty((n_vertex, 3), dtype=np.float64)
        for i in range(n_vertex):
            parts = f.readline().split()
            vertices[i, 0] = float(parts[0])
            vertices[i, 1] = float(parts[1])
            vertices[i, 2] = float(parts[2])

        faces = np.empty((n_face, 3), dtype=np.int64)
        for i in range(n_face):
            parts = f.readline().split()
            # Each face line is "k i0 i1 ... i(k-1)"; MaSIF writes triangles only.
            if int(parts[0]) != 3:
                raise ValueError(f"non-triangle face at row {i} in {path}")
            faces[i, 0] = int(parts[1])
            faces[i, 1] = int(parts[2])
            faces[i, 2] = int(parts[3])

    return vertices, faces
