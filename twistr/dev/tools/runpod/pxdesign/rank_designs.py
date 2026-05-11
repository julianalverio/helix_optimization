"""Rank PXDesign designs by helix-mediated interface quality.

Per design, computes 4 metrics against the configured hotspots and
writes a CSV ranked by an equal-weight z-score composite:

  helix_fraction    fraction of binder residues in contact with any
                    hotspot whose DSSP code is H (alpha helix)
  bsa_a2            buried surface area at the A/B interface (Å²)
  angle_deg         angle between binder interface-helix axis and
                    hotspot Cα-PCA axis, in [0°, 90°] (anti-parallel
                    treated as equally good as parallel)
  n_hotspot_contacts  hotspot residues with ≥1 binder heavy atom within
                    5 Å

Usage:
  python -m twistr.dev.tools.runpod.pxdesign.rank_designs \\
      --config runtime/configs/pxdesign.yaml \\
      --inputs runtime/outputs/design_runtime/outputs/3erd_b2_a runtime/outputs/design_runtime/outputs/3erd_b2_b ... \\
      --out runtime/outputs/rankings/face1.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import freesasa
import gemmi
import numpy as np
from scipy.spatial import cKDTree

from twistr.dev.tools.runpod.pxdesign.config import TargetChain, load_pxdesign_config

CONTACT_RADIUS = 5.0  # Å, heavy-atom


@dataclass
class DesignMetrics:
    design_id: str
    file: str
    helix_fraction: float
    bsa_a2: float
    angle_deg: float | None
    n_hotspot_contacts: int


def _heavy_atoms(chain: gemmi.Chain) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz [N,3], auth_seq_num [N]) for non-hydrogen atoms."""
    xyz: list[tuple[float, float, float]] = []
    seq: list[int] = []
    for res in chain:
        for atom in res:
            if atom.element.name == "H":
                continue
            xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
            seq.append(res.seqid.num)
    return np.asarray(xyz, dtype=np.float64), np.asarray(seq, dtype=np.int32)


def _ca_coords(chain: gemmi.Chain, residue_nums: list[int]) -> np.ndarray:
    """Cα coordinates for the given auth_seq numbers, in input order."""
    by_num = {r.seqid.num: r for r in chain}
    out = []
    for n in residue_nums:
        res = by_num.get(n)
        if res is None:
            continue
        ca = res.find_atom("CA", "*")
        if ca is None:
            continue
        out.append((ca.pos.x, ca.pos.y, ca.pos.z))
    return np.asarray(out, dtype=np.float64)


def _pca_axis(coords: np.ndarray) -> np.ndarray:
    """Unit vector along the first principal component of N x 3 points."""
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[0]


def _run_dssp_auth(structure: gemmi.Structure) -> dict[str, dict[int, str]]:
    """Run mkdssp; return {auth_chain: {auth_seq_num: ss_char}}.

    PXDesign's PDBs lack label_seq_id, so we call setup_entities() to
    populate the polymer numbering. DSSP keys results by (label_asym_id,
    label_seq_id); we map label_asym_id back to auth chain via the
    structure's subchains, and label_seq_id back to auth_seq_num by
    polymer-position in each chain.
    """
    structure.setup_entities()
    label_to_auth: dict[str, str] = {}
    polymer_by_label: dict[str, list[int]] = {}
    for chain in structure[0]:
        for sub in chain.subchains():
            sid = sub.subchain_id()
            label_to_auth[sid] = chain.name
            polymer_by_label[sid] = [r.seqid.num for r in sub]

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.cif"
        out_path = Path(td) / "out.cif"
        structure.make_mmcif_document().write_file(str(in_path))
        subprocess.run(
            ["mkdssp", "--output-format", "mmcif", str(in_path), str(out_path)],
            check=True, capture_output=True, timeout=180,
        )
        doc = gemmi.cif.read(str(out_path))
        table = doc.sole_block().find(
            "_dssp_struct_summary.",
            ["label_asym_id", "label_seq_id", "secondary_structure"],
        )
        out: dict[str, dict[int, str]] = {}
        for row in table:
            label_chain, seq_str, ss = row[0], row[1], row[2]
            if seq_str in (".", "?"):
                continue
            auth_chain = label_to_auth.get(label_chain)
            if auth_chain is None:
                continue
            label_seq = int(seq_str)
            polymer = polymer_by_label[label_chain]
            if label_seq < 1 or label_seq > len(polymer):
                continue
            auth_seq = polymer[label_seq - 1]
            out.setdefault(auth_chain, {})[auth_seq] = ss
        return out


def _interface_helix_axis(
    binder_chain: gemmi.Chain,
    contact_residues: set[int],
    ss_by_seq: dict[int, str],
) -> np.ndarray | None:
    """Find the contiguous H run on the binder that overlaps the most
    hotspot-contacting residues; return its Cα PCA axis. None if no
    contacting residue lies in any H run."""
    seq_nums = sorted(r.seqid.num for r in binder_chain)
    runs: list[list[int]] = []
    current: list[int] = []
    for n in seq_nums:
        if ss_by_seq.get(n) == "H":
            current.append(n)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    best_run: list[int] | None = None
    best_overlap = 0
    for run in runs:
        overlap = sum(1 for n in run if n in contact_residues)
        if overlap == 0:
            continue
        if overlap > best_overlap or (overlap == best_overlap and best_run is not None and len(run) > len(best_run)):
            best_overlap = overlap
            best_run = run
    if best_run is None or len(best_run) < 3:
        return None
    coords = _ca_coords(binder_chain, best_run)
    if len(coords) < 3:
        return None
    return _pca_axis(coords)


def _compute_bsa(pdb_path: Path) -> float:
    """BSA(complex) = SASA(A alone) + SASA(B alone) − SASA(complex), Å²."""
    freesasa.setVerbosity(freesasa.silent)
    text = pdb_path.read_text().splitlines(keepends=True)

    def _sasa(lines: list[str]) -> float:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.writelines(lines)
            tmp = f.name
        try:
            return float(freesasa.calc(freesasa.Structure(tmp)).totalArea())
        finally:
            os.unlink(tmp)

    keep_a = [ln for ln in text if not (ln.startswith(("ATOM", "HETATM")) and ln[21] == "B")]
    keep_b = [ln for ln in text if not (ln.startswith(("ATOM", "HETATM")) and ln[21] == "A")]
    return _sasa(keep_a) + _sasa(keep_b) - _sasa(text)


def _score_design(
    pdb_path: Path,
    hotspots: list[int],
    target_chain: str,
    binder_chain: str,
) -> DesignMetrics:
    structure = gemmi.read_structure(str(pdb_path))
    model = structure[0]
    target = model[target_chain]
    binder = model[binder_chain]

    binder_xyz, binder_seq = _heavy_atoms(binder)
    target_xyz, target_seq = _heavy_atoms(target)
    binder_tree = cKDTree(binder_xyz)

    contacted_hotspots: set[int] = set()
    contacting_binder: set[int] = set()
    target_seq_to_xyz_idx: dict[int, np.ndarray] = {}
    for hs in hotspots:
        idxs = np.where(target_seq == hs)[0]
        if len(idxs) == 0:
            continue
        target_seq_to_xyz_idx[hs] = idxs
        for hit_list in binder_tree.query_ball_point(target_xyz[idxs], r=CONTACT_RADIUS):
            if hit_list:
                contacted_hotspots.add(hs)
                for j in hit_list:
                    contacting_binder.add(int(binder_seq[j]))

    ss_all = _run_dssp_auth(structure)
    ss_binder = ss_all.get(binder_chain, {})
    if contacting_binder:
        helix_count = sum(1 for n in contacting_binder if ss_binder.get(n) == "H")
        helix_fraction = helix_count / len(contacting_binder)
    else:
        helix_fraction = 0.0

    binder_axis = _interface_helix_axis(binder, contacting_binder, ss_binder)
    if binder_axis is not None and len(hotspots) >= 2:
        hotspot_ca = _ca_coords(target, hotspots)
        if len(hotspot_ca) >= 2:
            target_axis = _pca_axis(hotspot_ca)
            cos_abs = abs(float(np.dot(binder_axis, target_axis)))
            cos_abs = min(1.0, max(0.0, cos_abs))
            angle_deg: float | None = math.degrees(math.acos(cos_abs))
        else:
            angle_deg = None
    else:
        angle_deg = None

    bsa = _compute_bsa(pdb_path)

    rel = pdb_path.relative_to(pdb_path.parents[5]) if len(pdb_path.parents) >= 6 else pdb_path
    parts = pdb_path.parts
    seed_part = next((p for p in parts if p.startswith("seed_")), "seed_?")
    run_part = parts[parts.index("design_outputs") + 1] if "design_outputs" in parts else parts[-6]
    sample_part = pdb_path.stem.replace("pxdesign_target_", "")
    design_id = f"{run_part}/{seed_part}/{sample_part}"

    return DesignMetrics(
        design_id=design_id,
        file=str(rel),
        helix_fraction=helix_fraction,
        bsa_a2=bsa,
        angle_deg=angle_deg,
        n_hotspot_contacts=len(contacted_hotspots),
    )


def _zscore(xs: np.ndarray) -> np.ndarray:
    sd = float(xs.std(ddof=0))
    if sd == 0.0:
        return np.zeros_like(xs)
    return (xs - xs.mean()) / sd


def _rank_and_write(rows: list[DesignMetrics], out_path: Path) -> list[tuple[int, DesignMetrics, float]]:
    n = len(rows)
    helix = np.array([r.helix_fraction for r in rows], dtype=np.float64)
    bsa = np.array([r.bsa_a2 for r in rows], dtype=np.float64)
    contacts = np.array([r.n_hotspot_contacts for r in rows], dtype=np.float64)
    angle = np.array([r.angle_deg if r.angle_deg is not None else np.nan for r in rows], dtype=np.float64)

    z_helix = _zscore(helix)
    z_bsa = _zscore(bsa)
    z_contacts = _zscore(contacts)
    valid_angle = ~np.isnan(angle)
    z_angle = np.full(n, np.nan, dtype=np.float64)
    if valid_angle.any():
        z_angle[valid_angle] = -_zscore(angle[valid_angle])

    composite = np.zeros(n, dtype=np.float64)
    has_helix = np.zeros(n, dtype=bool)
    for i in range(n):
        zs = [z_helix[i], z_bsa[i], z_contacts[i]]
        if not np.isnan(z_angle[i]):
            zs.append(z_angle[i])
            has_helix[i] = True
        composite[i] = float(np.mean(zs))

    order = sorted(
        range(n),
        key=lambda i: (has_helix[i], composite[i]),
        reverse=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "design_id", "file", "has_interface_helix",
            "helix_fraction", "bsa_a2", "angle_deg", "n_hotspot_contacts",
            "z_helix", "z_bsa", "z_angle", "z_contacts", "composite",
        ])
        ranked: list[tuple[int, DesignMetrics, float]] = []
        for rank, i in enumerate(order, start=1):
            r = rows[i]
            w.writerow([
                rank, r.design_id, r.file, bool(has_helix[i]),
                f"{r.helix_fraction:.4f}",
                f"{r.bsa_a2:.1f}",
                "" if r.angle_deg is None else f"{r.angle_deg:.2f}",
                r.n_hotspot_contacts,
                f"{z_helix[i]:.3f}",
                f"{z_bsa[i]:.3f}",
                "" if np.isnan(z_angle[i]) else f"{z_angle[i]:.3f}",
                f"{z_contacts[i]:.3f}",
                f"{composite[i]:.3f}",
            ])
            ranked.append((rank, r, float(composite[i])))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="PXDesign unified config YAML (for hotspots + target chain)")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Design output run directories (e.g. runtime/outputs/design_runtime/outputs/3erd_b2_a)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="B")
    args = parser.parse_args()

    cfg = load_pxdesign_config(args.config)
    chain_entry = cfg.target.chains.get(args.target_chain)
    if not isinstance(chain_entry, TargetChain) or chain_entry.hotspots is None:
        raise SystemExit(f"config has no hotspots on chain {args.target_chain}")
    hotspots = list(chain_entry.hotspots)

    pdbs: list[Path] = []
    for d in args.inputs:
        pdbs.extend(sorted(d.glob("**/predictions/converted_pdbs/*.pdb")))
    if not pdbs:
        raise SystemExit(f"no PDBs found under {args.inputs}")
    print(f"scoring {len(pdbs)} designs against {len(hotspots)} hotspots on chain {args.target_chain}")

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        rows = list(pool.map(
            lambda p: _score_design(p, hotspots, args.target_chain, args.binder_chain),
            pdbs,
        ))

    ranked = _rank_and_write(rows, args.out)
    print(f"wrote {args.out} ({len(rows)} designs)")
    print("top 5:")
    for rank, r, comp in ranked[:5]:
        helix_flag = "helix" if (r.angle_deg is not None) else "no-helix"
        angle = f"{r.angle_deg:.1f}°" if r.angle_deg is not None else "  -  "
        print(f"  {rank:>3}. {r.design_id:<40}  composite={comp:+.2f}  "
              f"helix_frac={r.helix_fraction:.2f}  bsa={r.bsa_a2:.0f}Å²  "
              f"angle={angle}  hs={r.n_hotspot_contacts}/{len(hotspots)}  [{helix_flag}]")


if __name__ == "__main__":
    main()
