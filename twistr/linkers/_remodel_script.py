"""Standalone PyRosetta worker — runs ONE linker design job.

Invoked as a subprocess from `remodel_runner.py` against the isolated
`.venv-rosetta` interpreter. Has zero `twistr` imports so it can run in
that env without the rest of the project being installed there.

Job spec is read from a JSON file passed on argv:

    {
      "subpose_pdb":      <path>,    # input pose: 2 anchors + Lk Ala placeholders
      "blueprint":        <path>,    # Remodel blueprint marking placeholders as L
      "out_dir":          <path>,    # where designs and scores.json land
      "nstruct":          int,       # independent design attempts
      "num_trajectory":   int,       # centroid trajectories per attempt
      "linker_lo":        int,       # 1-based seqpos of first linker residue
      "linker_hi":        int,       # 1-based seqpos of last linker residue
      "max_chainbreak_a": float      # max accepted CA-CA deviation (Å) from 3.80
    }

Writes one PDB per closed design plus a `scores.json` with one record
per attempt: `{index, path, total_score, error}`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CA_CA_BOND_A = 3.80


def _ca_dist(pose, i: int, j: int) -> float:
    a = pose.residue(i).xyz('CA')
    b = pose.residue(j).xyz('CA')
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _max_linker_chainbreak(pose, lo: int, hi: int) -> float:
    """Largest deviation of consecutive CA-CA distances from 3.80 Å across
    the linker region and its two anchor boundaries (residues lo-1..hi+1)."""
    first = max(1, lo - 1)
    last = min(pose.size() - 1, hi)
    return max((abs(_ca_dist(pose, i, i + 1) - CA_CA_BOND_A)
                for i in range(first, last + 1)), default=0.0)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    subpose_pdb = Path(job["subpose_pdb"]).resolve()
    blueprint = Path(job["blueprint"]).resolve()
    out_dir = Path(job["out_dir"]).resolve()
    nstruct = int(job["nstruct"])
    num_trajectory = int(job["num_trajectory"])
    linker_lo = int(job["linker_lo"])
    linker_hi = int(job["linker_hi"])
    max_chainbreak_a = float(job["max_chainbreak_a"])
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyrosetta
    pyrosetta.init(
        f"-mute core.io basic -ex1 -ex2 -use_input_sc -extrachi_cutoff 1 "
        f"-remodel:blueprint {blueprint} "
        f"-remodel:num_trajectory {num_trajectory} "
        f"-remodel:save_top 0"
    )

    sf = pyrosetta.create_score_function('ref2015')
    base_pose = pyrosetta.pose_from_pdb(str(subpose_pdb))

    # Placeholder Ala residues from the gemmi-built sub-pose have only
    # approximate bond geometry. Forcing ideal extended phi/psi rebuilds
    # their backbone with chemically valid bond lengths/angles before
    # Remodel starts fragment insertion.
    for seqpos in range(linker_lo, linker_hi + 1):
        base_pose.set_phi(seqpos, -135.0)
        base_pose.set_psi(seqpos, 135.0)
        base_pose.set_omega(seqpos, 180.0)

    scores: list[dict] = []
    for i in range(nstruct):
        pose = base_pose.clone()
        rm = pyrosetta.rosetta.protocols.forge.remodel.RemodelMover()
        rm.register_options()
        rm.max_linear_chainbreak(max_chainbreak_a)
        try:
            rm.apply(pose)
        except RuntimeError as e:
            scores.append({"index": i, "path": None, "total_score": None,
                           "error": f"{type(e).__name__}: {e}"})
            continue

        # RemodelMover returns SUCCESS even when no centroid trajectory
        # closed (the pose retains the centroid placeholder sequence and
        # has a stretched peptide bond at the cutpoint). Validate by
        # measuring CA-CA geometry across the linker; this catches both
        # the "no trajectory closed" and "centroid OK but design failed"
        # cases. mover_status itself is not checked because Remodel does
        # not flip it on this failure mode.
        cb = _max_linker_chainbreak(pose, linker_lo, linker_hi)
        if cb > max_chainbreak_a:
            scores.append({"index": i, "path": None, "total_score": None,
                           "error": f"closure_failed: max CA-CA deviation {cb:.2f} A"})
            continue

        out_pdb = out_dir / f"design_{i:03d}.pdb"
        pose.dump_pdb(str(out_pdb))
        total = float(sf(pose))
        scores.append({"index": i, "path": str(out_pdb), "total_score": total,
                       "error": None})

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: _remodel_script.py <job.json>\n")
        sys.exit(2)
    main(sys.argv[1])
