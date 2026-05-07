"""Generate synthetic test inputs for the linkers pipeline.

Creates a 65-residue idealized poly-Ala alpha helix, then slices it into:
  - framework.pdb: three discontinuous helix segments (residues 1-12,
    31-40, 59-65) saved as a single chain so the splicer's cut points
    pick them up cleanly.
  - helix1.pdb: residues 17-26 of the same helix (already in the
    framework's coordinate frame).
  - helix2.pdb: residues 45-54 of the same helix.

Run from the project root with:
    .venv-rosetta/bin/python data/linkers/test/build_test_inputs.py
"""
from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).parent

SEQ_LEN = 65
F1_RANGE     = (1, 12)
HELIX1_RANGE = (17, 26)
F2_RANGE     = (31, 40)
HELIX2_RANGE = (45, 54)
F3_RANGE     = (59, 65)


def main() -> None:
    import pyrosetta
    pyrosetta.init('-mute all')

    pose = pyrosetta.pose_from_sequence('A' * SEQ_LEN)
    for i in range(1, SEQ_LEN + 1):
        pose.set_phi(i, -57.0)
        pose.set_psi(i, -47.0)
        pose.set_omega(i, 180.0)

    pose.pdb_info(pyrosetta.rosetta.core.pose.PDBInfo(pose))
    for i in range(1, SEQ_LEN + 1):
        pose.pdb_info().chain(i, 'A')
        pose.pdb_info().number(i, i)

    framework_residues = sorted(set(
        i for lo, hi in (F1_RANGE, F2_RANGE, F3_RANGE)
        for i in range(lo, hi + 1)
    ))

    def _dump_subset(residue_ids: list[int], out: Path) -> None:
        subpose = pyrosetta.Pose(pose, residue_ids[0], residue_ids[0])
        for rid in residue_ids[1:]:
            extra = pyrosetta.Pose(pose, rid, rid)
            pyrosetta.rosetta.core.pose.append_pose_to_pose(subpose, extra, False)
        info = pyrosetta.rosetta.core.pose.PDBInfo(subpose)
        for new_idx, orig_idx in enumerate(residue_ids, start=1):
            info.chain(new_idx, 'A')
            info.number(new_idx, orig_idx)
        subpose.pdb_info(info)
        subpose.dump_pdb(str(out))

    _dump_subset(framework_residues,                          OUT_DIR / 'framework.pdb')
    _dump_subset(list(range(HELIX1_RANGE[0], HELIX1_RANGE[1] + 1)), OUT_DIR / 'helix1.pdb')
    _dump_subset(list(range(HELIX2_RANGE[0], HELIX2_RANGE[1] + 1)), OUT_DIR / 'helix2.pdb')

    print(f"wrote {OUT_DIR}/{{framework,helix1,helix2}}.pdb")


if __name__ == '__main__':
    main()
