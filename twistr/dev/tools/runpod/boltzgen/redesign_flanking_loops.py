"""Detect the flanking loop residues on either side of a grafted helix and
redesign them via Rosetta Remodel, reusing twistr.linkers's
PyRosetta subprocess runner. The designed chain A is then spliced back
into the original PDB with all other chains (e.g. the bound target)
preserved verbatim.

Currently supports flanks that are CONTIGUOUS in pose seqpos (i.e. only
left-flank, only right-flank, or a left+right pair that happen to abut
the helix). For two truly disjoint flank ranges, we'd need to either
run this twice or extend remodel_runner to accept multiple linker ranges.

Usage:
  python -m twistr.dev.tools.runpod.boltzgen.redesign_flanking_loops \\
      --pdb runtime/outputs/grafted/face1_into_93ite.pdb --chain A --helix-resi 13-22 \\
      --rosetta-python ~/.venv-rosetta/bin/python \\
      --out runtime/outputs/grafted/face1_into_93ite_redesigned.pdb
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import gemmi

from twistr.linkers.remodel_runner import run_remodel

DEFAULT_AA_WHITELIST = "AGSDNTPQEKR"
DEFAULT_NSTRUCT = 20
DEFAULT_NUM_TRAJECTORY = 20


def _ss(pdb_path: Path) -> dict[tuple[str, int], str]:
    """Run mkdssp; return {(chain, resnum) -> single-char SS}."""
    proc = subprocess.run(
        ["mkdssp", "--output-format", "dssp", str(pdb_path), "/dev/stdout"],
        capture_output=True, text=True, check=True,
    )
    out: dict[tuple[str, int], str] = {}
    in_data = False
    for line in proc.stdout.splitlines():
        if line.startswith("  #  RESIDUE"):
            in_data = True
            continue
        if not in_data or len(line) < 17:
            continue
        try:
            resnum = int(line[5:10].strip())
        except ValueError:
            continue
        ss = line[16]
        out[(line[11], resnum)] = ss if ss != " " else "-"
    return out


def _find_flanks(ss, chain, seqids, h_lo, h_hi):
    flank: list[int] = []
    for sid in sorted((x for x in seqids if x < h_lo), reverse=True):
        c = ss.get((chain, sid))
        if c is None or c == "E":
            break
        flank.append(sid)
    for sid in sorted(x for x in seqids if x > h_hi):
        c = ss.get((chain, sid))
        if c is None or c == "E":
            break
        flank.append(sid)
    return sorted(flank)


def _one_letter(resname: str) -> str:
    info = gemmi.find_tabulated_residue(resname)
    if info is None or not info.one_letter_code:
        raise ValueError(f"no 1-letter code for residue {resname!r}")
    return info.one_letter_code.upper()


def _write_subpose_and_blueprint(structure, chain_name, flanks, aa_whitelist,
                                 work_dir):
    """Emit subpose.pdb (the design chain only, renumbered 1..N so author
    seqid matches pose seqpos) + linker.blueprint marking flank residues
    designable. Returns (sub_pdb_path, blueprint_path, linker_lo,
    linker_hi)."""
    chain = structure[0][chain_name]

    new_struct = gemmi.Structure()
    new_struct.name = "subpose"
    new_struct.cell = gemmi.UnitCell()
    new_struct.spacegroup_hm = ""
    m = gemmi.Model("1")
    nc = gemmi.Chain(chain_name)
    seqid_to_pose: dict[int, int] = {}
    for i, r in enumerate(chain, start=1):
        new_r = gemmi.Residue()
        new_r.name = r.name
        new_r.seqid = gemmi.SeqId(i, " ")
        new_r.entity_type = gemmi.EntityType.Polymer
        new_r.het_flag = "A"
        for a in r:
            new_r.add_atom(a)
        nc.add_residue(new_r)
        seqid_to_pose[r.seqid.num] = i
    m.add_chain(nc)
    new_struct.add_model(m)
    new_struct.setup_entities()
    sub_pdb = work_dir / "subpose.pdb"
    sub_pdb.parent.mkdir(parents=True, exist_ok=True)
    new_struct.write_pdb(str(sub_pdb))

    flank_pose_ids = sorted(seqid_to_pose[s] for s in flanks)
    if flank_pose_ids != list(range(min(flank_pose_ids), max(flank_pose_ids) + 1)):
        raise SystemExit(
            f"flank residues are not contiguous in pose seqpos: {flank_pose_ids}. "
            f"This script currently only handles contiguous flanks."
        )
    linker_lo, linker_hi = min(flank_pose_ids), max(flank_pose_ids)

    flank_set = set(flank_pose_ids)
    lines: list[str] = []
    for i, r in enumerate(chain, start=1):
        aa = _one_letter(r.name)
        if i in flank_set:
            lines.append(f"{i} {aa} L PIKAA {aa_whitelist}")
        else:
            lines.append(f"{i} {aa} .")
    bp = work_dir / "linker.blueprint"
    bp.write_text("\n".join(lines) + "\n")
    return sub_pdb, bp, linker_lo, linker_hi


def _splice_back(designed_pdb, original_struct, chain_name, out_pdb):
    """Build the final PDB: designed chain (renumbered back to the
    original author seqids) + every other chain copied verbatim from
    original_struct (so the bound target survives untouched)."""
    designed = gemmi.read_structure(str(designed_pdb))
    designed_chain = designed[0][chain_name]
    designed_residues = list(designed_chain)

    orig_chain = original_struct[0][chain_name]
    orig_seqids = [r.seqid.num for r in orig_chain]
    if len(designed_residues) != len(orig_seqids):
        raise SystemExit(
            f"designed chain has {len(designed_residues)} residues, "
            f"original had {len(orig_seqids)}"
        )

    out = gemmi.Structure()
    out.name = "redesigned"
    out.cell = gemmi.UnitCell()
    out.spacegroup_hm = ""
    m = gemmi.Model("1")

    new_A = gemmi.Chain(chain_name)
    for orig_sid, dr in zip(orig_seqids, designed_residues):
        new_r = gemmi.Residue()
        new_r.name = dr.name
        new_r.seqid = gemmi.SeqId(orig_sid, " ")
        new_r.entity_type = gemmi.EntityType.Polymer
        new_r.het_flag = "A"
        for a in dr:
            new_r.add_atom(a)
        new_A.add_residue(new_r)
    m.add_chain(new_A)

    for c in original_struct[0]:
        if c.name == chain_name:
            continue
        new_c = gemmi.Chain(c.name)
        for r in c:
            new_r = gemmi.Residue()
            new_r.name = r.name
            new_r.seqid = r.seqid
            new_r.entity_type = r.entity_type
            new_r.het_flag = "A"
            for a in r:
                new_r.add_atom(a)
            new_c.add_residue(new_r)
        m.add_chain(new_c)

    out.add_model(m)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(out_pdb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--helix-resi", required=True, help="inclusive, e.g. 13-22")
    ap.add_argument("--rosetta-python", required=True,
                    help="Path to PyRosetta-enabled python (e.g. ~/.venv-rosetta/bin/python)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aa-whitelist", default=DEFAULT_AA_WHITELIST)
    ap.add_argument("--nstruct", type=int, default=DEFAULT_NSTRUCT)
    ap.add_argument("--num-trajectory", type=int, default=DEFAULT_NUM_TRAJECTORY)
    args = ap.parse_args()

    s = gemmi.read_structure(str(args.pdb.expanduser()))
    chain = s[0][args.chain]
    h_lo, h_hi = (int(x) for x in args.helix_resi.split("-"))
    seqids = sorted(r.seqid.num for r in chain)

    ss = _ss(args.pdb.expanduser())
    flanks = _find_flanks(ss, args.chain, seqids, h_lo, h_hi)
    if not flanks:
        raise SystemExit("no flanking loop residues between helix and beta sheet")
    print(f"redesigning {len(flanks)} flanking residues: {flanks}")

    work_root = args.out.parent / f"{args.out.stem}_work"
    sub_pdb, bp, lo, hi = _write_subpose_and_blueprint(
        s, args.chain, flanks, args.aa_whitelist, work_root)
    print(f"subpose: {sub_pdb}  blueprint: {bp}  pose linker: {lo}..{hi}")

    scores = run_remodel(
        rosetta_python=args.rosetta_python,
        subpose_pdb=sub_pdb,
        blueprint=bp,
        out_dir=work_root / "designs",
        nstruct=args.nstruct,
        num_trajectory=args.num_trajectory,
        linker_lo=lo,
        linker_hi=hi,
    )
    successes = [s for s in scores if s["error"] is None and s["path"] is not None]
    if not successes:
        raise SystemExit(
            f"no successful designs out of {args.nstruct}. First few errors: "
            f"{[s['error'] for s in scores[:3]]}"
        )
    best = min(successes, key=lambda s: s["total_score"])
    print(f"best design: {best['path']} (total_score={best['total_score']:.3f}, "
          f"successes={len(successes)}/{args.nstruct})")

    _splice_back(Path(best["path"]), s, args.chain, args.out.expanduser())
    print(f"wrote {args.out.expanduser()}")


if __name__ == "__main__":
    main()
