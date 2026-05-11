"""PDB splicing for the linkers pipeline.

The wrapper takes 3 input PDBs (framework + 2 helices, all in a shared
coordinate frame) and produces:
  - per-linker sub-pose PDBs containing only the two flanking anchor
    segments (Remodel inserts the linker residues via the blueprint).
  - a final assembled full-length PDB after the four chosen linker
    designs are merged back into the scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi

from .config import LinkersConfig


# ---------------------------------------------------------------------------
# Loading helpers


def _first_polymer_chain(structure: gemmi.Structure) -> gemmi.Chain:
    return structure[0][0]


def _residues_in_range(chain: gemmi.Chain, lo: int, hi: int) -> list[gemmi.Residue]:
    """Return residues whose author seqid is in [lo, hi] inclusive, in chain order."""
    out: list[gemmi.Residue] = []
    for res in chain:
        sid = res.seqid.num
        if lo <= sid <= hi:
            out.append(res)
    if not out:
        raise ValueError(f"No residues in author seqid range [{lo}, {hi}]")
    return out


def _all_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [res for res in chain]


def _slice_framework(framework: gemmi.Chain, cfg: LinkersConfig) -> tuple[
    list[gemmi.Residue], list[gemmi.Residue], list[gemmi.Residue],
]:
    """Split framework chain into F1, F2, F3 by the configured cut points."""
    seqids = [r.seqid.num for r in framework]
    if not seqids:
        raise ValueError("Framework chain is empty")
    cb1, ca1 = cfg.window1.cut_before, cfg.window1.cut_after
    cb2, ca2 = cfg.window2.cut_before, cfg.window2.cut_after
    if not (cb1 < ca1 <= cb2 < ca2):
        raise ValueError(
            f"Cut points must satisfy window1.cut_before < window1.cut_after "
            f"<= window2.cut_before < window2.cut_after; got "
            f"({cb1}, {ca1}, {cb2}, {ca2})"
        )
    lo, hi = min(seqids), max(seqids)
    f1 = _residues_in_range(framework, lo, cb1)
    f2 = _residues_in_range(framework, ca1, cb2)
    f3 = _residues_in_range(framework, ca2, hi)
    return f1, f2, f3


# ---------------------------------------------------------------------------
# PDB writing


def _clone_residues_to_chain(
    src_residues: list[gemmi.Residue],
    dest_chain: gemmi.Chain,
    start_seqid: int,
) -> int:
    """Append clones of src_residues to dest_chain renumbered from start_seqid.
    Returns the next available seqid after the last appended residue."""
    next_id = start_seqid
    for src in src_residues:
        clone = gemmi.Residue()
        clone.name = src.name
        clone.seqid = gemmi.SeqId(next_id, ' ')
        clone.entity_type = gemmi.EntityType.Polymer
        clone.het_flag = 'A'
        for atom in src:
            clone.add_atom(atom)
        dest_chain.add_residue(clone)
        next_id += 1
    return next_id


def _write_pdb(structure: gemmi.Structure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.setup_entities()
    structure.write_pdb(str(path))


# ---------------------------------------------------------------------------
# Sub-pose construction (one per linker)


@dataclass(frozen=True)
class SubposeLayout:
    """Indices into the sub-pose (1-based) for the blueprint generator."""
    upstream_anchor: tuple[int, int]      # (lo, hi) inclusive
    linker: tuple[int, int]               # (lo, hi) inclusive — Ala placeholder residues
    downstream_anchor: tuple[int, int]    # (lo, hi) inclusive
    insertion_length: int                 # Lk (== linker[1] - linker[0] + 1)
    upstream_aa: str                      # one-letter sequence of the upstream anchor
    downstream_aa: str                    # one-letter sequence of the downstream anchor


def _aa_one_letter(name: str) -> str:
    info = gemmi.find_tabulated_residue(name)
    if info is None or not info.one_letter_code:
        raise ValueError(
            f"Anchor residue {name!r} has no one-letter code; non-standard "
            f"residues are not supported in framework/helix anchors."
        )
    return info.one_letter_code.upper()


def _seq_one_letter(residues: list[gemmi.Residue]) -> str:
    return ''.join(_aa_one_letter(r.name) for r in residues)


def _ala_placeholder(seqid: int, ca_xyz: gemmi.Position) -> gemmi.Residue:
    """A minimal Ala residue with N, CA, C, O, CB clustered near `ca_xyz`.

    Real geometry is irrelevant — Remodel rebuilds the entire backbone of
    every linker residue. We just need atom records that PyRosetta will
    accept when loading the input pose.
    """
    res = gemmi.Residue()
    res.name = 'ALA'
    res.seqid = gemmi.SeqId(seqid, ' ')
    res.entity_type = gemmi.EntityType.Polymer
    res.het_flag = 'A'

    def _atom(name: str, element: str, dx: float, dy: float, dz: float) -> gemmi.Atom:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(ca_xyz.x + dx, ca_xyz.y + dy, ca_xyz.z + dz)
        atom.occ = 1.0
        atom.b_iso = 30.0
        return atom

    # Idealized Ala backbone offsets (from CA, in Å). Geometry is approximate
    # but bond lengths land in the chemically sane window so PyRosetta's
    # residue type matching does not reject the residue.
    res.add_atom(_atom('N',  'N', -1.20,  0.40,  0.00))
    res.add_atom(_atom('CA', 'C',  0.00,  0.00,  0.00))
    res.add_atom(_atom('C',  'C',  1.20,  0.40,  0.00))
    res.add_atom(_atom('O',  'O',  1.30,  1.60,  0.00))
    res.add_atom(_atom('CB', 'C',  0.00, -1.10, -0.90))
    return res


CA_CA_BOND_A = 3.80


def _extended_chain_cas(
    upstream_last_ca: gemmi.Position,
    downstream_first_ca: gemmi.Position,
    n: int,
) -> list[gemmi.Position]:
    """Place n placeholder CAs spaced CA_CA_BOND_A apart, walking from
    upstream toward downstream. Realistic local geometry — Remodel can
    close the residual chain break by KIC."""
    dx = downstream_first_ca.x - upstream_last_ca.x
    dy = downstream_first_ca.y - upstream_last_ca.y
    dz = downstream_first_ca.z - upstream_last_ca.z
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5
    if norm < 1e-6:
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux, uy, uz = dx / norm, dy / norm, dz / norm
    return [
        gemmi.Position(
            upstream_last_ca.x + ux * CA_CA_BOND_A * i,
            upstream_last_ca.y + uy * CA_CA_BOND_A * i,
            upstream_last_ca.z + uz * CA_CA_BOND_A * i,
        )
        for i in range(1, n + 1)
    ]


def _build_subpose(
    upstream: list[gemmi.Residue],
    downstream: list[gemmi.Residue],
    insertion_length: int,
    out_pdb: Path,
) -> SubposeLayout:
    """Write a sub-pose PDB containing the two anchor segments PLUS
    `insertion_length` Ala placeholder residues between them.

    The blueprint marks the placeholders for rebuild + design; their
    initial coordinates do not matter because Remodel rewrites the
    backbone via fragment insertion + KIC closure.
    """
    if not upstream or not downstream:
        raise ValueError("Both anchor segments must be non-empty")
    if insertion_length < 1:
        raise ValueError(f"insertion_length must be >= 1, got {insertion_length}")

    structure = gemmi.Structure()
    structure.name = 'subpose'
    structure.cell = gemmi.UnitCell()
    structure.spacegroup_hm = ''
    model = gemmi.Model('1')
    chain = gemmi.Chain('A')

    next_id = _clone_residues_to_chain(upstream, chain, 1)
    upstream_hi = next_id - 1

    last_ca = upstream[-1].find_atom('CA', '*')
    first_ca = downstream[0].find_atom('CA', '*')
    if last_ca is None or first_ca is None:
        raise ValueError("Anchor residues missing CA atoms")
    placeholder_cas = _extended_chain_cas(last_ca.pos, first_ca.pos, insertion_length)
    linker_lo = next_id
    for ca_pos in placeholder_cas:
        chain.add_residue(_ala_placeholder(next_id, ca_pos))
        next_id += 1
    linker_hi = next_id - 1

    next_id = _clone_residues_to_chain(downstream, chain, next_id)
    downstream_hi = next_id - 1

    model.add_chain(chain)
    structure.add_model(model)
    _write_pdb(structure, out_pdb)

    return SubposeLayout(
        upstream_anchor=(1, upstream_hi),
        linker=(linker_lo, linker_hi),
        downstream_anchor=(linker_hi + 1, downstream_hi),
        insertion_length=insertion_length,
        upstream_aa=_seq_one_letter(upstream),
        downstream_aa=_seq_one_letter(downstream),
    )


# ---------------------------------------------------------------------------
# Public sub-pose API: one builder per linker_id


def build_all_subposes(cfg: LinkersConfig, work_dir: Path) -> dict[str, tuple[Path, SubposeLayout]]:
    """Construct the four per-linker sub-pose PDBs.

    Returns a dict keyed by linker_id ('linker1'..'linker4') with
    (sub-pose PDB path, layout) tuples.
    """
    framework = _first_polymer_chain(gemmi.read_structure(cfg.framework_pdb))
    helix1 = _first_polymer_chain(gemmi.read_structure(cfg.helix1_pdb))
    helix2 = _first_polymer_chain(gemmi.read_structure(cfg.helix2_pdb))

    f1, f2, f3 = _slice_framework(framework, cfg)
    h1 = _all_residues(helix1)
    h2 = _all_residues(helix2)

    K = cfg.context_residues
    if K < 1:
        raise ValueError("context_residues must be >= 1")
    for name, segment in (('F1', f1), ('F2', f2), ('F3', f3),
                          ('helix1', h1), ('helix2', h2)):
        if len(segment) < K:
            raise ValueError(
                f"{name} has only {len(segment)} residues but "
                f"context_residues={K}; reduce context_residues or "
                f"use a longer segment."
            )

    anchors: dict[str, tuple[list[gemmi.Residue], list[gemmi.Residue], int]] = {
        'linker1': (f1[-K:], h1[:K], cfg.lengths.linker1),
        'linker2': (h1[-K:], f2[:K], cfg.lengths.linker2),
        'linker3': (f2[-K:], h2[:K], cfg.lengths.linker3),
        'linker4': (h2[-K:], f3[:K], cfg.lengths.linker4),
    }

    out: dict[str, tuple[Path, SubposeLayout]] = {}
    for lid, (up, dn, lk) in anchors.items():
        if lk < 1:
            raise ValueError(f"{lid}: insertion length must be >= 1, got {lk}")
        sub_pdb = work_dir / lid / 'subpose.pdb'
        layout = _build_subpose(up, dn, lk, sub_pdb)
        out[lid] = (sub_pdb, layout)
    return out


# ---------------------------------------------------------------------------
# Final-pose assembly


def _extract_linker_residues(
    designed_pdb: Path,
    layout: SubposeLayout,
) -> list[gemmi.Residue]:
    """Pull just the rebuilt linker residues from a Remodel output sub-pose."""
    structure = gemmi.read_structure(str(designed_pdb))
    chain = _first_polymer_chain(structure)
    lo, hi = layout.linker
    out = [res for res in chain if lo <= res.seqid.num <= hi]
    if len(out) != layout.insertion_length:
        raise ValueError(
            f"{designed_pdb}: expected {layout.insertion_length} linker "
            f"residues at seqids ({lo}..{hi}), found {len(out)}"
        )
    return out


def assemble_full_pose(
    cfg: LinkersConfig,
    chosen_designs: dict[str, Path],
    layouts: dict[str, SubposeLayout],
    out_pdb: Path,
) -> Path:
    """Splice the chosen per-linker designs back into the full scaffold."""
    framework = _first_polymer_chain(gemmi.read_structure(cfg.framework_pdb))
    helix1 = _first_polymer_chain(gemmi.read_structure(cfg.helix1_pdb))
    helix2 = _first_polymer_chain(gemmi.read_structure(cfg.helix2_pdb))

    f1, f2, f3 = _slice_framework(framework, cfg)
    h1 = _all_residues(helix1)
    h2 = _all_residues(helix2)

    linker_residues = {
        lid: _extract_linker_residues(chosen_designs[lid], layouts[lid])
        for lid in ('linker1', 'linker2', 'linker3', 'linker4')
    }

    structure = gemmi.Structure()
    structure.name = 'final'
    structure.cell = gemmi.UnitCell()
    structure.spacegroup_hm = ''
    model = gemmi.Model('1')
    chain = gemmi.Chain('A')

    next_id = 1
    for segment in (
        f1, linker_residues['linker1'], h1, linker_residues['linker2'],
        f2, linker_residues['linker3'], h2, linker_residues['linker4'], f3,
    ):
        next_id = _clone_residues_to_chain(segment, chain, next_id)

    model.add_chain(chain)
    structure.add_model(model)
    _write_pdb(structure, out_pdb)
    return out_pdb
