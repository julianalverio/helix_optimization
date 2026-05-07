"""Build BoltzGen design specs for the 3ERD face-1 / face-2 binder design.

The original 3ERD.cif chain A starts at label_seq=12 (auth=305) — there
are unmodelled N-terminal residues. PXDesign's hotspots ({55,59,...},
{34..38}) name positions in the original label_seq frame.

BoltzGen indexes chains starting at residue 1, so we pre-clean the
target: keep only chain A protein, renumber starting at 1, and
translate hotspots into the new frame.

Usage (per face):
    python -m tools.runpod_boltzgen.build_specs \\
        --target data/pdb/3ERD.cif \\
        --face 1 --hotspots 55,59,62,63,66,70 \\
        --binder-length 80 \\
        --out-dir boltzgen_specs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi


_AA_TO_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _renumber_chain_a(target_path: Path, dest_cif: Path) -> list[tuple[int, int, str]]:
    """Read `target_path`, keep only chain A protein residues, renumber
    them starting at 1, write to `dest_cif`. Return a list of
    (new_position, original_label_seq, residue_name) tuples for hotspot
    translation.

    BoltzGen reads `_entity_poly_seq` to map residue indices, so we must
    declare a proper polymer entity on the output structure (not just
    write atoms)."""
    src = gemmi.read_structure(str(target_path))
    if not len(src):
        sys.exit(f"no models in {target_path}")
    src_chain_a = None
    for c in src[0]:
        if c.name == "A":
            src_chain_a = c
            break
    if src_chain_a is None:
        sys.exit(f"no chain A in {target_path}")

    out_struct = gemmi.Structure()
    out_struct.name = "target"
    out_model = gemmi.Model("1")
    out_chain = gemmi.Chain("A")

    mapping: list[tuple[int, int, str]] = []
    seq_names: list[str] = []
    for residue in src_chain_a:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is None or not info.is_amino_acid():
            continue
        if residue.name not in _AA_TO_LETTER:
            continue
        new_pos = len(mapping) + 1
        new_res = gemmi.Residue()
        new_res.name = residue.name
        new_res.seqid = gemmi.SeqId(new_pos, " ")
        new_res.label_seq = new_pos
        new_res.subchain = "A"
        new_res.entity_type = gemmi.EntityType.Polymer
        for atom in residue:
            new_res.add_atom(atom)
        out_chain.add_residue(new_res)
        mapping.append((new_pos, residue.label_seq, residue.name))
        seq_names.append(residue.name)

    if not mapping:
        sys.exit(f"chain A in {target_path} has no amino acids")

    out_model.add_chain(out_chain)
    out_struct.add_model(out_model)

    # Declare the polymer entity explicitly. setup_entities() alone won't
    # populate _entity_poly_seq if no Entity object exists.
    ent = gemmi.Entity("1")
    ent.entity_type = gemmi.EntityType.Polymer
    ent.polymer_type = gemmi.PolymerType.PeptideL
    ent.subchains = ["A"]
    ent.full_sequence = seq_names
    out_struct.entities.append(ent)
    out_struct.setup_entities()
    out_struct.assign_label_seq_id()

    dest_cif.parent.mkdir(parents=True, exist_ok=True)
    out_struct.make_mmcif_document().write_file(str(dest_cif))
    return mapping


def _translate_hotspots(mapping: list[tuple[int, int, str]],
                        hotspots: list[int]) -> list[tuple[int, int, str]]:
    """Map each PXDesign hotspot (= original label_seq) to (new_pos,
    original, name). Hard-error on any miss."""
    by_orig = {orig: (new, name) for new, orig, name in mapping}
    out = []
    missing = []
    for h in hotspots:
        if h not in by_orig:
            missing.append(h)
            continue
        new, name = by_orig[h]
        out.append((new, h, name))
    if missing:
        sys.exit(f"hotspots not found in chain A: {missing}")
    return out


def _emit_spec_yaml(target_pod_path: str, hotspots_new: list[int],
                    binder_length: int, helix_bias: bool = False) -> str:
    """Match the schema in BoltzGen's example/peptide_against_specific_site_on_ragc/rragc.yaml.
    `helix_bias` is off for the smoke (the per-residue SS string format on a
    `protein:` entity isn't documented; we add it back for production once the
    syntax is verified)."""
    binding_str = ",".join(str(h) for h in hotspots_new)
    ss_line = f"      secondary_structure: {'H' * binder_length}\n" if helix_bias else ""
    return (
        "entities:\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {binder_length}\n"
        f"{ss_line}"
        "  - file:\n"
        f"      path: {target_pod_path}\n"
        "      include:\n"
        "        - chain:\n"
        "            id: A\n"
        "      binding_types:\n"
        "        - chain:\n"
        "            id: A\n"
        f"            binding: {binding_str}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True,
                        help="Original target CIF (e.g. data/pdb/3ERD.cif)")
    parser.add_argument("--face", type=int, required=True, choices=[1, 2])
    parser.add_argument("--hotspots", required=True,
                        help="Comma-separated PXDesign-frame residue numbers, "
                             "e.g. '55,59,62,63,66,70'")
    parser.add_argument("--binder-length", type=int, default=80)
    parser.add_argument("--helix-bias", action="store_true",
                        help="Constrain the binder to be all-helical "
                             "(secondary_structure per-residue string).")
    parser.add_argument("--out-dir", type=Path, default=Path("boltzgen_specs"))
    args = parser.parse_args()

    if not args.target.is_file():
        sys.exit(f"target not found: {args.target}")
    hotspots_orig = [int(x) for x in args.hotspots.split(",") if x.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_dest = args.out_dir / "3erd_chainA_renumbered.cif"
    mapping = _renumber_chain_a(args.target, target_dest)
    print(f"renumbered chain A → {target_dest} ({len(mapping)} residues)", flush=True)

    translated = _translate_hotspots(mapping, hotspots_orig)
    print(f"face-{args.face} hotspot translation:", flush=True)
    for new, orig, name in translated:
        print(f"  PXDesign {orig} ({name}) → BoltzGen {new}", flush=True)

    pod_path = f"/workspace/twistr/{target_dest}"
    spec_yaml = _emit_spec_yaml(
        target_pod_path=pod_path,
        hotspots_new=[new for new, _, _ in translated],
        binder_length=args.binder_length,
        helix_bias=args.helix_bias,
    )
    spec_dest = args.out_dir / f"face{args.face}.yaml"
    spec_dest.write_text(spec_yaml)
    print(f"wrote {spec_dest}", flush=True)


if __name__ == "__main__":
    main()
