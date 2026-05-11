"""Visualize Module 3 training examples to verify the crop is correct.

By default, walks the entire `runtime/data/examples/examples/` tree in a randomly
shuffled order — every Enter press advances to a new, randomly chosen
example. Pass explicit .npz paths if you want a specific set instead.

For each example the script writes three mmCIF files into
~/Desktop/debugging_pdbs/, then copies a single, fully self-contained PyMOL
command to the clipboard. The next clipboard command does `reinitialize`
first so PyMOL state is fully cleared before the next scene loads.

Three PyMOL objects, three colors:
  - expanded         gray70 (full assembly baseline)
  - helix_cropped    firebrick (dark red)
  - contacts_cropped / context_cropped   marine (dark blue)
                     Object name + scope depend on the mode:
        default        only interface contacts (is_interface_residue=True)
        --all          the full cropped context (everything not in the helix,
                       including the ±2 sequence-context residues)

Verification: the firebrick sticks should sit on top of the gray cartoon along
the cropped helix region, and the marine sticks should sit on top of the gray
cartoon along the cropped partner residues — confirming the cropped tensors
geometrically match the source structure.

Usage:
    python -m twistr.examples.visualize                                   # random examples, Enter for next
    python -m twistr.examples.visualize --all                             # same, full cropped context
    python -m twistr.examples.visualize --seed 7                          # reproducible random order
    python -m twistr.examples.visualize runtime/data/examples/examples/br/1brs_1_0.npz   # specific example
    python -m twistr.examples.visualize runtime/data/examples/examples/br/1brs_1_*.npz   # specific set
"""
from __future__ import annotations

import argparse
import gzip
import random
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from ..tensors.constants import ATOM14_SLOT_NAMES, RESIDUE_TYPE_NAMES

OUTPUT_DIR = Path.home() / "Desktop" / "debugging_pdbs"


def _decode(x):
    return str(x.item()) if hasattr(x, "item") else str(x)


def load_example(npz_path: Path) -> dict:
    d = np.load(npz_path)
    return {
        "coordinates": d["coordinates"],
        "atom_mask": d["atom_mask"],
        "residue_type": d["residue_type"],
        "chain_slot": d["chain_slot"],
        "seqres_position": d["seqres_position"],
        "is_helix": d["is_helix"].astype(bool),
        "is_interface_residue": d["is_interface_residue"].astype(bool),
        "chain_label": [_decode(c) for c in d["chain_label"]],
        "chain_role": d["chain_role"],
        "pdb_id": _decode(d["pdb_id"]),
        "assembly_id": int(d["assembly_id"]),
        "example_id": int(d["example_id"]),
        "helix_seqres_start": int(d["helix_seqres_start"]),
        "helix_seqres_end": int(d["helix_seqres_end"]),
        "helix_sequence": _decode(d["helix_sequence"]),
    }


def _plan_to_list(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return [dict(d) for d in val]
    except TypeError:
        return None


def _restrict_to_plan_chains(structure: gemmi.Structure, plan: list[dict]) -> None:
    wanted: set[str] = set()
    for entry in plan:
        a1, a2 = entry.get("asym_id_1"), entry.get("asym_id_2")
        if a1:
            wanted.add(str(a1))
        if a2:
            wanted.add(str(a2))
    if not wanted:
        return
    for model in structure:
        to_remove = [i for i, chain in enumerate(model) if chain.name not in wanted]
        for i in reversed(to_remove):
            del model[i]


def build_expanded_structure(pdb_id: str, m1_row: pd.Series, data_root: Path) -> gemmi.Structure:
    pdb_lower = pdb_id.lower()
    cif_path = data_root / "pdb" / pdb_lower[1:3] / f"{pdb_lower}.cif.gz"
    if not cif_path.exists():
        raise FileNotFoundError(f"Original mmCIF not found at {cif_path}")
    text = gzip.decompress(cif_path.read_bytes()).decode("utf-8", errors="replace")
    doc = gemmi.cif.read_string(text)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    structure.setup_entities()
    primary_asm = str(m1_row.get("primary_assembly_id") or "1")
    structure.transform_to_assembly(primary_asm, gemmi.HowToNameCopiedChain.Short)
    if bool(m1_row.get("large_assembly", False)):
        plan = _plan_to_list(m1_row.get("unique_interface_plan"))
        if plan:
            _restrict_to_plan_chains(structure, plan)
    structure.name = f"{pdb_lower}_{primary_asm}_expanded"
    return structure


def build_cropped_structure(ex: dict, residue_mask: np.ndarray, name: str) -> gemmi.Structure:
    """Build a gemmi Structure containing only the residues in `residue_mask`.
    gemmi copies chains/residues on add, so each chain must be fully populated
    before being attached to the model."""
    n = ex["coordinates"].shape[0]
    chains: dict[str, gemmi.Chain] = {}
    chain_order: list[str] = []
    for i in range(n):
        if not bool(residue_mask[i]):
            continue
        slot = int(ex["chain_slot"][i])
        chain_name = ex["chain_label"][slot]
        if chain_name not in chains:
            chains[chain_name] = gemmi.Chain(chain_name)
            chain_order.append(chain_name)

        rtype = int(ex["residue_type"][i])
        res = gemmi.Residue()
        res.name = RESIDUE_TYPE_NAMES[rtype]
        res.seqid.num = int(ex["seqres_position"][i])
        res.seqid.icode = " "
        res.subchain = chain_name
        res.entity_type = gemmi.EntityType.Polymer
        res.het_flag = "A"

        slot_names = ATOM14_SLOT_NAMES[rtype]
        for j in range(14):
            if int(ex["atom_mask"][i, j]) != 1:
                continue
            atom_name = slot_names[j]
            if not atom_name:
                continue
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(
                float(ex["coordinates"][i, j, 0]),
                float(ex["coordinates"][i, j, 1]),
                float(ex["coordinates"][i, j, 2]),
            )
            atom.occ = 1.0
            atom.b_iso = 30.0
            res.add_atom(atom)
        chains[chain_name].add_residue(res)

    structure = gemmi.Structure()
    structure.name = name
    model = gemmi.Model("1")
    for cname in chain_order:
        model.add_chain(chains[cname])
    structure.add_model(model)
    structure.setup_entities()
    return structure


def _resi_list(positions: list[int]) -> str:
    """PyMOL resi selector as a `+`-joined list. Bulletproof against negative
    auth_seq_id, which breaks PyMOL's `a-b` range syntax."""
    return "+".join(str(p) for p in sorted(set(positions)))


def build_pymol_command(
    expanded_path: Path,
    helix_path: Path,
    partners_path: Path,
    partners_obj_name: str,
) -> str:
    lines = [
        "reinitialize",
        "set cif_use_auth, 1",
        f"load {expanded_path}, expanded",
        f"load {helix_path}, helix_cropped",
        f"load {partners_path}, {partners_obj_name}",
        "dss expanded",
        "alter helix_cropped, ss='H'",
        "rebuild helix_cropped",
        "bg_color white",
        "hide everything",
        "show cartoon",
        f"show sticks, helix_cropped or {partners_obj_name}",
        "color gray70, expanded",
        "color firebrick, helix_cropped",
        f"color marine, {partners_obj_name}",
        f"orient helix_cropped or {partners_obj_name}",
        f"zoom helix_cropped or {partners_obj_name}, 5",
    ]
    return "\n".join(lines)


def _pbcopy(text: str) -> bool:
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def visualize_one(npz_path: Path, args, m1: pd.DataFrame) -> str:
    """Build the three mmCIF files for one example and return its PyMOL command."""
    ex = load_example(npz_path)
    rows = m1[m1["pdb_id"].astype(str).str.upper() == ex["pdb_id"].upper()]
    if len(rows) == 0:
        print(
            f"  WARNING: no Module 1 row for {ex['pdb_id']} — falling back to default assembly handling",
            file=sys.stderr,
        )
        m1_row = pd.Series({
            "primary_assembly_id": str(ex["assembly_id"]),
            "large_assembly": False,
            "unique_interface_plan": None,
        })
    else:
        m1_row = rows.iloc[0]

    expanded = build_expanded_structure(ex["pdb_id"], m1_row, args.data_root)

    pdb_l = ex["pdb_id"].lower()
    helix_name = f"{pdb_l}_{ex['assembly_id']}_{ex['example_id']}_helix"

    if args.all:
        partner_mask = ~ex["is_helix"]
        partners_obj_name = "context_cropped"
        partners_suffix = "context"
    else:
        partner_mask = (~ex["is_helix"]) & ex["is_interface_residue"]
        partners_obj_name = "contacts_cropped"
        partners_suffix = "contacts"
    partners_name = f"{pdb_l}_{ex['assembly_id']}_{ex['example_id']}_{partners_suffix}"

    helix_struct = build_cropped_structure(ex, ex["is_helix"], helix_name)
    partners_struct = build_cropped_structure(ex, partner_mask, partners_name)

    expanded_path = OUTPUT_DIR / f"{pdb_l}_{ex['assembly_id']}_expanded.cif"
    helix_path = OUTPUT_DIR / f"{helix_name}.cif"
    partners_path = OUTPUT_DIR / f"{partners_name}.cif"
    expanded.make_mmcif_document().write_file(str(expanded_path))
    helix_struct.make_mmcif_document().write_file(str(helix_path))
    partners_struct.make_mmcif_document().write_file(str(partners_path))

    n_helix = int(ex["is_helix"].sum())
    n_partners_shown = int(partner_mask.sum())
    n_partners_total = int((~ex["is_helix"]).sum())
    print(
        f"  pdb={ex['pdb_id']} assembly={ex['assembly_id']} example={ex['example_id']}  "
        f"helix=chain '{ex['chain_label'][0]}' resi {ex['helix_seqres_start']}-{ex['helix_seqres_end']} "
        f"({ex['helix_sequence']})  "
        f"residues: helix={n_helix} partners={n_partners_shown}/{n_partners_total}"
    )

    return build_pymol_command(
        expanded_path, helix_path, partners_path, partners_obj_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz_paths", type=Path, nargs="*",
        help="Specific Module 3 example .npz files. If omitted, the script walks "
             "--examples-root in a randomly shuffled order — Enter shows a new "
             "random example each time.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--examples-root", type=Path, default=Path("runtime/data/examples/examples"),
        help="Where to look for examples when no explicit npz paths are given.",
    )
    parser.add_argument(
        "--module1-manifest", type=Path,
        default=Path("runtime/data/manifests/module1_manifest.parquet"),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show the full cropped context (all non-helix residues including ±2 sequence "
             "neighbors) instead of only the interface contacts.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the random shuffle (only used when walking --examples-root). "
             "Default: nondeterministic.",
    )
    args = parser.parse_args()

    if not args.examples_root.exists():
        sys.exit(f"examples root not found: {args.examples_root}")

    explicit = [p.resolve() for p in args.npz_paths]
    missing = [p for p in explicit if not p.exists()]
    if missing:
        sys.exit("npz file(s) not found:\n  " + "\n  ".join(str(p) for p in missing))

    print(f"Scanning {args.examples_root} for examples…")
    tree_paths = [p.resolve() for p in args.examples_root.rglob("*.npz")]
    if not tree_paths:
        sys.exit(f"no .npz files under {args.examples_root}")

    rng = random.Random(args.seed)
    rng.shuffle(tree_paths)

    explicit_set = set(explicit)
    random_tail = [p for p in tree_paths if p not in explicit_set]
    all_paths = explicit + random_tail
    total = len(all_paths)
    npz_iter = iter(all_paths)

    seed_note = f"seed={args.seed}" if args.seed is not None else "random seed"
    if explicit:
        source_label = (
            f"{len(explicit)} explicit example(s), then {len(random_tail)} more "
            f"from {args.examples_root} shuffled ({seed_note})"
        )
    else:
        source_label = f"{total} examples under {args.examples_root}, shuffled ({seed_note})"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading Module 1 manifest at {args.module1_manifest}")
    m1 = pd.read_parquet(args.module1_manifest)

    mode_label = "--all (full cropped context)" if args.all else "default (interface contacts only)"
    partners_obj_name = "context_cropped" if args.all else "contacts_cropped"
    partner_label = (
        "cropped context (interface + ±2 sequence neighbors)"
        if args.all else "interface contacts"
    )

    print(f"\nSource: {source_label}")
    print(f"Mode:   {mode_label}")
    print(f"{'=' * 72}")
    print("Color legend (same for every example)")
    print(f"{'=' * 72}")
    print(f"  expanded            gray70     — full assembly baseline (cartoon)")
    print(f"  helix_cropped       firebrick  — cropped helix (sticks)")
    print(f"  {partners_obj_name:<20}marine     — cropped {partner_label} (sticks)")
    print("Verification: firebrick + marine sticks should sit on top of the gray cartoon.")
    print(f"{'=' * 72}\n")

    i = 0
    for npz_path in npz_iter:
        i += 1
        print(f"--- example {i}/{total}: {npz_path} ---")
        try:
            command = visualize_one(npz_path, args, m1)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue

        print(command)
        if _pbcopy(command):
            print("\n[copied to clipboard]")
        else:
            print("\n[pbcopy unavailable — copy the block above manually]")

        try:
            input("\nPress Enter for the next example (Ctrl-C to quit): ")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")
            return

    print("\nReached the end of the example list. Done.")


if __name__ == "__main__":
    main()
