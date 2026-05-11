"""Run ProteinMPNN on PXDesign-`infer` backbones and emit N decoded
sequences per backbone, copying each backbone PDB to a sibling
directory with the i-th decoded sequence applied.

PXDesign's `infer` subcommand produces backbone-only PDBs with
placeholder residues. ProteinMPNN ignores residue identities and
conditions on backbone coords only, so we can re-decode any number of
sequences from each backbone. The N decodings per backbone are the
candidate sequences fed to Protenix for refolding.

Outputs (written to --out-dir, default `runtime/outputs/mpnn_runtime/outputs/`):
  - runtime/outputs/mpnn_runtime/outputs/pdbs/<flat_id>_dec<i>.pdb       — backbone copy with residues renamed
  - runtime/outputs/mpnn_runtime/outputs/decoded_sequences.fasta         — all N×K records, headers like
        >{flat_id}_dec{i}|score={mpnn_score}

Usage:
  python -m twistr.dev.tools.runpod.pxdesign.sequence_design \\
      --inputs runtime/outputs/design_runtime/outputs/3erd_b2_a runtime/outputs/design_runtime/outputs/3erd_b2_b ... \\
      --num-decodings 8
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi

MPNN_DIR = Path(__file__).resolve().parents[2] / "twistr/external/ProteinMPNN"

AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_SCORE_RE = re.compile(r"score=([0-9.]+)")


def _run_mpnn(pdb_path: Path, design_chain: str, out_dir: Path,
              seed: int, num_decodings: int) -> list[tuple[str, float]]:
    """Returns [(sequence, score)] of length num_decodings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(MPNN_DIR / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path.resolve()),
        "--pdb_path_chains", design_chain,
        "--out_folder", str(out_dir.resolve()),
        "--num_seq_per_target", str(num_decodings),
        "--sampling_temp", "0.1",
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, cwd=MPNN_DIR)

    # ProteinMPNN FASTA layout:
    #   record 0 (header includes "score=NATIVE_SCORE, ..."): input/native sequence
    #   records 1..N (header "T=..., sample=N, score=..., ..."): the N samples
    fa = out_dir / "seqs" / f"{pdb_path.stem}.fa"
    lines = fa.read_text().splitlines()
    headers = [i for i, ln in enumerate(lines) if ln.startswith(">")]
    if len(headers) < num_decodings + 1:
        raise RuntimeError(
            f"{pdb_path.name}: expected at least {num_decodings + 1} FASTA "
            f"records (1 native + {num_decodings} samples), got {len(headers)}"
        )
    out: list[tuple[str, float]] = []
    for h_idx in headers[1:1 + num_decodings]:
        header = lines[h_idx]
        seq = lines[h_idx + 1].strip()
        m = _SCORE_RE.search(header)
        if not m:
            raise RuntimeError(f"{pdb_path.name}: could not parse score from header: {header!r}")
        out.append((seq, float(m.group(1))))
    return out


def _write_decoded_pdb(src_pdb: Path, dst_pdb: Path, chain_id: str, sequence: str) -> None:
    structure = gemmi.read_structure(str(src_pdb))
    chain = structure[0][chain_id]
    residues = [r for r in chain]
    if len(residues) != len(sequence):
        raise RuntimeError(
            f"{src_pdb.name}: chain {chain_id} has {len(residues)} residues but "
            f"MPNN gave a {len(sequence)}-residue sequence"
        )
    for res, aa in zip(residues, sequence):
        res.name = AA_1_TO_3[aa]
    dst_pdb.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(dst_pdb))


def _flat_id(pdb: Path) -> str:
    """Build a flat design id from a PDB path, e.g.
    runtime/outputs/design_runtime/outputs/3erd_b2_b/.../seed_480288/.../sample_1.pdb
        → 3erd_b2_b_seed_480288_sample_1
    """
    run_part = pdb.parts[pdb.parts.index("design_outputs") + 1]
    seed_part = next((p for p in pdb.parts if p.startswith("seed_")), "seed_unknown")
    sample_part = pdb.stem.replace("pxdesign_target_", "")
    return f"{run_part}_{seed_part}_{sample_part}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Run dirs under runtime/outputs/design_runtime/outputs/, e.g. runtime/outputs/design_runtime/outputs/3erd_b2_a")
    parser.add_argument("--design-chain", default="B")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--num-decodings", type=int, default=8,
                        help="Sequences sampled per backbone (Bennett 2023 standard: 8).")
    parser.add_argument("--out-dir", type=Path, default=Path("mpnn_outputs"),
                        help="Where to write decoded PDBs and the concatenated FASTA.")
    args = parser.parse_args()

    if not (MPNN_DIR / "protein_mpnn_run.py").is_file():
        sys.exit(f"ProteinMPNN not found at {MPNN_DIR}; clone it first.")

    pdbs: list[Path] = []
    for d in args.inputs:
        pdbs.extend(sorted(d.glob("**/predictions/converted_pdbs/*.pdb")))
    if not pdbs:
        sys.exit(f"no PDBs found under {args.inputs}")
    print(f"running ProteinMPNN on {len(pdbs)} backbones × {args.num_decodings} decodings "
          f"= {len(pdbs) * args.num_decodings} sequences; designing chain {args.design_chain}")

    pdbs_out_dir = args.out_dir / "pdbs"
    pdbs_out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.out_dir / "decoded_sequences.fasta"
    fasta_lines: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, pdb in enumerate(pdbs, start=1):
            flat = _flat_id(pdb)
            mpnn_workdir = td_path / flat
            decodings = _run_mpnn(pdb, args.design_chain, mpnn_workdir,
                                  args.seed, args.num_decodings)
            for j, (seq, score) in enumerate(decodings):
                dec_id = f"{flat}_dec{j}"
                dst_pdb = pdbs_out_dir / f"{dec_id}.pdb"
                _write_decoded_pdb(pdb, dst_pdb, args.design_chain, seq)
                fasta_lines.append(f">{dec_id}|score={score:.4f}")
                fasta_lines.append(seq)
            top_seq, top_score = min(decodings, key=lambda x: x[1])
            print(f"  [{i:>2}/{len(pdbs)}] {flat}: best score={top_score:.3f} "
                  f"({args.num_decodings} samples)")

    fasta_path.write_text("\n".join(fasta_lines) + "\n")
    print(f"\nwrote {len(pdbs) * args.num_decodings} decoded sequences to {fasta_path}")
    print(f"wrote {len(pdbs) * args.num_decodings} decoded PDBs to {pdbs_out_dir}/")
    print("PDBs are backbone-only with renamed residues; sidechain placement happens at refold.")


if __name__ == "__main__":
    main()
