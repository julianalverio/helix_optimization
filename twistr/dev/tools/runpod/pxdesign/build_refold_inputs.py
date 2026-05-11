"""Build per-design Protenix input JSONs for the all-atom refold.

Each design gets a single JSON containing:
  - target chain A sequence + cached MSA paths (paired+unpaired)
  - designed binder chain B sequence (single-sequence, no MSA)

Output JSONs are named `<design_id_with_underscores>.json` and written
to a flat output directory; the launcher shards them across pods.

Usage:
  python -m twistr.dev.tools.runpod.pxdesign.build_refold_inputs \\
      --target runtime/data/pdb/3ERD.cif \\
      --target-chain A \\
      --fasta runtime/outputs/rankings/designed_sequences.fasta runtime/outputs/rankings/designed_sequences_face1_rest.fasta runtime/outputs/rankings/designed_sequences_face2.fasta \\
      --out refold_inputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi

from twistr.dev.tools.runpod.pxdesign.launch import MSA_CACHE_DIR, REPO_ROOT, msa_cache_key


def _read_fastas(paths: list[Path]) -> dict[str, str]:
    """Returns {design_id: sequence}. Design IDs come from FASTA headers; any
    `|`-suffixed metadata (e.g. `|score=1.234`) is stripped."""
    out: dict[str, str] = {}
    for p in paths:
        lines = p.read_text().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                header = lines[i][1:].strip()
                design_id = header.split("|", 1)[0].strip()
                seq = lines[i + 1].strip()
                out[design_id] = seq
                i += 2
            else:
                i += 1
    return out


def _target_sequence(target_path: Path, chain_id: str) -> str:
    structure = gemmi.read_structure(str(target_path))
    chain = structure[0][chain_id]
    poly = chain.get_polymer()
    return gemmi.one_letter_code([r.name for r in poly])


def _pod_msa_path(local_path: Path) -> str:
    """Translate a local-repo path to where it lives after the tarball is
    extracted at /workspace/twistr on the pod."""
    return f"/workspace/twistr/{local_path.relative_to(REPO_ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--fasta", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    target_seq = _target_sequence(args.target, args.target_chain)
    msa_dir = MSA_CACHE_DIR / msa_cache_key(args.target, args.target_chain)
    paired = msa_dir / "pairing.a3m"
    unpaired = msa_dir / "non_pairing.a3m"
    if not paired.is_file() or not unpaired.is_file():
        raise SystemExit(f"missing MSA at {msa_dir}; run build_msas first.")

    paired_pod = _pod_msa_path(paired)
    unpaired_pod = _pod_msa_path(unpaired)

    designs = _read_fastas(args.fasta)
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for design_id, binder_seq in designs.items():
        flat = design_id.replace("/", "_")
        payload = [{
            "name": flat,
            "sequences": [
                {"proteinChain": {
                    "sequence": target_seq,
                    "count": 1,
                    "pairedMsaPath": paired_pod,
                    "unpairedMsaPath": unpaired_pod,
                }},
                {"proteinChain": {
                    "sequence": binder_seq,
                    "count": 1,
                }},
            ],
            "covalent_bonds": [],
        }]
        (args.out / f"{flat}.json").write_text(json.dumps(payload, indent=2))
        written += 1
    print(f"wrote {written} Protenix input JSONs to {args.out}/")
    print(f"  target: {args.target} chain {args.target_chain} ({len(target_seq)} aa)")
    print(f"  MSA: {msa_dir}  → pod paths: {paired_pod}, {unpaired_pod}")


if __name__ == "__main__":
    main()
