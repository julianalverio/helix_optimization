"""Build MSAs for any chains referenced by the given configs that don't
have a cache hit at `.cache/pxdesign_msa/<key>/`. HTTP requests run in
parallel via threads; the wall clock is dominated by ColabFold's
server queue + remote search time, not by our client.

Usage:
    python -m twistr.dev.tools.runpod.pxdesign.build_msas runtime/configs/pxdesign.yaml

After this runs, every chain referenced by the configs has either an
explicit `msa:` path (set by the user) or a populated cache entry.
`launch.py` pre-flight then resolves trivially.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from twistr.dev.tools.runpod.pxdesign._msa_client import fetch_unpaired_msa
from twistr.dev.tools.runpod.pxdesign._seq_extract import extract_chain_sequence
from twistr.dev.tools.runpod.pxdesign.config import PXDesignConfig, load_pxdesign_config
from twistr.dev.tools.runpod.pxdesign.launch import MSA_CACHE_DIR, msa_cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_WORKERS = 8  # ColabFold rate-limits per IP; >8 just queues server-side.


def _chains_to_build(cfg: PXDesignConfig) -> list[tuple[str, str, Path]]:
    """For one config, return (chain_id, sequence, cache_path) for each
    chain that needs a fresh MSA. Skips:
      - "all" shorthand (no per-chain config to attach an MSA to)
      - chains with an explicit `msa:` path
      - chains with a non-empty cache hit
    """
    target_path = REPO_ROOT / cfg.target.file
    if not target_path.is_file():
        sys.exit(f"target file not found: {target_path}")
    out: list[tuple[str, str, Path]] = []
    for cid, chain in cfg.target.chains.items():
        if isinstance(chain, str) or chain.msa is not None:
            continue
        cache_path = MSA_CACHE_DIR / msa_cache_key(target_path, cid)
        if cache_path.is_dir() and any(cache_path.iterdir()):
            continue
        out.append((cid, extract_chain_sequence(target_path, cid), cache_path))
    return out


def _build_one(sequence: str, cache_path: Path) -> None:
    """Run the MSA search and atomically populate cache_path. Raises on
    failure (caller logs and tracks per-job).

    PXDesign's `inputs.py` requires both `pairing.a3m` and `non_pairing.a3m`
    in the MSA dir even for single-chain targets — it raises FileNotFoundError
    on the missing one. We only fetch the unpaired MSA (true paired MSAs need
    multi-chain coordination); copy non_pairing.a3m → pairing.a3m as a
    single-chain stand-in so PXDesign's parser is happy.
    """
    staging = cache_path.with_suffix(".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        non_pairing = fetch_unpaired_msa(sequence, staging)
        shutil.copy(non_pairing, staging / "pairing.a3m")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if cache_path.exists():
        shutil.rmtree(cache_path)
    staging.rename(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path,
                        help="One or more PXDesign config YAMLs.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent HTTP requests (default {MAX_WORKERS}).")
    args = parser.parse_args()

    jobs: list[tuple[str, str, Path]] = []  # (label, sequence, cache_path)
    seen: set[Path] = set()
    for config_path in args.configs:
        cfg = load_pxdesign_config(config_path)
        for cid, sequence, cache_path in _chains_to_build(cfg):
            # Dedupe across configs that share the same target file + chain.
            if cache_path in seen:
                continue
            seen.add(cache_path)
            label = f"{config_path.name}:{cid}"
            jobs.append((label, sequence, cache_path))
            print(f"queued: {label} → {cache_path.relative_to(REPO_ROOT)}", flush=True)

    if not jobs:
        print("all MSAs already cached; nothing to do.")
        return

    print(f"building {len(jobs)} MSA(s) with up to {args.max_workers} concurrent requests...")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_build_one, seq, path): label for label, seq, path in jobs}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                fut.result()
                print(f"done: {label}", flush=True)
            except Exception as e:
                failures += 1
                print(f"FAILED: {label}: {e}", flush=True)

    if failures:
        sys.exit(f"{failures}/{len(jobs)} MSA build(s) failed; see above")
    print(f"all {len(jobs)} MSA(s) built and cached.")


if __name__ == "__main__":
    main()
