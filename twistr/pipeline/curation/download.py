from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd

from ... import paths
from ...config import Config

logger = logging.getLogger(__name__)


def build_candidate_paths_file(candidates_df: pd.DataFrame, out_path: Path) -> int:
    passing = candidates_df[candidates_df["passed_all_filters"]]
    lines = [paths.mmcif_rel_path(pid).removeprefix("pdb/") for pid in passing["pdb_id"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


CHUNK_SIZE = 5000


def _rsync_command(source: str, port: int, files_from: Path, dest: Path) -> list[str]:
    return [
        "rsync",
        "-rlpt",
        "-v",
        "-z",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--timeout=300",
        "--contimeout=60",
        f"--port={port}",
        f"--files-from={files_from}",
        source,
        str(dest) + "/",
    ]


def _split_files_from(files_from: Path) -> list[Path]:
    lines = [l for l in files_from.read_text().splitlines() if l]
    chunk_dir = files_from.parent / ".rsync_chunks"
    chunk_dir.mkdir(exist_ok=True)
    for stale in chunk_dir.glob("chunk_*.txt"):
        stale.unlink()
    chunks: list[Path] = []
    for i in range(0, len(lines), CHUNK_SIZE):
        path = chunk_dir / f"chunk_{i // CHUNK_SIZE:05d}.txt"
        path.write_text("\n".join(lines[i : i + CHUNK_SIZE]) + "\n")
        chunks.append(path)
    return chunks


def _rsync_chunks(source: str, port: int, chunks: list[Path], dest: Path, label: str) -> list[Path]:
    failed: list[Path] = []
    for i, chunk in enumerate(chunks, 1):
        cmd = _rsync_command(source, port, chunk, dest)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.warning("rsync %s chunk %d/%d exited %d", label, i, len(chunks), result.returncode)
            failed.append(chunk)
    return failed


def run_rsync(cfg: Config, data_root_path: Path, files_from: Path) -> None:
    dest = paths.pdb_dir(data_root_path)
    dest.mkdir(parents=True, exist_ok=True)
    chunks = _split_files_from(files_from)
    logger.info("rsync: %d chunks of up to %d files", len(chunks), CHUNK_SIZE)

    logger.info("rsync primary: %s", cfg.rsync_primary)
    failed = _rsync_chunks(cfg.rsync_primary, cfg.rsync_primary_port, chunks, dest, "primary")
    if not failed:
        return

    logger.warning("rsync primary: %d/%d chunks failed, retrying via fallback %s", len(failed), len(chunks), cfg.rsync_fallback)
    still_failed = _rsync_chunks(cfg.rsync_fallback, cfg.rsync_fallback_port, failed, dest, "fallback")
    if still_failed:
        logger.error("rsync fallback: %d/%d chunks failed; continuing with missing files (tracked via drop_reason=download_missing)", len(still_failed), len(failed))


def run_phase_b(cfg: Config, data_root_path: Path, candidates_path: Path) -> Path:
    candidates_df = pd.read_parquet(candidates_path)
    files_from = paths.manifests_dir(data_root_path) / "candidate_paths.txt"
    count = build_candidate_paths_file(candidates_df, files_from)
    if count == 0:
        return files_from
    run_rsync(cfg, data_root_path, files_from)
    return files_from
