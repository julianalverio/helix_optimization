from __future__ import annotations

import dataclasses
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yaml

from ... import paths
from ...config import PIPELINE_VERSION
from .config import ExamplesConfig, load_examples_config, examples_config_hash
from .constants import ENTRY_STATUS_COLUMNS, EXAMPLE_MANIFEST_COLUMNS


def _resolve_pipeline_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip() or PIPELINE_VERSION
    except (subprocess.CalledProcessError, FileNotFoundError):
        return PIPELINE_VERSION


_PIPELINE_VERSION = _resolve_pipeline_version()
_T0 = [0.0]

logger = logging.getLogger(__name__)


def _processing_date() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_log_handler(log_path: Path) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _write_config_snapshot(cfg: ExamplesConfig, output_dir: Path) -> None:
    out_path = output_dir / "config_used.yaml"
    tmp = out_path.with_suffix(".yaml.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)
    tmp.replace(out_path)


def _build_batches(entries: list[dict], batch_size: int) -> list[list[dict]]:
    return [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]


def _build_payload(batch: list[dict], cfg_raw: dict) -> list[dict]:
    items: list[dict] = []
    for entry in batch:
        m2_path = entry["m2_tensor_abs_path"]
        if not m2_path.exists():
            continue
        items.append({
            "pdb_id": entry["pdb_id"],
            "assembly_id": entry["assembly_id"],
            "module2_npz_bytes": m2_path.read_bytes(),
            "m2_meta": entry["m2_meta"],
            "cfg": cfg_raw,
        })
    return items


def _write_example(output_dir: Path, pdb_id: str, assembly_id: int,
                   example_id: int, tensor_bytes: bytes) -> str:
    rel = paths.example_rel_path(pdb_id, assembly_id, example_id)
    abs_path = output_dir / rel
    paths.atomic_write_bytes(abs_path, tensor_bytes)
    return rel


def _write_marker(output_dir: Path, pdb_id: str, assembly_id: int) -> None:
    abs_path = paths.marker_abs_path(output_dir, pdb_id, assembly_id)
    paths.atomic_write_bytes(abs_path, b"")


def _example_row(
    ex: dict,
    pdb_id: str,
    assembly_id: int,
    rel_path: str,
    m2_meta: dict,
    cfg_hash: str,
) -> dict:
    return {
        "example_id_full": f"{pdb_id}_{assembly_id}_{ex['example_id']}",
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "example_id": ex["example_id"],
        "helix_seqres_start": ex["helix_seqres_start"],
        "helix_seqres_end": ex["helix_seqres_end"],
        "helix_length": ex["helix_length"],
        "n_helix_residues": ex["n_helix_residues"],
        "n_partner_residues": ex["n_partner_residues"],
        "n_partner_chains": ex["n_partner_chains"],
        "n_helix_contacts": ex["n_helix_contacts"],
        "n_partner_interface_residues": ex["n_partner_interface_residues"],
        "n_residues_total": ex["n_residues_total"],
        "helix_sequence": ex["helix_sequence"],
        "resolution": m2_meta.get("resolution"),
        "r_free": m2_meta.get("r_free"),
        "source_method": m2_meta.get("method"),
        "sasa_used": ex["sasa_used"],
        "path_example": rel_path,
        "pipeline_version": _PIPELINE_VERSION,
        "config_hash": cfg_hash,
        "processing_date": _processing_date(),
    }


def _entry_status_row(result: dict) -> dict:
    return {
        "pdb_id": result["pdb_id"],
        "assembly_id": result["assembly_id"],
        "processing_status": result["processing_status"],
        "drop_reason": result["drop_reason"],
        "n_helix_segments": result.get("n_helix_segments"),
        "n_interacting_helices": result.get("n_interacting_helices"),
        "n_windows_before_filter": result.get("n_windows_before_filter"),
        "n_examples_emitted": result.get("n_examples_emitted"),
        "processing_date": _processing_date(),
    }


def _coerce_example_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        "example_id_full": "string",
        "pdb_id": "string",
        "helix_sequence": "string",
        "source_method": "string",
        "path_example": "string",
        "pipeline_version": "string",
        "config_hash": "string",
        "assembly_id": "int8",
        "example_id": "int32",
        "helix_seqres_start": "int32",
        "helix_seqres_end": "int32",
        "helix_length": "int16",
        "n_helix_residues": "int16",
        "n_partner_residues": "int16",
        "n_partner_chains": "int8",
        "n_helix_contacts": "int16",
        "n_partner_interface_residues": "int16",
        "n_residues_total": "int32",
        "resolution": "float32",
        "r_free": "float32",
        "sasa_used": "bool",
    })
    df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


def _coerce_entry_status_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        "pdb_id": "string",
        "processing_status": "string",
        "drop_reason": "string",
        "assembly_id": "int8",
        "n_helix_segments": "Int32",
        "n_interacting_helices": "Int32",
        "n_windows_before_filter": "Int32",
        "n_examples_emitted": "Int32",
    })
    df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_examples(
    examples_config_path: Path | str,
    test_mode: bool = False,
    force: bool = False,
) -> Path:
    cfg = load_examples_config(examples_config_path)
    if test_mode:
        cfg = dataclasses.replace(cfg, test_mode=True)

    output_root = Path(cfg.output_dir)
    output_dir = output_root / cfg.test_output_subdir if cfg.test_mode else output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    log_handler = _install_log_handler(output_dir / "examples.log")
    _T0[0] = time.time()
    logger.info("examples pipeline start (test_mode=%s)", cfg.test_mode)

    try:
        return _run(cfg, output_dir, force)
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def _plan_entries(cfg: ExamplesConfig, output_dir: Path, force: bool) -> tuple[list[dict], list[dict]]:
    m2_manifest = pd.read_parquet(cfg.tensors_manifest_path)
    m2_ok = m2_manifest[m2_manifest["processing_status"] == "ok"].copy()
    if cfg.test_mode:
        m2_ok = m2_ok.head(cfg.test_n_entries)

    tensors_output_dir = Path(cfg.tensors_output_dir)

    entries: list[dict] = []
    skipped_rows: list[dict] = []

    for _, row in m2_ok.iterrows():
        pdb_id = str(row["pdb_id"]).upper()
        assembly_id = int(row["assembly_id"])
        marker = paths.marker_abs_path(output_dir, pdb_id, assembly_id)
        if marker.exists() and not force:
            continue
        m2_rel = row["path_tensor"]
        if m2_rel is None or (isinstance(m2_rel, float) and np.isnan(m2_rel)):
            skipped_rows.append({
                "pdb_id": pdb_id, "assembly_id": assembly_id,
                "processing_status": "error",
                "drop_reason": "unparseable_module2_output",
                "n_helix_segments": None, "n_interacting_helices": None,
                "n_windows_before_filter": None, "n_examples_emitted": None,
                "processing_date": _processing_date(),
            })
            continue
        m2_abs = tensors_output_dir / str(m2_rel)
        if not m2_abs.exists():
            skipped_rows.append({
                "pdb_id": pdb_id, "assembly_id": assembly_id,
                "processing_status": "error",
                "drop_reason": "unparseable_module2_output",
                "n_helix_segments": None, "n_interacting_helices": None,
                "n_windows_before_filter": None, "n_examples_emitted": None,
                "processing_date": _processing_date(),
            })
            continue
        entries.append({
            "pdb_id": pdb_id,
            "assembly_id": assembly_id,
            "m2_tensor_abs_path": m2_abs,
            "m2_meta": {
                "resolution": _maybe_float(row.get("resolution")),
                "r_free": _maybe_float(row.get("r_free")),
                "method": _maybe_str(row.get("method")),
            },
        })
    return entries, skipped_rows


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _maybe_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


def _run(cfg: ExamplesConfig, output_dir: Path, force: bool) -> Path:
    cfg_hash = examples_config_hash(cfg)
    _write_config_snapshot(cfg, output_dir)

    entries, skipped_rows = _plan_entries(cfg, output_dir, force)
    logger.info("planning: %d entries to process, %d skipped/error",
                len(entries), len(skipped_rows))

    batch_size = cfg.test_modal_batch_size if cfg.test_mode else cfg.modal_batch_size

    example_rows: list[dict] = []
    entry_rows: list[dict] = list(skipped_rows)

    if entries:
        batches = _build_batches(entries, batch_size)
        logger.info("dispatching %d batches of up to %d", len(batches), batch_size)
        ex_rows, en_rows = _dispatch(batches, cfg, cfg_hash, output_dir)
        example_rows.extend(ex_rows)
        entry_rows.extend(en_rows)

    example_df = pd.DataFrame(example_rows, columns=EXAMPLE_MANIFEST_COLUMNS)
    if len(example_df):
        example_df = _coerce_example_dtypes(example_df)
    example_path = output_dir / "module3_manifest.parquet"
    _write_parquet(example_df, example_path)

    entry_df = pd.DataFrame(entry_rows, columns=ENTRY_STATUS_COLUMNS)
    if len(entry_df):
        entry_df = _coerce_entry_status_dtypes(entry_df)
    entry_path = output_dir / "module3_entry_status.parquet"
    _write_parquet(entry_df, entry_path)

    logger.info("wrote %s (%d rows), %s (%d rows)",
                example_path, len(example_df), entry_path, len(entry_df))

    from .report import build_summary_report, build_test_summary
    wall_time_sec = time.time() - _T0[0]
    build_summary_report(example_df, entry_df, cfg, output_dir, wall_time_sec)
    if cfg.test_mode:
        build_test_summary(example_df, entry_df, output_dir)

    logger.info("examples pipeline end (%.1f min wall time)", wall_time_sec / 60.0)
    return example_path


def _dispatch(
    batches: list[list[dict]],
    cfg: ExamplesConfig,
    cfg_hash: str,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    from concurrent.futures import ThreadPoolExecutor
    from .modal_app import app, process_batch

    cfg_raw = dataclasses.asdict(cfg)

    entry_lookup: dict[tuple[str, int], dict] = {}
    for batch in batches:
        for entry in batch:
            entry_lookup[(entry["pdb_id"], entry["assembly_id"])] = entry

    def payload_gen() -> Iterator[list[dict]]:
        for batch in batches:
            yield _build_payload(batch, cfg_raw)

    example_rows: list[dict] = []
    entry_rows: list[dict] = []
    completed_examples = 0
    seen_keys: set[tuple[str, int]] = set()
    batches_completed = 0
    t0 = time.time()
    total_batches = len(batches)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="m3-write") as writer, app.run():
        results_iter = process_batch.map(
            payload_gen(), return_exceptions=True, order_outputs=False,
        )
        pending_writes = []
        for result in results_iter:
            batches_completed += 1
            if isinstance(result, BaseException):
                logger.error(
                    "batch failed after retries (%d/%d completed): %r",
                    batches_completed, total_batches, result,
                )
                continue
            batch_ok = 0
            for entry_result in result:
                key = (entry_result["pdb_id"], entry_result["assembly_id"])
                source = entry_lookup.get(key)
                if source is None:
                    logger.warning("unexpected result key: %s", key)
                    continue
                seen_keys.add(key)
                m2_meta = source["m2_meta"]
                for ex in entry_result.get("examples") or []:
                    rel_path = paths.example_rel_path(
                        entry_result["pdb_id"], entry_result["assembly_id"], ex["example_id"],
                    )
                    fut = writer.submit(
                        paths.atomic_write_bytes,
                        output_dir / rel_path,
                        ex["tensor_bytes"],
                    )
                    pending_writes.append(fut)
                    example_rows.append(_example_row(
                        ex, entry_result["pdb_id"], entry_result["assembly_id"],
                        rel_path, m2_meta, cfg_hash,
                    ))
                    completed_examples += 1
                    if completed_examples % 1000 == 0:
                        elapsed = time.time() - t0
                        logger.info("progress: %d examples (%.1f min elapsed)",
                                    completed_examples, elapsed / 60.0)
                marker_path = paths.marker_abs_path(
                    output_dir, entry_result["pdb_id"], entry_result["assembly_id"],
                )
                pending_writes.append(writer.submit(paths.atomic_write_bytes, marker_path, b""))
                entry_rows.append(_entry_status_row(entry_result))
                if entry_result["processing_status"] == "ok":
                    batch_ok += 1
                for warn in entry_result.get("warnings") or []:
                    logger.warning("%s: %s", entry_result["pdb_id"], warn)
            if batches_completed % 50 == 0 or batches_completed == total_batches:
                logger.info("batches done %d/%d (last: %d/%d ok)",
                            batches_completed, total_batches, batch_ok, len(result))
        logger.info("draining %d pending writes", len(pending_writes))
        for fut in pending_writes:
            fut.result()

    missing = set(entry_lookup) - seen_keys
    if missing:
        logger.error("%d entries had no result; marking batch_retry_exhausted", len(missing))
        for pdb_id, assembly_id in missing:
            entry_rows.append({
                "pdb_id": pdb_id,
                "assembly_id": assembly_id,
                "processing_status": "error",
                "drop_reason": "batch_retry_exhausted",
                "n_helix_segments": None,
                "n_interacting_helices": None,
                "n_windows_before_filter": None,
                "n_examples_emitted": None,
                "processing_date": _processing_date(),
            })
    return example_rows, entry_rows
