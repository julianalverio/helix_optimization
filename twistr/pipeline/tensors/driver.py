from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import shutil
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
from .config import TensorsConfig, load_tensors_config, tensors_config_hash
from .constants import write_constants_npz


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

logger = logging.getLogger(__name__)

_MANIFEST_COLUMNS = [
    "pdb_id", "assembly_id", "processing_status", "drop_reason",
    "method", "resolution", "r_free", "deposition_date", "release_date",
    "n_chains_processed", "n_substantive_chains", "path_tensor",
    "pipeline_version", "config_hash", "processing_date",
]


def _infer_assembly_id(raw: str | None) -> int:
    if raw is None:
        return 1
    s = str(raw).strip()
    return int(s) if s.isdigit() else 1


def _plan_to_list(val) -> list[dict] | None:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        return [dict(d) for d in val]
    except TypeError:
        return None


def _load_mmcif_bytes(mmcif_path: Path) -> bytes:
    return mmcif_path.read_bytes()


def _write_tensor(output_dir: Path, pdb_id: str, assembly_id: int, tensor_bytes: bytes) -> Path:
    out_path = paths.tensor_abs_path(output_dir, pdb_id, assembly_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npz.tmp")
    tmp.write_bytes(tensor_bytes)
    tmp.replace(out_path)
    return out_path


def _processing_date() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_cfg(cfg: TensorsConfig) -> dict:
    return dataclasses.asdict(cfg)


def _build_batches(entries: list[tuple[str, pd.Series]], batch_size: int) -> list[list[tuple[str, pd.Series]]]:
    return [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]


def _build_payload(
    batch: list[tuple[str, pd.Series]],
    cfg_raw: dict,
    data_root_path: Path,
) -> list[dict]:
    items = []
    for pdb_id, row in batch:
        mmcif_path = paths.mmcif_abs_path(data_root_path, pdb_id)
        if not mmcif_path.exists():
            continue
        items.append({
            "pdb_id": pdb_id,
            "assembly_id": _infer_assembly_id(row.get("primary_assembly_id")),
            "mmcif_bytes": _load_mmcif_bytes(mmcif_path),
            "m1_meta": {
                "primary_assembly_id": str(row.get("primary_assembly_id") or "1"),
                "large_assembly": bool(row.get("large_assembly", False)),
                "unique_interface_plan": _plan_to_list(row.get("unique_interface_plan")),
            },
            "cfg": cfg_raw,
        })
    return items


def _missing_entry_row(pdb_id: str, assembly_id: int, row: pd.Series,
                      status: str, drop_reason: str, cfg_hash: str) -> dict:
    return {
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "processing_status": status,
        "drop_reason": drop_reason,
        "method": row.get("method"),
        "resolution": row.get("resolution"),
        "r_free": row.get("r_free"),
        "deposition_date": row.get("deposition_date"),
        "release_date": row.get("release_date"),
        "n_chains_processed": None,
        "n_substantive_chains": None,
        "path_tensor": None,
        "pipeline_version": _PIPELINE_VERSION,
        "config_hash": cfg_hash,
        "processing_date": _processing_date(),
    }


def _entry_row(result: dict, row: pd.Series, tensor_rel: str | None, cfg_hash: str) -> dict:
    return {
        "pdb_id": result["pdb_id"],
        "assembly_id": result["assembly_id"],
        "processing_status": result["processing_status"],
        "drop_reason": result["drop_reason"],
        "method": row.get("method"),
        "resolution": row.get("resolution"),
        "r_free": row.get("r_free"),
        "deposition_date": row.get("deposition_date"),
        "release_date": row.get("release_date"),
        "n_chains_processed": result["n_chains_processed"],
        "n_substantive_chains": result["n_substantive_chains"],
        "path_tensor": tensor_rel,
        "pipeline_version": _PIPELINE_VERSION,
        "config_hash": cfg_hash,
        "processing_date": _processing_date(),
    }


def _coerce_manifest_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        "pdb_id": "string",
        "processing_status": "string",
        "drop_reason": "string",
        "method": "string",
        "path_tensor": "string",
        "pipeline_version": "string",
        "config_hash": "string",
        "assembly_id": "int8",
        "resolution": "float32",
        "r_free": "float32",
        "n_chains_processed": "Int32",
        "n_substantive_chains": "Int32",
    })
    for col in ("deposition_date", "release_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["processing_date"] = pd.to_datetime(df["processing_date"], errors="coerce")
    return df


def _write_manifest(manifest_df: pd.DataFrame, output_dir: Path) -> Path:
    out_path = output_dir / "module2_manifest.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    manifest_df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path


def _write_config_snapshot(cfg: TensorsConfig, output_dir: Path) -> None:
    out_path = output_dir / "config_used.yaml"
    tmp = out_path.with_suffix(".yaml.tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)
    tmp.replace(out_path)


def _install_log_handler(log_path: Path) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def run_tensors(
    tensors_config_path: Path | str,
    test_mode: bool = False,
    force: bool = False,
) -> Path:
    cfg = load_tensors_config(tensors_config_path)
    if test_mode:
        cfg = dataclasses.replace(cfg, test_mode=True)

    output_root = Path(cfg.output_dir)
    if cfg.test_mode:
        output_dir = output_root / cfg.test_output_subdir
    else:
        output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    log_handler = _install_log_handler(output_dir / "tensors.log")
    _T0[0] = time.time()
    logger.info("tensors pipeline start (test_mode=%s)", cfg.test_mode)

    try:
        return _run(cfg, output_dir, force)
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def _tensor_has_valid_ss(tensor_path: Path) -> bool:
    """Open an existing tensor and check whether ss_8 has at least one non-null
    code among real residues. Used to detect tensors produced by the pre-fix
    Module 2 build (which had a chain-id lookup bug that left every ss_8 == 8).
    Treat unreadable tensors as broken so they get reprocessed."""
    try:
        d = np.load(tensor_path)
        am = d["atom_mask"]
        ss8 = d["ss_8"]
        real = (am != -1).any(axis=-1)
        ss8_real = ss8[real]
        if len(ss8_real) == 0:
            return False
        return bool(int(np.sum(ss8_real != 8)) > 0)
    except Exception:
        return False


def _run(cfg: TensorsConfig, output_dir: Path, force: bool) -> Path:
    cfg_hash = tensors_config_hash(cfg)
    write_constants_npz(output_dir / "constants.npz")
    _write_config_snapshot(cfg, output_dir)

    manifest_df = pd.read_parquet(cfg.module1_manifest_path)
    if cfg.test_mode:
        manifest_df = manifest_df.head(cfg.test_n_entries).copy()

    data_root_path = Path(cfg.local_mmcif_base_path)
    batch_size = cfg.test_modal_batch_size if cfg.test_mode else cfg.modal_batch_size

    from concurrent.futures import ThreadPoolExecutor

    entries: list[tuple[str, pd.Series]] = []
    rows_out: list[dict] = []
    preexisting_rows: list[dict] = []
    missing_on_disk = 0
    broken_tensor_count = 0

    rows_with_meta = []
    candidate_paths: list[Path | None] = []
    for _, row in manifest_df.iterrows():
        pdb_id = row["pdb_id"]
        assembly_id = _infer_assembly_id(row.get("primary_assembly_id"))
        tensor_path = paths.tensor_abs_path(output_dir, pdb_id, assembly_id)
        rows_with_meta.append((pdb_id, assembly_id, row, tensor_path))
        candidate_paths.append(tensor_path if (tensor_path.exists() and not force) else None)

    if not force:
        existing_indices = [i for i, p in enumerate(candidate_paths) if p is not None]
        logger.info("planning: validating SS in %d existing tensors", len(existing_indices))
        with ThreadPoolExecutor(max_workers=16) as pool:
            ss_valid = list(pool.map(
                _tensor_has_valid_ss,
                [candidate_paths[i] for i in existing_indices],
            ))
        valid_set = {existing_indices[k] for k, ok in enumerate(ss_valid) if ok}
    else:
        valid_set = set()

    for i, (pdb_id, assembly_id, row, tensor_path) in enumerate(rows_with_meta):
        if i in valid_set:
            preexisting_rows.append(_entry_row(
                {"pdb_id": pdb_id.upper(), "assembly_id": assembly_id,
                 "processing_status": "ok", "drop_reason": None,
                 "n_chains_processed": None, "n_substantive_chains": None},
                row, paths.tensor_rel_path(pdb_id, assembly_id), cfg_hash,
            ))
            continue
        if candidate_paths[i] is not None:
            broken_tensor_count += 1
        mmcif_path = paths.mmcif_abs_path(data_root_path, pdb_id)
        if not mmcif_path.exists():
            missing_on_disk += 1
            rows_out.append(_missing_entry_row(
                pdb_id.upper(), assembly_id, row, "error",
                "processing_error", cfg_hash,
            ))
            continue
        entries.append((pdb_id, row))

    logger.info(
        "planning: %d to process (incl. %d broken tensors to overwrite), %d skipped (good tensor), %d mmCIF missing",
        len(entries), broken_tensor_count, len(preexisting_rows), missing_on_disk,
    )

    if entries:
        batches = _build_batches(entries, batch_size)
        logger.info("dispatching %d batches of up to %d", len(batches), batch_size)
        rows_out.extend(_dispatch(batches, cfg, cfg_hash, output_dir, data_root_path))

    rows_out.extend(preexisting_rows)
    manifest_out = pd.DataFrame(rows_out, columns=_MANIFEST_COLUMNS)
    manifest_out = _coerce_manifest_dtypes(manifest_out)
    out_path = _write_manifest(manifest_out, output_dir)
    logger.info("wrote manifest %s (%d rows)", out_path, len(manifest_out))

    from .report import build_summary_report, build_test_summary
    wall_time_sec = time.time() - _T0[0]
    build_summary_report(manifest_out, cfg, output_dir, wall_time_sec)
    if cfg.test_mode:
        build_test_summary(manifest_out, cfg, output_dir)

    logger.info("tensors pipeline end (%.1f min wall time)", wall_time_sec / 60.0)
    return out_path


_T0 = [0.0]


def _dispatch(
    batches: list[list[tuple[str, pd.Series]]],
    cfg: TensorsConfig,
    cfg_hash: str,
    output_dir: Path,
    data_root_path: Path,
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    from .modal_app import app, process_batch

    cfg_raw = _serialize_cfg(cfg)

    entry_lookup: dict[tuple[str, int], pd.Series] = {}
    for batch in batches:
        for pdb_id, row in batch:
            asm = _infer_assembly_id(row.get("primary_assembly_id"))
            entry_lookup[(pdb_id.upper(), asm)] = row
    total_entries = len(entry_lookup)

    def payload_gen() -> Iterator[list[dict]]:
        for batch in batches:
            yield _build_payload(batch, cfg_raw, data_root_path)

    rows: list[dict] = []
    completed = 0
    seen_keys: set[tuple[str, int]] = set()
    batches_completed = 0
    t0 = time.time()
    total_batches = len(batches)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="m2-write") as writer, app.run():
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
                row = entry_lookup.get(key)
                if row is None:
                    logger.warning("unexpected result key: %s", key)
                    continue
                seen_keys.add(key)
                tensor_rel = None
                if entry_result["processing_status"] == "ok" and entry_result["tensor_bytes"]:
                    fut = writer.submit(
                        _write_tensor,
                        output_dir,
                        entry_result["pdb_id"],
                        entry_result["assembly_id"],
                        entry_result["tensor_bytes"],
                    )
                    pending_writes.append(fut)
                    tensor_rel = paths.tensor_rel_path(
                        entry_result["pdb_id"], entry_result["assembly_id"],
                    )
                    completed += 1
                    batch_ok += 1
                    if completed % 1000 == 0:
                        elapsed = time.time() - t0
                        logger.info(
                            "progress: %d successes (%.1f min elapsed)",
                            completed, elapsed / 60.0,
                        )
                for warn in entry_result.get("warnings") or []:
                    logger.warning("%s: %s", entry_result["pdb_id"], warn)
                rows.append(_entry_row(entry_result, row, tensor_rel, cfg_hash))
            if batches_completed % 50 == 0 or batches_completed == total_batches:
                logger.info(
                    "batches done %d/%d (last: %d/%d ok)",
                    batches_completed, total_batches, batch_ok, len(result),
                )
        logger.info("draining %d pending tensor writes", len(pending_writes))
        for fut in pending_writes:
            fut.result()

    missing = set(entry_lookup) - seen_keys
    if missing:
        logger.error("%d entries had no result returned; marking batch_retry_exhausted", len(missing))
        for pdb_id, assembly_id in missing:
            row = entry_lookup[(pdb_id, assembly_id)]
            rows.append(_missing_entry_row(
                pdb_id, assembly_id, row, "error", "batch_retry_exhausted", cfg_hash,
            ))
    return rows
