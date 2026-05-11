from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .curation import paths
from .curation.config import load_config, snapshot_now
from .curation import candidates, download, manifest, report, verify


def _data_root(cfg) -> Path:
    root = paths.data_root(cfg.data_root)
    paths.ensure_dirs(root)
    return root


def cmd_fetch_candidates(args):
    cfg = load_config(args.config)
    root = _data_root(cfg)
    snapshot = snapshot_now()
    out = candidates.run_phase_a(cfg, root, full_scale=args.full_scale, snapshot_date=snapshot)
    print(f"wrote {out}")


def cmd_download(args):
    cfg = load_config(args.config)
    root = _data_root(cfg)
    candidates_path = paths.manifests_dir(root) / "candidates.parquet"
    files_from = download.run_phase_b(cfg, root, candidates_path)
    print(f"rsync driven by {files_from}")


def cmd_verify(args):
    cfg = load_config(args.config)
    root = _data_root(cfg)
    candidates_path = paths.manifests_dir(root) / "candidates.parquet"
    entries, chains = verify.run_phase_c(cfg, root, candidates_path, workers=args.workers)
    print(f"wrote {entries} and {chains}")


def cmd_report(args):
    cfg = load_config(args.config)
    root = _data_root(cfg)
    snapshot = snapshot_now()
    final, audit = manifest.build_final_manifest(cfg, root, snapshot)
    report_path = report.build_report(cfg, root, snapshot)
    print(f"wrote {final}, {audit}, {report_path}")


def cmd_tensors(args):
    from .tensors.driver import run_tensors
    out = run_tensors(args.tensors_config, test_mode=args.test_mode, force=args.force)
    print(f"wrote {out}")


def cmd_examples(args):
    from .examples.driver import run_examples
    out = run_examples(args.examples_config, test_mode=args.test_mode, force=args.force)
    print(f"wrote {out}")


def cmd_linkers(args):
    from .linkers.driver import main as run_linkers
    out = run_linkers(Path(args.linkers_config))
    print(f"wrote {out}")


def cmd_pipeline_run(args):
    from .epitope_selection.manager.manager import run_pipeline
    pdb_list = Path(args.pdb_list) if args.pdb_list else None
    out = run_pipeline(pdb_list, Path(args.epitope_selection_config))
    print(f"wrote {out}")


def cmd_run_all(args):
    cfg = load_config(args.config)
    root = _data_root(cfg)
    snapshot = snapshot_now()

    candidates_path = candidates.run_phase_a(cfg, root, full_scale=args.full_scale, snapshot_date=snapshot)
    print(f"phase A -> {candidates_path}")

    download.run_phase_b(cfg, root, candidates_path)
    print("phase B complete")

    entries, chains = verify.run_phase_c(cfg, root, candidates_path, workers=args.workers)
    print(f"phase C -> {entries}, {chains}")

    final, audit = manifest.build_final_manifest(cfg, root, snapshot)
    report_path = report.build_report(cfg, root, snapshot)
    print(f"final -> {final}")
    print(f"audit -> {audit}")
    print(f"report -> {report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="twistr")
    parser.add_argument("--config", type=str, default="runtime/configs/curation.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, adds_full_scale, adds_workers in [
        ("fetch-candidates", cmd_fetch_candidates, True, False),
        ("download", cmd_download, False, False),
        ("verify", cmd_verify, False, True),
        ("report", cmd_report, False, False),
        ("run-all", cmd_run_all, True, True),
    ]:
        sp = sub.add_parser(name)
        if adds_full_scale:
            sp.add_argument("--full-scale", action="store_true")
        if adds_workers:
            sp.add_argument("--workers", type=int, default=None)
        sp.set_defaults(func=fn)

    sp_tensors = sub.add_parser("tensors")
    sp_tensors.add_argument("--tensors-config", type=str, default="runtime/configs/tensors.yaml")
    sp_tensors.add_argument("--test-mode", action="store_true")
    sp_tensors.add_argument("--force", action="store_true")
    sp_tensors.set_defaults(func=cmd_tensors)

    sp_examples = sub.add_parser("examples")
    sp_examples.add_argument("--examples-config", type=str, default="runtime/configs/examples.yaml")
    sp_examples.add_argument("--test-mode", action="store_true")
    sp_examples.add_argument("--force", action="store_true")
    sp_examples.set_defaults(func=cmd_examples)

    sp_linkers = sub.add_parser("linkers",
        help="Design 4 linkers connecting 2 helices to a framework via RosettaRemodel")
    sp_linkers.add_argument("--linkers-config", type=str, default="runtime/configs/linkers.yaml")
    sp_linkers.set_defaults(func=cmd_linkers)

    sp_epitope_sel = sub.add_parser("epitope-selection-run",
        help="Run the epitope pipeline (MaSIF + ScanNet + PPI-hotspot + viz) "
             "with stages selected by the YAML's `stages: [...]` list. Only "
             "PPI-hotspotID's critires.sh runs on Modal; the rest is local.")
    sp_epitope_sel.add_argument("--epitope-selection-config", type=str, default="runtime/configs/epitope_selection.yaml")
    sp_epitope_sel.add_argument("--pdb-list", type=str, default=None,
                             help="Text file with one PDB ID per line. Required only when "
                                  "the first stage in the config is 'masif'; other stages "
                                  "derive the PDB set from the upstream parquet.")
    sp_epitope_sel.set_defaults(func=cmd_pipeline_run)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
