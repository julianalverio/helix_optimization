"""Pod-side wrapper. Renders a PXDesign-format target YAML from the
unified config and invokes `pxdesign pipeline`.

Runs INSIDE the PXDesign conda env (activated by bootstrap.sh). The
`pxdesign` console-script comes from PXDesign's setup.py:60 and is on
PATH once the env is active.

MSAs are built locally before the pod is launched (see
`dev/tools/runpod/pxdesign/build_msas.py`); the unified config's `msa:`
fields point at cache locations, so this wrapper only renders + runs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from twistr.dev.tools.runpod.pxdesign.config import PXDesignConfig, load_pxdesign_config
from twistr.dev.tools.runpod.pxdesign.launch import MSA_CACHE_DIR, msa_cache_key


def _resolved_msa(cfg: PXDesignConfig, chain_id: str, explicit: str | None) -> str | None:
    """Return the msa path to write into the rendered YAML. Explicit
    paths (from the unified config) take priority; otherwise look up
    the cache. Returns None for preview preset (no MSA needed)."""
    if cfg.preset == "preview":
        return None
    if explicit is not None:
        return explicit
    cache_path = MSA_CACHE_DIR / msa_cache_key(Path(cfg.target.file), chain_id)
    if cache_path.is_dir() and any(cache_path.iterdir()):
        return str(cache_path)
    sys.exit(
        f"chain {chain_id}: no msa: in config and no cache hit at "
        f"{cache_path}. Build MSAs first via "
        "`python -m twistr.dev.tools.runpod.pxdesign.build_msas <config>`."
    )


def render_target_yaml(cfg: PXDesignConfig) -> str:
    """Convert the unified config's target spec into PXDesign's YAML format
    (see twistr/external/PXDesign/examples/PDL1_quick_start.yaml)."""
    chains_out = {}
    for cid, chain in cfg.target.chains.items():
        if isinstance(chain, str):
            chains_out[cid] = chain
            continue
        entry = {}
        if chain.crop is not None:
            entry["crop"] = list(chain.crop)
        if chain.hotspots is not None:
            entry["hotspots"] = list(chain.hotspots)
        msa = _resolved_msa(cfg, cid, chain.msa)
        if msa is not None:
            entry["msa"] = msa
        chains_out[cid] = entry
    payload = {
        "target": {"file": cfg.target.file, "chains": chains_out},
        "binder_length": cfg.binder_length,
    }
    return yaml.safe_dump(payload, sort_keys=False)


def build_cli(cfg: PXDesignConfig, target_yaml: Path, out_dir: Path) -> list[str]:
    # `infer` calls the raw-diffusion subcommand (no AF2/Protenix evaluation).
    # Avoids the eval-stage hangs we hit repeatedly with preview/extended.
    if cfg.preset == "infer":
        return [
            "pxdesign", "infer",
            "-i", str(target_yaml),
            "-o", str(out_dir),
            "--N_sample", str(cfg.n_sample),
            "--N_step", str(cfg.n_step),
            "--dtype", cfg.dtype,
            "--use_fast_ln", str(cfg.use_fast_ln),
            "--use_deepspeed_evo_attention", str(cfg.use_deepspeed_evo_attention),
        ]
    return [
        "pxdesign", "pipeline",
        "--preset", cfg.preset,
        "-i", str(target_yaml),
        "-o", str(out_dir),
        "--N_sample", str(cfg.n_sample),
        "--N_step", str(cfg.n_step),
        "--dtype", cfg.dtype,
        "--use_fast_ln", str(cfg.use_fast_ln),
        "--use_deepspeed_evo_attention", str(cfg.use_deepspeed_evo_attention),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("runtime/configs/pxdesign.yaml"))
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render the PXDesign target YAML to stdout and exit; do not run the pipeline.",
    )
    args = parser.parse_args()

    cfg = load_pxdesign_config(args.config)
    rendered = render_target_yaml(cfg)
    if args.render_only:
        sys.stdout.write(rendered)
        return

    target_yaml = Path("/tmp/pxdesign_target.yaml")
    target_yaml.write_text(rendered)
    out_dir = Path("design_outputs") / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cli(cfg, target_yaml, out_dir)
    print(f"==> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"==> outputs at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
