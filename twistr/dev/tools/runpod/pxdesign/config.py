"""Unified config for the PXDesign RunPod wrapper.

Holds both the PXDesign target spec (target file, chains, hotspots,
binder_length) AND the run-time/launcher knobs (preset, N_sample, dtype,
GPU preference, output dir). The pod-side wrapper splits this into a
PXDesign-format target YAML at runtime; the launcher reads the launcher
knobs (gpu_preferences, network_volume_id) before pod creation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TargetChain:
    crop: tuple[str, ...] | None = None
    hotspots: tuple[int, ...] | None = None
    # Optional explicit MSA dir. Leave None to use the cached MSA at
    # .cache/pxdesign_msa/<key>/ (built on the pod and pulled back on
    # the first run for this target+chain).
    msa: str | None = None


@dataclass(frozen=True)
class Target:
    file: str
    chains: dict[str, TargetChain | str]


@dataclass(frozen=True)
class PXDesignConfig:
    binder_length: int
    target: Target

    preset: str = "extended"
    n_sample: int = 100
    n_step: int = 400
    dtype: str = "bf16"
    use_fast_ln: bool = True
    use_deepspeed_evo_attention: bool = True

    gpu_preferences: tuple[str, ...] = (
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 80GB HBM3",
    )
    # When set, the launcher mounts this volume at /workspace/cache so
    # the conda env + ~10 GB of weights persist across runs. RunPod
    # requires data_center_id to match the volume's DC.
    network_volume_id: str | None = None
    data_center_id: str | None = None
    output_subdir: str = "task"


def _parse_chain(value) -> TargetChain | str:
    # `A:` with no body parses as None — treat as empty config (full chain,
    # no crop, no hotspots).
    if value is None:
        return TargetChain()
    if isinstance(value, str):
        if value != "all":
            raise ValueError(f"chain shorthand must be 'all', got {value!r}")
        return "all"
    if not isinstance(value, dict):
        raise ValueError(
            f"chain entry must be 'all' or a mapping, got {type(value).__name__}"
        )
    known = {f.name for f in fields(TargetChain)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"Unknown chain keys: {sorted(unknown)}")
    crop = value.get("crop")
    hotspots = value.get("hotspots")
    return TargetChain(
        crop=tuple(crop) if crop is not None else None,
        hotspots=tuple(hotspots) if hotspots is not None else None,
        msa=value.get("msa"),
    )


def _parse_target(raw) -> Target:
    if not isinstance(raw, dict):
        raise ValueError("target must be a mapping")
    if "file" not in raw or "chains" not in raw:
        raise ValueError("target must define both 'file' and 'chains'")
    if not isinstance(raw["chains"], dict):
        raise ValueError("target.chains must be a mapping")
    return Target(
        file=raw["file"],
        chains={cid: _parse_chain(v) for cid, v in raw["chains"].items()},
    )


def load_pxdesign_config(path: Path | str) -> PXDesignConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(PXDesignConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown pxdesign config keys: {sorted(unknown)}")
    for required in ("binder_length", "target"):
        if required not in raw:
            raise ValueError(f"pxdesign config must define '{required}'")

    rest = {k: v for k, v in raw.items() if k not in ("binder_length", "target")}
    if "gpu_preferences" in rest:
        rest["gpu_preferences"] = tuple(rest["gpu_preferences"])

    cfg = PXDesignConfig(
        binder_length=raw["binder_length"],
        target=_parse_target(raw["target"]),
        **rest,
    )
    if cfg.network_volume_id and not cfg.data_center_id:
        raise ValueError("data_center_id is required when network_volume_id is set")
    if cfg.preset not in ("preview", "extended", "infer"):
        raise ValueError(f"preset must be preview, extended, or infer, got {cfg.preset!r}")
    if cfg.dtype not in ("fp32", "bf16"):
        raise ValueError(f"dtype must be fp32 or bf16, got {cfg.dtype!r}")
    return cfg
