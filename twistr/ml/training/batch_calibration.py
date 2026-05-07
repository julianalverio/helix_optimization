"""Empirical (B, N_max) → memory calibration.

For each N_max in a sweep grid (dataset N quantiles), runs a
doubling-then-bisect search to find the largest batch size B that fits
through one forward + backward pass of `ExamplesModule._compute_losses`.
The resulting `{N_max: max_B}` table is consumed by
`LengthBucketBatchSampler` to size each batch.

Cached to `.cache/batch_calibration.json` keyed by
`sha256(GPU memory class + memory-relevant MLConfig fields)`. The hash
intentionally excludes loss weights, learning rate, data paths, and
training-loop knobs so a hyperparameter sweep over those reuses the
cache. The GPU memory class normalises the device name (e.g.
`NVIDIA A100 80GB PCIe` and `NVIDIA A100-SXM4-80GB` both map to
`A100-80GB`) so different SKUs of the same GPU + memory size share
calibration.

The probe constructs synthetic batches that match `pad_collate`'s output
schema (same dtypes and feature semantics) so memory usage faithfully
reflects what real training will see.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from twistr.ml.config import MLConfig

logger = logging.getLogger(__name__)


# Fields that affect the model's memory footprint. Anything not listed here
# (loss weights, learning rate, data paths, training-loop knobs, conditioning
# noise rates that drive only logits values, etc.) does not change activation
# or gradient memory shape.
_MEMORY_RELEVANT_FIELDS = (
    "c_s",
    "c_z",
    "pairformer_blocks",
    "n_heads_single",
    "n_heads_pair",
    "c_hidden_mul",
    "c_hidden_pair_att",
    "transition_n",
    "relpos_max_offset",
    "calibration_n_quantiles",
)


def _gpu_memory_class(device_name: str) -> str:
    """Normalise a CUDA device name to '{model}-{memory}GB' so different
    SKUs of the same GPU + memory size share a cache key. e.g.
    `NVIDIA A100 80GB PCIe` and `NVIDIA A100-SXM4-80GB` both map to
    `A100-80GB`. Falls back to the full name for unrecognised formats."""
    upper = device_name.upper()
    model = re.search(r"\b(A\d+|H\d+|V\d+|T\d+|L\d+|P\d+|RTX\s*\d+)\b", upper)
    memory = re.search(r"(\d+)\s*GB", upper)
    if model and memory:
        return f"{model.group(1).replace(' ', '')}-{memory.group(1)}GB"
    if model:
        return model.group(1).replace(" ", "")
    return device_name


def _config_hash(cfg: MLConfig, device_name_override: str | None = None) -> str:
    """Stable hash of (GPU memory class, memory-relevant MLConfig fields).
    Pass `device_name_override` to compute the key for a GPU other than the
    current one (used by the cache migration to re-key entries from before
    the memory-class normalisation was introduced)."""
    if device_name_override is not None:
        device_name = device_name_override
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    relevant = {f: getattr(cfg, f) for f in _MEMORY_RELEVANT_FIELDS}
    payload = json.dumps(
        {"device": _gpu_memory_class(device_name), "cfg": relevant},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def synthetic_batch(B: int, N: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a batch matching `pad_collate`'s output schema. Half the residues
    are flagged as helix (chain_slot 0), half as partner (chain_slot 1);
    coordinates and atom_mask are filled with valid heavy-atom data so
    `build_features` and the geometry losses run their full code paths."""
    half = N // 2
    chain_slot = torch.zeros(B, N, dtype=torch.long, device=device)
    chain_slot[:, half:] = 1
    is_helix = torch.zeros(B, N, dtype=torch.bool, device=device)
    is_helix[:, :half] = True
    return {
        "coordinates": torch.randn(B, N, 14, 3, device=device),
        "atom_mask": torch.ones(B, N, 14, dtype=torch.int8, device=device),
        "residue_type": torch.randint(0, 20, (B, N), device=device, dtype=torch.long),
        "chain_slot": chain_slot,
        "is_helix": is_helix,
        "is_interface_residue": torch.rand(B, N, device=device) < 0.2,
        "padding_mask": torch.ones(B, N, dtype=torch.bool, device=device),
    }


def _try_step(module, B: int, N: int, device: torch.device) -> bool:
    """One forward + backward at (B, N). True if it fits, False on CUDA OOM.
    Resets gradient and CUDA peak-memory state so the next probe sees a clean
    allocator."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        batch = synthetic_batch(B, N, device)
        # `training=False`: the probe module isn't attached to a trainer, so
        # `self.global_step` (used inside the train-mode coord-MSE annealing
        # path) would fail. The val-mode path goes through the same losses
        # with the same activation memory — just different scalar weights.
        losses = module._compute_losses(batch, training=False)
        total = sum(losses.values())
        total.backward()
    except torch.cuda.OutOfMemoryError:
        for p in module.parameters():
            p.grad = None
        torch.cuda.empty_cache()
        return False
    for p in module.parameters():
        p.grad = None
    torch.cuda.empty_cache()
    return True


def find_max_b(try_step_fn: Callable[[int], bool], b_cap: int = 1024) -> int:
    """Doubling probe to bracket OOM, then bisect to the largest fitting B.
    Returns 0 if even B=1 fails. Returns b_cap if b_cap fits (no upper bracket).
    `try_step_fn` is parameterized so unit tests can drive a synthetic OOM
    boundary without a GPU."""
    last_ok = 0
    B = 1
    while B <= b_cap:
        if try_step_fn(B):
            last_ok = B
            B *= 2
        else:
            break
    if last_ok == 0:
        return 0
    if B > b_cap:
        return last_ok
    lo, hi = last_ok, B
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if try_step_fn(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _quantile_grid(lengths: Sequence[int], quantiles: Sequence[float]) -> list[int]:
    """Integer N values at the given quantiles of the dataset's N distribution,
    deduplicated and sorted ascending."""
    arr = np.asarray(lengths, dtype=np.int64)
    return sorted({int(np.quantile(arr, q)) for q in quantiles})


def calibrate(
    module,
    lengths: Sequence[int],
    quantiles: Sequence[float],
    device: torch.device,
) -> dict[int, int]:
    """Build the (N_max → max_B) lookup table by probing each quantile of the
    dataset N distribution. Raises if the smallest probed N can't fit B=1
    (the model is too big for this GPU)."""
    grid = _quantile_grid(lengths, quantiles)
    table: dict[int, int] = {}
    for N in grid:
        max_b = find_max_b(lambda B: _try_step(module, B, N, device))
        if max_b == 0:
            raise RuntimeError(
                f"Calibration failed: B=1 OOMs at N={N}. "
                "Either the GPU is too small or the model config is too large."
            )
        logger.info("calibration: N=%d → max_B=%d", N, max_b)
        table[N] = max_b
    return table


def load_or_build_calibration(
    module,
    lengths: Sequence[int],
    cfg: MLConfig,
    cache_path: Path,
) -> dict[int, int]:
    """Look up a cached calibration table for the current (GPU, cfg) key; if
    none, run the probe and persist. On CPU the OOM probe is meaningless, so
    we return a single-entry table {max(lengths): 1} — callers fall through
    to a static B=1 schedule."""
    key = _config_hash(cfg)
    cache: dict[str, dict[str, int]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if key in cache:
        logger.info("calibration: loaded cached table (key=%s)", key)
        table = {int(k): int(v) for k, v in cache[key].items()}
    else:
        device = next(module.parameters()).device
        if device.type != "cuda":
            return {int(max(lengths)): 1}

        logger.info("calibration: building table (key=%s)", key)
        table = calibrate(module, lengths, cfg.calibration_n_quantiles, device)
        cache[key] = {str(k): v for k, v in table.items()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    # Cap each entry at `cfg.max_B_cap`. The OOM probe maximises GPU memory
    # use, but on a network-filesystem dataset a single worker has to load
    # every example in a batch sequentially, so very large B starves the
    # GPU on I/O. Capping trades GPU memory for I/O parallelism.
    cap = cfg.max_B_cap
    if cap > 0:
        capped = {n: min(b, cap) for n, b in table.items()}
        if any(b != table[n] for n, b in capped.items()):
            logger.info("calibration: capped table at max_B=%d → %s", cap, capped)
        table = capped
    return table
