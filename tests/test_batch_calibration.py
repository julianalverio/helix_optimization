"""Batch calibration — probe search logic, cache key invalidation."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from twistr.ml.config import MLConfig
from twistr.ml.training.batch_calibration import (
    _config_hash,
    _gpu_memory_class,
    _quantile_grid,
    find_max_b,
)


def test_find_max_b_finds_threshold_via_doubling_then_bisect():
    """Synthetic try_step that fits up to B=12 and OOMs above. Doubling should
    bracket [last_ok=8, oom=16], then bisect to 12."""
    threshold = 12
    calls: list[int] = []

    def try_step(B: int) -> bool:
        calls.append(B)
        return B <= threshold

    assert find_max_b(try_step) == threshold
    # Doubling probes 1, 2, 4, 8, 16 (16 OOMs); bisect 8..16.
    assert calls[:5] == [1, 2, 4, 8, 16]


def test_find_max_b_returns_zero_when_b_one_oomms():
    def never_fits(B: int) -> bool:
        return False

    assert find_max_b(never_fits) == 0


def test_find_max_b_returns_b_cap_when_everything_fits():
    def always_fits(B: int) -> bool:
        return True

    assert find_max_b(always_fits, b_cap=64) == 64


def test_find_max_b_at_threshold_one():
    def fits_at_one(B: int) -> bool:
        return B == 1

    assert find_max_b(fits_at_one) == 1


def test_find_max_b_at_power_of_two_threshold():
    """If the OOM boundary is exactly at a doubling point, no bisect is needed."""
    def fits_to_eight(B: int) -> bool:
        return B <= 8

    assert find_max_b(fits_to_eight) == 8


def test_config_hash_changes_with_memory_relevant_field():
    """Architecture changes invalidate the cache."""
    h1 = _config_hash(MLConfig())
    h2 = _config_hash(MLConfig(c_s=256))
    assert h1 != h2


def test_config_hash_stable_for_same_cfg():
    cfg = MLConfig()
    assert _config_hash(cfg) == _config_hash(cfg)
    assert _config_hash(MLConfig()) == _config_hash(MLConfig())


def test_config_hash_ignores_non_memory_fields():
    """Hyperparam sweeps over loss weights / learning rate / data paths /
    seeds reuse the cache — those don't change activation memory."""
    base = MLConfig()
    assert _config_hash(base) == _config_hash(MLConfig(learning_rate=0.5))
    assert _config_hash(base) == _config_hash(MLConfig(interaction_bce_weight=99.0))
    assert _config_hash(base) == _config_hash(MLConfig(seed=42))
    assert _config_hash(base) == _config_hash(MLConfig(num_workers=8))


def test_config_hash_collapses_pcie_and_sxm():
    """A100 PCIe and A100 SXM with the same memory map to the same key —
    they have the same VRAM and OOM behavior."""
    cfg = MLConfig()
    pcie = _config_hash(cfg, device_name_override="NVIDIA A100 80GB PCIe")
    sxm = _config_hash(cfg, device_name_override="NVIDIA A100-SXM4-80GB")
    assert pcie == sxm


def test_config_hash_separates_different_memory_sizes():
    """A100 80GB and A100 40GB must NOT share a cache — their max_B differs."""
    cfg = MLConfig()
    h80 = _config_hash(cfg, device_name_override="NVIDIA A100 80GB PCIe")
    h40 = _config_hash(cfg, device_name_override="NVIDIA A100 40GB PCIe")
    assert h80 != h40


def test_gpu_memory_class_normalisation():
    assert _gpu_memory_class("NVIDIA A100 80GB PCIe") == "A100-80GB"
    assert _gpu_memory_class("NVIDIA A100-SXM4-80GB") == "A100-80GB"
    assert _gpu_memory_class("NVIDIA A100 40GB PCIe") == "A100-40GB"
    assert _gpu_memory_class("NVIDIA H100 80GB HBM3") == "H100-80GB"
    assert _gpu_memory_class("NVIDIA A40") == "A40"


def test_quantile_grid_dedupes_and_sorts():
    lengths = [50, 60, 70, 80, 90, 100]
    grid = _quantile_grid(lengths, [0.0, 0.5, 0.5, 1.0])  # duplicate 0.5
    assert grid == sorted(set(grid))
    assert grid[0] == 50 and grid[-1] == 100


def test_load_or_build_calibration_cpu_returns_singleton(tmp_path: Path):
    """On CPU the OOM probe is meaningless; loader returns a single entry
    {max(lengths): 1} so callers fall through to a static B=1 schedule."""
    from twistr.ml.training.batch_calibration import load_or_build_calibration

    class CPUModule:
        def parameters(self):
            yield torch.zeros(1)

    table = load_or_build_calibration(
        CPUModule(), [10, 20, 30], MLConfig(), tmp_path / "cache.json"
    )
    assert table == {30: 1}


def test_load_or_build_calibration_uses_cache(tmp_path: Path):
    """If a cache entry for the current (GPU, cfg) hash exists, return it
    without invoking the probe."""
    from twistr.ml.training.batch_calibration import load_or_build_calibration

    cfg = MLConfig()
    key = _config_hash(cfg)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(f'{{"{key}": {{"100": 4, "200": 1}}}}')

    class ShouldNotProbe:
        def parameters(self):
            raise AssertionError("should not be probed when cache hit")

    table = load_or_build_calibration(ShouldNotProbe(), [50, 100, 200], cfg, cache_path)
    assert table == {100: 4, 200: 1}
