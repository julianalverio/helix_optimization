"""Tests for per-example random rotation augmentation in ExamplesDataset."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from twistr.pipeline.datasets.example_dataset import ExamplesDataset, random_rotation_matrix
from twistr.pipeline.features.interaction_matrix import clean_interaction_matrix

EXAMPLE_NPZ = Path("runtime/data/examples/examples/br/1brs_1_0.npz")


@pytest.fixture(autouse=True)
def _skip_if_no_example():
    if not EXAMPLE_NPZ.exists():
        pytest.skip(f"example file not present at {EXAMPLE_NPZ}")


def test_rotation_matrix_is_proper_so3():
    torch.manual_seed(0)
    for _ in range(20):
        R = random_rotation_matrix()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-5)


def test_no_rotate_is_deterministic():
    ds = ExamplesDataset([EXAMPLE_NPZ])
    a = ds[0]["coordinates"]
    b = ds[0]["coordinates"]
    assert torch.equal(a, b)


def test_rotate_changes_coords_per_call():
    ds = ExamplesDataset([EXAMPLE_NPZ], random_rotate=True)
    torch.manual_seed(0)
    a = ds[0]["coordinates"]
    b = ds[0]["coordinates"]
    # Two consecutive calls draw two different rotations.
    assert not torch.allclose(a, b)


def test_rotate_preserves_pairwise_distances():
    base = ExamplesDataset([EXAMPLE_NPZ])[0]["coordinates"][:, 1, :]   # CA only
    torch.manual_seed(0)
    rot = ExamplesDataset([EXAMPLE_NPZ], random_rotate=True)[0]["coordinates"][:, 1, :]
    d_base = (base[:, None] - base[None]).norm(dim=-1)
    d_rot = (rot[:, None] - rot[None]).norm(dim=-1)
    assert torch.allclose(d_base, d_rot, atol=1e-5)


def test_rotate_preserves_real_atom_centroid():
    torch.manual_seed(0)
    sample = ExamplesDataset([EXAMPLE_NPZ], random_rotate=True)[0]
    real = sample["atom_mask"] == 1
    centroid = sample["coordinates"][real].mean(dim=0)
    assert centroid.abs().max().item() < 1e-5


def test_rotate_preserves_clean_interaction_matrix():
    # The supervision target must be untouched by augmentation.
    base_sample = ExamplesDataset([EXAMPLE_NPZ])[0]
    torch.manual_seed(0)
    rot_sample = ExamplesDataset([EXAMPLE_NPZ], random_rotate=True)[0]

    base_batch = {k: v.unsqueeze(0) for k, v in base_sample.items()}
    rot_batch = {k: v.unsqueeze(0) for k, v in rot_sample.items()}

    base_im = clean_interaction_matrix(base_batch)
    rot_im = clean_interaction_matrix(rot_batch)
    assert torch.equal(base_im, rot_im)
