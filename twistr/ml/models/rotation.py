from __future__ import annotations

import torch


def rotation_from_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert a 6D continuous rotation parameterization (Zhou et al. 2019) to a
    3x3 rotation matrix via Gram-Schmidt. Input shape (..., 6); output (..., 3, 3).
    Continuous everywhere, no singularities — well-behaved when supervised
    only through coordinate-space losses."""
    a, b = x[..., :3], x[..., 3:]
    e1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b - (e1 * b).sum(dim=-1, keepdim=True) * e1
    e2 = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Inverse of rotation_from_6d: take the first two columns of R, concatenate
    them into a 6-vector. Input (..., 3, 3); output (..., 6). Lossless: applying
    rotation_from_6d to the result reconstructs R exactly (modulo numerics)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def frame_from_three_points(
    N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor,
) -> torch.Tensor:
    """Build the per-residue rotation matrix from atom positions (N, CA, C) in
    the AF2 convention: x-axis along CA→C, y-axis perpendicular in the N-CA-C
    plane with positive y-component (so N has positive y in the canonical frame),
    z-axis = x × y. All inputs shape (..., 3); output (..., 3, 3)."""
    x_axis = C - CA
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n_hat = N - CA
    n_hat = n_hat - (x_axis * n_hat).sum(dim=-1, keepdim=True) * x_axis
    y_axis = n_hat / n_hat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)
