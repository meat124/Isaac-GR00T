# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Batched, autograd-safe SE(3) / 6D-rotation helpers (torch).

These mirror the numpy conventions used by ``gr00t.data.state_action.pose``
so latents/predictions in the training graph stay consistent with the targets
computed offline via :class:`gr00t.data.state_action.pose.EndEffectorPose`.

**6D rotation convention (matches GR00T).** GR00T's
``EndEffectorPose._matrix_to_rot6d`` returns ``R[:2, :].flatten()`` — i.e. the
first two *rows* of the rotation matrix — and ``_rot6d_to_matrix`` Gram-Schmidt
orthonormalises those two rows. We replicate the *row* convention here so
``matrix_to_rot6d`` / ``rot6d_to_matrix`` round-trip against the numpy code
(verified in ``tests/gr00t_ext/test_rotation.py``).

All functions accept arbitrary leading batch dims (``...``).
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Convert 6D rotation to a rotation matrix via Gram-Schmidt.

    Args:
        rot6d: ``[..., 6]`` — the first two *rows* of the rotation matrix
            (GR00T convention), flattened row-major.

    Returns:
        ``[..., 3, 3]`` rotation matrix whose first two rows are the
        orthonormalised inputs.
    """
    m = rot6d.reshape(rot6d.shape[:-1] + (2, 3))
    a1, a2 = m[..., 0, :], m[..., 1, :]  # [..., 3] each
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1  # orthogonalise vs b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # rows -> [..., 3, 3]


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """Convert a rotation matrix ``[..., 3, 3]`` to 6D (first two rows) ``[..., 6]``."""
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def geodesic_distance(r1: Tensor, r2: Tensor) -> Tensor:
    """Angular distance (radians) between rotation matrices ``[..., 3, 3]``.

    Returns ``[...]`` with ``arccos((tr(R1 R2^T) - 1) / 2)``. Cosine is clamped to
    exactly ``[-1, 1]`` to absorb float overshoot (so identical rotations give 0,
    not ``arccos(1 - eps)``). Intended as an eval metric, not a loss — the
    gradient of ``arccos`` is singular at ``±1``.
    """
    rel = r1 @ r2.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.arccos(cos)


def se3_inverse(t: Tensor) -> Tensor:
    """Invert a homogeneous SE(3) transform ``[..., 4, 4]`` (assumes orthonormal R)."""
    r = t[..., :3, :3]
    trans = t[..., :3, 3]
    r_inv = r.transpose(-1, -2)
    out = torch.zeros_like(t)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -(r_inv @ trans.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def relative_pose(t0: Tensor, tt: Tensor) -> Tensor:
    """Relative transform ``inv(t0) @ tt`` — matches ``gr00t.relative_transformation``."""
    return se3_inverse(t0) @ tt


def se3_to_xyz_rot6d(t: Tensor) -> Tensor:
    """Flatten SE(3) ``[..., 4, 4]`` to ``[..., 9]`` = ``[translation(3); rot6d(6)]``."""
    trans = t[..., :3, 3]
    r6 = matrix_to_rot6d(t[..., :3, :3])
    return torch.cat((trans, r6), dim=-1)
