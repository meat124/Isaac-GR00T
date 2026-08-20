# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Hand -> parallel-jaw gripper retargeting (mirrors /lustre/meat124/lam_ws/ml-egodex-gripper).

EgoDex is human hand tracking; the manipulation stream of the ``[manip ; head]``
embodiment is a *gripper*, so we map each hand to a gripper pose + opening from 4
visionOS finger joints (index/thumb tip + index knuckle + thumb intermediate),
exactly as the reference ``gripper_mapping.py``:

  * TCP (gripper position) = midpoint(index_tip, thumb_tip)
  * pose recovered by Kabsch (SVD, reflection-safe) of canonical gripper points
  * opening = clip(||index_tip - thumb_tip||, 0, max_width) / max_width  in [0, 1]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# HDF5 joint = f"{side}{suffix}" (side in {left, right}).
GRIPPER_JOINT_SUFFIXES = {
    "index_tip": "IndexFingerTip",
    "thumb_tip": "ThumbTip",
    "index_knuckle": "IndexFingerKnuckle",
    "thumb_inter": "ThumbIntermediateTip",
}
GRIPPER_MAX_WIDTH = 0.085  # m, parallel-jaw stroke (Robotiq 2F-85)


def joint_keys(side: str) -> dict[str, str]:
    """Map roles -> full HDF5 transform keys for one hand side."""
    return {role: f"transforms/{side}{suffix}" for role, suffix in GRIPPER_JOINT_SUFFIXES.items()}


def kabsch(p: NDArray, q: NDArray) -> NDArray:
    """Reflection-safe rotation aligning ``p -> q`` (``q ~= R @ p + t``)."""
    pc = p - p.mean(0)
    qc = q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def gripper_pose_opening(
    joints: dict[str, NDArray], max_width: float = GRIPPER_MAX_WIDTH
) -> tuple[NDArray, float]:
    """Map 4 finger-joint world positions to (gripper SE(3) 4x4, opening in [0,1])."""
    it, tt = joints["index_tip"], joints["thumb_tip"]
    ik, ti = joints["index_knuckle"], joints["thumb_inter"]
    m_tip = 0.5 * (it + tt)
    m_base = 0.5 * (ik + ti)
    width = float(np.linalg.norm(it - tt))
    length = max(float(np.linalg.norm(m_tip - m_base)), 1e-4)
    w2 = 0.5 * width
    # canonical gripper points: +x open/close axis, +z approach (base -> tip).
    canonical = np.array(
        [[+w2, 0.0, 0.0], [-w2, 0.0, 0.0], [+w2, 0.0, -length], [-w2, 0.0, -length]],
        dtype=np.float64,
    )
    observed = np.stack([it, tt, ik, ti]).astype(np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = kabsch(canonical, observed)
    pose[:3, 3] = m_tip
    opening = float(np.clip(width, 0.0, max_width)) / max_width
    return pose, opening
