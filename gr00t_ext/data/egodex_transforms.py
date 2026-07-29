# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""EgoDex coordinate transforms: ARKit-origin SE(3) -> relative pose targets.

EgoDex stores each joint as an ``N x 4 x 4`` SE(3) trajectory in a fixed ARKit
world frame (Y-up, metres); ``transforms/camera`` is the head pose. We supervise
two latents with the relative motion over horizon ``H`` (frame ``t`` -> ``t+H``):

* **z_head** — relative head/camera pose ``inv(cam_t) @ cam_{t+H}`` (9D xyz+rot6d).
* **z_manip** — relative *gripper* pose + opening per hand. Following
  ``/lustre/meat124/ml-egodex-gripper``, each hand is retargeted to a parallel-jaw
  gripper (Kabsch on 4 finger joints) — so the manipulation latent supervises a
  robot-transferable gripper action, not the raw hand SE(3) (``gripper.py``).

Manip target layout (matches the reference's action layout — poses then openings)::

    [ gripper_rel_pose(9) per hand ... ; opening_{t+H}(1) per hand ... ]
    both hands -> 20D;  one hand -> 10D

The frame of the gripper delta is an ablation (``manip_target_frame``): ``head``
(gripper expressed in camera frame, decoupling head motion) or ``world``.

Rotation is GR00T's rot6d (first two matrix rows, via ``EndEffectorPose``) for
Stage-B compatibility — the reference uses the column convention internally; only
self-consistency within this pipeline matters.
"""

from __future__ import annotations

from typing import Literal

from gr00t.data.state_action.pose import EndEffectorPose, relative_transformation
import numpy as np
from numpy.typing import NDArray

from .gripper import GRIPPER_MAX_WIDTH, gripper_pose_opening


ManipFrame = Literal["head", "world"]
SIDES = ("left", "right")
POSE_DIM = 9  # [translation(3); rot6d(6)]


def se3_to_xyz_rot6d(transform: NDArray) -> NDArray[np.float64]:
    """Convert a 4x4 SE(3) matrix to a 9D ``[xyz; rot6d]`` vector (rotation renormalised)."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"expected (4, 4) SE(3) matrix, got {transform.shape}")
    out = EndEffectorPose(homogeneous=transform).xyz_rot6d
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite pose produced from SE(3) input")
    return out


def relative_camera_pose(cam_t: NDArray, cam_t_h: NDArray) -> NDArray[np.float64]:
    """z_head target: relative head/camera pose ``inv(cam_t) @ cam_t_h`` -> (9,)."""
    t_rel = relative_transformation(np.asarray(cam_t, np.float64), np.asarray(cam_t_h, np.float64))
    return se3_to_xyz_rot6d(t_rel)


def scale_pose_translation(pose: NDArray, scale: float, block: int = POSE_DIM) -> NDArray:
    """Scale the translation (first 3 of each ``block``) by ``scale`` (opening dims untouched)."""
    if scale == 1.0:
        return pose
    pose = np.asarray(pose, dtype=np.float64).copy()
    n_blocks = pose.shape[0] // block
    for b in range(n_blocks):  # only the pose blocks; trailing opening scalars are left as-is
        start = b * block
        pose[start : start + 3] *= scale
    return pose


def manip_hand_order(manip_hands: str) -> tuple[str, ...]:
    """Resolve ``manip_hands`` ('left'|'right'|'both') to ordered hand sides."""
    if manip_hands in SIDES:
        return (manip_hands,)
    if manip_hands == "both":
        return SIDES
    raise ValueError(f"manip_hands must be 'left'|'right'|'both', got {manip_hands!r}")


def relative_gripper_pose(
    cam_t: NDArray,
    cam_t_h: NDArray,
    gripper_t: NDArray,
    gripper_t_h: NDArray,
    frame: ManipFrame,
) -> NDArray[np.float64]:
    """Relative gripper pose over the horizon, in head or world frame -> (9,)."""
    if frame == "head":
        g_t = relative_transformation(cam_t, gripper_t)  # gripper in camera frame at t
        g_t_h = relative_transformation(cam_t_h, gripper_t_h)
        t_rel = relative_transformation(g_t, g_t_h)
    elif frame == "world":
        t_rel = relative_transformation(gripper_t, gripper_t_h)
    else:
        raise ValueError(f"manip_target_frame must be 'head' or 'world', got {frame!r}")
    return se3_to_xyz_rot6d(t_rel)


def compute_relative_targets(
    cam_t: NDArray,
    cam_t_h: NDArray,
    joints_t: dict[str, dict[str, NDArray]],
    joints_t_h: dict[str, dict[str, NDArray]],
    manip_hands: str = "both",
    manip_target_frame: ManipFrame = "head",
    max_width: float = GRIPPER_MAX_WIDTH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble (head_target [9], manip_target [10*n_hands]).

    ``joints_t`` / ``joints_t_h`` map a hand side ('left'/'right') to its 4 finger
    joint positions (role -> (3,)), as read from the HDF5 transforms.
    """
    head_target = relative_camera_pose(cam_t, cam_t_h)
    poses, openings = [], []
    for side in manip_hand_order(manip_hands):
        g_t, _ = gripper_pose_opening(joints_t[side], max_width)
        g_t_h, opening_t_h = gripper_pose_opening(joints_t_h[side], max_width)
        poses.append(relative_gripper_pose(cam_t, cam_t_h, g_t, g_t_h, manip_target_frame))
        openings.append(opening_t_h)
    manip_target = np.concatenate([*poses, np.asarray(openings, dtype=np.float64)])
    return head_target, manip_target
