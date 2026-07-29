# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Shared EgoDex HDF5 -> relative-pose-target reader.

Factors the per-``(episode, t, t+H)`` head/manip target computation out of
``EgoDexLAMDataset`` (Stage A) so the Stage B coupled dataset
(``gr00t_ext.stage_b.coupled_dataset``) can reuse the *identical* targets and
coordinate math without re-decoding video or duplicating the HDF5 reads. Both
callers go through :func:`relative_targets`, which wraps the same
``egodex_transforms.compute_relative_targets`` Stage A uses.

h5py handles are not fork-safe, so they are cached per-process and transparently
reopened after a fork (DataLoader workers).
"""

from __future__ import annotations

from collections import OrderedDict
import os

import h5py
import numpy as np
from numpy.typing import NDArray

from gr00t_ext.data.egodex_transforms import (
    compute_relative_targets,
    manip_hand_order,
    scale_pose_translation,
)
from gr00t_ext.data.gripper import joint_keys


CAMERA_KEY = "transforms/camera"

# Per-process h5py handle cache. h5py handles cannot cross a fork, so we key the
# cache on the current PID and drop it after a fork. The cache is a capped LRU:
# EgoDex has ~330k episodes, so an uncapped per-path cache accumulates open fds
# until the ulimit and freezes dataloader workers in Lustre i/o wait.
_HANDLES: OrderedDict[str, h5py.File] = OrderedDict()
_HANDLE_PID: int | None = None
_MAX_HANDLES = 256


def open_h5(path: str) -> h5py.File:
    """Return a process-local, fork-safe read handle to ``path`` (LRU-cached)."""
    global _HANDLE_PID, _HANDLES
    pid = os.getpid()
    if _HANDLE_PID != pid:  # forked since last open -> stale handles, reopen lazily
        _HANDLES = OrderedDict()
        _HANDLE_PID = pid
    handle = _HANDLES.get(path)
    if handle is None:
        handle = h5py.File(path, "r")
        _HANDLES[path] = handle
        while len(_HANDLES) > _MAX_HANDLES:
            _, evicted = _HANDLES.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass  # already closed / invalid handle: eviction must not fail a read
    else:
        _HANDLES.move_to_end(path)
    return handle


def manip_joint_keys(manip_hands: str) -> dict[str, dict[str, str]]:
    """``{side: {role: hdf5_key}}`` for the manipulation hand(s) of ``manip_hands``."""
    return {side: joint_keys(side) for side in manip_hand_order(manip_hands)}


def _read_joints(
    h5: h5py.File, t: int, joint_keys_by_side: dict[str, dict[str, str]]
) -> dict[str, dict[str, NDArray]]:
    """Per-side dict of the 4 finger-joint world positions at frame ``t``."""
    return {
        side: {role: np.asarray(h5[key][t][:3, 3]) for role, key in keys.items()}
        for side, keys in joint_keys_by_side.items()
    }


def relative_targets(
    h5: h5py.File,
    t: int,
    t_h: int,
    joint_keys_by_side: dict[str, dict[str, str]],
    manip_hands: str = "both",
    manip_target_frame: str = "head",
    translation_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Relative ``(head_target [9], manip_target [10*n_hands])`` over ``t -> t_h``.

    Identical to the target computation in ``EgoDexLAMDataset.__getitem__`` — head
    is the relative camera pose; manip is the per-hand gripper rel-pose + opening.
    """
    cam = h5[CAMERA_KEY]
    cam_t, cam_t_h = np.asarray(cam[t]), np.asarray(cam[t_h])
    joints_t = _read_joints(h5, t, joint_keys_by_side)
    joints_t_h = _read_joints(h5, t_h, joint_keys_by_side)
    head_target, manip_target = compute_relative_targets(
        cam_t, cam_t_h, joints_t, joints_t_h, manip_hands, manip_target_frame
    )
    head_target = scale_pose_translation(head_target, translation_scale)
    manip_target = scale_pose_translation(manip_target, translation_scale)
    return head_target, manip_target
