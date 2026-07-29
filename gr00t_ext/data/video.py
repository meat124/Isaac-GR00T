# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Frame-accurate MP4 reading via OpenCV.

We use OpenCV (already a GR00T dependency, with a self-contained FFmpeg) instead
of ``gr00t.utils.video_utils.get_frames_by_indices`` (torchcodec), because
torchcodec needs system FFmpeg shared libraries that aren't on the default
library path here. Frame accuracy matters: frame ``i`` must line up with
``transforms[i]`` in the HDF5, so we count every frame from 0 with ``grab()``
rather than relying on (keyframe-snapping) timestamp seeks.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames_by_indices(
    path: str | Path, indices: list[int], size: int | None = None
) -> list[np.ndarray]:
    """Return RGB uint8 ``[H, W, 3]`` frames at ``indices`` (frame-accurate).

    Seeks to the first wanted frame with ``CAP_PROP_POS_FRAMES`` (decoding only
    from the preceding keyframe, ~1 GOP) then decodes forward, taking wanted
    frames with ``read()`` and skipping the rest with ``grab()``. For these EgoDex
    H.264 clips the seek lands exactly on the requested frame — verified
    pixel-identical to a full decode-from-zero scan (see ``_read_frames_from_zero``
    and ``tests/gr00t_ext/test_video_seek.py``) — but avoids decoding every frame from
    0, which dominated dataloading on long clips. If ``size`` is given, frames are
    resized to ``size x size`` on the CPU (INTER_AREA) in the dataloader workers.
    """
    if not indices:
        return []
    wanted = sorted(set(indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    frames: dict[int, np.ndarray] = {}
    target = set(wanted)
    start, max_idx = wanted[0], wanted[-1]
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while i <= max_idx:
            if i in target:
                ok, bgr = cap.read()  # grab + retrieve frame i
                if not ok:
                    break
                if size is not None:
                    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
                frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif not cap.grab():  # skip frame i (demux only)
                break
            i += 1
    finally:
        cap.release()

    missing = [idx for idx in wanted if idx not in frames]
    if missing:
        raise IndexError(f"could not read frames {missing} from {path}")
    return [frames[idx] for idx in indices]


def _read_frames_from_zero(
    path: str | Path, indices: list[int], size: int | None = None
) -> list[np.ndarray]:
    """Reference decoder: scan every frame from 0 with ``grab()``. Slower but
    unambiguously frame-accurate; used to validate the seek path."""
    if not indices:
        return []
    wanted = sorted(set(indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    frames: dict[int, np.ndarray] = {}
    target = set(wanted)
    max_idx = wanted[-1]
    i = 0
    try:
        while i <= max_idx:
            if not cap.grab():
                break
            if i in target:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                if size is not None:
                    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
                frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()

    missing = [idx for idx in wanted if idx not in frames]
    if missing:
        raise IndexError(f"could not read frames {missing} from {path}")
    return [frames[idx] for idx in indices]
