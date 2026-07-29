# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Torch ``Dataset`` yielding (o_t, o_t+H) pairs with relative-pose targets.

Each item samples a frame ``t`` and its horizon partner ``t + H`` from one EgoDex
episode and returns the raw uint8 frames for on-the-fly DINO encoding (the M-Former
consumes patch tokens, so features cannot be precomputed/pooled ahead of time), plus
the relative head/hand pose targets and a deterministic supervision flag.
"""

from __future__ import annotations

import hashlib

import h5py
import numpy as np
from torch.utils.data import Dataset

from .egodex_targets import open_h5
from .egodex_transforms import compute_relative_targets, manip_hand_order, scale_pose_translation
from .gripper import joint_keys
from .target_stats import TargetStats
from .video import read_frames_by_indices


CAMERA_KEY = "transforms/camera"


def deterministic_supervision_flag(episode_id: str, t: int, seed: int, ratio: float) -> bool:
    """Stable per-(episode, frame) supervision decision, reproducible across workers.

    Uses a salted SHA-1 (not Python ``hash``, which is per-process randomised) so
    the supervised subset is fixed across epochs and supervision-ratio ablations
    are nested.
    """
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    digest = hashlib.sha1(f"{episode_id}:{t}:{seed}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") % 10_000) / 10_000.0 < ratio


class EgoDexLAMDataset(Dataset):
    """(o_t, o_t+H) pairs for Stage A LAM pretraining."""

    def __init__(
        self,
        records: list[dict],
        horizon: int,
        manip_hands: str = "both",
        manip_target_frame: str = "head",
        supervision_ratio: float = 1.0,
        pairs_per_episode: int = 8,
        min_confidence: float = 0.0,
        translation_scale: float = 1.0,
        image_size: int | None = None,
        seed: int = 42,
        max_resample: int = 8,
        target_stats: TargetStats | None = None,
    ) -> None:
        self.records = records
        self.horizon = horizon
        self.manip_hands = manip_hands
        self.manip_sides = manip_hand_order(manip_hands)
        self.joint_keys = {side: joint_keys(side) for side in self.manip_sides}
        self.manip_target_frame = manip_target_frame
        self.supervision_ratio = supervision_ratio
        self.pairs_per_episode = pairs_per_episode
        self.min_confidence = min_confidence
        self.translation_scale = translation_scale
        self.image_size = image_size
        self.seed = seed
        self.max_resample = max_resample
        self.target_stats = target_stats

    def __len__(self) -> int:
        return len(self.records) * self.pairs_per_episode

    def _confidence_ok(self, h5: h5py.File, t: int, t_h: int) -> bool:
        if self.min_confidence <= 0.0:
            return True
        # Gate on hand-tracking confidence only. EgoDex `confidences` holds ARKit body/hand
        # joint scores (no `camera` key — the head/camera pose is the AR device's own VIO
        # pose, always reliable), and this filter targets noisy HAND labels (the manip_rot
        # bottleneck), so the wrist `{side}Hand` score is the right per-hand proxy.
        # ~14% of EgoDex episodes ship no `confidences` group (and a file may miss a given
        # joint); we can't assess those, so KEEP the sample rather than crash or drop 14%
        # of the corpus by a property unrelated to label quality.
        conf_grp = h5.get("confidences")
        if conf_grp is None:
            return True
        joints = tuple(f"{side}Hand" for side in self.manip_sides)
        for joint in joints:
            if joint not in conf_grp:
                continue
            conf = conf_grp[joint]
            if float(conf[t]) < self.min_confidence or float(conf[t_h]) < self.min_confidence:
                return False
        return True

    def _sample_t(self, h5: h5py.File, n_frames: int, idx: int) -> int:
        rng = np.random.default_rng([self.seed, idx])
        high = n_frames - self.horizon  # t in [0, high) so t + H <= n_frames - 1
        t = int(rng.integers(0, high))
        for _ in range(self.max_resample):
            if self._confidence_ok(h5, t, t + self.horizon):
                return t
            t = int(rng.integers(0, high))
        return t  # give up: return last draw

    def _read_joints(self, h5: h5py.File, t: int) -> dict[str, dict[str, np.ndarray]]:
        """Per-side dict of the 4 finger-joint world positions at frame ``t``."""
        return {
            side: {role: np.asarray(h5[key][t][:3, 3]) for role, key in keys.items()}
            for side, keys in self.joint_keys.items()
        }

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx // self.pairs_per_episode]
        episode_id = record["episode_id"]
        h5 = open_h5(record["hdf5_path"])
        n_frames = int(record["n_frames"])
        t = self._sample_t(h5, n_frames, idx)
        t_h = t + self.horizon

        cam = h5[CAMERA_KEY]
        cam_t, cam_t_h = np.asarray(cam[t]), np.asarray(cam[t_h])
        joints_t = self._read_joints(h5, t)
        joints_t_h = self._read_joints(h5, t_h)

        head_target, manip_target = compute_relative_targets(
            cam_t, cam_t_h, joints_t, joints_t_h, self.manip_hands, self.manip_target_frame
        )
        head_target = scale_pose_translation(head_target, self.translation_scale)
        manip_target = scale_pose_translation(manip_target, self.translation_scale)
        if self.target_stats is not None:
            head_target = self.target_stats.normalize_head(head_target)
            manip_target = self.target_stats.normalize_manip(manip_target)
        sup = deterministic_supervision_flag(episode_id, t, self.seed, self.supervision_ratio)

        item: dict = {
            "head_target": head_target.astype(np.float32),
            "manip_target": manip_target.astype(np.float32),
            "sup_mask": np.float32(1.0 if sup else 0.0),
            "episode_id": episode_id,
            "t": t,
        }
        frame_t, frame_t_h = read_frames_by_indices(
            record["mp4_path"], [t, t_h], size=self.image_size
        )
        item["frame_t"] = np.asarray(frame_t, dtype=np.uint8)
        item["frame_t_h"] = np.asarray(frame_t_h, dtype=np.uint8)
        return item
