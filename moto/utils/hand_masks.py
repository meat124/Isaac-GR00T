# SPDX-License-Identifier: Apache-2.0
"""CaRe-Ego hand(+held object) masks for EgoDex tokenizer pretraining.

Store layout (precomputed 2026-07, full corpus 338,234 episodes):
  /scratch2/meat124/egodex_masks/carego_hoi_256/<part>/<task>/<ep>.npz
    hand_bits: [n_frames, 8192] uint8 packbits of a 256x256 binary mask per frame
    meta:      json string {"model": "CaRe-Ego", "dilate_px": 2, "shape": [...]}

The LeRobot episode_index maps to that path through the dataset's sidecar parquet
(episode_index -> episode_id like "part1/add_remove_lid/0" -> <root>/<episode_id>.npz).

Geometry: the video pipeline random-crops (train) at scale 0.95 then resizes to 224.
Masks are center-cropped at the same scale and area-resized; the residual train-time
misalignment is <= 2.5% of extent, well inside MFLA's own mask granularity (it pools
masks to 16x16 before use). Mask fetch is one npz member decompress per sample (~2 MB),
negligible next to the video decode.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_MASK_ROOT = "/scratch2/meat124/egodex_masks/carego_hoi_256"
DEFAULT_SIDECAR = "/lustre/meat124/datasets/egodex_lerobot/meta/egodex_sidecar.parquet"
MASK_SIZE = 256


class HandMaskStore:
    """episode_index + frame -> 256x256 uint8 mask, resolved via the sidecar."""

    def __init__(self, root: str = DEFAULT_MASK_ROOT, sidecar: str = DEFAULT_SIDECAR):
        self.root = Path(root)
        df = pd.read_parquet(sidecar, columns=["episode_index", "episode_id", "n_frames"])
        self._episode_id = dict(zip(df["episode_index"], df["episode_id"]))
        self._n_frames = dict(zip(df["episode_index"], df["n_frames"]))

    @functools.lru_cache(maxsize=8)
    def _bits(self, episode_index: int) -> np.ndarray:
        path = self.root / f"{self._episode_id[episode_index]}.npz"
        with np.load(path) as z:
            return z["hand_bits"]

    def get(self, episode_index: int, frame: int) -> np.ndarray:
        """Binary mask [256, 256] for a frame (clamped to the episode end, matching the
        video loader's pad-with-last behaviour on short tails)."""
        bits = self._bits(int(episode_index))
        t = min(int(frame), bits.shape[0] - 1, int(self._n_frames[int(episode_index)]) - 1)
        return np.unpackbits(bits[max(t, 0)]).reshape(MASK_SIZE, MASK_SIZE)


def masks_to_model_frame(
    masks: np.ndarray, crop_scale: float = 0.95, out_size: int = 224
) -> torch.Tensor:
    """[B, 256, 256] binary -> [B, out, out] float in {0,1} in the model's pixel frame:
    center crop at ``crop_scale`` then area resize, binarised at 0.5."""
    m = torch.from_numpy(np.ascontiguousarray(masks)).float().unsqueeze(1)  # [B,1,H,W]
    side = int(round(MASK_SIZE * crop_scale))
    off = (MASK_SIZE - side) // 2
    m = m[:, :, off : off + side, off : off + side]
    m = F.interpolate(m, size=(out_size, out_size), mode="area")
    return (m.squeeze(1) > 0.5).float()
