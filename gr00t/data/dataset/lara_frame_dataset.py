# SPDX-License-Identifier: Apache-2.0
#
# LARA representation alignment — the frame pair the latent-motion tokenizer encodes.
"""``LaraFrameDataset`` — GR00T VLA samples plus the ``(frame[t], frame[t+H])`` pair.

The official stack gets this pair by asking the data pipeline for video observation
indices ``[0, H]``, which would also hand the VLM a second frame and move the policy off
the anchor's input distribution. Attaching the pair as extra keys keeps the VLA input
identical to the baseline's, so the two arms differ only by the alignment terms.

``Gr00tN1d7DataCollator`` ``np.stack``s any extra key and ``prepare_input`` passes the
batch through whole, so these arrive at the action head with no collator change (keep
``remove_unused_columns`` False).
"""

from typing import Any

import numpy as np
from PIL import Image

from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset


class LaraFrameDataset(ShardedSingleStepDataset):
    """ShardedSingleStepDataset that also yields the LAM's frame pair.

    Args (beyond the base class):
        lam_horizon: partner offset ``H`` (frame ``t -> t+H``).
        lam_image_size: side length the frames are resized to (the tokenizer's input).
        video_key: modality video key (``zed_cam_left`` for AV-ALOHA).
    """

    def __init__(
        self,
        *args: Any,
        lam_horizon: int,
        lam_image_size: int = 224,
        video_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lam_horizon = lam_horizon
        self.lam_image_size = lam_image_size
        self.video_key = video_key

    def get_shard(self, idx: int) -> list:
        datapoints = []
        for ep_idx, step_indices in self.sharded_episodes[idx]:
            episode_data = self.episode_loader[ep_idx]
            n = len(episode_data)
            for step_index in step_indices:
                t = int(step_index)
                item = self.get_datapoint(episode_data, t)
                t_h = t + self.lam_horizon
                item["lam_frame_t"] = self._frame(episode_data, t)
                # Steps within H of the episode end have no true partner frame; they are
                # clamped so the batch stacks, and masked out of both losses.
                item["lam_frame_th"] = self._frame(episode_data, min(t_h, n - 1))
                item["lam_valid"] = np.float32(1.0 if t_h <= n - 1 else 0.0)
                datapoints.append(item)
        return datapoints

    def _frame(self, episode_data, t: int) -> np.ndarray:
        """RGB ``uint8`` HWC frame at ``t`` resized to ``lam_image_size``."""
        frame = episode_data[f"video.{self.video_key}"].iloc[t]
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame))
        frame = frame.convert("RGB")
        if frame.size != (self.lam_image_size, self.lam_image_size):
            frame = frame.resize((self.lam_image_size, self.lam_image_size))
        return np.asarray(frame, dtype=np.uint8)
