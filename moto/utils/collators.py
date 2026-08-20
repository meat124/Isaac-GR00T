"""Batch collation for latent-motion-tokenizer pretraining."""

from typing import Any

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorMixin


def collate_moto_tokenizer(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not features or any("video" not in feature for feature in features):
        raise ValueError("Tokenizer samples must all contain a video field")

    video = torch.from_numpy(np.stack([sample["video"] for sample in features]))
    if video.ndim != 6 or video.shape[1] != 2 or video.shape[2] != 1:
        raise ValueError(
            "Expected video shape [batch, 2 frames, 1 camera, height, width, channels], "
            f"got {tuple(video.shape)}"
        )

    video = video.permute(0, 1, 2, 5, 3, 4).to(torch.float32) / 255.0
    return {
        "rgb_initial": video[:, 0],
        "rgb_future": video[:, 1],
    }


class DefaultDataCollatorMotoTokenizer(DataCollatorMixin):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_moto_tokenizer(features)


class MaskedDataCollatorMotoTokenizer(DataCollatorMixin):
    """Default collation + the TARGET-frame hand mask for MFLA-style factorization.

    Samples must carry ``episode_index``/``base_index`` (MaskedStepDataset). The target
    frame is ``base + target_offset`` (the tokenizer's [0, 15] pair), clamped inside the
    store to the episode end exactly like the video loader's tail padding.
    """

    def __init__(self, mask_store, target_offset: int = 15,
                 crop_scale: float = 0.95, out_size: int = 224):
        from moto.utils.hand_masks import masks_to_model_frame
        self._store = mask_store
        self._offset = target_offset
        self._to_frame = lambda m: masks_to_model_frame(m, crop_scale, out_size)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = collate_moto_tokenizer(features)
        masks = np.stack([
            self._store.get(f["episode_index"], f["base_index"] + self._offset)
            for f in features
        ])
        batch["hand_mask"] = self._to_frame(masks)
        return batch
