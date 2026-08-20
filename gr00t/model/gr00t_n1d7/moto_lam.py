# SPDX-License-Identifier: Apache-2.0
#
# LARA representation alignment — the official latent-motion tokenizer as the align target.
"""Loads moto's ``LatentMotionTokenizer`` (vendored under ``moto/``) and presents the two
calls the action head makes of it.

Two details are copied deliberately from the official implementation rather than
reinvented, because they define what "the LARA target" means and a plausible-looking
substitute would silently change the experiment:

* The target is the tokenizer's ``embed`` (post-``vq_down``, pre-quantisation) FLATTENED
  across motion tokens, not a mean over them. LARA predicts a single
  ``token_count * codebook_dim`` vector with one Linear, so pooling the tokens here would
  align to a strictly coarser target than the paper's.
* Pixels are ImageNet-normalised after ``/255``. moto's ViT-MAE front-end was trained that
  way and the official stack feeds it exactly that; frames arrive here as uint8 HWC, so the
  conversion belongs on this side.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

# ImageNet statistics, matching the official GR00TTransform.apply_single.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass
class MotoLamConfig:
    """Shape metadata the action head reads to size its projector."""

    image_size: int
    token_count: int
    codebook_dim: int
    align_dim: int


def _patch_legacy_vit_kwargs() -> None:
    """Let moto's 4.51-era ``self.encoder(...)`` calls run on transformers 4.57+.

    ``MFormer`` and ``LMDViTModel`` both hold a stock ``ViTEncoder`` and call it with
    ``output_attentions`` / ``output_hidden_states`` / ``return_dict``. transformers 4.57
    dropped those parameters (the encoder now records outputs declaratively), so the call
    raises ``TypeError`` on this venv while working fine on the official stack's 4.51.
    Neither flag is ever read downstream -- moto takes ``encoder_outputs[0]`` and stores the
    rest in a dataclass it does not consult -- so dropping them is behaviour-preserving.

    A no-op on transformers versions that still accept the kwargs.
    """
    import inspect

    from transformers.models.vit.modeling_vit import ViTEncoder

    if getattr(ViTEncoder, "_moto_lam_patched", False):
        return
    accepted = set(inspect.signature(ViTEncoder.forward).parameters)
    legacy = {"output_attentions", "output_hidden_states", "return_dict"}
    if legacy <= accepted:
        return
    inner = ViTEncoder.forward

    def forward(self, *a, **kw):
        return inner(self, *a, **{k: v for k, v in kw.items() if k in accepted})

    ViTEncoder.forward = forward
    ViTEncoder._moto_lam_patched = True


class MotoLam(nn.Module):
    """moto ``LatentMotionTokenizer`` behind the small contract the action head uses."""

    def __init__(self, tokenizer_path: str, image_encoder_path: Optional[str] = None):
        super().__init__()
        # Imported here rather than at module scope: moto hard-imports lpips and hydra, so a
        # run that never enables LARA should not pay for (or require) either.
        from moto.latent_motion_tokenizer.loading import load_latent_motion_tokenizer

        _patch_legacy_vit_kwargs()

        self.tokenizer = load_latent_motion_tokenizer(
            tokenizer_path, image_encoder_path=image_encoder_path
        )
        # These stay frozen even in the co-trained arm: the official stack trains the
        # tokenizer proper but never its ViT-MAE encoder or the LPIPS network.
        self.tokenizer.image_encoder.requires_grad_(False)
        self.tokenizer.loss_fn_lpips.requires_grad_(False)

        tc = int(self.tokenizer.config.m_former_config["config"]["query_num"])
        cd = int(self.tokenizer.config.codebook_dim)
        self.config = MotoLamConfig(
            image_size=int(self.tokenizer.config.decoder_config["config"]["image_size"]),
            token_count=tc,
            codebook_dim=cd,
            align_dim=tc * cd,
        )
        self.register_buffer("_mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def _pix(self, frames: Tensor) -> Tensor:
        """uint8 ``[B, H, W, C]`` (the dataset's LAM frames) -> normalised ``[B, C, H, W]``."""
        x = frames.to(self._mean.device)
        if x.ndim != 4:
            raise ValueError(f"expected [B, H, W, C] LAM frames, got {tuple(x.shape)}")
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        # Frames arrive as uint8; a caller that already scaled them would double-divide.
        if x.max() > 1.5:
            x = x / 255.0
        return (x - self._mean) / self._std

    def encode_and_lam_loss(self, frame_t: Tensor, frame_th: Tensor) -> tuple[Tensor, Tensor]:
        """Flattened pre-quantisation embedding + moto's own ``L_LAM`` (recon + LPIPS + VQ)."""
        out = self.tokenizer(
            cond_pixel_values=self._pix(frame_t),
            target_pixel_values=self._pix(frame_th),
        )
        if out.embed is None:
            raise RuntimeError("moto tokenizer returned no embed output")
        z = out.embed.reshape(out.embed.shape[0], -1)  # [B, token_count * codebook_dim]
        return z, out.loss.mean()

    def encode(self, frame_t: Tensor, frame_th: Tensor) -> Tensor:
        z, _ = self.encode_and_lam_loss(frame_t, frame_th)
        return z


def is_moto_checkpoint(path: str) -> bool:
    """True when ``path`` holds a moto tokenizer."""
    cfg = Path(path) / "config.json"
    if not cfg.is_file():
        return False
    try:
        arch = json.loads(cfg.read_text()).get("architectures") or []
    except (OSError, ValueError):
        return False
    return "LatentMotionTokenizer" in arch
