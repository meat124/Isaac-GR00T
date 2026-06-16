"""
Real-Time Chunking (RTC) for GR00T inference.

Ported from X-VLA's rtc_xvla.py for GR00T's flow-matching (velocity prediction)
denoising loop.  Uses direct x0-blending guidance on the Euler-integrated action
estimate at each denoising step.

Reference:
    https://www.physicalintelligence.company/download/real_time_chunking.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class RTCSchedule(str, Enum):
    LINEAR = "linear"
    EXP = "exp"
    ONES = "ones"
    ZEROS = "zeros"


@dataclass
class GR00TRTCConfig:
    enabled: bool = False
    max_guidance_weight: float = 1.0
    execution_horizon: int = 10
    schedule: str = "linear"

    @classmethod
    def from_dict(cls, d: dict | None) -> "GR00TRTCConfig":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GR00TRTCProcessor:
    """Direct guidance RTC processor for GR00T flow-matching inference."""

    def __init__(self, config: GR00TRTCConfig):
        self.config = config

    def guide_prediction(
        self,
        action: Tensor,
        prev_chunk_left_over: Tensor | None,
        time_value: float,
    ) -> Tensor:
        """Apply RTC guidance to the current action estimate after an Euler step.

        For GR00T's flow-matching loop, this is called after each Euler integration:
            actions = actions + dt * pred_velocity
            actions = rtc.guide_prediction(actions, prev_chunk, noise_level)

        Args:
            action: Current action estimate, shape (B, T, A).
            prev_chunk_left_over: Unexecuted tail from previous chunk,
                shape (B, T_prev, A).  None on the first chunk.
            time_value: Noise level in [0, 1] where 1 = pure noise (beginning),
                0 = clean (end).  For GR00T: ``1.0 - (t + 1) * dt``.

        Returns:
            Guided action with the same shape as *action*.
        """
        if prev_chunk_left_over is None:
            return action

        squeezed = False
        if action.dim() < 3:
            action = action.unsqueeze(0)
            squeezed = True
        if prev_chunk_left_over.dim() < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        B, T, A = action.shape
        exec_horizon = min(self.config.execution_horizon, prev_chunk_left_over.shape[1])

        # Pad prev_chunk if shorter than current chunk
        if prev_chunk_left_over.shape[1] < T or prev_chunk_left_over.shape[2] < A:
            padded = torch.zeros(B, T, A, device=action.device, dtype=action.dtype)
            pt = min(prev_chunk_left_over.shape[1], T)
            pa = min(prev_chunk_left_over.shape[2], A)
            padded[:, :pt, :pa] = prev_chunk_left_over[:, :pt, :pa]
            prev_chunk_left_over = padded

        # Prefix weights: (T,) → (1, T, 1) for broadcasting
        weights = (
            self._get_prefix_weights(0, exec_horizon, T)
            .to(device=action.device, dtype=action.dtype)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        # Guidance weight: strong when noisy (early), weak when clean (late).
        # Clamped to [0, 1] for stable direct blending.
        gw = min(self.config.max_guidance_weight * time_value, 1.0)

        # Direct guidance: blend action toward previous chunk's unexecuted tail
        err = (prev_chunk_left_over[:, :T, :A] - action) * weights
        guided = action + gw * err

        if squeezed:
            guided = guided.squeeze(0)
        return guided

    # ------------------------------------------------------------------
    # Prefix weight schedules
    # ------------------------------------------------------------------
    def _get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)
        sched = RTCSchedule(self.config.schedule)

        if sched == RTCSchedule.ZEROS:
            w = torch.zeros(total)
            w[:start] = 1.0
        elif sched == RTCSchedule.ONES:
            w = torch.ones(total)
            w[end:] = 0.0
        elif sched == RTCSchedule.LINEAR:
            lin = self._linweights(start, end, total)
            w = self._add_trailing_zeros(lin, total, end)
            w = self._add_leading_ones(w, start, total)
        elif sched == RTCSchedule.EXP:
            lin = self._linweights(start, end, total)
            lin = lin * torch.expm1(lin).div(math.e - 1)
            w = self._add_trailing_zeros(lin, total, end)
            w = self._add_leading_ones(w, start, total)
        else:
            raise ValueError(f"Unknown schedule: {sched}")
        return w

    @staticmethod
    def _linweights(start: int, end: int, total: int) -> Tensor:
        skip = max(total - end, 0)
        n = total - skip - start
        if end <= start or n <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, n + 2)[1:-1]

    @staticmethod
    def _add_trailing_zeros(weights: Tensor, total: int, end: int) -> Tensor:
        zlen = total - end
        if zlen <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zlen)])

    @staticmethod
    def _add_leading_ones(weights: Tensor, start: int, total: int) -> Tensor:
        olen = min(start, total)
        if olen <= 0:
            return weights
        return torch.cat([torch.ones(olen), weights])
