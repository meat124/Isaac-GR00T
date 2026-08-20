# SPDX-License-Identifier: Apache-2.0
#
# LARA representation alignment (arXiv:2606.07100) on the N1.7 action head.
"""Alignment objective, DiT token pooling, and the ``f_psi`` projector.

The released LARA objective is ``mean(1 - cos(pred, target))`` between a projection of one
DiT hidden token and the latent-motion tokenizer's pre-quantisation embedding. Everything
here is that objective plus diagnostics; the anti-collapse guards are default-off so the
paper arm runs unmodified.
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PoolMode = Literal["last", "action", "state", "all"]


def latent_alignment_loss(
    pred: Tensor,
    target: Tensor,
    *,
    center: bool = False,
    w_var: float = 0.0,
    var_floor: float = 0.4,
    w_cov: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """LARA Eq. 6 cosine plus the optional anti-collapse guards.

    Returns ``(loss, stats)``. With every guard off and ``center=False`` this is exactly
    the released objective; the extra tensors are then diagnostics only. Computed in fp32:
    the residual of a DC-dominated target rides on a mean ~15x its size, past a bf16
    mantissa, and the released tokenizer's embedding is DC-dominated (a constant predictor
    scores cosine 0.997 on it).

    center - subtract each side's batch mean before the cosine, so matching the constant
             batch mean scores 0 instead of ~1.
    w_var  - hinge pulling each latent dim's batch std up to ``var_floor``, blocking the
             amplitude-shrink escape that centring alone leaves open.
    w_cov  - squared off-diagonal batch correlation of the target, blocking the remaining
             rank-1 escape (all variance in one direction).

    Both guards are dimensionless so the weights transfer across latents of different
    scale: the raw ``mean(relu(floor - std))`` and ``off_diag_sq / D`` forms scale as
    ``std`` and ``std**4``, which made one tuned set of weights ~800x weaker on a latent
    whose healthy per-dim std was 0.116 instead of 0.617. ``var_floor`` is then the only
    scale-carrying knob, set from the measured healthy std.
    """
    p = pred.float()
    z = target.float()
    stats: dict[str, Tensor] = {}
    with torch.no_grad():
        n = z.shape[0]
        mu = z.mean(dim=0)
        zc0 = z - mu
        dc = mu.pow(2).sum() / (mu.pow(2).sum() + zc0.pow(2).sum() / max(n, 1) + 1e-12)
        stats["latent_align_raw"] = (1 - F.cosine_similarity(p, z, dim=-1)).mean()
        stats["latent_align_centered"] = (
            1 - F.cosine_similarity(p - p.mean(0, keepdim=True), zc0, dim=-1)
        ).mean()
        stats["latent_dc_fraction"] = dc
        stats["latent_z_std"] = z.std(dim=0).mean() if n > 1 else z.new_zeros(())

    if center and p.shape[0] > 1:
        p = p - p.mean(dim=0, keepdim=True)
        z = z - z.mean(dim=0, keepdim=True)
    loss = (1 - F.cosine_similarity(p, z, dim=-1)).mean()

    if w_var > 0 and z.shape[0] > 1:
        var = F.relu(1.0 - z.std(dim=0) / max(var_floor, 1e-8)).mean()
        stats["latent_var_hinge"] = var.detach()
        loss = loss + w_var * var
    if w_cov > 0 and z.shape[0] > 1:
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        diag = cov.diagonal()
        denom = diag.clamp(min=1e-12).sqrt()
        corr = cov / (denom[:, None] * denom[None, :])
        d = z.shape[1]
        cov_pen = (corr.pow(2).sum() - d) / max(d * (d - 1), 1)
        stats["latent_cov_penalty"] = cov_pen.detach()
        loss = loss + w_cov * cov_pen
    return loss, stats


@torch.no_grad()
def latent_stats(z: Tensor, prefix: str) -> dict[str, Tensor]:
    """Collapse-guard stats for a ``[B, D]`` latent batch.

    ``<prefix>_eff_rank`` is the participation ratio of the covariance eigenvalues: a
    trivially matched align target shows std -> 0 and eff_rank -> 1 early, long before the
    loss curve gives it away. The SVD makes this logging-cadence-only -- do not call it
    every training step.
    """
    z = z.detach().float()
    if z.shape[0] < 2:  # std/rank undefined for a single sample
        zero = z.new_zeros(())
        return {f"{prefix}_std": zero, f"{prefix}_eff_rank": zero, f"{prefix}_mean_norm": zero}
    mean = z.mean(dim=0, keepdim=True)
    eigvals = torch.linalg.svdvals(z - mean).pow(2)
    eff_rank = eigvals.sum().pow(2) / (eigvals.pow(2).sum() + 1e-12)
    return {
        f"{prefix}_std": z.std(dim=0).mean(),
        f"{prefix}_eff_rank": eff_rank,
        f"{prefix}_mean_norm": mean.squeeze(0).norm(),
    }


def pool_dit_tokens(
    hidden: Tensor,
    action_horizon: int,
    mode: PoolMode = "last",
    chunk_len: Optional[Tensor] = None,
) -> Tensor:
    """Pool a DiT hidden state ``[B, S, D]`` (``S = state tokens + action_horizon``) to ``[B, D]``.

    The action slot is PADDED: the processor pads every chunk to the model config's
    ``action_horizon`` (40 for the released N1.7) with zero actions and a zeroed
    ``action_mask``, so a 16-step dataset chunk occupies action tokens 0..15 and tokens
    16..39 are loss-masked padding whose input embedding is pure flow noise. ``chunk_len``
    ``[B]`` (derived from ``action_mask``) marks the true chunk end per sample.

    ``last`` (LARA paper A.2) is the token of the FINAL VALID action step, so the
    completed-trajectory representation matches the LAM's predicted visual effect. Without
    ``chunk_len`` it falls back to the final action TOKEN, which is a pad token whenever the
    dataset chunk is shorter than the model horizon. ``action`` averages the valid action
    tokens; ``state`` is the single state token; ``all`` averages every token (pads
    included; diagnostic only).
    """
    act = hidden[:, -action_horizon:, :]  # [B, H, D] action-token block (pads at the tail)
    if mode == "last":
        if chunk_len is None:
            return act[:, -1, :]
        idx = (chunk_len.long().clamp(min=1) - 1).to(act.device)
        return act[torch.arange(act.shape[0], device=act.device), idx, :]
    if mode == "action":
        if chunk_len is None:
            return act.mean(dim=1)
        mask = (
            torch.arange(act.shape[1], device=act.device)[None, :]
            < chunk_len.long().to(act.device)[:, None]
        ).to(act.dtype)
        return (act * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    if mode == "state":
        return hidden[:, 0, :]
    if mode == "all":
        return hidden.mean(dim=1)
    raise ValueError(f"unknown pool mode: {mode!r}")


def latent_motion_head_width(head: nn.Module) -> int:
    """Output width of a projector built by ``build_latent_motion_head``."""
    last = head if isinstance(head, nn.Linear) else head[-1]
    return last.out_features


def build_latent_motion_head(in_dim: int, out_dim: int, num_layers: int = 0) -> nn.Module:
    """``f_psi``, mapping a DiT token to the flattened latent-motion target.

    ``num_layers <= 0`` is the paper's single ``Linear``. That layer can satisfy a
    centring-free cosine with ``W = 0, b = mu_z`` -- it need not read the DiT at all -- so
    the deeper variants exist for the guarded arms.
    """
    if num_layers <= 0:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = [nn.Linear(in_dim, in_dim)]
    for i in range(num_layers):
        layers += [
            nn.LayerNorm(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim if i == num_layers - 1 else in_dim),
        ]
    return nn.Sequential(*layers)
