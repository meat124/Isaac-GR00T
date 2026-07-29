# SPDX-License-Identifier: Apache-2.0
#
# Latent-action NTP pipeline — research extension on top of Isaac GR00T N1.7.
"""Standardization of the Stage A pose-regression targets.

The LAM readouts regress relative poses in PHYSICAL units: translations in metres
next to rot6d entries and a [0, 1] gripper opening. The blocks live on different
scales, and since 99.5% of the targets fall in smooth_l1's quadratic regime, each
block's contribution to the loss goes as the SQUARE of its spread. Measured on
EgoDex (30.5k transitions, h=50, mean-centered — the rot6d DC term is a constant
the readout bias fits for free and carries no signal):

    head  trans 0.0129   rot6d 0.0335    -> rot6d ~2.6x  the spread (~7x the loss)
    hand  trans 0.056 / 0.108            -> rot6d ~4-6x  the spread (~20-35x the loss)
    hand  rot6d 0.319 / 0.395   opening 0.249

The head translation block is the starved one, and the baseline arm shows it:
lam_headhand_pix_h50 plateaus at head_trans_err 0.0112 m against a no-skill (predict
the mean) reference of 0.0126 m — 11% better than a constant. Its hands, whose
translation spread is 4-8x larger, reach 0.0384 m against a no-skill 0.0857 m.

Standardizing removes the imbalance WITHOUT distorting the pose geometry: every
block is centered per-dim and divided by ONE scalar (its RMS). A rot6d block is
therefore only uniformly scaled — it still inverts exactly, so the physical-unit
metrics (translation error in metres, geodesic rotation error in degrees) stay
comparable across a normalized and an unnormalized run.

Stats are computed once over the corpus (``scripts/egodex_compute_target_stats.py``)
and read at train time. They depend on the target definition, so ``meta`` records
the knobs that change it (horizon, manip hands/frame, translation_scale) and
:meth:`TargetStats.check_compatible` refuses a mismatched file rather than
silently normalizing with the wrong constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# One pose block = translation(3) + rot6d(6). Trailing dims (gripper openings) form
# their own block: they are already O(1) and carry no rotation geometry.
POSE_BLOCK = 9
TRANS_DIM = 3

# A block whose RMS is below this is treated as constant — divide by 1.0 instead of
# exploding it into noise.
SCALE_FLOOR = 1e-4

# Target-definition knobs the stats depend on; a mismatch changes the constants. The
# horizon matters most: refitting at h=15 halves every pose scale (h=100 raises them ~30%),
# so a stats file is only valid for the horizon it was fitted at.
META_KEYS = ("horizon", "manip_hands", "manip_target_frame", "translation_scale")

REPO_ROOT = Path(__file__).resolve().parents[2]


def block_slices(dim: int) -> list[slice]:
    """Blocks of a target vector: each 9-D pose block split into trans(3) + rot6d(6),
    plus one trailing block for whatever remains (the gripper openings)."""
    slices: list[slice] = []
    n_pose = dim // POSE_BLOCK
    for b in range(n_pose):
        start = b * POSE_BLOCK
        slices.append(slice(start, start + TRANS_DIM))
        slices.append(slice(start + TRANS_DIM, start + POSE_BLOCK))
    if n_pose * POSE_BLOCK < dim:
        slices.append(slice(n_pose * POSE_BLOCK, dim))
    return slices


def _fit(samples: NDArray) -> tuple[NDArray, NDArray]:
    """Per-dim mean and per-BLOCK scalar scale (broadcast to per-dim) for ``[N, D]``."""
    mean = samples.mean(axis=0)
    centered = samples - mean
    scale = np.ones(samples.shape[1], dtype=np.float64)
    for sl in block_slices(samples.shape[1]):
        rms = float(np.sqrt((centered[:, sl] ** 2).mean()))
        scale[sl] = rms if rms > SCALE_FLOOR else 1.0
    return mean, scale


@dataclass
class TargetStats:
    """Centering/scaling constants for the head (9-D) and manip (20-D) targets."""

    head_mean: NDArray
    head_scale: NDArray
    manip_mean: NDArray
    manip_scale: NDArray
    meta: dict[str, Any]

    @classmethod
    def fit(cls, head: NDArray, manip: NDArray, meta: dict[str, Any]) -> TargetStats:
        head_mean, head_scale = _fit(np.asarray(head, dtype=np.float64))
        manip_mean, manip_scale = _fit(np.asarray(manip, dtype=np.float64))
        return cls(head_mean, head_scale, manip_mean, manip_scale, meta)

    def normalize_head(self, x: NDArray) -> NDArray:
        return (x - self.head_mean) / self.head_scale

    def normalize_manip(self, x: NDArray) -> NDArray:
        return (x - self.manip_mean) / self.manip_scale

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "head_mean": self.head_mean.tolist(),
            "head_scale": self.head_scale.tolist(),
            "manip_mean": self.manip_mean.tolist(),
            "manip_scale": self.manip_scale.tolist(),
            "meta": self.meta,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, indent=2))
        tmp.replace(path)  # atomic: a half-written file never replaces a good one

    @classmethod
    def load(cls, path: str | Path) -> TargetStats:
        blob = json.loads(Path(path).read_text())
        return cls(
            head_mean=np.asarray(blob["head_mean"], dtype=np.float64),
            head_scale=np.asarray(blob["head_scale"], dtype=np.float64),
            manip_mean=np.asarray(blob["manip_mean"], dtype=np.float64),
            manip_scale=np.asarray(blob["manip_scale"], dtype=np.float64),
            meta=blob.get("meta", {}),
        )

    def check_compatible(self, expected: dict[str, Any]) -> None:
        """Raise if the stats were fitted to a different target definition."""
        bad = {
            k: (self.meta.get(k), expected[k])
            for k in META_KEYS
            if k in expected and self.meta.get(k) != expected[k]
        }
        if bad:
            detail = ", ".join(
                f"{k}: stats={got!r} config={want!r}" for k, (got, want) in bad.items()
            )
            raise ValueError(
                f"target stats were fitted to a different target definition ({detail}). "
                f"Recompute with scripts/egodex_compute_target_stats.py."
            )


def load_target_stats(path: str | Path | None, expected: dict[str, Any]) -> TargetStats | None:
    """Load + validate the stats file, or return None when normalization is off.

    A relative path resolves against the repo root, so the constants travel with the
    code (they are ~2 KB and horizon-specific — versioning them next to the config that
    uses them beats parking them in the dataset directory)."""
    if not path:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"target stats not found: {path}\n"
            f"Fit them (~20 s) with:\n"
            f"  python scripts/egodex_compute_target_stats.py "
            f"--horizon {expected.get('horizon', 50)} --out {path}"
        )
    stats = TargetStats.load(path)
    stats.check_compatible(expected)
    return stats
