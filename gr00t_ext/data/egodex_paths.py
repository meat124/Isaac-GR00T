# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""EgoDex episode discovery and manifest building.

EgoDex is laid out as ``<root>/<split>/<task>/<id>.hdf5`` with a co-located
``<id>.mp4``. There are ~169k episodes, so we build a parquet manifest once
(``scripts/egodex_build_manifest.py``) and read it at train time instead of
re-walking the tree.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import pandas as pd


# EgoDex splits. ``test`` is held out for validation by default.
TRAIN_SPLITS = ("part1", "part2", "part3", "part4", "part5", "extra")
VAL_SPLIT = "test"
CAMERA_KEY = "transforms/camera"


def episode_num_frames(hdf5_path: str) -> int:
    """Read the frame count from an EgoDex HDF5 (shape only, no data load)."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            return int(f[CAMERA_KEY].shape[0])
    except (OSError, KeyError):
        return -1


def discover_episodes(root: str | Path, splits: tuple[str, ...]) -> list[dict]:
    """List ``{hdf5_path, mp4_path, task, split, episode_id}`` for episodes with video."""
    root = Path(root)
    records: list[dict] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for hdf5_path in sorted(split_dir.glob("*/*.hdf5")):
            mp4_path = hdf5_path.with_suffix(".mp4")
            if not mp4_path.exists():
                continue
            task = hdf5_path.parent.name
            records.append(
                {
                    "hdf5_path": str(hdf5_path),
                    "mp4_path": str(mp4_path),
                    "task": task,
                    "split": split,
                    "episode_id": f"{split}/{task}/{hdf5_path.stem}",
                }
            )
    return records


def build_manifest(
    root: str | Path,
    splits: tuple[str, ...],
    out_path: str | Path,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Discover episodes, attach frame counts, and write a parquet manifest."""
    records = discover_episodes(root, splits)
    paths = [r["hdf5_path"] for r in records]
    if num_workers > 1 and paths:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            frames = list(ex.map(episode_num_frames, paths, chunksize=64))
    else:
        frames = [episode_num_frames(p) for p in paths]
    for record, n in zip(records, frames):
        record["n_frames"] = n

    df = pd.DataFrame(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def load_manifest(path: str | Path, min_frames: int | None = None) -> pd.DataFrame:
    """Load a manifest parquet, optionally dropping episodes shorter than ``min_frames``."""
    df = pd.read_parquet(path)
    if min_frames is not None:
        df = df[df["n_frames"] >= min_frames].reset_index(drop=True)
    return df
