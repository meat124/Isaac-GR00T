# SPDX-License-Identifier: Apache-2.0
#
# SaPaVe — active perception + manipulation, research extension on Isaac GR00T N1.7.
"""Convert HD-EPIC (HDF5 + MP4) to LeRobot v2.1 for the SaPaVe Stage-1 corpus.

Manifest-driven sibling of ``egodex_to_lerobot_v2.py``. HD-EPIC is already converted to
the EgoDex on-disk schema at ``/scratch2/meat124/dataset/hd-epic``; this step only
re-packages it as LeRobot v2.1 so the **stock** GR00T data path (statistics, modality
config, and the ``mix_ratio`` mixture machinery) can be used unchanged.

Why HD-EPIC and not EgoDex, for Stage 1: median head angular velocity is 12.2 deg/s vs
EgoDex's 2.97 (same protocol, 400 episodes each) — EgoDex is ~4x too static to teach
camera control — and **every** HD-EPIC episode carries a real natural-language sentence.

Four deltas from the EgoDex converter:

1. Episode discovery is the manifest parquet, not a directory walk (HD-EPIC uses parallel
   ``hdf5/`` and ``clips/`` trees, which ``egodex_paths.discover_episodes`` cannot express).
2. The instruction is the manifest's ``narration`` sentence ("Open the upper cupboard by
   holding the handle of the cupboard with the left hand."), **not** the directory verb
   slug ("open"). The HDF5 stores no text, so this is a manifest-side join by filename.
3. ``mp4_path`` already points at a 256x256 clip, so it is symlinked directly (no
   ``--video-root`` remap and no re-encode).
4. The video key is named ``zed_cam_left`` to match AV-ALOHA, so Stage 2 can mix both
   corpora under one embodiment tag without relying on positional video-key auto-mapping.

Note HD-EPIC HDF5s carry no ``confidences/*`` group, but the 29-D frame only reads
``transforms/camera`` plus the 4 finger joints per hand, so no guard is needed.

    python gr00t_ext/scripts/hdepic_to_lerobot.py \
        --manifest /lustre/meat124/runs/groot_runs/hdepic_manifest_carego.parquet \
        --out /lustre/meat124/datasets/hdepic_lerobot --splits hdepic_train
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from gr00t_ext.embodiment.egodex_embodiment import manip_head_modality_json
from gr00t_ext.scripts.egodex_to_lerobot_v2 import (
    CHUNK_SIZE,
    IMG_H,
    IMG_W,
    STATE_DIM,
    _episode_frame,
    write_stats,
)
import h5py
import numpy as np
import pandas as pd


FPS = 30  # HD-EPIC clips are 30 fps (AV-ALOHA is 50 — see the plan's temporal-rate note)
ROBOT_TYPE = "human_hdepic_manip_head"
# Named to match AV-ALOHA so one NEW_EMBODIMENT tag covers both corpora in the Stage-2
# mixture (see module docstring, delta 4).
VIDEO_KEY = "zed_cam_left"
VIDEO_ORIGINAL_KEY = f"observation.images.{VIDEO_KEY}"
DEFAULT_MANIFEST = "/lustre/meat124/runs/groot_runs/hdepic_manifest_carego.parquet"


def clean_narration(text: object) -> str:
    """Normalize a raw HD-EPIC narration into an instruction string.

    The shipped narrations carry a leading space and irregular internal whitespace
    (e.g. ``" Open the upper cupboard by holding the handle ..."``); left as-is they
    would show up verbatim in the language conditioning and split otherwise-identical
    instructions into distinct task vocabulary entries.
    """
    return " ".join(str(text).split())


def load_manifest(manifest: Path, splits: tuple[str, ...], limit: int | None) -> pd.DataFrame:
    """Rows of the HD-EPIC manifest for ``splits``, in a stable order."""
    df = pd.read_parquet(manifest)
    missing = {"hdf5_path", "mp4_path", "narration", "n_frames", "episode_id", "split"} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"{manifest} is missing required columns: {sorted(missing)}")
    df = df[df["split"].isin(splits)].copy()
    if df.empty:
        raise ValueError(f"no rows for splits={splits} in {manifest}")
    df["narration"] = df["narration"].map(clean_narration)
    # episode_id is unique per (video, narration segment); sorting makes the episode_index
    # assignment reproducible across runs.
    df = df.sort_values("episode_id", kind="stable").reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


def convert_episode(
    record: dict, episode_index: int, out: Path, task_index: int
) -> tuple[int, dict]:
    """Write one episode's parquet + video symlink; return (n_frames, sidecar row)."""
    with h5py.File(record["hdf5_path"], "r") as h5:
        n = int(h5["transforms/camera"].shape[0])
        states = np.stack([_episode_frame(h5, t) for t in range(n)])  # [N, 29]
    rows = {
        "observation.state": list(states),
        "action": list(states),  # GR00T's RELATIVE ActionConfig deltas this at load
        "task_index": np.full(n, task_index, np.int64),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, episode_index, np.int64),
        "index": np.arange(n, dtype=np.int64) + episode_index * 1_000_000,
        "timestamp": (np.arange(n) / FPS).astype(np.float32),
    }
    data_dir = out / f"data/chunk-{0:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)

    video_dir = out / f"videos/chunk-{0:03d}/{VIDEO_ORIGINAL_KEY}"
    video_dir.mkdir(parents=True, exist_ok=True)
    link = video_dir / f"episode_{episode_index:06d}.mp4"
    link.unlink(missing_ok=True)
    link.symlink_to(Path(record["mp4_path"]).resolve())  # already 256x256, no re-encode

    sidecar = {
        "episode_index": episode_index,
        "hdf5_path": record["hdf5_path"],
        "episode_id": record["episode_id"],
        "n_frames": n,
        "manifest_n_frames": int(record["n_frames"]),
        "narration": record["narration"],
    }
    return n, sidecar


def write_meta(out: Path, lengths: list[int], task_of_episode: list[int], tasks: list[str]) -> None:
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "total_tasks": len(tasks),
        "chunks_size": CHUNK_SIZE,  # all episodes in chunk-000
        "fps": FPS,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [STATE_DIM]},
            VIDEO_ORIGINAL_KEY: {
                "dtype": "video",
                "shape": [IMG_H, IMG_W, 3],
                "info": {"video.fps": FPS, "video.height": IMG_H, "video.width": IMG_W},
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    with (meta / "episodes.jsonl").open("w") as f:
        for i, (length, ti) in enumerate(zip(lengths, task_of_episode)):
            f.write(json.dumps({"episode_index": i, "tasks": [tasks[ti]], "length": length}) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for i, task in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")
    (meta / "modality.json").write_text(json.dumps(manip_head_modality_json([VIDEO_KEY]), indent=2))


def write_sidecar(out: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(out / "meta" / "hdepic_sidecar.parquet", index=False)


def _convert_one(job: tuple) -> tuple[int, int, dict]:
    record, i, out, task_idx = job
    n, sidecar = convert_episode(record, i, out, task_idx)
    return i, n, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HD-EPIC -> LeRobot v2.1 (29-D manip;head)"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["hdepic_train"])
    parser.add_argument("--limit", type=int, default=None, help="max episodes (debug)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.out)
    df = load_manifest(Path(args.manifest), tuple(args.splits), args.limit)
    records = df.to_dict("records")
    print(f"{len(records)} episodes from splits={args.splits}", flush=True)

    # Language conditioning: the narration sentence itself is the task string. Many
    # narrations repeat verbatim across episodes, so the vocabulary is the unique set.
    task_names = sorted({r["narration"] for r in records})
    task_to_idx = {name: i for i, name in enumerate(task_names)}
    task_of_episode = [task_to_idx[r["narration"]] for r in records]
    jobs = [(rec, i, out, task_of_episode[i]) for i, rec in enumerate(records)]
    print(
        f"{len(task_names)} unique narrations; converting with {args.workers} workers", flush=True
    )

    results: list[tuple[int, dict] | None] = [None] * len(jobs)
    done = 0
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, n, sidecar in pool.imap_unordered(_convert_one, jobs, chunksize=8):
                results[i] = (n, sidecar)
                done += 1
                if done % 2000 == 0:
                    print(f"converted {done}/{len(jobs)} episodes", flush=True)
    else:
        for job in jobs:
            i, n, sidecar = _convert_one(job)
            results[i] = (n, sidecar)

    lengths = [results[i][0] for i in range(len(jobs))]
    sidecar_rows = [results[i][1] for i in range(len(jobs))]

    # The manifest's n_frames drives window counting downstream, so a mismatch against the
    # HDF5 would silently mis-slice episodes. Fail loudly instead.
    bad = [r for r in sidecar_rows if r["n_frames"] != r["manifest_n_frames"]]
    if bad:
        raise ValueError(
            f"{len(bad)} episodes disagree with the manifest on n_frames, e.g. {bad[:3]}"
        )

    write_meta(out, lengths, task_of_episode, task_names)
    write_stats(out)
    write_sidecar(out, sidecar_rows)
    print(
        f"done. {len(records)} episodes, {sum(lengths)} frames, {len(task_names)} narrations -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
