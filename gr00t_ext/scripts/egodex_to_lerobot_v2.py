# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Convert EgoDex (HDF5 + MP4) to LeRobot v2.1 for the GR00T VLA pipeline (Stage B).

Writes a LeRobot v2.1 dataset for the unified **29-D ``[manip ; head]`` eef** embodiment
(``gr00t_ext.embodiment.egodex_embodiment``), per frame an absolute state/action vector

    [ left_eef(9) ; left_gripper(1) ; right_eef(9) ; right_gripper(1) ; middle_eef(9) ]

(GR00T's RELATIVE ActionConfig turns the eef poses into the delta action chunk at load).
Both hands are retargeted to a parallel-jaw gripper (Kabsch on 4 finger joints, like the
Stage A LAM); the head/camera pose is ``middle_eef`` (= the active-vision channel that
``z_head`` aligns to). Poses come from the original HDF5; the **video is symlinked to the
256x256 transcoded clips** (``egodex_small``) — GR00T resizes to 256 anyway, and this
avoids the 1080p decode cliff while keeping the full episode coverage.

A ``meta/egodex_sidecar.parquet`` map (``episode_index -> hdf5_path / episode_id /
n_frames``) is written so the Stage B coupled dataset can fetch the LAM pose targets from
the source HDF5 (byte-stable with the LeRobot episode ordering).

    python gr00t_ext/scripts/egodex_to_lerobot_v2.py \
        --egodex-root /lustre/dataset/EgoDex --video-root /scratch2/meat124/egodex_small \
        --out /lustre/meat124/egodex_lerobot --splits part1 part2 part3 part4 part5 extra
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from gr00t.data.state_action.pose import EndEffectorPose
from gr00t_ext.data.egodex_paths import discover_episodes
from gr00t_ext.data.gripper import gripper_pose_opening, joint_keys
from gr00t_ext.embodiment.egodex_embodiment import manip_head_modality_json
import h5py
import numpy as np
import pandas as pd


FPS = 30
STATE_DIM = 29  # left_eef(9)+left_grip(1)+right_eef(9)+right_grip(1)+middle_eef(9)
# All episodes are written to chunk-000; chunks_size must exceed the episode count so the
# LeRobot loader maps every episode_index -> chunk-000 (episode_index // chunks_size == 0).
CHUNK_SIZE = 10_000_000
IMG_H, IMG_W = 256, 256  # egodex_small transcode resolution
VIDEO_KEY = "ego_view"
VIDEO_ORIGINAL_KEY = f"observation.images.{VIDEO_KEY}"
_JOINT_KEYS = {side: joint_keys(side) for side in ("left", "right")}


def _hand_eef_opening(h5: h5py.File, t: int, side: str) -> tuple[np.ndarray, float]:
    """Retarget one hand's 4 finger joints to (gripper eef_9d, opening in [0, 1])."""
    joints = {role: np.asarray(h5[key][t][:3, 3]) for role, key in _JOINT_KEYS[side].items()}
    pose, opening = gripper_pose_opening(joints)
    return EndEffectorPose(homogeneous=pose).xyz_rot6d, opening


def _episode_frame(h5: h5py.File, t: int) -> np.ndarray:
    """The 29-D absolute state/action vector at frame ``t``."""
    l_eef, l_open = _hand_eef_opening(h5, t, "left")
    r_eef, r_open = _hand_eef_opening(h5, t, "right")
    head = EndEffectorPose(homogeneous=np.asarray(h5["transforms/camera"][t])).xyz_rot6d
    return np.concatenate([l_eef, [l_open], r_eef, [r_open], head]).astype(np.float32)  # [29]


def _small_video_path(mp4_path: str, egodex_root: Path, video_root: Path) -> Path:
    """Map a 1080p EgoDex MP4 path to its 256x256 transcode under ``video_root``."""
    return video_root / Path(mp4_path).resolve().relative_to(egodex_root.resolve())


def convert_episode(
    record: dict,
    episode_index: int,
    out: Path,
    egodex_root: Path,
    video_root: Path,
    task_index: int,
    chunk: int = 0,
) -> tuple[int, dict]:
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
    data_dir = out / f"data/chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)

    video_dir = out / f"videos/chunk-{chunk:03d}/{VIDEO_ORIGINAL_KEY}"
    video_dir.mkdir(parents=True, exist_ok=True)
    link = video_dir / f"episode_{episode_index:06d}.mp4"
    link.unlink(missing_ok=True)
    link.symlink_to(_small_video_path(record["mp4_path"], egodex_root, video_root))

    sidecar = {
        "episode_index": episode_index,
        "hdf5_path": record["hdf5_path"],
        "episode_id": record["episode_id"],
        "n_frames": n,
    }
    return n, sidecar


def write_meta(out: Path, lengths: list[int], task_of_episode: list[int], tasks: list[str]) -> None:
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "human_egodex_manip_head",
        "total_episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "total_tasks": len(tasks),
        "chunks_size": CHUNK_SIZE,  # all episodes in chunk-000 (see CHUNK_SIZE note)
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


def write_stats(out: Path, max_episodes: int = 2000) -> None:
    """Compute mean/std/min/max/q01/q99 over state+action from a sample of parquets.

    GR00T's own ``generate_stats`` recomputes exact dataset statistics at training
    time, so a bounded sample here keeps the (single-pass) converter memory-safe on
    the full corpus while still writing a reasonable ``meta/stats.json``.
    """
    paths = sorted((out / "data").rglob("*.parquet"))[:max_episodes]
    arrays = [pd.read_parquet(p)["observation.state"].to_list() for p in paths]
    # Episodes have different frame counts, so concatenate along the time axis
    # (np.stack would require equal lengths).
    data = (
        np.concatenate([np.stack(a) for a in arrays], axis=0)
        if arrays
        else np.zeros((1, STATE_DIM))
    )
    stat = {
        "mean": data.mean(0).tolist(),
        "std": (data.std(0) + 1e-6).tolist(),
        "min": data.min(0).tolist(),
        "max": data.max(0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }
    (out / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": stat, "action": stat}, indent=2)
    )


def write_sidecar(out: Path, rows: list[dict]) -> None:
    """episode_index -> {hdf5_path, episode_id, n_frames} for the Stage B coupled dataset."""
    pd.DataFrame(rows).to_parquet(out / "meta" / "egodex_sidecar.parquet", index=False)


def _convert_one(job: tuple) -> tuple[int, int, dict]:
    """Pool worker: convert one episode -> (episode_index, n_frames, sidecar)."""
    record, i, out, egodex_root, video_root, task_idx = job
    n, sidecar = convert_episode(record, i, out, egodex_root, video_root, task_idx)
    return i, n, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EgoDex -> LeRobot v2.1 (29-D manip;head)")
    parser.add_argument(
        "--egodex-root", default="/lustre/dataset/EgoDex", help="HDF5 (pose) source"
    )
    parser.add_argument(
        "--video-root",
        default="/scratch2/meat124/egodex_small",
        help="256x256 transcoded MP4 source for the video symlinks",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["part1", "part2", "part3", "part4", "part5"]
    )
    parser.add_argument("--limit", type=int, default=None, help="max episodes (debug)")
    parser.add_argument("--workers", type=int, default=16, help="parallel conversion workers")
    args = parser.parse_args()

    out, egodex_root, video_root = Path(args.out), Path(args.egodex_root), Path(args.video_root)
    print(f"discovering episodes in {args.splits} ...", flush=True)
    episodes = discover_episodes(args.egodex_root, tuple(args.splits))
    if args.limit:
        episodes = episodes[: args.limit]

    # Task vocabulary (language conditioning): one index per unique EgoDex task name.
    task_names = sorted({r["task"].replace("_", " ") for r in episodes})
    task_to_idx = {name: i for i, name in enumerate(task_names)}
    jobs = [
        (rec, i, out, egodex_root, video_root, task_to_idx[rec["task"].replace("_", " ")])
        for i, rec in enumerate(episodes)
    ]
    print(f"{len(jobs)} episodes; converting with {args.workers} workers", flush=True)

    results: list[tuple[int, dict] | None] = [None] * len(jobs)
    done = 0
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, n, sidecar in pool.imap_unordered(_convert_one, jobs, chunksize=8):
                results[i] = (n, sidecar)
                done += 1
                if done % 500 == 0:
                    print(f"converted {done}/{len(jobs)} episodes", flush=True)
    else:
        for job in jobs:
            i, n, sidecar = _convert_one(job)
            results[i] = (n, sidecar)

    lengths = [results[i][0] for i in range(len(jobs))]
    sidecar_rows = [results[i][1] for i in range(len(jobs))]
    task_of_episode = [task_to_idx[rec["task"].replace("_", " ")] for rec in episodes]
    write_meta(out, lengths, task_of_episode, task_names)
    write_stats(out)
    write_sidecar(out, sidecar_rows)
    print(f"done. {len(episodes)} episodes, {len(task_names)} tasks -> {out}", flush=True)


if __name__ == "__main__":
    main()
