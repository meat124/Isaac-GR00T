# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert an AV-ALOHA simulation dataset (LeRobot v1.6) to GR00T LeRobot v2.1.

The AV-ALOHA HF datasets (e.g. ``iantc104/gv_sim_slot_insertion_3arms``) store:
  * a single ``data/train-*.parquet`` with all episodes (21-d state/action for
    3 arms, 14-d for 2 arms),
  * per-camera, per-episode AV1-encoded mp4s under ``videos/``,
  * ``meta_data/info.json`` (fps, codec).

GR00T expects LeRobot v2.1: one parquet + one mp4 per camera *per episode*, plus
``meta/{info,episodes,tasks,modality}.json``. GR00T's only video backend
(torchcodec) does not reliably decode AV1, so every video is transcoded to H.264.

Example
-------
    python examples/AVAloha/convert_avaloha_to_gr00t.py \
        --task slot_insertion --num-arms 3 \
        --output-dir datasets/avaloha_slot_insertion_3arms \
        --cameras all
"""

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# avaloha_layout lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaloha_layout import (  # noqa: E402
    CAMERAS_BY_NUM_ARMS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    TASKS,
    default_instruction,
    dim_for,
    groups_for,
    hf_repo_id,
    state_dim,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 1000  # GR00T/LeRobot default episodes-per-chunk; we only use chunk-000.


def _read_yaml_defaults() -> dict:
    """Best-effort read of avaloha_config.yaml for task/num_arms defaults."""
    cfg_path = os.environ.get("AVALOHA_CONFIG") or str(
        Path(__file__).with_name("avaloha_config.yaml")
    )
    try:
        import yaml

        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    d = _read_yaml_defaults()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--task",
        default=d.get("task", "slot_insertion"),
        help="Task short-name (insert_peg, slot_insertion, sew_needle, tube_transfer, hook_package).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Pool multiple tasks into ONE language-conditioned dataset ('all' or a list of "
        "names). One model trained on this handles all tasks (the GR00T convention). "
        "Default: just --task.",
    )
    p.add_argument("--num-arms", type=int, default=int(d.get("num_arms", 3)), choices=[2, 3])
    p.add_argument(
        "--action-space",
        default=d.get("action_space", "eef"),
        choices=["joint", "eef"],
        help="State/action space: 'joint' (raw joint targets) or 'eef' (per-arm "
        "end-effector pose xyz+rot6d, via FK). Default from YAML (eef).",
    )
    p.add_argument("--repo-id", default=None, help="Override HF dataset repo id.")
    p.add_argument(
        "--src-dir",
        default=None,
        help="Use an already-downloaded raw dataset directory instead of HF.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output GR00T dataset dir (default: datasets/avaloha_<task>_<N>arms).",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["all"],
        help="Cameras to include ('all' or explicit names). Default: all available.",
    )
    p.add_argument(
        "--max-episodes", type=int, default=None, help="Limit number of episodes (for quick tests)."
    )
    p.add_argument(
        "--instruction", default=None, help="Language instruction (default: per-task built-in)."
    )
    p.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better, larger).")
    p.add_argument("--num-workers", type=int, default=4, help="Parallel ffmpeg transcodes.")
    p.add_argument(
        "--no-stats", action="store_true", help="Skip stats.json / relative_stats.json generation."
    )
    p.add_argument("--overwrite", action="store_true", help="Delete output dir if it exists.")
    return p.parse_args()


def download_raw(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[convert] downloading {repo_id} from HuggingFace ...")
    local = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return Path(local)


def load_source_frame(src_dir: Path) -> tuple[pd.DataFrame, int]:
    """Load and concatenate all source parquet shards; return (df, fps)."""
    parquets = sorted((src_dir / "data").glob("train-*.parquet"))
    if not parquets:
        parquets = sorted((src_dir / "data").glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet found under {src_dir / 'data'}")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    fps = 50
    info_path = src_dir / "meta_data" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            fps = int(json.load(f).get("fps", 50))
    return df, fps


def transcode_av1_to_h264(src: Path, dst: Path, crf: int) -> None:
    """Re-encode one mp4 to H.264 yuv420p, preserving frame count/rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def write_episode_parquet(
    rows: pd.DataFrame,
    dst: Path,
    ep_new: int,
    index_start: int,
    fps: int,
    kin=None,
    task_index: int = 0,
) -> int:
    """Write one episode parquet in GR00T schema. Returns the new running index.

    If ``kin`` (AvalohaKinematics) is given, the raw joint state/action vectors are
    mapped to end-effector pose vectors (eef_9d per arm) via forward kinematics.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    states = [np.asarray(x, dtype=np.float32) for x in rows["observation.state"].to_numpy()]
    actions = [np.asarray(x, dtype=np.float32) for x in rows["action"].to_numpy()]
    if kin is not None:
        states = [kin.fk(s) for s in states]
        actions = [kin.fk(a) for a in actions]
    frame_index = np.arange(n, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array((frame_index / fps).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(n, ep_new, dtype=np.int64), type=pa.int64()),
            "index": pa.array(
                np.arange(index_start, index_start + n, dtype=np.int64), type=pa.int64()
            ),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, dst)
    return index_start + n


def build_info(
    num_arms: int,
    cameras: list[str],
    n_episodes: int,
    total_frames: int,
    fps: int,
    action_space: str = "joint",
    n_tasks: int = 1,
) -> dict:
    dim = dim_for(action_space, num_arms)
    prefix = "eef" if action_space == "eef" else "joint"
    feat_names = [f"{prefix}_{i}" for i in range(dim)]
    features = {
        "action": {"dtype": "float32", "shape": [dim], "names": feat_names},
        "observation.state": {"dtype": "float32", "shape": [dim], "names": feat_names},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": IMAGE_HEIGHT,
                "video.width": IMAGE_WIDTH,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        features[k] = {
            "dtype": "float32" if k == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": f"av_aloha_{num_arms}arm",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": n_tasks,
        "total_videos": n_episodes * len(cameras),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_modality(num_arms: int, cameras: list[str], action_space: str = "joint") -> dict:
    groups = groups_for(action_space, num_arms)
    state = {g: {"start": s, "end": e} for g, (s, e) in groups.items()}
    return {
        "state": state,
        "action": dict(state),
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def main() -> None:
    args = parse_args()

    cameras_available = list(CAMERAS_BY_NUM_ARMS[args.num_arms])
    if args.cameras == ["all"]:
        cameras = cameras_available
    else:
        bad = [c for c in args.cameras if c not in cameras_available]
        if bad:
            raise SystemExit(
                f"Cameras {bad} unavailable for num_arms={args.num_arms}. "
                f"Available: {cameras_available}"
            )
        cameras = list(args.cameras)

    # Resolve the task list (single task, or pooled multi-task).
    if args.tasks:
        tasks = list(TASKS.keys()) if args.tasks == ["all"] else list(args.tasks)
    else:
        tasks = [args.task]
    multitask = len(tasks) > 1
    suffix = "_eef" if args.action_space == "eef" else ""
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif multitask:
        out_dir = (
            REPO_ROOT / "datasets" / f"avaloha_multitask{len(tasks)}_{args.num_arms}arms{suffix}"
        )
    else:
        out_dir = REPO_ROOT / "datasets" / f"avaloha_{tasks[0]}_{args.num_arms}arms{suffix}"
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output dir {out_dir} exists. Use --overwrite to replace.")

    kin = None
    if args.action_space == "eef":
        from avaloha_kinematics import AvalohaKinematics

        # Arm FK is task-independent (same robot), so one kinematics object serves all tasks.
        kin = AvalohaKinematics(num_arms=args.num_arms, task=tasks[0])
        print(f"[convert] FK joints -> eef_9d (dim {dim_for('eef', args.num_arms)})")

    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    print(
        f"[convert] tasks={tasks}, cameras={cameras}, action_space={args.action_space} -> {out_dir}"
    )

    episodes_meta = []
    tasks_jsonl = []  # one entry per task -> language instruction
    transcode_jobs = []  # (src, dst)
    running_index = 0
    total_frames = 0
    ep_new = 0
    fps = 50
    for ti, task in enumerate(tasks):
        repo_id = (
            args.repo_id if (args.repo_id and not multitask) else hf_repo_id(task, args.num_arms)
        )
        instruction = (
            args.instruction if (args.instruction and not multitask) else default_instruction(task)
        )
        tasks_jsonl.append({"task_index": ti, "task": instruction})
        src_dir = Path(args.src_dir) if (args.src_dir and not multitask) else download_raw(repo_id)
        df, fps = load_source_frame(src_dir)
        got = len(np.asarray(df["observation.state"].iloc[0]))
        if got != state_dim(args.num_arms):
            raise SystemExit(
                f"State dim {got} != {state_dim(args.num_arms)} for num_arms={args.num_arms} "
                f"(task {task}). Did you set --num-arms correctly?"
            )
        ep_ids = sorted(df["episode_index"].unique().tolist())
        if args.max_episodes is not None:
            ep_ids = ep_ids[: args.max_episodes]
        print(f"[convert]   task[{ti}] '{task}' ({repo_id}): {len(ep_ids)} episodes")
        for ep_orig in ep_ids:
            rows = (
                df[df["episode_index"] == ep_orig].sort_values("frame_index").reset_index(drop=True)
            )
            n = len(rows)
            pq_dst = out_dir / "data" / "chunk-000" / f"episode_{ep_new:06d}.parquet"
            running_index = write_episode_parquet(
                rows, pq_dst, ep_new, running_index, fps, kin=kin, task_index=ti
            )
            episodes_meta.append({"episode_index": ep_new, "tasks": [instruction], "length": n})
            total_frames += n
            for cam in cameras:
                src_vid = src_dir / "videos" / f"observation.images.{cam}_episode_{ep_orig:06d}.mp4"
                if not src_vid.exists():
                    raise FileNotFoundError(f"Missing source video: {src_vid}")
                dst_vid = (
                    out_dir
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{cam}"
                    / f"episode_{ep_new:06d}.mp4"
                )
                transcode_jobs.append((src_vid, dst_vid))
            ep_new += 1

    n_episodes = ep_new

    print(
        f"[convert] transcoding {len(transcode_jobs)} videos (AV1 -> H.264) with {args.num_workers} workers ..."
    )
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(transcode_av1_to_h264, s, d, args.crf) for s, d in transcode_jobs]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            fut.result()
            if i % 20 == 0 or i == len(futures):
                print(f"[convert]   {i}/{len(futures)} videos done")

    # ---- meta files ----
    with open(out_dir / "meta" / "info.json", "w") as f:
        json.dump(
            build_info(
                args.num_arms, cameras, n_episodes, total_frames, fps, args.action_space, len(tasks)
            ),
            f,
            indent=4,
        )
    with open(out_dir / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(out_dir / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks_jsonl:
            f.write(json.dumps(t) + "\n")
    with open(out_dir / "meta" / "modality.json", "w") as f:
        json.dump(build_modality(args.num_arms, cameras, args.action_space), f, indent=4)

    print(
        f"[convert] wrote {n_episodes} episodes ({len(tasks)} tasks), {total_frames} frames to {out_dir}"
    )

    if not args.no_stats:
        cfg_path = os.environ.get("AVALOHA_CONFIG") or str(
            Path(__file__).with_name("avaloha_config.yaml")
        )
        env = {**os.environ, "AVALOHA_CONFIG": cfg_path}
        cmd = [
            sys.executable,
            str(REPO_ROOT / "gr00t" / "data" / "stats.py"),
            "--dataset-path",
            str(out_dir),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(Path(__file__).with_name("avaloha_config.py")),
        ]
        print(f"[convert] generating stats: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
        print("[convert] stats generated.")

    print(f"[convert] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
