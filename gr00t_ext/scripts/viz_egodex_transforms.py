# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Coordinate-convention visual gate for EgoDex relative-pose targets.

For each sampled episode this renders frame ``t`` and frame ``t+H`` side by side
and overlays the relative HEAD pose and the relative MANIP (hand) pose in both
``head`` and ``world`` frames. A human checks that the head-yaw sign and hand
motion direction agree with the video BEFORE any large-scale training (CLAUDE.md
§7/§9.3). Needs no DINO weights, so it can run before HF access is granted.

    python gr00t_ext/scripts/viz_egodex_transforms.py \
        --egodex-root /lustre/dataset/EgoDex --out-dir /lustre/meat124/runs/groot_runs/viz \
        --num-episodes 10 --horizon 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gr00t.data.state_action.pose import EndEffectorPose
from gr00t_ext.data.egodex_paths import VAL_SPLIT, discover_episodes
from gr00t_ext.data.egodex_transforms import relative_camera_pose, relative_gripper_pose
from gr00t_ext.data.gripper import gripper_pose_opening, joint_keys
from gr00t_ext.data.video import read_frames_by_indices
import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _trans_euler(xyz_rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = EndEffectorPose(translation=xyz_rot6d[:3], rotation=xyz_rot6d[3:], rotation_type="rot6d")
    return pose.translation, pose.euler_xyz  # translation (m), euler xyz (deg)


def _pick_episodes(root: str, splits: tuple[str, ...], num: int, seed: int) -> list[dict]:
    episodes = discover_episodes(root, splits)
    by_task: dict[str, list[dict]] = {}
    for ep in episodes:
        by_task.setdefault(ep["task"], []).append(ep)
    rng = np.random.default_rng(seed)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    picked = [by_task[task][0] for task in tasks[:num]]
    return picked


def _gripper_at(f, side: str, t: int):
    joints = {role: np.asarray(f[key][t][:3, 3]) for role, key in joint_keys(side).items()}
    return gripper_pose_opening(joints)


def visualize(episode: dict, horizon: int, out_path: Path) -> bool:
    """Render frame t / t+H with the 29-D embodiment's relative targets overlaid:
    HEAD (z_head <-> middle_eef) and BOTH hands' gripper poses (z_manip)."""
    with h5py.File(episode["hdf5_path"], "r") as f:
        n = f["transforms/camera"].shape[0]
        if n <= horizon + 1:
            return False
        t = max(0, min(n // 3, n - 1 - horizon))
        t_h = t + horizon
        cam_t = np.asarray(f["transforms/camera"][t])
        cam_t_h = np.asarray(f["transforms/camera"][t_h])
        grips = {  # per hand: (pose_t, opening_t, pose_t+H, opening_t+H)
            side: (*_gripper_at(f, side, t), *_gripper_at(f, side, t_h))
            for side in ("left", "right")
        }

    frame_t, frame_t_h = read_frames_by_indices(episode["mp4_path"], [t, t_h])
    head_t_, head_e = _trans_euler(relative_camera_pose(cam_t, cam_t_h))

    lines = [
        f"{episode['episode_id']}  (H={horizon})",
        f"HEAD rel (z_head<->middle_eef):  dxyz(m)={np.round(head_t_, 3)}  euler(deg)={np.round(head_e, 1)}",
    ]
    for side, (g_t, o_t, g_t_h, o_t_h) in grips.items():
        mh_t, mh_e = _trans_euler(relative_gripper_pose(cam_t, cam_t_h, g_t, g_t_h, "head"))
        mw_t, _ = _trans_euler(relative_gripper_pose(cam_t, cam_t_h, g_t, g_t_h, "world"))
        lines.append(
            f"{side.upper():5s} manip head-frame dxyz={np.round(mh_t, 3)} euler={np.round(mh_e, 1)}"
            f"  world dxyz={np.round(mw_t, 3)}  grip {o_t:.2f}->{o_t_h:.2f}"
        )
    lines.append("(relative gripper pose via Kabsch — NOT 2D reprojection GT)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(frame_t)
    axes[0].set_title(f"t = {t}")
    axes[1].imshow(frame_t_h)
    axes[1].set_title(f"t+H = {t_h}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("\n".join(lines), fontsize=8, family="monospace", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="EgoDex coordinate visualization gate")
    parser.add_argument("--egodex-root", default="/lustre/dataset/EgoDex")
    parser.add_argument("--out-dir", default="/lustre/meat124/runs/groot_runs/viz")
    parser.add_argument("--splits", nargs="+", default=[VAL_SPLIT])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = _pick_episodes(args.egodex_root, tuple(args.splits), args.num_episodes, args.seed)
    out_dir = Path(args.out_dir)
    written = 0
    for episode in episodes:
        out_path = out_dir / f"{episode['episode_id'].replace('/', '__')}.png"
        if visualize(episode, args.horizon, out_path):
            written += 1
            print(f"wrote {out_path}")
    print(f"done. {written} visualizations in {out_dir}")


if __name__ == "__main__":
    main()
