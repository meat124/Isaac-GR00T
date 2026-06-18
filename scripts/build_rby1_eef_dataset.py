"""Build a relative-EEF variant of an rby1 dataset.

For each frame, converts the absolute end-effector poses
(right_eef_pose / left_eef_pose = xyz + quat[xyzw], 7D) into GR00T's 9D
xyz+rot6d action format, and writes two new parquet columns:

  eef_state_20d  = [right_eef_9d(9), left_eef_9d(9), right_gripper, left_gripper]  (state grippers)
  eef_action_20d = [right_eef_9d(9), left_eef_9d(9), right_gripper, left_gripper]  (action grippers)

The eef pose block is identical in both (absolute per-frame pose); only the gripper
source differs (observation.state vs action). With rep=RELATIVE the processor reads
future eef poses from eef_action_20d and computes the SE3-relative transform vs the
current eef pose in eef_state_20d.

Then writes a HEAD-ONLY meta/modality.json for the EEF embodiment and removes stale
stats (regenerated separately).

Usage: python scripts/build_rby1_eef_dataset.py <src_name> <dst_name>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gr00t.data.state_action.pose import EndEffectorPose

DEMO = Path("/lustre/meat124/rby1_demo")


def quat_xyzw_to_xyz_rot6d(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """xyz(3)+quat[xyzw](4) -> xyz(3)+rot6d(6) = 9D, matching GR00T's parser."""
    R = Rotation.from_quat(quat_xyzw).as_matrix()  # (3,3)
    rot6d = R[:2].flatten()  # first two ROWS (GR00T convention)
    return np.concatenate([xyz, rot6d]).astype(np.float32)


def _verify_convention():
    """Assert the scipy fast-path matches GR00T's EndEffectorPose.xyz_rot6d exactly."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        xyz = rng.normal(size=3)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        fast = quat_xyzw_to_xyz_rot6d(xyz, q)
        gt = EndEffectorPose(
            translation=xyz, rotation=q, rotation_type="quat", rotation_order="xyzw"
        ).xyz_rot6d
        assert np.allclose(fast, gt, atol=1e-6), (fast, gt)
    print("convention check OK (scipy fast-path == GR00T EndEffectorPose.xyz_rot6d)")


def convert(src_name: str, dst_name: str):
    src = DEMO / src_name
    dst = DEMO / dst_name
    print(f"=== {src} -> {dst} ===")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    n_frames = 0
    for pq in sorted((dst / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        rep = df["right_eef_pose"].to_numpy()
        lep = df["left_eef_pose"].to_numpy()
        st = df["observation.state"].to_numpy()
        ac = df["action"].to_numpy()
        eef_state, eef_action = [], []
        for i in range(len(df)):
            r9 = quat_xyzw_to_xyz_rot6d(np.asarray(rep[i][:3], float), np.asarray(rep[i][3:7], float))
            l9 = quat_xyzw_to_xyz_rot6d(np.asarray(lep[i][:3], float), np.asarray(lep[i][3:7], float))
            s = np.asarray(st[i], np.float32)
            a = np.asarray(ac[i], np.float32)
            eef_state.append(np.concatenate([r9, l9, s[14:15], s[15:16]]).astype(np.float32))
            eef_action.append(np.concatenate([r9, l9, a[14:15], a[15:16]]).astype(np.float32))
        df["eef_state_20d"] = eef_state
        df["eef_action_20d"] = eef_action
        df.to_parquet(pq, index=False)
        n_frames += len(df)
    print(f"  converted {n_frames} frames across parquet files")

    # head-only EEF modality.json
    modality = {
        "state": {
            "eef_right": {"original_key": "eef_state_20d", "start": 0, "end": 9},
            "eef_left": {"original_key": "eef_state_20d", "start": 9, "end": 18},
            "right_gripper": {"original_key": "eef_state_20d", "start": 18, "end": 19},
            "left_gripper": {"original_key": "eef_state_20d", "start": 19, "end": 20},
        },
        "action": {
            "eef_right": {"original_key": "eef_action_20d", "start": 0, "end": 9},
            "eef_left": {"original_key": "eef_action_20d", "start": 9, "end": 18},
            "right_gripper": {"original_key": "eef_action_20d", "start": 18, "end": 19},
            "left_gripper": {"original_key": "eef_action_20d", "start": 19, "end": 20},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    (dst / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))

    # Register the new columns in info.json features so generate_stats computes
    # their stats (the loader's get_dataset_statistics keys off these).
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    def _names9(side):
        return [f"{side}_{c}" for c in ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12")]

    eef_names = _names9("right") + _names9("left") + ["right_gripper", "left_gripper"]
    for col in ("eef_state_20d", "eef_action_20d"):
        info["features"][col] = {"dtype": "float32", "shape": [20], "names": eef_names}
    info_path.write_text(json.dumps(info, indent=4))

    for stale in ("stats.json", "relative_stats.json"):
        p = dst / "meta" / stale
        if p.exists():
            p.unlink()
    print(f"  wrote meta/modality.json + info.json eef features; removed stale stats")


if __name__ == "__main__":
    _verify_convention()
    convert(sys.argv[1], sys.argv[2])
