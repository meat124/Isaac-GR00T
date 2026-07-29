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

"""FK / IK between AV-ALOHA joint vectors and GR00T end-effector (eef_9d) vectors.

Used by:
  - convert_avaloha_to_gr00t.py  (FK: joint state/action -> eef_9d, for the eef dataset)
  - deploy_avaloha.py            (IK: predicted eef_9d -> joint targets, for the sim)

`eef_9d = [x, y, z, rot6d(6)]` where rot6d is the first two ROWS of the rotation
matrix (matches GR00T's EndEffectorPose; decode via Gram-Schmidt on the rows).

Uses raw ``mujoco`` (no GL context needed, so it runs headless on CPU login nodes
too). FK reads site poses directly; IK is damped-least-squares on the site
Jacobian, seeded at the current joints (actions are small relative deltas).
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaloha_layout import (  # noqa: E402
    EEF_ARMS,
    EEF_GRIPPERS,
    TASKS,
    eef_dim,
    eef_state_action_groups,
    state_action_groups,
)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> 6D (first two rows, flattened) — GR00T convention."""
    return np.concatenate([R[0], R[1]]).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D -> rotation matrix via Gram-Schmidt on the two rows (GR00T convention)."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row1 = r[0] / (np.linalg.norm(r[0]) + 1e-9)
    row2 = r[1] - np.dot(row1, r[1]) * row1
    row2 = row2 / (np.linalg.norm(row2) + 1e-9)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


class AvalohaKinematics:
    """FK/IK on a scratch copy of the AV-ALOHA MuJoCo model (does not touch the sim)."""

    def __init__(self, num_arms: int = 3, task: str = "slot_insertion"):
        from gym_guided_vision.constants import XML_DIR

        self.num_arms = num_arms
        xml = os.path.join(XML_DIR, f"task_{TASKS[task]['stem']}.xml")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        self.joint_groups = state_action_groups(num_arms)
        self.eef_layout = eef_state_action_groups(num_arms)
        self.eef_groups = [g for g in self.eef_layout if g in EEF_ARMS]

        # Precompute site ids and per-arm joint qpos/dof addresses + limits.
        self._site_id = {}
        self._qadr = {}  # eef group -> array of qpos addresses (one per arm joint)
        self._dadr = {}  # eef group -> array of dof addresses
        self._jrange = {}  # eef group -> (n,2) joint limits (nan if unlimited)
        for g in self.eef_groups:
            self._site_id[g] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, EEF_ARMS[g]["site"]
            )
            qadr, dadr, rng = [], [], []
            for jn in EEF_ARMS[g]["joint_names"]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                qadr.append(self.model.jnt_qposadr[jid])
                dadr.append(self.model.jnt_dofadr[jid])
                lim = (
                    self.model.jnt_range[jid] if self.model.jnt_limited[jid] else (-np.inf, np.inf)
                )
                rng.append(lim)
            self._qadr[g] = np.array(qadr)
            self._dadr[g] = np.array(dadr)
            self._jrange[g] = np.array(rng, dtype=np.float64)

    def _arm_slice(self, eef_group: str) -> tuple[int, int]:
        return self.joint_groups[eef_group.replace("_eef", "_arm")]

    def _set_arm_qpos(self, joints21: np.ndarray) -> None:
        for g in self.eef_groups:
            s, e = self._arm_slice(g)
            self.data.qpos[self._qadr[g]] = np.asarray(joints21, dtype=np.float64)[s:e]

    def _site_pose(self, g: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self._site_id[g]
        pos = self.data.site_xpos[sid].copy()
        R = self.data.site_xmat[sid].reshape(3, 3).copy()
        return pos, R

    # ---- FK: joint vector -> eef vector ----------------------------------------
    def fk(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        self._set_arm_qpos(joints)
        mujoco.mj_kinematics(self.model, self.data)
        out = np.zeros(eef_dim(self.num_arms), dtype=np.float32)
        for g in self.eef_groups:
            pos, R = self._site_pose(g)
            s, e = self.eef_layout[g]
            out[s : s + 3] = pos
            out[s + 3 : e] = matrix_to_rot6d(R)
        for grp, spec in EEF_GRIPPERS.items():
            js, je = spec["joint_slice"]
            es, ee = self.eef_layout[grp]
            out[es:ee] = joints[js:je]
        return out

    # ---- IK: eef vector -> joint vector (damped least squares, seeded) ---------
    def ik(
        self,
        eef_vec: np.ndarray,
        seed_joints: np.ndarray,
        max_iters: int = 80,
        tol: float = 1e-3,
        damping: float = 1e-2,
        max_dq: float = 0.2,
    ) -> np.ndarray:
        eef_vec = np.asarray(eef_vec, dtype=np.float64)
        out = np.array(seed_joints, dtype=np.float32).copy()
        self._set_arm_qpos(seed_joints)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for g in self.eef_groups:
            es, ee = self.eef_layout[g]
            target_pos = eef_vec[es : es + 3]
            target_R = rot6d_to_matrix(eef_vec[es + 3 : ee])
            qadr, dadr, rng = self._qadr[g], self._dadr[g], self._jrange[g]
            for _ in range(max_iters):
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_comPos(self.model, self.data)
                cur_pos, cur_R = self._site_pose(g)
                pos_err = target_pos - cur_pos
                rot_err = Rotation.from_matrix(target_R @ cur_R.T).as_rotvec()
                err = np.concatenate([pos_err, rot_err])
                if np.linalg.norm(err) < tol:
                    break
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id[g])
                J = np.vstack([jacp[:, dadr], jacr[:, dadr]])  # (6, n)
                dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(6), err)
                dq = np.clip(dq, -max_dq, max_dq)
                q = self.data.qpos[qadr] + dq
                q = np.clip(q, rng[:, 0], rng[:, 1])
                self.data.qpos[qadr] = q
            s, e = self._arm_slice(g)
            out[s:e] = self.data.qpos[qadr]
        for grp, spec in EEF_GRIPPERS.items():
            js, je = spec["joint_slice"]
            es, ee = self.eef_layout[grp]
            out[js:je] = eef_vec[es:ee]
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-arms", type=int, default=3)
    ap.add_argument("--task", default="slot_insertion")
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()

    kin = AvalohaKinematics(num_arms=args.num_arms, task=args.task)
    from gym_guided_vision.constants import LEFT_ARM_POSE, MIDDLE_ARM_POSE, RIGHT_ARM_POSE

    home = np.array(
        list(LEFT_ARM_POSE)
        + list(RIGHT_ARM_POSE)
        + (list(MIDDLE_ARM_POSE) if args.num_arms == 3 else []),
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    max_pose_err = 0.0
    for _ in range(args.trials):
        q = home + rng.uniform(-0.25, 0.25, size=home.shape).astype(np.float32)
        eef = kin.fk(q)
        q_ik = kin.ik(eef, seed_joints=home)
        eef2 = kin.fk(q_ik)
        max_pose_err = max(max_pose_err, float(np.abs(eef - eef2).max()))
    print(f"FK/IK round-trip: max |eef - FK(IK(eef))| = {max_pose_err:.5f}  (want < ~0.02)")
    print("OK" if max_pose_err < 0.05 else "WARN: IK not converging well; check damping/iters")
