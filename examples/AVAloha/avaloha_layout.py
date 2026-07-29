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

"""Shared, side-effect-free layout constants for the AV-ALOHA <-> GR00T bridge.

The AV-ALOHA simulator (``gym_guided_vision``) records a flat joint vector for
both ``observation.state`` and ``action``. This module is the single source of
truth for how that vector splits into named GR00T modality groups, which
cameras exist, and how task short-names map to HuggingFace datasets / gym envs.

Importing this module has NO side effects (it does not register anything), so
the data converter, the modality config, and the deployment script can all
share it freely.

Joint vector layout (AV-ALOHA, ViperX 300S arms; grippers normalized to [0, 1],
1 = open, 0 = closed):

    3 arms (21-dim):
        left_arm      indices [0:6]    left ViperX 6-DoF arm
        left_gripper  index   [6:7]    left gripper (normalized)
        right_arm     indices [7:13]   right ViperX 6-DoF arm
        right_gripper index   [13:14]  right gripper (normalized)
        middle_arm    indices [14:21]  7-DoF active-vision arm carrying the ZED

    2 arms (14-dim): the ``middle_arm`` group is dropped (no ZED cameras).
"""

from __future__ import annotations


# All cameras the simulator can render. The two "zed_cam_*" views are the
# active-vision stereo pair mounted on the middle arm; they only exist on the
# 3-arm embodiment.
ALL_CAMERAS: tuple[str, ...] = (
    "zed_cam_left",
    "zed_cam_right",
    "wrist_cam_left",
    "wrist_cam_right",
    "overhead_cam",
    "worms_eye_cam",
)

# Cameras available per arm count (the middle arm — and thus the ZED pair — is
# hidden on the 2-arm embodiment).
CAMERAS_BY_NUM_ARMS: dict[int, tuple[str, ...]] = {
    2: ("wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"),
    3: ALL_CAMERAS,
}

# Frame size produced by the simulator / stored in the datasets.
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def state_action_groups(num_arms: int) -> dict[str, tuple[int, int]]:
    """Return an ordered {group_name: (start, end)} split of the joint vector.

    The insertion order is also the concatenation order used to rebuild the
    flat action vector for ``env.step`` during deployment, so it must match the
    simulator's expected actuator ordering (left, right, middle).
    """
    if num_arms not in (2, 3):
        raise ValueError(f"num_arms must be 2 or 3, got {num_arms}")
    groups: dict[str, tuple[int, int]] = {
        "left_arm": (0, 6),
        "left_gripper": (6, 7),
        "right_arm": (7, 13),
        "right_gripper": (13, 14),
    }
    if num_arms == 3:
        groups["middle_arm"] = (14, 21)
    return groups


def state_dim(num_arms: int) -> int:
    """Total joint-vector dimensionality (21 for 3 arms, 14 for 2 arms)."""
    return 21 if num_arms == 3 else 14


# Which groups are grippers (normalized [0, 1]) vs. continuous joint angles.
# Grippers train better as ABSOLUTE targets; arms as RELATIVE deltas.
GRIPPER_GROUPS: frozenset[str] = frozenset({"left_gripper", "right_gripper"})


# =============================================================================
# End-effector / task-space action layout (the optional "eef" action space)
# =============================================================================
# Instead of joint angles, each arm is represented by its end-effector POSE as
# eef_9d = [x, y, z, rot6d(6)] (rot6d = first two rows of the rotation matrix,
# matching GR00T's EndEffectorPose). Grippers stay scalar. The active-vision
# camera's "EEF" is the ZED camera frame on the middle arm.
#
#   3 arms (29-dim): left_eef[0:9] left_gripper[9:10] right_eef[10:19]
#                    right_gripper[19:20] middle_eef[20:29]
#   2 arms (20-dim): drops middle_eef.

ACTION_SPACES: tuple[str, ...] = ("joint", "eef")

# eef group (9-D pose) -> simulator spec used for FK (data conversion) and IK
# (deployment). `joint_slice` is where this arm's joints sit in the 21-D joint
# vector (so the IK result can be assembled into a sim joint action).
EEF_ARMS: dict[str, dict] = {
    "left_eef": {
        "site": "left_gripper_control",
        "joint_names": [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
        ],
        "joint_slice": (0, 6),
    },
    "right_eef": {
        "site": "right_gripper_control",
        "joint_names": [
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
        ],
        "joint_slice": (7, 13),
    },
    "middle_eef": {
        "site": "middle_zed_camera_center",
        "joint_names": [
            "middle_waist",
            "middle_shoulder",
            "middle_elbow",
            "middle_forearm_roll",
            "middle_wrist_1_joint",
            "middle_wrist_2_joint",
            "middle_wrist_3_joint",
        ],
        "joint_slice": (14, 21),
    },
}

# gripper group -> (eef-vector slice, joint-21 slice). Grippers pass through unchanged.
EEF_GRIPPERS: dict[str, dict] = {
    "left_gripper": {"joint_slice": (6, 7)},
    "right_gripper": {"joint_slice": (13, 14)},
}

# The 9-D EEF pose groups (vs. the 1-D gripper groups).
EEF_POSE_GROUPS: frozenset[str] = frozenset({"left_eef", "right_eef", "middle_eef"})


def eef_state_action_groups(num_arms: int) -> dict[str, tuple[int, int]]:
    """Ordered {group: (start, end)} split of the eef state/action vector."""
    if num_arms not in (2, 3):
        raise ValueError(f"num_arms must be 2 or 3, got {num_arms}")
    groups: dict[str, tuple[int, int]] = {
        "left_eef": (0, 9),
        "left_gripper": (9, 10),
        "right_eef": (10, 19),
        "right_gripper": (19, 20),
    }
    if num_arms == 3:
        groups["middle_eef"] = (20, 29)
    return groups


def eef_dim(num_arms: int) -> int:
    """Total eef state/action dimensionality (29 for 3 arms, 20 for 2 arms)."""
    return 29 if num_arms == 3 else 20


def groups_for(action_space: str, num_arms: int) -> dict[str, tuple[int, int]]:
    """Dispatch to the joint or eef group layout."""
    if action_space == "joint":
        return state_action_groups(num_arms)
    if action_space == "eef":
        return eef_state_action_groups(num_arms)
    raise ValueError(f"action_space must be one of {ACTION_SPACES}, got {action_space!r}")


def dim_for(action_space: str, num_arms: int) -> int:
    return state_dim(num_arms) if action_space == "joint" else eef_dim(num_arms)


# Task short-name -> (gym EnvName prefix, HF dataset stem, default instruction).
# HF datasets live under the "iantc104/gv_sim_<stem>_<N>arms" repo ids.
TASKS: dict[str, dict[str, str]] = {
    "insert_peg": {
        "env": "InsertPeg",
        "stem": "insert_peg",
        "instruction": "Pick up the peg and insert it into the hole.",
    },
    "slot_insertion": {
        "env": "SlotInsertion",
        "stem": "slot_insertion",
        "instruction": "Pick up the stick and insert it through the slot.",
    },
    "sew_needle": {
        "env": "SewNeedle",
        "stem": "sew_needle",
        "instruction": "Pick up the needle and pass it through the hole.",
    },
    "tube_transfer": {
        "env": "TubeTransfer",
        "stem": "tube_transfer",
        "instruction": "Transfer the tube from one rack to the other.",
    },
    "hook_package": {
        "env": "HookPackage",
        "stem": "hook_package",
        "instruction": "Pick up the package and hang it on the hook.",
    },
}


def hf_repo_id(task: str, num_arms: int) -> str:
    """HuggingFace dataset repo id for a task / arm-count, e.g.
    ``iantc104/gv_sim_slot_insertion_3arms``."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Known: {sorted(TASKS)}")
    return f"iantc104/gv_sim_{TASKS[task]['stem']}_{num_arms}arms"


def gym_env_id(task: str, num_arms: int) -> str:
    """gym_guided_vision environment id, e.g.
    ``gym_guided_vision/SlotInsertion-3Arms-v0``."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Known: {sorted(TASKS)}")
    return f"gym_guided_vision/{TASKS[task]['env']}-{num_arms}Arms-v0"


def default_instruction(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Known: {sorted(TASKS)}")
    return TASKS[task]["instruction"]
