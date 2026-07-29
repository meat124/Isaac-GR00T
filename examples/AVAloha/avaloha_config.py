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

"""GR00T modality config for the AV-ALOHA embodiment (registered as NEW_EMBODIMENT).

Importing this module registers a modality config whose set of cameras and
action horizon are read from a YAML file (``avaloha_config.yaml`` next to this
file, or the path in the ``AVALOHA_CONFIG`` environment variable). This is the
file you pass to ``--modality-config-path`` for both ``gr00t/data/stats.py`` and
``gr00t/experiment/launch_finetune.py``.

State/action are split into per-arm joint groups (see ``avaloha_layout.py``):
arms use a RELATIVE action representation (deltas from the current pose, which
GR00T N1.7 reconstructs to absolute targets at inference), grippers use ABSOLUTE.
"""

import os
from pathlib import Path

from avaloha_layout import CAMERAS_BY_NUM_ARMS, EEF_POSE_GROUPS, GRIPPER_GROUPS, groups_for
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
import yaml


# Language key shared by the converted dataset (meta/modality.json annotation)
# and the deployment observation.
LANGUAGE_KEY = "annotation.human.task_description"


def _load_yaml_config() -> dict:
    """Load the YAML pointed to by $AVALOHA_CONFIG, else the sibling yaml."""
    cfg_path = os.environ.get("AVALOHA_CONFIG")
    if cfg_path is None:
        cfg_path = Path(__file__).with_name("avaloha_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_avaloha_modality_config(
    cameras: list[str],
    action_horizon: int,
    num_arms: int,
    action_space: str = "eef",
) -> dict[str, ModalityConfig]:
    """Build the GR00T modality config for AV-ALOHA in 'joint' or 'eef' action space.

    - joint: arm groups are RELATIVE NON_EEF joint deltas; grippers ABSOLUTE NON_EEF.
    - eef:   arm groups are RELATIVE EEF poses (XYZ_ROT6D, reconstructed to absolute
             at inference); grippers ABSOLUTE NON_EEF.
    """
    groups = list(groups_for(action_space, num_arms).keys())

    available = set(CAMERAS_BY_NUM_ARMS[num_arms])
    bad = [c for c in cameras if c not in available]
    if bad:
        raise ValueError(
            f"Cameras {bad} are not available for num_arms={num_arms}. "
            f"Available: {sorted(available)}"
        )
    if not cameras:
        raise ValueError("At least one camera must be selected in the config.")

    action_configs = []
    for group in groups:
        if group in GRIPPER_GROUPS:
            # Near-binary target -> ABSOLUTE scalar.
            action_configs.append(
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            )
        elif action_space == "eef" and group in EEF_POSE_GROUPS:
            # End-effector pose -> RELATIVE EEF (xyz + rot6d), referenced to the
            # same-named state group; GR00T reconstructs absolute poses at inference.
            action_configs.append(
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key=group,
                )
            )
        else:
            # Joint-space arm -> RELATIVE joint deltas.
            action_configs.append(
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            )

    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(cameras)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=groups),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=groups,
            action_configs=action_configs,
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }


_cfg = _load_yaml_config()
avaloha_config = build_avaloha_modality_config(
    cameras=list(_cfg["cameras"]),
    action_horizon=int(_cfg.get("action_horizon", 16)),
    num_arms=int(_cfg.get("num_arms", 3)),
    action_space=str(_cfg.get("action_space", "eef")),
)

register_modality_config(avaloha_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
