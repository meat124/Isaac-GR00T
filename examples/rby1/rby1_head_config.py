"""RB-Y1 modality config — HEAD CAMERA ONLY variant.

Identical to rby1_config.py except the video modality uses only the head
(ego_view) camera; the two wrist cameras are dropped. State/action spaces are
unchanged (joint-space: relative arms, absolute grippers).

Use with: launch_finetune.py --modality-config-path examples/rby1/rby1_head_config.py
The dataset's meta/modality.json may still list all three cameras — the config
just selects a subset, so no dataset changes or stats regeneration are needed.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

rby1_head_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # head camera only
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm",
            "left_arm",
            "right_gripper",
            "left_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "right_arm",
            "left_arm",
            "right_gripper",
            "left_gripper",
        ],
        action_configs=[
            # right_arm (7 joints) - relative for smoother control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            # left_arm (7 joints) - relative for smoother control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            # right_gripper - absolute (binary open/close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper - absolute (binary open/close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(rby1_head_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
