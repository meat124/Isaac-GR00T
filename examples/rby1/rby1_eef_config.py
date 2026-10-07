"""RB-Y1 modality config — RELATIVE END-EFFECTOR (EEF) variant, head camera only.

Uses GR00T's native relative-EEF action space (matching the N1.7 pretrain
embodiments like OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT):

  - state/action eef_right / eef_left are 9D absolute poses (xyz + rot6d), read
    from the eef_state_20d / eef_action_20d columns built by
    scripts/build_rby1_eef_dataset.py.
  - rep=RELATIVE + type=EEF + format=XYZ_ROT6D: the processor computes the
    SE3-relative transform of each future eef pose vs the current eef state, and
    inverts it back to absolute at decode time (needs IK on the robot to convert
    the predicted eef target to joint commands).
  - grippers stay absolute (NON_EEF), as in the joint-space config.

Use with: launch_finetune.py --modality-config-path examples/rby1/rby1_eef_config.py
on the *_eef datasets (which carry the eef_state_20d / eef_action_20d columns).
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

rby1_eef_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],  # head camera only
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_right",
            "eef_left",
            "right_gripper",
            "left_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "eef_right",
            "eef_left",
            "right_gripper",
            "left_gripper",
        ],
        action_configs=[
            # right end-effector: relative SE3 delta in xyz+rot6d
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_right",
            ),
            # left end-effector: relative SE3 delta in xyz+rot6d
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_left",
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

register_modality_config(rby1_eef_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
