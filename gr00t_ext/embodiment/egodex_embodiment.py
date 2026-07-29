# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""The unified 29-D ``[manip ; head]`` embodiment shared by Stage B and Stage C.

Both stages use the **same** AV-ALOHA end-effector action layout so the GR00T DiT /
action-encoder transfer cleanly and the ``z_head``/``z_manip`` correspondence is
consistent (CLAUDE.md §2-B; user decision: unify to 29-D):

    [ left_eef(9) ; left_gripper(1) ; right_eef(9) ; right_gripper(1) ; middle_eef(9) ]
      \________________ z_manip ________________/        \____ z_head ____/

* ``z_manip`` <-> the two arm EEF poses + grippers (manipulation),
* ``z_head``  <-> ``middle_eef`` — the active-vision camera pose (Stage C: the
  AV-ALOHA middle arm carrying the ZED; Stage B: EgoDex head/camera).

The layout is single-sourced from ``examples/AVAloha/avaloha_layout.py`` (the same
module the AV-ALOHA Stage C tooling uses) so there is exactly one definition. Stage B
(EgoDex) registers this with its own ego camera key; Stage C registers it via
``avaloha_config.py`` with the ZED/wrist cameras — same action layout, same
``NEW_EMBODIMENT`` slot, different video keys (the frozen VLM is camera-agnostic).
"""

from __future__ import annotations

from pathlib import Path
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# avaloha_layout lives in examples/AVAloha (run as a path-added module, not a package),
# importing it has no side effects. Add that dir to sys.path once, then import the
# single-source layout helpers.
_AVALOHA_DIR = Path(__file__).resolve().parents[2] / "examples" / "AVAloha"
if str(_AVALOHA_DIR) not in sys.path:
    sys.path.insert(0, str(_AVALOHA_DIR))

from avaloha_layout import (  # noqa: E402  (path-dependent import, intentional)
    EEF_POSE_GROUPS,
    GRIPPER_GROUPS,
    eef_state_action_groups,
    groups_for,
)


LANGUAGE_KEY = "annotation.human.task_description"
DEFAULT_VIDEO_KEYS = ["ego_view"]  # EgoDex single ego/head camera


def _action_configs(groups: list[str], action_space: str) -> list[ActionConfig]:
    """Per-group ActionConfig: arms RELATIVE EEF (xyz+rot6d), grippers ABSOLUTE scalar.

    Mirrors ``examples/AVAloha/avaloha_config.build_avaloha_modality_config`` so the
    action representation is identical to the Stage C embodiment.
    """
    configs: list[ActionConfig] = []
    for group in groups:
        if group in GRIPPER_GROUPS:
            configs.append(
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            )
        elif action_space == "eef" and group in EEF_POSE_GROUPS:
            configs.append(
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key=group,
                )
            )
        else:
            configs.append(
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            )
    return configs


def build_manip_head_modality_config(
    video_keys: list[str],
    action_horizon: int = 16,
    num_arms: int = 3,
    action_space: str = "eef",
    language_key: str = LANGUAGE_KEY,
) -> dict[str, ModalityConfig]:
    """Assemble the ``{video, state, action, language}`` 29-D modality config.

    Action ``modality_keys`` order is the chunk layout
    ``[left_eef, left_gripper, right_eef, right_gripper, middle_eef]``.
    """
    groups = list(groups_for(action_space, num_arms).keys())
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(video_keys)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=groups),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=groups,
            action_configs=_action_configs(groups, action_space),
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[language_key]),
    }


def register_manip_head_embodiment(
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT,
    video_keys: list[str] | None = None,
    action_horizon: int = 16,
    num_arms: int = 3,
) -> None:
    """Register the 29-D ``[manip ; head]`` modality config (idempotent).

    ``register_modality_config`` errors if the tag is already registered, so this is
    a no-op when it already exists (safe to call per-process / in tests).
    """
    if embodiment_tag.value in MODALITY_CONFIGS:
        return
    config = build_manip_head_modality_config(
        video_keys=video_keys or DEFAULT_VIDEO_KEYS,
        action_horizon=action_horizon,
        num_arms=num_arms,
    )
    register_modality_config(config, embodiment_tag=embodiment_tag)


def manip_head_modality_json(
    video_keys: list[str] | None = None, num_arms: int = 3, action_space: str = "eef"
) -> dict:
    """The dataset's ``meta/modality.json`` (29-D state/action slices + video map).

    Used by ``gr00t_ext/scripts/egodex_to_lerobot_v2.py``. State and action share the
    same slicing: left_eef[0:9], left_gripper[9:10], right_eef[10:19],
    right_gripper[19:20], middle_eef[20:29].
    """
    groups = (
        eef_state_action_groups(num_arms)
        if action_space == "eef"
        else groups_for(action_space, num_arms)
    )
    slices = {g: {"start": s, "end": e} for g, (s, e) in groups.items()}
    vkeys = video_keys or DEFAULT_VIDEO_KEYS
    return {
        "state": dict(slices),
        "action": dict(slices),
        "video": {k: {"original_key": f"observation.images.{k}"} for k in vkeys},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
