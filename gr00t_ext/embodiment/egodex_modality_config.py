# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""GR00T modality config for EgoDex action-BC pretraining (registered as NEW_EMBODIMENT).

Pass this file to ``--modality-config-path`` for ``gr00t/experiment/launch_finetune.py``
when finetuning on the ``/lustre/meat124/egodex_lerobot`` dataset. Importing it registers
the **same 29-D ``[manip ; head]`` eef action layout as AV-ALOHA** (single-sourced from
``examples/AVAloha/avaloha_layout`` via ``gr00t_ext.embodiment.egodex_embodiment``), but
with EgoDex's single ``ego_view`` head/ego camera. Because the layout, action horizon and
NEW_EMBODIMENT slot are identical to ``examples/AVAloha/avaloha_config.py``, an N1.7
checkpoint pretrained here transfers 1:1 into the AV-ALOHA Stage C finetune (the only
difference the model sees is the video key, and the frozen/finetuned VLM is
camera-agnostic).

This is the "fair baseline" entry point: it lets a plain GR00T N1.7 see EgoDex through its
real action labels (behavior cloning) before the AV-ALOHA comparison, matching the EgoDex
exposure our latent-NTP models get in Stages A/B.
"""

from gr00t_ext.embodiment.egodex_embodiment import register_manip_head_embodiment


# ego_view is the only EgoDex camera; action_horizon=16 and num_arms=3 mirror the
# AV-ALOHA Stage C modality config so the action chunk layout matches exactly.
register_manip_head_embodiment(video_keys=["ego_view"], action_horizon=16, num_arms=3)
