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

"""Low-rank adaptation (LoRA) of the transformers of a GR00T model.

A LoRA layer keeps the weight it adapts frozen and learns a low-rank update next to it, so a
transformer with a billion weights is fine-tuned through a few million. The adapters are injected
in place: the model keeps its class and its forward pass, and only the adapted linear layers change
type, which also changes their names in the state dict (``to_q.weight`` becomes
``to_q.base_layer.weight``, next to ``to_q.lora_A.default.weight`` and ``to_q.lora_B.default.weight``).

Which layers are adapted, and with what defaults, follows the LoRA fine-tuning of GR00T N1.5
(``gr00t/utils/peft.py`` there).
"""

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
import torch
from torch import nn


# The query, key and value projections of the attention layers, as they are named in the action
# head's transformers and in the backbone's language model. This is what GR00T N1.5 adapts.
_QKV = ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj")
# What a linear layer's name has to contain for the layer to be adapted, per choice of targets.
TARGETS = {
    "qkv": _QKV,
    # Also the attention output projection and the feed-forward layers, as openpi adapts.
    "all": (
        *_QKV,
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}
# What the adapters' parameters have in their names, which nothing else does.
LORA_PARAMETER_MARKER = "lora_"


def target_layers(module: nn.Module, targets: str = "qkv") -> list[str]:
    """Name the linear layers of a module that a choice of targets adapts.

    Args:
        module: Module to look in.
        targets: "qkv" for the query, key and value projections of the attention layers, "all" to
            add the attention output projection and the feed-forward layers.

    Returns:
        The names of the layers, relative to the module.
    """
    patterns = TARGETS[targets]
    return [
        name
        for name, layer in module.named_modules()
        if isinstance(layer, nn.Linear) and any(pattern in name for pattern in patterns)
    ]


def add_lora(
    module: nn.Module,
    rank: int,
    alpha: float = 16,
    dropout: float = 0.1,
    targets: str = "qkv",
) -> int:
    """Add LoRA adapters to the linear layers of a module, in place.

    The adapters start as a no-op: one of their two factors is zero, so the module's output is
    unchanged until they are trained. Every parameter of the module other than the adapters'
    is left frozen.

    Args:
        module: Module whose layers are adapted.
        rank: Rank of the update learned for each layer.
        alpha: Scale of the update, applied as ``alpha / rank``.
        dropout: Dropout on the input of each adapter.
        targets: Which layers are adapted; see `target_layers`.

    Returns:
        The number of layers adapted, which is zero for a module without such layers.
    """
    names = target_layers(module, targets)
    if not names:
        return 0
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=names, bias="none"
    )
    inject_adapter_in_model(config, module)
    for layer in module.modules():
        if isinstance(layer, LoraLayer):
            # A new layer starts out in training mode, with its dropout active, also inside a
            # model that is in evaluation mode.
            layer.train(layer.get_base_layer().training)
    return len(names)


def has_lora(module: nn.Module) -> bool:
    """Whether any layer of a module carries a LoRA adapter."""
    return any(isinstance(layer, LoraLayer) for layer in module.modules())


def merge_lora(module: nn.Module) -> int:
    """Fold the LoRA adapters of a module into the weights they adapt and remove them, in place.

    The module computes the same as before and is left with the layers, and so the state dict
    names, it had before the adapters were added.

    Args:
        module: Module with adapters added by `add_lora`.

    Returns:
        The number of adapters merged.
    """
    adapted = [
        (name, layer) for name, layer in module.named_modules() if isinstance(layer, LoraLayer)
    ]
    for name, layer in adapted:
        # In fp32, whatever the model is held in: a merged weight rounded to 16 bits would lose
        # much of the update, which is small next to the weight it is added to.
        layer.to(torch.float32)
        layer.merge(safe_merge=True)  # refuses to write a weight that is not finite
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, layer.get_base_layer())
    if hasattr(module, "peft_config"):
        del module.peft_config  # left behind by the injection
    return len(adapted)


def set_only_lora_trainable(module: nn.Module, trainable: bool = True):
    """Freeze every parameter of a module except, if `trainable`, those of its LoRA adapters."""
    for name, parameter in module.named_parameters():
        parameter.requires_grad = trainable and LORA_PARAMETER_MARKER in name
