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

"""
Test LoRA fine-tuning of the action head: what the adapters change, what they leave trainable,
and that a model with adapters saves, loads and merges without changing what it computes.

The action head is instantiated directly, at a small size and without a backbone.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules import lora
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


RANK = 4
BLOCKS = 2  # of the diffusion model; the self-attention over the backbone features has one
# Layers adapted in each block: the query, key and value projections of its attention, to which
# "all" adds the attention's output projection and the two layers of the feed-forward.
ADAPTED_PER_BLOCK = {"qkv": 3, "all": 6}


def _small_config(**overrides) -> Gr00tN1d7Config:
    transformer = {
        "positional_embeddings": None,
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "dropout": 0.0,
        "final_dropout": False,
    }
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            **transformer,
            "num_layers": BLOCKS,
            "norm_type": "ada_norm",
            "output_dim": 64,
            "interleave_self_attention": True,
        },
        vl_self_attention_cfg={**transformer, "num_layers": 1},
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _inputs(config, batch_size=2, seq_len=8):
    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )
    return backbone_output, action_input


def _actions(head, inputs):
    """The actions the head predicts, from the same noise every time."""
    backbone_output, action_input = inputs
    # Without the ground-truth action: given one, the head takes it for a chunk to continue.
    action_input = {key: value for key, value in action_input.items() if key != "action"}
    torch.manual_seed(0)
    with torch.no_grad():
        return head.eval().get_action(
            BatchFeature(data=dict(backbone_output)), BatchFeature(data=action_input)
        )["action_pred"]


def _randomize_adapters(head):
    """Make the adapters do something, as training would: they start out as a no-op."""
    torch.manual_seed(1)
    for name, parameter in head.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.05)


@pytest.fixture
def head():
    torch.manual_seed(0)
    return Gr00tN1d7ActionHead(_small_config())


@pytest.mark.parametrize("targets", ["qkv", "all"])
def test_adapters_are_added_to_the_attention_of_every_block(head, targets):
    assert not lora.has_lora(head)
    head.add_lora(RANK, targets=targets)

    assert head.lora_rank == RANK
    adapted = [name for name, layer in head.named_modules() if isinstance(layer, lora.LoraLayer)]
    in_dit = [name for name in adapted if name.startswith("model.")]
    in_self_attention = [name for name in adapted if name.startswith("vl_self_attention.")]
    assert len(in_dit) == BLOCKS * ADAPTED_PER_BLOCK[targets]
    assert len(in_self_attention) == ADAPTED_PER_BLOCK[targets]
    assert len(adapted) == len(in_dit) + len(in_self_attention)  # and nowhere else


def test_the_default_targets_are_the_query_key_and_value_projections(head):
    names = lora.target_layers(head.model.transformer_blocks[0])
    assert sorted(names) == ["attn1.to_k", "attn1.to_q", "attn1.to_v"]
    assert set(lora.target_layers(head.model.transformer_blocks[0], "all")) == {
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }


def test_adapters_start_out_changing_nothing(head):
    inputs = _inputs(head.config)
    before = _actions(head, inputs)
    head.add_lora(RANK)
    torch.testing.assert_close(_actions(head, inputs), before)


def test_adapters_added_to_a_model_in_evaluation_mode_are_in_evaluation_mode(head):
    head.eval()
    head.add_lora(RANK, dropout=0.5)
    assert not any(module.training for module in head.modules())

    head.train()
    assert all(module.training for module in head.modules() if isinstance(module, lora.LoraLayer))


def test_by_default_nothing_but_the_adapters_is_trained(head):
    """As GR00T N1.5 fine-tunes with LoRA."""
    head.add_lora(RANK)
    trainable = {name for name, parameter in head.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert any("lora_A" in name for name in trainable) and any(
        "lora_B" in name for name in trainable
    )


def test_the_parts_without_adapters_can_be_left_to_train(head):
    head.add_lora(RANK, only=False)
    trainable = {name for name, parameter in head.named_parameters() if parameter.requires_grad}

    for name in trainable:
        if name.startswith("vl_self_attention."):
            assert "lora_" in name, name
        if name.startswith("model."):
            small = ("model.timestep_encoder.", "model.proj_out_1.", "model.proj_out_2.")
            assert "lora_" in name or name.startswith(small), name
    assert any("lora_" in name for name in trainable)
    # What feeds and reads the transformers is trained as without adapters.
    assert all(
        parameter.requires_grad
        for name, parameter in head.named_parameters()
        if name.startswith(("state_encoder.", "action_encoder.", "action_decoder.", "vlln."))
    )


@pytest.mark.parametrize("only", [True, False])
def test_a_training_step_moves_the_adapters_and_not_the_weights_they_adapt(head, only):
    head.add_lora(RANK, only=only)
    layer = head.get_submodule("model.transformer_blocks.0.attn1.to_q")
    weight_before = layer.base_layer.weight.detach().clone()
    b_before = layer.lora_B["default"].weight.detach().clone()
    encoder_before = head.state_encoder.layer1.W.detach().clone()

    optimizer = torch.optim.SGD([p for p in head.parameters() if p.requires_grad], lr=0.1)
    head.train()
    head.forward(*_inputs(head.config))["loss"].backward()
    optimizer.step()

    assert layer.base_layer.weight.grad is None
    torch.testing.assert_close(layer.base_layer.weight, weight_before)
    assert not torch.equal(layer.lora_B["default"].weight, b_before)
    assert torch.equal(head.state_encoder.layer1.W, encoder_before) == only


@pytest.mark.parametrize("only", [True, False])
def test_setting_trainable_parameters_again_keeps_the_adapted_weights_frozen(head, only):
    head.add_lora(RANK, only=only)
    head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True, tune_vlln=True)
    to_q = head.model.transformer_blocks[0].attn1.to_q
    assert not to_q.base_layer.weight.requires_grad
    assert to_q.lora_A["default"].weight.requires_grad
    assert head.state_encoder.layer1.W.requires_grad != only


def test_a_checkpoint_with_adapters_loads_into_a_model_built_from_its_config(head):
    head.add_lora(RANK)
    _randomize_adapters(head)
    inputs = _inputs(head.config)

    # As when loading a checkpoint: the config says there are adapters, so the model is built with them.
    restored = Gr00tN1d7ActionHead(_small_config(lora_rank=RANK))
    assert restored.lora_rank == RANK
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(_actions(restored, inputs), _actions(head, inputs))


@pytest.mark.parametrize("targets", ["qkv", "all"])
def test_merging_keeps_what_the_model_computes_and_removes_the_adapters(head, targets):
    plain_names = set(head.state_dict())
    head.add_lora(RANK, targets=targets)
    _randomize_adapters(head)
    inputs = _inputs(head.config)
    with_adapters = _actions(head, inputs)
    assert set(head.state_dict()) != plain_names

    head.merge_lora()

    assert head.lora_rank == 0 and not lora.has_lora(head)
    assert set(head.state_dict()) == plain_names
    torch.testing.assert_close(_actions(head, inputs), with_adapters, rtol=1e-4, atol=1e-5)
    # And it is a model that is fine-tuned in the usual way again.
    assert all(parameter.requires_grad for parameter in head.parameters())

    # A merged checkpoint loads into a model that knows nothing of adapters.
    plain = Gr00tN1d7ActionHead(_small_config())
    plain.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(_actions(plain, inputs), with_adapters, rtol=1e-4, atol=1e-5)


def test_adapters_do_change_the_output_once_trained(head):
    inputs = _inputs(head.config)
    before = _actions(head, inputs)
    head.add_lora(RANK)
    _randomize_adapters(head)
    assert not torch.allclose(_actions(head, inputs), before, atol=1e-4)


def test_adapters_cannot_be_added_twice(head):
    head.add_lora(RANK)
    with pytest.raises(ValueError):
        head.add_lora(RANK)


def test_alpha_scales_the_update():
    torch.manual_seed(0)
    layer = torch.nn.Sequential()
    layer.add_module("to_q", torch.nn.Linear(8, 8))
    layer.add_module("other", torch.nn.Linear(8, 8))
    assert lora.add_lora(layer, rank=4, alpha=8.0) == 1  # only the layer named like a target
    assert layer.to_q.scaling["default"] == pytest.approx(2.0)
    lora.merge_lora(layer)
    assert isinstance(layer.to_q, torch.nn.Linear) and not hasattr(layer, "peft_config")


def test_a_module_without_target_layers_is_left_alone():
    assert lora.add_lora(torch.nn.Identity(), rank=4) == 0


def test_merging_a_half_precision_model_keeps_the_update():
    """A model trained in bf16 is merged in fp32: rounded to 16 bits, the update would be lost."""
    torch.manual_seed(0)
    block = torch.nn.Sequential()
    block.add_module("to_q", torch.nn.Linear(64, 64))
    lora.add_lora(block, rank=4, alpha=4)  # a scale of 1
    torch.nn.init.normal_(block.to_q.lora_B["default"].weight, std=1e-3)  # a small update
    block.to(torch.bfloat16)
    base = block.to_q.base_layer.weight.detach().float().clone()
    update = (
        block.to_q.lora_B["default"].weight.float() @ block.to_q.lora_A["default"].weight.float()
    ).detach()

    lora.merge_lora(block)

    assert block.to_q.weight.dtype == torch.float32
    torch.testing.assert_close(block.to_q.weight - base, update, rtol=1e-3, atol=1e-7)
