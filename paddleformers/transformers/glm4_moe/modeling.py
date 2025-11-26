# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Union

import paddle.distributed as dist
from paddle.distributed import fleet
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from paddleformers.transformers.gpt_provider import GPTModelProvider

from ..model_utils import PretrainedModel, register_base_model
from .configuration import Glm4MoeConfig

if TYPE_CHECKING:
    from paddlefleet.transformer import LayerSpec


@dataclass
class GLMMoEModelProvider(GPTModelProvider):
    """Base provider for GLM MoE Models."""

    transformer_layer_spec: Union[
        "LayerSpec", Callable[["GPTModelProvider"], "LayerSpec"]
    ] = get_gpt_decoder_block_spec


class Glm4MoePreTrainedModel(PretrainedModel):
    config: Glm4MoeConfig
    config_class = Glm4MoeConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["mlp.gate.weight", "e_score_correction_bias"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Glm4MoeConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        LAYER_COLWISE = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ]
        FUSE_LAYER_COLWISE = [
            "self_attn.qkv_proj.weight",
        ]

        LAYER_ROWWISE = ["self_attn.o_proj.weight"]

        EXPERT_LAYER_COLWISE = [
            "up_proj.weight",
            "gate_proj.weight",
        ]
        FUSE_EXPERT_LAYER_COLWISE = [
            "up_gate_proj.weight",
        ]

        EXPERT_LAYER_ROWWISE = ["down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]
        FUSE_BIAS_KEYS = [
            "self_attn.qkv_proj.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "model.embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                if not config.fuse_attention_qkv:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                            for k in FUSE_LAYER_COLWISE
                        }
                    )

                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                try:
                    moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
                except:
                    moe_group = None
                expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1
                # TODO: merge disable_ffn_model_parallel and expert_parallel_degree
                if expert_parallel_degree <= 1:
                    # # if disable_ffn_model_parallel is True, disable expert layer tp plan
                    # if not config.disable_ffn_model_parallel:
                    if not config.fuse_attention_ffn:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in EXPERT_LAYER_COLWISE
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                    fn, is_column=True, is_naive_2fuse=True
                                )
                                for e in range(config.n_routed_experts)
                                for k in FUSE_EXPERT_LAYER_COLWISE
                            }
                        )
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(
                                fn, is_column=False
                            )
                            for e in range(config.n_routed_experts)
                            for k in EXPERT_LAYER_ROWWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=False)
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(fn, is_column=True)
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )

                if not config.fuse_attention_ffn:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True
                            )
                            for k in EXPERT_LAYER_COLWISE
                        }
                    )
                else:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                                fn, is_column=True, is_naive_2fuse=True
                            )
                            for k in FUSE_EXPERT_LAYER_COLWISE
                        }
                    )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.shared_experts.{k}": partial(
                            fn, is_column=False
                        )
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                # bias
                if config.attention_bias:
                    if not config.fuse_attention_qkv:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in BIAS_KEYS
                            }
                        )
                    else:
                        actions.update(
                            {
                                f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                                for b in FUSE_BIAS_KEYS
                            }
                        )
            return actions

        mappings = make_base_actions()
        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Glm4MoeConfig, is_fuse=False):
        # return parameter fuse utils
        from ..conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = [
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.qkv_proj.weight",
            ),
            (
                "layers.0.self_attn.q_proj.bias",
                "layers.0.self_attn.k_proj.bias",
                "layers.0.self_attn.v_proj.bias",
                "layers.0.self_attn.qkv_proj.bias",
            ),
        ]
        fuse_gate_up_keys = [
            (
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.experts.0.gate_proj.weight",
                "layers.0.mlp.experts.0.up_proj.weight",
                "layers.0.mlp.experts.0.up_gate_proj.weight",
            ),
            (
                "layers.0.mlp.shared_experts.gate_proj.weight",
                "layers.0.mlp.shared_experts.up_proj.weight",
                "layers.0.mlp.shared_experts.up_gate_proj.weight",
            ),
        ]
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)
        num_experts = getattr(config, "n_routed_experts", 128)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )

            if fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = fn
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = fn

        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn,
                            split_nums=3,
                            is_qkv=True,
                            num_heads=num_heads,
                            num_key_value_heads=num_key_value_heads,
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_gate_up_keys:
                        keys = [key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys]
                        if "experts.0." in keys[0]:
                            for j in range(num_experts):
                                experts_keys = tuple([key.replace("experts.0.", f"experts.{j}.") for key in keys])
                                final_actions[experts_keys] = partial(fn, split_nums=2)
                        else:
                            experts_keys = tuple(keys)
                            final_actions[experts_keys] = partial(fn, split_nums=2)
        return final_actions

    @classmethod
    def _gen_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> {model_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias",
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            ]
        }

        # attention qkv
        if not config.fuse_attention_qkv:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if not config.fuse_attention_ffn:
            aoa_config["aoa_statements"] += (
                [
                    f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
                + [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{p}_proj.weight"
                    for p in ("gate", "up")
                ]
            )
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: Glm4MoeConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            # do cast
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
            # do transpose
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.e_score_correction_bias -> model.layers.$LAYER_ID.mlp.gate.e_score_correction_bias",
        ]

        if not config.fuse_attention_qkv:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
                for x in ("q", "k", "v")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}, axis = 0",
            ]
            aoa_statements += [
                f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                for layer_id in range(config.num_hidden_layers)
                for x in ("q", "k", "v")
            ]

        if not config.fuse_attention_ffn:
            aoa_statements += (
                [
                    f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_experts.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
                + [
                    f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{y}_proj.weight"
                    for y in ("gate", "up")
                ]
            )
        else:
            aoa_statements += [
                f"{model_prefix}layers.0.mlp.up_gate_proj.weight -> model.layers.0.mlp.gate_proj.weight, model.layers.0.mlp.up_proj.weight, fused_ffn",
                "model.layers.0.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
            ]
            aoa_statements += (
                [
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight^T -> model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                    for expert_id in range(config.n_routed_experts)
                ]
                + [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                    for layer_id in range(1, config.num_hidden_layers)
                    for expert_id in range(config.n_routed_experts)
                ]
            )
        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Glm4MoeForCausalLM(Glm4MoePreTrainedModel):
    def __new__(cls, config):
        model_provider_class = GLMMoEModelProvider
        if hasattr(config, "n_routed_experts") and config.n_routed_experts > 0:
            config.moe_layer_freq = 1  # Default to expert layer every layer
        model_provider = model_provider_class.from_config(config)
        return model_provider.provide()


@register_base_model
class Glm4MoeModel(Glm4MoePreTrainedModel):
    pass


class Glm4MoeForCausalLMPipe:
    pass


__all__ = ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM"]
