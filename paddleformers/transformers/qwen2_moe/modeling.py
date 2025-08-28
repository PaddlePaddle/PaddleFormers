# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen2Moe model."""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Qwen2MLP
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate import PretrainedMoEGate
from ..moe_layer import MoELayer
from ..qwen2.modeling import Qwen2Attention, Qwen2RotaryEmbedding
from .configuration import Qwen2MoeConfig

__all__ = [
    "Qwen2MoeModel",
    "Qwen2MoePretrainedModel",
    "Qwen2MoeForCausalLM",
]


def load_balancing_loss_func(gate_logits, num_experts, top_k=2, attention_mask=None):
    """
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Paddle.
    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.
    Args:
        gate_logits (Union[`paddle.Tensor`, Tuple[paddle.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`):
            Number of experts.
        top_k (`int`):
            Number of top k experts to be considered for the loss computation.
        attention_mask (`paddle.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        concatenated_gate_logits = paddle.concat(
            gate_logits, axis=0
        )  # [num_hidden_layers X batch_size X sequence_length, num_experts]

    routing_weights = F.softmax(concatenated_gate_logits, axis=-1)
    _, selected_experts = paddle.topk(routing_weights, top_k, axis=-1)
    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    )  # [num_hidden_layers X batch_size X sequence_length, top_k, num_experts]

    if attention_mask is None or len(attention_mask.shape) == 4:
        # Only intokens strategy has 4-D attention_mask, we currently do not support excluding padding tokens.
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = paddle.mean(expert_mask.astype("float32"), axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = paddle.mean(routing_weights, axis=0)
    else:
        # Exclude the load balancing loss of padding tokens.
        if len(attention_mask.shape) == 2:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape([-1, top_k, num_experts])
            )  # [num_hidden_layers * batch_size * sequence_length, top_k, num_experts]

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = paddle.sum(expert_mask.astype("float32") * expert_attention_mask, axis=0) / paddle.sum(
                expert_attention_mask, axis=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape([-1, num_experts])
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = paddle.sum(
                routing_weights * router_per_expert_attention_mask, axis=0
            ) / paddle.sum(router_per_expert_attention_mask, axis=0)

    overall_loss = paddle.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Qwen2MoeRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class Qwen2MoeMLP(Qwen2MLP):
    def __init__(self, config: Qwen2MoeConfig, intermediate_size=None):
        super().__init__(config, intermediate_size=intermediate_size)


class Qwen2MoeAttention(Qwen2Attention):
    pass


class Qwen2MoeGate(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]
        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        # compute gating score
        logits = F.linear(hidden_states, self.weight, None)

        with paddle.amp.auto_cast(False):
            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.get_default_dtype())

        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.topkgating(scores)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss


class Qwen2MoeSparseMoeBlock(MoELayer):
    def __init__(self, config: Qwen2MoeConfig):
        gate = Qwen2MoeGate(
            config,
            config.num_experts,
            config.hidden_size,
            top_k=config.num_experts_per_tok,
            drop_tokens=False,
        )

        super().__init__(
            config,
            moe_num_experts=config.num_experts,
            expert_class=Qwen2MoeMLP,
            expert_kwargs={"config": config, "intermediate_size": config.moe_intermediate_size},
            gate=gate,
            capacity=2.0,
        )

        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias_attr=False)

    def forward(self, hidden_states):
        final_hidden_states, l_aux, _ = super().forward(hidden_states)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states, l_aux


class Qwen2MoeDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = Qwen2MoeAttention(config, layer_idx)

        if config.num_experts > 0:
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            # num_experts == 0 or this layer is not sparse layer
            self.mlp = Qwen2MoeMLP(config, config.itermediate_size)

        self.input_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            position_ids,
            past_key_value,
            attention_mask,
            output_attentions,
            use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embedding=position_embedding,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Qwen2MoePretrainedModel(PretrainedModel):
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "shared_expert_gate",
        "lm_head",
    ]

    @classmethod
    def _get_name_mappings(cls, config: Qwen2MoeConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.q_proj.bias", None],
                [f"layers.{layer_index}.self_attn.k_proj.bias", None],
                [f"layers.{layer_index}.self_attn.v_proj.bias", None],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

            for expert_idx in range(config.num_experts):
                expert_mappings = [
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.down_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.experts.{expert_idx}.up_proj.weight", None, "transpose"],
                ]
                model_mappings.extend(expert_mappings)
            model_mappings.append([f"layers.{layer_index}.mlp.gate.weight", None, "transpose"])

            model_mappings.append([f"layers.{layer_index}.mlp.shared_expert.gate_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_expert.down_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_expert.up_proj.weight", None, "transpose"])
            model_mappings.append([f"layers.{layer_index}.mlp.shared_expert_gate.weight", None, "transpose"])

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "Qwen2MoeModel"
        if "Qwen2MoeModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "model." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen2MoeConfig, is_split=True):
        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers, num_experts):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            # Add tp split for expert params.
            base_actions = {
                "layers.0.mlp.experts.0.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.experts.0.down_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.experts.0.up_proj.weight": partial(fn, is_column=True),
            }
            for key, action in base_actions.items():
                for i in range(num_layers):
                    newkey = key.replace("layers.0.", f"layers.{i}.")
                    for j in range(num_experts):
                        newkey2 = newkey.replace("experts.0.", f"experts.{j}.")
                        final_actions[newkey2] = action

            # Add tp split for shared expert params.
            base_actions = {
                "layers.0.mlp.shared_expert.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.shared_expert.up_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.shared_expert.down_proj.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers, config.num_experts)

        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Qwen2MoeConfig, is_fuse=False):
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

        fuse_gate_up_keys = (
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.gate_up_fused_proj.weight",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)
        fuse_attention_ffn = getattr(config, "fuse_attention_ffn", False)

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
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = fn
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for fuse_keys in fuse_qkv_keys:
                        keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_keys])
                        final_actions[keys] = partial(
                            fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )
            if not fuse_attention_ffn:
                for i in range(config.num_hidden_layers):
                    keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in fuse_gate_up_keys])
                    final_actions[keys] = partial(fn, split_nums=2)
        return final_actions


@register_base_model
class Qwen2MoeModel(Qwen2MoePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2MoeDecoderLayer`]
    Args:
        config: Qwen2MoeConfig
    """

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__(config)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )
        self.rotary_emb = Qwen2MoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        if config.sequence_parallel:
            self.norm.enable_sequence_parallel()

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        output_router_logits: bool,
        past_key_value: Tensor,
        use_cache: bool,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            output_router_logits,
            past_key_value,
            use_cache,
            position_embedding,
            attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> Union[Tuple, MoEModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        else:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if attn_mask_startend_row_indices is not None:
            attention_mask = None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embedding = self.rotary_emb(hidden_states, seq_length_with_past)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    output_router_logits,
                    past_key_value,
                    use_cache,
                    position_embedding=position_embedding,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    output_router_logits,
                    past_key_value,
                    use_cache,
                    position_embedding=position_embedding,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class Qwen2MoeForCausalLM(Qwen2MoePretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__(config)
        self.model = Qwen2MoeModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing

        if config.sliding_window:
            self.config.sliding_window = False
            logger.warning("We do not support sliding window attention for now.")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, MoECausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            # TODO: support attention mask for other models
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )
            elif len(attention_mask.shape) == 4:
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                    axis=-1,
                )[:, :, -1:, :]

        return model_kwargs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,  # [bs, seq_len]
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
