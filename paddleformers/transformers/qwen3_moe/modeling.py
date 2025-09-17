# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen3Moe model."""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..moe_layer import MoELayer
from ..qwen2_moe.modeling import Qwen2MoeGate, Qwen2MoeMLP, load_balancing_loss_func
from ..qwen3.modeling import Qwen3Attention, Qwen3RotaryEmbedding
from .configuration import Qwen3MoeConfig

__all__ = [
    "Qwen3MoeModel",
    "Qwen3MoePretrainedModel",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForCausalLMPipe",
]


class Qwen3MoeRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Qwen3MoeMLP(Qwen2MoeMLP):
    pass


class Qwen3MoeAttention(Qwen3Attention):
    pass


class Qwen3MoeGate(Qwen2MoeGate):
    pass


class ExpertParallelQwen3MoeSparseMoeBlock(MoELayer):
    def __init__(self, config: Qwen3MoeConfig):
        gate = Qwen3MoeGate(
            config,
            config.num_experts,
            config.hidden_size,
            top_k=config.num_experts_per_tok,
            drop_tokens=False,
            norm_topk_prob=config.norm_topk_prob,
        )

        super().__init__(
            config,
            moe_num_experts=config.num_experts,
            expert_class=Qwen3MoeMLP,
            expert_kwargs={"config": config, "intermediate_size": config.moe_intermediate_size},
            gate=gate,
            capacity=2.0,
        )
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states):
        final_hidden_states, l_aux, _ = super().forward(hidden_states)

        return final_hidden_states, l_aux


class Qwen3MoeSparseMoeBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = GeneralLinear.create(config.hidden_size, config.num_experts, has_bias=False, linear_type="default")
        self.experts = nn.LayerList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """ """

        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
            sequence_length = max_sequence_length
            hidden_dim = hidden_states.shape[1]
        else:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view([-1, hidden_dim])
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, axis=1, dtype=paddle.float32)
        # (batch * sequence_length, topk)
        routing_weights, selected_experts = paddle.topk(routing_weights, self.top_k, axis=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(axis=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = paddle.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])
        # [num_experts, topk, bs*seq]
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(axis=-1)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, hidden_dim])
            current_hidden_states = expert_layer(current_state) * routing_weights[idx, top_x].unsqueeze(-1)
            final_hidden_states.index_add_(
                index=idx.reshape([-1]), axis=0, value=current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape([batch_size, sequence_length, hidden_dim])
        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if config.num_experts > 0:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            # num_experts == 0 or this layer is not sparse layer
            self.mlp = Qwen3MoeMLP(config)

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
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
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
            position_embeddings=position_embeddings,
            batch_size=batch_size,
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


class Qwen3MoePretrainedModel(PretrainedModel):
    config_class = Qwen3MoeConfig
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
        "gate",
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen3MoeConfig, is_split=True):
        """Generate tensor parallel mappings for model conversion."""
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

        LAYER_ROWWISE = ["self_attn.o_proj.weight"]

        EXPERT_LAYER_COLWISE = [
            "up_proj.weight",
            "gate_proj.weight",
        ]

        EXPERT_LAYER_ROWWISE = ["down_proj.weight"]

        BIAS_KEYS = [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ]

        def make_base_actions():
            actions = {
                "lm_head.weight": partial(fn, is_column=False),
                "embed_tokens.weight": partial(fn, is_column=False),
            }
            for layer_idx in range(config.num_hidden_layers):
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=True)
                        for k in LAYER_COLWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.{k}": partial(fn, is_column=False)
                        for k in LAYER_ROWWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(fn, is_column=True)
                        for e in range(config.num_experts)
                        for k in EXPERT_LAYER_COLWISE
                    }
                )
                actions.update(
                    {
                        f"{cls.base_model_prefix}.layers.{layer_idx}.mlp.experts.{e}.{k}": partial(fn, is_column=False)
                        for e in range(config.num_experts)
                        for k in EXPERT_LAYER_ROWWISE
                    }
                )
                # bias
                if config.attention_bias:
                    actions.update(
                        {
                            f"{cls.base_model_prefix}.layers.{layer_idx}.{b}": partial(fn, is_column=True)
                            for b in BIAS_KEYS
                        }
                    )

            return actions

        mappings = make_base_actions()
        return mappings


@register_base_model
class Qwen3MoeModel(Qwen3MoePretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3MoeDecoderLayer`]
    Args:
        config: Qwen3MoeConfig
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )
        self.rotary_emb = Qwen3MoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

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
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
        batch_size: int = None,
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
            position_embeddings,
            attn_mask_startend_row_indices,
            batch_size,
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
        position_embeddings = self.rotary_emb(hidden_states, seq_length_with_past)

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
                    position_embeddings=position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    batch_size=batch_size,
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
                    position_embeddings=position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    batch_size=batch_size,
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


class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
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
        **kwargs,
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
        loss_mask: Optional[paddle.Tensor] = None,
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


class Qwen3MoeForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Qwen3MoeConfig
    _decoder_layer_cls = Qwen3MoeDecoderLayer
    _get_tensor_parallel_mappings = Qwen3MoeModel._get_tensor_parallel_mappings
    _init_weights = Qwen3MoeModel._init_weights
    _keep_in_fp32_modules = Qwen3MoeModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Qwen3MoeModel.transpose_weight_keys
