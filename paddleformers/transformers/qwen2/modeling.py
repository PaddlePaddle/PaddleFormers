# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle Qwen2 model."""
from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Qwen2MLP
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..contrastive_loss import SimpleContrastiveLoss
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..embedding_utils import dist_gather_tensor_with_gradient
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Qwen2Config

__all__ = [
    "Qwen2Model",
    "Qwen2PretrainedModel",
    "Qwen2ForCausalLM",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2SentenceEmbedding",
]


class Qwen2RotaryEmbedding(nn.Layer):
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (
            self.base ** (paddle.cast(paddle.arange(0, self.head_dim, 2), dtype="float32") / self.head_dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        if self.inv_freq.dtype != paddle.float32:
            self.inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.head_dim, 2), dtype="float32") / self.head_dim)
            )
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        # Note: Only for Qwen2MoEForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.attention_bias = config.attention_bias
        self.attn_implementation = config._attn_implementation
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.sequence_parallel = config.sequence_parallel
        self.fuse_attention_qkv = config.fuse_attention_qkv

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.num_attention_heads * self.head_dim

        self.q_proj = GeneralLinear.create(
            self.hidden_size,
            q_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            self.hidden_size,
            kv_hidden_size,
            has_bias=self.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel:
            target_query_shape = [batch_size, -1, self.num_heads, self.head_dim]
            target_key_value_shape = [batch_size, -1, self.num_key_value_heads, self.head_dim]
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        query_states = query_states.reshape(target_query_shape)
        key_states = key_states.reshape(target_key_value_shape)
        value_states = value_states.reshape(target_key_value_shape)

        if attn_mask_startend_row_indices is None and attention_mask is None:
            self.attn_implementation = "sdpa"
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        cos, sin = position_embedding
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.get("attention_dropout", 0.0) if self.training else 0.0,
            scaling=self.scaling,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2DecoderLayer(nn.Layer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
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
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embedding=position_embedding,
            batch_size=batch_size,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Qwen2PretrainedModel(PretrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

    @classmethod
    def _get_name_mappings(cls, config: Qwen2Config) -> list[StateDictNameMapping]:
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
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "Qwen2MoEModel"
        if "Qwen2Model" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "model." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen2Config, is_split=True):

        from ..conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if config.tie_word_embeddings:
                base_actions["lm_head.weight"] = partial(fn, is_column=False)
            else:
                base_actions["lm_head.weight"] = partial(fn, is_column=True)

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                base_actions.pop("lm_head.weight")
                base_actions.pop("embed_tokens.weight")
            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.qkv_proj.bias"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                else:
                    final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: Qwen2Config, is_fuse=False):
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
class Qwen2Model(Qwen2PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(
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
        past_key_value: Tensor,
        use_cache: bool,
        position_embedding: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
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
            past_key_value,
            use_cache,
            position_embedding,
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
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # fmt:skip
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
                    past_key_value,
                    use_cache,
                    position_embedding,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    batch_size=batch_size,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    position_embedding,
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

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM(Qwen2PretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
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

        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
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
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2ForSequenceClassification(Qwen2PretrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = GeneralLinear.create(config.hidden_size, self.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = paddle.equal(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths
            else:
                sequence_lengths = -1

        # pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]
        pooled_logits = logits.gather_nd(paddle.stack([paddle.arange(logits.shape[0]), sequence_lengths], axis=-1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Qwen2ForTokenClassification(Qwen2PretrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = GeneralLinear.create(config.hidden_size, config.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2SentenceEmbedding(Qwen2PretrainedModel):
    def __init__(
        self,
        config: Qwen2Config,
        embedding_temperature: float = 0.02,
    ):
        """Qwen2SentenceEmbedding
        For getting larger batch_size, we use tensor parallel to get larger batch_size.

        Args:
            config (Qwen2Config): _description_
            model (Qwen2Model): _description_
            embedding_temperature (float, optional): _description_. Defaults to 0.02.
        """
        super(Qwen2SentenceEmbedding, self).__init__(config)
        self.config = config
        self.model = Qwen2Model(config)
        self.in_batch_negative_loss = SimpleContrastiveLoss(embedding_temperature)
        self.world_size = dist.get_world_size()
        self.process_rank = dist.get_rank()
        self.embedding_negatives_cross_device = config.embedding_negatives_cross_device
        if self.world_size <= 1:
            self.embedding_negatives_cross_device = False

    def forward(
        self,
        query: Optional[Dict[str, paddle.Tensor]] = None,
        passages: Optional[Dict[str, paddle.Tensor]] = None,
        return_encode=False,
    ):
        """forward"""
        q_reps = self.encode(**query)
        p_reps = self.encode(**passages)

        q_reps = nn.functional.normalize(q_reps, axis=-1)
        p_reps = nn.functional.normalize(p_reps, axis=-1)

        if return_encode:
            return q_reps, p_reps

        if self.embedding_negatives_cross_device:
            q_reps = dist_gather_tensor_with_gradient(q_reps)
            p_reps = dist_gather_tensor_with_gradient(p_reps)

        loss = self.in_batch_negative_loss(q_reps, p_reps)
        return loss

    def encode(
        self,
        input_ids,
        position_ids=None,
        embedding_indices=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **kwargs,
    ):
        """encode"""
        input_type = type(input_ids)
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]
        last_hidden_states = hidden_states.gather_nd(embedding_indices)
        return last_hidden_states
