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

from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

from ...generation import GenerationMixin
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as BaseMLP
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..activations import ACT2FN
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Gemma3Config, Gemma3TextConfig


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, config):
        num_embeddings = config.vocab_size
        embedding_dim = config.hidden_size
        padding_idx = config.pad_token_id

        # TODO: config cannot be updated when pp!=1, temporarily hard-coded
        embed_scale = config.hidden_size**0.5

        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", paddle.tensor(embed_scale), persistable=False)

    def forward(self, input_ids: paddle.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(BaseMLP):
    def __init__(self, config: Gemma3TextConfig, fuse_up_gate=False):
        super().__init__(config, fuse_up_gate=fuse_up_gate)
        self.act_fn = ACT2FN[config.hidden_activation]


class Gemma3RMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6, input_is_parallel=False):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )

        if input_is_parallel:
            self.enable_sequence_parallel()

    def _norm(self, x):
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class Gemma3RMSNormPipe(Gemma3RMSNorm):
    def __init__(self, config):
        hidden_size = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)
        input_is_parallel = getattr(config, "sequence_parallel", False)
        super().__init__(hidden_size, eps, input_is_parallel)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return super().forward(x)


class Gemma3RotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        rope_parameters = self.config.rope_parameters
        self.rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))

        # TODO: The rope_type here is the 'default', which supports some models such as `gemma-3-1b-it`.
        # Other models, such as `gemma-3-4b-it`, require other types, such as 'linear', which is not supported now.
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Gemma3TextConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`PreTrainedConfig`]):
                The model configuration.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


class Gemma3Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not config.use_bidirectional_attention
        self.attn_implementation = config._attn_implementation
        self.fuse_attention_qkv = config.fuse_attention_qkv

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        assert config.num_attention_heads // config.num_key_value_heads

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = config.num_key_value_heads * self.head_dim
        q_hidden_size = config.num_attention_heads * self.head_dim

        if not self.fuse_attention_qkv:
            self.q_proj = GeneralLinear.create(
                config.hidden_size,
                q_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.k_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
            self.v_proj = GeneralLinear.create(
                config.hidden_size,
                kv_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
        else:
            self.qkv_proj = GeneralLinear.create(
                config.hidden_size,
                q_hidden_size + 2 * kv_hidden_size,
                has_bias=config.attention_bias,
                config=config,
                tp_plan="colwise",
            )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="rowwise",
        )

        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps)

        if config.sequence_parallel:
            self.q_norm.enable_sequence_parallel()
            self.k_norm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[tuple[paddle.Tensor]]]:
        if not self.fuse_attention_qkv:
            if self.config.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
            else:
                bsz, q_len, _ = hidden_states.shape

            hidden_shape = (bsz, q_len, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).reshape(hidden_shape)
            key_states = self.k_proj(hidden_states).reshape(hidden_shape)
            value_states = self.v_proj(hidden_states).reshape(hidden_shape)
        else:
            mix_layer = self.qkv_proj(hidden_states)
            if self.config.sequence_parallel:
                max_sequence_length = self.config.max_sequence_length
                bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
                q_len = max_sequence_length
                target_shape = [
                    bsz,
                    q_len,
                    self.num_key_value_heads,
                    (self.num_key_value_groups + 2) * self.head_dim,
                ]
            else:
                target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
            mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )
            query_states = query_states.reshape([0, 0, -1, self.head_dim])

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # b l h d -> b h l d
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if attn_mask_startend_row_indices is None and attention_mask is None:
            self.attn_implementation = "sdpa"
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Layer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config, fuse_up_gate=config.fuse_attention_ffn)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings: Tuple[paddle.Tensor, paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> tuple[paddle.FloatTensor, Optional[tuple[paddle.FloatTensor, paddle.FloatTensor]]]:
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Gemma3PreTrainedModel(PretrainedModel):
    config_class = Gemma3Config
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
        "lm_head",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Gemma3TextConfig):
        model_prefix = "" if cls == cls.base_model_prefix else "model."
        aoa_config = {
            "aoa_statements": [
                # load tied weight
                "model.embed_tokens.weight -> lm_head.weight",
                # others
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.layers.$LAYER_ID.pre_feedforward_layernorm.weight -> {model_prefix}layers.$LAYER_ID.pre_feedforward_layernorm.weight",
                f"model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight",
                # do transpose
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
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
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.{p}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{p}_proj.weight"
                for p in ("gate", "up")
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
            ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: Gemma3TextConfig):
        model_prefix = "" if cls == cls.base_model_prefix else "model."
        aoa_statements = [
            # ignore tied weights
            "lm_head.weight -> _",
            # do transpose
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            # others
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.pre_feedforward_layernorm.weight -> model.layers.$LAYER_ID.pre_feedforward_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_feedforward_layernorm.weight -> model.layers.$LAYER_ID.post_feedforward_layernorm.weight",
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
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.{y}_proj.weight^T -> model.layers.$LAYER_ID.mlp.{y}_proj.weight"
                for y in ("gate", "up")
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.0.mlp.up_gate_proj.weight -> model.layers.0.mlp.gate_proj.weight, model.layers.0.mlp.up_proj.weight, fused_ffn",
                "model.layers.0.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
            ]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Gemma3TextModel(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.sequence_parallel = config.sequence_parallel

        self.embed_tokens = Gemma3TextScaledWordEmbedding(config)
        self.layers = nn.LayerList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNormPipe(config)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

        if config.sequence_parallel:
            self.norm.enable_sequence_parallel()

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        attention_mask: paddle.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            position_ids,
            output_attentions,
            use_cache,
            attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if self.config.use_bidirectional_attention:
            logger.warning(
                "Bidirectional attention is currently unsupported. "
                "Disabling 'use_bidirectional_attention' automatically."
            )
            self.config.use_bidirectional_attention = False

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        # Create the causal mask and row indices
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        causal_mask_mapping = {"full_attention": full_mask}
        attn_mask_startend_row_indices_mapping = {"full_attention": full_indices}

        # if model has sliding layer
        if self.has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                attn_mask_startend_row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        # TODO: apply different RoPE settings based on 'layer_type'
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # decoder layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final Norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return outputs
        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Gemma3TextConfig
    # TODO: base_model_prefix should be same with submodel variable name
    # base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "position_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

        >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
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
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = paddle.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

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


class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel):
    """
    Gemma3TextForSequenceClassification is a text-only sequence classification model that works with Gemma3TextConfig.
    It uses the generic sequence classification implementation for efficiency and consistency.
    """

    config_class = Gemma3TextConfig


class Gemma3ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Gemma3TextConfig
    _decoder_layer_cls = Gemma3DecoderLayer
    _get_tensor_parallel_mappings = Gemma3TextModel._get_tensor_parallel_mappings
    _init_weights = Gemma3TextModel._init_weights
    _rotary_emb_cls = Gemma3RotaryEmbedding
    _embed_cls = Gemma3TextScaledWordEmbedding
    _rms_norm_pipe_cls = Gemma3RMSNormPipe
    _keep_in_fp32_modules = Gemma3TextModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Gemma3TextModel.transpose_weight_keys
    _gen_aoa_config = Gemma3ForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Gemma3ForCausalLM._gen_inv_aoa_config


__all__ = [
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
    "Gemma3ForCausalLMPipe",
]
