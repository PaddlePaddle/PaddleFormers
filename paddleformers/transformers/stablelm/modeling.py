# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import copy
from contextlib import nullcontext
from functools import partial
from typing import Optional

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..conversion_utils import split_or_merge_func
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import StableLmConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class StableLmRotaryEmbedding(nn.Layer):
    def __init__(self, config: StableLmConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = self.config.rope_parameters.get("rope_type", "default")
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[StableLmConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0

        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type and hasattr(self, "None_max_seq_len_cached"):
            # dynamic_frequency_update writes this name when layer_type is None.
            self.max_seq_len_cached = int(self.None_max_seq_len_cached)
            delattr(self, "None_max_seq_len_cached")
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class StableLmLayerNormPerHead(nn.Layer):
    def __init__(self, dim, num_heads, eps=1e-5, bias=False, tensor_parallel_size=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        self.tensor_parallel_size = tensor_parallel_size
        # A flat head-major parameter permits concatenating the original per-head checkpoint keys.
        self.weight = self.create_parameter([num_heads * dim], default_initializer=nn.initializer.Constant(1.0))
        self.bias = (
            self.create_parameter([num_heads * dim], is_bias=True, default_initializer=nn.initializer.Constant(0.0))
            if bias
            else None
        )
        for parameter in (self.weight, self.bias):
            if parameter is not None:
                parameter.is_distributed = tensor_parallel_size > 1
                if parameter.is_distributed:
                    parameter.split_axis = 0

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        with paddle.amp.auto_cast(enable=False):
            states = hidden_states.astype("float32")
            mean = states.mean(axis=-1, keepdim=True)
            variance = (states - mean).square().mean(axis=-1, keepdim=True)
            states = (states - mean) * paddle.rsqrt(variance + self.eps)
            states = states * self.weight.astype("float32").reshape([1, self.num_heads, 1, self.dim])
            if self.bias is not None:
                states = states + self.bias.astype("float32").reshape([1, self.num_heads, 1, self.dim])
        return states.astype(dtype)

    def sharded_state_dict(self, structured_name_prefix=""):
        if self.tensor_parallel_size > 1:
            axes = {"weight": 0}
            if self.bias is not None:
                axes["bias"] = 0
            return build_sharded_state_dict(self.state_dict(), axes, structured_name_prefix)
        return super().sharded_state_dict(structured_name_prefix)


def _validate_execution_config(config):
    if config.sequence_parallel and config.tensor_model_parallel_size <= 1:
        raise ValueError("sequence_parallel requires tensor_model_parallel_size > 1")
    if config.apply_rope_fusion:
        raise ValueError("StableLM partial RoPE does not support apply_rope_fusion; use the unfused RoPE path")
    if config.fuse_swiglu and config.hidden_act != "silu":
        raise ValueError("fuse_swiglu requires hidden_act='silu'")
    if config.recompute_granularity is not None and (
        config.recompute_granularity != "full"
        or config.recompute_method != "uniform"
        or config.recompute_num_layers != 1
    ):
        raise ValueError("StableLM recompute supports full/uniform with recompute_num_layers=1")


class StableLmAttention(nn.Layer):
    def __init__(self, config: StableLmConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        _validate_execution_config(config)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim

        q_hidden_size = config.num_attention_heads * self.head_dim
        kv_hidden_size = config.num_key_value_heads * self.head_dim

        if config.tensor_model_parallel_size > 1:
            if self.num_heads % config.tensor_model_parallel_size:
                raise ValueError("num_attention_heads must be divisible by tensor_model_parallel_size")
            if self.num_key_value_heads % config.tensor_model_parallel_size:
                raise ValueError("num_key_value_heads must be divisible by tensor_model_parallel_size")
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        self.rotary_ndims = int(self.head_dim * config.rope_parameters["partial_rotary_factor"])
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            q_hidden_size,
            has_bias=config.use_qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.use_qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            kv_hidden_size,
            has_bias=config.use_qkv_bias,
            config=config,
            tp_plan="colwise",
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = StableLmLayerNormPerHead(
                self.head_dim,
                self.num_heads,
                eps=config.layer_norm_eps,
                tensor_parallel_size=config.tensor_model_parallel_size,
            )
            self.k_layernorm = StableLmLayerNormPerHead(
                self.head_dim,
                self.num_key_value_heads,
                eps=config.layer_norm_eps,
                tensor_parallel_size=config.tensor_model_parallel_size,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        if self.config.sequence_parallel:
            q_len = position_embeddings[0].shape[-2]
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // q_len
        else:
            bsz, q_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(1, 2)
        )

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        cos, sin = position_embeddings
        query_rot, query_pass = (
            query_states[..., : self.rotary_ndims],
            query_states[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_ndims],
            key_states[..., self.rotary_ndims :],
        )
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        query_states = paddle.concat((query_rot, query_pass), axis=-1)
        key_states = paddle.concat((key_rot, key_pass), axis=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        backend = self.config._attn_implementation
        if backend == "flashmask" and (attention_mask is not None or (self.training and self.attention_dropout > 0)):
            backend = "sdpa"
        attention_interface = ALL_ATTENTION_FUNCTIONS[backend]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class StableLmDecoderLayer(nn.Layer):
    def __init__(self, config: StableLmConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.use_parallel_residual = config.use_parallel_residual

        self.self_attn = StableLmAttention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="layer_norm",
            hidden_size=config.hidden_size,
            has_bias=True,
            norm_eps=config.layer_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = None
        if not self.use_parallel_residual:
            self.post_attention_layernorm = GeneralNorm.create(
                config=config,
                norm_type="layer_norm",
                hidden_size=config.hidden_size,
                has_bias=True,
                norm_eps=config.layer_norm_eps,
                input_is_parallel=config.sequence_parallel,
            )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> paddle.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        self_attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if self.use_parallel_residual:
            mlp_output = self.mlp(hidden_states)
            mlp_output = self.dropout(mlp_output)
            hidden_states = residual + self_attn_output + mlp_output
        else:
            residual = residual + self_attn_output
            mlp_output = self.mlp(self.post_attention_layernorm(residual))
            mlp_output = self.dropout(mlp_output)
            hidden_states = residual + mlp_output

        return hidden_states


class StableLmPretrainedModel(PretrainedModel):
    config_class = StableLmConfig
    base_model_class = None
    base_model_prefix = "model"

    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    def _init_weights(self, layer):
        if isinstance(layer, GeneralLMHead) and self.config.tie_word_embeddings:
            return
        if isinstance(layer, tuple(GeneralLinear._global_mapping.values()) + (GeneralLMHead,)):
            context = get_rng_state_tracker().rng_state() if layer.weight.is_distributed else nullcontext()
            with context:
                nn.initializer.Normal(std=self.config.initializer_range)(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, tuple(GeneralEmbedding._global_mapping.values())):
            context = get_rng_state_tracker().rng_state() if layer.weight.is_distributed else nullcontext()
            with context:
                nn.initializer.Normal(std=self.config.initializer_range)(layer.weight)
            padding_idx = self.config.pad_token_id
            if padding_idx is not None and padding_idx >= 0:
                if self.config.tensor_model_parallel_size > 1:
                    padding_idx -= self.config.tensor_parallel_rank * layer.weight.shape[0]
                with paddle.no_grad():
                    if 0 <= padding_idx < layer.weight.shape[0]:
                        layer.weight[padding_idx] = 0

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        actions = {"embed_tokens.weight": partial(fn, is_column=False), "lm_head.weight": partial(fn, is_column=False)}
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}."
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                actions[prefix + f"self_attn.{name}.weight"] = partial(fn, is_column=name != "o_proj")
            if config.use_qkv_bias:
                for name in ("q_proj", "k_proj", "v_proj"):
                    actions[prefix + f"self_attn.{name}.bias"] = partial(fn, is_column=True)
            for name in ("gate_proj", "up_proj", "down_proj"):
                actions[prefix + f"mlp.{name}.weight"] = partial(fn, is_column=name != "down_proj")
            if config.qk_layernorm:
                for name in ("q_layernorm", "k_layernorm"):
                    actions[prefix + f"self_attn.{name}.weight"] = partial(fn, is_column=False)
        return actions

    @classmethod
    def _gen_aoa_config(cls, config: StableLmConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.norm.bias -> {model_prefix}norm.bias",
        ]

        # Per-layer LayerNorm weights (with bias, StableLm uses standard LayerNorm)
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.input_layernorm.bias -> {model_prefix}layers.$LAYER_ID.input_layernorm.bias",
            ]
        )
        if not config.use_parallel_residual:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                    f"model.layers.$LAYER_ID.post_attention_layernorm.bias -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.bias",
                ]
            )

        # Attention Linear weights (with ^T transpose: HF [out, in] -> Paddle [in, out])
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.{proj}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{proj}.weight"
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )

        # QKV bias (no transpose needed, but needs to map if use_qkv_bias=True)
        if config.use_qkv_bias:
            aoa_statements.extend(
                [
                    f"model.layers.$LAYER_ID.self_attn.{proj}.bias -> {model_prefix}layers.$LAYER_ID.self_attn.{proj}.bias"
                    for proj in ["q_proj", "k_proj", "v_proj"]
                ]
            )

        if config.qk_layernorm:
            for name, heads in (
                ("q_layernorm", config.num_attention_heads),
                ("k_layernorm", config.num_key_value_heads),
            ):
                sources = ", ".join(
                    f"model.layers.$LAYER_ID.self_attn.{name}.norms.{head}.weight" for head in range(heads)
                )
                aoa_statements.append(
                    f"{sources} -> {model_prefix}layers.$LAYER_ID.self_attn.{name}.weight"
                    + (", axis=0" if heads > 1 else "")
                )

        # MLP Linear weights (with ^T transpose)
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.mlp.{proj}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{proj}.weight"
                for proj in ["gate_proj", "up_proj", "down_proj"]
            ]
        )

        # lm_head weight (StableLm does NOT tie word embeddings by default)
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: StableLmConfig):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""

        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}norm.bias -> model.norm.bias",
        ]

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
                f"{model_prefix}layers.$LAYER_ID.input_layernorm.bias -> model.layers.$LAYER_ID.input_layernorm.bias",
            ]
        )
        if not config.use_parallel_residual:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
                    f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.bias -> model.layers.$LAYER_ID.post_attention_layernorm.bias",
                ]
            )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{proj}.weight^T -> model.layers.$LAYER_ID.self_attn.{proj}.weight"
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )

        if config.use_qkv_bias:
            aoa_statements.extend(
                [
                    f"{model_prefix}layers.$LAYER_ID.self_attn.{proj}.bias -> model.layers.$LAYER_ID.self_attn.{proj}.bias"
                    for proj in ["q_proj", "k_proj", "v_proj"]
                ]
            )

        if config.qk_layernorm:
            for name, heads in (
                ("q_layernorm", config.num_attention_heads),
                ("k_layernorm", config.num_key_value_heads),
            ):
                targets = ", ".join(
                    f"model.layers.$LAYER_ID.self_attn.{name}.norms.{head}.weight" for head in range(heads)
                )
                aoa_statements.append(
                    f"{model_prefix}layers.$LAYER_ID.self_attn.{name}.weight -> {targets}"
                    + (", axis=0" if heads > 1 else "")
                )

        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.mlp.{proj}.weight^T -> model.layers.$LAYER_ID.mlp.{proj}.weight"
                for proj in ["gate_proj", "up_proj", "down_proj"]
            ]
        )

        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


@register_base_model
class StableLmModel(StableLmPretrainedModel):
    def __init__(self, config: StableLmConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="layer_norm",
            hidden_size=config.hidden_size,
            has_bias=True,
            norm_eps=config.layer_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = StableLmRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ) -> BaseModelOutputWithPast | tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_attentions:
            raise ValueError("StableLM does not currently return attention weights")
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        bsz, seq_length, _ = inputs_embeds.shape

        if self.config.sequence_parallel and seq_length != self.config.max_sequence_length:
            raise ValueError("sequence_parallel requires input length equal to max_sequence_length")

        recompute_enabled = self.training and paddle.is_grad_enabled() and self.config.recompute_granularity == "full"
        if recompute_enabled and past_key_values is not None:
            raise ValueError("past_key_values must be None during recompute training")
        if recompute_enabled and use_cache:
            logger.warning_once("use_cache=True is incompatible with recompute; disabling cache for this forward")
            use_cache = False
        if past_key_values is not None and not use_cache:
            if getattr(past_key_values, "offloading", False):
                raise ValueError("Reading an offloaded cache requires use_cache=True")
            past_key_values = copy.copy(past_key_values)
            past_key_values.layers = [copy.copy(layer) for layer in past_key_values.layers]
            for layer in past_key_values.layers:
                for name, value in vars(layer).items():
                    if isinstance(value, paddle.Tensor):
                        setattr(layer, name, value.clone())

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile([bsz, 1])
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        if kv_seq_len > 0 and seq_length > 1 and causal_mask is None and attn_mask_startend_row_indices is None:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask=paddle.ones([bsz, kv_seq_len + seq_length], dtype="bool"),
                input_shape=(bsz, seq_length),
                past_key_values_length=kv_seq_len,
                dtype=inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_forward = decoder_layer
            if recompute_enabled:
                layer_forward = partial(
                    recompute,
                    decoder_layer,
                    use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
                )
            hidden_states = layer_forward(
                hidden_states,
                attention_mask=causal_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        if not use_cache:
            past_key_values = None
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )
        return output if return_dict else output.to_tuple()


class StableLmForCausalLM(StableLmPretrainedModel):
    base_model_class = StableLmModel
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: StableLmConfig):
        super().__init__(config)
        self.config = config
        self.model = StableLmModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        past_key_values: Cache | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels, loss_mask=loss_mask)

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )


class StableLmEmbeddingPipe(EmbeddingPipe):
    def forward(self, args):
        input_ids, mask, _, _, _ = parse_args(args, is_embed=True)
        outputs = super().forward(args)
        hidden_states, _, position_ids, position_embeddings, offset = parse_args(outputs)
        row_indices = mask if mask is not None and mask.ndim >= 3 and mask.dtype == paddle.int32 else None
        attention_mask = mask.astype("bool") if mask is not None and row_indices is None else None
        attention_mask, row_indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=hidden_states,
            batch_size=input_ids.shape[0],
            seq_length=input_ids.shape[1],
            cache_length=0,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=row_indices,
            prepare_decoder_attention_mask=StableLmPretrainedModel._prepare_decoder_attention_mask,
        )
        outputs = (hidden_states,)
        for value in (
            row_indices if row_indices is not None else attention_mask,
            position_ids,
            position_embeddings,
            offset,
        ):
            if value is not None:
                outputs += (value,)
        return outputs


class StableLmDecoderLayerPipe(StableLmDecoderLayer):
    def forward(self, args):
        hidden_states, mask, position_ids, position_embeddings, _ = parse_args(args)
        row_indices = mask if mask is not None and mask.dtype == paddle.int32 else None
        attention_mask = mask if row_indices is None else None
        embeddings = None if position_embeddings is None else (position_embeddings[0], position_embeddings[1])
        if embeddings is None:
            raise ValueError("StableLM PP requires explicit rotary embeddings")
        layer_forward = super().forward
        if self.training and paddle.is_grad_enabled() and self.config.recompute_granularity == "full":
            layer_forward = partial(
                recompute,
                layer_forward,
                use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
            )
        hidden_states = layer_forward(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=row_indices,
            position_ids=position_ids,
            position_embeddings=embeddings,
        )
        if isinstance(args, tuple):
            return (hidden_states,) + args[1:]
        return hidden_states


class StableLmCriterionLayerPipe(CriterionLayer):
    def __init__(self, config):
        super().__init__(config, return_tuple=False)

    def forward(self, logits, labels):
        loss_mask = None
        if isinstance(labels, tuple):
            labels, loss_mask = labels
        return super().forward(logits, labels, loss_mask=loss_mask)


class StableLmForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = StableLmConfig
    base_model_class = StableLmModel
    base_model_prefix = "model"
    _decoder_layer_cls = StableLmDecoderLayer
    _decoder_layer_pipe_cls = StableLmDecoderLayerPipe
    _embedding_pipe_cls = StableLmEmbeddingPipe
    _rotary_emb_cls = StableLmRotaryEmbedding
    _criterion_pipe_cls = StableLmCriterionLayerPipe
    _norm_cls = "layer_norm"
    _get_tensor_parallel_mappings = StableLmPretrainedModel._get_tensor_parallel_mappings
    _init_weights = StableLmPretrainedModel._init_weights
    _keep_in_fp32_modules = StableLmPretrainedModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = StableLmPretrainedModel.transpose_weight_keys
    _gen_aoa_config = StableLmForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = StableLmForCausalLM._gen_inv_aoa_config

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        # The shared-weight path hardcodes EmbeddingPipe instead of using _embedding_pipe_cls.
        if name_prefix == "model" and layer_desc.layer_func is EmbeddingPipe:
            layer_desc.layer_func = StableLmEmbeddingPipe
        super().add_sequential_layer(layer_desc, name_prefix)


__all__ = [
    "StableLmModel",
    "StableLmForCausalLM",
    "StableLmForCausalLMPipe",
    "StableLmPretrainedModel",
    "StableLmRotaryEmbedding",
    "StableLmDecoderLayer",
    "StableLmAttention",
]
