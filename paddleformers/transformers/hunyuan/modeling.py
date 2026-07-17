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
#
# This file incorporates code from the Qwen team, Alibaba Group, the
# HuggingFace Inc. team, and EleutherAI's GPT-NeoX library. Copyright 2025
# The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights
# reserved. The code has been adapted for Hunyuan architectural differences.
"""Paddle Hunyuan model."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

try:
    from paddleformers.transformers.gpt_provider import GPTModelProvider
except ImportError:
    # paddlefleet is optional for the single-card implementation below. Keep
    # the provider importable so the standard Hunyuan model has no Fleet-only
    # runtime dependency; Fleet construction still fails at its own entrypoint.
    class GPTModelProvider:
        @classmethod
        def from_config(cls, config):
            raise ImportError("paddlefleet is required to construct HunyuanForCausalLMPipeFleet")

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as HunyuanMLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..contrastive_loss import SimpleContrastiveLoss
from ..embedding_utils import dist_gather_tensor_with_gradient
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import HunyuanConfig


@dataclass
class HunyuanModelProvider(GPTModelProvider):
    """Base provider for Hunyuan Models."""

    model_type = "hunyuan_v1_dense"

    use_qk_norm: bool = True

    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True

    transform_rules = {
        "dtype": "params_dtype",
    }

    persist_layer_norm: bool = True
    share_embeddings_and_output_weights: bool = False

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)
        config_dict = asdict(self)

        # Filter out non-serializable values
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if make_serializable(v) is not None}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj if make_serializable(item) is not None]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Skip non-serializable types like partial, function, etc.
                return None

        serializable_config = make_serializable(config_dict)

        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(serializable_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        logger.info(f"Configuration saved in {output_config_file}")


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


class HunyuanAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: HunyuanConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads

        self.tensor_parallel = config.tensor_model_parallel_size > 1
        self.sequence_parallel = config.sequence_parallel
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert (
                self.num_key_value_heads % config.tensor_model_parallel_size == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        kv_hidden_size = self.config.num_key_value_heads * self.head_dim
        q_hidden_size = self.config.num_attention_heads * self.head_dim

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
        self.q_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )  # unlike olmo, only on the head dim!
        self.k_norm = GeneralNorm.create(
            config,
            norm_type="rms_norm",
            hidden_size=self.head_dim,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=self.tensor_parallel,
        )  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        mix_layer = self.qkv_proj(hidden_states)
        if self.sequence_parallel:
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
        if self.gqa_or_mqa:
            query_states = paddle.reshape_(query_states, [0, 0, self.num_heads, self.head_dim])

        # HunYuan applies RoPE before its per-head Q/K RMSNorm layers.
        # Keeping this ordering is necessary for Hugging Face checkpoint parity.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        # key_states shape: [bs, seq_len, num_head, head_dim]
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class HunyuanDecoderLayer(nn.Layer):
    def __init__(self, config: HunyuanConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = HunyuanAttention(config, layer_idx)

        self.mlp = HunyuanMLP(config, fuse_up_gate=True)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.attention_type = config.layer_types[layer_idx]

        if config.sequence_parallel:
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            batch_size=batch_size,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HunyuanPretrainedModel(PretrainedModel):
    config_class = HunyuanConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: HunyuanConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
                f"model.layers.$LAYER_ID.self_attn.query_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight",
                f"model.layers.$LAYER_ID.self_attn.key_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.self_attn.query_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.layers.$LAYER_ID.self_attn.key_layernorm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
        ]

        # lm_head
        if config.tie_word_embeddings:
            if is_fleet:
                aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
            else:
                aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]
        else:
            if is_fleet:
                aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: HunyuanConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if is_fleet:
            aoa_statements += [
                f"{model_prefix}embedding.embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight -> model.layers.$LAYER_ID.self_attn.query_layernorm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight -> model.layers.$LAYER_ID.self_attn.key_layernorm.weight",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.query_layernorm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.key_layernorm.weight",
            ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
        ]
        for layer_id in range(config.num_hidden_layers):
            for x in ("q", "k", "v"):
                aoa_statements += [
                    f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                ]
        if config.attention_bias:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
        ]
        for layer_id in range(config.num_hidden_layers):
            aoa_statements += [
                f"model.layers.{layer_id}.mlp.gate_proj.weight^T -> model.layers.{layer_id}.mlp.gate_proj.weight",
                f"model.layers.{layer_id}.mlp.up_proj.weight^T -> model.layers.{layer_id}.mlp.up_proj.weight",
            ]

        if config.tie_word_embeddings:
            if is_fleet:
                aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
            else:
                aoa_statements += ["lm_head.weight -> _"]
        else:
            if is_fleet:
                aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]
        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class HunyuanRotaryEmbedding(nn.Layer):
    def __init__(self, config: HunyuanConfig):
        super().__init__()
        self.config = config
        self.rope_parameters = config.rope_parameters
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.attention_scaling = 1.0

        # HunYuan Dense V1 uses a fixed DynamicNTKAlpha base, rather than the
        # sequence-dependent generic dynamic-NTK formula used by Qwen models.
        base = self.rope_parameters["rope_theta"]
        rope_type = self.rope_parameters.get("rope_type", self.rope_parameters.get("type", "default"))
        alpha = self.rope_parameters.get("alpha")
        if rope_type == "dynamic" and alpha is not None:
            base *= float(alpha) ** (self.head_dim / (self.head_dim - 2))

        inv_freq = 1.0 / (
            base ** (paddle.arange(0, self.head_dim, 2, dtype="int64").astype("float32") / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@register_base_model
class HunyuanModel(HunyuanPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HunyuanDecoderLayer`]

    Args:
        config: HunyuanConfig
    """

    def __init__(self, config: HunyuanConfig):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [HunyuanDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = HunyuanRotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        past_key_values: Cache,
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
            attention_mask,
            past_key_values,
            use_cache,
            position_embeddings,
            attn_mask_startend_row_indices,
            batch_size,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

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

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

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

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, (decoder_layer) in enumerate(self.layers):
            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    past_key_values,
                    use_cache,
                    position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    batch_size=batch_size,
                )

        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HunyuanForCausalLMFleet(HunyuanPretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = HunyuanModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class HunyuanForCausalLM(HunyuanPretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    @staticmethod
    def _load_single_card_hf_weights(model, model_dir: Union[str, os.PathLike], config: HunyuanConfig) -> None:
        """Stream local Hunyuan Dense V1 weights into a non-Flex single-card model."""
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ImportError("safetensors is required to load Hunyuan HF checkpoints") from error

        weights_path = Path(model_dir) / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing Hunyuan safetensors checkpoint: {weights_path}")

        state_dict = model.state_dict()
        logged_dtype_conversion = False

        def assign(name: str, value: np.ndarray) -> None:
            nonlocal logged_dtype_conversion
            parameter = state_dict[name]
            target_dtype = str(parameter.dtype).removeprefix("paddle.")
            if target_dtype == "bfloat16":
                if not logged_dtype_conversion:
                    logger.debug(
                        f"Converting HF safetensors weight {name} from {value.dtype} "
                        f"to target Paddle dtype {parameter.dtype}"
                    )
                    logged_dtype_conversion = True
                # NumPy cannot construct np.dtype("bfloat16") on every supported
                # version. Convert through float32 and let Paddle perform the
                # final BF16 cast when assigning the parameter.
                value = value.astype(np.float32)
                parameter.set_value(paddle.to_tensor(value, dtype="float32").astype("bfloat16"))
                return
            if str(value.dtype) == "bfloat16":
                if not logged_dtype_conversion:
                    logger.debug(
                        f"Converting HF safetensors weight {name} from {value.dtype} "
                        f"to target Paddle dtype {parameter.dtype}"
                    )
                    logged_dtype_conversion = True
                value = value.astype(np.dtype(target_dtype))
            parameter.set_value(value)

        head_dim = config.head_dim
        num_kv_heads = config.num_key_value_heads
        num_kv_groups = config.num_attention_heads // num_kv_heads
        with safe_open(str(weights_path), framework="np") as checkpoint:
            embedding = checkpoint.get_tensor("model.embed_tokens.weight")
            assign("model.embed_tokens.weight", embedding)
            # LM head is tied in the source checkpoint but appears as a separate
            # parameter in the regular single-card state dict.
            assign("lm_head.weight", embedding)
            assign("model.norm.weight", checkpoint.get_tensor("model.norm.weight"))

            for layer_id in range(config.num_hidden_layers):
                source_prefix = f"model.layers.{layer_id}"
                attention_prefix = f"{source_prefix}.self_attn"
                mlp_prefix = f"{source_prefix}.mlp"
                q_proj = checkpoint.get_tensor(f"{attention_prefix}.q_proj.weight").T.reshape(
                    config.hidden_size, num_kv_heads, num_kv_groups, head_dim
                )
                k_proj = checkpoint.get_tensor(f"{attention_prefix}.k_proj.weight").T.reshape(
                    config.hidden_size, num_kv_heads, 1, head_dim
                )
                v_proj = checkpoint.get_tensor(f"{attention_prefix}.v_proj.weight").T.reshape(
                    config.hidden_size, num_kv_heads, 1, head_dim
                )
                assign(
                    f"{source_prefix}.self_attn.qkv_proj.weight",
                    np.concatenate((q_proj, k_proj, v_proj), axis=2).reshape(config.hidden_size, -1),
                )
                assign(
                    f"{source_prefix}.self_attn.o_proj.weight",
                    checkpoint.get_tensor(f"{attention_prefix}.o_proj.weight").T,
                )
                assign(
                    f"{source_prefix}.self_attn.q_norm.weight",
                    checkpoint.get_tensor(f"{attention_prefix}.query_layernorm.weight"),
                )
                assign(
                    f"{source_prefix}.self_attn.k_norm.weight",
                    checkpoint.get_tensor(f"{attention_prefix}.key_layernorm.weight"),
                )
                assign(
                    f"{source_prefix}.input_layernorm.weight",
                    checkpoint.get_tensor(f"{source_prefix}.input_layernorm.weight"),
                )
                assign(
                    f"{source_prefix}.post_attention_layernorm.weight",
                    checkpoint.get_tensor(f"{source_prefix}.post_attention_layernorm.weight"),
                )
                assign(
                    f"{source_prefix}.mlp.up_gate_proj.weight",
                    np.concatenate(
                        (
                            checkpoint.get_tensor(f"{mlp_prefix}.gate_proj.weight").T,
                            checkpoint.get_tensor(f"{mlp_prefix}.up_proj.weight").T,
                        ),
                        axis=1,
                    ),
                )
                assign(
                    f"{source_prefix}.mlp.down_proj.weight",
                    checkpoint.get_tensor(f"{mlp_prefix}.down_proj.weight").T,
                )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load local HF weights without requiring FlexCheckpoint on a single card."""
        subfolder = kwargs.get("subfolder") or ""
        model_dir = Path(pretrained_model_name_or_path) / subfolder
        if (
            kwargs.get("convert_from_hf", True)
            and kwargs.get("load_checkpoint_format") != "flex_checkpoint"
            and kwargs.get("state_dict") is None
            and model_dir.is_dir()
            and (model_dir / "model.safetensors").is_file()
        ):
            # Keep this fast path's argument boundary aligned with
            # PretrainedModel.from_pretrained: AutoModel deliberately forwards
            # these loader options, but HunyuanForCausalLM.__init__ only accepts
            # the config object.
            config = kwargs.pop("config", None)
            cache_dir = kwargs.pop("cache_dir", None)
            force_download = kwargs.pop("force_download", False)
            kwargs.pop("from_hf_hub", None)
            kwargs.pop("from_aistudio", None)
            kwargs.pop("convert_from_torch", None)
            kwargs.pop("ignore_mismatched_sizes", None)
            kwargs.pop("download_hub", None)
            kwargs.pop("load_via_cpu", None)
            kwargs.pop("load_checkpoint_format", None)
            kwargs.pop("variant", None)
            kwargs.pop("use_safetensors", None)
            kwargs.pop("low_cpu_mem_usage", None)
            kwargs.pop("load_state_as_np", None)
            kwargs.pop("key_mapping", None)
            kwargs.pop("revision", None)
            kwargs.pop("local_files_only", None)
            kwargs.pop("token", None)
            kwargs.pop("use_auth_token", None)
            kwargs.pop("proxies", None)
            kwargs.pop("resume_download", None)
            kwargs.pop("trust_remote_code", None)
            kwargs.pop("_fast_init", None)
            kwargs.pop("convert_from_hf", None)
            kwargs.pop("subfolder", None)

            dtype = kwargs.pop("dtype", None)
            supported_dtypes = (None, "float32", "bfloat16", paddle.float32, paddle.bfloat16)
            if dtype not in supported_dtypes:
                raise ValueError(
                    "single-card Hunyuan HF loading supports dtype=None, 'float32', or 'bfloat16', "
                    f"but received {dtype!r}"
                )

            if config is None:
                config, kwargs = HunyuanConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    subfolder=subfolder,
                    return_unused_kwargs=True,
                    **kwargs,
                )
            if not isinstance(config, HunyuanConfig):
                return super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

            model = cls(config, *model_args, **kwargs)
            cls._load_single_card_hf_weights(model, model_dir, config)
            return model
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(self, config: HunyuanConfig):
        super().__init__(config)
        self.model = HunyuanModel(config)
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
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        loss_mask: Optional[paddle.Tensor] = None,
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
        >>> from transformers import AutoTokenizer, HunyuanForCausalLM

        >>> model = HunyuanForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

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


class HunyuanForSequenceClassification(HunyuanPretrainedModel):
    def __init__(self, config: HunyuanConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = HunyuanModel(config)
        self.score = GeneralLinear.create(config.hidden_size, self.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
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
                sequence_lengths = paddle.eq(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
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


class HunyuanForTokenClassification(HunyuanPretrainedModel):
    def __init__(self, config: HunyuanConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = HunyuanModel(config)
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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
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


class HunyuanSentenceEmbedding(HunyuanPretrainedModel):
    def __init__(
        self,
        config: HunyuanConfig,
        embedding_temperature: float = 0.02,
    ):
        """HunyuanSentenceEmbedding
        For getting larger batch_size, we use tensor parallel to get larger batch_size.

        Args:
            config (HunyuanConfig): _description_
            model (HunyuanModel): _description_
            embedding_temperature (float, optional): _description_. Defaults to 0.02.
        """
        super(HunyuanSentenceEmbedding, self).__init__(config)
        self.config = config
        self.model = HunyuanModel(config)
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
        attention_mask=None,
        position_ids=None,
        embedding_indices=None,
        return_dict=False,
        **kwargs,
    ):
        """encode"""
        input_type = type(input_ids)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            **kwargs,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]
        last_hidden_states = hidden_states.gather_nd(embedding_indices)
        return last_hidden_states


class HunyuanForCausalLMPipeFleet(HunyuanPretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = HunyuanModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


# Exact Hugging Face architecture name used by Hunyuan Dense V1 checkpoints.
HunYuanDenseV1ForCausalLM = HunyuanForCausalLM


class HunyuanForCausalLMPipe(HunyuanForCausalLM):
    config_class = HunyuanConfig
    _decoder_layer_cls = HunyuanDecoderLayer
    _get_tensor_parallel_mappings = HunyuanModel._get_tensor_parallel_mappings
    _init_weights = HunyuanModel._init_weights
    _keep_in_fp32_modules = HunyuanModel._keep_in_fp32_modules
    _rotary_emb_cls = HunyuanRotaryEmbedding
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = HunyuanModel.transpose_weight_keys
    _gen_aoa_config = HunyuanForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = HunyuanForCausalLM._gen_inv_aoa_config


__all__ = [
    "HunyuanModel",
    "HunyuanPretrainedModel",
    "HunyuanForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunyuanForCausalLMPipe",
    "HunyuanForSequenceClassification",
    "HunyuanForTokenClassification",
    "HunyuanSentenceEmbedding",
    "HunyuanForCausalLMFleet",
    "HunyuanForCausalLMPipeFleet",
]
