# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.auto_parallel.intermediate.tensor_parallel import (
    PrepareLayerInput,
)
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


import paddle.distributed as dist

from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ..model_utils import PretrainedModel, register_base_model
from .configuration import (
    LLAMA_PRETRAINED_INIT_CONFIGURATION,
    LLAMA_PRETRAINED_RESOURCE_FILES_MAP,
    LlamaConfig,
)
from .modeling import (
    Llama3RotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaNTKScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    _expand_2d_mask,
    _make_causal_mask,
    apply_rotary_pos_emb,
    build_alibi_tensor,
    get_triangle_upper_mask,
    repeat_kv,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

__all__ = [
    "LlamaForCausalLMNet",
    "LlamaForCausalLMNetDPO",
    "LlamaPretrainingCriterionNet",
]


def enable_fuse_ffn_qkv_pass():
    if os.getenv("FLAGS_enable_fused_ffn_qkv_pass") in [
        "True",
        "true",
        "1",
    ]:
        return True
    else:
        return False


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    attn_mask_startend_row_indices=None,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, _, _ = value_states.shape

    if config.use_flash_attention and flash_attention:
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
        version = paddle.version.full_version
        if version != "0.0.0" and version <= "2.5.2":
            if alibi is not None:
                raise ValueError("Flash Attention doesn't support alibi")
            attn_output, attn_weights = flash_attention(
                query_states,
                key_states,
                value_states,
                causal=True,
                return_softmax=output_attentions,
            )
        else:
            if alibi is not None:
                attention_mask = attention_mask.cast(alibi.dtype) + alibi
            if attn_mask_startend_row_indices is not None:
                if len(attn_mask_startend_row_indices.shape) == 2:
                    attn_mask_startend_row_indices = paddle.unsqueeze(attn_mask_startend_row_indices, axis=1)
                attn_output = F.flashmask_attention(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices=attn_mask_startend_row_indices.unsqueeze(-1),
                    causal=True,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    is_causal=attention_mask is None and query_states.shape[1] != 1,
                )
            attn_weights = None

        attn_output = attn_output.reshape([bsz, q_len, head_dim * query_states.shape[-2]])
        return (attn_output, attn_weights) if output_attentions else attn_output
    else:
        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next transpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])
        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))
        # then add alibi bias
        if alibi is not None:
            attn_weights = attn_weights + alibi
        if list(attn_weights.shape) != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        # NOTE: we only call get_triangle_upper_mask under PP setup
        # FIXME ZHUI when we use pipeline parallel, the attention_mask can be None
        # we just make it triangle_upper_mask
        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if list(attention_mask.shape) != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        # [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, num_heads * head_dim]
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


class LlamaRMSNormNet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        if self.config.use_fused_rms_norm:
            return paddle.incubate.nn.functional.fused_rms_norm_ext(hidden_states, self.weight, self.variance_epsilon)[
                0
            ].astype(self.weight.dtype)

        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)

        return hidden_states * self.weight


class LlamaMLPNet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_attention_ffn = config.fuse_attention_ffn
        self.config = config

        if config.fuse_attention_ffn and not enable_fuse_ffn_qkv_pass():
            self.gate_up_fused_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        if self.fuse_attention_ffn and not enable_fuse_ffn_qkv_pass():
            x = swiglu(self.gate_up_fused_proj(x))
        else:
            x = swiglu(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(x)
        return out


class LlamaAttentionNet(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layerwise_recompute: bool = False):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.gqa_or_mqa = config.num_attention_heads != config.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.seq_length = config.seq_length

        self.fuse_attention_qkv = config.fuse_attention_qkv

        self.kv_indices = None
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        self.use_fused_rope = config.use_fused_rope
        if self.use_fused_rope:
            if "gpu" not in paddle.device.get_device() or fused_rotary_position_embedding is None:
                warnings.warn(
                    "Enable fuse rope in the config, but fuse rope is not available. "
                    "Will disable fuse rope. Try using latest gpu version of Paddle."
                )
                self.use_fused_rope = False

        if self.fuse_attention_qkv and not enable_fuse_ffn_qkv_pass():
            self.qkv_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size + 2 * self.config.num_key_value_heads * self.head_dim,
                bias_attr=False,
            )

        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )

            self.k_proj = nn.Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=False,
            )

            self.v_proj = nn.Linear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                bias_attr=False,
            )
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias_attr=False,
        )

        if config.rope:
            self._init_rope()

        self.config = config

    def _init_rope(self):
        if (
            hasattr(self.config, "rope_scaling")
            and self.config.rope_scaling is not None
            and self.config.rope_scaling.get("rope_type", None) == "llama3"
        ):
            self.rotary_emb = Llama3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
                factor=self.config.rope_scaling["factor"],
                high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"],
            )

        elif self.config.rope_scaling_type is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "ntk":
            self.rotary_emb = LlamaNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        elif self.config.rope_scaling_type == "dynamic_ntk":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.config.rope_scaling_factor,
                base=self.config.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {self.config.rope_scaling_type}")

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        alibi: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] or [seq_len / n, bs, num_head * head_dim] (if sequence_parallel)
        # enter tp region

        if self.fuse_attention_qkv and not enable_fuse_ffn_qkv_pass():
            target_shape = [0, 0, self.num_key_value_heads, (self.num_key_value_groups + 2) * self.head_dim]
            mix_layer = self.qkv_proj(hidden_states)
            mix_layer = paddle.reshape_(mix_layer, target_shape)
            query_states, key_states, value_states = paddle.split(
                mix_layer,
                num_or_sections=[self.num_key_value_groups * self.head_dim, self.head_dim, self.head_dim],
                axis=-1,
            )
            if self.gqa_or_mqa:
                query_states = paddle.reshape(query_states, [0, 0, self.num_heads, self.head_dim])
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]

            query_states = self.q_proj(hidden_states).reshape(shape=target_query_shape)
            key_states = self.k_proj(hidden_states).reshape(shape=target_key_value_shape)
            value_states = self.v_proj(hidden_states).reshape(shape=target_key_value_shape)

        kv_seq_len = key_states.shape[-3]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if self.config.rope:
            if self.use_fused_rope:
                assert past_key_value is None, "fuse rotary not support cache kv for now"
                batch_size, seq_length, num_heads, head_dim = query_states.shape
                _, kv_seq_len, num_key_value_heads, _ = key_states.shape
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                paddle_version = float(paddle.__version__[:3])
                if ((paddle_version != 0.0) and (paddle_version <= 2.6)) and (num_heads != num_key_value_heads):
                    query_states, _, _ = fused_rotary_position_embedding(
                        query_states,
                        None,
                        None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
                    key_states, _, _ = fused_rotary_position_embedding(
                        key_states,
                        None,
                        None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
                else:
                    query_states, key_states, _ = fused_rotary_position_embedding(
                        query_states,
                        key_states,
                        v=None,
                        sin=sin,
                        cos=cos,
                        position_ids=position_ids,
                        use_neox_rotary_style=False,
                    )
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                # hack here, because elementwise infer spmd not support broadcast now
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.kv_indices is not None:
            key_states = paddle.index_select(key_states, self.kv_indices, axis=2)
            value_states = paddle.index_select(value_states, self.kv_indices, axis=2)

        # TODO(wj-Mcat): use broadcast strategy when n_kv_heads = 1
        # repeat k/v heads if n_kv_heads < n_heads
        # paddle version > 2.6 or develop support flash-attn with gqa/mqa
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or (paddle_version != 0.0) and (paddle_version <= 2.6):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                scaled_dot_product_attention,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = scaled_dot_product_attention(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # [bs, q_len, num_head * head_dim]
        attn_output = self.o_proj(attn_output)

        # enter sp region
        if not output_attentions:
            attn_weights = None

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class LlamaDecoderLayerNet(nn.Layer):
    def __init__(self, config, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionNet(config, layerwise_recompute)
        self.mlp = LlamaMLPNet(config)
        self.input_layernorm = LlamaRMSNormNet(config)
        self.post_attention_layernorm = LlamaRMSNormNet(config)
        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        alibi: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        # [bs, seq_len, embed_dim] or [seq_len / n, bs, embed_dim] (if sequence_parallel)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        ):
            outputs = recompute(
                self.self_attn,
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                attn_mask_startend_row_indices,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.self_attn(
                hidden_states,
                position_ids,
                past_key_value,
                attention_mask,
                output_attentions,
                use_cache,
                alibi,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )

        if type(outputs) is tuple:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if output_attentions:
            self_attn_weights = outputs[1]

        if use_cache:
            present_key_value = outputs[2 if output_attentions else 1]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # enter tp region

        hidden_states = self.mlp(hidden_states)

        # enter sp region

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class ReshardLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class GlobalOutputNet(nn.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.reshard_replicate = ReshardLayer()

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape, past_key_values_length=past_key_values_length
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    def forward(
        self, position_ids, attention_mask, seq_length, batch_size, seq_length_with_past, cache_length, emb_dtype
    ):
        if position_ids is None and self.config.sep_parallel_degree > 1:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        if not self.config.use_flash_attention and attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            attention_mask = self.reshard_replicate(attention_mask)
        if self.config.alibi:
            if attention_mask is None:
                attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                attention_mask = self.reshard_replicate(attention_mask)

            alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=emb_dtype)
            alibi = self.reshard_replicate(alibi)
        else:
            alibi = None
        if self.config.use_flash_attention and not self.config.alibi:
            # attention_mask in flash_attn is always None for pretrain
            # atttenton_mask is used in scaled_dot_product_attention with alibi_tensor
            attention_mask = None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, emb_dtype
            )  # [bs, 1, seq_len, seq_len]
            attention_mask = self.reshard_replicate(attention_mask)
        return position_ids, attention_mask, alibi


class LlamaPretrainedModelNet(PretrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "llama"
    pretrained_init_configuration = LLAMA_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = LLAMA_PRETRAINED_RESOURCE_FILES_MAP
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]

    # TODO(): wa that loading weight first, then parallelize.
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split):
        return {}


@register_base_model
class LlamaModelNet(LlamaPretrainedModelNet):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayerNet`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )
        self.global_layer = GlobalOutputNet(config=config)

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(LlamaDecoderLayerNet(config, i not in self.no_recompute_layers))

        self.layers = nn.LayerList(decoder_layers)
        self.norm = LlamaRMSNormNet(config)

        self.gradient_checkpointing = False

        self.reshard_row = ReshardLayer()
        self.reshard_row_and_col = ReshardLayer()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
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

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            with paddle.amp.auto_cast(False):
                inputs_embeds = self.embed_tokens(input_ids)

        """
        if position_ids is None and self.config.sep_parallel_degree > 1:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))
        # embed positions
        if not self.config.use_flash_attention and attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        if self.config.alibi:
            if attention_mask is None:
                attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
            alibi = build_alibi_tensor(attention_mask, self.config.num_attention_heads, dtype=inputs_embeds.dtype)
        else:
            alibi = None
        if self.config.use_flash_attention and not self.config.alibi:
            # attention_mask in flash_attn is always None for pretrain
            # atttenton_mask is used in scaled_dot_product_attention with alibi_tensor
            attention_mask = None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]
        """
        position_ids, attention_mask, alibi = self.global_layer(
            position_ids,
            attention_mask,
            seq_length,
            batch_size,
            seq_length_with_past,
            cache_length,
            inputs_embeds.dtype,
        )
        if attention_mask is not None:
            attention_mask = self.reshard_row(attention_mask)
        if alibi is not None:
            alibi = self.reshard_row_and_col(alibi)
        # print(position_ids, attention_mask, alibi)
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    alibi,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    alibi,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )

            if type(layer_outputs) is tuple:
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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class LlamaPretrainingCriterionNet(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, config):

        super(LlamaPretrainingCriterionNet, self).__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = config.tensor_parallel_degree > 1 and config.tensor_parallel_output
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                warnings.warn(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        # Force entropy same kernel
        with paddle.amp.auto_cast(False):
            if isinstance(prediction_scores, paddle.Tensor):
                masked_lm_loss = self.loss_func(
                    prediction_scores.astype("float32")._use_gpudnn(False),
                    masked_lm_labels.unsqueeze(2),
                )
            else:

                masked_lm_loss = self.loss_func(
                    prediction_scores.astype("float32"),
                    masked_lm_labels.unsqueeze(2),
                )

            masked_lm_loss = paddle.masked_select(masked_lm_loss, masked_lm_loss > 0).astype("float32")
            loss = paddle.mean(masked_lm_loss)
        return loss


class LlamaLMHeadNet(nn.Layer):
    def __init__(self, config: LlamaConfig):
        super(LlamaLMHeadNet, self).__init__()
        self.config = config
        vocab_size = config.vocab_size
        self.weight = self.create_parameter(
            shape=[config.hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=False)
        return logits


def layer_input_parallel_row_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Replicate()])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Replicate()]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Shard(0), dist.Replicate()]))
        return tuple(res_inputs)

    return hook


def layer_input_parallel_row_and_col_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Shard(0), dist.Shard(1)])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Shard(0), dist.Shard(1)]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Shard(0), dist.Shard(1)]))
        return tuple(res_inputs)

    return hook


def layer_input_replicate_hook(process_mesh):
    def hook(layer, inputs, output=None):
        res_inputs = []
        for input in inputs:
            if not input.is_dist():
                x = dist.shard_tensor(input, process_mesh, [dist.Replicate(), dist.Replicate()])
                res_inputs.append(dist.reshard(x, process_mesh, [dist.Replicate(), dist.Replicate()]))
            else:
                res_inputs.append(dist.reshard(input, process_mesh, [dist.Replicate(), dist.Replicate()]))
        return tuple(res_inputs)

    return hook


class LlamaForCausalLMNet(LlamaPretrainedModelNet):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.llama = LlamaModelNet(config)
        self.lm_head = LlamaLMHeadNet(config)

    def get_input_embeddings(self):
        return self.llama.embed_tokens

    def set_input_embeddings(self, value):
        self.llama.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llama = decoder

    def get_decoder(self):
        return self.llama

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        attention_mask = kwargs.get("attention_mask", None)
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

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
            )

        return model_kwargs

    def forward(
        self,
        input_ids=None,
        labels=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        attn_mask_startend_row_indices=None,
    ):
        input_ids.stop_gradient = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.llama(
            input_ids,  # [bs, seq_len]
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

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )

        logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

        return logits

    def auto_dist_config(self, prefix=""):
        if prefix != "":
            assert prefix.endswith(".")
        config = {
            "sp_config": {
                "parallelize_plan": {
                    f"{prefix}llama.embed_tokens": [
                        dist.ColWiseParallel(),
                        dist.SequenceParallelBegin(),
                    ],
                    f"{prefix}llama.reshard_row": PrepareLayerInput(layer_input_parallel_row_hook),
                    f"{prefix}llama.reshard_row_and_col": PrepareLayerInput(layer_input_parallel_row_and_col_hook),
                    f"{prefix}llama.global_layer.reshard_replicate": PrepareLayerInput(layer_input_replicate_hook),
                    f"{prefix}llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn": dist.SequenceParallelDisable(),
                    f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}llama.layers.*.mlp": dist.SequenceParallelDisable(need_transpose=False),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                    f"{prefix}lm_head": dist.SequenceParallelEnd(),
                }
            },
            "mp_config": {
                "parallelize_plan": {
                    f"{prefix}llama.embed_tokens": dist.ColWiseParallel(gather_output=True),
                    f"{prefix}llama.reshard_row": PrepareLayerInput(layer_input_parallel_row_hook),
                    f"{prefix}llama.reshard_row_and_col": PrepareLayerInput(layer_input_parallel_row_and_col_hook),
                    f"{prefix}llama.global_layer.reshard_replicate": PrepareLayerInput(layer_input_replicate_hook),
                    f"{prefix}llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.k_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.v_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.gate_up_fused_proj": dist.ColWiseParallel(),
                    f"{prefix}llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                }
            },
            "pp_config": {"split_spec": f"{prefix}llama.layers", "global_spec": f"{prefix}llama.global_layer"},
        }

        return config


class LlamaForCausalLMNetDPO(LlamaForCausalLMNet):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        response_indexs=None,
        attention_mask=None,
        chosen_labels=None,
        rejected_labels=None,
        attn_mask_startend_row_indices=None,
        labels=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        logits = super().forward(
            input_ids=input_ids,
            labels=labels,
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
        return logits
