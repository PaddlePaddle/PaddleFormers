# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
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
"""Paddle DeepSeek_V2 model."""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn import Linear

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from ...utils.log import logger
from ...utils.tools import get_env_device
from ..activations import ACT2FN
from ..llama import fusion_ops
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..moe_gate_auto import PretrainedMoEGate
from ..moe_layer_auto import MoELayer
from .configuration import DeepseekV2Config
from .modeling import (
    DeepseekV2DynamicNTKScalingRotaryEmbedding,
    DeepseekV2LinearScalingRotaryEmbedding,
    DeepseekV2PretrainingCriterion,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
    DeepseekV2YarnRotaryEmbedding,
    _expand_2d_mask,
    _make_causal_mask,
    apply_rotary_pos_emb,
    get_triangle_upper_mask,
    is_casual_mask,
    yarn_get_mscale,
)

__all__ = [
    "DeepseekV2LMHeadAuto",
    "DeepseekV2ForCausalLMAuto",
    "DeepseekV2ModelAuto",
    "DeepseekV2PretrainedModelAuto",
]


def is_pp_enable():
    global_mesh = dist.auto_parallel.get_mesh()
    return "pp" in global_mesh.dim_names


def scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    attn_mask_startend_row_indices=None,
    softmax_scale=1.0,
    training=True,
    sequence_parallel=False,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, v_num_heads, v_head_dim = value_states.shape

    if config.use_flash_attention and flash_attention:
        # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
        # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]

        # Note: Flash Attention does not support softmax_scale, so we need to scale the query_states
        q_head_dim = query_states.shape[-1]
        softmax_scale = softmax_scale * (q_head_dim**0.5)
        query_states = query_states * softmax_scale
        value_padding = paddle.zeros(
            [bsz, kv_seq_len, v_num_heads, head_dim - v_head_dim],
            dtype=value_states.dtype,
        )
        value_states = paddle.concat([value_states, value_padding], axis=-1)

        outputs = fusion_ops.fusion_flash_attention(
            query_states,
            config,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            sequence_parallel=False,
        )

        if isinstance(outputs, tuple):
            outputs[0] = outputs[0].reshape([bsz, q_len, v_num_heads, head_dim])
            outputs[0] = outputs[0][..., :v_head_dim]
            outputs[0] = outputs[0].reshape([bsz, q_len, -1])
        else:
            outputs = outputs.reshape([bsz, q_len, v_num_heads, head_dim])
            outputs = outputs[..., :v_head_dim]
            outputs = outputs.reshape([bsz, q_len, -1])

        if sequence_parallel:
            attn_output = outputs.reshape([bsz * q_len, v_head_dim * num_heads])
        else:
            attn_output = outputs.reshape([bsz, q_len, v_head_dim * num_heads])
        return attn_output

    else:
        #  [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        # merge with the next transpose
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
        value_states = paddle.transpose(value_states, [0, 2, 1, 3])

        # matmul and divide by sqrt(head_dim)
        attn_weights = paddle.matmul(query_states * softmax_scale, key_states.transpose([0, 1, 3, 2]))

        if attn_weights.shape != [bsz, num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)
        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )

        attn_weights = attn_weights + attention_mask
        if not paddle.in_dynamic_mode():
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_weights = F.dropout(attn_weights, p=config.attention_dropout, training=training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        if sequence_parallel:
            attn_output = attn_output.reshape([bsz * q_len, v_head_dim * num_heads])
        else:
            attn_output = attn_output.reshape([bsz, q_len, v_head_dim * num_heads])
        return (attn_output, attn_weights) if output_attentions else attn_output


class MoEGate(PretrainedMoEGate):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super().__init__(config, num_experts, expert_hidden_size, **kwargs)
        # [hidden_size, n_expert]

        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method

        self.weight = paddle.create_parameter(
            shape=[expert_hidden_size, num_experts],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            default_initializer=nn.initializer.Constant(1.0),
        )

        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = paddle.create_parameter(
                shape=[num_experts],
                dtype=paddle.get_default_dtype(),
                default_initializer=nn.initializer.Constant(0.0),
            )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (_type_): [batch_size * seq_len, hidden_size]
        """
        _, h_dim = hidden_states.shape

        # compute gating score
        logits = F.linear(hidden_states, self.weight, None)

        with paddle.amp.auto_cast(False):
            scores = self.gate_score_func(logits=logits)
            scores = scores.cast(paddle.get_default_dtype())

        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = self.topkgating(scores)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        # assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            # grad_loss = paddle.ones(1, dtype=ctx.dtype)
            grad_loss = paddle.to_tensor(1, dtype=ctx.dtype)
            mesh = grad_output.process_mesh
            grad_loss = dist.auto_parallel.api.dtensor_from_local(grad_loss, mesh, [dist.Replicate()])
        return grad_output, grad_loss


class DeepseekV2MLPAuto(nn.Layer):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None, is_moe=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def redistribute_expert(self, mesh, placements):
        """
        Place the experts on different devices.
        """
        self.gate_proj.weight = dist.shard_tensor(self.gate_proj.weight, mesh, placements)
        if self.gate_proj.bias is not None:
            self.gate_proj.bias = dist.shard_tensor(self.gate_proj.bias, mesh, placements)

        self.up_proj.weight = dist.shard_tensor(self.up_proj.weight, mesh, placements)
        if self.up_proj.bias is not None:
            self.up_proj.bias = dist.shard_tensor(self.up_proj.bias, mesh, placements)

        self.down_proj.weight = dist.shard_tensor(self.down_proj.weight, mesh, placements)
        if self.down_proj.bias is not None:
            self.down_proj.bias = dist.shard_tensor(self.down_proj.bias, mesh, placements)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekV2MoEAuto(MoELayer):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config, ipp=None):
        gate = MoEGate(
            config=config,
            num_experts=config.n_routed_experts,
            expert_hidden_size=config.hidden_size,
            top_k=config.num_experts_per_tok,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            drop_tokens=False,
            ipp=ipp,
        )

        super().__init__(
            config=config,
            moe_num_experts=config.n_routed_experts,
            expert_class=DeepseekV2MLPAuto,
            expert_kwargs={"config": config, "intermediate_size": config.moe_intermediate_size},
            gate=gate,
            capacity=2.0,
            ipp=ipp,
        )
        self.alpha = config.aux_loss_alpha
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLPAuto(config=config, intermediate_size=intermediate_size, is_moe=True)

    def forward(self, hidden_states):
        final_hidden_states, l_aux, l_zloss = super().forward(hidden_states)
        if self.training and self.alpha > 0.0:
            final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, l_aux)

        if self.config.n_shared_experts is not None:
            shared_expert_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2AttentionAuto(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.seq_length = config.seq_length

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        # Note (@DrownFish19): For tensor parallel we consider that q_a_proj and kv_a_proj_with_mqa
        # are the small weight and cannot achieve performance gain. So we use the original
        # linear layers. We use the tensor parallel linear layers for q_proj，q_b_proj and kv_b_proj
        # for which are the large weight and can achieve performance gain.

        # fmt: off
        # for without tensor parallel
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias_attr=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias_attr=config.attention_bias)
            self.q_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias_attr=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias_attr=config.attention_bias)
        self.kv_a_layernorm = DeepseekV2RMSNorm(config=config, hidden_size=config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias_attr=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias_attr=config.attention_bias)
        # fmt: on

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_func = scaled_dot_product_attention

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.v_head_dim]).transpose([1, 0, 2, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        # DeepSeekV2 q_lora_rank=1536
        # DeepSeekV2-lite q_lora_rank=None
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape([bsz, q_len, self.num_heads, self.q_head_dim])
        q_nope, q_pe = paddle.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        # DeepSeekV2 kv_lora_rank+qk_rope_head_dim=512+64
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = paddle.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        k_pe = k_pe.reshape([bsz, q_len, 1, self.qk_rope_head_dim])

        # self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim = 128+64
        # self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim) = config.qk_nope_head_dim + self.v_head_dim = 128+128
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).reshape(
            [bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim]
        )

        k_nope, value_states = paddle.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
        kv_seq_len = value_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = paddle.concat([q_nope, q_pe], axis=-1)
        key_states = paddle.concat([k_nope, k_pe.expand([bsz, q_len, self.num_heads, k_pe.shape[-1]])], axis=-1)

        # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            outputs = recompute(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            outputs = self.attn_func(
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                softmax_scale=self.softmax_scale,
                training=self.training,
            )
        if output_attentions:
            attn_output, attn_weights = outputs
        else:
            attn_output = outputs

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DeepseekV2DecoderLayerAuto(nn.Layer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int, layerwise_recompute: bool = False, ipp=None):
        super().__init__()
        self.config = config

        self.enable_recompute = False
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = config.recompute_granularity

        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2AttentionAuto(config=config, layerwise_recompute=layerwise_recompute)
        self.ipp = ipp

        self.mlp = (
            DeepseekV2MoEAuto(config, ipp=self.ipp)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLPAuto(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(config)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_axis)`
            attention_mask (`paddle.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
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
            hidden_states, self_attn_weights, present_key_value = recompute(
                self.self_attn,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **kwargs,
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


class DeepseekV2MTPLayerAuto(DeepseekV2DecoderLayerAuto):
    def __init__(self, config: DeepseekV2Config, layer_idx: int, layerwise_recompute: bool = False, ipp=None):
        super(DeepseekV2MTPLayerAuto, self).__init__(config, layer_idx, layerwise_recompute, ipp)

        self.enorm = DeepseekV2RMSNorm(config)
        self.hnorm = DeepseekV2RMSNorm(config)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        nextn_hidden_state: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:

        hidden_states = self.hnorm(hidden_states)
        nextn_hidden_state = self.enorm(nextn_hidden_state)

        hidden_states = self.eh_proj(paddle.concat([hidden_states, nextn_hidden_state], axis=-1))

        layer_outputs = super(DeepseekV2MTPLayerAuto, self).forward(
            hidden_states,
            position_ids,
            attention_mask,
            output_attentions,
            past_key_value,
            use_cache,
            attn_mask_startend_row_indices,
            **kwargs,
        )

        if type(layer_outputs) is tuple:
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs

        return hidden_states


class DeepseekV2PretrainedModelAuto(PretrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "deepseek_v2"
    _no_split_modules = ["DeepseekV2DecoderLayerAuto"]


class GlobalOutputNet(nn.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        if get_env_device() == "xpu":
            x = paddle.to_tensor(0.0, dtype="float32")
            y = paddle.to_tensor(-1.7005809656952787e38, dtype="float32")
            expanded_attn_mask = paddle.where(expanded_attn_mask, x, y)
        else:
            expanded_attn_mask = paddle.where(expanded_attn_mask.cast("bool"), 0.0, paddle.finfo(dtype).min).astype(
                dtype
            )
        return expanded_attn_mask

    def forward(
        self,
        position_ids,
        attention_mask,
        seq_length,
        batch_size,
        seq_length_with_past,
        cache_length,
        emb_dtype,
        attn_mask_startend_row_indices,
    ):
        if position_ids is None:
            position_ids = paddle.arange(cache_length, seq_length + cache_length, dtype=paddle.int64)
            position_ids = position_ids.unsqueeze(0)

        if (
            attn_mask_startend_row_indices is not None
            or get_use_casual_mask()
            or (self.config.use_flash_attention and self.training)
        ):
            attention_mask = None
        else:
            # [bs, seq_len]
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, emb_dtype
            )  # [bs, 1, seq_len, seq_len]
            if self.config.use_flash_attention:
                attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        return position_ids, attention_mask


@register_base_model
class DeepseekV2ModelAuto(DeepseekV2PretrainedModelAuto):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayerAuto`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.global_layer = GlobalOutputNet(config=config)

        def divide_list_indices(n, k):
            n = n + self.config.pp_extra_layer_num
            base_size = n // k
            extra = n % k

            indices = []
            current_index = -1

            for i in range(k):
                current_index += base_size
                if i < extra:
                    current_index += 1
                indices.append(current_index)
            return indices

        if is_pp_enable():
            mesh = dist.auto_parallel.get_mesh()
            self.pp_indices = divide_list_indices(
                config.num_hidden_layers + config.num_nextn_predict_layers, mesh.get_dim_size("pp")
            )

        def get_pp_stage_id(layer_idx):
            if not is_pp_enable():
                return None
            else:
                for idx, end_idx in enumerate(self.pp_indices):
                    if layer_idx <= end_idx:
                        return idx

        decoder_layers = []
        for layer_idx in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            pp_stage_id = get_pp_stage_id(layer_idx)
            logger.info(f"layer_idx:{layer_idx}, pp_stage_id:{pp_stage_id}")
            if layer_idx < config.num_hidden_layers:
                decoder_layers.append(
                    DeepseekV2DecoderLayerAuto(
                        config, layer_idx, layer_idx not in self.no_recompute_layers, pp_stage_id
                    )
                )
            else:
                decoder_layers.append(
                    DeepseekV2MTPLayerAuto(config, layer_idx, layer_idx not in self.no_recompute_layers, pp_stage_id)
                )

        self.layers = nn.LayerList(decoder_layers)

        self.norm = DeepseekV2RMSNorm(config)

        self.enable_recompute = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        attn_mask_startend_row_indices: Optional[Tensor] = None,
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
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        seq_length -= self.config.num_nextn_predict_layers

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids, attention_mask = self.global_layer(
            position_ids,
            attention_mask,
            seq_length,
            batch_size,
            seq_length_with_past,
            cache_length,
            inputs_embeds.dtype,
            attn_mask_startend_row_indices,
        )

        if self.config.num_nextn_predict_layers > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.num_nextn_predict_layers :, :]  # [B, S, D]
            inputs_embeds = inputs_embeds[:, : -self.config.num_nextn_predict_layers, :]
            inputs_embeds_ori = inputs_embeds

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        mtp_outputs = []

        for idx in range(self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]

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
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.config.num_nextn_predict_layers > 0:
            mtp_outputs.append(hidden_states)

            for nextn in range(self.config.num_nextn_predict_layers):
                decoder_layer = self.layers[nextn + self.config.num_hidden_layers]

                # 构建输入向量
                inputs_embeds_cur_depth = paddle.concat(
                    [inputs_embeds_ori[:, (nextn + 1) :, :], inputs_embeds_extra[:, : (nextn + 1), :]], axis=1
                )

                if inputs_embeds_cur_depth.process_mesh != hidden_states.process_mesh:
                    inputs_embeds_cur_depth = paddle.distributed.reshard(
                        inputs_embeds_cur_depth,
                        hidden_states.process_mesh,
                        inputs_embeds_cur_depth.placements,
                    )
                # 通过该层的decoder_layer进行预测
                past_key_value = None
                layer_outputs = decoder_layer(
                    hidden_states,
                    inputs_embeds_cur_depth,
                    position_ids,
                    attention_mask,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    attn_mask_startend_row_indices,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, mtp_outputs] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV2LMHeadAuto(nn.Layer):
    def __init__(self, config: DeepseekV2Config):
        super(DeepseekV2LMHeadAuto, self).__init__()

        self.config = config

        self.weight = self.create_parameter(
            shape=[config.hidden_size, config.vocab_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.XavierNormal(1.0),
        )

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output
        logits = paddle.matmul(hidden_states, self.weight)
        return logits


class DeepseekV2ForCausalLMAuto(DeepseekV2PretrainedModelAuto):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.config = config
        self.deepseek_v2 = DeepseekV2ModelAuto(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DeepseekV2LMHeadAuto(config)
        self.criterion = DeepseekV2PretrainingCriterion(config)

    def get_input_embeddings(self):
        return self.deepseek_v2.embed_tokens

    def set_input_embeddings(self, value):
        self.deepseek_v2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.deepseek_v2 = decoder

    def get_decoder(self):
        return self.deepseek_v2

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
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLMAuto

        >>> model = DeepseekV2ForCausalLMAuto.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        input_ids.stop_gradient = True
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
        outputs = self.deepseek_v2(
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
        mtp_outputs = outputs[-1]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

        mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs] if len(mtp_outputs) > 0 else []

        return self.criterion(logits, labels, mtp_logits=mtp_logits)

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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def auto_dist_config(self, prefix=""):
        if prefix != "":
            assert prefix.endswith(".")
        config = {
            "mp_config": {
                "parallelize_plan": {
                    f"{prefix}deepseek_v2.embed_tokens": dist.ColWiseParallel(gather_output=True),
                    f"{prefix}deepseek_v2.layers.*.self_attn.q_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v2.layers.*.self_attn.kv_b_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v2.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
                    f"{prefix}deepseek_v2.layers.*.mlp.gate_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v2.layers.*.mlp.up_proj": dist.ColWiseParallel(),
                    f"{prefix}deepseek_v2.layers.*.mlp.down_proj": dist.RowWiseParallel(),
                    f"{prefix}lm_head.weight": dist.ColWiseParallel(),
                }
            },
        }
        return config
