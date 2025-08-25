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

from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddle.nn import functional as F

from ...nn.mlp import MLP as Ernie4_5MLP
from ...nn.norm import Norm as GeneralNorm
from ..conversion_utils import StateDictNameMapping, init_name_mappings
from ..ernie4_5.modeling import Ernie4_5Attention

# from ...cache_utils import Cache, DynamicCache
# from ...generation import GenerationMixin
# from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
# from ...modeling_layers import GradientCheckpointingLayer
from ..llama.modeling import get_use_casual_mask
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast

# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ..model_utils import PretrainedModel
from ..qwen2.modeling import (
    Qwen2PretrainingCriterion,
    _expand_2d_mask,
    _make_causal_mask,
    is_casual_mask,
)
from ..refined_recompute import recompute as rr_recompute
from ..tensor_parallel_utils import model_parallel_dropout

# from ...processing_utils import Unpack
# from ...utils import TransformersKwargs, can_return_tuple
# from ...utils.generic import OutputRecorder, check_model_inputs
from .configuration import GptOssConfig

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None


class GptOssRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GptOssRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size], default_initializer=paddle.nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(paddle.bfloat16)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GptOssExperts(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size, 2 * self.expert_dim],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.gate_up_proj_bias = paddle.create_parameter(
            shape=[self.num_experts, 2 * self.expert_dim],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.down_proj = paddle.create_parameter(
            shape=[self.num_experts, self.expert_dim, self.hidden_size],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.down_proj_bias = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: paddle.Tensor, router_indices=None, routing_weights=None) -> paddle.Tensor:
        """
        When training is is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.
        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.
        Args:
            hidden_states (paddle.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (paddle.Tensor): (batch_size * token_num, top_k)
            routing_weights (paddle.Tensor): (batch_size * token_num, num_experts)
        Returns:
            paddle.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        if self.training:
            next_states = paddle.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with paddle.no_grad():
                expert_mask = F.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence lenght to get which experts
                # are hit this time around
                expert_hitted = paddle.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted[:]:
                with paddle.no_grad():
                    _, token_idx = paddle.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * F.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = paddle.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * paddle.sigmoid(gate * self.alpha)
            next_states = paddle.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states


class GptOssTopKRouter(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_dim],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.bias = paddle.create_parameter(
            shape=[self.num_experts], dtype="bfloat16", default_initializer=paddle.nn.initializer.Uniform()
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = paddle.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = paddle.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class GptOssMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores


class GptOssRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="bfloat16") / self.dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        if self.inv_freq.dtype != paddle.bfloat16:
            self.inv_freq = 1.0 / (
                self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="bfloat16") / self.dim)
            )
        # [seq_len]
        t = paddle.arange(seq_len, dtype="bfloat16")
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


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rotary_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    first_half, second_half = paddle.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return paddle.concat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = paddle.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = paddle.concat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = F.dropout(scores, p=dropout, training=module.training)
    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GptOssAttention(Ernie4_5Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: GptOssConfig, layer_idx=0):
        super().__init__(config, layer_idx)
        self.sinks = paddle.create_parameter(
            shape=[config.num_attention_heads], dtype="bfloat16", default_initializer=paddle.nn.initializer.Uniform()
        )
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs, seq_len, num_head * head_dim] -> [seq_len / n, bs, num_head * head_dim] (n is model parallelism)

        if self.fuse_attention_qkv:
            mix_layer = self.qkv_proj(hidden_states)
            if self.sequence_parallel:
                target_shape = [
                    batch_size,
                    -1,
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
            query_states = self.q_proj(query_states)
            key_states = self.k_proj(key_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if self.sequence_parallel:
                target_query_shape = [batch_size, -1, self.num_heads, self.head_dim]
                target_key_value_shape = [batch_size, -1, self.num_key_value_heads, self.head_dim]
            else:
                target_query_shape = [0, 0, self.num_heads, self.head_dim]
                target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
            query_states = self.q_proj(query_states.reshape(shape=target_query_shape))
            key_states = self.k_proj(key_states.reshape(shape=target_key_value_shape))
            value_states = value_states.reshape(shape=target_key_value_shape)

        if position_ids is not None and not self.use_fused_rope:
            kv_seq_len = position_ids.max().item() + 1
        else:
            kv_seq_len = key_states.shape[-3]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-3]
        if self.use_fused_rope:
            assert past_key_value is None, "fuse rotary not support cache kv for now"
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
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
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        paddle_version = float(paddle.__version__[:3])
        if not self.config.use_flash_attention or ((paddle_version != 0.0) and (paddle_version <= 2.6)):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        has_gradient = not (query_states.stop_gradient and key_states.stop_gradient and value_states.stop_gradient)
        if (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        ):
            recompute_fn = rr_recompute if any(self.skip_recompute_ops.values()) else recompute
            outputs = recompute_fn(
                self.attn_func,
                query_states,
                self.config,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                training=self.training,
                sequence_parallel=self.sequence_parallel,
                s_aux=self.sinks,
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
                training=self.training,
                sequence_parallel=self.sequence_parallel,
                s_aux=self.sinks,
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

        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Ernie4_5DecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5Config): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = Ernie4_5Attention(config, layer_idx)
        self.mlp = Ernie4_5MLP(config, fuse_up_gate=True)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
        )

        self.hidden_dropout = nn.Dropout(p=config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
                if config.use_bias:
                    mark_as_sequence_parallel_parameter(self.self_attn.o_proj.bias)
                    mark_as_sequence_parallel_parameter(self.mlp.down_proj.bias)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_start_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (paddle.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[paddle.Tensor]): Indices for variable length attention
            position_ids (Optional[paddle.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[paddle.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            attn_mask_start_row_indices=attn_mask_start_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        with model_parallel_dropout(self.config):
            hidden_states = self.hidden_dropout(hidden_states) + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]
        return outputs


class GptOssDecoderLayer(nn.Layer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = GptOssAttention(config=config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[paddle.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = residual + hidden_states
        return hidden_states


class GptOssPreTrainedModel(PretrainedModel):
    config: GptOssConfig
    config_class = GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        # "router_logits": OutputRecorder(GptOssTopKRouter, index=0),
        "hidden_states": GptOssDecoderLayer,
        "attentions": GptOssAttention,
    }
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
    _supports_flash_attention = False
    _supports_flex_attention = False

    @classmethod
    def _get_name_mappings(cls, config: GptOssConfig) -> list[StateDictNameMapping]:
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
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
                [f"layers.{layer_index}.self_attn.q_norm.weight"],
                [f"layers.{layer_index}.self_attn.k_norm.weight"],
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

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "GPTOssModel"
        if "GptOssModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "model." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: GptOssConfig, is_split=True):
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
            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

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
            base_actions = {}
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers, config.num_experts)

        return mappings

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: GptOssConfig, is_fuse=False):
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

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
            if module.bias is not None:
                module.bias.set_value(paddle.zeros_like(module.bias))
        elif isinstance(module, paddle.framework.Parameter):
            module.set_value(paddle.normal(mean=0.0, std=std, shape=module.shape))
        elif isinstance(module, nn.Embedding):
            module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
            if module._padding_idx is not None:
                weight_data = module.weight.numpy()
                weight_data[module._padding_idx] = 0.0
                module.weight.set_value(paddle.to_tensor(weight_data))
        elif isinstance(module, GptOssRMSNorm):
            module.weight.set_value(paddle.ones_like(module.weight))
        elif isinstance(module, GptOssExperts):
            module.gate_up_proj.set_value(paddle.normal(mean=0.0, std=std, shape=module.gate_up_proj.shape))
            module.gate_up_proj_bias.set_value(paddle.zeros_like(module.gate_up_proj_bias))
            module.down_proj.set_value(paddle.normal(mean=0.0, std=std, shape=module.down_proj.shape))
            module.down_proj_bias.set_value(paddle.zeros_like(module.down_proj_bias))
        elif isinstance(module, GptOssAttention):
            module.sinks.set_value(paddle.normal(mean=0.0, std=std, shape=module.sinks.shape))
        elif isinstance(module, GptOssTopKRouter):
            module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
            module.bias.set_value(paddle.normal(mean=0.0, std=std, shape=module.bias.shape))


class GptOssModel(GptOssPreTrainedModel):
    """
    Args:
        config: GptOssConfig
    """

    _no_split_modules = ["GptOssDecoderLayer"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.sequence_parallel = config.sequence_parallel
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList(
            [
                GptOssDecoderLayer(
                    config=config,
                    layer_idx=layer_idx not in self.no_recompute_layers,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GptOssRMSNorm(config.hidden_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        expanded_attn_mask = paddle.where(expanded_attn_mask.to("bool"), 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

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
            attn_mask_startend_row_indices,
            use_reentrant=self.config.recompute_use_reentrant,
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

        if self.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        # embed positions
        if attn_mask_startend_row_indices is not None or get_use_casual_mask():
            attention_mask = None
        else:
            # [bs, seq_len]
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
            )  # [bs, 1, seq_len, seq_len]
            if self.config.use_flash_attention:
                attention_mask = None if is_casual_mask(attention_mask) else attention_mask

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        hidden_states = inputs_embeds

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
                    output_router_logits,
                    past_key_value,
                    use_cache,
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
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None
            if type(layer_outputs) is tuple:
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


def load_balancing_loss_func(
    gate_logits: Union[paddle.Tensor, tuple[paddle.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[paddle.Tensor] = None,
) -> Union[paddle.Tensor, int]:
    r"""
    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`paddle.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = paddle.concat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = paddle.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = paddle.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = paddle.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = paddle.sum(expert_mask.float() * expert_attention_mask, dim=0) / paddle.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = paddle.sum(routing_weights * router_per_expert_attention_mask, dim=0) / paddle.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = paddle.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class GptOssPretrainingCriterion(Qwen2PretrainingCriterion):
    pass


class GptOssForCausalLM(GptOssPreTrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.config = config

        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(10, 10, bias_attr=False)
        self.criterion = GptOssPretrainingCriterion(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing

        if config.sliding_window:
            self.config.sliding_window = False
            # logger.warning("We do not support sliding window attention for now.")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
            # logger.warning(
            #     "You have provided both attn_mask_startend_row_indices and attention_mask. "
            #     "The attn_mask_startend_row_indices will be used."
            # )
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

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        if labels is not None and self.config.use_fused_linear_cross_entropy:
            from paddlenlp_kernel.triton.cut_cross_entropy import linear_cross_entropy

            assert (
                self.config.tensor_parallel_degree <= 1
            ), "The argument `use_fused_linear_cross_entropy` is imcompatiable with tensor parallel "

            masked_lm_loss = linear_cross_entropy(hidden_states, self.lm_head.weight, targets=labels)

            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count
            logits = None
        else:
            logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

            loss = None
            if labels is not None:
                loss = self.criterion(logits, labels)

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


__all__ = ["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel"]
