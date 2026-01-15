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
from paddle import Tensor, nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp
from paddle.nn import functional as F

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import GptOssConfig


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.cat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class GptOssExperts(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.sequence_parallel = config.sequence_parallel
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size, 2 * self.expert_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.gate_up_proj_bias = paddle.create_parameter(
            shape=[self.num_experts, 2 * self.expert_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.down_proj = paddle.create_parameter(
            shape=[self.num_experts, self.expert_dim, self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.down_proj_bias = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size],
            dtype=paddle.get_default_dtype(),
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
        hidden_states = hidden_states.reshape([-1, self.hidden_size])  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        if self.training:
            next_states = paddle.zeros_like(hidden_states, dtype=hidden_states.dtype)
            with paddle.no_grad():
                expert_mask = F.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.transpose(perm=[2, 1, 0])
                # we sum on the top_k and on the sequence lenght to get which experts
                # are hit this time around
                expert_hitted = paddle.nonzero(
                    paddle.greater_than(expert_mask.sum(axis=(-1, -2)), paddle.to_tensor(0, dtype=expert_mask.dtype))
                )
            for expert_idx in expert_hitted[:]:
                with paddle.no_grad():
                    _, token_idx = paddle.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = paddle.clip(gate, min=None, max=self.limit)
                up = paddle.clip(up, min=-self.limit, max=self.limit)
                glu = gate * F.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states = paddle.index_add(
                    next_states,
                    token_idx,
                    0,
                    weighted_output.astype(hidden_states.dtype),
                )
            if self.sequence_parallel:
                next_states = next_states.reshape([-1, self.hidden_size])
            else:
                next_states = next_states.reshape([batch_size, -1, self.hidden_size])

        else:
            hidden_states = paddle.tile(hidden_states, repeat_times=[num_experts, 1])
            hidden_states = hidden_states.reshape((num_experts, -1, self.hidden_size))
            gate_up = paddle.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = paddle.clip(gate, min=None, max=self.limit)
            up = paddle.clip(up, min=-self.limit, max=self.limit)
            glu = gate * F.sigmoid(gate * self.alpha)
            next_states = paddle.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            if self.sequence_parallel:
                next_states = next_states.reshape((num_experts, -1, self.hidden_size))
                next_states = next_states * routing_weights.transpose([0, 1]).reshape((num_experts, -1))[..., None]
            else:
                next_states = next_states.reshape((num_experts, batch_size, -1, self.hidden_size))
                next_states = (
                    next_states * routing_weights.transpose([0, 1]).reshape((num_experts, batch_size, -1))[..., None]
                )
            next_states = next_states.sum(axis=0)

        return next_states


class GptOssTopKRouter(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )
        self.bias = paddle.create_parameter(
            shape=[self.num_experts],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape([-1, self.hidden_dim])
        router_logits = F.linear(hidden_states, self.weight.t(), self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = paddle.topk(router_logits, self.top_k, axis=-1)  # (seq_len, top_k)
        router_top_value = F.softmax(router_top_value, axis=1, dtype=router_top_value.dtype)
        router_scores = paddle.zeros_like(router_logits)
        router_scores = paddle.put_along_axis(router_scores, router_indices, router_top_value, axis=1)
        return router_scores, router_indices


class GptOssMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)
        self.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states):
        if self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        if self.sequence_parallel:
            routed_out = ScatterOp.apply(routed_out)
        return routed_out, router_scores


class GptOssRotaryEmbedding(nn.Layer):
    def __init__(self, config: GptOssConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", config.rope_parameters.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq
        self.inv_freq.stop_gradient = True

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[GptOssConfig] = None,
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

    @paddle.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = freqs

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    first_half, second_half = paddle.chunk(x, 2, axis=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return paddle.cat((first_, second_), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


class GptOssAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx=0):
        """Initialize the attention layer.

        Args:
            config (GptOssConfig): Model configuration.
            layer_idx (int, optional): Index in transformer stack. Defaults to 0.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        assert config.num_attention_heads // config.num_key_value_heads

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sequence_parallel = config.sequence_parallel
        self.attention_bias = config.attention_bias

        self.sequence_parallel = config.sequence_parallel

        self.scaling = self.head_dim**-0.5

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

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
            has_bias=self.attention_bias,
            config=config,
            tp_plan="rowwise",
        )

        self.sinks = paddle.create_parameter(
            shape=[self.num_heads],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Uniform(),
        )

    def forward(
        self,
        hidden_states,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Compute attention outputs.

        Args:
            hidden_states (paddle.Tensor): Input tensor [bsz, seq_len, hidden_size]
            past_key_values (Optional[Cache]): Cached key/value states
            attention_mask (Optional[paddle.Tensor]): Attention mask tensor
            attn_mask_startend_row_indices (Optional[paddle.Tensor]): Variable length attention indices
            position_ids (Optional[paddle.Tensor]): Position indices for RoPE
            output_attentions (bool): Return attention weights if True
            use_cache (bool): Cache key/value states if True

        Returns:
            Tuple containing:
                - attention_output: [bsz, seq_len, hidden_size]
                - attention_weights: Optional attention probabilities
                - updated_key_value_cache: Optional updated cache
        """
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel:
            if batch_size is None:
                batch_size = (
                    hidden_states.shape[0] * self.config.tensor_model_parallel_size // self.config.max_sequence_length
                )
            q_len = self.config.max_sequence_length
            target_query_shape = [batch_size, q_len, self.num_heads, self.head_dim]
            target_key_value_shape = [batch_size, q_len, self.num_key_value_heads, self.head_dim]
        else:
            target_query_shape = [0, 0, self.num_heads, self.head_dim]
            target_key_value_shape = [0, 0, self.num_key_value_heads, self.head_dim]
        # b l h d -> b h l d
        query_states = query_states.reshape(target_query_shape).transpose(1, 2)
        key_states = key_states.reshape(target_key_value_shape).transpose(1, 2)
        value_states = value_states.reshape(target_key_value_shape).transpose(1, 2)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            sink=self.sinks,
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
        return attn_output, attn_weights


class GptOssDecoderLayer(nn.Layer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        if config.sequence_parallel:
            self.post_attention_layernorm.enable_sequence_parallel()
            if not hasattr(config, "disable_ffn_model_parallel"):
                self.input_layernorm.enable_sequence_parallel()
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
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
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value object
        """
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class GptOssPreTrainedModel(PretrainedModel):
    config: GptOssConfig
    config_class = GptOssConfig
    base_model_prefix = "model"
    keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: GptOssConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_config = {
            "aoa_statements": [
                f"_ -> {model_prefix}rotary_emb.inv_freq",
                f"_ -> {model_prefix}rotary_emb.original_inv_freq",
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
                f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate_up_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_up_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_up_proj.weight, fused_ffn",
            ]

        return aoa_config

    # NOTE: These aoa_config items will be removed later. The subsequent AOA parsing module will automatically generate the reverse AOA based on the forward (from_pretrained) AOA.
    @classmethod
    def _gen_inv_aoa_config(cls, config: GptOssConfig):
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
                f"{model_prefix}layers.0.mlp.gate_up_proj.weight -> model.layers.0.mlp.gate_proj.weight, model.layers.0.mlp.up_proj.weight, fused_ffn",
                "model.layers.0.mlp.gate_proj.weight^T -> model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight^T -> model.layers.0.mlp.up_proj.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_up_proj.weight -> model.layers.$LAYER_ID.mlp.shared_experts.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_experts.up_proj.weight, fused_ffn",
                f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_up_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
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


@register_base_model
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
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )

        self.layers = nn.LayerList(
            [
                GptOssDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=config.use_bias,
            norm_eps=self.config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(
            self.config, "sliding_window", None
        ) is not None and "sliding_attention" in getattr(self.config, "layer_types", [])

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Tensor,
        output_attentions: bool,
        output_router_logits: bool,
        past_key_values: Tensor,
        use_cache: bool,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
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
            past_key_values,
            use_cache,
            position_embeddings,
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
        past_key_values: Optional[Cache] = None,
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

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            # [bs, seq_len, dim]
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.sequence_parallel:
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

        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

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
                layer_outputs = self.recompute_training_full(
                    layer_module=decoder_layer,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values] if v is not None)

        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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
        concatenated_gate_logits = paddle.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

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


class GptOssForCausalLM(GptOssPreTrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

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

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        loss_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
    ):
        return_dict = True
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


class GptOssForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = GptOssConfig
    _decoder_layer_cls = GptOssDecoderLayer
    _get_tensor_parallel_mappings = GptOssModel._get_tensor_parallel_mappings
    _init_weights = GptOssModel._init_weights
    _rotary_emb_cls = GptOssRotaryEmbedding
    _keep_in_fp32_modules = GptOssModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = GptOssModel.transpose_weight_keys
    _gen_aoa_config = GptOssForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = GptOssForCausalLM._gen_inv_aoa_config


__all__ = ["GptOssForCausalLM", "GptOssModel", "GptOssPreTrainedModel", "GptOssForCausalLMPipe"]
