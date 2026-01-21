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
"""Paddle Qwen3-Next model."""

from typing import Any, List, Optional

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp

from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from ...nn.norm import mark_as_sequence_parallel_parameter
from ...nn.pp_model import GeneralModelForCausalLMPipe, RMSNormPipe, parse_args
from ...utils.log import logger
from ..configuration_utils import PretrainedConfig
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..qwen2_moe.modeling import Qwen2MoeSparseMoeBlock, load_balancing_loss_func
from ..qwen3_moe.modeling import Qwen3MoeAttention, Qwen3MoeMLP
from .configuration import Qwen3NextConfig

__all__ = [
    "Qwen3NextPretrainedModel",
    "Qwen3NextModel",
    "Qwen3NextForCausalLM",
    "Qwen3NextForCausalLMPipe",
]

causal_conv1d_update, causal_conv1d_fn = None, None
chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
FusedRMSNormGated = None
is_fast_path_available = False


class Qwen3NextRMSNormGated(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(paddle.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(paddle.float32))

        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"dim={tuple(self.weight.shape)}, variance_epsilon={self.variance_epsilon}, dtype={self.weight.dtype}"


class Qwen3NextDynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the linear attention
    cache (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for gated deltanet cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For linear attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `recurrent_states` represents the recurrent state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    is_compileable = False

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")

        # Initialize everything to None -> will be lazy initialized to allow multi-gpu (device_map) inference
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Tensor, Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = paddle.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = paddle.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        return self.conv_states[self.last_linear_layer] is not None


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    seq_len: Optional[int] = None,
) -> tuple[Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`paddle.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype="int64").to(dtype="float") / dim))
    return inv_freq, attention_factor


class Qwen3NextRotaryEmbedding(nn.Layer):
    inv_freq: Tensor

    def __init__(self, config: Qwen3NextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        assert self.rope_type == "default", f"Currently only supports default rope_type, but got {self.rope_type}"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Qwen3NextConfig] = None,
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
        partial_rotary_factor = config.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    @paddle.no_grad()
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])

            position_ids_expanded = position_ids[:, None, :].float()

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            emb = paddle.concat((freqs, freqs), axis=-1)

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextRMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6, input_is_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(paddle.zeros(dim))

        if input_is_parallel:
            mark_as_sequence_parallel_parameter(self.weight)

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}, dtype={self.weight.dtype}"


class Qwen3NextAttention(Qwen3MoeAttention):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = GeneralLinear.create(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.q_norm = Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            input_is_parallel=self.sequence_parallel,
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            input_is_parallel=self.sequence_parallel,
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        query_states, gate = paddle.chunk(query_states.view(bsz, q_len, -1, self.head_dim * 2), chunks=2, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)

        query_states = self.q_norm(query_states.view(bsz, q_len, -1, self.head_dim))
        key_states = self.k_norm(key_states.view(bsz, q_len, -1, self.head_dim))
        value_states = value_states.reshape(bsz, q_len, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=query_states.transpose(1, 2),
            key=key_states.transpose(1, 2),
            value=value_states.transpose(1, 2),
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output * paddle.sigmoid(gate)

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape(-1, attn_output.shape[-1])

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def paddle_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = paddle.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = paddle.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def paddle_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(paddle.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = paddle.triu(paddle.ones(chunk_size, chunk_size, dtype=paddle.bool), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + paddle.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        paddle.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = paddle.zeros_like(value)
    mask = paddle.triu(paddle.ones(chunk_size, chunk_size, dtype=paddle.bool), diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def paddle_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(paddle.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = paddle.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
    last_recurrent_state = (
        paddle.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(num_heads):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.squeeze(2)  # => [batch_size, seq_len, head_dim]
    sin = sin.squeeze(2)  # => [batch_size, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = paddle.cat([q_embed, q_pass], dim=-1)
    k_embed = paddle.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3NextGatedDeltaNet(nn.Layer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.sequence_parallel = config.sequence_parallel

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1D(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias_attr=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias_attr=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias_attr=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(paddle.ones(self.num_v_heads))

        A = paddle.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(paddle.log(A))

        self.norm = (
            Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=paddle.device.get_device(),
                dtype=config.dtype if config.dtype is not None else paddle.get_default_dtype(),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias_attr=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or paddle_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or paddle_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or paddle_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "paddle implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation "
                "and https://github.com/Dao-AILab/causal-conv1d"
            )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.shape[:-1] + [
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        ]
        new_tensor_shape_ba = mixed_ba.shape[:-1] + [self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads]

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = paddle.split(mixed_qkvz, split_arg_list_qkvz, axis=3)
        b, a = paddle.split(mixed_ba, split_arg_list_ba, axis=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)
        z = z.reshape(z.shape[0], z.shape[1], -1, self.head_v_dim)
        b = b.reshape(b.shape[0], b.shape[1], self.num_v_heads)
        a = a.reshape(a.shape[0], a.shape[1], self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: Tensor,
        cache_params: Optional[Qwen3NextDynamicCache] = None,
        cache_position: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        if self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states).unsqueeze(0)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = paddle.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = paddle.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            axis=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, axis=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, axis=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)

        if self.sequence_parallel:
            output = ScatterOp.apply(output.squeeze(0))

        return output


class Qwen3NextDecoderLayer(nn.Layer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config

        # token mixer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None
        expert_model_parallel_size = dist.get_world_size(moe_group) if moe_group is not None else 1

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = (
                QuickAccessMoEFactory.create_from_model_name(
                    pretrained_config=config,
                    expert_class=Qwen3MoeMLP,
                    gate_activation="softmax",
                    expert_activation="silu",
                    train_topk_method="greedy",
                    inference_topk_method="greedy",
                    transpose_gate_weight=False,
                )
                if expert_model_parallel_size > 1
                else Qwen2MoeSparseMoeBlock(config)
            )
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: tuple[Tensor, Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
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
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3NextPretrainedModel(PretrainedModel):
    config_class = Qwen3NextConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    transpose_weight_keys = [
        "gate",
        "gate_proj",
        "in_proj_ba",
        "in_proj_qkvz",
        "out_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "shared_expert_gate",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3NextConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
            # gate
            f"model.layers.$LAYER_ID.mlp.gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
            f"model.layers.$LAYER_ID.mlp.shared_expert_gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_expert_gate.weight",
            # linear_attn
            f"model.layers.$LAYER_ID.linear_attn.A_log -> {model_prefix}layers.$LAYER_ID.linear_attn.A_log",
            f"model.layers.$LAYER_ID.linear_attn.conv1d.weight -> {model_prefix}layers.$LAYER_ID.linear_attn.conv1d.weight",
            f"model.layers.$LAYER_ID.linear_attn.dt_bias -> {model_prefix}layers.$LAYER_ID.linear_attn.dt_bias",
            f"model.layers.$LAYER_ID.linear_attn.in_proj_ba.weight^T -> {model_prefix}layers.$LAYER_ID.linear_attn.in_proj_ba.weight",
            f"model.layers.$LAYER_ID.linear_attn.in_proj_qkvz.weight^T -> {model_prefix}layers.$LAYER_ID.linear_attn.in_proj_qkvz.weight",
            f"model.layers.$LAYER_ID.linear_attn.norm.weight -> {model_prefix}layers.$LAYER_ID.linear_attn.norm.weight",
            f"model.layers.$LAYER_ID.linear_attn.out_proj.weight^T -> {model_prefix}layers.$LAYER_ID.linear_attn.out_proj.weight",
        ]

        # self_attn
        aoa_statements += [
            f"model.layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v", "o")
        ]
        aoa_statements += [
            f"model.layers.$LAYER_ID.self_attn.{x}_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.{x}_norm.weight"
            for x in ("q", "k")
        ]

        # experts
        aoa_statements += [
            f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{x}_proj.weight"
            for x in ("gate", "up", "down")
        ]
        aoa_statements += [
            f"model.layers.$LAYER_ID.mlp.shared_expert.{x}_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_expert.{x}_proj.weight"
            for x in ("gate", "up", "down")
        ]

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3NextConfig):
        model_prefix = "" if cls == cls.base_model_class else "model."
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            # gate
            f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight^T -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
            f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert_gate.weight^T -> model.layers.$LAYER_ID.mlp.shared_expert_gate.weight",
            # linear_attn
            f"{model_prefix}layers.$LAYER_ID.linear_attn.A_log -> model.layers.$LAYER_ID.linear_attn.A_log",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.conv1d.weight -> model.layers.$LAYER_ID.linear_attn.conv1d.weight",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.dt_bias -> model.layers.$LAYER_ID.linear_attn.dt_bias",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.in_proj_ba.weight^T -> model.layers.$LAYER_ID.linear_attn.in_proj_ba.weight",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.in_proj_qkvz.weight^T -> model.layers.$LAYER_ID.linear_attn.in_proj_qkvz.weight",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.norm.weight -> model.layers.$LAYER_ID.linear_attn.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.linear_attn.out_proj.weight^T -> model.layers.$LAYER_ID.linear_attn.out_proj.weight",
        ]

        # self_attn
        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_proj.weight^T -> model.layers.$LAYER_ID.self_attn.{x}_proj.weight"
            for x in ("q", "k", "v", "o")
        ]
        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.{x}_norm.weight -> model.layers.$LAYER_ID.self_attn.{x}_norm.weight"
            for x in ("q", "k")
        ]

        # experts
        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{x}_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.{x}_proj.weight"
            for x in ("gate", "up", "down")
        ]
        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert.{x}_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_expert.{x}_proj.weight"
            for x in ("gate", "up", "down")
        ]

        return {"aoa_statements": aoa_statements}

    def _init_weights(self, module):
        if isinstance(module, Qwen3NextGatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.uniform_(0, 16).log_()


@register_base_model
class Qwen3NextModel(Qwen3NextPretrainedModel):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = nn.LayerList(
            [Qwen3NextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config)
        if config.rope_scaling:
            logger.warning_once("The rope_scaling is not implemented.")
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs,
    ) -> MoEModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3NextDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            # [bs * seq_len / mp_degree, num_head * head_dim]
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        causal_mask = None
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and paddle.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


class Qwen3NextForCausalLM(Qwen3NextPretrainedModel):
    enable_to_static_method = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.model = Qwen3NextModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(
        self,
        input_ids: Tensor = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[Tensor]] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ) -> MoECausalLMOutputWithPast:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3NextForCausalLM

        >>> model = Qwen3NextForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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


class Qwen3NextRMSNormPipe(RMSNormPipe):
    def _norm(self, x):
        self.eps = self.variance_epsilon
        return Qwen3NextRMSNorm._norm(self, x)

    def forward(self, args):
        hidden_states, _, _, _, _ = parse_args(args)
        return Qwen3NextRMSNorm.forward(self, hidden_states)


class Qwen3NextForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Qwen3NextConfig
    _rotary_emb_cls = Qwen3NextRotaryEmbedding
    _rms_norm_pipe_cls = Qwen3NextRMSNormPipe
    _decoder_layer_cls = Qwen3NextDecoderLayer
    _get_tensor_parallel_mappings = Qwen3NextModel._get_tensor_parallel_mappings
    _init_weights = Qwen3NextModel._init_weights
    _keep_in_fp32_modules = Qwen3NextModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Qwen3NextModel.transpose_weight_keys
    _gen_aoa_config = Qwen3NextPretrainedModel._gen_aoa_config
    _gen_inv_aoa_config = Qwen3NextPretrainedModel._gen_inv_aoa_config
