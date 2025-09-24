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

import copy
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.recompute.recompute import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...generation import GenerationMixin
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...utils.log import logger

# from ...utils.masking_utils import is_casual_mask
from ..activations import ACT2FN
from ..configuration_utils import PretrainedConfig

# from ..llama.modeling import get_use_casual_mask
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import Gemma3Config, Gemma3TextConfig
from .utils import ignore_causal_mask_sdpa, prepare_sliding_window_startend_row_indices


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", paddle.tensor(embed_scale), persistable=False)

    def forward(self, input_ids: paddle.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Layer):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.up_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.down_proj = GeneralLinear.create(
            self.intermediate_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3RMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(paddle.zeros(dim))

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["paddle.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["paddle.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`paddle.device`):
            The device to use for initialization of the inverse frequencies.
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
    inv_freq = 1.0 / (
        base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).to(device=device, dtype=paddle.float32) / dim)
    )
    return inv_freq, attention_factor


class Gemma3RotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # TODO: rope_type is default in `gemma-3-1b-it` but linear in `gemma-3-4b-it`
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    @paddle.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.place)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = paddle.matmul(inv_freq_expanded.float(), position_ids_expanded.float()).transpose(1, 2)
        emb = paddle.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.transpose(1, 2).astype(q.dtype), k_embed.transpose(1, 2).astype(q.dtype)


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
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not self.config.use_bidirectional_attention

        kv_hidden_size = config.num_key_value_heads * self.head_dim
        q_hidden_size = config.num_attention_heads * self.head_dim
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
        self.o_proj = GeneralLinear.create(
            q_hidden_size,
            config.hidden_size,
            has_bias=config.attention_bias,
            config=config,
            tp_plan="colwise",
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        position_embeddings: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[tuple[paddle.Tensor]]]:
        if self.config.sequence_parallel:
            max_sequence_length = self.config.max_sequence_length
            bsz = hidden_states.shape[0] * self.config.tensor_parallel_degree // max_sequence_length
            q_len = max_sequence_length
        else:
            bsz, q_len, _ = hidden_states.shape

        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)
        past_key_value = (key_states, value_states) if use_cache else None

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # if sequence_parallel is true, out shape are [q_len / n, bs, num_head * head_dim]
        # else their shape are [bs, q_len, num_head * head_dim], n is mp parallelism.
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Gemma3DecoderLayer(nn.Layer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_embeddings_global: paddle.Tensor,
        position_embeddings_local: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> tuple[paddle.FloatTensor, Optional[tuple[paddle.FloatTensor, paddle.FloatTensor]]]:
        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
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
        if use_cache:
            outputs += (present_key_value,)
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Gemma3PreTrainedModel(PretrainedModel):
    config_class = Gemma3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Gemma3DecoderLayer",
        # "SiglipVisionEmbeddings",
        # "SiglipEncoderLayer",
        # "SiglipMultiheadAttentionPoolingHead",
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Gemma3DecoderLayer,
        "attentions": Gemma3Attention,
    }

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

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gemma3MultiModalProjector):
            module.mm_input_projection_weight.data.zero_()


class Gemma3TextModel(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.sequence_parallel = config.sequence_parallel

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.LayerList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        if config.sequence_parallel:
            self.norm.enable_sequence_parallel()

        if self._supports_sdpa:
            self.config._attn_implementation = "sdpa"

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: Optional[paddle.Tensor],
        past_key_value: paddle.Tensor,
        use_cache: bool,
        output_attentions: bool,
        position_embeddings_global: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        position_embeddings_local: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices=None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states=hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        return hidden_states

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor]] = None,
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

        # Compute cache length
        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        else:
            cache_length = past_key_values[0][0].shape[-2]
            seq_length_with_past += cache_length

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        # Prepare causal mask mapping
        if attn_mask_startend_row_indices is not None:
            attention_mask = None
            causal_mask_mapping = {}
            attn_mask_startend_row_indices_mapping = {}
            causal_mask_mapping["full_attention"] = None
            causal_mask_mapping["sliding_attention"] = None

            attn_mask_startend_row_indices_mapping["full_attention"] = attn_mask_startend_row_indices

            if self.config.sliding_window is not None:
                attn_mask_startend_row_indices_mapping[
                    "sliding_attention"
                ] = prepare_sliding_window_startend_row_indices(
                    attn_mask_startend_row_indices, window_size=self.config.sliding_window
                )
            else:
                attn_mask_startend_row_indices_mapping["sliding_attention"] = None
        else:
            attention_mask = (
                paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
                if attention_mask is None
                else attention_mask
            )
            causal_mask_mapping = {}
            attn_mask_startend_row_indices_mapping = {}
            attn_mask_startend_row_indices_mapping["full_attention"] = None
            attn_mask_startend_row_indices_mapping["sliding_attention"] = None

            mask_kwargs = {
                "attention_mask": attention_mask,
                "input_shape": (batch_size, seq_length),
                "past_key_values_length": cache_length,
                "dtype": inputs_embeds.dtype,
            }

            if self.config._attn_implementation == "sdpa":
                # There is no return of None in `_prepare_decoder_attention_mask`
                should_ignore = ignore_causal_mask_sdpa(**mask_kwargs) and not self.training
            full_attention_mask = None if should_ignore else self._prepare_decoder_attention_mask(**mask_kwargs)

            if self.config.sliding_window is not None:
                mask_kwargs["sliding_window_size"] = self.config.sliding_window
                if self.config._attn_implementation == "sdpa":
                    should_ignore_sliding_window = ignore_causal_mask_sdpa(**mask_kwargs) and not self.training
                    sliding_attention_mask = (
                        None if should_ignore_sliding_window else self._prepare_decoder_attention_mask(**mask_kwargs)
                    )
            else:
                sliding_attention_mask = None

            causal_mask_mapping = {
                "full_attention": full_attention_mask,
                "sliding_attention": sliding_attention_mask,
            }

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64").expand((batch_size, seq_length))

        # Compute position embeddings
        position_embeddings_global = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_local = self.rotary_emb_local(inputs_embeds, position_ids)

        # decoder layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.config.recompute and self.config.recompute_granularity == "full" and has_gradient:
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_mapping[
                        decoder_layer.attention_type
                    ],
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        # Final RMS Norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Return outputs
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3TextConfig
    # TODO: base_model_prefix should be same with submodel variable name
    # base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: Optional[paddle.LongTensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[Tuple[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices=None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
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
        return_dict = True
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

        if not return_dict:
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
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


class Gemma3MultiModalProjector(nn.Layer):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            paddle.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: paddle.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = paddle.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel):
    """
    Gemma3TextForSequenceClassification is a text-only sequence classification model that works with Gemma3TextConfig.
    It uses the generic sequence classification implementation for efficiency and consistency.
    """

    config_class = Gemma3TextConfig
    # TODO: implement the class refer to deepseekv2 & QWen3


__all__ = [
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
]
