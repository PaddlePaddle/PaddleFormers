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

"""Paddle implementation of the LFM2 text backbone."""

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead
from ...nn.mlp import MLP
from ...nn.norm import Norm as GeneralNorm
from ...utils.masking_utils import _gen_from_sparse_attn_mask_indices
from ..cache_utils import DynamicCache
from ..llama.modeling import LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Lfm2Config


class Lfm2HybridConvCache(DynamicCache):
    """Standard dynamic KV cache plus bounded causal-convolution history."""

    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.conv_states = [None] * config.num_hidden_layers
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        # Also works for all-convolution models and when layer zero is a conv.
        return self.seen_tokens

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        for index, state in enumerate(self.conv_states):
            if state is not None:
                self.conv_states[index] = paddle.index_select(state, beam_idx, axis=0)

    def reset(self):
        self.__init__(self.config)


class Lfm2MLP(MLP):
    def __init__(self, config):
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                multiple = config.block_multiple_of
                intermediate_size = multiple * ((intermediate_size + multiple - 1) // multiple)
        super().__init__(config, intermediate_size=intermediate_size, has_bias=False)


class Lfm2Attention(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.is_causal = True
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        degree = config.tensor_model_parallel_size
        if self.num_heads % degree or self.num_key_value_heads % degree:
            raise ValueError("LFM2 attention and KV heads must be divisible by tensor_model_parallel_size")
        self.q_proj = GeneralLinear.create(
            config.hidden_size, self.num_heads * self.head_dim, has_bias=False, config=config, tp_plan="colwise"
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.out_proj = GeneralLinear.create(
            self.num_heads * self.head_dim, config.hidden_size, has_bias=False, config=config, tp_plan="rowwise"
        )
        self.num_heads //= degree
        self.num_key_value_heads //= degree
        self.q_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=self.head_dim, has_bias=False, norm_eps=config.norm_eps
        )
        self.k_layernorm = GeneralNorm.create(
            config=config, norm_type="rms_norm", hidden_size=self.head_dim, has_bias=False, norm_eps=config.norm_eps
        )
        if degree > 1:
            from paddle.distributed import fleet

            group = fleet.get_hybrid_communicate_group().get_model_parallel_group()

            def sum_head_gradients(gradient):
                # Q/K norm weights are shared across sharded heads, even without SP.
                # Reduce the incoming gradient, not the accumulated .grad buffer.
                gradient = gradient.clone()
                paddle.distributed.all_reduce(gradient, group=group)
                return gradient

            self.q_layernorm.weight.register_hook(sum_head_gradients)
            self.k_layernorm.weight.register_hook(sum_head_gradients)

    def forward(
        self, hidden_states, cos, sin, attention_mask=None, attn_mask_startend_row_indices=None, past_key_values=None
    ):
        sequence_length = cos.shape[1]
        batch_size = (
            hidden_states.shape[0] * self.config.tensor_model_parallel_size // sequence_length
            if self.config.sequence_parallel
            else hidden_states.shape[0]
        )
        query = self.q_proj(hidden_states).reshape([batch_size, sequence_length, self.num_heads, self.head_dim])
        key = self.k_proj(hidden_states).reshape(
            [batch_size, sequence_length, self.num_key_value_heads, self.head_dim]
        )
        value = self.v_proj(hidden_states).reshape(
            [batch_size, sequence_length, self.num_key_value_heads, self.head_dim]
        )
        query = self.q_layernorm(query).transpose([0, 2, 1, 3])
        key = self.k_layernorm(key).transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        output, _ = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            scaling=self.scaling,
            dropout=0.0,
        )
        if self.config.sequence_parallel:
            output = output.reshape([-1, output.shape[-1]])
        return self.out_proj(output)


class Lfm2ShortConv(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.kernel_size = config.conv_L_cache
        degree = config.tensor_model_parallel_size
        if config.hidden_size % degree:
            raise ValueError("ShortConv hidden_size must be divisible by tensor_model_parallel_size")
        channels = config.hidden_size // degree
        # Separate projections preserve corresponding B/C/value channels on each TP rank.
        # Splitting the old fused [B, C, X] weight contiguously would mix the three gates.
        self.b_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=config.conv_bias, config=config, tp_plan="colwise"
        )
        self.c_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=config.conv_bias, config=config, tp_plan="colwise"
        )
        self.x_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=config.conv_bias, config=config, tp_plan="colwise"
        )
        self.conv = nn.Conv1D(
            channels,
            channels,
            self.kernel_size,
            groups=channels,
            padding=self.kernel_size - 1,
            bias_attr=config.conv_bias,
        )
        for parameter in (self.conv.weight, self.conv.bias):
            if parameter is not None:
                parameter.is_distributed = degree > 1
                if degree > 1:
                    parameter.split_axis = 0
        self.out_proj = GeneralLinear.create(
            config.hidden_size, config.hidden_size, has_bias=config.conv_bias, config=config, tp_plan="rowwise"
        )

    def forward(
        self, hidden_states, attention_mask=None, packed_mask=None, past_key_values=None, sequence_length=None
    ):
        sequence_length = hidden_states.shape[1] if sequence_length is None else sequence_length
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).astype(hidden_states.dtype)
            if self.config.sequence_parallel:
                mask = ScatterOp.apply(mask.reshape([-1, 1]))
            hidden_states = hidden_states * mask
        projected = [
            projection(hidden_states).reshape([-1, sequence_length, self.conv.weight.shape[0]]).transpose([0, 2, 1])
            for projection in (self.b_proj, self.c_proj, self.x_proj)
        ]
        b_gate, c_gate, values = projected
        values = b_gate * values
        values_dtype = values.dtype
        if past_key_values is not None:
            history = past_key_values.conv_states[self.layer_idx]
            history = (
                F.pad(values, [self.kernel_size - 1, 0])
                if history is None
                else paddle.concat([history, values], axis=-1)
            )
            # Keep K-1 values regardless of prompt length; supports both one-token
            # decoding and a multi-token continuation without rerunning the prefix.
            past_key_values.conv_states[self.layer_idx] = (
                history[:, :, -(self.kernel_size - 1) :] if self.kernel_size > 1 else history[:, :, :0]
            )
            values = F.conv1d(
                history.astype("float32"),
                self.conv.weight.astype("float32"),
                None if self.conv.bias is None else self.conv.bias.astype("float32"),
                groups=values.shape[1],
            ).astype(values_dtype)
        elif packed_mask is None:
            values = F.conv1d(
                values.astype("float32"),
                self.conv.weight.astype("float32"),
                None if self.conv.bias is None else self.conv.bias.astype("float32"),
                padding=self.kernel_size - 1,
                groups=values.shape[1],
            )[:, :, :sequence_length].astype(values_dtype)
        else:
            # A causal convolution must obey the same document boundaries as
            # attention. Mask each of its K lag contributions, including bias
            # exactly once; do not convolve across concatenated documents.
            length = values.shape[-1]
            result = paddle.zeros_like(values, dtype="float32")
            allowed = packed_mask if packed_mask.dtype == paddle.bool else packed_mask >= 0
            for lag in range(min(self.kernel_size, length)):
                valid = paddle.diagonal(allowed[:, 0], offset=-lag, axis1=-2, axis2=-1).unsqueeze(1)
                shifted = values[:, :, : length - lag].astype("float32") * valid.astype("float32")
                shifted = F.pad(shifted, [lag, 0])
                weight = self.conv.weight[:, 0, self.kernel_size - 1 - lag].astype("float32").reshape([1, -1, 1])
                result = result + shifted * weight
            if self.conv.bias is not None:
                result = result + self.conv.bias.astype("float32").reshape([1, -1, 1])
            values = result.astype(values_dtype)
        output = (c_gate * values).transpose([0, 2, 1]).contiguous()
        if self.config.sequence_parallel:
            output = output.reshape([-1, output.shape[-1]])
        return self.out_proj(output)


class Lfm2DecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.ffn_norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        causal_mask=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        packed_mask=None,
        past_key_values=None,
    ):
        residual = hidden_states
        normalized = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            hidden_states = self.self_attn(
                normalized, cos, sin, causal_mask, attn_mask_startend_row_indices, past_key_values
            )
        else:
            hidden_states = self.conv(
                normalized, attention_mask, packed_mask, past_key_values, sequence_length=cos.shape[1]
            )
        hidden_states = residual + hidden_states
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm2PreTrainedModel(PretrainedModel):
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    config_class = Lfm2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    @staticmethod
    def _paddle_weight_name(name):
        for old, new in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            name = name.replace(f"feed_forward.{old}.", f"feed_forward.{new}.")
        return name

    @classmethod
    def _weight_statements(cls, weights, prefix, inverse=False):
        statements = []
        for name, transpose in weights:
            source, target = "model." + name, prefix + cls._paddle_weight_name(name)
            if ".conv.in_proj." in name:
                targets = [target.replace(".conv.in_proj.", f".conv.{gate}_proj.") for gate in ("b", "c", "x")]
                if inverse:
                    left = ", ".join(item + ("^T" if transpose else "") for item in targets)
                    statements.append(f"{left} -> {source}, axis=0")
                else:
                    statements.append(
                        f"{source}{'^T' if transpose else ''} -> {', '.join(targets)}, axis={1 if transpose else 0}"
                    )
            else:
                if inverse:
                    source, target = target, source
                statements.append(f"{source}{'^T' if transpose else ''} -> {target}")
        return statements

    @classmethod
    def _checkpoint_weights(cls, config):
        weights = [("embed_tokens.weight", False), ("embedding_norm.weight", False)]
        for index, layer_type in enumerate(config.layer_types):
            prefix = f"layers.{index}."
            weights.extend((prefix + name + ".weight", False) for name in ("operator_norm", "ffn_norm"))
            weights.extend((prefix + "feed_forward." + name + ".weight", True) for name in ("w1", "w2", "w3"))
            if layer_type == "full_attention":
                weights.extend(
                    (prefix + "self_attn." + name + ".weight", True)
                    for name in ("q_proj", "k_proj", "v_proj", "out_proj")
                )
                weights.extend(
                    (prefix + "self_attn." + name + ".weight", False) for name in ("q_layernorm", "k_layernorm")
                )
            else:
                for name in ("in_proj", "out_proj", "conv"):
                    weights.append((prefix + "conv." + name + ".weight", name != "conv"))
                    if config.conv_bias:
                        weights.append((prefix + "conv." + name + ".bias", False))
        return weights

    @classmethod
    def _gen_aoa_config(cls, config):
        prefix = "" if cls == cls.base_model_class else "model."
        statements = cls._weight_statements(cls._checkpoint_weights(config), prefix)
        if prefix and not config.tie_word_embeddings:
            statements.append("lm_head.weight -> lm_head.weight")
        elif prefix:
            statements.append("model.embed_tokens.weight -> lm_head.weight")
        return {"aoa_statements": statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        prefix = "" if cls == cls.base_model_class else "model."
        statements = cls._weight_statements(cls._checkpoint_weights(config), prefix, inverse=True)
        if prefix and not config.tie_word_embeddings:
            statements.append("lm_head.weight -> lm_head.weight")
        elif prefix:
            statements.append("lm_head.weight -> _")
        return {"aoa_statements": statements}


@register_base_model
class Lfm2Model(Lfm2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.LayerList([Lfm2DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.embedding_norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape
        if self.config.sequence_parallel and use_cache:
            raise ValueError("LFM2 sequence-parallel training requires use_cache=False")
        if self.config.sequence_parallel and sequence_length != self.config.max_sequence_length:
            raise ValueError("LFM2 sequence-parallel input length must match max_sequence_length for LMHead gathering")
        if self.config.sequence_parallel and batch_size * sequence_length % self.config.tensor_model_parallel_size:
            raise ValueError("Batch times sequence length must be divisible by the sequence-parallel degree")
        if use_cache and self.training and getattr(self.config, "recompute_granularity", None) == "full":
            raise ValueError("LFM2 recompute training requires use_cache=False to avoid updating cache twice")
        if use_cache and (
            attn_mask_startend_row_indices is not None or (attention_mask is not None and attention_mask.ndim == 4)
        ):
            raise ValueError("LFM2 cached decoding requires a 2D padding mask; packed training uses use_cache=False")
        if use_cache:
            if past_key_values is None:
                past_key_values = Lfm2HybridConvCache(self.config)
            elif not isinstance(past_key_values, Lfm2HybridConvCache):
                raise TypeError("LFM2 requires Lfm2HybridConvCache for both KV and convolution history")
        else:
            past_key_values = None
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                paddle.arange(cache_length, cache_length + sequence_length, dtype="int64")
                .unsqueeze(0)
                .expand([batch_size, -1])
            )
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        packed_mask = None
        if attn_mask_startend_row_indices is not None:
            indices = attn_mask_startend_row_indices
            if indices.ndim == 3:
                indices = indices.unsqueeze(-1)
            packed_mask = _gen_from_sparse_attn_mask_indices(indices, inputs_embeds.dtype, is_causal=True)
        elif attention_mask is not None and attention_mask.ndim == 4:
            packed_mask = attention_mask
        causal_mask, row_indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=inputs_embeds,
            batch_size=batch_size,
            seq_length=sequence_length,
            cache_length=cache_length,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            prepare_decoder_attention_mask=self._prepare_decoder_attention_mask,
        )
        padding_mask = (
            attention_mask[:, -sequence_length:] if attention_mask is not None and attention_mask.ndim == 2 else None
        )
        hidden_states = inputs_embeds
        if self.config.sequence_parallel:
            hidden_states = ScatterOp.apply(hidden_states.reshape([-1, hidden_states.shape[-1]]))

        def create_custom_forward(module):
            def custom_forward(layer_hidden_states):
                return module(layer_hidden_states, cos, sin, causal_mask, padding_mask, row_indices, packed_mask)

            return custom_forward

        for layer in self.layers:
            if (
                self.training
                and not hidden_states.stop_gradient
                and getattr(self.config, "recompute_granularity", None) == "full"
            ):
                hidden_states = recompute(
                    create_custom_forward(layer),
                    hidden_states,
                    use_reentrant=getattr(self.config, "recompute_use_reentrant", True),
                )
            else:
                hidden_states = layer(
                    hidden_states, cos, sin, causal_mask, padding_mask, row_indices, packed_mask, past_key_values
                )
        hidden_states = self.embedding_norm(hidden_states)
        if past_key_values is not None:
            past_key_values.seen_tokens += sequence_length
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Lfm2ForCausalLM(Lfm2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Lfm2Model(config)
        self.lm_head = LMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, labels=None, **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = self.criterion(logits, labels)[0] if labels is not None else None
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


def convert_lfm2_legacy_state_dict(state_dict):
    """Migrate historical Paddle w1/w2/w3 and fused ShortConv keys without changing values."""
    converted = {}
    for name, value in state_dict.items():
        target = Lfm2PreTrainedModel._paddle_weight_name(name)
        if ".conv.in_proj." in target:
            pieces = value.chunk(3, axis=1 if name.endswith(".weight") else 0)
            items = [
                (target.replace(".conv.in_proj.", f".conv.{gate}_proj."), piece)
                for gate, piece in zip(("b", "c", "x"), pieces)
            ]
        else:
            items = [(target, value)]
        for key, tensor in items:
            if key in converted:
                raise ValueError(f"Checkpoint contains conflicting old/new LFM2 parameter names: {key}")
            converted[key] = tensor
    return converted


__all__ = [
    "Lfm2ForCausalLM",
    "Lfm2Model",
    "Lfm2PreTrainedModel",
    "Lfm2HybridConvCache",
    "convert_lfm2_legacy_state_dict",
]
