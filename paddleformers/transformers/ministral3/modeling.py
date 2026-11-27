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

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.mlp import MLP as Ministral3MLP
from ...nn.norm import Norm as GeneralNorm
from ...nn.pp_model import EmbeddingPipe, GeneralModelForCausalLMPipe, parse_args
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import Ministral3Config


def _create_attention_mask(config, **kwargs):
    mask_fn = (
        create_sliding_window_causal_mask_and_row_indices
        if config.sliding_window is not None
        else create_causal_mask_and_row_indices
    )
    return mask_fn(config=config, **kwargs)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


def get_llama_4_attn_scale(position_ids: Tensor, beta: float, max_position_embeddings: int) -> Tensor:
    positions = position_ids.astype("float32")
    scaling = 1.0 + beta * paddle.log(1.0 + paddle.floor(positions / max_position_embeddings))
    return scaling[:, None, :, None]


class Ministral3RotaryEmbedding(nn.Layer):
    def __init__(self, config: Ministral3Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_parameters = config.rope_parameters
        self.rope_type = self.rope_parameters.get("rope_type", self.rope_parameters.get("type", "default"))

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistable=False)

    @staticmethod
    def compute_default_rope_parameters(config: Ministral3Config, seq_len: Optional[int] = None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype("float32") / dim))
        return inv_freq, 1.0

    @dynamic_rope_update
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].astype("float32").expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].astype("float32")
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose([0, 2, 1])
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Ministral3Attention(nn.Layer):
    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        q_hidden_size = config.num_attention_heads * self.head_dim
        kv_hidden_size = config.num_key_value_heads * self.head_dim

        if config.tensor_model_parallel_size > 1:
            if self.num_heads % config.tensor_model_parallel_size:
                raise ValueError("num_attention_heads must be divisible by tensor_model_parallel_size")
            if self.num_key_value_heads % config.tensor_model_parallel_size:
                raise ValueError("num_key_value_heads must be divisible by tensor_model_parallel_size")
            self.num_heads = self.num_heads // config.tensor_model_parallel_size
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        self.q_proj = GeneralLinear.create(
            config.hidden_size, q_hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.k_proj = GeneralLinear.create(
            config.hidden_size, kv_hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.v_proj = GeneralLinear.create(
            config.hidden_size, kv_hidden_size, has_bias=False, config=config, tp_plan="colwise"
        )
        self.o_proj = GeneralLinear.create(
            q_hidden_size, config.hidden_size, has_bias=False, config=config, tp_plan="rowwise"
        )

        self.llama_4_scaling_beta = config.rope_parameters.get("llama_4_scaling_beta")
        self.original_max_position_embeddings = config.rope_parameters.get(
            "original_max_position_embeddings", config.max_position_embeddings
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.config.sequence_parallel:
            q_len = position_embeddings[0].shape[-2]
            bsz = hidden_states.shape[0] * self.config.tensor_model_parallel_size // q_len
        else:
            bsz, q_len = hidden_states.shape[:2]

        query_states = (
            self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )
        key_states = (
            self.k_proj(hidden_states)
            .reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        value_states = (
            self.v_proj(hidden_states)
            .reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.llama_4_scaling_beta is not None and position_ids is not None:
            query_states = query_states * get_llama_4_attn_scale(
                position_ids, self.llama_4_scaling_beta, self.original_max_position_embeddings
            ).astype(query_states.dtype)

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

        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Ministral3DecoderLayer(nn.Layer):
    def __init__(self, config: Ministral3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Ministral3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Ministral3MLP(config)
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states


def _validate_execution_config(config: Ministral3Config):
    if config.sequence_parallel and config.tensor_model_parallel_size <= 1:
        raise ValueError("sequence_parallel requires tensor_model_parallel_size > 1")
    if config.apply_rope_fusion:
        raise ValueError("Ministral3 does not support apply_rope_fusion; disable it to use the unfused RoPE path")


class Ministral3PretrainedModel(PretrainedModel):
    config_class = Ministral3Config
    base_model_class = None
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: Ministral3Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            f"model.norm.weight -> {model_prefix}norm.weight",
            f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
            f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"model.layers.$LAYER_ID.mlp.{proj_name}.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: Ministral3Config):
        model_prefix = cls.base_model_prefix + "." if cls != cls.base_model_class else ""
        aoa_statements = [
            f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
        ]
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.self_attn.{proj_name}.weight^T -> model.layers.$LAYER_ID.self_attn.{proj_name}.weight"
                for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ]
        )
        aoa_statements.extend(
            [
                f"{model_prefix}layers.$LAYER_ID.mlp.{proj_name}.weight^T -> model.layers.$LAYER_ID.mlp.{proj_name}.weight"
                for proj_name in ["gate_proj", "up_proj", "down_proj"]
            ]
        )
        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")
        return {"aoa_statements": aoa_statements}

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


@register_base_model
class Ministral3Model(Ministral3PretrainedModel):
    def __init__(self, config: Ministral3Config):
        _validate_execution_config(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = GeneralEmbedding.create(
            config=config,
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.LayerList(
            [Ministral3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = Ministral3RotaryEmbedding(config=config)

    @staticmethod
    def _recompute_layer(
        layer,
        hidden_states,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        position_ids=None,
        position_embeddings=None,
        past_key_values=None,
        use_cache=False,
    ):
        # Keep Layer hooks without sharing the PP wrapper's unbound forward signature cache.
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_attentions and self.config._attn_implementation != "eager":
            raise ValueError("output_attentions is only supported by the eager attention implementation")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            bsz, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            bsz, seq_length = inputs_embeds.shape[:2]
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        recompute_enabled = (
            self.training
            and paddle.is_grad_enabled()
            and self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
        )
        if recompute_enabled and output_attentions:
            raise ValueError("output_attentions is not supported together with activation recompute")
        if recompute_enabled and past_key_values is not None:
            raise ValueError("past_key_values must be None during recompute training")
        if recompute_enabled and use_cache:
            logger.warning_once("use_cache=True is incompatible with recompute; disabling cache for this forward")
            use_cache = False

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(cache_length, seq_length + cache_length, dtype=paddle.int64).unsqueeze(0).tile([bsz, 1])
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": cache_length,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = _create_attention_mask(**mask_kwargs)
        if cache_length > 0 and seq_length > 1 and causal_mask is None and attn_mask_startend_row_indices is None:
            causal_mask = self._prepare_decoder_attention_mask(
                attention_mask=paddle.ones([bsz, cache_length + seq_length], dtype="bool"),
                input_shape=(bsz, seq_length),
                past_key_values_length=cache_length,
                dtype=inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if recompute_enabled:
                layer_outputs = recompute(
                    self._recompute_layer,
                    decoder_layer,
                    hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    use_reentrant=self.config.recompute_use_reentrant and not hidden_states.stop_gradient,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                hidden_states, attn_weights = layer_outputs
                all_self_attns += (attn_weights,)
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class Ministral3ForCausalLM(Ministral3PretrainedModel):
    base_model_class = Ministral3Model
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Ministral3Config):
        super().__init__(config)
        self.model = Ministral3Model(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[Tensor] = None,
        attn_mask_startend_row_indices: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and attention_mask.dtype != paddle.bool:
            attention_mask = attention_mask.astype(paddle.bool)

        if attn_mask_startend_row_indices is not None and attention_mask is not None:
            logger.warning(
                "You have provided both attn_mask_startend_row_indices and attention_mask. "
                "The attn_mask_startend_row_indices will be used."
            )
            attention_mask = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        logits = self.lm_head(outputs.last_hidden_state)

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


class Ministral3DecoderLayerPipe(Ministral3DecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, position_embeddings, _ = parse_args(args)

        max_seq_len = hidden_states.shape[1]
        if self.config.sequence_parallel:
            max_seq_len = hidden_states.shape[0] * self.config.tensor_model_parallel_size
        if attention_mask is None:
            tgt_mask = None
            attn_mask_startend_row_indices = None
        elif attention_mask.dtype == paddle.int32:
            tgt_mask = None
            attn_mask_startend_row_indices = attention_mask
        else:
            tgt_mask = attention_mask[:, :, :max_seq_len, :max_seq_len]
            attn_mask_startend_row_indices = None
            assert len(tgt_mask.shape) == 4, f"Attention mask should be 4D tensor, but got {tgt_mask.shape}."

        # The Llama-4 attention scaling is applied from the per-token position ids, which the
        # generic decoder-layer pipe does not forward.
        position_ids_decoder = position_ids[:, :max_seq_len] if position_ids is not None else None

        if position_embeddings is not None:
            position_embeddings = position_embeddings[..., :max_seq_len, :]
            tuple_position_embeddings = (position_embeddings[0], position_embeddings[1])
        else:
            tuple_position_embeddings = None

        has_gradient = not hidden_states.stop_gradient
        if (
            self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and self.config.recompute_num_layers == 1
            and has_gradient
        ):
            hidden_states = recompute(
                super().forward,
                hidden_states,
                attention_mask=tgt_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids_decoder,
                position_embeddings=tuple_position_embeddings,
                use_reentrant=self.config.recompute_use_reentrant,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                attention_mask=tgt_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids_decoder,
                position_embeddings=tuple_position_embeddings,
            )

        if isinstance(hidden_states, paddle.Tensor):
            ret = (hidden_states,)
        else:
            ret = hidden_states
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if position_embeddings is not None:
            ret += (position_embeddings.clone(),)
        if len(ret) == 1:
            (ret,) = ret
        return ret


class Ministral3EmbeddingPipe(EmbeddingPipe):
    def forward(self, args):
        input_ids, mask, _, _, _ = parse_args(args, is_embed=True)
        outputs = super().forward(args)
        hidden_states, _, position_ids, position_embeddings, offset = parse_args(outputs)
        row_indices = mask if mask is not None and mask.ndim >= 3 and mask.dtype == paddle.int32 else None
        attention_mask = mask.astype("bool") if mask is not None and row_indices is None else None
        attention_mask, row_indices = _create_attention_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            batch_size=input_ids.shape[0],
            seq_length=input_ids.shape[1],
            cache_length=0,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=row_indices,
            prepare_decoder_attention_mask=Ministral3PretrainedModel._prepare_decoder_attention_mask,
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
        return outputs[0] if len(outputs) == 1 else outputs


class Ministral3CriterionLayerPipe(CriterionLayer):
    def __init__(self, config):
        super().__init__(config, return_tuple=False)

    def forward(self, logits, labels):
        loss_mask = None
        if isinstance(labels, tuple) and "sft" in self.loss_type:
            labels, loss_mask = labels
        return super().forward(logits, labels, loss_mask=loss_mask)


class Ministral3ForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = Ministral3Config
    base_model_class = Ministral3Model
    base_model_prefix = "model"
    _decoder_layer_cls = Ministral3DecoderLayer
    _decoder_layer_pipe_cls = Ministral3DecoderLayerPipe
    _embedding_pipe_cls = Ministral3EmbeddingPipe
    _criterion_pipe_cls = Ministral3CriterionLayerPipe
    _rotary_emb_cls = Ministral3RotaryEmbedding
    _init_weights = Ministral3PretrainedModel._init_weights
    _keep_in_fp32_modules = Ministral3PretrainedModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = Ministral3PretrainedModel.transpose_weight_keys
    _gen_aoa_config = Ministral3ForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = Ministral3ForCausalLM._gen_inv_aoa_config

    def __init__(self, config):
        _validate_execution_config(config)
        super().__init__(config)

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        # The shared-weight path hardcodes EmbeddingPipe instead of using _embedding_pipe_cls.
        if name_prefix == "model" and layer_desc.layer_func is EmbeddingPipe:
            layer_desc.layer_func = Ministral3EmbeddingPipe
        super().add_sequential_layer(layer_desc, name_prefix)


__all__ = [
    "Ministral3PretrainedModel",
    "Ministral3Model",
    "Ministral3ForCausalLM",
    "Ministral3ForCausalLMPipe",
    "Ministral3RotaryEmbedding",
    "Ministral3DecoderLayer",
    "Ministral3Attention",
]
