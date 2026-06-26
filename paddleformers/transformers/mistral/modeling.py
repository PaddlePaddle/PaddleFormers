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

from typing import Callable, Optional, Tuple, Union, cast

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.activation import ACT2FN
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
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .configuration import MistralConfig


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies rotary positional embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    original_dtype = q.dtype
    q_embed = (q.astype("float32") * cos) + (rotate_half(q).astype("float32") * sin)
    k_embed = (k.astype("float32") * cos) + (rotate_half(k).astype("float32") * sin)
    return q_embed.astype(original_dtype), k_embed.astype(original_dtype)


class MistralMLP(nn.Layer):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=config.mlp_bias,
            config=config,
            tp_plan="colwise",
        )
        self.up_proj = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size,
            has_bias=config.mlp_bias,
            config=config,
            tp_plan="colwise",
        )
        self.down_proj = GeneralLinear.create(
            self.intermediate_size,
            self.hidden_size,
            has_bias=config.mlp_bias,
            config=config,
            tp_plan="rowwise",
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MistralAttention(nn.Layer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        assert config.num_attention_heads % config.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads. "
            f"Found {config.num_attention_heads} and {config.num_key_value_heads}."
        )

        if config.tensor_model_parallel_size > 1:
            assert (
                self.num_heads % config.tensor_model_parallel_size == 0
            ), f"num_heads: {self.num_heads}, tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            self.num_heads = self.num_heads // config.tensor_model_parallel_size

            assert self.num_key_value_heads % config.tensor_model_parallel_size == 0, (
                f"num_key_value_heads: {self.num_key_value_heads}, "
                f"tensor_model_parallel_size: {config.tensor_model_parallel_size}"
            )
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_model_parallel_size

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        q_hidden_size = self.head_dim * config.num_attention_heads
        kv_hidden_size = self.head_dim * config.num_key_value_heads

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
            tp_plan="rowwise",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor, list[paddle.Tensor] | None]:
        if self.config.sequence_parallel:
            seq_len = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_len
        else:
            batch_size, seq_len = hidden_states.shape[:2]

        q_shape = (batch_size, seq_len, -1, self.head_dim)
        kv_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).reshape(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if self.config._attn_implementation != "sdpa":
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


class MistralDecoderLayer(nn.Layer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.post_attention_layernorm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if len(outputs) == 1 and isinstance(outputs, tuple):
            outputs = outputs[0]

        return outputs


class MistralRotaryEmbedding(nn.Layer):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = config.head_dim

        self.rope_type = "default"
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self.rope_type = config.rope_parameters.get("rope_type", "default")

        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[MistralConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MistralPretrainedModel(PretrainedModel):
    config_class = MistralConfig
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

    @classmethod
    def _gen_aoa_config(cls, config: MistralConfig):
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

        if "CausalLM" not in cls.__name__:
            aoa_statements = [s for s in aoa_statements if "lm_head.weight" not in s]

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: MistralConfig):
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

        if cls != cls.base_model_class and not config.tie_word_embeddings:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        if "CausalLM" not in cls.__name__:
            aoa_statements = [s for s in aoa_statements if "lm_head.weight" not in s]

        return {"aoa_statements": aoa_statements}


@register_base_model
class MistralModel(MistralPretrainedModel):
    def __init__(self, config: MistralConfig):
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
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = MistralRotaryEmbedding(config=config)
        self.has_sliding_layers = getattr(self.config, "sliding_window", None) is not None and (
            "sliding_attention" in getattr(self.config, "layer_types", [])
        )

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = False,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        inputs_embeds = cast(paddle.Tensor, inputs_embeds)
        batch_size, seq_length, _ = inputs_embeds.shape

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        cache_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(cache_length, seq_length + cache_length, dtype=paddle.int64)
                .unsqueeze(0)
                .tile((batch_size, 1))
            )

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
        full_mask, full_indices = create_causal_mask_and_row_indices(**mask_kwargs)

        causal_mask_mapping = {"full_attention": full_mask}
        row_indices_mapping = {"full_attention": full_indices}
        if self.has_sliding_layers:
            (
                causal_mask_mapping["sliding_attention"],
                row_indices_mapping["sliding_attention"],
            ) = create_sliding_window_causal_mask_and_row_indices(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        all_hidden_states = [] if output_hidden_states else None

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            attention_type = getattr(decoder_layer, "attention_type", "full_attention")
            layer_mask = causal_mask_mapping[attention_type]
            layer_row_indices = row_indices_mapping[attention_type]

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    layer_mask,
                    layer_row_indices,
                    position_ids,
                    position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    attn_mask_startend_row_indices=layer_row_indices,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple | list) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        if not return_dict:
            outputs = [hidden_states]
            if output_hidden_states:
                outputs.append(all_hidden_states)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        attn_mask_startend_row_indices: paddle.Tensor | None,
        position_ids: paddle.Tensor,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor],
        past_key_values: Cache | None,
        use_cache: bool,
    ):
        cos, sin = position_embeddings
        position_embeddings_safe = (cos.clone(), sin.clone())

        hidden_states = recompute(
            layer_module,
            hidden_states,
            attention_mask,
            attn_mask_startend_row_indices,
            position_ids,
            position_embeddings_safe,
            past_key_values,
            use_cache,
        )
        return hidden_states


class MistralForCausalLM(MistralPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        self.model = MistralModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
        output_hidden_states: bool | None = False,
        return_dict: bool = False,
        **kwargs,
    ):
        if kwargs.get("attn_mask_start_row_indices", None) is not None and attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = kwargs.pop("attn_mask_start_row_indices")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            output_hidden_states=output_hidden_states,
            return_dict=True,
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


class MistralForSequenceClassification(MistralPretrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        self.score = GeneralLinear.create(config.hidden_size, self.num_labels, has_bias=False, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
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
            sequence_lengths = paddle.full([batch_size], logits.shape[1] - 1, dtype="int64")
        else:
            if input_ids is not None:
                sequence_lengths = paddle.eq(input_ids, self.config.pad_token_id).astype("int32").argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = paddle.full([batch_size], logits.shape[1] - 1, dtype="int64")

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


class MistralForTokenClassification(MistralPretrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
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
        input_ids: paddle.Tensor | None = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        sequence_output = self.dropout(outputs[0])
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


class MistralForQuestionAnswering(MistralPretrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.model = MistralModel(config)
        self.qa_outputs = GeneralLinear.create(config.hidden_size, 2, has_bias=True, linear_type="default")

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        start_positions: Optional[paddle.Tensor] = None,
        end_positions: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        logits = self.qa_outputs(outputs[0])
        start_logits, end_logits = paddle.unstack(logits, axis=-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            seq_len = start_logits.shape[-1]
            start_positions = start_positions.clip(0, seq_len - 1)
            end_positions = end_positions.clip(0, seq_len - 1)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MistralForCausalLMPipe(GeneralModelForCausalLMPipe):
    config_class = MistralConfig
    _decoder_layer_cls = MistralDecoderLayer
    _get_tensor_parallel_mappings = MistralModel._get_tensor_parallel_mappings
    _init_weights = MistralModel._init_weights
    _keep_in_fp32_modules = MistralModel._keep_in_fp32_modules
    _tied_weights_keys = ["lm_head.weight"]
    transpose_weight_keys = MistralModel.transpose_weight_keys
    _gen_aoa_config = MistralForCausalLM._gen_aoa_config
    _gen_inv_aoa_config = MistralForCausalLM._gen_inv_aoa_config


__all__ = [
    "MistralAttention",
    "MistralDecoderLayer",
    "MistralForCausalLM",
    "MistralForCausalLMPipe",
    "MistralForQuestionAnswering",
    "MistralForSequenceClassification",
    "MistralForTokenClassification",
    "MistralMLP",
    "MistralModel",
    "MistralPretrainedModel",
    "MistralRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
