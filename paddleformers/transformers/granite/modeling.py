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

from typing import Callable, Optional, cast

import paddle
from paddle import nn
import paddle.nn.functional as F

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ...utils.log import logger
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import create_causal_mask_and_row_indices
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_utils import PretrainedModel, register_base_model
from ..modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ..activations import ACT2FN
from .configuration import GraniteConfig

try:
    from paddle.distributed.flex_checkpoint.dcp.sharded_weight import build_sharded_state_dict
except ImportError:
    build_sharded_state_dict = None


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    original_dtype = q.dtype
    q_embed = paddle.add(paddle.multiply(q.astype("float32"), cos), paddle.multiply(rotate_half(q).astype("float32"), sin))
    k_embed = paddle.add(paddle.multiply(k.astype("float32"), cos), paddle.multiply(rotate_half(k).astype("float32"), sin))
    return q_embed.astype(original_dtype), k_embed.astype(original_dtype)


class GraniteAttention(nn.Layer):
    def __init__(self, config: GraniteConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        assert config.num_attention_heads % config.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # Granite difference: scaling = attention_multiplier instead of head_dim**-0.5
        self.scaling = config.attention_multiplier
        self.attention_dropout = config.attention_dropout

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
        batch_size, seq_len = hidden_states.shape[:2]
        q_shape = (batch_size, seq_len, -1, self.head_dim)
        kv_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = paddle.reshape(self.q_proj(hidden_states), [batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = paddle.reshape(self.k_proj(hidden_states), [batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = paddle.reshape(self.v_proj(hidden_states), [batch_size, seq_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

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
        attn_output = attn_output.reshape([batch_size, seq_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GraniteMLP(nn.Layer):
    """Granite shared MLP: uses input_linear (fused gate+up) and output_linear (down)."""

    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.input_linear = GeneralLinear.create(
            self.hidden_size,
            self.intermediate_size * 2,
            has_bias=False,
            config=config,
            tp_plan="colwise",
        )
        self.output_linear = GeneralLinear.create(
            self.intermediate_size,
            self.hidden_size,
            has_bias=False,
            config=config,
            tp_plan="rowwise",
        )

    def forward(self, x):
        x = self.input_linear(x)
        chunked = x.chunk(2, axis=-1)
        x = self.act_fn(chunked[0]) * chunked[1]
        x = self.output_linear(x)
        return x


class GraniteDecoderLayer(nn.Layer):
    def __init__(self, config: GraniteConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = GraniteAttention(config=config, layer_idx=layer_idx)
        self.shared_mlp = GraniteMLP(config)
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
        # Granite difference: residual_multiplier
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor]:
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
        # Granite difference: residual + hidden_states * residual_multiplier
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteRotaryEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

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
        config: Optional[GraniteConfig] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["paddle.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.int64).astype(dtype=paddle.float32) / dim))
        return inv_freq, attention_factor

    @dynamic_rope_update
    def forward(self, x, position_ids):
        with paddle.amp.auto_cast(enable=False):
            inv_freq = self.inv_freq.astype("float32")
            # Use broadcasting instead of matmul: [1, 1, 32] * [batch, seq, 1] = [batch, seq, 32]
            freqs = paddle.multiply(
                inv_freq.reshape([1, 1, -1]),
                position_ids.astype("float32").unsqueeze(-1)
            )  # [batch, seq, head_dim/2]
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GranitePretrainedModel(PretrainedModel):
    config_class = GraniteConfig
    base_model_prefix = "model"
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "input_linear",
        "output_linear",
    ]

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        if build_sharded_state_dict is None:
            raise ImportError(
                "The current version of paddlepaddle does not support 'build_sharded_state_dict'. "
                "Please install paddlepaddle>=3.2."
            )
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, None, structured_name_prefix)

    @classmethod
    def _gen_aoa_config(cls, config: GraniteConfig):
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
                f"model.layers.$LAYER_ID.shared_mlp.{mlp_name}.weight^T -> {model_prefix}layers.$LAYER_ID.shared_mlp.{mlp_name}.weight"
                for mlp_name in ["input_linear", "output_linear"]
            ]
        )
        if cls != cls.base_model_class:
            if config.tie_word_embeddings:
                aoa_statements.append("model.embed_tokens.weight -> lm_head.weight")
            else:
                aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}

    @classmethod
    def _gen_inv_aoa_config(cls, config: GraniteConfig):
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
                f"{model_prefix}layers.$LAYER_ID.shared_mlp.{mlp_name}.weight^T -> model.layers.$LAYER_ID.shared_mlp.{mlp_name}.weight"
                for mlp_name in ["input_linear", "output_linear"]
            ]
        )

        if not config.tie_word_embeddings and cls != cls.base_model_class:
            aoa_statements.append("lm_head.weight -> lm_head.weight")

        return {"aoa_statements": aoa_statements}


@register_base_model
class GraniteModel(GranitePretrainedModel):
    def __init__(self, config: GraniteConfig):
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
            [GraniteDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )
        self.rotary_emb = GraniteRotaryEmbedding(config=config)
        # Granite difference: embedding_multiplier
        self.embedding_multiplier = config.embedding_multiplier

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
        # Granite difference: multiply by embedding_multiplier
        inputs_embeds = inputs_embeds * self.embedding_multiplier
        bsz, seq_length, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile((bsz, 1))
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "batch_size": bsz,
            "seq_length": seq_length,
            "cache_length": kv_seq_len,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "prepare_decoder_attention_mask": self._prepare_decoder_attention_mask,
        }
        causal_mask, attn_mask_startend_row_indices = create_causal_mask_and_row_indices(**mask_kwargs)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        all_hidden_states = [] if output_hidden_states else None

        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
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
            return tuple(output for output in (hidden_states, past_key_values, all_hidden_states) if output is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class GraniteForCausalLM(GranitePretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: GraniteConfig):
        super().__init__(config)
        self.config = config
        self.model = GraniteModel(config)
        self.lm_head = GeneralLMHead(config)
        self.criterion = CriterionLayer(config)
        # Granite difference: logits_scaling
        self.logits_scaling = config.logits_scaling
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
        # Granite difference: divide by logits_scaling
        logits = logits / self.logits_scaling

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape([-1, logits.shape[-1]])
            shift_labels = labels[:, 1:].reshape([-1])
            valid = shift_labels != -100
            if loss_mask is not None:
                valid = valid & paddle.cast(loss_mask[:, 1:].reshape([-1]), paddle.bool)
            safe_labels = paddle.where(valid, shift_labels, paddle.zeros_like(shift_labels))
            selected_log_probs = paddle.take_along_axis(
                F.log_softmax(shift_logits, axis=-1), safe_labels.unsqueeze(-1), axis=-1
            ).squeeze(-1)
            valid_float = paddle.cast(valid, selected_log_probs.dtype)
            valid_count = valid_float.sum()
            loss = -(selected_log_probs * valid_float).sum() / paddle.clip(valid_count, min=1.0)

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
