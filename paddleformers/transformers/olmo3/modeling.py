# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

"""OLMo3 model implementation for PaddlePaddle.

Migrated from transformers.models.olmo3.modeling_olmo3
"""

from typing import Optional, cast

import paddle
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache, DynamicCache
from ..masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)
from ..model_outputs import BaseModelOutputWithPast
from ..model_utils import register_base_model
from ..olmo2.modeling import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2ForCausalLMPipe,
    Olmo2PretrainedModel,
    Olmo2RotaryEmbedding,
)
from .configuration import Olmo3Config


class Olmo3Attention(Olmo2Attention):
    """OLMo3 attention with sliding window support.

    Identical to OLMo2 attention except:
    - Sliding window attention is used for 3 out of 4 layers
    - Each layer has an attention_type attribute
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None


class Olmo3DecoderLayer(Olmo2DecoderLayer):
    """OLMo3 decoder layer with sliding window attention support."""

    _attention_cls = Olmo3Attention


class Olmo3RotaryEmbedding(Olmo2RotaryEmbedding):
    """OLMo3 rotary embedding with rope_type support.

    Supports explicit rope_type parameter (used for sliding_attention layers).
    Also handles BC: "rope_type" was originally "type" in the config.
    """

    def __init__(self, config: Olmo3Config, rope_type: Optional[str] = None):
        # Handle BC: "rope_type" was originally "type" in the config
        if rope_type is None and hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            rope_type = config.rope_parameters.get("rope_type", config.rope_parameters.get("type"))
        super().__init__(config, rope_type=rope_type)


class Olmo3PretrainedModel(Olmo2PretrainedModel):
    """Base class for OLMo3 models."""

    config_class = Olmo3Config
    pass


@register_base_model
class Olmo3Model(Olmo3PretrainedModel):
    """OLMo3 model (base model without LM head).

    Key differences from OLMo2:
    - Sliding window attention is used for 3 out of 4 layers
    - RoPE scaling is not applied to sliding window attention layers
    - Uses ModuleDict with two RotaryEmbeddings instead of a single one
    """

    def __init__(self, config: Olmo3Config):
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
            [Olmo3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config,
            norm_type="rms_norm",
            hidden_size=config.hidden_size,
            has_bias=False,
            norm_eps=config.rms_norm_eps,
            input_is_parallel=config.sequence_parallel,
        )

        # Two separate RoPE embeddings: sliding_attention uses default (no scaling), full_attention uses configured rope
        self.rotary_embs = nn.LayerDict(
            {
                "sliding_attention": Olmo3RotaryEmbedding(config=config, rope_type="default"),
                "full_attention": Olmo3RotaryEmbedding(config=config),
            }
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
        bsz, seq_length, _ = inputs_embeds.shape

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        kv_seq_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = (
                paddle.arange(kv_seq_len, seq_length + kv_seq_len, dtype=paddle.int64).unsqueeze(0).tile((bsz, 1))
            )

        # Create separate masks for full_attention and sliding_attention
        if not isinstance(causal_mask_mapping := attention_mask, dict):
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
            causal_mask_mapping = {
                "full_attention": create_causal_mask_and_row_indices(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask_and_row_indices(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        position_embeddings_mapping = {
            "sliding_attention": self.rotary_embs["sliding_attention"](hidden_states, position_ids),
            "full_attention": self.rotary_embs["full_attention"](hidden_states, position_ids),
        }

        all_hidden_states = [] if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            attention_type = decoder_layer.self_attn.attention_type

            if isinstance(causal_mask_mapping, dict):
                mask_result = causal_mask_mapping[attention_type]
                if isinstance(mask_result, tuple):
                    layer_causal_mask, layer_attn_mask_startend_row_indices = mask_result
                else:
                    layer_causal_mask = mask_result
                    layer_attn_mask_startend_row_indices = attn_mask_startend_row_indices
            else:
                layer_causal_mask = causal_mask_mapping
                layer_attn_mask_startend_row_indices = attn_mask_startend_row_indices

            layer_position_embeddings = position_embeddings_mapping[attention_type]

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
                    layer_causal_mask,
                    layer_attn_mask_startend_row_indices,
                    position_ids,
                    layer_position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_causal_mask,
                    attn_mask_startend_row_indices=layer_attn_mask_startend_row_indices,
                    position_ids=position_ids,
                    position_embeddings=layer_position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple | list) else layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states) if all_hidden_states else None

        if not return_dict:
            outputs = []
            outputs.append(hidden_states)
            if output_hidden_states:
                outputs.append(all_hidden_states)
            return tuple(outputs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class Olmo3ForCausalLM(Olmo2ForCausalLM):
    """OLMo3 model for causal language modeling.

    Inherits from Olmo2ForCausalLM — identical forward pass, only model class differs.
    """

    config_class = Olmo3Config
    _model_cls = Olmo3Model


class Olmo3ForCausalLMPipe(Olmo2ForCausalLMPipe):
    """OLMo3 model for pipeline parallel training."""

    config_class = Olmo3Config
    _decoder_layer_cls = Olmo3DecoderLayer
