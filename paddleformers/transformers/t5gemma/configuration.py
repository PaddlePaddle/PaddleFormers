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

"""T5Gemma model configuration."""

from typing import Any, Optional, Union

from ..configuration_utils import PretrainedConfig, layer_type_validation
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class T5GemmaModuleConfig(PretrainedConfig):
    model_type = "t5_gemma_module"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        rope_scaling=None,
        rope_parameters=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.layer_types = layer_types
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_scaling or rope_parameters

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)


class T5GemmaConfig(PretrainedConfig):
    model_type = "t5gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "encoder": T5GemmaModuleConfig,
        "decoder": T5GemmaModuleConfig,
    }

    def __init__(
        self,
        encoder: Optional[Union[T5GemmaModuleConfig, dict[Any, Any]]] = None,
        decoder: Optional[Union[T5GemmaModuleConfig, dict[Any, Any]]] = None,
        is_encoder_decoder: bool = True,
        dropout_rate: float = 0.0,
        classifier_dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        vocab_size: int = 256000,
        **kwargs,
    ):
        if isinstance(encoder, dict):
            encoder = T5GemmaModuleConfig(**encoder)
        elif encoder is None:
            encoder = T5GemmaModuleConfig()
        elif not isinstance(encoder, T5GemmaModuleConfig):
            raise TypeError(f"{type(encoder)} is not supported.")

        if isinstance(decoder, dict):
            decoder = T5GemmaModuleConfig(**decoder)
        elif decoder is None:
            decoder = encoder
        elif not isinstance(decoder, T5GemmaModuleConfig):
            raise TypeError(f"{type(decoder)} is not supported.")

        encoder = T5GemmaModuleConfig(**encoder.to_dict())
        decoder = T5GemmaModuleConfig(**decoder.to_dict())

        encoder.is_decoder = False
        encoder.dropout_rate = dropout_rate
        encoder.attention_dropout = attention_dropout
        self.encoder = encoder

        decoder.is_decoder = True
        decoder.use_cache = True
        decoder.dropout_rate = dropout_rate
        decoder.attention_dropout = attention_dropout
        decoder.cross_attention_hidden_size = encoder.hidden_size
        self.decoder = decoder

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(decoder, special_token_key)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = kwargs.get("use_cache", decoder.use_cache)
        self.initializer_range = kwargs.get("initializer_range", decoder.initializer_range)
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.classifier_dropout_rate = classifier_dropout_rate
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size

    def __setattr__(self, key, value):
        shared_attr_with_submodules = [
            "output_hidden_states",
            "output_attentions",
            "_attn_implementation",
            "dropout_rate",
            "attention_dropout",
            "vocab_size",
        ]
        if key in shared_attr_with_submodules:
            if hasattr(self, "encoder"):
                setattr(self.encoder, key, value)
            if hasattr(self, "decoder"):
                setattr(self.decoder, key, value)
        super().__setattr__(key, value)


__all__ = ["T5GemmaConfig", "T5GemmaModuleConfig"]
