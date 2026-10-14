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

"""Cohere model configuration."""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class CohereConfig(PretrainedConfig):
    model_type = "cohere"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=8192,
        intermediate_size=22528,
        max_position_embeddings=8192,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_parameters=None,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        head_dim=None,
        logit_scale=0.0625,
        max_sequence_length=None,
        ignored_index=-100,
        pp_seg_method="layer:CohereDecoderLayer",
        dpo_config=None,
        kto_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rms_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.logit_scale = logit_scale
        self.max_sequence_length = max_sequence_length if max_sequence_length is not None else max_position_embeddings
        self.ignored_index = ignored_index
        self.pp_seg_method = pp_seg_method
        self.dpo_config = dpo_config
        self.kto_config = kto_config

        self.rope_theta = rope_theta
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        if rope_parameters is None:
            rope_parameters = self.rope_scaling
        elif "rope_theta" not in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.register_unsavable_keys(
            ["ignored_index", "pp_seg_method", "dpo_config", "kto_config", "max_sequence_length"]
        )
        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)


__all__ = ["CohereConfig"]
