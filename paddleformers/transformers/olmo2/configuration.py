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

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class Olmo2Config(PretrainedConfig):
    model_type = "olmo2"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_hidden_layers=32,
        num_attention_heads=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        hidden_act="silu",
        num_key_value_heads=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        head_dim=None,
        rope_theta=10000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        self.rope_theta = rope_theta
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling
        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
