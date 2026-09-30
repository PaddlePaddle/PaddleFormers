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

"""SeedOss model configuration."""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class SeedOssConfig(PretrainedConfig):
    model_type = "seed_oss"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=155136,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=64,
        num_attention_heads=80,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=524288,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_theta=10000000.0,
        attention_bias=True,
        attention_out_bias=False,
        attention_dropout=0.1,
        residual_dropout=0.1,
        mlp_bias=False,
        head_dim=128,
        use_bias=False,
        fuse_rms_norm=False,
        ignored_index=-100,
        pp_seg_method="layer:SeedOssDecoderLayer",
        dpo_config=None,
        kto_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.use_bias = use_bias
        self.fuse_rms_norm = fuse_rms_norm
        self.ignored_index = ignored_index
        self.pp_seg_method = pp_seg_method
        self.dpo_config = dpo_config
        self.kto_config = kto_config

        self.rope_scaling = kwargs.pop("rope_scaling", None)
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if rope_parameters is None:
            rope_parameters = self.rope_scaling
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        elif "rope_theta" not in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = rope_theta

        self.rope_parameters = rope_parameters
        self.rope_theta = rope_parameters.get("rope_theta", rope_theta)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.register_unsavable_keys(
            [
                "ignored_index",
                "pp_seg_method",
                "dpo_config",
                "kto_config",
            ]
        )

        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)


__all__ = ["SeedOssConfig"]
