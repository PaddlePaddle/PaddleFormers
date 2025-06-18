# Copyright (c) 2023 Alibaba Cloud and PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["QWenConfig"]


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        emb_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=8192,
        scale_attn_weights=True,
        use_cache=True,
        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        intermediate_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        long_sequence_strategy_type=None,
        long_sequence_strategy_name=None,
        long_sequence_init_args=None,
        use_long_sequence_strategies=False,
        dpo_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.kv_channels = kv_channels

        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.no_bias = no_bias
        self.long_sequence_strategy_type = long_sequence_strategy_type
        self.long_sequence_strategy_name = long_sequence_strategy_name
        self.long_sequence_init_args = {} if long_sequence_init_args is None else long_sequence_init_args
        self.use_long_sequence_strategies = use_long_sequence_strategies
        self.dpo_config = dpo_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
