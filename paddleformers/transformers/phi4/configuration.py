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

import math

from ..configuration_utils import PretrainedConfig


class Phi4Config(PretrainedConfig):
    model_type = "phi4"

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=32,
        num_attention_heads=40,
        num_key_value_heads=20,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        bos_token_id=199999,
        eos_token_id=199999,
        pad_token_id=199999,
        sliding_window=512,
        mb_per_layer=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.mb_per_layer = mb_per_layer
        self.pad_token_id = pad_token_id
        self.mlp_bias = mlp_bias
        self.lm_head_bias = lm_head_bias
        if isinstance(sliding_window, list):
            self.sliding_window = sliding_window
        else:
            self.sliding_window = [
                sliding_window if layer_idx < num_hidden_layers // 2 and layer_idx % 2 == 1 else None
                for layer_idx in range(num_hidden_layers)
            ]

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        layer_block_types = []
        for i in range(self.num_hidden_layers):
            if i % 2 == 1:
                layer_block_type = "attention" if i <= (self.num_hidden_layers // 2 + 1) else "shared_attention"
            else:
                layer_block_type = "mamba"
            layer_block_types.append(layer_block_type)
        return layer_block_types
