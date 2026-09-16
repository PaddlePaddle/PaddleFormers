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


class TelechatConfig(PretrainedConfig):
    model_type = "telechat"
    attribute_map = {"num_hidden_layers": "n_layer", "num_attention_heads": "n_head"}

    def __init__(
        self,
        vocab_size=120000,
        hidden_size=2048,
        ffn_hidden_size=5460,
        n_layer=16,
        n_head=32,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cache=True,
        training_seqlen=8192,
        base_seqlen=8192,
        embed_layernorm=False,
        apply_residual_connection_post_layernorm=False,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.training_seqlen = training_seqlen
        self.base_seqlen = base_seqlen
        self.embed_layernorm = embed_layernorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
