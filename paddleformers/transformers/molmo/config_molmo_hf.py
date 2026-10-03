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
"""Minimal upstream-compatible Molmo configuration for exported checkpoints."""

from transformers import PretrainedConfig


class MolmoConfig(PretrainedConfig):
    model_type = "molmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        embedding_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        layer_norm_eps=1e-5,
        rope_theta=10000.0,
        clip_qkv=None,
        qkv_bias=False,
        weight_tying=False,
        use_position_ids=True,
        tie_word_embeddings=True,
        attention_layer_norm=False,
        norm_after=False,
        layer_norm_type="rms",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.clip_qkv = clip_qkv
        self.qkv_bias = qkv_bias
        self.weight_tying = weight_tying
        self.use_position_ids = use_position_ids
        self.attention_layer_norm = attention_layer_norm
        self.norm_after = norm_after
        self.layer_norm_type = layer_norm_type
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
