# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


"""LFM2 text configuration."""

from ..configuration_utils import PretrainedConfig


class Lfm2Config(PretrainedConfig):
    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=2560,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=128000,
        initializer_range=0.02,
        norm_eps=1e-5,
        use_cache=True,
        rope_theta=1000000.0,
        rope_parameters=None,
        conv_bias=False,
        conv_L_cache=3,
        layer_types=None,
        full_attn_idxs=None,
        block_auto_adjust_ff_dim=False,
        block_ffn_dim_multiplier=None,
        block_multiple_of=256,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        fuse_rms_norm=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = kwargs.pop("block_ff_dim", intermediate_size)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps
        # Preserve the reference implementation's unfused FP32 RMSNorm computation.
        self.fuse_rms_norm = fuse_rms_norm
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_theta = self.rope_parameters.get("rope_theta", rope_theta)
        self.rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters.setdefault("rope_type", "default")
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_multiple_of = block_multiple_of
        self.full_attn_idxs = full_attn_idxs
        if layer_types is None:
            full_attn_idxs = full_attn_idxs if full_attn_idxs is not None else list(range(num_hidden_layers))
            layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(num_hidden_layers)]
        allowed_layer_types = ("full_attention", "conv")
        if not all(layer_type in allowed_layer_types for layer_type in layer_types):
            raise ValueError(f"The `layer_types` entries must be in {allowed_layer_types}")
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
                f"({len(layer_types)})"
            )
        self.layer_types = layer_types
        super().__init__(
            tie_word_embeddings=kwargs.pop("tie_embedding", tie_word_embeddings),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = ["Lfm2Config"]
