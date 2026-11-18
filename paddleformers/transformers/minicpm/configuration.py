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
"""MiniCPM model configuration."""

from ..configuration_utils import PretrainedConfig
from ..modeling_rope_utils import rope_config_validation, standardize_rope_params


class MiniCPMConfig(PretrainedConfig):
    r"""MiniCPM configuration, with defaults matching the original HF config.

    Args:
        vocab_size (int): Vocabulary size. Defaults to 32000.
        hidden_size (int): Decoder width. Defaults to 4096.
        intermediate_size (int): Feed-forward width. Defaults to 11008.
        num_hidden_layers (int): Decoder layer count. Defaults to 32.
        num_attention_heads (int): Query head count. Defaults to 32.
        num_key_value_heads (int, optional): KV heads; defaults to query heads.
        hidden_act (str): MLP activation. Defaults to ``silu``.
        max_position_embeddings (int): Maximum context length. Defaults to 2048.
        initializer_range (float): Weight initialization std. Defaults to 0.02.
        rms_norm_eps (float): RMSNorm epsilon. Defaults to 1e-6.
        use_cache (bool): Return KV cache. Defaults to True.
        pad_token_id (int, optional): Padding token ID. Defaults to None.
        bos_token_id (int): Beginning-of-sequence ID. Defaults to 1.
        eos_token_id (int): End-of-sequence ID. Defaults to 2.
        pretraining_tp (int): Original pretraining TP factor. Defaults to 1.
        tie_word_embeddings (bool): Share embedding/head weights. Defaults to True.
        rope_theta (float): RoPE base period. Defaults to 10000.0.
        rope_scaling (dict, optional): RoPE scaling parameters. Defaults to None.
        attention_bias (bool): Enable attention projection biases. Defaults to False.
        attention_dropout (float): Attention dropout probability. Defaults to 0.0.
        scale_emb (float): Token embedding multiplier. Defaults to 1.
        dim_model_base (int): Reference width for LM-head scaling. Defaults to 1.
        scale_depth (float): Residual depth multiplier. Defaults to 1.
        **kwargs: Additional PaddleFormers configuration options.
    """

    model_type = "minicpm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        scale_emb=1,
        dim_model_base=1,
        scale_depth=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth

        self.layer_types = ["full_attention"] * num_hidden_layers

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling

        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniCPMConfig"]
