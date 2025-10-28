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

# from ..configuration_utils import PretrainedConfig, layer_type_validation
from ..configuration_utils import PretrainedConfig

# from ...modeling_rope_utils import rope_config_validation


class GptOssConfig(PretrainedConfig):
    r"""
    This will yield a configuration to that of the BERT
    [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.
    """

    model_type = "gpt_oss"

    def __init__(
        self,
        num_hidden_layers: int = 24,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings=False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings=131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
        attention_dropout: float = 0.0,
        num_experts_per_tok=4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits=False,
        use_cache=True,
        layer_types=None,
        pp_seg_method="layer:GptOssDecoderLayer",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_local_experts
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        # layer_type_validation(self.layer_types)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        # rope_config_validation(self)

        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        self.use_bias = False
        self.pp_seg_method = pp_seg_method

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["GptOssConfig"]
