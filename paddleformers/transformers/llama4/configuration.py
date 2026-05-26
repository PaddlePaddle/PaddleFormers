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


class Llama4TextConfig(PretrainedConfig):
    model_type = "llama4_text"

    def __init__(
        self,
        vocab_size=202048,
        hidden_size=5120,
        intermediate_size=8192,
        intermediate_size_mlp=16384,
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        attention_bias=False,
        num_experts_per_tok=1,
        num_local_experts=16,
        interleave_moe_layer_step=1,
        use_qk_norm=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        no_rope_layer_interval=4,
        attention_chunk_size=8192,
        attn_temperature_tuning=True,
        floor_scale=8192,
        attn_scale=0.1,
        fuse_rms_norm=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.interleave_moe_layer_step = interleave_moe_layer_step
        self.use_qk_norm = use_qk_norm
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.no_rope_layer_interval = no_rope_layer_interval
        self.attention_chunk_size = attention_chunk_size
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.fuse_rms_norm = fuse_rms_norm

        self.moe_layers = kwargs.pop("moe_layers", None)
        if self.moe_layers is None:
            self.moe_layers = list(
                range(
                    interleave_moe_layer_step - 1,
                    num_hidden_layers,
                    interleave_moe_layer_step,
                )
            )

        self.no_rope_layers = kwargs.pop("no_rope_layers", None)
        if not self.no_rope_layers:
            self.no_rope_layers = [int((i + 1) % no_rope_layer_interval != 0) for i in range(num_hidden_layers)]

        self.layer_types = kwargs.pop("layer_types", None)
        if self.layer_types is None:
            self.layer_types = [
                "chunked_attention" if self.no_rope_layers[i] else "full_attention" for i in range(num_hidden_layers)
            ]

        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = self.rope_scaling.copy() if isinstance(self.rope_scaling, dict) else {}
            rope_parameters.setdefault("rope_type", rope_parameters.pop("type", "default"))
            rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters = rope_parameters
        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

        kwargs.setdefault("_attn_implementation", "eager")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
