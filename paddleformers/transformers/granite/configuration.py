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


class GraniteConfig(PretrainedConfig):
    model_type = "granitemoehybrid"
    tokenizer_class = "GPT2Tokenizer"

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=1024,
        intermediate_size=2048,
        shared_intermediate_size=2048,
        max_position_embeddings=32768,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=4,
        initializer_range=0.1,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=100256,
        bos_token_id=100257,
        eos_token_id=100257,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        tie_word_embeddings=True,
        # MuP scaling factors
        embedding_multiplier=1.0,
        attention_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
        # MoE (unused for 350m-base, all zero)
        num_local_experts=0,
        num_experts_per_tok=0,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        # Mamba (unused for 350m-base, all attention layers)
        mamba_n_heads=128,
        mamba_n_groups=1,
        mamba_d_state=256,
        mamba_d_head="auto",
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=256,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        layer_types=None,
        position_embedding_type="rope",
        rope_theta=10000000.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
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
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        # MuP scaling factors
        self.embedding_multiplier = embedding_multiplier
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier
        self.logits_scaling = logits_scaling

        # MoE params
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # Mamba params (for architecture compatibility, unused in 350m-base)
        self.mamba_n_heads = mamba_n_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_head = mamba_d_head
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        # Layer types
        if layer_types is None:
            layer_types = ["attention"] * num_hidden_layers
        if any(layer_type != "attention" for layer_type in layer_types):
            raise ValueError("GraniteForCausalLM supports attention-only Granite configurations.")
        if num_local_experts or num_experts_per_tok:
            raise ValueError("GraniteForCausalLM does not support MoE configurations.")
        self.layer_types = layer_types
        self.position_embedding_type = position_embedding_type

        # RoPE
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.rope_parameters = self.rope_scaling if self.rope_scaling is not None else {"rope_type": "default", "rope_theta": rope_theta}
        standardize_rope_params(self, rope_theta=self.rope_theta)
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
