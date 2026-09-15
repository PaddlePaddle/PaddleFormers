# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""InternLM model configuration."""

from ..configuration_utils import PretrainedConfig


class InternLMConfig(PretrainedConfig):
    model_type = "internlm"

    def __init__(
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        recompute_use_reentrant=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        bias=True,
        rotary=None,
        attention_dropout=0.0,
        ignored_index=-100,
        micro_batch_size=-1,
        pp_seg_method="layer:InternLMDecoderLayer|EmptyLayer",
        dpo_config=None,
        kto_config=None,
        recompute_granularity=None,
        recompute_method=None,
        recompute_modules=None,
        recompute_num_layers=None,
        recompute_mtp_granularity=None,
        recompute_mtp_method=None,
        recompute_mtp_modules=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.bias = bias
        self.attention_bias = bias
        self.mlp_bias = False
        self.attention_dropout = attention_dropout
        self.head_dim = self.hidden_size // self.num_attention_heads

        rotary = {"base": 10000, "type": "dynamic"} if rotary is None else rotary
        self.rotary = rotary
        self.rope_theta = rotary.get("base", 10000)
        rotary_type = rotary.get("type", "dynamic")
        if rotary_type == "origin":
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}
        elif rotary_type == "dynamic":
            self.rope_parameters = {
                "rope_theta": self.rope_theta,
                "rope_type": "dynamic",
                "factor": rotary.get("scaling_factor", 1.0),
            }
        else:
            raise ValueError("InternLM only supports rotary type 'origin' or 'dynamic'.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.recompute_use_reentrant = recompute_use_reentrant
        self.ignored_index = ignored_index
        self.micro_batch_size = micro_batch_size
        self.pp_seg_method = pp_seg_method
        self.dpo_config = dpo_config
        self.kto_config = kto_config
        self.recompute_granularity = recompute_granularity
        self.recompute_method = recompute_method
        self.recompute_modules = recompute_modules
        self.recompute_num_layers = recompute_num_layers
        self.recompute_mtp_granularity = recompute_mtp_granularity
        self.recompute_mtp_method = recompute_mtp_method
        self.recompute_mtp_modules = recompute_mtp_modules
        self.register_unsavable_keys(
            [
                "ignored_index",
                "recompute_use_reentrant",
                "pp_seg_method",
                "micro_batch_size",
                "dpo_config",
                "kto_config",
                "recompute_granularity",
                "recompute_method",
                "recompute_modules",
                "recompute_num_layers",
                "recompute_mtp_granularity",
                "recompute_mtp_method",
                "recompute_mtp_modules",
            ]
        )
