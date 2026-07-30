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

from paddleformers.transformers.qwen3_5.configuration import Qwen3_5TextConfig
from paddleformers.transformers.qwen3_5.modeling_fleet import Qwen3_5TextModelProvider
from paddleformers.transformers.qwen3_5_moe.configuration import Qwen3_5MoETextConfig


def provider_from(config):
    return Qwen3_5TextModelProvider.from_config(config)


def test_dense_provider_clears_default_expert_fields():
    config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        head_dim=16,
    )

    provider = provider_from(config)

    assert provider.model_type == "qwen3_5_text"
    assert provider.n_routed_experts is None
    assert provider.n_shared_experts == 0
    assert provider.moe_shared_expert_gate is False


def test_moe_provider_preserves_routed_experts():
    config = Qwen3_5MoETextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
    )

    provider = provider_from(config)

    assert provider.model_type == "qwen3_5_moe_text"
    assert provider.n_routed_experts == 8
    assert provider.n_shared_experts == 1
    assert provider.moe_shared_expert_gate is True
