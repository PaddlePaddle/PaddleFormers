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


from modular_moe_layer import ModularMoELayer


class QuickAccessMoEFactory:
    @classmethod
    def create_from_model_name(
        cls,
        config,
        model_name: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 1,
        expert_parallel_degree: int = 1,
    ) -> ModularMoELayer:
        return ModularMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_shared_experts=num_shared_experts,
            expert_parallel_degree=expert_parallel_degree,
            **config,
        )
