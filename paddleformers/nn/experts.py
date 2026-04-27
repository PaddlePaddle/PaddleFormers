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

import paddle
import paddle.nn as nn

from .activation import ACT2FN


class MoeExpertsBase(nn.Layer):
    pass


class MoeExperts(MoeExpertsBase):
    def __init__(self, config):
        super().__init__()
        if hasattr(config, "n_routed_experts"):
            self.num_experts = config.n_routed_experts
        else:
            self.num_experts = config.num_experts
        if hasattr(config, "moe_intermediate_size"):
            self.intermediate_dim = config.moe_intermediate_size
        else:
            self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_proj = self.create_parameter(
            shape=[self.num_experts, self.hidden_dim, 2 * self.intermediate_dim],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
        )
        self.down_proj = self.create_parameter(
            shape=[self.num_experts, self.intermediate_dim, self.hidden_dim],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = paddle.zeros_like(hidden_states)
        with paddle.no_grad():
            expert_mask = paddle.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = paddle.greater(expert_mask.sum(dim=(-1, -2)), paddle.to_tensor(0, dtype="int32")).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = paddle.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
