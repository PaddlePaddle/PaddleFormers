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

"""Diagnostic equivalent MoE accumulation without in-place index_add_."""
import paddle
import paddle.nn.functional as F


def forward(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states = hidden_states * paddle.uniform(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            min=1.0 - self.jitter_noise,
            max=1.0 + self.jitter_noise,
        )
    flat = hidden_states.reshape([-1, hidden_dim])
    result = paddle.zeros_like(flat)
    logits = F.softmax(self.gate(flat).astype("float32"), axis=-1)
    weights, selected = paddle.topk(logits, self.top_k, axis=-1)
    weights = (weights / weights.sum(axis=-1, keepdim=True)).astype(hidden_states.dtype)
    mask = F.one_hot(selected, num_classes=self.num_experts).transpose([2, 1, 0])
    for i in range(self.num_experts):
        positions, indices = paddle.where(mask[i])
        values = self.experts[i](flat[indices])
        values = values * weights[indices, positions, None]
        result = paddle.index_add(result, index=indices, value=values.astype(result.dtype), axis=0)
    return result.reshape([batch_size, sequence_length, hidden_dim])


def install():
    from paddleformers.transformers.minimax.modeling import MiniMaxSparseMoeBlock

    MiniMaxSparseMoeBlock.forward = forward
