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

from .act_quant import act_quant


def fp8_simulate(x: paddle.Tensor, block_size: int):
    y, scale = act_quant(x.contiguous(), block_size, "ue8m0")
    shape = [*list(y.shape[:-1]), -1, block_size]
    y = y.reshape(shape).astype("float32") * scale.unsqueeze(-1)
    return y.flatten(-2, -1).astype(x.dtype)


class DeepSeekV4LinearQATFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply
