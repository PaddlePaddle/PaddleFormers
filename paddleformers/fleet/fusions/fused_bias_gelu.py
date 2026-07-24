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
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import paddle

from paddleformers.fleet.jit import jit_fuser

# BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + paddle.erf(x * 0.70710678))


@jit_fuser
def bias_gelu(bias, y):
    x = bias + y
    return (
        x * 0.5 * (1.0 + paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
    )


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + paddle.erf(x * 0.70710678)) + 0.3989423 * x * paddle.exp(-0.5 * x * x)
@jit_fuser
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(paddle.autograd.PyLayer):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensor()
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

    # This is required to make Sphinx happy :-(
    @classmethod
    def apply(cls, *args, **kwargs):
        return super().apply(*args, **kwargs)


bias_gelu_impl = GeLUFunction.apply
