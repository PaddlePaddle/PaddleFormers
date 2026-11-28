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

"""RMS normalization with the source model's opmath epsilon semantics."""
import paddle
from paddle import nn
from paddle.nn import functional as F


def rms_norm(x, normalized_shape, weight=None, eps=None):
    # PyTorch 2.11 uses the float32 compute epsilon for half/bfloat16 inputs.
    if eps is None:
        eps = 2.220446049250313e-16 if x.dtype == paddle.float64 else 1.1920928955078125e-7
    if x.place.is_gpu_place():
        return F.rms_norm(x, normalized_shape, weight=weight, eps=eps)
    # Accumulate half precision inputs in float32, matching the fused kernel.
    compute_dtype = "float64" if x.dtype == paddle.float64 else "float32"
    with paddle.amp.auto_cast(False):
        values = x.astype(compute_dtype)
        axes = list(range(values.ndim - len(normalized_shape), values.ndim))
        values = values * paddle.rsqrt(values.square().mean(axis=axes, keepdim=True) + eps)
        if weight is not None:
            values = values * weight.astype(compute_dtype)
        return values.astype(x.dtype)


class RMSNorm(nn.Layer):
    def __init__(self, dim, eps=None):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter([dim], default_initializer=nn.initializer.Constant(1.0))

    def forward(self, x):
        return rms_norm(x, [x.shape[-1]], weight=self.weight, eps=self.eps)
