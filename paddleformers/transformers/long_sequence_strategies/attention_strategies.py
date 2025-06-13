# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math

import numpy as np
import paddle
from paddle import Tensor, nn

__all__ = ["AttentionWithLinearBias"]


class AttentionWithLinearBias(nn.Layer):
    def __init__(self, **init_args):
        super().__init__()

    def _get_interleave(self, n):
        def _get_interleave_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return np.array([start * start**i for i in range(n)]).astype(np.float32)

        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                _get_interleave_power_of_2(closest_power_of_2)
                + self._get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    def forward(self, bool_attention_mask: Tensor, num_heads: int, dtype: paddle.dtype):
        attention_mask = bool_attention_mask.astype("float32")
        batch_size, seq_length = attention_mask.shape[0], attention_mask.shape[-1]
        slopes = paddle.to_tensor(self._get_interleave(num_heads), dtype="float32")
        with paddle.amp.auto_cast(enable=False):
            alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(
                axis=[0, 1]
            ).expand([num_heads, -1, -1])
        alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
        return paddle.cast(alibi, dtype)
