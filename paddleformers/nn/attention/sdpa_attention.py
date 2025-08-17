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

from typing import Optional

import paddle
import paddle.nn as nn

from .utils import repeat_kv


def sdpa_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):
    # query: b l h d
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    if is_causal is None:
        is_causal = query.shape[1] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    attn_output = nn.functional.scaled_dot_product_attention(
        query, key, value, attention_mask, dropout, is_causal=is_causal, training=module.training
    )
    attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])
    return attn_output, None
