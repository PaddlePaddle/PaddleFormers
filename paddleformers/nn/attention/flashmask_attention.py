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
from paddle.nn.functional.flash_attention import flashmask_attention


def flashmask_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    attn_mask_start_row_indices=None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs
):
    if attn_mask_start_row_indices is not None:
        attn_mask_start_row_indices = attn_mask_start_row_indices.unsqueeze(-1)

    # b,l,h,d
    out = flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=attn_mask_start_row_indices,
        causal=True,
    )
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    return out, None
