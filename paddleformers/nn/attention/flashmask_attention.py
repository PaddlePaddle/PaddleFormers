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

from .sink_impl import sink_attention_forward


def flashmask_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attn_mask_startend_row_indices: paddle.Tensor,
    dropout: float = 0.0,
    sink: Optional[paddle.Tensor] = None,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs
):
    # b,l,h,d
    if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.ndim == 3:
        attn_mask_startend_row_indices = attn_mask_startend_row_indices.unsqueeze(-1)
    if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 1:
        is_causal = True
    if sink is None:
        out = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=attn_mask_startend_row_indices,
            causal=True,
        )
    else:
        out = sink_attention_forward(
            query,
            key,
            value,
            sink,
            startend_row_indices=attn_mask_startend_row_indices,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=is_causal if is_causal is not None else False,
        )
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    return out, None
