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

from ...utils.masking_utils import _gen_from_sparse_attn_mask_indices


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    dropout: float = 0.0,
    sink: Optional[paddle.Tensor] = None,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):

    if hasattr(module, "num_key_value_groups"):
        num_key_value_groups = module.num_key_value_groups
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)

    if attention_mask is None and kwargs.get("attn_mask_startend_row_indices", None) is not None:
        attn_mask_startend_row_indices = kwargs["attn_mask_startend_row_indices"]
        if attn_mask_startend_row_indices.ndim == 3:
            attn_mask_startend_row_indices = attn_mask_startend_row_indices.unsqueeze(-1)
        if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 1:
            is_causal = True
        if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 4:
            is_causal = False

        attention_mask = _gen_from_sparse_attn_mask_indices(attn_mask_startend_row_indices, query.dtype, is_causal)

    attn_weights = paddle.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    if sink is not None:
        sink = sink.reshape([1, -1, 1, 1]).expand([query.shape[0], -1, query.shape[-2], -1])
        combined_logits = paddle.cat([attn_weights, sink], axis=-1)
        probs = nn.functional.softmax(combined_logits, axis=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    else:
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).astype(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = paddle.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = paddle.reshape(x=attn_output, shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]])

    return attn_output, attn_weights
