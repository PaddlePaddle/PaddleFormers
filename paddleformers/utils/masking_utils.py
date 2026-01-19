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

import os
from typing import Optional

import numpy as np
import paddle

from .tools import get_env_device


def _gen_from_sparse_attn_mask_indices(
    attn_mask_startend_row_indices: paddle.Tensor,
    dtype: Optional[paddle.dtype] = paddle.bfloat16,
    is_causal: Optional[bool] = None,
):
    """
    Recover 4-D attention_mask from attn_mask_startend_row_indices.

    Args:
        attn_mask_startend_row_indices (paddle.Tensor):
            A column-wise sparse attention mask row indices tensor.
            A 4-D tensor with shape [batch_size, k_num_heads, k_seq_len, {1, 2, 4}].
            The dtype must be int32. k_num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
            Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings.

            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 1],
              indicating unidirectional attention. The value represents the starting row index of the left
              lower triangular mask in the dense mask. The value startend_row_indices[..., 0] indicates that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked.
            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating unidirectional attention. The values represent the starting and ending row indices of
              the left lower triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1]-th row (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating bidirectional attention. The values represent the starting row index of the left
              lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 1]-th row upwards (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 4] ,
              indicating bidirectional attention. The values represent the start and end row indices of the
              left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:4] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1] row (exclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 2]-th row downwards (inclusive) but above the startend_row_indices[..., 3] row (exclusive) will be masked.
        dtype (paddle.dtype): The data type of the tensor.
        causal (bool): Whether to enable causal mode.

    Returns:
        paddle.Tensor: The dense attention mask recovered from attn_mask_startend_row_indices.
    """

    if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.ndim == 3:
        attn_mask_startend_row_indices = attn_mask_startend_row_indices.unsqueeze(-1)
    if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 1:
        is_causal = True
    if attn_mask_startend_row_indices is not None and attn_mask_startend_row_indices.shape[-1] == 4:
        is_causal = False

    if is_causal is None:
        raise ValueError(
            "The `is_causal` argument must be specified when recovering the dense attention mask from the column-wise sparse attention mask row indices."
        )

    batch_size, num_head, seq_len, bound_num = attn_mask_startend_row_indices.shape
    has_end = (is_causal and bound_num == 2) or ((not is_causal) and bound_num == 4)

    attention_mask = paddle.ones([seq_len, seq_len], dtype="bool").expand([batch_size, num_head, seq_len, seq_len])
    if is_causal:
        attention_mask = paddle.tril(attention_mask)

    base = paddle.arange(seq_len, dtype="int32").unsqueeze(1).expand([batch_size, num_head, -1, seq_len])

    # [batch_size, k_num_heads, k_seq_len, {1, 2, 4}] -> [batch_size, k_num_heads, {1, 2, 4}, k_seq_len]
    mask_indices = attn_mask_startend_row_indices.transpose([0, 1, 3, 2])

    downstart_mask_indices = mask_indices[:, :, 0:1, :]
    downstart_mask_indices = downstart_mask_indices.expand([batch_size, num_head, seq_len, -1])
    lower_tri = base < downstart_mask_indices
    if has_end:
        downend_mask_indices = mask_indices[:, :, 1:2, :]
        downend_mask_indices = downend_mask_indices.expand([batch_size, num_head, seq_len, -1])
        lower_tri = paddle.logical_or(lower_tri, base >= downend_mask_indices)

    attention_mask = paddle.logical_and(attention_mask, lower_tri)

    if not is_causal:
        if has_end:
            upstart_mask_indices = mask_indices[:, :, 2:3, :]
            upstart_mask_indices = upstart_mask_indices.expand([batch_size, num_head, seq_len, -1])
            upend_mask_indices = mask_indices[:, :, 3:4, :]
            upend_mask_indices = upend_mask_indices.expand([batch_size, num_head, seq_len, -1])
            upper_tri = base >= upend_mask_indices
            upper_tri = paddle.logical_or(upper_tri, base < upstart_mask_indices)
        else:
            upend_mask_indices = mask_indices[:, :, 1:2, :]
            upend_mask_indices = upend_mask_indices.expand([batch_size, num_head, seq_len, -1])
            upper_tri = base >= upend_mask_indices

        attention_mask = paddle.logical_and(attention_mask, upper_tri)

    attention_mask = paddle.scale(
        x=attention_mask.astype(dtype),
        scale=1000000.0,
        bias=-1.0,
        bias_after_scale=False,
    )
    return attention_mask


def get_use_casual_mask():
    """Get the value of the 'USE_CASUAL_MASK' environment variable."""
    return os.getenv("USE_CASUAL_MASK", "False") == "True"


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask.to("bool"), y, x)


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make casual mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    if get_env_device() == "npu":
        mask = paddle.tril(paddle.ones((target_length, target_length))).astype("int32")
    else:
        mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.cat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    if get_env_device() == "npu":
        mask = mask[:, None, None, :].astype(dtype)
    else:
        mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if np.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = int(2 ** np.floor(np.log2(n)))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def build_alibi_tensor(
    bool_attention_mask: paddle.Tensor, num_heads: int, dtype: paddle.dtype, tensor_model_parallel_size=1
) -> paddle.Tensor:
    batch_size, seq_length = bool_attention_mask.shape[0], bool_attention_mask.shape[-1]
    slopes = paddle.to_tensor(_get_interleave(num_heads), dtype="float32")
    alibi = slopes.unsqueeze(axis=[1, 2]) * paddle.arange(seq_length, dtype="float32").unsqueeze(axis=[0, 1]).expand(
        [num_heads, -1, -1]
    )
    alibi = alibi.reshape(shape=(1, num_heads, 1, seq_length)).expand([batch_size, -1, -1, -1])
    return paddle.cast(alibi, dtype)
