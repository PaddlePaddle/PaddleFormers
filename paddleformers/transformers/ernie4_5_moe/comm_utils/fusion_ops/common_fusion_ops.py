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

"""
Common fusion operators
"""

import paddle
import paddle.nn.functional as F
from paddle import matmul, tensor

if paddle.is_compiled_with_xpu():
    try:
        from paddle_xpu.layers.nn import Linear
    except ImportError:
        from paddle.nn import Linear
else:
    from paddle.nn import Linear
from paddle.nn.functional.flash_attention import flashmask_attention

__all__ = [
    "matmul",
    "Linear",
]


def _fusion_flash_attention(
    q,
    k,
    v,
    training_mode,
    attention_probs_dropout_prob,
    use_sparse_flash_attn,
    attention_mask=None,
    attn_mask_start_row_indices=None,
    rr_flash_attn=None,
):
    """
    Performs fused flash attention with multiple implementation variants.

    Args:
        q (paddle.Tensor): Query tensor with shape [batch, heads, seq_len, dim_head]
        k (paddle.Tensor): Key tensor with shape [batch, heads, seq_len, dim_head]
        v (paddle.Tensor): Value tensor with shape [batch, heads, seq_len, dim_head]
        training_mode (bool): Whether in training mode (affects dropout)
        attention_probs_dropout_prob (float): Dropout probability for attention weights
        use_sparse_flash_attn (bool): Whether to use sparse flash attention optimization
        attention_mask (Optional[paddle.Tensor]): Dense attention mask (default: None)
        attn_mask_start_row_indices (Optional[paddle.Tensor]): Sparse mask indices (default: None)
        rr_flash_attn (Optional[Callable]): Recomputation wrapper for flash attention (default: None)

    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
            - Output tensor with shape [batch, seq_len, heads*dim_head]
            - Attention weights (None for flash attention implementations)

    Raises:
        Warning: If sparse flash attention is requested but unavailable
        ValueError: If invalid combination of mask inputs is provided
    """

    # version = paddle.version.full_version
    if attn_mask_start_row_indices is not None:
        if use_sparse_flash_attn:
            if rr_flash_attn is None:
                out = flashmask_attention(
                    q,
                    k,
                    v,
                    startend_row_indices=attn_mask_start_row_indices.unsqueeze(-1),
                    causal=True,
                )
            else:
                out = rr_flash_attn(
                    flashmask_attention,
                    q,
                    k,
                    v,
                    startend_row_indices=attn_mask_start_row_indices.unsqueeze(-1),
                    causal=True,
                )
        else:
            attention_mask = _gen_from_sparse_attn_mask_indices(attn_mask_start_row_indices, q.dtype)
            if rr_flash_attn is None:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    is_causal=False,
                )
            else:
                out = rr_flash_attn(
                    F.scaled_dot_product_attention,
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    is_causal=False,
                )
        weights = None
    else:
        if rr_flash_attn is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and q.shape[1] != 1,
            )
            weights = None
        else:
            out = rr_flash_attn(
                F.scaled_dot_product_attention,
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None and q.shape[1] != 1,
            )
            weights = None

    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
    return out, weights


def _gen_from_sparse_attn_mask_indices(attn_mask_start_row_indices, dtype):
    """
    Recover 4-D attention_mask from attn_mask_start_row_indices.

    Args:
        attn_mask_start_row_indices (paddle.Tensor): The start row indices for the attention mask.
        dtype (str): The data type of the tensor.

    Returns:
        paddle.Tensor: The dense attention mask recovered from attn_mask_start_row_indices.
    """
    batch_size, _, max_seq_len = attn_mask_start_row_indices.shape
    base = paddle.arange(max_seq_len, dtype="int32").unsqueeze(1).expand([batch_size, -1, max_seq_len]).unsqueeze(1)
    mask_indices = attn_mask_start_row_indices.unsqueeze(1)

    tril = paddle.tril(
        paddle.ones([max_seq_len, max_seq_len], dtype="bool").expand([batch_size, 1, max_seq_len, max_seq_len])
    )
    attention_mask = paddle.logical_and(base < mask_indices, tril)
    attention_mask = paddle.scale(
        x=attention_mask.astype(dtype),
        scale=1000000.0,
        bias=-1.0,
        bias_after_scale=False,
    )

    return attention_mask
