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
Fusion operators
"""
import paddle
from paddle.incubate.nn.functional import fused_rms_norm_ext
from paddle.incubate.nn.functional import fused_rotary_position_embedding as fused_rope
from paddle.incubate.nn.functional import swiglu as fused_swiglu

from .common_fusion_ops import Linear, matmul

if paddle.device.is_compiled_with_custom_device("npu"):
    from .npu_fusion_ops import npu_cal_aux_loss_func as cal_aux_loss
else:
    from paddle.incubate.nn.functional import cal_aux_loss

__all__ = [
    "fused_rope",
    "fused_swiglu",
    "fused_rms_norm_ext",
    "Linear",
    "matmul",
    "cal_aux_loss",
]


def fusion_flash_attention(
    q,
    k,
    v,
    training_mode,
    attention_probs_dropout_prob,
    use_sparse_flash_attn,
    attention_mask=None,
    attn_mask_start_row_indices=None,
    seq_length=None,
    use_var_len_flash_attn=False,
    rr_flash_attn=None,
):
    """
    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        v (Tensor): Value tensor.
        training_mode (bool): Whether in training mode.
        attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
        use_sparse_flash_attn (bool): Whether to use sparse flash attention.
        attention_mask (Tensor, optional): Attention mask. Defaults to None.
        attn_mask_start_row_indices (Tensor, optional): Start row indices for attention mask. Defaults to None.
        seq_length (int, optional): Sequence length. Defaults to None.
        use_var_len_flash_attn (bool, optional): Whether to use variable length flash attention. Defaults to False.
        rr_flash_attn (bool, optional): Whether to use round-robin flash attention. Defaults to None.

    Returns:
        Tensor: Output tensor after applying fusion flash attention.
    """
    from .common_fusion_ops import _fusion_flash_attention

    return _fusion_flash_attention(
        q,
        k,
        v,
        training_mode=training_mode,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        use_sparse_flash_attn=use_sparse_flash_attn,
        attention_mask=attention_mask,
        attn_mask_start_row_indices=attn_mask_start_row_indices,
        rr_flash_attn=rr_flash_attn,
    )
