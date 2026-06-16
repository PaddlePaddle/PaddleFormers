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

import inspect
from functools import partial

import paddle

from . import is_flash_mask_available

if is_flash_mask_available():
    try:
        from .flash_mask import (
            flash_attention as _flash_attention,
            flashmask_attention as _flashmask_attention,
        )
    except (ImportError, ModuleNotFoundError):
        from .flash_mask.cute.interface import (
            flash_attention as _flash_attention,
            flashmask_attention as _flashmask_attention,
        )
else:
    from paddle.nn.functional.flash_attention import (
        flash_attention as _flash_attention,
        flashmask_attention as _flashmask_attention,
    )


def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
    use_varlen: bool = False,
):
    if use_varlen:
        assert (
            "use_varlen" in inspect.signature(_flashmask_attention).parameters
        ), "The flash_mask installed does not support use_varlen"

    if "xpu" in paddle.get_device():
        fa_version = 2
    else:
        fa_version = paddle.base.framework.get_flags(
            ["FLAGS_flash_attn_version"]
        )["FLAGS_flash_attn_version"]

    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]
    is_fa4_support_d192_dv128 = (
        "use_varlen" in inspect.signature(_flashmask_attention).parameters
    )
    need_value_padding = (
        not (
            fa_version == 4
            and use_varlen
            and is_fa4_support_d192_dv128
            and q_head_dim == 192
            and v_head_dim == 128
        )
    ) and q_head_dim != v_head_dim

    if need_value_padding:
        value_padding = paddle.zeros(
            [bsz, q_len, value.shape[2], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    if use_varlen:
        flashmask_attention_func = partial(
            _flashmask_attention, use_varlen=True
        )
    else:
        flashmask_attention_func = _flashmask_attention

    outs = flashmask_attention_func(
        query=query,
        key=key,
        value=value,
        startend_row_indices=startend_row_indices.clone(),
        dropout=dropout,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=return_softmax_lse,
        return_seed_offset=return_seed_offset,
        fixed_seed_offset=fixed_seed_offset,
        rng_name=rng_name,
        training=training,
        name=name,
        softmax_scale=softmax_scale,
        block_mask=block_mask,
    )

    if return_softmax_lse:
        attn_out, lse = outs
        lse = lse.reshape([bsz, q_len])
    else:
        attn_out = outs

    if need_value_padding:
        attn_out = attn_out[..., :v_head_dim]

    attn_out = attn_out.reshape([bsz, q_len, num_heads, v_head_dim])

    if return_softmax_lse:
        return [attn_out, lse]
    else:
        return attn_out


def flash_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
    softmax_scale=None,
):
    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]
    need_value_padding = q_head_dim != v_head_dim

    if need_value_padding:
        value_padding = paddle.zeros(
            [bsz, q_len, value.shape[2], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    attn_output, softmax_result = _flash_attention(
        query=query,
        key=key,
        value=value,
        dropout=dropout,
        causal=causal,
        return_softmax=return_softmax,
        fixed_seed_offset=fixed_seed_offset,
        rng_name=rng_name,
        training=training,
        name=name,
        softmax_scale=softmax_scale,
    )

    if need_value_padding:
        attn_output = attn_output[..., :v_head_dim]

    attn_output = attn_output.reshape([bsz, q_len, num_heads, v_head_dim])

    return attn_output, softmax_result


__all__ = [
    "flashmask_attention",
    "flash_attention",
]
