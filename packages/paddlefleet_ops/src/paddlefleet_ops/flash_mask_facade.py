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


def get_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version for the given head dims.

    Dispatch rules:
      * XPU device -> FA2.
      * Otherwise, respect ``FLAGS_flash_attn_version`` by default.
      * If ``fa_version == 3`` and deterministic is required, FA3 only supports
        ``head_dim <= 128``. For ``head_dim > 128``, fall back to FA2.
      * FA4 is only used when both ``hdim_ok`` and ``mask_ok`` hold:

        - ``hdim_ok``: one of
          * ``head_dim <= 128`` and ``head_dim_v <= 128``
          * ``head_dim == 192`` and ``head_dim_v == 128``
          * ``head_dim == 256`` and ``head_dim_v == 256``
        - ``mask_ok``: ``startend_row_indices is None`` or
          ``startend_row_indices.shape[-1] != 4``

        When ``startend_row_indices`` is not provided (``None``), ``mask_ok``
        is treated as ``True`` -- this covers the ``flash_attention`` path
        which has no mask tensor. Aligned with flash-attention ``interface.py``.

    Args:
        head_dim: Query/Key head dim (always equal).
        head_dim_v: Value head dim. Defaults to ``head_dim`` when not provided.
        startend_row_indices: FlashMask indices tensor. Pass ``None`` (default)
            for the plain ``flash_attention`` path where no mask check is needed.

    Returns:
        The FlashAttention version to use (2, 3 or 4).
    """
    if "xpu" in paddle.get_device():
        return 2

    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]

    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]

    if fa_version == 3:
        if deterministic and head_dim > 128:
            return 2

    if fa_version == 4:
        _head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        fa4_hdim_ok = (
            (head_dim <= 128 and _head_dim_v <= 128)
            or (head_dim == 192 and _head_dim_v == 128)
            or (head_dim == 256 and _head_dim_v == 256)
        )
        fa4_mask_ok = (
            startend_row_indices is None or startend_row_indices.shape[-1] != 4
        )
        if not (fa4_hdim_ok and fa4_mask_ok):
            return 2

    return fa_version


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
    learnable_sink: paddle.Tensor | None = None,
):
    if use_varlen:
        assert (
            "use_varlen" in inspect.signature(_flashmask_attention).parameters
        ), "The flash_mask installed does not support use_varlen"

    if learnable_sink is not None:
        if (
            "learnable_sink"
            not in inspect.signature(_flashmask_attention).parameters
        ):
            raise NotImplementedError(
                "learnable_sink (softmax sink) requires FA4 (cute backend); the "
                "installed flash_mask / current device (e.g. H-card fa2/fa3) does "
                "not support it. Disable the attention sink or run on a "
                "FA4-capable device."
            )

    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]

    fa_version = get_fa_version(q_head_dim, v_head_dim, startend_row_indices)

    need_value_padding = (
        not (fa_version == 4 and q_head_dim == 192 and v_head_dim == 128)
    ) and q_head_dim != v_head_dim

    if need_value_padding:
        value_padding = paddle.zeros(
            [bsz, q_len, value.shape[2], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    extra_kwargs = {}
    if use_varlen:
        # use_varlen is no longer used and will be removed soon.
        extra_kwargs["use_varlen"] = True
    if learnable_sink is not None:
        extra_kwargs["learnable_sink"] = learnable_sink

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
        **extra_kwargs,
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

    # startend_row_indices is None
    fa_version = get_fa_version(q_head_dim, v_head_dim)

    need_value_padding = (
        not (fa_version == 4 and q_head_dim == 192 and v_head_dim == 128)
    ) and q_head_dim != v_head_dim

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
    "get_fa_version",
]
