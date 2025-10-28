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
from paddle.autograd.py_layer import PyLayer

from .utils import repeat_kv

_C_ops = paddle._C_ops


def _get_fa_version():
    """Get the FlashAttention version based on environment flags."""
    if paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]:
        return 2
    return paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]


def _flash_attention_forward_dispatch(
    query,
    key,
    value,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    attention_mask: Optional[paddle.Tensor] = None,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
    softmax_scale=None,
):
    """
    Dispatch FlashAttention forward pass based on version.
    Note: For FlashAttention, seq_k = seq_v is required.
    """
    assert not return_softmax, "return_softmax must be false"

    # Validate sequence length consistency for FlashAttention
    seq_k, seq_v = key.shape[1], value.shape[1]
    assert seq_k == seq_v, f"FlashAttention requires equal sequence lengths: seq_k={seq_k}, seq_v={seq_v}"

    fa_version = _get_fa_version()

    if fa_version == 2:
        # FlashAttention v2 supports custom softmax_scale
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn"):
            out, _, lse, _ = _C_ops.flash_attn(
                query, key, value, fixed_seed_offset, attention_mask, dropout, causal, False, not training, rng_name
            )
        else:
            assert False, "flash_attn_v2 is not supported, may be due to paddle version"
        lse = lse[:, :, : query.shape[1]]
    elif fa_version == 3:
        # FlashAttention v3 supports custom softmax_scale
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3"):
            out, lse = _C_ops.flash_attn_v3(
                query, key, value, None, None, None, None, softmax_scale, causal, -1, -1, 0.0, 1, False, False, 0
            )
        else:
            assert False, "flash_attn_v3 is not supported, may be due to paddle version"

        assert attention_mask is None, "FA3 do not support dense mask(attention_mask)"
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return out, lse


def _flash_attention_backward_dispatch(
    grad_output,
    query,
    key,
    value,
    output,
    lse,
    dropout=0.0,
    attention_mask: Optional[paddle.Tensor] = None,
    causal=False,
    softmax_scale=None,
):
    """
    Dispatch FlashAttention backward pass based on version.
    """
    fa_version = _get_fa_version()

    if fa_version == 2:
        # FlashAttention v2 supports custom softmax_scale
        seed_offset = paddle.zeros(shape=[2], dtype="int64")
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_grad"):
            grad_q, grad_k, grad_v = _C_ops.flash_attn_grad(
                query, key, value, output, lse, seed_offset, attention_mask, grad_output, dropout, causal
            )
        else:
            assert False, "flash_attn_v2_grad is not supported, may be due to paddle version"
    elif fa_version == 3:
        # FlashAttention v3 supports custom softmax_scale
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3_grad"):
            grad_q, grad_k, grad_v = _C_ops.flash_attn_v3_grad(
                query, key, value, output, lse, grad_output, softmax_scale, causal, -1, -1, 0.0, 0
            )
        else:
            assert False, "flash_attn_v3_grad is not supported, may be due to paddle version"
        assert attention_mask is None, "FA3 do not support dense mask(attention_mask)"
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return grad_q, grad_k, grad_v


def _flashmask_attention_forward_dispatch(
    query,
    key,
    value,
    startend_row_indices,
    dropout=0.0,
    causal=False,
    training=True,
    softmax_scale=None,
):
    """
    Dispatch FlashMask attention forward pass.
    FlashMask supports variable sequence lengths through startend_row_indices.
    Note: Only FlashMask v1 doesn't support custom softmax_scale.
    """
    fa_version = _get_fa_version()

    if fa_version == 2:
        # FlashMask v1 doesn't support custom softmax_scale
        if softmax_scale is not None and softmax_scale != 1.0 / (query.shape[-1] ** 0.5):
            print(
                f"Warning: FlashMask v1 doesn't support custom softmax_scale, ignoring provided value: {softmax_scale}"
            )

        output, log_sum_exp = paddle.nn.functional.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            dropout=dropout,
            return_softmax_lse=True,
            training=training,
        )
    else:
        # FlashMask v2 and later support custom softmax_scale
        output, log_sum_exp = paddle.nn.functional.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            training=training,
        )

    return output, log_sum_exp


def _flashmask_attention_backward_dispatch(
    grad_output,
    query,
    key,
    value,
    output,
    lse,
    startend_row_indices,
    dropout=0.0,
    causal=False,
    softmax_scale=None,
):
    """
    Dispatch FlashMask attention backward pass based on version.
    Note: Only FlashMask v1 doesn't support custom softmax_scale.
    """
    fa_version = _get_fa_version()

    if fa_version == 2:
        # FlashMask v1 doesn't support custom softmax_scale
        seed_offset = paddle.zeros(shape=[2], dtype="int64")
        if hasattr(paddle.base.libpaddle.pir.ops, "flashmask_attention_grad"):
            grad_q, grad_k, grad_v = _C_ops.flashmask_attention_grad(
                query, key, value, startend_row_indices, output, lse, seed_offset, grad_output, dropout, causal
            )
        else:
            assert False, "flashmask_attention_grad is not supported, may be due to paddle version"
    elif fa_version == 3:
        # FlashMask v2 supports custom softmax_scale
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flashmask_attention_v2_grad"):
            grad_q, grad_k, grad_v = _C_ops.flashmask_attention_v2_grad(
                query, key, value, output, lse, startend_row_indices, grad_output, softmax_scale, causal
            )
        else:
            assert False, "flashmask_attention_v2_grad is not supported, may be due to paddle version"
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return grad_q, grad_k, grad_v


class FlashMaskSinkPyLayer(PyLayer):
    """
    Custom PyLayer implementing FlashAttention/FlashMask with Sink mechanism.

    The Sink mechanism modifies attention outputs by applying a learned sink parameter
    that affects the attention distribution. This is particularly useful for handling
    attention sinks in long sequences.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        sink,
        startend_row_indices,
        attention_mask: Optional[paddle.Tensor] = None,
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
        """
        Forward pass of FlashMask with Sink mechanism.

        Args:
            query: Query tensor [B, S, H_q, D]
            key: Key tensor [B, S, H_kv, D]
            value: Value tensor [B, S, H_kv, D]
            sink: Sink parameter tensor [H_q]
            startend_row_indices: Optional indices for FlashMask (variable length sequences)
            attention_mask: Dense mask tensor [B, H_q, S, S]
            dropout: Dropout probability
            causal: Whether to apply causal mask
            softmax_scale: Custom softmax scaling factor
        """
        # Input validation
        assert query.ndim == 4, f"Query must be 4D tensor, got {query.ndim}D"
        assert key.ndim == 4, f"Key must be 4D tensor, got {key.ndim}D"
        assert value.ndim == 4, f"Value must be 4D tensor, got {value.ndim}D"
        assert sink.ndim == 1, f"Sink must be 1D tensor, got {sink.ndim}D"

        batch_q, seq_q, num_q_heads, head_dim_q = query.shape
        batch_k, seq_k, num_kv_heads, head_dim_k = key.shape
        batch_v, seq_v, num_kv_heads_v, head_dim_v = value.shape

        # Validate batch dimensions
        assert (
            batch_q == batch_k == batch_v
        ), f"Batch sizes must match: query={batch_q}, key={batch_k}, value={batch_v}"

        # Validate head dimensions
        assert (
            head_dim_q == head_dim_k == head_dim_v
        ), f"Head dimensions must match: query={head_dim_q}, key={head_dim_k}, value={head_dim_v}"
        assert (
            num_kv_heads == num_kv_heads_v
        ), f"Key and value must have same number of heads: key={num_kv_heads}, value={num_kv_heads_v}"

        # Validate GQA compatibility
        assert (
            num_q_heads % num_kv_heads == 0
        ), f"Query heads ({num_q_heads}) must be divisible by key/value heads ({num_kv_heads})"

        # Validate sink parameter
        assert (
            sink.shape[0] == num_q_heads
        ), f"Sink parameter size ({sink.shape[0]}) must match number of query heads ({num_q_heads})"

        # Sequence length validation based on attention type
        if startend_row_indices is None:
            # FlashAttention requires equal sequence lengths
            assert (
                seq_q == seq_k == seq_v
            ), f"FlashAttention requires equal sequence lengths: seq_q={seq_q}, seq_k={seq_k}, seq_v={seq_v}"

        else:
            # FlashMask allows variable sequence lengths, but key and value must match
            assert seq_k == seq_v, f"Key and value sequence lengths must match: seq_k={seq_k}, seq_v={seq_v}"
            assert attention_mask is None, "Flashmask do not support dense mask(attention_mask)"

        # Handle GQA by repeating key/value heads if necessary
        num_attention_heads = query.shape[2]
        num_key_value_heads = key.shape[2]
        num_key_value_groups = num_attention_heads // num_key_value_heads
        if startend_row_indices is None:
            key_states = repeat_kv(key, num_key_value_groups)
            value_states = repeat_kv(value, num_key_value_groups)
        else:
            key_states = key
            value_states = value

        # Choose between FlashAttention and FlashMask based on startend_row_indices
        if startend_row_indices is None:
            # Use standard FlashAttention
            raw_output, lse_original = _flash_attention_forward_dispatch(
                query,
                key_states,
                value_states,
                dropout,
                causal,
                attention_mask=attention_mask,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=softmax_scale,
            )
        else:
            # Use FlashMask attention for variable length sequences
            raw_output, lse_original = _flashmask_attention_forward_dispatch(
                query,
                key_states,
                value_states,
                startend_row_indices,
                dropout,
                causal,
                training=training,
                softmax_scale=softmax_scale,
            )

        # Apply sink mechanism
        origin_dtype = raw_output.dtype
        scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)

        # Reshape tensors for sink computation
        lse_transposed = lse_original.transpose(perm=[0, 2, 1]).unsqueeze(-1)
        sink_reshaped = sink.reshape(shape=[1, 1, -1, 1])

        batch_size, seq_len, num_heads, _ = query.shape
        sink_expanded = sink_reshaped.expand([batch_size, seq_len, num_heads, 1])

        # Compute sink multiplier: 1 / (exp(sink - lse) + 1)
        multiplier = 1 / (paddle.exp(sink_expanded - lse_transposed) + 1)
        final_out = (raw_output * multiplier).to(origin_dtype)

        # Save tensors for backward pass
        ctx.save_for_backward(
            query, key, value, sink, attention_mask, raw_output, lse_original, multiplier, startend_row_indices
        )
        ctx.dropout = dropout
        ctx.causal = causal
        ctx.softmax_scale = scale
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.rng_name = rng_name
        ctx.training = training
        ctx.name = name
        ctx.num_key_value_groups = num_key_value_groups

        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computing gradients for all inputs.
        """
        (
            query,
            key,
            value,
            sink,
            attention_mask,
            raw_output,
            lse_original,
            multiplier,
            startend_row_indices,
        ) = ctx.saved_tensor()

        # Restore context variables
        num_key_value_groups = ctx.num_key_value_groups
        if startend_row_indices is None:
            key_states = repeat_kv(key, num_key_value_groups)
            value_states = repeat_kv(value, num_key_value_groups)
        else:
            key_states = key
            value_states = value

        dropout, causal, scale = ctx.dropout, ctx.causal, ctx.softmax_scale
        fixed_seed_offset, rng_name = ctx.fixed_seed_offset, ctx.rng_name
        training, name = ctx.training, ctx.name

        # Compute gradient w.r.t. raw attention output
        grad_raw_output = (grad_output * multiplier).to(query.dtype)

        # Compute main gradients using appropriate attention backward
        if startend_row_indices is None:
            grad_q_main, grad_k_repeated, grad_v_repeated = _flash_attention_backward_dispatch(
                grad_raw_output,
                query,
                key_states,
                value_states,
                raw_output,
                lse_original,
                dropout=dropout,
                attention_mask=attention_mask,
                causal=causal,
                softmax_scale=scale,
            )
        else:
            grad_q_main, grad_k_repeated, grad_v_repeated = _flashmask_attention_backward_dispatch(
                grad_raw_output,
                query,
                key_states,
                value_states,
                raw_output,
                lse_original,
                startend_row_indices,
                dropout,
                causal,
                scale,
            )

        # Handle GQA: sum gradients across repeated heads
        # Only if grad_k_repeated.shape[2] == num_kv_heads * num_key_value_groups, (kv_head is expanded)
        if num_key_value_groups > 1 and grad_k_repeated.shape[2] == key.shape[2] * num_key_value_groups:
            batch, seq_len, num_kv_heads, head_dim = key.shape
            grad_k_main = grad_k_repeated.reshape([batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]).sum(
                axis=3
            )
            grad_v = grad_v_repeated.reshape([batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]).sum(
                axis=3
            )
        else:
            grad_k_main = grad_k_repeated
            grad_v = grad_v_repeated

        # Compute sink-related gradients
        g_r = paddle.sum(grad_output * raw_output, axis=-1)
        multiplier_for_grad = multiplier.squeeze(-1)
        g_ell = g_r * multiplier_for_grad * (1 - multiplier_for_grad)

        # Gradient w.r.t. sink parameter
        grad_sink_temp = -paddle.sum(g_ell, axis=1)
        grad_sink = grad_sink_temp.sum(axis=0)

        # Compute additional gradients through sink mechanism
        if startend_row_indices is None:
            # Use FlashAttention for computing mu_k (attention between query and key)
            mu_k, lse_k = _flash_attention_forward_dispatch(
                query,
                key_states,
                key_states,
                dropout,
                causal,
                attention_mask=attention_mask,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=scale,
            )
            x = (g_ell.unsqueeze(-1) * query).to(query.dtype)
            _, grad_k_extra_repeated, _ = _flash_attention_backward_dispatch(
                x,
                query,
                key_states,
                key_states,
                mu_k,
                lse_k,
                dropout=dropout,
                attention_mask=attention_mask,
                causal=causal,
                softmax_scale=scale,
            )
        else:
            # Use FlashMask for computing mu_k
            mu_k, lse_k = _flashmask_attention_forward_dispatch(
                query,
                key_states,
                key_states,
                startend_row_indices,
                dropout,
                causal,
                training=training,
                softmax_scale=scale,
            )
            x = (g_ell.unsqueeze(-1) * query).to(query.dtype)
            _, grad_k_extra_repeated, _ = _flashmask_attention_backward_dispatch(
                x, query, key_states, key_states, mu_k, lse_k, startend_row_indices, dropout, causal, scale
            )

        # Additional gradients from sink mechanism
        grad_q_extra = scale * g_ell.unsqueeze(-1) * mu_k

        if num_key_value_groups > 1 and grad_k_extra_repeated.shape[2] == key.shape[2] * num_key_value_groups:
            batch, seq_len, num_kv_heads, head_dim = key.shape
            grad_k_extra_repeated = grad_k_extra_repeated.reshape(
                [batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]
            )
            grad_k_extra = scale * grad_k_extra_repeated.sum(axis=3)
        else:
            grad_k_extra = scale * grad_k_extra_repeated

        # Combine main and extra gradients
        grad_q = grad_q_main + grad_q_extra
        grad_k = grad_k_main + grad_k_extra
        if query.dtype != grad_q.dtype:
            grad_q = grad_q.cast(query.dtype)
        if key.dtype != grad_k.dtype:
            grad_k = grad_k.cast(key.dtype)
        if value.dtype != grad_v.dtype:
            grad_v = grad_v.cast(value.dtype)
        if sink.stop_gradient:
            # Return gradients (number of return values must match forward inputs)
            if startend_row_indices is None:
                return grad_q, grad_k, grad_v, None  # grad_sink
            else:
                return grad_q, grad_k, grad_v, None, None
        else:
            if startend_row_indices is None:
                return grad_q, grad_k, grad_v, grad_sink
            else:
                return grad_q, grad_k, grad_v, grad_sink, None


def sink_attention_forward(
    q,
    k,
    v,
    sink: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    startend_row_indices: Optional[paddle.Tensor] = None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    """
    A unified, high-performance attention implementation with Sink mechanism support.

    This function automatically chooses between FlashAttention and FlashMask based on
    the presence of startend_row_indices:
    - If startend_row_indices is None: Uses standard FlashAttention (requires seq_q = seq_k = seq_v)
    - If startend_row_indices is provided: Uses FlashMask attention (supports variable length sequences)

    The Sink mechanism modifies attention outputs by applying a learned sink parameter
    that affects the attention distribution, which is useful for handling attention sinks
    in long sequences.

    Also supports GQA (Grouped-Query Attention) where the number of key/value heads
    is smaller than the number of query heads.

    Args:
        q: Query tensor with shape [batch_size, seq_len, num_q_heads, head_dim]
        k: Key tensor with shape [batch_size, seq_len, num_kv_heads, head_dim]
        v: Value tensor with shape [batch_size, seq_len, num_kv_heads, head_dim]
        sink: Sink parameter tensor with shape [num_q_heads]
        attention_mask: Dense mask, only supported for FA2
        startend_row_indices: Optional tensor for FlashMask attention to handle variable length sequences
        dropout_p: Dropout probability (default: 0.0)
        softmax_scale: Custom softmax scaling factor (default: 1/sqrt(head_dim))
                      Note: Only FlashMask v1 doesn't support custom softmax_scale
        causal: Whether to apply causal masking (default: False)

    Returns:
        Attention output tensor with shape [batch_size, seq_len, num_q_heads, head_dim]

    Notes:
        - For standard FlashAttention: seq_q = seq_k = seq_v is required
        - FlashMask allows variable sequence lengths, but key and value lengths must match
        - Only FlashMask v1 doesn't support custom softmax_scale parameter
        - The function automatically handles GQA by repeating key/value heads when necessary
        - Input tensors must be 4D with proper shape validation
        - Sink parameter size must match the number of query heads

    Raises:
        AssertionError: If input tensor shapes are incompatible or requirements are not met
        ValueError: If unsupported FlashAttention version is detected
    """
    return FlashMaskSinkPyLayer.apply(
        q,
        k,
        v,
        sink,
        startend_row_indices,
        attention_mask=attention_mask,
        dropout=dropout_p,
        causal=causal,
        return_softmax=False,
        softmax_scale=softmax_scale,
    )
