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

import inspect

import paddle
from paddle import distributed as dist
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet
from paddle.nn.functional.flash_attention import flashmask_attention

_flash_mask_available = False
try:
    if (
        paddle.cuda.is_available()
        and paddle.cuda.get_device_capability()[0] == 10
    ):
        from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
            FlashMaskInfoPaddle,
        )
        from paddlefleet_ops.flash_mask.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        _flash_mask_available = True
except (ImportError, AttributeError):
    _flash_mask_available = False


def mark_context_parallel_parameter_disable_scale_grad(param_or_layer):
    """
    Mark parameters or layers to disable context parallel gradient scaling.

    This function sets the attribute `context_parallel_disable_scale_grad` to `True` for the given parameter,
    tensor, or layer. When set, this flag indicates that the specified parameter or layer should not have
    its gradient scaled during context parallel training.
    - If a `paddle.nn.Layer` is provided, both its `weight` and (if present) `bias` will be marked.
    - If a `paddle.base.framework.Parameter` or `paddle.Tensor` is provided, it will be marked directly.
    - Raises a `TypeError` if the input is not a supported type.
    Args:
        param_or_layer (paddle.nn.Layer or paddle.base.framework.Parameter or paddle.Tensor):
            The parameter, tensor, or layer to mark as disabling context parallel gradient scaling.
    Raises:
        TypeError: If `param_or_layer` is not a `Parameter`, `Tensor`, or `Layer`.
    Example:
        >>> mark_context_parallel_parameter_disable_scale_grad(layer)
        >>> mark_context_parallel_parameter_disable_scale_grad(param)
    """

    if isinstance(param_or_layer, paddle.nn.Layer):
        param_or_layer.weight.context_parallel_disable_scale_grad = True
        if hasattr(param_or_layer, "bias") and param_or_layer.bias is not None:
            param_or_layer.bias.context_parallel_disable_scale_grad = True
    elif isinstance(
        param_or_layer, (paddle.base.framework.Parameter, paddle.Tensor)
    ):
        param_or_layer.context_parallel_disable_scale_grad = True
    else:
        raise TypeError(
            f"param should be 'Parameter' or 'Tensor' or 'Layer', but received {type(param_or_layer)}"
        )


def context_parallel_parameter_disable_scale_grad(param):
    """
    Check whether context parallel gradient scaling is disabled for the parameter or tensor.
    Returns the value of the `context_parallel_disable_scale_grad` attribute for the given parameter or tensor.
    If the attribute is not set, returns `False` by default.
    Args:
        param (paddle.base.framework.Parameter or paddle.Tensor):
            The parameter or tensor to check.
    Returns:
        bool: True if context parallel gradient scaling is disabled, False otherwise.
    Example:
        >>> if context_parallel_parameter_disable_scale_grad(param):
        ...     # Handle parameter that should not have its gradient scaled
        ...     pass
    """
    return getattr(param, "context_parallel_disable_scale_grad", False)


def scatter_balance(input_tensor, group=None, axis=0):
    """
    Evenly split input tensor along the specified axis across model parallel ranks.
    This function implements balanced scattering by taking chunks from both ends
    of the tensor to ensure load balancing across ranks.
    Args:
        input_tensor (paddle.Tensor): Input tensor to be scattered
        group (paddle.distributed.Group, optional): Communication group.
            If None, uses model parallel group from fleet
        axis (int, optional): Axis along which to scatter. Defaults to 0
    Returns:
        paddle.Tensor: Scattered tensor chunk for current rank
    Note:
        This API is different from distributed.scatter - it performs balanced
        splitting by taking chunks from both ends of the sequence.
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    rank = group.rank
    seq_len = input_tensor.shape[axis]

    # Ensure sequence length is divisible by parallelism * 2 for balanced splitting
    assert seq_len % (parallelism * 2) == 0, (
        f"Input sequence length {seq_len} can't be divided exactly by sequence parallelism * 2 {parallelism * 2}"
    )

    interval = seq_len // parallelism // 2
    total_len = input_tensor.shape[axis]

    # Take chunk from the beginning
    chunk_start = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[interval * rank],
        ends=[interval * (rank + 1)],
    )

    # Take chunk from the end (in reverse order)
    chunk_end = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[total_len - interval * (rank + 1)],
        ends=[total_len - interval * rank],
    )

    # Concatenate chunks
    result = paddle.concat([chunk_start, chunk_end], axis=axis)

    # Use assign to free the memory of the whole input tensor to avoid OOM
    # since slice uses stride and maintains reference to original tensor
    result = paddle.assign(result)
    return result


def all_gather_balance(input_tensor, group=None, axis=0):
    """
    All-gather operation with balanced reconstruction.
    This function performs all-gather to reconstruct the original tensor
    from balanced scattered chunks.
    Args:
        input_tensor (paddle.Tensor): Input tensor chunk
        group (paddle.distributed.Group, optional): Communication group
        axis (int, optional): Axis along which to gather. Defaults to 0
    Returns:
        paddle.Tensor: Reconstructed full tensor
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    # Split input into two halves (start and end chunks)
    chunk_start, chunk_end = paddle.split(input_tensor, 2, axis=axis)

    if axis == 0:
        # Handle axis=0 case with optimized memory layout
        output_shape_start = list(chunk_start.shape)
        output_shape_start[axis] = output_shape_start[axis] * parallelism

        gathered_start = paddle.empty(
            shape=output_shape_start, dtype=input_tensor.dtype
        )
        dist.stream.all_gather(
            gathered_start, chunk_start, group=group, use_calc_stream=True
        )

        # Gather end chunks
        gathered_end_list = [
            paddle.empty(chunk_end.shape, dtype=input_tensor.dtype)
            for _ in range(parallelism)
        ]
        dist.stream.all_gather(
            gathered_end_list, chunk_end, group=group, use_calc_stream=True
        )

        # Reverse the end chunks to reconstruct original order
        gathered_end_list.reverse()

        result = paddle.concat([gathered_start, *gathered_end_list], axis=axis)
        return result
    else:
        # Handle other axes
        gathered_start_list = [
            paddle.empty(chunk_start.shape, dtype=input_tensor.dtype)
            for _ in range(parallelism)
        ]
        dist.stream.all_gather(
            gathered_start_list, chunk_start, group=group, use_calc_stream=True
        )

        gathered_end_list = [
            paddle.empty(chunk_end.shape, dtype=input_tensor.dtype)
            for _ in range(parallelism)
        ]
        dist.stream.all_gather(
            gathered_end_list, chunk_end, group=group, use_calc_stream=True
        )

        # Reverse the end chunks
        gathered_end_list = gathered_end_list[::-1]

        result = paddle.concat(
            gathered_start_list + gathered_end_list, axis=axis
        )
        return result


def reduce_scatter_any_axis(input_tensor, axis, group=None):
    """
    Reduce-scatter operation along any axis.
    Performs element-wise reduction (sum) across ranks and scatters the result
    so each rank gets a portion of the reduced tensor.
    Args:
        input_tensor (paddle.Tensor): Input tensor to reduce and scatter
        axis (int): Axis along which to perform reduce-scatter
        group (paddle.distributed.Group, optional): Communication group
    Returns:
        paddle.Tensor: Reduced and scattered tensor chunk
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % parallelism == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be ",
        f"divided exactly by context parallelism {parallelism}",
    )

    if axis == 0:
        # Optimized path for axis=0
        output_shape = list(input_tensor.shape)
        output_shape[0] = output_shape[0] // parallelism

        output = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
        dist.stream.reduce_scatter(
            output,
            input_tensor,
            op=dist.ReduceOp.SUM,
            group=group,
            use_calc_stream=True,
        )
        return output
    else:
        # General case for other axes using alltoall
        input_chunks = paddle.split(input_tensor, parallelism, axis=axis)

        output_buffers = [
            paddle.empty(input_chunks[0].shape, dtype=input_tensor.dtype)
            for _ in range(parallelism)
        ]

        dist.stream.alltoall(
            output_buffers, input_chunks, group=group, use_calc_stream=True
        )

        # Sum the received chunks
        result = paddle.stack(output_buffers, axis=0).sum(axis=0)
        return result


def reduce_scatter_any_axis_balance(input_tensor, axis, group=None):
    """
    Balanced reduce-scatter operation along any axis.
    Similar to reduce_scatter_any_axis but uses balanced splitting strategy
    by processing chunks from both ends of the tensor.
    Args:
        input_tensor (paddle.Tensor): Input tensor to reduce and scatter
        axis (int): Axis along which to perform reduce-scatter
        group (paddle.distributed.Group, optional): Communication group
    Returns:
        paddle.Tensor: Reduced and scattered tensor chunk with balanced distribution
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % (parallelism * 2) == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be ",
        f"divided exactly by context parallelism * 2 {parallelism * 2}",
    )

    # Split input into two halves
    input_start, input_end = paddle.split(input_tensor, 2, axis=axis)

    # Split each half across ranks
    chunks_start = paddle.split(input_start, parallelism, axis=axis)
    chunks_end = paddle.split(input_end, parallelism, axis=axis)

    # Reverse end chunks for balanced distribution
    chunks_end = chunks_end[::-1]

    # Combine corresponding start and end chunks
    combined_chunks = [
        paddle.concat([start_chunk, end_chunk], axis=axis)
        for start_chunk, end_chunk in zip(chunks_start, chunks_end)
    ]

    # Perform alltoall communication
    output_buffers = [
        paddle.empty(combined_chunks[0].shape, dtype=input_tensor.dtype)
        for _ in range(parallelism)
    ]

    dist.stream.alltoall(
        output_buffers, combined_chunks, group=group, use_calc_stream=True
    )

    # Sum the received chunks
    result = paddle.stack(output_buffers, axis=0).sum(axis=0)
    return result


# ===========================================================================
# Contiguous CP primitives (rank r owns [r*chunk, (r+1)*chunk])
# ===========================================================================


def scatter_contiguous(input_tensor, group=None, axis=0):
    """Contiguous scatter: rank r gets slice [r*chunk, (r+1)*chunk] along axis."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    rank = group.rank
    chunk_size = input_tensor.shape[axis] // nranks
    result = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[rank * chunk_size],
        ends=[(rank + 1) * chunk_size],
    )
    return paddle.assign(result)


def all_gather_contiguous(input_tensor, group=None, axis=0):
    """Contiguous all-gather: concatenate all ranks' local tensors in rank order."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    if axis == 0:
        shape = list(input_tensor.shape)
        shape[0] *= nranks
        gathered = paddle.empty(shape=shape, dtype=input_tensor.dtype)
        dist.stream.all_gather(
            gathered,
            input_tensor.contiguous(),
            group=group,
            use_calc_stream=True,
        )
        return gathered
    else:
        tensor_list = [
            paddle.empty(input_tensor.shape, dtype=input_tensor.dtype)
            for _ in range(nranks)
        ]
        dist.stream.all_gather(
            tensor_list,
            input_tensor.contiguous(),
            group=group,
            use_calc_stream=True,
        )
        return paddle.concat(tensor_list, axis=axis)


def reduce_scatter_contiguous(input_tensor, axis, group=None):
    """Contiguous reduce-scatter: reduce_scatter for axis=0, alltoall+sum otherwise."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    if axis == 0:
        output_shape = list(input_tensor.shape)
        output_shape[0] //= nranks
        output = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
        dist.stream.reduce_scatter(
            output,
            input_tensor.contiguous(),
            op=dist.ReduceOp.SUM,
            group=group,
            use_calc_stream=True,
        )
        return output
    else:
        chunks = paddle.split(input_tensor, nranks, axis=axis)
        bufs = [
            paddle.empty(chunks[0].shape, dtype=input_tensor.dtype)
            for _ in range(nranks)
        ]
        dist.stream.alltoall(
            bufs,
            [c.contiguous() for c in chunks],
            group=group,
            use_calc_stream=True,
        )
        return (
            paddle.stack(bufs).cast("float32").sum(0).cast(input_tensor.dtype)
        )


class ContextParallelScatterOp(PyLayer):
    """
    Context parallel scatter operation using PyLayer for automatic differentiation.
    Forward: Scatter input tensor (balanced or contiguous based on mode)
    Backward: All-gather gradients (inverse of forward scatter)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis=0, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "ScatterOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return scatter_contiguous(input_tensor, group=group, axis=axis)
        return scatter_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return all_gather_contiguous(
                grad_output, group=ctx.group, axis=ctx.axis
            )
        return all_gather_balance(grad_output, axis=ctx.axis, group=ctx.group)


class ContextParallelGatherOp(PyLayer):
    """
    Context parallel gather operation using PyLayer for automatic differentiation.
    Forward: All-gather input tensor (balanced or contiguous based on mode)
    Backward: Scatter gradients (inverse of forward gather)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis=0, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "GatherOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return scatter_contiguous(
                grad_output, group=ctx.group, axis=ctx.axis
            )
        return scatter_balance(grad_output, axis=ctx.axis, group=ctx.group)


class ContextParallelAllGatherOp(PyLayer):
    """
    Context parallel all-gather operation with gradient reduction.
    Forward: All-gather input tensor (balanced or contiguous based on mode)
    Backward: Reduce-scatter gradients (sum + scatter)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "AllGatherOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return reduce_scatter_contiguous(
                grad_output, axis=ctx.axis, group=ctx.group
            )
        return reduce_scatter_any_axis_balance(
            grad_output, axis=ctx.axis, group=ctx.group
        )


def preprocess_index(
    startend_row_indices, chunk_id, seq_blocksize, max_seqlen_q
):
    """
    Preprocess startend row indices for a single chunk.
    Adjusts the startend_row_indices relative to the chunk's starting position and
    clips them to valid range.
    Args:
        startend_row_indices (paddle.Tensor): Original startend row indices
        chunk_id (int): ID of the current chunk
        seq_blocksize (int): Size of each sequence block
        max_seqlen_q (int): Maximum sequence length for queries
    Returns:
        paddle.Tensor: Preprocessed row indices
    """
    rows_min = chunk_id * seq_blocksize
    adjusted_indices = startend_row_indices - rows_min
    clipped_indices = paddle.clip(adjusted_indices, min=0, max=max_seqlen_q)
    return clipped_indices


def preprocess_index_dual_chunks(
    startend_row_indices,
    chunk_id_first,
    chunk_id_second,
    seq_blocksize,
    max_seqlen_q,
):
    """
    Preprocess row indices for dual chunks (DualChunkSwap strategy).
    This function handles the index preprocessing for the balanced dual-chunk
    strategy where each rank processes chunks from both ends of the sequence.
    Args:
        startend_row_indices (paddle.Tensor): Original row indices
        chunk_id_first (int): ID of the first chunk
        chunk_id_second (int): ID of the second chunk
        seq_blocksize (int): Size of each sequence block
        max_seqlen_q (int): Maximum sequence length for queries
    Returns:
        paddle.Tensor: Preprocessed row indices for dual chunks
    """
    # Calculate starting positions for both chunks
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    # Process first chunk indices
    indices_first = startend_row_indices - rows_min_first
    indices_first = paddle.clip(indices_first, min=0, max=max_seqlen_q)

    # Process second chunk indices
    indices_second = startend_row_indices - rows_min_second
    indices_second = paddle.clip(indices_second, min=0, max=max_seqlen_q)

    # Offset second chunk indices to avoid overlap
    indices_second = paddle.where(
        indices_second != 0, indices_second + max_seqlen_q, indices_second
    )

    # Combine indices from both chunks
    combined_indices = paddle.maximum(indices_first, indices_second)
    return combined_indices


def cp_flashmask_allgatherkv_balance_forward(
    query, key, value, startend_row_indices, group, causal, is_training
):
    """
    Forward pass of context parallel flashmask attention with balanced all-gather strategy.
    This function implements the forward pass of flash attention with context parallelism
    using the DualChunkSwap strategy for load balancing.
    Args:
        query (paddle.Tensor): Query tensor with shape [batch, seq_len/n, num_heads, head_dim]
        key (paddle.Tensor): Key tensor with shape [batch, seq_len/n, num_heads, head_dim]
        value (paddle.Tensor): Value tensor with shape [batch, seq_len/n, num_heads, head_dim]
        startend_row_indices (paddle.Tensor): Row indices for attention mask
        group (paddle.distributed.Group): Communication group
        causal (bool): Whether to use causal attention
        is_training (bool): Whether in training mode
    Returns:
        tuple: (output, log_sum_exp, processed_indices, fa_version)
            ``fa_version`` is the effective FlashAttention version actually
            used by the forward kernel and must be passed to the backward
            counterpart to keep fwd/bwd consistent.
    """
    paddle.base.core.nvprof_nvtx_push(
        "cp_flashmask_allgatherkv_balance_forward"
    )

    rank = group.rank
    cp_size = group.world_size

    # All-gather key tensors across context parallel ranks
    key_gathered = all_gather_balance(key, axis=1, group=group)

    # All-gather value tensors across context parallel ranks
    value_gathered = all_gather_balance(value, axis=1, group=group)

    # Calculate sequence block size for dual-chunk strategy
    seq_blocksize = query.shape[1] // 2

    # Preprocess indices for dual-chunk strategy
    startend_row_indices = preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )

    # Perform flashmask attention with startend_row_indices
    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    # Apply deterministic override here so forward and backward use the same
    # effective fa_version (mirrors backward's previous logic and the
    # framework flashmask_attention's internal deterministic fallback).
    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]
    if "block_mask" in inspect.signature(flashmask_attention).parameters:
        if deterministic and query.shape[-1] > 128:
            fa_version = 2
    elif deterministic:
        fa_version = 2

    if fa_version == 4 and _flash_mask_available:
        output, log_sum_exp = _flash_attn_fwd(
            query,
            key_gathered,
            value_gathered,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
        )
    else:
        output, log_sum_exp = flashmask_attention(
            query,
            key_gathered,
            value_gathered,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True,
            training=is_training,
        )

    paddle.base.core.nvprof_nvtx_pop()
    return output, log_sum_exp, startend_row_indices, fa_version


def cp_flashmask_allgatherkv_balance_backward(
    query,
    key,
    value,
    startend_row_indices,
    output,
    log_sum_exp,
    output_grad,
    group,
    causal,
    fa_version: int,
):
    """
    Backward pass of context parallel flashmask attention with balanced all-gather strategy.
    This function implements the backward pass of flashmask attention with context parallelism,
    computing gradients for query, key, and value tensors.
    Args:
        query (paddle.Tensor): Query tensor
        key (paddle.Tensor): Key tensor
        value (paddle.Tensor): Value tensor
        startend_row_indices (paddle.Tensor): Processed startend_row_indices
        output (paddle.Tensor): Forward pass output
        log_sum_exp (paddle.Tensor): Log-sum-exp from forward pass
        output_grad (paddle.Tensor): Gradient of output
        group (paddle.distributed.Group): Communication group
        causal (bool): Whether causal attention was used
        fa_version (int): FlashAttention version that was actually used by the
            forward kernel. Must be propagated from the forward call to keep
            fwd/bwd consistent.
    Returns:
        tuple: (query_grad, key_grad, value_grad)
    """
    paddle.base.core.nvprof_nvtx_push(
        "cp_flashmask_allgatherkv_balance_backward"
    )

    # All-gather key and value tensors (same as forward pass)
    key_gathered = all_gather_balance(key, axis=1, group=group)
    value_gathered = all_gather_balance(value, axis=1, group=group)

    if fa_version == 2:
        # Create seed offset tensor (required for gradient computation)
        seed_offset = paddle.zeros(
            shape=[query.shape[1], query.shape[2]], dtype=paddle.int64
        )

        # Compute gradients using flashmask attention backward pass
        query_grad, key_grad_gathered, value_grad_gathered = (
            paddle._C_ops.flashmask_attention_grad(
                query,
                key_gathered,
                value_gathered,
                startend_row_indices,
                output,
                log_sum_exp,
                seed_offset,
                output_grad,
                0.0,  # dropout probability
                causal,
            )
        )
    elif fa_version == 3:
        sig_params = inspect.signature(flashmask_attention).parameters
        if "group" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                    0,  # rank
                    1,  # nranks
                )
            )
        elif "block_mask" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
        else:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
    elif fa_version == 4 and _flash_mask_available:
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None
        query_grad, key_grad_gathered, value_grad_gathered = _flash_attn_bwd(
            query,
            key_gathered,
            value_gathered,
            output,
            output_grad,
            log_sum_exp,
            flashmask_info,
            causal=causal,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ],
        )
    else:
        raise ValueError(
            f"FlashAttention version {fa_version} is not supported."
        )

    # Reduce-scatter key and value gradients
    key_grad = reduce_scatter_any_axis_balance(
        key_grad_gathered, axis=1, group=group
    )
    value_grad = reduce_scatter_any_axis_balance(
        value_grad_gathered, axis=1, group=group
    )

    paddle.base.core.nvprof_nvtx_pop()
    return query_grad, key_grad, value_grad


def scatter_with_padding(input_tensor, num_pad, axis, group):
    """scatter_with_padding"""
    cp_degree = group.nranks
    cp_rank = group.rank

    total_num = input_tensor.shape[axis]
    avg_num = (total_num + num_pad) // cp_degree

    split_sections = []
    cnt = 0
    rank_idx = 0
    rank_pad = 0
    for _ in range(0, cp_degree):
        if cnt + avg_num < total_num:
            split_sections.append(avg_num)
        elif cnt < total_num:
            split_sections.append(total_num - cnt)
            rank_pad = avg_num - total_num + cnt
        else:
            break
        cnt += avg_num
        rank_idx += 1

    if cp_rank < rank_idx:
        list_of_res = paddle.split(input_tensor, num_or_sections=split_sections)
        cur_res = list_of_res[cp_rank]
        if rank_pad > 0 and cp_rank == rank_idx - 1:
            pad_list = [0 for _ in range(0, input_tensor.ndim * 2)]
            pad_list[axis * input_tensor.ndim * 2 + 1] = rank_pad
            cur_res = paddle.nn.functional.pad(
                cur_res, pad_list, mode="constant", value=0
            )
    else:
        shape = input_tensor.shape
        shape[axis] = avg_num
        cur_res = paddle.zeros(shape, input_tensor.dtype)
        cur_res.stop_gradient = False
    return cur_res


def all_gather_without_padding(input_tensor, num_pad, axis, group):
    """all_gather_without_padding"""
    output_shape = list(input_tensor.shape)
    output_shape[axis] = output_shape[axis] * group.nranks
    output_tensor = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
    dist.stream.all_gather(output_tensor, input_tensor, group)
    if num_pad > 0:
        pad_start = output_tensor.shape[axis] - num_pad
        output_tensor = paddle.slice(
            output_tensor, axes=[axis], starts=[0], ends=[pad_start]
        )
    return output_tensor


class ContextParallelNormalScatter(PyLayer):
    """ContextParallelNormalScatter"""

    @staticmethod
    def forward(ctx, input_tensor, num_pad, axis=0):
        """forward"""
        ctx.axis = axis
        hcg = fleet.get_hybrid_communicate_group()
        cp_degree = hcg.get_context_parallel_world_size()

        if cp_degree == 1:
            return input_tensor.clone()

        group = hcg.get_context_parallel_group()
        ctx.group = group
        ctx.num_pad = num_pad
        ctx.axis = axis

        return scatter_with_padding(input_tensor, num_pad, axis, ctx.group)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        if ctx.group.nranks == 1:
            return grad_output.clone()

        return all_gather_without_padding(
            grad_output, ctx.num_pad, ctx.axis, ctx.group
        )


class ContextParallelNormalGather(PyLayer):
    """ContextParallelNormalGather"""

    @staticmethod
    def forward(ctx, input_tensor, num_pad, axis=0):
        """forward"""
        ctx.axis = axis
        hcg = fleet.get_hybrid_communicate_group()
        cp_degree = hcg.get_context_parallel_world_size()
        group = hcg.get_context_parallel_group()
        ctx.group = group
        ctx.num_pad = num_pad

        if cp_degree == 1:
            return input_tensor.clone()

        return all_gather_without_padding(input_tensor, num_pad, axis, group)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        if ctx.group.nranks == 1:
            return grad_output.clone()

        return scatter_with_padding(
            grad_output, ctx.num_pad, ctx.axis, ctx.group
        )


class FlashMaskContextParallel(PyLayer):
    """
    FlashMask attention with context parallelism implementation.
    This class implements flashmask attention with context parallelism (CP) using PyLayer
    for automatic differentiation. CP partitions tensors along the sequence dimension
    to enable long-context LLMs in a distributed fashion.
    The implementation uses the DualChunkSwap strategy to ensure load balancing
    across CP ranks by processing chunks from both ends of the sequence.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        mode="allgather_kv",
    ):
        """
        Forward pass of FlashMask attention with context parallelism.
        Args:
            ctx: Context object for saving information for backward pass
            query (paddle.Tensor): Query tensor, pre-divided by CP size
            key (paddle.Tensor): Key tensor, pre-divided by CP size
            value (paddle.Tensor): Value tensor, pre-divided by CP size
            startend_row_indices (paddle.Tensor): Row indices for attention mask
            fixed_seed_offset (paddle.Tensor, optional): Fixed seed offset for dropout
            dropout (float): Dropout probability
            causal (bool): Whether to use causal attention
            training (bool): Whether in training mode
            mode (str): Attention mode, currently supports "allgather_kv"
        Returns:
            paddle.Tensor: Attention output
        Raises:
            NotImplementedError: If dropout > 0.0 or causal=True
            AssertionError: If query sequence length is not divisible by 2
        """
        # Validate input parameters
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in FlashMask context parallel yet."
            )

        if causal:
            raise NotImplementedError(
                "FlashMaskContextParallel does not support causal=True yet."
            )

        if fixed_seed_offset is not None:
            raise NotImplementedError("Fixed seed offset is not supported yet.")

        # Get communication group
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

        # Validate query sequence length for DualChunkSwap strategy
        assert query.shape[1] % 2 == 0, (
            f"Query sequence length must be divisible by 2. "
            f"FlashMaskContextParallel uses DualChunkSwap strategy for load balancing. "
            f"Current query sequence length: {query.shape[1]}"
        )

        # Perform forward pass
        output, log_sum_exp, startend_row_indices, fa_version = (
            cp_flashmask_allgatherkv_balance_forward(
                query, key, value, startend_row_indices, group, causal, training
            )
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            query, key, value, output, log_sum_exp, startend_row_indices
        )
        ctx.group = group
        ctx.causal = causal
        ctx.fa_version = fa_version

        return output

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass of FlashMask attention with context parallelism.
        Args:
            ctx: Context object with saved information
            output_grad (paddle.Tensor): Gradient of output
        Returns:
            tuple: Gradients for all input arguments
        """
        # Retrieve saved tensors
        query, key, value, output, log_sum_exp, startend_row_indices = (
            ctx.saved_tensor()
        )
        group = ctx.group
        causal = ctx.causal
        fa_version = ctx.fa_version

        # Compute gradients
        query_grad, key_grad, value_grad = (
            cp_flashmask_allgatherkv_balance_backward(
                query,
                key,
                value,
                startend_row_indices,
                output,
                log_sum_exp,
                output_grad,
                group,
                causal,
                fa_version,
            )
        )

        return query_grad, key_grad, value_grad


def flashmask_attention_cp(
    query,
    key,
    value,
    startend_row_indices,
    fixed_seed_offset=None,
    dropout=0.0,
    causal=False,
    training=True,
    mode="allgather_kv",
):
    """
    FlashMask attention with context parallelism - public API.
    This is the main entry point for using FlashMask attention with context parallelism.
    It provides a convenient interface that wraps the FlashMaskContextParallel PyLayer.
    Args:
        query (paddle.Tensor): Query tensor with shape [batch, seq_len/n, num_heads, head_dim]
        key (paddle.Tensor): Key tensor with shape [batch, seq_len/n, num_heads, head_dim]
        value (paddle.Tensor): Value tensor with shape [batch, seq_len/n, num_heads, head_dim]
        startend_row_indices (paddle.Tensor): Row indices for attention mask
        fixed_seed_offset (paddle.Tensor, optional): Fixed seed offset for dropout
        dropout (float, optional): Dropout probability. Defaults to 0.0
        causal (bool, optional): Whether to use causal attention. Defaults to False
        training (bool, optional): Whether in training mode. Defaults to True
        mode (str, optional): Attention mode. Defaults to "allgather_kv"
    Returns:
        paddle.Tensor: Attention output with shape [batch, seq_len/n, num_heads, head_dim]
    Example:
        ```python
        # Initialize tensors (assuming context parallelism is set up)
        query = paddle.randn([2, 512, 8, 64])  # [batch, seq_len/n, heads, head_dim]
        key = paddle.randn([2, 512, 8, 64])    # [batch, seq_len/n, heads, head_dim]
        value = paddle.randn([2, 512, 8, 64])  # [batch, seq_len/n, heads, head_dim]
        mask_indices = paddle.randint(0, 1024, [100, 2])
        # Apply FlashMask attention with context parallelism
        output = flashmask_attention_cp(
            query=query,
            key=key,
            value=value,
            startend_row_indices=mask_indices,
            training=True
        )
        ```
    """
    output = FlashMaskContextParallel.apply(
        query,
        key,
        value,
        startend_row_indices,
        fixed_seed_offset,
        dropout,
        causal,
        training,
        mode,
    )
    return output
