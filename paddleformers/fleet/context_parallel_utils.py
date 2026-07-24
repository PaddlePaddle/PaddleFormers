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

import paddlefleet_ops.flash_mask_facade
from paddlefleet_ops.flash_mask_facade import get_fa_version

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
        from paddlefleet_ops.flash_mask.utils import bshd_slice_contiguous_kv

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
    Balanced all-gather operation using Triton reorder kernel.

    Gathers tensors from all ranks via all_gather, then reorders the gathered data
    using a Triton kernel (balanced_gather_reorder_kernel) to reconstruct the original
    sequence order from the DualChunkSwap balanced layout. Each rank's local tensor
    contains two chunks (one from the start, one from the end of the sequence), and
    this function reassembles them into the full contiguous sequence.

    This is the inverse of reduce_scatter_any_axis_balance and scatter_balance.

    Args:
        input_tensor (paddle.Tensor): Local tensor chunk to gather. Each rank's
            chunk size along `axis` must be even (split into two halves by the
            balanced strategy).
        group (paddle.distributed.Group, optional): Communication group. If None,
            uses the model parallel group from fleet.
        axis (int, optional): Axis along which to gather and reorder. Defaults to 0.

    Returns:
        paddle.Tensor: Full gathered tensor with shape[axis] = input_shape[axis] * parallelism,
            reordered to restore the original sequence order.
    """
    import triton

    from paddleformers.fleet.triton_ops.balanced_reorder import (
        balanced_gather_reorder_kernel,
    )

    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    # Single all_gather (gathers along axis=0)
    shape = list(input_tensor.shape)
    gathered_shape = list(shape)
    gathered_shape[0] = shape[0] * parallelism
    gathered = paddle.empty(gathered_shape, dtype=input_tensor.dtype)
    dist.stream.all_gather(
        gathered, input_tensor.contiguous(), group=group, use_calc_stream=True
    )

    # Compute strides for reorder kernel
    axis_size = shape[axis]
    chunk_size = axis_size // 2
    N = parallelism

    # outer_size: product of all dims left of axis in the *original* (per-rank) shape
    outer_size = 1
    for i in range(axis):
        outer_size *= shape[i]

    # inner_size: product of all dims right of axis
    inner_size = 1
    for i in range(axis + 1, len(shape)):
        inner_size *= shape[i]

    # src is gathered along axis=0: shape = [N*S0, S1, ..., S_axis, ..., S_last]
    # src_rank_stride = elements per rank = product of original shape
    src_rank_stride = 1
    for s in shape:
        src_rank_stride *= s

    # src_outer_stride = elements to skip per outer index = S_axis * inner_size
    src_outer_stride = axis_size * inner_size

    out_shape = list(shape)
    out_shape[axis] = 2 * N * chunk_size
    output = paddle.empty(out_shape, dtype=input_tensor.dtype)

    BLOCK_SIZE = 1024
    num_blocks_per_chunk = triton.cdiv(chunk_size * inner_size, BLOCK_SIZE)
    grid = (num_blocks_per_chunk * 2 * N, outer_size, 1)

    balanced_gather_reorder_kernel[grid](
        gathered,
        output,
        N,
        chunk_size,
        inner_size,
        src_rank_stride,
        src_outer_stride,
        num_blocks_per_chunk,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


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
    Balanced reduce-scatter operation along any axis using Triton reorder kernel.

    Performs reduce-scatter with the DualChunkSwap balanced strategy: first reorders
    the input tensor via a Triton kernel (balanced_scatter_reorder_kernel) to prepare
    balanced chunks for each rank, then uses alltoall_single to exchange data, and
    finally sums the received chunks to produce the reduced result.

    This is the inverse of all_gather_balance and is used in backward passes of
    context parallel attention (e.g., to reduce-scatter key/value gradients).

    Args:
        input_tensor (paddle.Tensor): Input tensor to reduce and scatter. The size
            along `axis` must be divisible by (parallelism * 2).
        axis (int): Axis along which to perform the balanced reduce-scatter.
        group (paddle.distributed.Group, optional): Communication group. If None,
            uses the context parallel group from fleet.

    Returns:
        paddle.Tensor: Reduced tensor with shape[axis] = input_shape[axis] / parallelism.
    """
    import triton

    from paddleformers.fleet.triton_ops.balanced_reorder import (
        balanced_scatter_reorder_kernel,
    )

    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    N = parallelism
    shape = list(input_tensor.shape)

    assert shape[axis] % (N * 2) == 0, (
        f"Input sequence length {shape[axis]} can't be "
        f"divided exactly by context parallelism * 2 {N * 2}"
    )

    chunk_size = shape[axis] // (2 * N)

    outer_size = 1
    for i in range(axis):
        outer_size *= shape[i]

    inner_size = 1
    for i in range(axis + 1, len(shape)):
        inner_size *= shape[i]

    src_outer_stride = shape[axis] * inner_size
    dst_outer_stride = 2 * chunk_size * inner_size
    dst_rank_stride = outer_size * dst_outer_stride

    per_rank_shape = list(shape)
    per_rank_shape[axis] = 2 * chunk_size
    # send_buf: [N, *per_rank_shape], contiguous, kernel writes into it
    send_buf = paddle.empty([N, *per_rank_shape], dtype=input_tensor.dtype)

    BLOCK_SIZE = 1024
    num_blocks_per_chunk = triton.cdiv(chunk_size * inner_size, BLOCK_SIZE)
    grid = (num_blocks_per_chunk * 2 * N, outer_size, 1)

    balanced_scatter_reorder_kernel[grid](
        input_tensor,
        send_buf,
        N,
        chunk_size,
        inner_size,
        src_outer_stride,
        dst_rank_stride,
        dst_outer_stride,
        num_blocks_per_chunk,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # alltoall_single: send_buf[r] -> rank r's recv_buf[my_rank]
    recv_buf = paddle.empty_like(send_buf)
    dist.stream.alltoall_single(
        recv_buf.reshape([-1]),
        send_buf.reshape([-1]),
        group=group,
        use_calc_stream=True,
    )

    # sum across N received chunks: same order as original stack+sum
    result = recv_buf.reshape([N, *per_rank_shape]).sum(axis=0)
    return result


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

        if mode.startswith("contiguous"):
            return scatter_contiguous(input_tensor, group=group, axis=axis)
        return scatter_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode.startswith("contiguous"):
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

        if mode.startswith("contiguous"):
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode.startswith("contiguous"):
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

        if mode.startswith("contiguous"):
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode.startswith("contiguous"):
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
    query,
    key,
    value,
    startend_row_indices,
    learnable_sink,
    group,
    causal,
    is_training,
    softmax_scale,
    mode: str = "dualchunk_allgather",
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
        softmax_scale (float): softmax scaling factor
        mode (str): Attention mode, support 'dualchunk_allgather' and 'contiguous_allgather'
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

    if mode == "dualchunk_allgather":
        key_gathered = all_gather_balance(key, axis=1, group=group)
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
    elif mode == "contiguous_allgather":
        key_gathered = all_gather_contiguous(key, axis=1, group=group)
        value_gathered = all_gather_contiguous(value, axis=1, group=group)

        startend_row_indices = preprocess_index(
            startend_row_indices,
            chunk_id=group.rank,
            seq_blocksize=query.shape[1],
            max_seqlen_q=query.shape[1],
        )
    else:
        raise ValueError(f"Unsupported FlashMask context parallel mode: {mode}")

    q_head_dim = query.shape[-1]
    v_head_dim = value_gathered.shape[-1]
    fa_version = get_fa_version(q_head_dim, v_head_dim, startend_row_indices)

    if fa_version == 4 and _flash_mask_available:
        output, log_sum_exp = _flash_attn_fwd(
            query,
            key_gathered,
            value_gathered,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            learnable_sink=learnable_sink,
            pack_gqa=False,
            softmax_scale=softmax_scale,
        )
    else:
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
        output, log_sum_exp = flashmask_attention(
            query,
            key_gathered,
            value_gathered,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True,
            training=is_training,
            softmax_scale=softmax_scale,
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
    learnable_sink,
    group,
    causal,
    fa_version: int,
    softmax_scale,
    mode: str = "dualchunk_allgather",
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
        softmax_scale (float): Softmax scaling factor
        mode (str): Attention mode, support 'dualchunk_allgather' and 'contiguous_allgather'
    Returns:
        tuple: (query_grad, key_grad, value_grad, sink_grad)
    """
    paddle.base.core.nvprof_nvtx_push(
        "cp_flashmask_allgatherkv_balance_backward"
    )

    # All-gather key and value tensors (same as forward pass)
    if mode == "dualchunk_allgather":
        key_gathered = all_gather_balance(key, axis=1, group=group)
        value_gathered = all_gather_balance(value, axis=1, group=group)
    elif mode == "contiguous_allgather":
        key_gathered = all_gather_contiguous(key, axis=1, group=group)
        value_gathered = all_gather_contiguous(value, axis=1, group=group)
    else:
        raise ValueError(f"Unsupported FlashMask context parallel mode: {mode}")

    sink_grad = None
    if fa_version == 2:
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
        if softmax_scale is not None:
            raise NotImplementedError(
                "fa_version==2 does not support setting softmax_scale"
            )
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
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
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
                    query.shape[-1] ** (-0.5)
                    if softmax_scale is None
                    else softmax_scale,
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
                    query.shape[-1] ** (-0.5)
                    if softmax_scale is None
                    else softmax_scale,
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
                    query.shape[-1] ** (-0.5)
                    if softmax_scale is None
                    else softmax_scale,
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
        query_grad, key_grad_gathered, value_grad_gathered, sink_grad = (
            _flash_attn_bwd(
                query,
                key_gathered,
                value_gathered,
                output,
                output_grad,
                log_sum_exp,
                flashmask_info,
                learnable_sink=learnable_sink,
                causal=causal,
                softmax_scale=softmax_scale,
                deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                    "FLAGS_cudnn_deterministic"
                ],
            )
        )
    else:
        raise ValueError(
            f"FlashAttention version {fa_version} is not supported."
        )

    # Reduce-scatter key and value gradients
    if mode == "dualchunk_allgather":
        key_grad = reduce_scatter_any_axis_balance(
            key_grad_gathered, axis=1, group=group
        )
        value_grad = reduce_scatter_any_axis_balance(
            value_grad_gathered, axis=1, group=group
        )
    elif mode == "contiguous_allgather":
        key_grad = reduce_scatter_contiguous(
            key_grad_gathered, axis=1, group=group
        )
        value_grad = reduce_scatter_contiguous(
            value_grad_gathered, axis=1, group=group
        )
    else:
        raise ValueError(f"Unsupported FlashMask context parallel mode: {mode}")

    paddle.base.core.nvprof_nvtx_pop()
    return query_grad, key_grad, value_grad, sink_grad


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
        learnable_sink=None,
        softmax_scale=None,
        mode="dualchunk_allgather",
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
            mode (str): Attention mode, supports "dualchunk_allgather" and "contiguous_allgather"
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
                query,
                key,
                value,
                startend_row_indices,
                learnable_sink,
                group,
                causal,
                training,
                softmax_scale,
                mode,
            )
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            query, key, value, output, log_sum_exp, startend_row_indices
        )
        ctx.group = group
        ctx.causal = causal
        ctx.fa_version = fa_version
        ctx.learnable_sink = learnable_sink
        ctx.softmax_scale = softmax_scale
        # Only a trainable sink (a Parameter) needs a gradient returned from
        # backward. A fixed off-by-one sink is created as a stop_gradient=True
        # Tensor, and Paddle's PyLayer requires None in that return slot.
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        ctx.mode = mode

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
        learnable_sink = ctx.learnable_sink
        softmax_scale = ctx.softmax_scale
        mode = ctx.mode

        # Compute gradients
        query_grad, key_grad, value_grad, sink_grad = (
            cp_flashmask_allgatherkv_balance_backward(
                query,
                key,
                value,
                startend_row_indices,
                output,
                log_sum_exp,
                output_grad,
                learnable_sink,
                group,
                causal,
                fa_version,
                softmax_scale,
                mode,
            )
        )

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: query(0)/key(1)/value(2)/startend_row_indices(3)/
        # learnable_sink(4). startend_row_indices is stop_gradient=True, so its
        # slot (position 3) must be None -- sink_grad belongs in position 4.
        # A fixed off-by-one sink is also stop_gradient=True, so for it the
        # 3-tuple (sink slot omitted) is correct.
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, None, sink_grad
        return query_grad, key_grad, value_grad


# ======================== P2P SWA CP fast path Layer ========================


def _wait_all(tasks):
    """Wait for all asynchronous communication tasks."""
    for task in tasks:
        task.wait()


def _exchange_prev_window(key, value, group, window_size=128):
    """Exchange each rank's tail KV window with the next CP rank."""
    rank = group.rank
    cp_size = group.world_size

    assert len(key.shape) == 4, (
        f"SWA P2P expects BSHD KV, got key.shape={key.shape}"
    )
    assert key.shape[1] >= window_size, (
        f"SWA window requires local KV sequence length >= {window_size}, "
        f"got {key.shape[1]}"
    )
    assert value.shape == key.shape, (
        f"key/value shape mismatch: key={key.shape}, value={value.shape}"
    )

    recv_window = paddle.empty(
        [2, key.shape[0], window_size, key.shape[2], key.shape[3]],
        dtype=key.dtype,
    )
    ops = []

    if rank > 0:
        recv_rank = group.ranks[rank - 1]
        ops.append(dist.P2POp(dist.irecv, recv_window, recv_rank, group))

    if rank < cp_size - 1:
        send_rank = group.ranks[rank + 1]
        send_window = paddle.stack(
            [key[:, -window_size:, :, :], value[:, -window_size:, :, :]], axis=0
        ).contiguous()
        ops.append(dist.P2POp(dist.isend, send_window, send_rank, group))

    if ops:
        _wait_all(dist.batch_isend_irecv(ops))

    return recv_window[0], recv_window[1]


def _scatter_kv_to_global_tensor(key, value, recv_key, recv_value, group):
    """Place local KV and received previous-window KV into global sequence layout."""
    rank = group.rank
    cp_size = group.world_size
    local_seqlen = key.shape[1]
    total_seqlen = local_seqlen * cp_size
    local_start = rank * local_seqlen
    local_end = local_start + local_seqlen

    if cp_size == 1:
        return key, value

    scratch_shape = [key.shape[0], total_seqlen, key.shape[2], key.shape[3]]
    key_tensor = paddle.empty(scratch_shape, dtype=key.dtype)
    value_tensor = paddle.empty(scratch_shape, dtype=value.dtype)
    key_tensor[:, local_start:local_end, :, :] = key
    value_tensor[:, local_start:local_end, :, :] = value

    if rank > 0:
        window_size = recv_key.shape[1]
        window_start = local_start - window_size
        key_tensor[:, window_start:local_start, :, :] = recv_key
        value_tensor[:, window_start:local_start, :, :] = recv_value

    return key_tensor, value_tensor


def _send_window_grad_back(
    key_grad_tensor, value_grad_tensor, key, value, group, window_size
):
    """Return remote-window KV gradients to owner ranks and accumulate them."""
    rank = group.rank
    cp_size = group.world_size
    local_seqlen = key.shape[1]
    local_start = rank * local_seqlen

    if cp_size == 1:
        return key_grad_tensor, value_grad_tensor

    key_grad, value_grad = bshd_slice_contiguous_kv(
        key_grad_tensor, value_grad_tensor, local_start, local_seqlen
    )

    recv_grad_window = paddle.empty(
        [2, key.shape[0], window_size, key.shape[2], key.shape[3]],
        dtype=key.dtype,
    )
    ops = []

    if rank < cp_size - 1:
        send_rank = group.ranks[rank + 1]
        ops.append(dist.P2POp(dist.irecv, recv_grad_window, send_rank, group))

    if rank > 0:
        recv_rank = group.ranks[rank - 1]
        window_start = local_start - window_size
        send_grad_window = paddle.stack(
            [
                key_grad_tensor[:, window_start:local_start, :, :],
                value_grad_tensor[:, window_start:local_start, :, :],
            ],
            axis=0,
        ).contiguous()
        ops.append(dist.P2POp(dist.isend, send_grad_window, recv_rank, group))

    if ops:
        _wait_all(dist.batch_isend_irecv(ops))

    if rank < cp_size - 1:
        key_grad[:, -window_size:, :, :].add_(recv_grad_window[0])
        value_grad[:, -window_size:, :, :].add_(recv_grad_window[1])

    return key_grad, value_grad


def cp_flashmask_swa_p2p_forward(
    query,
    key,
    value,
    startend_row_indices,
    learnable_sink,
    group,
    causal,
    is_training,
    softmax_scale,
    window_size,
):
    """Run forward SWA FlashMask CP with one-hop P2P KV exchange."""
    paddle.base.core.nvprof_nvtx_push("cp_flashmask_swa_p2p_forward")

    startend_row_indices = preprocess_index(
        startend_row_indices,
        chunk_id=group.rank,
        seq_blocksize=query.shape[1],
        max_seqlen_q=query.shape[1],
    )

    recv_key, recv_value = _exchange_prev_window(key, value, group, window_size)

    key_tensor, value_tensor = _scatter_kv_to_global_tensor(
        key, value, recv_key, recv_value, group
    )

    output, log_sum_exp = _flash_attn_fwd(
        query,
        key_tensor,
        value_tensor,
        startend_row_indices=startend_row_indices,
        learnable_sink=learnable_sink,
        causal=causal,
        return_lse=True,
        pack_gqa=False,
        softmax_scale=softmax_scale,
    )

    paddle.base.core.nvprof_nvtx_pop()

    return output, log_sum_exp, recv_key, recv_value, startend_row_indices


def cp_flashmask_swa_p2p_backward(
    query,
    key,
    value,
    recv_key,
    recv_value,
    startend_row_indices,
    output,
    log_sum_exp,
    output_grad,
    learnable_sink,
    group,
    causal,
    softmax_scale,
    window_size,
):
    """Run backward SWA FlashMask CP and return P2P KV gradients."""
    paddle.base.core.nvprof_nvtx_push("cp_flashmask_swa_p2p_backward")

    key_tensor, value_tensor = _scatter_kv_to_global_tensor(
        key, value, recv_key, recv_value, group
    )

    flashmask_info = None
    if startend_row_indices is not None:
        flashmask_info = FlashMaskInfoPaddle(
            startend_row_indices=startend_row_indices,
            is_causal=causal,
        )

    local_seqlen = key.shape[1]
    local_start = group.rank * local_seqlen
    local_end = local_start + local_seqlen
    kv_postprocess_start = (
        local_start if group.rank == 0 else local_start - window_size
    )

    query_grad, key_grad_tensor, value_grad_tensor, grad_sink = _flash_attn_bwd(
        query,
        key_tensor,
        value_tensor,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info=flashmask_info,
        learnable_sink=learnable_sink,
        causal=causal,
        softmax_scale=softmax_scale,
        deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ],
        kv_postprocess_start=kv_postprocess_start,
        kv_postprocess_end=local_end,
    )

    key_grad, value_grad = _send_window_grad_back(
        key_grad_tensor, value_grad_tensor, key, value, group, window_size
    )

    paddle.base.core.nvprof_nvtx_pop()

    return query_grad, key_grad, value_grad, grad_sink


class FlashMaskSwaP2P(PyLayer):
    """PyLayer for FlashMask SWA context parallelism using one-hop P2P KV exchange."""

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
        learnable_sink=None,
        softmax_scale=None,
        group=None,
        mode="contiguous_allgather",
        window_size=None,
    ):
        """Forward pass for SWA P2P FlashMask attention."""
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in FlashMask context parallel yet."
            )
        if fixed_seed_offset is not None:
            raise NotImplementedError("Fixed seed offset is not supported yet.")
        window_size = 128 if window_size is None else window_size
        if window_size <= 0:
            raise ValueError(
                f"SWA P2P window_size must be positive, got {window_size}"
            )

        output, log_sum_exp, recv_key, recv_value, startend_row_indices = (
            cp_flashmask_swa_p2p_forward(
                query,
                key,
                value,
                startend_row_indices,
                learnable_sink,
                group,
                causal,
                training,
                softmax_scale,
                window_size,
            )
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            recv_key,
            recv_value,
            output,
            log_sum_exp,
            startend_row_indices,
        )
        ctx.learnable_sink = learnable_sink
        ctx.softmax_scale = softmax_scale
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        ctx.group = group
        ctx.causal = causal
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, output_grad):
        """Backward pass for SWA P2P FlashMask attention."""
        (
            query,
            key,
            value,
            recv_key,
            recv_value,
            output,
            log_sum_exp,
            startend_row_indices,
        ) = ctx.saved_tensor()
        query_grad, key_grad, value_grad, grad_sink = (
            cp_flashmask_swa_p2p_backward(
                query,
                key,
                value,
                recv_key,
                recv_value,
                startend_row_indices,
                output,
                log_sum_exp,
                output_grad,
                ctx.learnable_sink,
                ctx.group,
                ctx.causal,
                ctx.softmax_scale,
                ctx.window_size,
            )
        )
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, None, grad_sink
        return query_grad, key_grad, value_grad


# ===========================================================================
# Ulysses Context Parallel (All-to-All based sequence parallelism)
#
# DeepSpeed-Ulysses partitions the input sequence across P GPUs. Before
# attention, an all-to-all redistributes Q/K/V so that each GPU holds the
# *full sequence* but only h/P attention heads. After local attention, a
# reverse all-to-all restores the original sequence-partitioned layout.
# ===========================================================================


def _ulysses_generate_layout_params(
    scatter_idx, batch_dim_idx, seq_world_size, input
):
    """
    Generate reshape/permute parameters for the all-to-all in Ulysses SP.

    With batch_dim_idx=0 (tensor layout [batch, seq, heads, head_dim]):
      - scatter_idx < 2 (scatter_idx=0 or 1, i.e. scatter along sequence dim):
            Input  [b, full_seq, h/P, d] -> Output [b, full_seq/P, h, d]
            Scatters sequence across ranks, gathers heads from all ranks.
      - scatter_idx >= 2 (scatter_idx=2, i.e. scatter along heads dim):
            Input  [b, seq/P, h, d]      -> Output [b, seq, h/P, d]
            Scatters heads across ranks, gathers sequence from all ranks.
    """
    if batch_dim_idx == 0:
        if scatter_idx < 2:
            # Scatter sequence, gather heads
            bs, global_seq_len, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [
                bs,
                seq_world_size,
                global_seq_len // seq_world_size,
                num_local_head,
                head_dim,
            ]
            pre_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [
                bs,
                global_seq_len // seq_world_size,
                seq_world_size * num_local_head,
                head_dim,
            ]
        else:
            # Scatter heads, gather sequence
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, (
                f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            )
            pre_all2all_inp_shape = [
                bs,
                local_seq_len,
                seq_world_size,
                num_total_head // seq_world_size,
                head_dim,
            ]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)
            post_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_res_shape = [
                bs,
                seq_world_size * local_seq_len,
                num_total_head // seq_world_size,
                head_dim,
            ]
    else:
        if scatter_idx < 2:
            # batch_dim_idx=1: tensor layout [seq, batch, heads, head_dim]
            global_seq_len, bs, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [
                seq_world_size,
                global_seq_len // seq_world_size,
                bs,
                num_local_head,
                head_dim,
            ]
            pre_all2all_permute_idx = None
            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [
                global_seq_len // seq_world_size,
                bs,
                seq_world_size * num_local_head,
                head_dim,
            ]
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, (
                f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            )
            pre_all2all_inp_shape = [
                local_seq_len,
                bs,
                seq_world_size,
                num_total_head // seq_world_size,
                head_dim,
            ]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)
            post_all2all_permute_idx = None
            post_all2all_res_shape = [
                local_seq_len * seq_world_size,
                bs,
                num_total_head // seq_world_size,
                head_dim,
            ]

    return (
        pre_all2all_permute_idx,
        pre_all2all_inp_shape,
        post_all2all_permute_idx,
        post_all2all_res_shape,
    )


def _ulysses_single_all_to_all(
    input, scatter_idx, gather_idx, batch_dim_idx, group
):
    """
    Perform a single all-to-all with reshape/permute for Ulysses SP.
    """
    seq_world_size = dist.get_world_size(group)
    (
        pre_all2all_permute_idx,
        pre_all2all_inp_shape,
        post_all2all_permute_idx,
        post_all2all_res_shape,
    ) = _ulysses_generate_layout_params(
        scatter_idx, batch_dim_idx, seq_world_size, input
    )

    # Pre-process: reshape and permute
    input_t = input.reshape(pre_all2all_inp_shape).contiguous()
    if pre_all2all_permute_idx is not None:
        input_t = input_t.permute(pre_all2all_permute_idx).contiguous()

    # All-to-all communication
    output = paddle.empty_like(input_t)
    dist.alltoall(output, input_t, group=group)

    # Post-process: permute and reshape
    if post_all2all_permute_idx is not None:
        output = output.permute(post_all2all_permute_idx).contiguous()
    output = output.reshape(post_all2all_res_shape).contiguous()

    return output


# ---------------------------------------------------------------------------
# Fused Ulysses all-to-all permute (Triton).
#
# For the production path (batch_dim_idx=0), the pre/post reshape+transpose+
# .contiguous() around the all-to-all are pure data movement whose cost (two
# physical copies) rivals the communication itself. A single Triton kernel
# fuses reshape+transpose+contiguous into one coalesced copy, and the post
# output reuses the send buffer for a lower peak. The kernels live in
# paddleformers.fleet.triton_ops.ulysses_alltoall_fused; the thin wrappers below keep
# stable module-level names. A cheap non-Triton guard runs first so that
# unsupported layouts, CPU/non-CUDA inputs, or Triton-less environments fall
# back to the reference _ulysses_single_all_to_all path; the Triton import is
# deferred to call time and is itself guarded.
# ---------------------------------------------------------------------------
def _ulysses_fused_supported(scatter_idx, batch_dim_idx, input):
    """Whether the fused Triton path can handle this call.

    Only batch_dim_idx=0 with a 4-D CUDA input is fused. Everything else
    (other layouts, CPU/non-CUDA inputs, or an environment where the Triton
    op cannot be imported) returns False so the caller keeps using the
    reference reshape/permute all-to-all.
    """
    if (
        batch_dim_idx != 0
        or len(input.shape) != 4
        or not paddle.is_compiled_with_cuda()
        or not input.place.is_gpu_place()
    ):
        return False

    try:
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            ulysses_alltoall_fused_supported,
        )
    except (ImportError, OSError):
        # ImportError: Triton not installed. OSError: Triton present but its
        # binary deps fail to load (e.g. shared library errors). Either way,
        # fall back to the reference reshape/permute path.
        return False

    return ulysses_alltoall_fused_supported(scatter_idx, batch_dim_idx, input)


def _ulysses_single_all_to_all_fused(input, scatter_idx, group):
    """Fused seq<->head all-to-all for batch_dim_idx=0 (bit-exact, 2x peak)."""
    from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
        ulysses_single_all_to_all_fused,
    )

    return ulysses_single_all_to_all_fused(input, scatter_idx, group)


class UlyssesAlltoAll(PyLayer):
    """
    Ulysses All-to-All for sequence parallelism.

    Forward performs all-to-all with the given scatter/gather indices.
    Backward performs the inverse all-to-all (swap scatter and gather indices).

    When the layout matches the production path (batch_dim_idx=0, 4-D input), a
    fused Triton kernel replaces the reshape+transpose+.contiguous() permutes
    for lower latency and peak memory. Otherwise it falls back to the reference
    reshape/permute path. Both are bit-exact.
    """

    @staticmethod
    def forward(ctx, input, scatter_idx, gather_idx, batch_dim_idx, group):
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.batch_dim_idx = batch_dim_idx
        ctx.group = group
        if _ulysses_fused_supported(scatter_idx, batch_dim_idx, input):
            return _ulysses_single_all_to_all_fused(input, scatter_idx, group)
        return _ulysses_single_all_to_all(
            input, scatter_idx, gather_idx, batch_dim_idx, group
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Inverse a2a swaps scatter/gather; the fused check uses the backward
        # scatter index (ctx.gather_idx).
        if _ulysses_fused_supported(
            ctx.gather_idx, ctx.batch_dim_idx, grad_output
        ):
            return _ulysses_single_all_to_all_fused(
                grad_output, ctx.gather_idx, ctx.group
            )
        return _ulysses_single_all_to_all(
            grad_output,
            ctx.gather_idx,
            ctx.scatter_idx,
            ctx.batch_dim_idx,
            ctx.group,
        )


def flashmask_attention_ulysses(
    query,
    key,
    value,
    startend_row_indices,
    causal=False,
    learnable_sink=None,
    softmax_scale=None,
):
    """
    FlashMask attention with Ulysses context parallelism.

    Each CP rank initially holds a sequence partition [b, N/P, h, d]. The Ulysses
    all-to-all redistributes Q/K/V so each rank holds the full sequence but only
    h/P heads: [b, N, h/P, d]. Local flashmask attention is then computed per rank.
    A reverse all-to-all restores the original sequence-partitioned layout.

    Requires: num_heads % cp_size == 0, and q_heads == k_heads == v_heads (no GQA).

    Args:
        query: [batch, seq_len/P, num_heads, head_dim] - sequence-partitioned query
        key:   [batch, seq_len/P, num_kv_heads, head_dim] - sequence-partitioned key
        value: [batch, seq_len/P, num_kv_heads, head_dim] - sequence-partitioned value
        startend_row_indices: [b, num_mask_heads, seq_len, cols] attention mask indices
            num_mask_heads must be 1 (broadcast) or equal to num_kv_heads.
        dropout: dropout probability
        causal: whether to use causal attention
        training: whether in training mode

    Returns:
        [batch, seq_len/P, num_heads, head_dim] - sequence-partitioned output
    """
    if learnable_sink is not None:
        raise NotImplementedError(
            "flashmask_attention_ulysses does not support learnable_sink "
            "(softmax sink)"
        )

    if softmax_scale is not None:
        raise NotImplementedError(
            "flashmask_attention_ulysses does not support setting softmax_scale"
        )
    hcg = fleet.get_hybrid_communicate_group()
    cp_group = hcg.get_context_parallel_group()
    cp_size = cp_group.nranks
    cp_rank = cp_group.rank

    num_q_heads = query.shape[2]
    num_k_heads = key.shape[2]
    num_v_heads = value.shape[2]

    assert num_q_heads == num_k_heads == num_v_heads, (
        f"Ulysses a2a CP requires q_heads == k_heads == v_heads, "
        f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
    )
    assert num_q_heads % cp_size == 0, (
        f"num_heads ({num_q_heads}) must be divisible by cp_size ({cp_size}) for Ulysses"
    )

    # Validate and slice startend_row_indices along head dimension
    # startend_row_indices shape: [b, num_mask_heads, seq_len, cols]
    num_mask_heads = startend_row_indices.shape[1]
    assert num_mask_heads == 1 or num_mask_heads == num_k_heads, (
        f"startend_row_indices head dim must be 1 or num_kv_heads ({num_k_heads}), "
        f"got {num_mask_heads}"
    )

    # When mask has per-head indices, slice the heads belonging to this rank
    if num_mask_heads != 1:
        heads_per_rank = num_mask_heads // cp_size
        head_start = cp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        startend_row_indices = startend_row_indices[
            :, head_start:head_end, :, :
        ]

    # Before attention: scatter heads across ranks, gather full sequence from all ranks
    # [b, N/P, h, d] -> [b, N, h/P, d]
    query = UlyssesAlltoAll.apply(
        query, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=cp_group
    )
    key = UlyssesAlltoAll.apply(
        key, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=cp_group
    )
    value = UlyssesAlltoAll.apply(
        value, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=cp_group
    )

    # Local flashmask attention on full sequence with h/P heads
    attn_output = paddlefleet_ops.flash_mask_facade.flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=startend_row_indices,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    # After attention: scatter sequence across ranks, gather full heads from all ranks
    # [b, N, h/P, d] -> [b, N/P, h, d]
    attn_output = UlyssesAlltoAll.apply(
        attn_output,
        scatter_idx=1,
        gather_idx=2,
        batch_dim_idx=0,
        group=cp_group,
    )

    return attn_output


def flashmask_attention_cp(
    query,
    key,
    value,
    startend_row_indices,
    fixed_seed_offset=None,
    dropout=0.0,
    causal=False,
    training=True,
    learnable_sink=None,
    softmax_scale=None,
    mode="dualchunk_allgather",
    window_size=None,
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
        mode (str, optional): Attention mode. Defaults to "dualchunk_allgather"
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
    if mode == "contiguous_swap2p":
        hcg = fleet.get_hybrid_communicate_group()
        cp_group = hcg.get_context_parallel_group()

        assert _flash_mask_available, (
            "P2P SWA fast path requires flashmask installed. Please check."
        )

        return FlashMaskSwaP2P.apply(
            query,
            key,
            value,
            startend_row_indices,
            fixed_seed_offset,
            dropout,
            causal,
            training,
            learnable_sink,
            softmax_scale,
            cp_group,
            mode,
            window_size,
        )
    elif mode == "dualchunk_allgather":
        output = FlashMaskContextParallel.apply(
            query,
            key,
            value,
            startend_row_indices,
            fixed_seed_offset,
            dropout,
            causal,
            training,
            learnable_sink,
            softmax_scale,
            mode,
        )
    elif mode == "contiguous_a2a":
        if fixed_seed_offset is not None:
            raise NotImplementedError(
                "flashmask_attention_ulysses does not support setting fixed_seed_offset"
            )

        if dropout != 0.0:
            raise NotImplementedError(
                "flashmask_attention_ulysses does not support dropout"
            )

        if not training:
            raise NotImplementedError(
                "flashmask_attention_ulysses does not support setting training"
            )

        output = flashmask_attention_ulysses(
            query=query,
            key=key,
            value=value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            learnable_sink=learnable_sink,
            softmax_scale=softmax_scale,
        )
    else:
        raise ValueError(f"invalid cp_balance_mode: {mode}")
    return output


# ===================== MTP Distillation Loss Shift Layer =====================


def _mtp_distillation_loss_shift_forward(tensor, nextn, group):
    ops = []

    bs, _, hidden = tensor.shape
    rank = group.rank
    cp_size = group.world_size

    if rank > 0:
        send_rank = group.ranks[rank - 1]
        send_window = tensor[:, :nextn].contiguous()
        ops.append(dist.P2POp(dist.isend, send_window, send_rank, group))

    if rank < cp_size - 1:
        recv_rank = group.ranks[rank + 1]
        recv_window = paddle.empty([bs, nextn, hidden], tensor.dtype)
        ops.append(dist.P2POp(dist.irecv, recv_window, recv_rank, group))
    else:
        recv_window = paddle.zeros([bs, nextn, hidden], tensor.dtype)

    _wait_all(dist.batch_isend_irecv(ops))

    tensor = paddle.concat([tensor[:, 1:], recv_window], axis=1)
    return tensor


def _mtp_distillation_loss_shift_backward(tensor, nextn, group):
    ops = []

    bs, _, hidden = tensor.shape
    rank = group.rank
    cp_size = group.world_size

    if rank < cp_size - 1:
        send_rank = group.ranks[rank + 1]
        send_window = tensor[:, -nextn:].contiguous()
        ops.append(dist.P2POp(dist.isend, send_window, send_rank, group))

    if rank > 0:
        recv_rank = group.ranks[rank - 1]
        recv_window = paddle.empty([bs, nextn, hidden], tensor.dtype)
        ops.append(dist.P2POp(dist.irecv, recv_window, recv_rank, group))
    else:
        recv_window = paddle.zeros([bs, nextn, hidden], tensor.dtype)

    _wait_all(dist.batch_isend_irecv(ops))

    # output has shape [bs, seq_len, hidden]
    output = paddle.nn.functional.pad(
        tensor[:, :-nextn], [0, 0, 1], mode="constant", value=0
    )
    output[:, :nextn] += recv_window
    return output


class MTPDistillationLossShift(PyLayer):
    """
    Shift LMHead logits for MTP distillation loss computation.

    Given a local input tensor of shape [B, S, H] on each rank of a CP group, this function is
    conceptually equivalent to first all-gathering the global tensor of shape [B, S*cp_size, H]
    and then returning the slice [:, S*cp_rank+1 : S*(cp_rank+1)+nextn, :] on each rank.
    In practice, only the boundary tokens are exchanged between neighboring ranks instead of
    performing a full gather.
    """

    @staticmethod
    def forward(
        ctx, tensor, num_nextn_predict_layers, mode="contiguous_allgather"
    ):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

        assert len(tensor.shape) == 3, (
            f"Expect input of shape [B, S, H], got {tensor.shape}"
        )
        batch_size, seq_len, hidden_size = tensor.shape
        assert seq_len > num_nextn_predict_layers, (
            "The local seq_len per-rank should be greater than nextn, "
            f"got {seq_len=} and {num_nextn_predict_layers=}"
        )
        assert num_nextn_predict_layers > 0, (
            f"num_nextn_predict_layers must be greater than 0, got {num_nextn_predict_layers}"
        )
        assert mode == "contiguous_allgather", (
            f"MTPDistillationLossShift only supports 'contiguous_allgather' mode, got {mode}"
        )

        ctx.group = group
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.hidden_size = hidden_size
        ctx.num_nextn_predict_layers = num_nextn_predict_layers

        return _mtp_distillation_loss_shift_forward(
            tensor, num_nextn_predict_layers, group
        )

    @staticmethod
    def backward(ctx, output_grad):
        output_grad = output_grad.reshape(
            [
                ctx.batch_size,
                ctx.seq_len + ctx.num_nextn_predict_layers - 1,
                ctx.hidden_size,
            ]
        )
        return _mtp_distillation_loss_shift_backward(
            output_grad, ctx.num_nextn_predict_layers, ctx.group
        )
