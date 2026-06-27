# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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

import queue

import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group
from paddlefleet_ops import (
    is_deep_ep_available,
    is_hybrid_ep_available,
    is_sonic_moe_available,
)

from paddleformers.fleet.refined_recompute.queue_check import global_rr_queue_log

from .fp8_utils import FP8_ALIGN
from .moe_utils import manual_backward

if is_deep_ep_available():
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import deep_ep
    else:
        from paddle.distributed.communication import deep_ep

    HAVE_DEEP_EP = True
else:
    HAVE_DEEP_EP = False

if is_hybrid_ep_available():
    from paddlefleet_ops import hybrid_ep

    HAVE_HYBRID_EP = True
else:
    HAVE_HYBRID_EP = False

if is_sonic_moe_available():
    HAVE_SONIC_MOE = True
    try:
        from paddlefleet_ops.sonicmoe.quack_utils import (
            quantize_activation_blockscaled_fast,
        )
    except ImportError:
        quantize_activation_blockscaled_fast = None
else:
    quantize_activation_blockscaled_fast = None
    HAVE_SONIC_MOE = False

_buffer = None
_hybrid_ep_buffer = None

# HybridEP dispatch/combine kernels use 64-token chunks for public APIs.
HYBRIDEP_TOKEN_ALIGNMENT = 64


def barrier_ep(ep_group):
    """barrier_ep"""
    paddle.distributed.barrier(ep_group)


def wait_for_deepep(group_id):
    """wait_for_deepep"""
    comm_event = deep_ep.get_event_from_comm_stream(group_id)
    comm_event.calc_stream_wait(group_id)


def get_hidden_bytes(x: paddle.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (paddle.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.shape[1] * max(x.element_size(), 2)


def _normalize_fp8_scale_for_deepep(
    x_fp8: paddle.Tensor, scale: paddle.Tensor, use_ue8m0: bool = False
):
    num_tokens = x_fp8.shape[0]
    num_scales = x_fp8.shape[1] // 128
    if use_ue8m0:
        num_scales //= 4
    if scale.shape[0] == num_scales:
        scale = scale.T.contiguous()
    else:
        scale = scale.contiguous()
    if scale.shape[0] > num_tokens:
        scale = scale[:num_tokens, :]
    if scale.shape[0] != num_tokens or scale.shape[1] != num_scales:
        raise RuntimeError(
            "Invalid FP8 scale shape for DeepEP dispatch: "
            f"scale={scale.shape}, x_fp8={x_fp8.shape}, "
            f"expected [{num_tokens}, {num_scales}]"
        )
    return scale


def configure_buffer(num_sms=None, dispatch_config=None, combine_config=None):
    """
    Configure the runtime parameters for deep_ep kernels.
    Must be called before calling get_buffer() to take effect.

    Args:
        num_sms (int): Number of SMs allocated to deep_ep kernels.
        dispatch_config (List[int]):
            Token capacity parameters for dispatch kernels, in the form
            [nvl_send_tokens, nvl_recv_tokens, rdma_send_tokens, rdma_recv_tokens].
            Trailing values may be omitted to use the defaults.
        combine_config (List[int]): Same as above, but for combine kernels.
    """
    if num_sms is not None and HAVE_DEEP_EP:
        deep_ep.Buffer.set_num_sms(num_sms)
    if dispatch_config is not None and HAVE_DEEP_EP:
        deep_ep.Buffer.get_dispatch_config = staticmethod(
            lambda _: deep_ep.Config(deep_ep.Buffer.num_sms, *dispatch_config)
        )
    if combine_config is not None and HAVE_DEEP_EP:
        deep_ep.Buffer.get_combine_config = staticmethod(
            lambda _: deep_ep.Config(deep_ep.Buffer.num_sms, *combine_config)
        )


def get_buffer(group: Group, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (paddle.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.world_size),
        deep_ep.Buffer.get_combine_config(group.world_size),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size),
            num_rdma_bytes,
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def reset_hybrid_ep_buffer():
    """Reset the shared HybridEP communication buffer."""
    global _hybrid_ep_buffer

    _hybrid_ep_buffer = None


def _need_new_hybrid_ep_buffer(
    group: Group,
    hidden_dim: int,
    max_num_of_tokens_per_rank: int,
    num_local_experts: int,
    num_sms_dispatch_api: int | None,
    num_sms_combine_api: int | None,
    num_sms_preprocessing_api: int | None,
):
    if _hybrid_ep_buffer is None:
        return True

    config = _hybrid_ep_buffer.config
    need_new_buffer = (
        _hybrid_ep_buffer.group != group
        or config.hidden_dim != hidden_dim
        or config.max_num_of_tokens_per_rank < max_num_of_tokens_per_rank
        or config.num_of_experts_per_rank != num_local_experts
    )
    if num_sms_dispatch_api is not None:
        need_new_buffer |= (
            _hybrid_ep_buffer.num_sms_dispatch_api != num_sms_dispatch_api
        )
    if num_sms_combine_api is not None:
        need_new_buffer |= (
            _hybrid_ep_buffer.num_sms_combine_api != num_sms_combine_api
        )
    if num_sms_preprocessing_api is not None:
        need_new_buffer |= (
            _hybrid_ep_buffer.num_sms_preprocessing_api
            != num_sms_preprocessing_api
        )
    return need_new_buffer


def get_hybrid_ep_buffer(
    group: Group,
    hidden_dim: int,
    max_num_of_tokens_per_rank: int,
    num_local_experts: int,
    load_cached_kernels: bool = True,
    num_sms_dispatch_api: int | None = None,
    num_sms_combine_api: int | None = None,
    num_sms_preprocessing_api: int | None = None,
):
    """Get or create the shared HybridEP communication buffer."""
    global _hybrid_ep_buffer

    if _need_new_hybrid_ep_buffer(
        group,
        hidden_dim,
        max_num_of_tokens_per_rank,
        num_local_experts,
        num_sms_dispatch_api,
        num_sms_combine_api,
        num_sms_preprocessing_api,
    ):
        _hybrid_ep_buffer = hybrid_ep.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_of_tokens_per_rank,
            num_local_experts=num_local_experts,
            use_fp8=False,
            num_sms_dispatch_api=num_sms_dispatch_api,
            num_sms_combine_api=num_sms_combine_api,
            num_sms_preprocessing_api=num_sms_preprocessing_api,
            load_cached_kernels=load_cached_kernels,
        )
    return _hybrid_ep_buffer


def fused_dispatch_forward_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Forward pass of fused dispatch."""
    if moe_ep_barrier:
        barrier_ep(group)
    # Calculate layout before actual dispatch
    if isinstance(x, tuple):
        buffer = get_buffer(group, get_hidden_bytes(x[0]))
    else:
        buffer = get_buffer(group, get_hidden_bytes(x))
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event_,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    assert token_probs.dtype == paddle.float32
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive,
    # so this is not compatible with CUDA graph
    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    states = {}
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    return recv_x, recv_token_probs, states, event


def fused_dispatch_backward_func(
    grad_output,
    grad_token_probs,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Backward pass of fused dispatch."""
    if moe_ep_barrier:
        barrier_ep(group)

    buffer = get_buffer(group, get_hidden_bytes(grad_output))

    grad_x, grad_token_probs, event = buffer.combine(
        grad_output.contiguous(),
        handle,
        topk_weights=grad_token_probs.cast(paddle.float32),
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return grad_x, None, grad_token_probs


def fused_combine_forward_func(
    x,
    group,
    states,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Forward pass of fused combine."""
    if moe_ep_barrier:
        barrier_ep(group)

    handle = states["handle"]
    buffer = get_buffer(group, get_hidden_bytes(x))
    combined_x, _, event = buffer.combine(
        x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return combined_x


def fused_combine_backward_func(
    grad_output,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Backward pass of fused combine."""
    if moe_ep_barrier:
        barrier_ep(group)

    if isinstance(grad_output, tuple):
        buffer = get_buffer(group, get_hidden_bytes(grad_output[0]))
        grad_x, _, _, _, _, event = buffer.dispatch(
            (grad_output[0].contiguous(), grad_output[1].contiguous()),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    else:
        buffer = get_buffer(group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    return grad_x


class DeepEPDispatch(PyLayer):
    """DeepEP dispatch operation for MoE routing and expert parallel communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        moe_ep_barrier: bool = True,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        """Forward pass of fused dispatch."""
        if fp8_dispatch:
            if using_sonic_moe:
                assert quantize_activation_blockscaled_fast is not None, (
                    "Cannot find quantize_activation_blockscaled_fast, please update sonicmoe."
                )
                if not x.is_contiguous():
                    x = x.contiguous()
                x_fp8, scale = quantize_activation_blockscaled_fast(
                    x, scale_dtype=paddle.int32
                )
            else:
                x_fp8, scale = (
                    paddle.incubate.nn.functional.fp8_quant_blockwise(
                        x,
                        quant_method="1x128",
                        input_transpose=False,
                        output_scale_transpose=True,
                        return_transpose_only=False,
                        using_ue8m0_scale=use_ue8m0,
                    )
                )
                scale = scale.T.contiguous()
            x = (x_fp8, scale)
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            moe_ep_barrier=moe_ep_barrier,
        )

        ctx.group = group
        ctx.handle = states["handle"]
        ctx.event = event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.set_grad_in_dtype_consistent(False)
        ctx.moe_ep_barrier = moe_ep_barrier
        if fp8_dispatch:
            recv_x, scale = recv_x
            return recv_x, recv_token_probs, states, {"scale": scale}
        return recv_x, recv_token_probs, states, None

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        return fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            ctx.group,
            ctx.handle,
            None,  # previous_event
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            moe_ep_barrier=ctx.moe_ep_barrier,
        )


class DeepEPCombine(PyLayer):
    """DeepEP combine operation for restoring MoE outputs across expert parallel ranks."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        moe_ep_barrier: bool = True,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        """Forward pass of fused combine."""
        combined_x = fused_combine_forward_func(
            x, group, states, previous_event, moe_ep_barrier=moe_ep_barrier
        )

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.previous_event = previous_event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.moe_ep_barrier = moe_ep_barrier
        ctx.fp8_dispatch = fp8_dispatch
        ctx.combine_grad_handle = combine_grad_handle

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        grad_for_comm = grad_output
        if ctx.fp8_dispatch:
            assert quantize_activation_blockscaled_fast is not None, (
                "Cannot find quantize_activation_blockscaled_fast, please update sonicmoe."
            )
            grad_output = grad_output.contiguous()
            grad_for_comm = quantize_activation_blockscaled_fast(
                grad_output, scale_dtype=paddle.int32
            )

        grad_x = fused_combine_backward_func(
            grad_for_comm,
            ctx.group,
            ctx.handle,
            ctx.previous_event,
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            moe_ep_barrier=ctx.moe_ep_barrier,
        )

        if ctx.fp8_dispatch:
            assert ctx.combine_grad_handle is not None, (
                "For fp8_dispatch, combine_grad_handle must be provided in combine backward."
            )
            grad_x, grad_scale = grad_x
            ctx.combine_grad_handle["data"] = grad_x
            ctx.combine_grad_handle["scale"] = grad_scale

        return grad_x


class DeepEPCombineAsync(PyLayer):
    """DeepEP combine with shared expert overlap."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        """Forward pass of fused combine."""
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            async_finish=True,
        )

        assert fn is not None, "use DeepEPCombineAsync async, but fn is None."
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.fp8_dispatch = fp8_dispatch
        ctx.combine_grad_handle = combine_grad_handle

        wait_for_deepep(group.id)

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        """Backward pass of fused combine."""
        grad_for_comm = grad_output
        if ctx.fp8_dispatch:
            assert quantize_activation_blockscaled_fast is not None, (
                "Cannot find quantize_activation_blockscaled_fast, please update sonicmoe."
            )
            grad_output = grad_output.contiguous()
            grad_for_comm = quantize_activation_blockscaled_fast(
                grad_output, scale_dtype=paddle.int32
            )

        grad_x = fused_combine_backward_func(
            grad_for_comm,
            ctx.group,
            ctx.handle,
            async_finish=True,
        )

        fn_args_grads = ctx.bwf(*fn_out_grads)

        wait_for_deepep(ctx.group.id)

        if ctx.fp8_dispatch:
            assert ctx.combine_grad_handle is not None, (
                "For fp8_dispatch, combine_grad_handle must be provided in combine backward."
            )
            grad_x, grad_scale = grad_x
            ctx.combine_grad_handle["data"] = grad_x
            ctx.combine_grad_handle["scale"] = grad_scale

        return (grad_x,) + fn_args_grads  # noqa: RUF005


class DeepEPCombineAsyncFunctor(PyLayer):
    """DeepEPCombineAsyncFunctor for deepep combine with overlap (Refined Recompute)."""

    @staticmethod
    def forward(
        ctx,
        hold_tensors,
        x,
        group,
        states,
        *fn_args,
        fn,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        """Forward pass of fused combine with overlap, get cached output directly."""
        combined_x = hold_tensors["res_output"]

        # Re-run fn with grad tracking to build backward graph and obtain bwf
        ctx.bwf, fn_out = manual_backward(fn, False, *fn_args)

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.fp8_dispatch = fp8_dispatch
        ctx.combine_grad_handle = combine_grad_handle

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        """Backward pass of fused combine with overlap."""
        grad_for_comm = grad_output
        if ctx.fp8_dispatch:
            assert quantize_activation_blockscaled_fast is not None, (
                "Cannot find quantize_activation_blockscaled_fast, please update sonicmoe."
            )
            grad_output = grad_output.contiguous()
            grad_for_comm = quantize_activation_blockscaled_fast(
                grad_output, scale_dtype=paddle.int32
            )

        grad_x = fused_combine_backward_func(
            grad_for_comm,
            ctx.group,
            ctx.handle,
            async_finish=True,
        )

        fn_args_grads = ctx.bwf(*fn_out_grads)

        wait_for_deepep(ctx.group.id)

        if ctx.fp8_dispatch:
            assert ctx.combine_grad_handle is not None, (
                "For fp8_dispatch, combine_grad_handle must be provided in combine backward."
            )
            grad_x, grad_scale = grad_x
            ctx.combine_grad_handle["data"] = grad_x
            ctx.combine_grad_handle["scale"] = grad_scale

        return (grad_x,) + fn_args_grads  # noqa: RUF005


class DeepEPCombineAsyncRefinedRecompute:
    """RefinedRecompute class for deepep fused_combine with overlap."""

    def __init__(self):
        """__init__"""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(
            self._hold_tensors_queue, "DeepEPCombineAsync"
        )

    def forward(self, x, group, states, *fn_args, fn):
        """forward"""
        tracer = framework._dygraph_tracer()
        is_first_fwd = not tracer._has_grad
        if is_first_fwd:
            # _first_fwd runs under @no_grad: returned tensors have no gradient.
            # The backward graph is rebuilt in the second forward (recompute) pass
            # via DeepEPCombineAsyncFunctor, so callers must not rely on gradients here.
            fwd_output, fn_out = self._first_fwd(x, group, states, fn, *fn_args)
            self._hold_tensors_queue.put({"res_output": fwd_output.detach()})
            return (fwd_output, *fn_out)
        else:
            if self._hold_tensors_queue.empty():
                raise RuntimeError(
                    "[DeepEPCombineAsyncRefinedRecompute] Queue is empty during the second forward "
                    "(recompute) pass. This usually indicates a first-forward / recompute-forward call count mismatch."
                )
            hold_tensors = self._hold_tensors_queue.get()
            output = self._second_fwd(
                hold_tensors, x, group, states, fn, *fn_args
            )
            return output

    @paddle.no_grad()
    def _first_fwd(self, x, group, states, fn, *fn_args):
        """_first_fwd"""
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            async_finish=True,
        )

        if fn is None:
            raise ValueError(
                "[DeepEPCombineAsyncRefinedRecompute] fn must not be None when using RefinedRecompute."
            )
        _, fn_out = manual_backward(fn, True, *fn_args)

        # After wait, the handle in states still holds metadata needed for backward
        # (same pattern as DeepEPCombineAsync). Do not remove this wait.
        wait_for_deepep(group.id)

        return combined_x, fn_out

    def _second_fwd(self, hold_tensors, x, group, states, fn, *fn_args):
        """_second_fwd"""
        return DeepEPCombineAsyncFunctor.apply(
            hold_tensors, x, group, states, *fn_args, fn=fn
        )

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.forward(*args, **kwargs)


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group: Group,
        previous_event=None,
        fp8_dispatch: bool = False,
        async_finish=False,
        allocate_on_comm_stream=False,
        moe_ep_barrier: bool = True,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event
            moe_ep_barrier: Whether to use barrier for expert parallelism

        Returns:
            Result of DeepEPDispatch
        """
        return DeepEPDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
            moe_ep_barrier,
            use_ue8m0,
            using_sonic_moe,
        )

    def fused_combine(
        x,
        group,
        handle,
        *,
        _rr_fusedcombined=None,
        previous_event=None,
        combine_overlap_handle=None,
        async_finish=False,
        moe_ep_barrier: bool = True,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            _rr_fusedcombined: RefinedRecompute functor for deepep combine
            previous_event: Previous CUDA event
            combine_overlap_handle: Handle for overlapping with shared experts
            moe_ep_barrier: Whether to use barrier for expert parallelism
            use_rr_deepep_combine: Whether to use refined recompute for deepep combine

        Returns:
            Result of DeepEPCombine
        """
        states = {}
        states["handle"] = handle
        if combine_overlap_handle is None:
            if use_rr_deepep_combine:
                raise ValueError(
                    "use_rr_deepep_combine requires combine_overlap_handle to be provided (not None)."
                )
            return DeepEPCombine.apply(
                x,
                group,
                states,
                previous_event,
                async_finish,
                moe_ep_barrier=moe_ep_barrier,
                fp8_dispatch=fp8_dispatch,
                combine_grad_handle=combine_grad_handle,
            )
        else:
            if previous_event is not None:
                raise ValueError(
                    "previous_event must be None when combine_overlap_handle is provided."
                )
            if not isinstance(combine_overlap_handle, dict):
                raise TypeError("combine_overlap_handle must be a dict.")
            if "fn" not in combine_overlap_handle:
                raise ValueError(
                    "combine_overlap_handle must contain 'fn' key."
                )
            if "fn_args" not in combine_overlap_handle:
                raise ValueError(
                    "combine_overlap_handle must contain 'fn_args' key."
                )
            if not isinstance(combine_overlap_handle["fn_args"], tuple):
                raise TypeError(
                    "combine_overlap_handle['fn_args'] must be a tuple."
                )
            if not use_rr_deepep_combine:
                combined_x, *fn_out = DeepEPCombineAsync.apply(
                    x,
                    group,
                    states,
                    *(combine_overlap_handle["fn_args"]),
                    fn=combine_overlap_handle["fn"],
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    fp8_dispatch=fp8_dispatch,
                    combine_grad_handle=combine_grad_handle,
                )
                combine_overlap_handle["fn_out"] = fn_out
                return combined_x
            else:
                if _rr_fusedcombined is None:
                    raise ValueError(
                        "_rr_fusedcombined must be provided when use_rr_deepep_combine is True with combine_overlap_handle."
                    )
                combined_x, *fn_out = _rr_fusedcombined(
                    x,
                    group,
                    states,
                    *(combine_overlap_handle["fn_args"]),
                    fn=combine_overlap_handle["fn"],
                )
                combine_overlap_handle["fn_out"] = fn_out
                return combined_x

else:
    fused_dispatch = None
    fused_combine = None


class HybridEPDispatch(PyLayer):
    """Fused HybridEP dispatch bridge for Paddle autograd."""

    @staticmethod
    def forward(
        ctx, x, token_indices, token_probs, manager, fp8_dispatch=False
    ):
        recv_x, recv_token_probs, scale = manager._dispatch_with_permute_impl(
            x, token_indices, token_probs, use_fp8=fp8_dispatch
        )
        ctx.buffer = manager._active_buffer
        ctx.handle = manager.handle
        ctx.token_indices = token_indices
        ctx.num_unpadded_tokens = x.shape[0]
        ctx.hidden_dtype = x.dtype
        ctx.use_fp8_dispatch = fp8_dispatch
        ctx.set_grad_in_dtype_consistent(False)
        return recv_x, recv_token_probs, scale

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs, grad_scale=None):
        del grad_scale
        if grad_output.dtype != ctx.hidden_dtype:
            grad_output = grad_output.astype(ctx.hidden_dtype)
        grad_x, grad_dense_probs = ctx.buffer.combine_with_unpermute(
            hidden=grad_output.contiguous(),
            probs=None
            if grad_token_probs is None
            else grad_token_probs.astype("float32"),
            handle=ctx.handle,
            pad_multiple=FP8_ALIGN if ctx.use_fp8_dispatch else None,
        )
        if grad_x.shape[0] != ctx.num_unpadded_tokens:
            grad_x = grad_x[: ctx.num_unpadded_tokens]
            if grad_dense_probs is not None:
                grad_dense_probs = grad_dense_probs[: ctx.num_unpadded_tokens]
        grad_probs = None
        if grad_dense_probs is not None:
            grad_probs = paddle.take_along_axis(
                grad_dense_probs,
                ctx.token_indices,
                axis=-1,
            )
        return grad_x, None, grad_probs


def _replay_hybrid_ep_dispatch_backward(
    buffer,
    handle,
    grad_output,
    num_permuted_tokens,
    use_fp8_dispatch,
):
    replay_handle = handle
    if use_fp8_dispatch:
        replay_config = buffer.update_template_config(
            hidden_dim=grad_output.shape[-1],
            num_of_tokens_per_rank=handle[6],
            num_local_experts=handle[7].num_of_experts_per_rank,
            use_fp8=False,
        )
        replay_handle = (
            *handle[:7],
            replay_config,
            handle[8],
        )
    grad_x, _, _, _, _ = buffer.dispatch_with_permute(
        hidden=grad_output.contiguous(),
        handle=replay_handle,
        num_permuted_tokens=num_permuted_tokens,
        pad_multiple=FP8_ALIGN if use_fp8_dispatch else None,
        non_blocking=False,
    )
    return grad_x[:num_permuted_tokens]


class HybridEPCombine(PyLayer):
    """Fused HybridEP combine bridge for Paddle autograd."""

    @staticmethod
    def forward(ctx, x, manager, num_permuted_tokens=None):
        handle = manager.handle
        if num_permuted_tokens is None:
            num_permuted_tokens = x.shape[0]
        assert x.shape[0] == num_permuted_tokens, (
            "HybridEP combine expects active permuted rows, got "
            f"{x.shape[0]} rows but num_permuted_tokens is "
            f"{num_permuted_tokens}."
        )
        use_fp8_dispatch = "UINT8" in str(handle[7].token_data_type)
        combined_x, _ = manager._active_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=FP8_ALIGN if use_fp8_dispatch else None,
        )
        combined_x.stop_gradient = False
        ctx.buffer = manager._active_buffer
        ctx.handle = handle
        ctx.use_fp8_dispatch = use_fp8_dispatch
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = _replay_hybrid_ep_dispatch_backward(
            ctx.buffer,
            ctx.handle,
            grad_output,
            ctx.num_permuted_tokens,
            ctx.use_fp8_dispatch,
        )
        return grad_x


def hybrid_ep_dispatch(
    x,
    token_indices,
    token_probs,
    manager,
    fp8_dispatch: bool = False,
):
    """Perform HybridEP dispatch_with_permute with explicit Paddle autograd."""
    return HybridEPDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        manager,
        fp8_dispatch,
    )


def hybrid_ep_combine(x, manager, num_permuted_tokens=None):
    """Perform HybridEP combine_with_unpermute with explicit Paddle autograd."""
    return HybridEPCombine.apply(
        x,
        manager,
        num_permuted_tokens,
    )


class DispatchNode:
    def __init__(self, name="dispatch"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(
        self,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        self.group = group
        self.handle = states["handle"]
        self.event = event

        return recv_x, recv_token_probs, states

    def backward(
        self,
        grad_output,
        grad_token_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Backward pass of fused dispatch."""
        out = fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.reset_statue()
        return out


class CombineNode:
    def __init__(self, name="combine"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(
        self,
        x,
        group,
        handle,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused combine."""
        states = {}
        states["handle"] = handle
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        self.handle = handle
        self.group = group
        self.previous_event = previous_event

        return combined_x

    def backward(
        self,
        grad_output,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Backward pass of fused combine."""
        out = fused_combine_backward_func(
            grad_output,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.reset_statue()
        return out
