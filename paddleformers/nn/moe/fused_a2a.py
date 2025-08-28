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

try:
    import paddle.distributed.communication.deep_ep as deep_ep

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import paddle
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group

_buffer = None


def get_hidden_bytes(x: paddle.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (paddle.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.shape[1] * max(x.element_size(), 2)


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
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size), num_rdma_bytes)

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


class FusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group, previous_event=None):
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

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
            topk_weights=token_probs.cast(paddle.float32),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event
        tokens_per_expert = paddle.to_tensor(num_recv_tokens_per_expert_list)

        states = dict()
        states["dispatched_indices"] = recv_token_indices
        states["tokens_per_expert"] = tokens_per_expert
        states["handle"] = handle

        return recv_x, recv_token_probs, states

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.cast(paddle.float32),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs


class FusedCombine(PyLayer):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, states, previous_event=None):
        """Forward pass of fused combine."""
        handle = states["handle"]
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x, handle=handle, async_finish=False, previous_event=None, allocate_on_comm_stream=False
        )
        ctx.handle = handle
        ctx.group = group
        ctx.previous_event = previous_event

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=ctx.previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x


if HAVE_DEEP_EP:

    def fused_dispatch(x, token_indices, token_probs, num_experts, group: Group, previous_event=None):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(x.contiguous(), token_indices, token_probs, num_experts, group, previous_event)

    def fused_combine(x, group, handle, previous_event=None):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        states = dict()
        states["handle"] = handle
        return FusedCombine.apply(x, group, states, previous_event)

else:
    fused_dispatch = None
    fused_combine = None
