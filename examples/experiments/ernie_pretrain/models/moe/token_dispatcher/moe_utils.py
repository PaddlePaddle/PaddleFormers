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

import warnings

import numpy as np
import paddle
from paddle import framework


def inplace_offload(x):
    """Offload tensor to CPU in-place to save GPU memory.

    Args:
        x (paddle.Tensor): The tensor to be offloaded to CPU.

    Note:
        This operation modifies the tensor in-place by sharing data with a CPU copy.
    """
    if not x.place._equals(paddle.CPUPlace()):
        y = x.cpu()
        if y is not x:
            x_t = x.value().get_tensor()
            y_t = y.value().get_tensor()
            x_t._share_data_with(y_t)


def inplace_offload_if_needed(x, threshold=2 * 1024 * 1024 * 1024):
    """Conditionally offload tensor to CPU if it exceeds memory threshold.

    Args:
        x (paddle.Tensor): The tensor to potentially offload.
        threshold (int, optional): Memory threshold in bytes. Defaults to 2GB.

    Note:
        Only offloads tensors during gradient computation when memory usage exceeds threshold.
        Issues a warning when offloading occurs.
    """
    if not framework._dygraph_tracer()._has_grad:
        return

    memory_size = np.prod(x.shape) * paddle.core.size_of_dtype(x.dtype)
    if memory_size >= threshold:
        inplace_offload(x)
        warnings.warn(f"Offload tensor with shape: {x.shape}, dtype: {x.dtype}, memory size {memory_size}")


def topk_to_permuted_indices_single(x, num_tokens, expert_id, topk):
    """Convert topk indices to permuted indices for a single expert.

    Args:
        x (paddle.Tensor): Input tensor containing expert assignments.
        num_tokens (int): Number of tokens assigned to this expert.
        expert_id (int): ID of the expert to filter for.
        topk (int): Number of experts selected per token (top-k value).

    Returns:
        tuple: (token_permuted_indices, prob_permuted_indices)
            - token_permuted_indices: Indices of tokens assigned to this expert
            - prob_permuted_indices: Indices of probabilities for the expert assignments
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.tensor.search._restrict_nonzero(x == expert_id, num_tokens).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    """Convert topk indices to permuted indices for all experts.

    Args:
        x (paddle.Tensor): Input tensor containing expert assignments.
        num_tokens_per_expert_list (list[int]): List of token counts per expert.
        topk (int): Number of experts selected per token (top-k value).

    Returns:
        tuple: (token_permuted_indices, prob_permuted_indices)
            - token_permuted_indices: Indices of tokens assigned to experts
            - prob_permuted_indices: Indices of probabilities for all expert assignments
    """
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.concat(
        [
            paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
            for i, total_true_num in enumerate(num_tokens_per_expert_list)
        ]
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def permute(
    tokens,
    token_permuted_indices,
    drop_and_pad: bool = False,
):
    """Permute tokens based on expert assignment indices.

    Args:
        tokens (paddle.Tensor): Input tokens to be permuted.
        token_permuted_indices (paddle.Tensor): Indices for permutation.
        drop_and_pad (bool, optional): Whether to drop and pad tokens. Not supported yet.

    Returns:
        paddle.Tensor: Permuted tokens.

    Raises:
        AssertionError: If drop_and_pad is True (not supported).
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    permuted_input = paddle.gather(tokens, token_permuted_indices)
    return permuted_input


def unpermute(
    permuted_tokens: paddle.Tensor,
    token_permuted_indices: paddle.Tensor,
    prob_permuted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """Restore original token order from permuted tokens.

    Args:
        permuted_tokens (paddle.Tensor): Permuted tokens to be restored.
        token_permuted_indices (paddle.Tensor): Original token positions.
        prob_permuted_indices (paddle.Tensor): Indices for probability values.
        restore_shape (paddle.shape): Original shape of the tensor.
        probs (paddle.Tensor, optional): Probability values for weighted restoration.
        drop_and_pad (bool, optional): Whether to drop and pad tokens. Not supported yet.

    Returns:
        paddle.Tensor: Restored tokens in original order.

    Raises:
        AssertionError: If drop_and_pad is True (not supported).
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape
    if probs is not None:
        permuted_probs = paddle.gather(probs.flatten(), prob_permuted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    output_tokens.scatter_(index=token_permuted_indices, updates=permuted_tokens, overwrite=False)
    return output_tokens


class UnZipNode:
    """Handles the unzipping (high performance permute) of tokens for expert processing in Mixture of Experts,
    in an efficient, deterministic manner.

    This class manages the process of expanding tokens assigned to experts, including:
    - Forward pass: Distributes tokens to experts
    - Backward pass: Collects gradients from experts

    Attributes:
        token_dispatcher: Reference to the parent token dispatcher.
        name (str): Identifier for this node.
        unzipped_probs (paddle.Tensor): Probability values after unzipping.
        zipped_expertwise_rowmap (paddle.Tensor): Mapping between original and expanded tokens.
    """

    def __init__(self, token_dispatcher, name="unzip"):
        """Initialize the UnZipNode.

        Args:
            token_dispatcher: Parent token dispatcher instance.
            name (str, optional): Name identifier. Defaults to "unzip".
        """
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_status(self):
        """Reset internal state between forward/backward passes."""
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    @paddle.no_grad()
    def forward(
        self,
        hs_2d_dispatched,
        dispatched_indices,
        dispatched_probs,
        topk,
        num_experts,
        tokens_per_expert,
    ):
        """Forward pass - distribute tokens to experts.

        Args:
            hs_2d_dispatched (paddle.Tensor): Dispatched hidden states (2D).
            dispatched_indices (paddle.Tensor): Indices of expert assignments.
            dispatched_probs (paddle.Tensor): Routing probabilities.
            topk (int): Number of experts selected per token.
            num_experts (int): Total number of experts.
            tokens_per_expert (int): Tokens allocated per expert.

        Returns:
            tuple: (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs)
                - unzipped_tokens: Expanded tokens for expert processing
                - zipped_expertwise_rowmap: Mapping between original and expanded tokens
                - unzipped_probs: Expanded routing probabilities
        """
        with paddle.amp.auto_cast(False):
            (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, _,) = paddle.nn.functional.moe_permute(
                hs_2d_dispatched,
                None,
                dispatched_indices,
                dispatched_probs,
                num_experts=num_experts,
                tokens_per_expert=tokens_per_expert,
                padding_alignment=128,
            )
        self.unzipped_probs = unzipped_probs
        self.zipped_expertwise_rowmap = zipped_expertwise_rowmap
        return (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
        )

    @paddle.no_grad()
    def backward(self, dx, hidden_states_out_grad, probs_grad, dispatched_indices, num_experts):
        """Backward pass - collect gradients from experts.

        Args:
            dx (paddle.Tensor): Gradient from experts.
            hidden_states_out_grad (paddle.Tensor): Gradient of output hidden states.
            probs_grad (paddle.Tensor): Gradient of routing probabilities.
            dispatched_indices (paddle.Tensor): Original expert assignment indices.
            num_experts (int): Total number of experts.

        Returns:
            tuple: (weighted_zipped_tokens, probs_grad_zipped)
                - weighted_zipped_tokens: Compressed gradients from experts
                - probs_grad_zipped: Compressed probability gradients
        """
        with paddle.amp.auto_cast(False):
            weighted_zipped_tokens, probs_grad_zipped = paddle.nn.functional.moe_unpermute(
                dx,
                self.zipped_expertwise_rowmap,
                dispatched_indices,
                probs_grad,
                total_zipped_tokens=hidden_states_out_grad.shape[0],
                num_experts=num_experts,
            )
        self.reset_status()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    """Handles the zipping (high performance unpermute) of expert outputs in Mixture of Experts,
    in an efficient, deterministic manner.

    This class manages the process of combining expert outputs, including:
    - Forward pass: Combines expert outputs
    - Backward pass: Distributes gradients to experts

    Attributes:
        token_dispatcher: Reference to the parent token dispatcher.
        name (str): Identifier for this node.
    """

    def __init__(self, token_dispatcher, name="zip"):
        """Initialize the ZipNode.

        Args:
            token_dispatcher: Parent token dispatcher instance.
            name (str, optional): Name identifier. Defaults to "zip".
        """
        self.token_dispatcher = token_dispatcher
        self.name = name

    @paddle.no_grad()
    def forward(
        self,
        expert_out,
        zipped_expertwise_rowmap,
        routemap_topk,
        unzipped_probs,
        total_zipped_tokens,
        num_experts,
    ):
        """Forward pass - combine expert outputs.

        Args:
            expert_out (paddle.Tensor): Outputs from all experts.
            zipped_expertwise_rowmap (paddle.Tensor): Mapping between original and expanded tokens.
            routemap_topk (paddle.Tensor): Top-k routing information.
            unzipped_probs (paddle.Tensor): Expanded routing probabilities.
            total_zipped_tokens (int): Total number of original tokens.
            num_experts (int): Total number of experts.

        Returns:
            paddle.Tensor: Combined expert outputs.
        """
        with paddle.amp.auto_cast(False):
            expert_out_zipped, zipped_probs_topk = paddle.nn.functional.moe_unpermute(
                expert_out,
                zipped_expertwise_rowmap,
                routemap_topk,
                unzipped_probs,
                total_zipped_tokens,
                num_experts,
            )
        return expert_out_zipped

    @paddle.no_grad()
    def backward(
        self,
        grad_output,
        dispatched_indices,
        dispatched_probs,
        top_k,
        num_experts,
        tokens_per_expert,
    ):
        """Backward pass - distribute gradients to experts.

        Args:
            grad_output (paddle.Tensor): Gradient of the combined output.
            dispatched_indices (paddle.Tensor): Original expert assignment indices.
            dispatched_probs (paddle.Tensor): Original routing probabilities.
            top_k (int): Number of experts selected per token.
            num_experts (int): Total number of experts.
            tokens_per_expert (int): Tokens allocated per expert.

        Returns:
            paddle.Tensor: Expanded gradients to be sent to experts.
        """
        with paddle.amp.auto_cast(False):
            (unzipped_grad, zipped_expertwise_rowmap_grad, unzipped_probs_grad, _,) = paddle.nn.functional.moe_permute(
                grad_output,
                None,
                dispatched_indices,
                dispatched_probs,
                num_experts,
                tokens_per_expert,
                padding_alignment=128,
            )

        return unzipped_grad
