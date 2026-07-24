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
import logging
import queue

import paddle
from paddle import _C_ops, framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.nn.functional.flash_attention import flashmask_attention

from paddlefleet_ops import is_flash_mask_available
from paddlefleet_ops.flash_mask_facade import get_fa_version
from paddleformers.fleet.context_parallel_utils import (
    UlyssesAlltoAll,
    _ulysses_fused_supported,
    _ulysses_single_all_to_all,
    _ulysses_single_all_to_all_fused,
    cp_flashmask_allgatherkv_balance_backward,
    cp_flashmask_allgatherkv_balance_forward,
    cp_flashmask_swa_p2p_backward,
    cp_flashmask_swa_p2p_forward,
)
from paddleformers.fleet.refined_recompute.queue_check import (
    global_rr_queue_log,
)

if is_flash_mask_available():
    from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
        FlashMaskInfoPaddle,
    )
    from paddlefleet_ops.flash_mask.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )

logger = logging.getLogger(__name__)


def flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16):
    """
    A utility function to ensure that the Query, Key, and Value tensors
    are cast to a specific data type (typically bfloat16) before being
    passed to the FlashAttention kernel, which often requires a specific precision.

    Args:
        q (paddle.Tensor): The query tensor.
        k (paddle.Tensor): The key tensor.
        v (paddle.Tensor): The value tensor.
        dtype (paddle.dtype, optional): The target data type. Defaults to paddle.bfloat16.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: The casted Q, K, and V tensors.
    """
    if q.dtype != dtype:
        q = q.astype(dtype)
    if k.dtype != dtype:
        k = k.astype(dtype)
    if v.dtype != dtype:
        v = v.astype(dtype)
    return q, k, v


class FlashAttnFunctor(PyLayer):
    """
    A custom PyLayer designed for the refined recompute strategy.

    This class does not perform any actual computation in its forward pass. Instead,
    it serves as a "surrogate" or "fake" layer during the second forward pass of
    recomputation. Its primary role is to reconstruct the computation graph,
    allowing PaddlePaddle's autograd engine to correctly execute the custom
    backward pass defined here.
    """

    @staticmethod
    def forward(ctx, q, k, v, hold_tensors):
        """
        The forward pass of the surrogate layer. It simply retrieves the pre-computed
        attention output from the `hold_tensors` dictionary and saves all necessary
        tensors for the backward pass using `ctx.save_for_backward`.

        Args:
            ctx (Context): The context object to save tensors for backward.
            q, k, v (paddle.Tensor): The input tensors from the second forward pass.
            hold_tensors (dict): A dictionary containing intermediate results from the
                                 first forward pass (e.g., the actual attention output, softmax_lse).

        Returns:
            paddle.Tensor: The pre-computed attention output.
        """

        # startend_row_indices is None
        fa_version = get_fa_version(q.shape[-1], v.shape[-1])
        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")

        # Save the necessary tensors that will be needed to compute the gradient.
        if fa_version == 2:
            result_attention = hold_tensors["result_attention"]
            result_softmax = hold_tensors["result_softmax"]
            softmax_lse = hold_tensors["softmax_lse"]
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        elif fa_version == 3:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q, k, v, result_attention, softmax_lse, causal
            )
        elif fa_version == 4:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q, k, v, result_attention, softmax_lse, causal
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Return the actual output computed during the first forward pass.
        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for FlashAttention.
        It retrieves the saved tensors from the context and calls the low-level
        C++ gradient operator (`_C_ops.flash_attn_grad`) to compute the gradients
        for Q, K, and V.

        Args:
            ctx (Context): The context object to retrieve saved tensors.
            grad (paddle.Tensor): The gradient of the output tensor.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: The gradients for Q, K, and V.
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            # Call the underlying C++ gradient kernel.
            q_grad, k_grad, v_grad = _C_ops.flash_attn_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                seed_offset,
                None,  # attn_mask (dense mask)
                grad,
                dropout,
                causal,
            )
            seed_offset._clear_dataptr()
        elif fa_version == 3:
            q, k, v, result_attention, softmax_lse, causal = ctx.saved_tensor()
            q_grad, k_grad, v_grad = _C_ops.flash_attn_v3_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                grad,
                q.shape[-1] ** (-0.5)
                if ctx.softmax_scale is None
                else ctx.softmax_scale,  # softmax_scale
                causal,
                -1,  # window_size_left
                -1,  # window_size_right
                0.0,  # softcap
                0,  # sm_margin
            )
        elif fa_version == 4:
            flashmask_info = None
            q, k, v, result_attention, softmax_lse, causal = ctx.saved_tensor()
            q_grad, k_grad, v_grad, _ = _flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                grad,
                softmax_lse,
                flashmask_info,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
                deterministic=bool(
                    paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                        "FLAGS_cudnn_deterministic"
                    ]
                ),
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Manually release memory of intermediate tensors to save GPU memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashAttention:
    """
    Implements the refined recompute strategy for standard (non-masked) FlashAttention.
    This class is designed to be used within a `recompute` block.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "flash_attention")

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
        softmax_scale=None,
    ):
        """
        The main entry point for the forward pass.
        It checks if autograd is active. If not, it executes the first forward pass.
        If autograd is active (which happens during recomputation's backward pass),
        it executes the second forward pass.
        """
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output, attn_weights = self._first_fwd(
                query_states,
                key_states,
                value_states,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
                softmax_scale=softmax_scale,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output, attn_weights = self._second_fwd(
                query_states, key_states, value_states
            )

        return attn_output, attn_weights

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
        softmax_scale=None,
    ):
        """
        The first forward pass. It runs the actual FlashAttention computation
        without tracking gradients (`@paddle.no_grad()`). It saves the necessary
        intermediate tensors for the backward pass into a queue and returns the final output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        # startend_row_indices is None
        fa_version = get_fa_version(
            query_states.shape[-1], value_states.shape[-1]
        )
        if fa_version == 2:
            if softmax_scale is not None:
                raise NotImplementedError(
                    "fa_version==2 does not support setting softmax_scale"
                )
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flash_attn(
                    query_states,
                    key_states,
                    value_states,
                    None,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
            # Store all tensors needed for the backward pass in a dictionary.
            hold_tensors = {
                "result_attention": result_attention,
                "result_softmax": result_softmax,
                "softmax_lse": softmax_lse,
                "seed_offset": seed_offset,
                "dropout": dropout,
                "causal": causal,
            }
        elif fa_version == 3:
            (result_attention, softmax_lse) = _C_ops.flash_attn_v3(
                query_states,
                key_states,
                value_states,
                None,
                None,
                None,
                None,
                query_states.shape[-1] ** (-0.5)
                if softmax_scale is None
                else softmax_scale,
                causal,
                -1,
                -1,
                0.0,
                1,
                False,
                False,
                0,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
            result_softmax = None  # FA v3 does not return softmax.
        elif fa_version == 4:
            (result_attention, softmax_lse) = _flash_attn_fwd(
                query_states,
                key_states,
                value_states,
                causal=causal,
                return_lse=True,
                startend_row_indices=None,
                softmax_scale=softmax_scale,
                pack_gqa=False,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
            result_softmax = None
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Put the dictionary of saved tensors into the queue.
        self._hold_tensors_queue.put(hold_tensors)
        return result_attention, result_softmax if return_softmax else None

    def _second_fwd(self, query_states, key_states, value_states):
        """
        The second forward pass. It retrieves the saved tensors from the queue
        and passes them to the `FlashAttnFunctor` surrogate layer. This action
        reconstructs the computation graph, enabling the custom backward pass to run.
        """
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        # Call the surrogate PyLayer to link the backward pass.
        output = FlashAttnFunctor.apply(
            query_states, key_states, value_states, hold_tensors
        )
        return output, hold_tensors.get(
            "result_softmax"
        )  # Use .get for safety with FA v3

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)


class FlashMaskAttnFunctor(PyLayer):
    """
    A custom PyLayer for the **masked** version of FlashAttention.

    This class serves the same purpose as `FlashAttnFunctor` but is tailored for
    the `flashmask_attention` operator, which takes an additional `startend_row_indices`
    tensor to handle variable-length sequences or sparse attention patterns.
    """

    @staticmethod
    def forward(
        ctx, q, k, v, startend_row_indices, learnable_sink, hold_tensors
    ):
        """
        The forward pass for the masked attention surrogate layer.
        It saves all necessary tensors, including `startend_row_indices`, for the backward pass.
        """
        fa_version = get_fa_version(
            q.shape[-1], v.shape[-1], startend_row_indices
        )
        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )

        if fa_version == 2:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        elif fa_version == 3:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
            )
        elif fa_version == 4:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
                learnable_sink,
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for masked FlashAttention.
        It calls the corresponding low-level C++ gradient operator (`_C_ops.flashmask_attention_grad`).
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            # Call the underlying C++ gradient kernel for masked attention.
            q_grad, k_grad, v_grad = _C_ops.flashmask_attention_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                grad,
                dropout,
                causal,
            )
            seed_offset._clear_dataptr()
        elif fa_version == 3:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
            ) = ctx.saved_tensor()

            sig_params = inspect.signature(flashmask_attention).parameters

            softmax_scale = (
                q.shape[-1] ** (-0.5)
                if ctx.softmax_scale is None
                else ctx.softmax_scale
            )

            if "group" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    softmax_scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif "block_mask" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    softmax_scale,
                    causal,
                )
            else:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    grad,
                    softmax_scale,
                    causal,
                )
        elif fa_version == 4:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
                learnable_sink,
            ) = ctx.saved_tensor()
            if startend_row_indices is not None:
                flashmask_info = FlashMaskInfoPaddle(
                    startend_row_indices=startend_row_indices,
                    is_causal=causal,
                )
            else:
                flashmask_info = None
            q_grad, k_grad, v_grad, sink_grad = _flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                grad,
                softmax_lse,
                flashmask_info,
                learnable_sink=learnable_sink,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
                deterministic=bool(
                    paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                        "FLAGS_cudnn_deterministic"
                    ]
                ),
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Manually release memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: q(0)/k(1)/v(2)/startend_row_indices(3)/learnable_sink(4).
        # startend_row_indices is stop_gradient=True, so its slot (position 3)
        # must be None -- sink_grad belongs in position 4. A fixed off-by-one
        # sink is also stop_gradient=True, so for it the 3-tuple is correct.
        if ctx.sink_requires_grad:
            return q_grad, k_grad, v_grad, None, sink_grad
        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashMaskAttention:
    """
    Implements the refined recompute strategy for masked FlashAttention.
    This class is designed to be used within a `recompute` block.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(
            self._hold_tensors_queue, "flashmask_attention"
        )

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
        learnable_sink=None,
        softmax_scale=None,
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if learnable_sink is not None:
            fa_version = get_fa_version(
                query_states.shape[-1],
                value_states.shape[-1],
                startend_row_indices,
            )
            if fa_version != 4:
                raise NotImplementedError(
                    "learnable_sink only supported on fa_version==4 cute backend"
                )
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output = self._first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                learnable_sink=learnable_sink,
            )

        return attn_output

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
        learnable_sink=None,
        softmax_scale=None,
    ):
        """
        The first forward pass for masked attention. It runs the actual computation,
        saves intermediate tensors to the queue, and returns the output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        fa_version = get_fa_version(
            query_states.shape[-1], value_states.shape[-1], startend_row_indices
        )
        if fa_version == 2:
            if softmax_scale is not None:
                raise NotImplementedError(
                    "fa_version==2 does not support setting softmax_scale"
                )
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flashmask_attention(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "seed_offset": seed_offset,
                "result_softmax": result_softmax,
                "dropout": dropout,
                "causal": causal,
            }
        elif fa_version == 3:
            sig_params = inspect.signature(flashmask_attention).parameters
            scale = (
                query_states.shape[-1] ** (-0.5)
                if softmax_scale is None
                else softmax_scale
            )
            if "group" in sig_params:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,  # block_mask
                    None,  # nvshmem unique id
                    scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif "block_mask" in sig_params:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,  # block_mask
                    scale,
                    causal,
                )
            else:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    scale,
                    causal,
                )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
        elif fa_version == 4:
            (result_attention, softmax_lse) = _flash_attn_fwd(
                query_states,
                key_states,
                value_states,
                causal=causal,
                return_lse=True,
                startend_row_indices=startend_row_indices,
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
                pack_gqa=False,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "learnable_sink": learnable_sink,
                "softmax_scale": softmax_scale,
            }
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention

    def _second_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        learnable_sink=None,
    ):
        """
        The second forward pass for masked attention. It reconstructs the graph
        by calling the `FlashMaskAttnFunctor` surrogate layer.
        """
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        output = FlashMaskAttnFunctor.apply(
            query_states,
            key_states,
            value_states,
            startend_row_indices,
            learnable_sink,
            hold_tensors,
        )
        return output

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)


class FlashMaskAttnCpFunctor(PyLayer):
    """
    A custom PyLayer for the **masked** version of FlashAttention.

    This class serves the same purpose as `FlashAttnFunctor` but is tailored for
    the `flashmask_attention` operator, which takes an additional `startend_row_indices`
    tensor to handle variable-length sequences or sparse attention patterns.
    """

    @staticmethod
    def forward(ctx, q, k, v, learnable_sink, hold_tensors):
        """
        The forward pass for the masked attention surrogate layer.
        It saves all necessary tensors, including `startend_row_indices`, for the backward pass.
        """

        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        startend_row_indices = hold_tensors["startend_row_indices"]
        fa_version = hold_tensors["fa_version"]
        group = hold_tensors["group"]
        causal = hold_tensors["causal"]
        mode = hold_tensors["mode"]

        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")
        ctx.mode = mode
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
            learnable_sink,
        )

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for masked FlashAttention.
        It calls the corresponding low-level C++ gradient operator (`_C_ops.flashmask_attention_grad`).
        """
        # Retrieve saved tensors
        (
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
            learnable_sink,
        ) = ctx.saved_tensor()
        fa_version = ctx.fa_version

        # Compute gradients
        query_grad, key_grad, value_grad, sink_grad = (
            cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                grad,
                learnable_sink,
                group,
                causal,
                fa_version,
                ctx.softmax_scale,  # softmax_scale
                ctx.mode,
            )
        )

        # Manually release memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: q(0)/k(1)/v(2)/learnable_sink(3). A trainable sink needs its
        # grad returned in slot 3; a fixed off-by-one sink (stop_gradient=True)
        # or sink=None uses the 3-tuple.
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, sink_grad
        return query_grad, key_grad, value_grad


def _ulysses_single_all_to_all_rr(
    input, scatter_idx, gather_idx, batch_dim_idx, group
):
    if _ulysses_fused_supported(scatter_idx, batch_dim_idx, input):
        return _ulysses_single_all_to_all_fused(input, scatter_idx, group)
    return _ulysses_single_all_to_all(
        input, scatter_idx, gather_idx, batch_dim_idx, group
    )


class FlashMaskUlyssesCpFunctor(PyLayer):
    """Surrogate PyLayer for RR Ulysses FlashMask CP attention."""

    @staticmethod
    def forward(ctx, q, k, v, hold_tensors):
        """Return saved final output and keep local tensors for backward."""
        local_hold_tensors = hold_tensors["local_hold_tensors"]
        ctx.group = hold_tensors["group"]
        ctx.softmax_scale = local_hold_tensors.get("softmax_scale")
        ctx.fa_version = local_hold_tensors["fa_version"]
        ctx.dropout = local_hold_tensors.get("dropout", 0.0)
        if ctx.fa_version == 2:
            ctx.save_for_backward(
                hold_tensors["local_query"],
                hold_tensors["local_key"],
                hold_tensors["local_value"],
                hold_tensors["startend_row_indices"],
                local_hold_tensors["result_attention"],
                local_hold_tensors["softmax_lse"],
                local_hold_tensors["causal"],
                local_hold_tensors["seed_offset"],
            )
        else:
            ctx.save_for_backward(
                hold_tensors["local_query"],
                hold_tensors["local_key"],
                hold_tensors["local_value"],
                hold_tensors["startend_row_indices"],
                local_hold_tensors["result_attention"],
                local_hold_tensors["softmax_lse"],
                local_hold_tensors["causal"],
            )
        return hold_tensors["result_attention"]

    @staticmethod
    def backward(ctx, grad):
        """Run local attention backward, then map local grads to input layout."""
        if ctx.fa_version == 2:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
                seed_offset,
            ) = ctx.saved_tensor()
        else:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
            ) = ctx.saved_tensor()
            seed_offset = None
        group = ctx.group
        local_grad = _ulysses_single_all_to_all_rr(
            grad,
            scatter_idx=2,
            gather_idx=1,
            batch_dim_idx=0,
            group=group,
        )
        if ctx.fa_version == 2:
            q_grad, k_grad, v_grad = _C_ops.flashmask_attention_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                local_grad,
                ctx.dropout,
                causal,
            )
            seed_offset._clear_dataptr()
        elif ctx.fa_version == 3:
            sig_params = inspect.signature(flashmask_attention).parameters
            scale = (
                q.shape[-1] ** (-0.5)
                if ctx.softmax_scale is None
                else ctx.softmax_scale
            )
            if "group" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,
                    local_grad,
                    scale,
                    causal,
                    0,
                    1,
                )
            elif "block_mask" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,
                    local_grad,
                    scale,
                    causal,
                )
            else:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    local_grad,
                    scale,
                    causal,
                )
        elif ctx.fa_version == 4:
            if startend_row_indices is not None:
                flashmask_info = FlashMaskInfoPaddle(
                    startend_row_indices=startend_row_indices,
                    is_causal=causal,
                )
            else:
                flashmask_info = None
            q_grad, k_grad, v_grad, _ = _flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                local_grad,
                softmax_lse,
                flashmask_info,
                learnable_sink=None,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
                deterministic=bool(
                    paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                        "FLAGS_cudnn_deterministic"
                    ]
                ),
            )
        else:
            raise ValueError(
                f"Invalid flash attention version: {ctx.fa_version}"
            )

        query_grad = _ulysses_single_all_to_all_rr(
            q_grad,
            scatter_idx=1,
            gather_idx=2,
            batch_dim_idx=0,
            group=group,
        )
        key_grad = _ulysses_single_all_to_all_rr(
            k_grad,
            scatter_idx=1,
            gather_idx=2,
            batch_dim_idx=0,
            group=group,
        )
        value_grad = _ulysses_single_all_to_all_rr(
            v_grad,
            scatter_idx=1,
            gather_idx=2,
            batch_dim_idx=0,
            group=group,
        )
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()
        return query_grad, key_grad, value_grad


class FlashMaskSwaP2PFunctor(PyLayer):
    """Surrogate PyLayer for RR P2P SWA FlashMask attention."""

    @staticmethod
    def forward(ctx, q, k, v, learnable_sink, hold_tensors):
        """Return saved first-forward output and keep tensors for backward."""
        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        recv_key = hold_tensors["recv_key"]
        recv_value = hold_tensors["recv_value"]
        startend_row_indices = hold_tensors["startend_row_indices"]
        group = hold_tensors["group"]
        causal = hold_tensors["causal"]

        ctx.learnable_sink = learnable_sink
        ctx.softmax_scale = hold_tensors.get("softmax_scale")
        ctx.window_size = hold_tensors["window_size"]
        ctx.group = group
        ctx.causal = causal
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            recv_key,
            recv_value,
            result_attention,
            softmax_lse,
            startend_row_indices,
        )
        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """Run explicit P2P SWA FlashMask backward from saved tensors."""
        (
            q,
            k,
            v,
            recv_key,
            recv_value,
            result_attention,
            softmax_lse,
            startend_row_indices,
        ) = ctx.saved_tensor()
        query_grad, key_grad, value_grad, grad_sink = (
            cp_flashmask_swa_p2p_backward(
                q,
                k,
                v,
                recv_key,
                recv_value,
                startend_row_indices,
                result_attention,
                softmax_lse,
                grad,
                ctx.learnable_sink,
                ctx.group,
                ctx.causal,
                ctx.softmax_scale,
                ctx.window_size,
            )
        )
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, grad_sink
        return query_grad, key_grad, value_grad


def slice_ulysses_mask_heads(startend_row_indices, num_k_heads, cp_group):
    """Slice per-head FlashMask indices for the local Ulysses head shard."""
    num_mask_heads = startend_row_indices.shape[1]
    assert num_mask_heads == 1 or num_mask_heads == num_k_heads, (
        f"startend_row_indices head dim must be 1 or num_kv_heads ({num_k_heads}), "
        f"got {num_mask_heads}"
    )
    if num_mask_heads == 1:
        return startend_row_indices
    assert num_mask_heads % cp_group.nranks == 0, (
        f"startend_row_indices head dim ({num_mask_heads}) must be divisible "
        f"by cp_size ({cp_group.nranks}) for Ulysses"
    )
    heads_per_rank = num_mask_heads // cp_group.nranks
    head_start = cp_group.rank * heads_per_rank
    return startend_row_indices[
        :, head_start : head_start + heads_per_rank, :, :
    ]


def ulysses_local_flashmask_first_fwd(
    query_states,
    key_states,
    value_states,
    startend_row_indices,
    causal,
    softmax_scale,
):
    """Run local Ulysses FlashMask first forward and save RR tensors."""
    fa_version = get_fa_version(
        query_states.shape[-1], value_states.shape[-1], startend_row_indices
    )
    if fa_version == 2:
        if softmax_scale is not None:
            raise NotImplementedError(
                "fa_version==2 does not support setting softmax_scale"
            )
        result_attention, result_softmax, softmax_lse, seed_offset = (
            _C_ops.flashmask_attention(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                None,
                0.0,
                causal,
                False,
                False,
                "",
            )
        )
        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "seed_offset": seed_offset,
            "result_softmax": result_softmax,
            "dropout": 0.0,
            "causal": causal,
        }
    elif fa_version == 3:
        sig_params = inspect.signature(flashmask_attention).parameters
        scale = (
            query_states.shape[-1] ** (-0.5)
            if softmax_scale is None
            else softmax_scale
        )
        if "group" in sig_params:
            result_attention, softmax_lse = _C_ops.flashmask_attention_v2(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                None,
                None,
                scale,
                causal,
                0,
                1,
            )
        elif "block_mask" in sig_params:
            result_attention, softmax_lse = _C_ops.flashmask_attention_v2(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                None,
                scale,
                causal,
            )
        else:
            result_attention, softmax_lse = _C_ops.flashmask_attention_v2(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                scale,
                causal,
            )
        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "causal": causal,
            "softmax_scale": softmax_scale,
        }
    elif fa_version == 4:
        result_attention, softmax_lse = _flash_attn_fwd(
            query_states,
            key_states,
            value_states,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
            softmax_scale=softmax_scale,
        )
        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "causal": causal,
            "softmax_scale": softmax_scale,
        }
    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")
    hold_tensors["fa_version"] = fa_version
    return result_attention, hold_tensors


class RefinedRcomputeFlashMaskCpAttention:
    """
    Implements the refined recompute strategy for masked FlashAttention.
    This class is designed to be used within a `recompute` block in Context Parallel.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(
            self._hold_tensors_queue, "flashmask_attention_rr"
        )

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        mode="dualchunk_allgather",
        learnable_sink=None,
        softmax_scale=None,
        window_size=None,
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if learnable_sink is not None:
            fa_version = get_fa_version(
                query_states.shape[-1],
                value_states.shape[-1],
                startend_row_indices,
            )
            if not fa_version == 4:
                raise NotImplementedError(
                    "learnable_sink only supported on fa_version==4 cute backend"
                )
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output = self._first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                fixed_seed_offset=fixed_seed_offset,
                dropout=dropout,
                causal=causal,
                training=training,
                mode=mode,
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states,
                key_states,
                value_states,
                learnable_sink=learnable_sink,
            )

        return attn_output

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        mode="dualchunk_allgather",
        learnable_sink=None,
        softmax_scale=None,
        window_size=None,
    ):
        """
        The first forward pass for masked attention. It runs the actual computation,
        saves intermediate tensors to the queue, and returns the output.
        """

        # Validate input parameters
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in FlashMask context parallel yet."
            )

        if causal and mode != "contiguous_a2a":
            raise NotImplementedError(
                "FlashMaskContextParallel does not support causal=True for mode other than 'contiguous_a2a'"
            )

        if fixed_seed_offset is not None:
            raise NotImplementedError("Fixed seed offset is not supported yet.")

        # Get communication group
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

        if mode == "dualchunk_allgather":
            assert query_states.shape[1] % 2 == 0, (
                f"Query sequence length must be divisible by 2. "
                f"FlashMaskContextParallel uses DualChunkSwap strategy for load balancing. "
                f"Current query sequence length: {query_states.shape[1]}"
            )

        if mode in ("dualchunk_allgather", "contiguous_allgather"):
            result_attention, softmax_lse, startend_row_indices, fa_version = (
                cp_flashmask_allgatherkv_balance_forward(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    learnable_sink,
                    group,
                    causal,
                    training,
                    softmax_scale,
                    mode,
                )
            )
            hold_tensors = {
                "mode": mode,
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "startend_row_indices": startend_row_indices,
                "fa_version": fa_version,
                "group": group,
                "causal": causal,
                "learnable_sink": learnable_sink,
                "softmax_scale": softmax_scale,
            }

            self._hold_tensors_queue.put(hold_tensors)
            return result_attention
        elif mode == "contiguous_swap2p":
            assert is_flash_mask_available(), (
                "P2P SWA fast path requires flashmask installed. Please check."
            )
            if window_size is None or window_size <= 0:
                raise ValueError(
                    f"SWA P2P window_size must be positive, got {window_size}"
                )
            (
                result_attention,
                softmax_lse,
                recv_key,
                recv_value,
                startend_row_indices,
            ) = cp_flashmask_swa_p2p_forward(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                learnable_sink,
                group,
                causal,
                training,
                softmax_scale,
                window_size,
            )
            hold_tensors = {
                "mode": mode,
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "recv_key": recv_key,
                "recv_value": recv_value,
                "startend_row_indices": startend_row_indices,
                "group": group,
                "causal": causal,
                "learnable_sink": learnable_sink,
                "softmax_scale": softmax_scale,
                "window_size": window_size,
            }
            self._hold_tensors_queue.put(hold_tensors)
            return result_attention
        elif mode == "contiguous_a2a":
            return self._ulysses_first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                group,
                causal,
                learnable_sink,
                softmax_scale,
            )
        else:
            raise ValueError(f"invalid cp_balance_mode: {mode}")

    def _ulysses_alltoall_qkv(self, query, key, value, group):
        """Redistribute Q/K/V from sequence shards to Ulysses head shards."""
        query = UlyssesAlltoAll.apply(
            query, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )
        key = UlyssesAlltoAll.apply(
            key, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )
        value = UlyssesAlltoAll.apply(
            value, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )
        return query, key, value

    def _ulysses_alltoall_output(self, output, group):
        """Redistribute Ulysses local output back to sequence shards."""
        return UlyssesAlltoAll.apply(
            output,
            scatter_idx=1,
            gather_idx=2,
            batch_dim_idx=0,
            group=group,
        )

    def _ulysses_first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        group,
        causal,
        learnable_sink,
        softmax_scale,
    ):
        """Run first forward for RR Ulysses FlashMask CP."""
        if learnable_sink is not None:
            raise NotImplementedError(
                "flashmask_attention_ulysses does not support learnable_sink "
                "(softmax sink)"
            )
        if softmax_scale is not None:
            raise NotImplementedError(
                "flashmask_attention_ulysses does not support setting softmax_scale"
            )
        num_q_heads = query_states.shape[2]
        num_k_heads = key_states.shape[2]
        num_v_heads = value_states.shape[2]
        assert num_q_heads == num_k_heads == num_v_heads, (
            f"Ulysses a2a CP requires q_heads == k_heads == v_heads, "
            f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
        )
        assert num_q_heads % group.nranks == 0, (
            f"num_heads ({num_q_heads}) must be divisible by cp_size ({group.nranks}) for Ulysses"
        )

        startend_row_indices = slice_ulysses_mask_heads(
            startend_row_indices, num_k_heads, group
        )
        query, key, value = self._ulysses_alltoall_qkv(
            query_states, key_states, value_states, group
        )
        local_attention, local_hold_tensors = ulysses_local_flashmask_first_fwd(
            query,
            key,
            value,
            startend_row_indices,
            causal,
            softmax_scale,
        )
        result_attention = self._ulysses_alltoall_output(local_attention, group)
        self._hold_tensors_queue.put(
            {
                "mode": "contiguous_a2a",
                "group": group,
                "result_attention": result_attention,
                "local_query": query,
                "local_key": key,
                "local_value": value,
                "startend_row_indices": startend_row_indices,
                "local_hold_tensors": local_hold_tensors,
            }
        )
        return result_attention

    def _second_fwd(
        self, query_states, key_states, value_states, learnable_sink=None
    ):
        """
        The second forward pass for masked attention. It reconstructs the graph
        by calling the mode-specific surrogate layer.
        """
        hold_tensors = self._hold_tensors_queue.get()
        mode = hold_tensors["mode"]
        if mode in ("dualchunk_allgather", "contiguous_allgather"):
            return FlashMaskAttnCpFunctor.apply(
                query_states,
                key_states,
                value_states,
                learnable_sink,
                hold_tensors,
            )
        elif mode == "contiguous_swap2p":
            return FlashMaskSwaP2PFunctor.apply(
                query_states,
                key_states,
                value_states,
                learnable_sink,
                hold_tensors,
            )
        elif mode == "contiguous_a2a":
            return FlashMaskUlyssesCpFunctor.apply(
                query_states, key_states, value_states, hold_tensors
            )
        else:
            raise ValueError(f"invalid cp_balance_mode: {mode}")

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)
