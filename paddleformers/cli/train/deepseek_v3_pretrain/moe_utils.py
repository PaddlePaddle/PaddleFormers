# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

try:
    import paddlefleet as paddlefleet
except ImportError:
    paddlefleet = None

from paddleformers.transformers.fp8_utils import FP8LinearFunctionBase

if not hasattr(paddle.Tensor, "_clear_to_zero_allocation"):

    def _clear_to_zero_allocation(self):
        """
        _clear_to_zero_allocation
        """
        old_shape = self.shape
        dst = paddle.empty([0], dtype=self.dtype)
        dst_t = dst.value().get_tensor()
        src_t = self.value().get_tensor()
        src_t._share_data_with(dst_t)
        src_t._set_dims(old_shape)

    setattr(paddle.Tensor, "_clear_to_zero_allocation", _clear_to_zero_allocation)


if not hasattr(paddle.Tensor, "_holder_size"):

    def _holder_size(self):
        """
        _holder_size
        """
        if self._is_initialized():
            return int(np.prod(self.shape)) * paddle.core.size_of_dtype(self.dtype)
        else:
            return 0

    setattr(paddle.Tensor, "_holder_size", _holder_size)


def topk_to_permuted_indices(x, num_tokens_per_expert_list, topk):
    x = paddle.flatten(x)
    prob_permuted_indices = paddle.cat(
        [
            paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
            for i, total_true_num in enumerate(num_tokens_per_expert_list)
        ]
    ).flatten()
    token_permuted_indices = prob_permuted_indices // topk
    return token_permuted_indices, prob_permuted_indices


def permute_fast(
    tokens,
    token_permuted_indices,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    # permuted_input = paddle.gather(tokens, token_permuted_indices)
    permuted_input = tokens.index_select(axis=0, index=token_permuted_indices)
    return permuted_input


def unpermute_fast(
    permuted_tokens: paddle.Tensor,
    token_permuted_indices: paddle.Tensor,
    prob_permuted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        token_permuted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape
    if probs is not None:
        permuted_probs = paddle.gather(probs.flatten(), prob_permuted_indices)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions

    output_tokens.put_along_axis_(
        axis=0,
        indices=token_permuted_indices.unsqueeze(1).expand([-1, hidden]),
        values=permuted_tokens,
        reduce="add",
        include_self=True,
    )
    return output_tokens


class UnZipNode:
    def __init__(self, name="unzip"):
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_statue(self):
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
        if isinstance(hs_2d_dispatched, tuple):
            with paddle.amp.auto_cast(False):
                (
                    unzipped_tokens,
                    zipped_expertwise_rowmap,
                    unzipped_probs,
                    unzipped_scale,
                ) = paddle.nn.functional.moe_permute(
                    hs_2d_dispatched[0],
                    hs_2d_dispatched[1],
                    dispatched_indices,
                    dispatched_probs,
                    num_experts=num_experts,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                )
        else:
            with paddle.amp.auto_cast(False):
                (
                    unzipped_tokens,
                    zipped_expertwise_rowmap,
                    unzipped_probs,
                    unzipped_scale,
                ) = paddle.nn.functional.moe_permute(
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
        return (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_scale)

    @paddle.no_grad()
    def backward(self, dx, total_zipped_tokens, probs_grad, dispatched_indices, num_experts):
        with paddle.amp.auto_cast(False):
            weighted_zipped_tokens, probs_grad_zipped = paddle.nn.functional.moe_unpermute(
                dx,
                self.zipped_expertwise_rowmap,
                dispatched_indices,
                probs_grad,
                total_zipped_tokens=total_zipped_tokens,
                num_experts=num_experts,
            )
        self.reset_statue()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    def __init__(self, name="zip"):
        self.name = name

    @paddle.no_grad()
    def forward(
        self, expert_out, zipped_expertwise_rowmap, routemap_topk, unzipped_probs, total_zipped_tokens, num_experts
    ):
        with paddle.amp.auto_cast(False):
            expert_out_zipped, zipped_probs_topk = paddle.nn.functional.moe_unpermute(
                expert_out, zipped_expertwise_rowmap, routemap_topk, unzipped_probs, total_zipped_tokens, num_experts
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
        if isinstance(grad_output, tuple):
            with paddle.amp.auto_cast(False):
                (
                    unzipped_grad,
                    zipped_expertwise_rowmap_grad,
                    unzipped_probs_grad,
                    unzipped_scale_grad,
                ) = paddle.nn.functional.moe_permute(
                    grad_output[0],
                    grad_output[1],
                    dispatched_indices,
                    dispatched_probs,
                    num_experts,
                    tokens_per_expert,
                    padding_alignment=128,
                )
                return (unzipped_grad, unzipped_scale_grad)
        else:
            with paddle.amp.auto_cast(False):
                (
                    unzipped_grad,
                    zipped_expertwise_rowmap_grad,
                    unzipped_probs_grad,
                    unzipped_scale_grad,
                ) = paddle.nn.functional.moe_permute(
                    grad_output,
                    None,
                    dispatched_indices,
                    dispatched_probs,
                    num_experts,
                    tokens_per_expert,
                    padding_alignment=128,
                )

        return unzipped_grad


class PermuteNode:
    def __init__(self, token_dispatcher, name="permute"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def reset_status(self):
        self.token_permuted_indices = None
        self.prob_permuted_indices = None

    def forward(self, hidden_states, hidden_states_scale, dispatched_indices):
        self.token_dispatcher._comm_manager.hidden_shape_before_permute = hidden_states.shape
        self.hidden_shape_before_permute = hidden_states.shape
        self.token_permuted_indices, self.prob_permuted_indices = topk_to_permuted_indices(
            dispatched_indices,
            self.token_dispatcher._comm_manager.tokens_per_expert,
            self.token_dispatcher._comm_manager.router_topk,
        )
        hidden_states = permute_fast(hidden_states, self.token_permuted_indices)
        # permute scale
        hidden_states_scale = permute_fast(hidden_states_scale, self.token_permuted_indices)

        return hidden_states, hidden_states_scale, self.token_permuted_indices, self.prob_permuted_indices

    def backward(self, out_grad, dispatched_probs):
        input_dtype = out_grad.dtype
        hidden_states_grad = unpermute_fast(
            permuted_tokens=out_grad,
            token_permuted_indices=self.token_permuted_indices,
            prob_permuted_indices=self.prob_permuted_indices,
            restore_shape=self.hidden_shape_before_permute,
            probs=dispatched_probs,
        )
        self.reset_status()
        return hidden_states_grad.to(input_dtype)


class UnPermuteNode:
    def __init__(self, token_dispatcher, name="unpermute"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def reset_status(self):
        self.token_permuted_indices = None
        self.hidden_states = None
        self.prob_permuted_indices = None
        self.faltten_dispatched_probs = None
        self.hidden = None
        self.permuted_tokens = None
        self.output_tokens = None

    def forward(
        self,
        hidden_states,
        token_permuted_indices,
        prob_permuted_indices,
        dispatched_probs,
    ):
        self.token_permuted_indices = token_permuted_indices
        self.input_dtype = hidden_states.dtype
        self.hidden_states = hidden_states
        self.prob_permuted_indices = prob_permuted_indices
        self.dispatched_probs_shape = dispatched_probs.shape
        # permute
        _, self.hidden = self.token_dispatcher._comm_manager.hidden_shape_before_permute

        self.faltten_dispatched_probs = dispatched_probs.flatten()

        self.permuted_probs = paddle.gather(self.faltten_dispatched_probs, self.prob_permuted_indices)
        permuted_tokens = self.hidden_states * self.permuted_probs.unsqueeze(-1)
        permuted_tokens = permuted_tokens.cast(self.hidden_states.dtype)

        # Create an output tensor filled with zeros
        output_tokens = paddle.zeros(
            self.token_dispatcher._comm_manager.hidden_shape_before_permute, dtype=self.hidden_states.dtype
        )
        # Scatter add the permuted_input back to the original positions
        output_tokens.put_along_axis_(
            axis=0,
            indices=self.token_permuted_indices.cast("int32").unsqueeze(1).expand([-1, self.hidden]),
            values=permuted_tokens,
            reduce="add",
            include_self=True,
        )
        with paddle.base.device_guard("cpu"):
            self.output_tokens = paddle.empty(shape=output_tokens.shape, dtype=output_tokens.dtype)

        return output_tokens.to(self.input_dtype)

    def backward(self, out_grad, out_grad_scale):
        hidden_states_grad = paddle.gather(out_grad, self.token_permuted_indices)

        output_tokens_grad = FP8LinearFunctionBase.dequantize_fp8_to_fp32(out_grad, out_grad_scale)
        permuted_tokens = self.hidden_states * self.permuted_probs.unsqueeze(-1)
        permuted_tokens = permuted_tokens.cast(self.hidden_states.dtype)

        _, permuted_tokens_grad = paddle._C_ops.put_along_axis_grad(
            self.output_tokens,
            self.token_permuted_indices.cast("int32").unsqueeze(1).expand([-1, self.hidden]),
            permuted_tokens,
            self.output_tokens,
            output_tokens_grad,
            0,
            "add",
            True,
        )

        permuted_probs_grad = (permuted_tokens_grad * self.hidden_states).sum(axis=-1)

        faltten_dispatched_probs_grad = paddle._C_ops.gather_grad(
            self.faltten_dispatched_probs, self.prob_permuted_indices, permuted_probs_grad, 0
        )

        # dispatched_probs_grad = paddle._C_ops.flatten_grad(self.dispatched_probs, faltten_dispatched_probs_grad)
        dispatched_probs_grad = faltten_dispatched_probs_grad.reshape(self.dispatched_probs_shape)

        self.reset_status()
        return hidden_states_grad, dispatched_probs_grad


def tokens_zip_unique_add_with_subbatch(zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows=None):
    """
    tokens_zip_unique_add_with_subbatch
    """
    if subbatch_rows is None or subbatch_rows <= 0 or zipped_rows <= 0:
        return paddlefleet.extensions.ops.tokens_zip_unique_add(zipped, unzipped, index_unzipped, zipped_rows)
    else:
        if isinstance(zipped, paddle.Tensor):
            num_split = (zipped_rows + subbatch_rows - 1) // subbatch_rows
            remainder = zipped_rows % subbatch_rows
            if remainder == 0:
                rows = [subbatch_rows] * num_split
            else:
                rows = [subbatch_rows] * (num_split - 1) + [remainder]

            if zipped.shape[0] == 0:
                dtype = zipped.dtype
                hidden_size = zipped.shape[1]
                zipped = [paddle.zeros([r, hidden_size], dtype=dtype) for r in rows]
            else:
                zipped = paddle.split(zipped, rows, axis=0)
        return paddlefleet.extensions.ops.tokens_zip_unique_add_subbatch(
            zipped, unzipped, index_unzipped, zipped_rows, subbatch_rows
        )


def merge_subbatch_cast(x, dtype):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
            return x.cast(dtype) if x.dtype != dtype else x
        else:
            return paddlefleet.extensions.ops.merge_subbatch_cast(x, dtype)
    else:
        return x.cast(dtype) if x.dtype != dtype else x


def get_env_device():
    """
    Return the device name of running environment.
    """
    if paddle.is_compiled_with_cuda():
        return "gpu"
    elif "npu" in paddle.device.get_all_custom_device_type():
        return "npu"
    elif "mlu" in paddle.device.get_all_custom_device_type():
        return "mlu"
    elif "gcu" in paddle.device.get_all_custom_device_type():
        return "gcu"
    elif "intel_hpu" in paddle.device.get_all_custom_device_type():
        return "intel_hpu"
    elif paddle.is_compiled_with_rocm():
        return "rocm"
    elif paddle.is_compiled_with_xpu():
        return "xpu"
    return "cpu"
