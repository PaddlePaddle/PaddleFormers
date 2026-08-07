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
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# pylint: disable=missing-function-docstring, missing-class-docstring

import logging

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.jit import jit_fuser
from paddleformers.fleet.utils import nvtx_decorator

logger = logging.getLogger(__name__)

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (paddle.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        paddle.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    return F.swiglu(y)


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (paddle.Tensor): Input tensor.
        bias (paddle.Tensor): Bias tensor to be added to input.

    Returns:
        paddle.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + paddle.erf(x * 0.70710678)) + 0.3989423 * x * paddle.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (paddle.Tensor): Gradient tensor from the subsequent layer.
        y (paddle.Tensor): Input tensor that was used in the forward pass.

    Returns:
        paddle.Tensor: Gradient with respect to the input tensor, computed using the
            Paddle native SwiGLU gradient operator.
    """
    dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
    return dx


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (paddle.Tensor): Gradient tensor from the subsequent layer.
        y (paddle.Tensor): Input tensor that was used in the forward pass.
        bias (paddle.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        paddle.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precision of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = paddle.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


@jit_fuser
def clamped_swiglu(y, clamp_value):
    """SwiGLU with clamped inputs for numerical stability.

    Clamps y1 (gate) to (-inf, clamp_value] and y2 (value) to
    [-clamp_value, clamp_value] before computing SiLU(y1) * y2.
    Computation is performed in float32 and cast back to original dtype.

    Args:
        y (paddle.Tensor): Input tensor, split into two halves along last dim.
        clamp_value (float): Clamp bound.

    Returns:
        paddle.Tensor: SiLU(clamp(y1)) * clamp(y2), same dtype as input.
    """
    dtype = y.dtype
    y_1, y_2 = paddle.chunk(y.cast(paddle.float32), 2, axis=-1)
    y_1 = y_1.clip(max=clamp_value)
    y_2 = y_2.clip(min=-clamp_value, max=clamp_value)
    res = F.silu(y_1) * y_2
    return res.cast(dtype)


@jit_fuser
def clamped_swiglu_back(g, y, clamp_value):
    """Backward pass for clamped_swiglu.

    Gradient is zeroed out where inputs were clamped.
    Computation is performed in float32; g is used in its
    original dtype so auto-promotion to float32 occurs naturally through
    the float32 terms; masks are cast to g.dtype matching
    ``.to(g.dtype)`` in the reference implementation.

    Args:
        g (paddle.Tensor): Upstream gradient.
        y (paddle.Tensor): Original (un-clamped) input tensor from forward pass.
        clamp_value (float): Clamp bound used in forward pass.

    Returns:
        paddle.Tensor: Gradient w.r.t. y, same dtype as y.
    """
    dtype = y.dtype
    y_fp32 = y.cast(paddle.float32)
    y_1, y_2 = paddle.chunk(y_fp32, 2, axis=-1)
    y_1_clamped = y_1.clip(max=clamp_value)
    y_2_clamped = y_2.clip(min=-clamp_value, max=clamp_value)
    # Clamp masks in g.dtype
    y1_mask = (y_1 <= clamp_value).cast(g.dtype)
    y2_mask = ((y_2 >= -clamp_value) & (y_2 <= clamp_value)).cast(g.dtype)
    sig = F.sigmoid(y_1_clamped)
    grad_y1 = (
        g * sig * (1.0 + y_1_clamped * (1.0 - sig)) * y_2_clamped * y1_mask
    )
    grad_y2 = g * (y_1_clamped * sig) * y2_mask  # silu = x * sigmoid(x)
    return paddle.concat([grad_y1, grad_y2], axis=-1).cast(dtype)


@jit_fuser
def clamped_bias_swiglu(y, bias, clamp_value):
    """Clamped SwiGLU with bias addition.

    Adds bias to the input tensor, then applies clamped SwiGLU activation.
    Gate (y1) is clamped to (-inf, clamp_value] and value (y2) to
    [-clamp_value, clamp_value] before computing SiLU(y1) * y2.

    Args:
        y (paddle.Tensor): Input tensor, split into gate/value halves along last dim.
        bias (paddle.Tensor): Bias tensor added to input before activation.
        clamp_value (float): Clamp bound for numerical stability.

    Returns:
        paddle.Tensor: clamped_swiglu(y + bias), same dtype as y.
    """
    y = y + bias
    return clamped_swiglu(y, clamp_value)


@jit_fuser
def clamped_bias_swiglu_back(g, y, bias, clamp_value):
    """Backward pass for clamped_bias_swiglu.

    Adds bias to the input tensor and delegates to clamped_swiglu_back
    which zeros out gradients where inputs were clamped in the forward pass.

    Args:
        g (paddle.Tensor): Upstream gradient.
        y (paddle.Tensor): Original (un-clamped) input tensor from forward pass.
        bias (paddle.Tensor): Bias tensor that was added in the forward pass.
        clamp_value (float): Clamp bound used in forward pass.

    Returns:
        paddle.Tensor: Gradient w.r.t. (y + bias), same dtype as y.
    """
    y = y + bias
    return clamped_swiglu_back(g, y, clamp_value)


@jit_fuser
def clamped_weighted_swiglu(y, weights, clamp_value):
    """ClampedSwiGLU with per-token weight scaling.

    Args:
        y (paddle.Tensor): Input tensor.
        weights (paddle.Tensor): Per-token weights, shape [..., 1].
        clamp_value (float): Clamp bound.

    Returns:
        paddle.Tensor: clamped_swiglu(y) * weights, same dtype as y.
    """
    dtype = y.dtype
    res = clamped_swiglu(y, clamp_value) * weights
    return res.cast(dtype)


@jit_fuser
def clamped_weighted_swiglu_back(g, y, weights, clamp_value):
    """Backward pass for clamped_weighted_swiglu.

    Delegates to clamped_swiglu_back for input grad and recomputes
    clamped_swiglu(y) for weights grad. This re-executes one SwiGLU forward
    (clip + sigmoid + silu) in the backward — a conscious trade-off:
    correctness and code clarity take priority over the marginal FLOPs,
    matching the reference implementation structure. The overhead is ~1
    SwiGLU forward which is negligible compared to the main matmul.

    Precision note: ``paddle.sum`` internally accumulates bf16 inputs in
    float32, matching the CUDA kernel's ``float`` accumulation for
    ``local_d_probs_sum`` in the same-type branch.

    Args:
        g (paddle.Tensor): Upstream gradient.
        y (paddle.Tensor): Original input tensor from forward pass.
        weights (paddle.Tensor): Per-token weights, broadcastable to g's shape.
        clamp_value (float): Clamp bound used in forward pass.

    Returns:
        tuple: (grad_y, grad_weights), matching dtypes of inputs.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = clamped_swiglu_back(g * weights, y, clamp_value)
    weights_grad = clamped_swiglu(y, clamp_value) * g.cast(w_dtype)
    weights_grad = paddle.sum(weights_grad, axis=-1, keepdim=True)
    return input_grad.cast(input_dtype), weights_grad.cast(w_dtype)


class BiasSwiGLUFunction(paddle.autograd.PyLayer):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx, input, bias, fp8_input_store, cpu_offload_input, clamp_value=None
    ):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (paddle.Tensor): Input tensor to apply SwiGLU to.
            bias (paddle.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.
            cpu_offload_input (bool): If True, enables CPU activation offloading.
            clamp_value (float, optional): If provided and > 0, clamps gate to
                (-inf, clamp_value] and value to [-clamp_value, clamp_value] for
                numerical stability.

        Returns:
            paddle.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = (
            input.to(paddle.float8_e4m3fn) if fp8_input_store else input
        )
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        if clamp_value is not None and clamp_value > 0:
            return clamped_bias_swiglu(input, bias, clamp_value)
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (paddle.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
        """
        input, bias = ctx.saved_tensor()
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp = clamped_bias_swiglu_back(
                grad_output, input, bias, ctx.clamp_value
            )
        else:
            tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp


class SwiGLUFunction(paddle.autograd.PyLayer):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx, input, fp8_input_store, cpu_offload_input, clamp_value=None
    ):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (paddle.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.
            cpu_offload_input (bool): If True, enables CPU activation offloading.
            clamp_value (float, optional): If provided and > 0, clamps gate to
                (-inf, clamp_value] and value to [-clamp_value, clamp_value] for
                numerical stability.

        Returns:
            paddle.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = (
            input.to(paddle.float8_e4m3fn) if fp8_input_store else input
        )
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        if clamp_value is not None and clamp_value > 0:
            return clamped_swiglu(input, clamp_value)
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (paddle.Tensor): Gradient of the loss with respect to the output.

        Returns:
            paddle.Tensor: Gradient with respect to the input tensor.
        """
        input = ctx.saved_tensor()[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp = clamped_swiglu_back(grad_output, input, ctx.clamp_value)
        else:
            tmp = swiglu_back(grad_output, input)
        return tmp


class WeightedSwiGLUFunction(paddle.autograd.PyLayer):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store, clamp_value=None):
        input_for_backward = (
            input.to(paddle.float8_e4m3fn) if fp8_input_store else input
        )
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = clamp_value
        if clamp_value is not None and clamp_value > 0:
            res = clamped_weighted_swiglu(input, weights, clamp_value)
        else:
            res = weighted_swiglu(input, weights)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensor()
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.clamp_value is not None and ctx.clamp_value > 0:
            tmp, wgrad = clamped_weighted_swiglu_back(
                grad_output, input, weights, ctx.clamp_value
            )
        else:
            tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (paddle.Tensor): Input tensor to apply SwiGLU activation.
        bias (paddle.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.
        cpu_offload_input (bool, optional): If True, enables CPU activation offloading.
            Defaults to False.
        clamp_value (float, optional): If provided and > 0, clamps gate to
            (-inf, clamp_value] and value to [-clamp_value, clamp_value] for
            numerical stability.

    Returns:
        paddle.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(
            input, bias, fp8_input_store, cpu_offload_input, clamp_value
        )
    else:
        output = SwiGLUFunction.apply(
            input, fp8_input_store, cpu_offload_input, clamp_value
        )

    return (
        output
        if len(ori_shape) == 2
        else output.view(ori_shape[0], ori_shape[1], -1)
    )


def weighted_bias_swiglu_impl(
    input, bias, weights, fp8_input_store=False, clamp_value=None
):
    """
    Token-wise-weighted bias swiglu fusion.

    Args:
        input: Input tensor.
        bias: Optional bias (not supported for weighted variant).
        weights: Per-token weights, shape [..., 1].
        fp8_input_store (bool): Whether to store intermediate values in FP8 format.
        clamp_value (float, optional): If provided and > 0, clamps gate to
            (-inf, clamp_value] and value to [-clamp_value, clamp_value] for
            numerical stability.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if len(ori_shape) == 3:
        weights = weights.view(-1, weights.shape[-1])
    if bias is not None:
        raise NotImplementedError(
            "Bias is not supported for weighted swiglu fusion"
        )
    else:
        output = WeightedSwiGLUFunction.apply(
            input, weights, fp8_input_store, clamp_value
        )

    return (
        output
        if len(ori_shape) == 2
        else output.view(ori_shape[0], ori_shape[1], -1)
    )


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
