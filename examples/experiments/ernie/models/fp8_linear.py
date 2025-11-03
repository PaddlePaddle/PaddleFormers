# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

"""
FP8 Linear Layer Implementation for PaddlePaddle

This module implements FP8 (8-bit floating point) linear layers using PaddlePaddle's
incubate APIs for low-precision training. Key features include:

1. FP8 matrix multiplication with block-wise quantization
2. Memory-efficient forward/backward passes
3. PaddlePaddle-specific optimizations like:
   - Using paddle.incubate.fp8 APIs
   - Leveraging Paddle's automatic differentiation system
   - Optimized for Paddle's tensor layout and memory management
"""


import numpy
import paddle
from paddle.incubate.fp8 import deep_gemm
from paddle.incubate.nn.functional import swiglu

# Keep reference to original linear op for fallback if needed
original_linear = paddle.nn.functional.linear


# Expose only the main class to public API
__all__ = ["Fp8FusedMlp"]


def fp8_gemm(
    x_fp8,
    x_scale,
    w_fp8,
    w_scale,
    is_a_1d_scaled,
    is_b_1d_scaled,
    out=None,
    rtn_dtype=paddle.bfloat16,
):
    """
    Performs FP8 matrix multiplication (GEMM) operation, using blockwise GEMM algorithm.

    Args:
        x_fp8 (Tensor): Input tensor in FP8 format
        x_scale (Tensor): Scaling factor for input tensor
        w_fp8 (Tensor): Weight tensor in FP8 format
        w_scale (Tensor): Scaling factor for weight tensor
        is_a_1d_scaled (bool): Whether input tensor uses 1D scaling
        is_b_1d_scaled (bool): Whether weight tensor uses 1D scaling
        out (Tensor, optional): Output tensor for accumulation. Defaults to None
        rtn_dtype (dtype, optional): Return data type. Defaults to paddle.bfloat16

    Returns:
        Tensor: Result of the matrix multiplication
    """
    accumulate = out is not None
    if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
        # Using Paddle's blockwise FP8 GEMM with split accumulator for numerical stability
        y = paddle.incubate.nn.functional.fp8_gemm_blockwise(
            a=x_fp8,
            a_decode_scale=x_scale,  # Input scaling factors
            b=w_fp8,
            b_decode_scale=w_scale,  # Weight scaling factors
            out_dtype=rtn_dtype,  # Output dtype (bfloat16)
            out=out,  # Optional output tensor for accumulation
            accumulate=accumulate,  # Whether to accumulate into out tensor
            use_split_accumulator=True,  # Paddle-specific optimization
            is_a_1d_scaled=is_a_1d_scaled,  # 1D scaling for input
            is_b_1d_scaled=is_b_1d_scaled,  # 1D scaling for weights
        )
    else:
        y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], rtn_dtype)
        if out is not None:
            out = out + y
            return out

    return y


def padding(x, axis):
    """
    Pads the input tensor along specified axis to make its size divisible by 512 or 128.

    Args:
        x (Tensor): Input tensor to be padded
        axis (int): Axis along which to pad (0 for rows, 1 for columns)

    Returns:
        Tensor: Padded tensor
    """
    if x.shape[axis] % 512 != 0:
        if (x.shape[axis] + 128 - (x.shape[axis] % 128)) % 512 != 0:
            padding_size = 512
        else:
            padding_size = 128
        pad_size = padding_size - (x.shape[axis] % padding_size)
        if axis == 0:
            x = paddle.concat([x, paddle.zeros([pad_size, x.shape[-1]], dtype=x.dtype)], axis=0)
        else:
            x = paddle.concat([x, paddle.zeros([x.shape[0], pad_size], dtype=x.dtype)], axis=-1)
    return x


class Fp8FusedMlpFunc(paddle.autograd.PyLayer):
    """
    Custom PyLayer implementation of FP8 fused MLP operation.

    This class implements both forward and backward passes for a memory-efficient
    FP8 (8-bit floating point) multi-layer perceptron using PaddlePaddle's
    FP8 quantization APIs.
    """

    @staticmethod
    def forward(ctx, x, w1, w2):
        """
        Forward pass for FP8 fused multi-layer perceptron (MLP) operation.

        Args:
            ctx (PyLayerContext): Context object to save tensors for backward pass
            x (paddle.Tensor): Input tensor of shape [batch_size, hidden_size]
            w1 (paddle.Tensor): First weight matrix of shape [hidden_size, intermediate_size*2]
            w2 (paddle.Tensor): Second weight matrix of shape [intermediate_size, hidden_size]

        Returns:
            paddle.Tensor: Output tensor of shape [batch_size, hidden_size]

        Note:
            - Uses Paddle's FP8 quantization for memory efficiency
            - Implements SWiGLU activation internally
            - Handles tensor padding for optimal FP8 GEMM performance
        """
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        if x.shape[0] % 512 != 0:
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            x = padding(x, 0)
            _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )

        else:
            x_fp8, x_scale, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )

        _, _, w1_fp8, w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w1,
            quant_method="128x128",
            input_transpose=True,
            output_scale_transpose=False,
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_scale), o1)

        o2 = swiglu(o1)
        o2_fp8, o2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2, quant_method="1x128", input_transpose=False, output_scale_transpose=True
        )

        _, _, w2_t_fp8, w2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w2,
            quant_method="128x128",
            input_transpose=True,
            output_scale_transpose=False,
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale.T), (w2_t_fp8, w2_t_scale), o3)
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        ctx.save_for_backward(
            x_t_fp8,
            x_t_scale,
            w1,
            o1,
            w2,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        """
        Memory-efficient backward pass for FP8 fused MLP operation.

        Args:
            ctx: Context object containing saved tensors from forward pass
            do3 (Tensor): Gradient of the loss with respect to the output

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Gradients with respect to x, w1, and w2
        """
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        x_t_fp8, x_t_scale, w1, o1, w2, x_orig_shape = ctx.saved_tensor()
        x_orig_shape = x_orig_shape.numpy()

        o2 = swiglu(o1)
        if do3.shape[0] % 512 != 0:
            do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do3 = padding(do3, 0)
            _, _, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        else:
            do3_fp8, do3_scale, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        w2_fp8, w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w2,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale.T), (w2_fp8, w2_scale), do2)

        o2 = padding(o2, 0)
        _, _, o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2,
            quant_method="1x128",
            input_transpose=True,
            output_scale_transpose=True,
        )

        dw2 = fp8_gemm(
            o2_t_fp8,
            o2_t_scale,
            do3_t_fp8,
            do3_t_scale,
            True,
            True,
            rtn_dtype=paddle.float32,
        )

        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        if do1.shape[0] % 512 != 0:
            do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do1 = padding(do1, 0)
            _, _, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        else:
            do1_fp8, do1_scale, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        w1_fp8, w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w1,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale.T), (w1_fp8, w1_scale), dx)
        if len(x_orig_shape) > 2:
            dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

        dw1 = fp8_gemm(
            x_t_fp8,
            x_t_scale,
            do1_t_fp8,
            do1_t_scale,
            True,
            True,
            rtn_dtype=paddle.float32,
        )
        return dx, dw1, dw2


class MemEfficientFp8FusedMlpFunc(paddle.autograd.PyLayer):
    """
    Memory-optimized version of FP8 fused MLP operation.

    This implementation reduces memory usage during training by:
    - Avoiding redundant tensor storage in forward pass
    - Recomputing intermediate values during backward pass
    - Using optimized FP8 quantization strategies

    Inherits from paddle.autograd.PyLayer to implement custom backward pass.
    """

    @staticmethod
    def forward(ctx, x, w1, w2):
        """
        Memory-efficient forward pass for FP8 fused MLP operation.

        Args:
            ctx (PyLayerContext): Context object to save minimal tensors for backward pass
            x (paddle.Tensor): Input tensor of shape [batch_size, hidden_size]
            w1 (paddle.Tensor): First weight matrix of shape [hidden_size, intermediate_size*2]
            w2 (paddle.Tensor): Second weight matrix of shape [intermediate_size, hidden_size]

        Returns:
            paddle.Tensor: Output tensor of shape [batch_size, hidden_size]

        Note:
            - Saves only essential tensors for backward pass to reduce memory usage
            - Uses recomputation strategy during backward pass
            - Maintains same numerical accuracy as standard implementation
        """
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        x_fp8, x_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            x, quant_method="1x128", input_transpose=False, output_scale_transpose=True
        )

        _, _, w1_fp8, w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w1,
            quant_method="128x128",
            input_transpose=True,
            output_scale_transpose=False,
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=x.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_scale), o1)

        o2 = swiglu(o1)
        o2_fp8, o2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2, quant_method="1x128", input_transpose=False, output_scale_transpose=True
        )

        _, _, w2_t_fp8, w2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w2,
            quant_method="128x128",
            input_transpose=True,
            output_scale_transpose=False,
        )
        o3 = paddle.empty([o2_fp8.shape[0], w2_t_fp8.shape[0]], dtype=o1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((o2_fp8, o2_scale.T), (w2_t_fp8, w2_t_scale), o3)
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        ctx.save_for_backward(
            x_fp8,
            x_scale,
            w1,
            w2,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        x_fp8, x_scale, w1, w2, x_orig_shape = ctx.saved_tensor()
        x_orig_shape = x_orig_shape.numpy()

        _, _, w1_fp8, w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w1,
            quant_method="128x128",
            input_transpose=True,
            output_scale_transpose=False,
        )
        o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_scale), o1)

        x_dequant_fp16 = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale.T.contiguous())
        x_dequant_fp16 = padding(x_dequant_fp16, 0)

        _, _, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            x_dequant_fp16,
            quant_method="1x128",
            input_transpose=True,
            output_scale_transpose=True,
        )

        o2 = swiglu(o1)

        if do3.shape[0] % 512 != 0:
            do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do3 = padding(do3, 0)
            _, _, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        else:
            do3_fp8, do3_scale, do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        w2_fp8, w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w2,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        do2 = paddle.empty([do3_fp8.shape[0], w2_fp8.shape[0]], do3.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do3_fp8, do3_scale.T), (w2_fp8, w2_scale), do2)

        o2 = padding(o2, 0)
        _, _, o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2,
            quant_method="1x128",
            input_transpose=True,
            output_scale_transpose=True,
        )

        dw2 = fp8_gemm(
            o2_t_fp8,
            o2_t_scale,
            do3_t_fp8,
            do3_t_scale,
            True,
            True,
            rtn_dtype=paddle.float32,
        )

        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        if do1.shape[0] % 512 != 0:
            do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do1 = padding(do1, 0)
            _, _, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        else:
            do1_fp8, do1_scale, do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1,
                quant_method="1x128",
                input_transpose=True,
                output_scale_transpose=True,
            )
        w1_fp8, w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            w1,
            quant_method="128x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        dx = paddle.empty([do1_fp8.shape[0], w1_fp8.shape[0]], do1.dtype)
        deep_gemm.gemm_fp8_fp8_bf16_nt((do1_fp8, do1_scale.T), (w1_fp8, w1_scale), dx)
        if len(x_orig_shape) > 2:
            dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

        dw1 = fp8_gemm(
            x_t_fp8,
            x_t_scale,
            do1_t_fp8,
            do1_t_scale,
            True,
            True,
            rtn_dtype=paddle.float32,
        )
        return dx, dw1, dw2


class Fp8FusedMlp(paddle.nn.Layer):
    """
    PaddlePaddle Layer implementing FP8 fused multi-layer perceptron (MLP).

    This layer combines:
    - FP8 precision matrix operations for improved performance
    - Fused MLP architecture with SWiGLU activation
    - Memory-efficient training through custom PyLayer implementation

    """

    def __init__(self, config):
        """
        Initializes the FP8 Fused MLP layer.

        Args:
            config (object): Configuration object containing:
                - hidden_size (int): Dimension of the input/output features
                - intermediate_size (int): Dimension of the intermediate features

        Note:
            - Weights are initialized using Paddle's create_parameter
            - Uses bfloat16 precision for weight storage
            - No bias terms are used in this implementation
        """

        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = self.create_parameter(
            shape=[self.hidden_size, self.intermediate_size * 2],
            dtype="bfloat16",  # Using Paddle's bfloat16 dtype
            is_bias=False,  # Paddle-specific parameter attribute
        )
        self.w2 = self.create_parameter(
            shape=[self.intermediate_size, self.hidden_size],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        """
        Forward pass of the FP8 fused MLP layer.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor after MLP transformation
        """
        return Fp8FusedMlpFunc.apply(x, self.w1, self.w2)
