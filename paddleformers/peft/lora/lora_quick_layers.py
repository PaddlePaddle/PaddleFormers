# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.autograd import PyLayer
from paddle.distributed.communication.reduce import ReduceOp, _get_reduce_op
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.framework import core

__all__ = ["quick_lora"]


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm() or paddle.is_compiled_with_xpu():
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    return False


if is_fused_matmul_bias_supported():
    linear_func = paddle.incubate.nn.functional.fused_linear
else:
    linear_func = paddle.nn.functional.linear


def quick_lora(
    input: paddle.Tensor,
    lora_A: paddle.Tensor,
    lora_B: paddle.Tensor,
    weight: paddle.Tensor,
    bias: paddle.Tensor = None,
    scaling: float = 1.0,
    is_column: bool = False,
    is_row: bool = False,
    group=None,
    world_size: int = 1,
):
    r"""
    Definition of the quick_lora function for efficient low-rank adaptation (LORA) operations

    Parameters:
        input: The input data for the LORA operation
        lora_A: The LORA matrix A
        lora_B: The LORA matrix B
        weight: The weight matrix
        bias: The bias vector (optional, defaults to None)
        scaling: The scaling factor (optional, defaults to 1.0)
        is_column: Flag indicating whether to perform LORA operation by column (optional, defaults to False)
        is_row: Flag indicating whether to perform LORA operation by row (optional, defaults to False)
        group: Group information (optional, defaults to None)
        world_size: World size for distributed operations (optional, defaults to 1)

    Returns:
        The result of the LORA operation based on the specified parameters

    """
    assert weight.stop_gradient, "When using Quick LoRA, it is necessary that weight.stop_gradient is set to True."
    if bias is not None:
        assert bias.stop_gradient, "When using Quick LoRA, it is necessary that bias.stop_gradient is set to True."

    input_stop_gradient = input.stop_gradient
    if is_column:
        # If is_column is True, apply the LORA operation by column using the ColumnQuickLora class
        return ColumnQuickLora.apply(
            input, lora_A, lora_B, weight, bias, scaling, group, input_stop_gradient=input_stop_gradient
        )
    elif is_row:
        # If is_row is True, apply the LORA operation by row using the RowQuickLora class
        return RowQuickLora.apply(
            input, lora_A, lora_B, weight, bias, scaling, group, world_size, input_stop_gradient=input_stop_gradient
        )
    else:
        # If neither is_column nor is_row is True, apply the regular LORA operation using the QuickLora class
        return QuickLora.apply(input, lora_A, lora_B, weight, bias, scaling, input_stop_gradient=input_stop_gradient)


class QuickLora(PyLayer):
    @staticmethod
    def forward(
        ctx,
        input,
        lora_A,
        lora_B,
        weight,
        bias: paddle.Tensor = None,
        scaling: float = 1.0,
        input_stop_gradient: bool = False,
    ):
        merged_weight = paddle.addmm(weight, lora_A, lora_B, beta=1.0, alpha=scaling)
        ctx.input_stop_gradient = input_stop_gradient
        ctx.scaling = scaling
        ctx.save_for_backward(input, weight, lora_A, lora_B)
        result = linear_func(input, merged_weight, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, lora_A, lora_B = ctx.saved_tensor()
        grad_output = grad_output.flatten(0, 1)
        input_fused = input.flatten(0, 1)
        lora_B_input_grad = paddle.matmul(grad_output, lora_B, transpose_y=True)
        input_grad = None

        if not ctx.input_stop_gradient:
            input_grad = paddle.addmm(
                paddle.matmul(grad_output, weight, transpose_y=True),
                lora_B_input_grad,
                lora_A.T,
                beta=1.0,
                alpha=ctx.scaling,
            ).reshape(input.shape)

        lora_A_grad = paddle.matmul(input_fused, lora_B_input_grad, transpose_x=True) * ctx.scaling

        lora_B_grad = paddle.matmul(paddle.matmul(input_fused, lora_A), grad_output, transpose_x=True) * ctx.scaling

        return input_grad, lora_A_grad, lora_B_grad


class ColumnQuickLora(PyLayer):
    @staticmethod
    def forward(
        ctx, input, lora_A, lora_B, weight, bias=None, scaling=1.0, group=None, input_stop_gradient: bool = False
    ):
        merged_weight = paddle.addmm(weight, lora_A, lora_B, beta=1.0, alpha=scaling)
        ctx.group = group
        ctx.op_type = _get_reduce_op(ReduceOp.SUM, "_c_identity")
        ctx.input_stop_gradient = input_stop_gradient
        ctx.scaling = scaling
        ctx.save_for_backward(input, weight, lora_A, lora_B)
        result = linear_func(input, merged_weight, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, lora_A, lora_B = ctx.saved_tensor()
        grad_output = grad_output.flatten(0, 1)
        input_fused = input.flatten(0, 1)
        lora_B_input_grad = paddle.matmul(grad_output, lora_B, transpose_y=True)
        input_grad = None
        if not ctx.input_stop_gradient:
            input_grad = paddle.addmm(
                paddle.matmul(grad_output, weight, transpose_y=True),
                lora_B_input_grad,
                lora_A.T,
                beta=1.0,
                alpha=ctx.scaling,
            ).reshape(input.shape)

        if ctx.group is not None:
            ctx.group.process_group.all_reduce_on_calc_stream(lora_B_input_grad, ctx.op_type)
        lora_A_grad = paddle.matmul(input_fused, lora_B_input_grad, transpose_x=True) * ctx.scaling

        lora_B_grad = paddle.matmul(paddle.matmul(input_fused, lora_A), grad_output, transpose_x=True) * ctx.scaling

        return input_grad, lora_A_grad, lora_B_grad


class RowQuickLora(PyLayer):
    @staticmethod
    def forward(
        ctx,
        input,
        lora_A,
        lora_B,
        weight,
        bias=None,
        scaling: float = 1.0,
        group=None,
        world_size: int = 1,
        input_stop_gradient: bool = False,
    ):
        if world_size > 1 and bias is not None:
            bias = paddle.scale(bias, 1.0 / world_size)
        merged_weight = paddle.addmm(weight, lora_A, lora_B, beta=1.0, alpha=scaling)
        ctx.input_stop_gradient = input_stop_gradient
        ctx.group = group
        ctx.scaling = scaling
        ctx.save_for_backward(input, weight, lora_A, lora_B)
        result = linear_func(input, merged_weight, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, lora_A, lora_B = ctx.saved_tensor()

        grad_output = grad_output.flatten(0, 1)
        input_fused = input.flatten(0, 1)

        lora_B_input_grad = paddle.matmul(grad_output, lora_B, transpose_y=True)

        input_grad = None
        if not ctx.input_stop_gradient:
            input_grad = paddle.addmm(
                paddle.matmul(grad_output, weight, transpose_y=True),
                lora_B_input_grad,
                lora_A.T,
                beta=1.0,
                alpha=ctx.scaling,
            ).reshape(input.shape)

        lora_A_grad = paddle.matmul(input_fused, lora_B_input_grad, transpose_x=True) * ctx.scaling

        x_lora_A = paddle.matmul(input_fused, lora_A)
        if ctx.group is not None:
            x_lora_A = mp_ops._mp_allreduce(
                x_lora_A,
                group=ctx.group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        lora_B_grad = paddle.matmul(x_lora_A, grad_output, transpose_x=True) * ctx.scaling
        return input_grad, lora_A_grad, lora_B_grad
