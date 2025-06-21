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


import paddle
from paddle.autograd import PyLayer

from .hadamard_utils import apply_hadamard_matmul

try:
    from transformer_engine import transformer_engine_paddle as tex
    from transformer_engine.paddle.constants import FP8BwdTensors, FP8FwdTensors
    from transformer_engine.paddle.cpp_extensions import fp8_gemm
    from transformer_engine.paddle.layer.base import get_workspace

    TE_DType = {
        paddle.float8_e4m3fn: tex.DType.kFloat8E4M3,
        paddle.float8_e5m2: tex.DType.kFloat8E5M2,
    }
    SUPPORT_TE = True
except ImportError:
    SUPPORT_TE = False

from paddle.linalg import fp8_fp8_half_gemm_fused

QMIN_QMAX_MAPPING = {
    "a8w8linear_activation": (-128, 127),
    "a8w4linear_activation": (-128, 127),
    "a8w8linear_weight": (-128, 127),
    "a8w4linear_weight": (-8, 7),
    "float8_e4m3fn": (-488, 488),
    "float8_e5m2": (-57344, 57344),
}


def quantize(
    x,
    weight_quantize_algo,
    tensor_type,
    quantization_config,
    side="right",
    apply_hadamard=False,
    activation_scale=None,
    state=0,
    training=False,
    group=None,
):
    if apply_hadamard:
        target_x = apply_hadamard_matmul(x, side, quantization_config.hadamard_block_size)
        hadamard_scale = quantization_config.hadamard_block_size
    else:
        target_x, hadamard_scale = x, 1.0
    if weight_quantize_algo in ["fp8linear"]:
        qmin, qmax = QMIN_QMAX_MAPPING[quantization_config.fp8_format[tensor_type]]
    else:
        qmin, qmax = QMIN_QMAX_MAPPING[weight_quantize_algo + "_" + tensor_type]
    if tensor_type == "activation":
        if activation_scale is not None:
            if training:
                scale = (paddle.max(paddle.abs(target_x)) / qmax + quantization_config.scale_epsilon).reshape([1])
                if group is not None:
                    paddle.distributed.all_reduce(scale, op=paddle.distributed.ReduceOp.MAX, group=group, sync_op=True)
                if state < quantization_config.apply_online_actscale_step:
                    activation_scale[:] = (state * activation_scale + scale) / (state + 1)
                else:
                    scale = (
                        1 - quantization_config.actscale_moving_rate
                    ) * activation_scale + quantization_config.actscale_moving_rate * scale
                    activation_scale[:] = scale
            else:
                scale = activation_scale
        else:
            scale = (paddle.max(paddle.abs(target_x)) / qmax + quantization_config.scale_epsilon).reshape([1])
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            quant_x = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
        elif weight_quantize_algo in ["fp8linear"]:
            quant_x = (target_x / scale).astype(quantization_config.fp8_format[tensor_type])
        else:
            raise NotImplementedError(f"Unknown {weight_quantize_algo}.")
    elif tensor_type == "weight":
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            # channelwise
            scale = paddle.max(paddle.abs(target_x), axis=0, keepdim=True) / qmax + quantization_config.scale_epsilon
            if group is not None:
                paddle.distributed.all_reduce(scale, op=paddle.distributed.ReduceOp.MAX, group=group, sync_op=True)
            quant_x = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
            scale = scale.squeeze(0) / hadamard_scale
        elif weight_quantize_algo in ["fp8linear"]:
            scale = paddle.max(paddle.abs(target_x)) / qmax + quantization_config.scale_epsilon
            if group is not None:
                paddle.distributed.all_reduce(scale, op=paddle.distributed.ReduceOp.MAX, group=group, sync_op=True)
            quant_x = (target_x / scale).astype(quantization_config.fp8_format[tensor_type]).view("int8")
            scale = (scale / hadamard_scale).reshape([1])
        else:
            raise NotImplementedError(f"Unknown {weight_quantize_algo}.")
    elif tensor_type == "grad_output":
        if weight_quantize_algo in ["fp8linear"]:
            scale = (paddle.max(paddle.abs(target_x)) / qmax + quantization_config.scale_epsilon).reshape([1])
            quant_x = (target_x / scale).astype(quantization_config.fp8_format[tensor_type])
            scale = scale / hadamard_scale
        else:
            raise NotImplementedError(f"Unknown {weight_quantize_algo}.")
    else:
        raise NotImplementedError(f"Unknown {tensor_type}.")
    scale.stop_gradient = True
    return quant_x, scale


def dequantize(
    quant_x, scale, tensor_type, weight_quantize_algo, quantization_config, apply_hadamard=False, side="left"
):
    if tensor_type == "weight":
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            x = quant_x.astype(scale.dtype)
        elif weight_quantize_algo in ["fp8linear"]:
            x = quant_x.view(quantization_config.fp8_format[tensor_type]).astype(scale.dtype)
        else:
            raise NotImplementedError(f"Unknown weight_quantize_algo: {weight_quantize_algo}")
        if apply_hadamard:
            x = apply_hadamard_matmul(x, side, quantization_config.hadamard_block_size)
        x *= scale
    else:
        raise NotImplementedError(f"Unknown {tensor_type}.")
    return x


def int8_forward(
    x,
    quant_w,
    scale_w,
    weight_quantize_algo,
    bias=None,
    quantization_config=None,
    state=0,
    training=False,
    activation_scale=None,
    group=None,
):
    quant_x, scale_x = quantize(
        x=x,
        weight_quantize_algo=weight_quantize_algo,
        tensor_type="activation",
        quantization_config=quantization_config,
        side="right",
        apply_hadamard=quantization_config.apply_hadamard,
        activation_scale=activation_scale,
        state=state,
        training=training,
        group=group,
    )

    out = paddle.matmul(quant_x, quant_w).astype(scale_w.dtype) * (scale_x * scale_w)
    if bias is not None:
        out += bias
    return out, quant_x, scale_x


def int8_backward(ctx, x, grad_output, quant_weight, weight_scale, quant_x, x_scale):
    if not ctx.x_stop_gradient:
        qdq_weight = dequantize(
            quant_weight,
            weight_scale,
            "weight",
            ctx.weight_quantize_algo,
            ctx.quantization_config,
            ctx.quantization_config.apply_hadamard,
            "left",
        )
        input_grad = paddle.matmul(grad_output, qdq_weight.T)
    else:
        input_grad = None

    if not ctx.w_stop_gradient:
        if len(x.shape) == 2:
            weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
        else:
            weight_grad = paddle.matmul(
                x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
            )
    else:
        weight_grad = None

    return input_grad, weight_grad


def fp8_forward(
    x,
    w_fp8,
    w_scale,
    weight_quantize_algo,
    bias=None,
    dtype=None,
    quantization_config=None,
    state=0,
    training=False,
    activation_scale=None,
    group=None,
):
    x_fp8, x_scale = quantize(
        x,
        weight_quantize_algo,
        "activation",
        quantization_config,
        side="right",
        apply_hadamard=quantization_config.apply_hadamard,
        activation_scale=activation_scale,
        state=state,
        training=training,
        group=group,
    )
    w_fp8 = w_fp8.view(quantization_config.fp8_format["weight"]).T
    origin_shape = x_fp8.shape[:-1]
    out = fp8_fp8_half_gemm_fused(
        x_fp8.reshape([-1, x_fp8.shape[-1]]),
        w_fp8,
        transpose_x=False,
        transpose_y=True,
        bias=bias,
        scale=x_scale * w_scale,
        output_dtype=dtype,
    )
    return out.reshape(origin_shape + out.shape[-1:]), x_fp8, x_scale


def fp8_backward(ctx, x, grad_output, quant_weight, weight_scale, quant_x, x_scale):
    if not ctx.x_stop_gradient:
        if ctx.quantization_config.quant_input_grad:
            grad_output_fp8, grad_output_scale = quantize(
                grad_output,
                ctx.weight_quantize_algo,
                "grad_output",
                ctx.quantization_config,
                side="left",
                apply_hadamard=False,
            )
            quant_weight = quant_weight.view(ctx.quantization_config.fp8_format["weight"])
            if SUPPORT_TE:
                grad_output_shape = grad_output_fp8.shape
                grad_output_fp8 = grad_output_fp8.view((-1, grad_output_fp8.shape[-1]))
                fwd_scales = paddle.concat([x_scale.astype("float32"), weight_scale.astype("float32")])
                bwd_scales = grad_output_scale[None].astype("float32")
                input_grad, _ = fp8_gemm(
                    A=quant_weight,
                    A_scale_inv=fwd_scales,
                    A_fp8_tensor=FP8FwdTensors.GEMM1_WEIGHT,
                    A_dtype=TE_DType[quant_weight.dtype],
                    B=grad_output_fp8,
                    B_scale_inv=bwd_scales,
                    B_fp8_tensor=FP8BwdTensors.GRAD_OUTPUT1,
                    B_dtype=TE_DType[grad_output_fp8.dtype],
                    out_dtype=ctx.dtype,
                    workspace=get_workspace(),
                    use_split_accumulator=True,
                )
                input_grad = input_grad.view((*grad_output_shape[:-1], -1))
            else:
                grad_output_ = grad_output_fp8.astype(ctx.dtype) * grad_output_scale
                weight_ = quant_weight.astype(ctx.dtype) * weight_scale
                input_grad = paddle.matmul(grad_output_, weight_).astype(ctx.dtype)
            if ctx.quantization_config.apply_hadamard:
                input_grad = apply_hadamard_matmul(input_grad, "right", ctx.quantization_config.hadamard_block_size)
        else:
            qdq_weight = dequantize(
                quant_weight,
                weight_scale,
                "weight",
                ctx.weight_quantize_algo,
                ctx.quantization_config,
                apply_hadamard=ctx.quantization_config.apply_hadamard,
                side="left",
            )
            input_grad = paddle.matmul(grad_output, qdq_weight.T)
    else:
        input_grad = None

    if not ctx.w_stop_gradient:
        if ctx.quantization_config.quant_weight_grad:
            grad_output_fp8, grad_output_scale = quantize(
                x=grad_output,
                weight_quantize_algo=ctx.weight_quantize_algo,
                tensor_type="grad_output",
                quantization_config=ctx.quantization_config,
                apply_hadamard=False,
            )
            if SUPPORT_TE:
                quant_x = quant_x.view((-1, quant_x.shape[-1]))
                grad_output_fp8 = grad_output_fp8.view((-1, grad_output_fp8.shape[-1]))
                fwd_scales = paddle.concat([x_scale.astype("float32"), weight_scale.astype("float32")])
                bwd_scales = grad_output_scale[None].astype("float32")
                # FP8 gemm need k % 16 = 0
                ALIGNMENT_SIZE = 16

                def pad_tensor_to_multiple(tensor, dtype):
                    current_size = tensor.shape[0]
                    padding_size = ALIGNMENT_SIZE - current_size % ALIGNMENT_SIZE
                    # Create padding zeros with matching shape and dtype
                    padding_shape = [padding_size, tensor.shape[1]]
                    padding = paddle.zeros(padding_shape, dtype=dtype)
                    padded_tensor = paddle.concat([tensor, padding], axis=0)
                    return padded_tensor

                if quant_x.shape[0] % ALIGNMENT_SIZE != 0:
                    quant_x = pad_tensor_to_multiple(quant_x, ctx.quantization_config.fp8_format["activation"])
                    grad_output_fp8 = pad_tensor_to_multiple(
                        grad_output_fp8, ctx.quantization_config.fp8_format["grad_output"]
                    )

                weight_grad, _ = fp8_gemm(
                    A=grad_output_fp8.T,
                    A_scale_inv=bwd_scales,
                    A_fp8_tensor=FP8BwdTensors.GRAD_OUTPUT1,
                    A_dtype=TE_DType[grad_output_fp8.dtype],
                    B=quant_x.T,
                    B_scale_inv=fwd_scales,
                    B_fp8_tensor=FP8FwdTensors.GEMM1_INPUT,
                    B_dtype=TE_DType[quant_x.dtype],
                    out_dtype=ctx.dtype,
                    workspace=get_workspace(),
                    use_split_accumulator=True,
                )
            else:
                grad_output_ = grad_output_fp8.astype(ctx.dtype) * grad_output_scale
                x_ = quant_x.astype(ctx.dtype) * x_scale
                if len(x_.shape) == 2:
                    weight_grad = paddle.matmul(x_.transpose([1, 0]), grad_output_).astype(ctx.dtype)
                else:
                    weight_grad = paddle.matmul(
                        x_.reshape([-1, x_.shape[-1]]).transpose([1, 0]),
                        grad_output_.reshape([-1, grad_output_.shape[-1]]),
                    ).astype(ctx.dtype)
            if ctx.quantization_config.apply_hadamard:
                weight_grad = weight_grad / ctx.quantization_config.hadamard_block_size
                weight_grad = apply_hadamard_matmul(weight_grad, "left", ctx.quantization_config.hadamard_block_size)
        else:
            if len(x.shape) == 2:
                weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
            else:
                weight_grad = paddle.matmul(
                    x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
                )
    else:
        weight_grad = None

    return input_grad, weight_grad


class QATFunc(PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        quant_weight,
        bias,
        weight_scale,
        quantization_config,
        dtype,
        state,
        training,
        activation_scale,
        weight_quantize_algo,
        group,
    ):
        quant_x, x_scale = None, None
        if weight_quantize_algo in ["fp8linear"]:
            output, quant_x, x_scale = fp8_forward(
                x,
                quant_weight,
                w_scale=weight_scale,
                weight_quantize_algo=weight_quantize_algo,
                bias=bias,
                dtype=dtype,
                quantization_config=quantization_config,
                state=state,
                training=training,
                activation_scale=activation_scale,
                group=group,
            )
        else:
            output, quant_x, x_scale = int8_forward(
                x,
                quant_w=quant_weight,
                scale_w=weight_scale,
                weight_quantize_algo=weight_quantize_algo,
                bias=bias,
                quantization_config=quantization_config,
                state=state,
                training=training,
                activation_scale=activation_scale,
                group=group,
            )
        ctx.quantization_config = quantization_config
        ctx.weight_quantize_algo = weight_quantize_algo
        ctx.dtype = dtype
        ctx.x_stop_gradient = x.stop_gradient
        ctx.w_stop_gradient = quant_weight.stop_gradient
        ctx.b_stop_gradient = bias.stop_gradient if bias is not None else True
        ctx.save_for_backward(
            x if not quantization_config.quant_weight_grad else None,
            quant_weight,
            bias,
            weight_scale,
            quant_x if quantization_config.quant_weight_grad else None,
            x_scale if quantization_config.quant_weight_grad else None,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, quant_weight, bias, weight_scale, quant_x, x_scale = ctx.saved_tensor()

        if ctx.quantization_config.weight_quantize_algo in ["fp8linear"]:
            input_grad, weight_grad = fp8_backward(ctx, x, grad_output, quant_weight, weight_scale, quant_x, x_scale)
        else:
            input_grad, weight_grad = int8_backward(ctx, x, grad_output, quant_weight, weight_scale, quant_x, x_scale)

        if not ctx.b_stop_gradient:
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None

        return input_grad, weight_grad, bias_grad
