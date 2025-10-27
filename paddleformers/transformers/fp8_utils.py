# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial

import numpy
import paddle
import paddle.nn.functional as F

try:
    import fused_ln
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import WeightGradStore

USE_DS_GEMM = os.getenv("USE_DS_GEMM", "False").lower() == "true"

try:
    if USE_DS_GEMM:
        import deep_gemm
    else:
        from paddle.incubate.fp8 import deep_gemm
except:
    pass


__all__ = [
    "FP8LinearFunctionBase",
    "FP8Linear",
    "FP8GroupGemmMlpFunctionNode",
]


def get_sm_num():
    return 112


def set_parameter_color(
    parameters, color, group=None, offline_quant_expert_weight=True, clear_origin_weight_when_offline_quant=True
):
    if offline_quant_expert_weight and clear_origin_weight_when_offline_quant:
        if group is None:
            for p in parameters:
                if hasattr(p, "color") and p.color is not None:
                    continue
                setattr(p, "color", {"color": color})
        else:
            for p in parameters:
                if hasattr(p, "color") and p.color is not None:
                    continue
                setattr(p, "color", {"color": color, "group": group})


def extract_first_if_tuple(x):
    return x[0] if isinstance(x, tuple) else x


def _get_fp8_weight_and_scale(weight, stacked=False, transpose=False):
    """_get_fp8_weight_and_scale"""
    if stacked:
        if transpose:
            fp8_weight, fp8_scale = weight.fp8_weight_stacked_transpose, weight.fp8_scale_stacked_transpose
        else:
            fp8_weight, fp8_scale = weight.fp8_weight_stacked, weight.fp8_scale_stacked
    else:
        if transpose:
            fp8_weight, fp8_scale = weight.fp8_weight_transpose, weight.fp8_scale_transpose
        else:
            fp8_weight, fp8_scale = weight.fp8_weight, weight.fp8_scale
    return fp8_weight, fp8_scale


def fused_stack_quant(expert_weight_list, transpose=False):
    if transpose is False and hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], stacked=True, transpose=False)
    elif transpose is True and hasattr(expert_weight_list[0], "fp8_weight_stacked_transpose"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], stacked=True, transpose=True)
    elif transpose is True and hasattr(expert_weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], stacked=True, transpose=False)
    elif transpose is False and hasattr(expert_weight_list[0], "fp8_weight_stacked_transpose"):
        w, scale = _get_fp8_weight_and_scale(expert_weight_list[0], stacked=True, transpose=True)
    else:
        w, scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(expert_weight_list, transpose=transpose)
    return w, scale


def weight_quant(weight, transpose=False):
    if transpose:
        if hasattr(weight, "fp8_weight_transpose"):
            return weight.fp8_weight_transpose, weight.fp8_scale_transpose
        elif hasattr(weight, "fp8_weight"):
            return weight.fp8_weight.T.contiguous(), weight.fp8_scale.T.contiguous()
        else:
            return paddle.incubate.nn.functional.fp8_quant_blockwise(
                weight,
                output_scale_transpose=False,
                quant_method="128x128",
                input_transpose=True,
                return_transpose_only=True,
            )
    else:
        if hasattr(weight, "fp8_weight"):
            return weight.fp8_weight, weight.fp8_scale
        elif hasattr(weight, "fp8_weight_transpose"):
            return weight.fp8_weight_transpose.T.contiguous(), weight.fp8_scale_transpose.T.contiguous()
        else:
            return paddle.incubate.nn.functional.fp8_quant_blockwise(
                weight,
                output_scale_transpose=False,
                quant_method="128x128",
                input_transpose=False,
                return_transpose_only=False,
            )


class FP8LinearFunctionBase:
    @staticmethod
    def dequantize_fp8_to_fp32(fp8_tensor, scale):
        res = fp8_tensor.reshape([-1, 128]).astype("bfloat16") * (scale.reshape([-1, 1]))
        return res.reshape(fp8_tensor.shape)

    @staticmethod
    def padding(x, axis):
        if x.shape[axis] % 512 != 0:
            if (x.shape[axis] + 128 - (x.shape[axis] % 128)) % 512 != 0:
                padding_size = 512
            else:
                padding_size = 128
            pad_size = padding_size - (x.shape[axis] % padding_size)
            if axis == 0:
                x = paddle.cat([x, paddle.zeros([pad_size, x.shape[-1]], dtype=x.dtype)], axis=0)
            else:
                x = paddle.cat([x, paddle.zeros([x.shape[0], pad_size], dtype=x.dtype)], axis=-1)
        return x

    @staticmethod
    def padding_and_quant_input(tensor):
        """Quantize input to FP8, with fallback to padded transposed version if shape not aligned."""
        if tensor.shape[0] % 512 != 0:
            tensor_fp8, tensor_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                tensor, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            tensor = FP8LinearFunctionBase.padding(tensor, 0)
            tensor_t_fp8, tensor_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                tensor,
                output_scale_transpose=True,
                quant_method="1x128",
                input_transpose=True,
                return_transpose_only=True,
            )
            return tensor_fp8, tensor_scale, tensor_t_fp8, tensor_t_scale
        else:
            tensor_fp8, tensor_scale, tensor_t_fp8, tensor_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                tensor, output_scale_transpose=True, quant_method="1x128", input_transpose=True
            )
            return tensor_fp8, tensor_scale, tensor_t_fp8, tensor_t_scale

    @staticmethod
    def kitchen_gemm(
        x_fp8, x_scale, w_fp8, w_scale, is_a_1d_scaled, is_b_1d_scaled, out=None, rtn_dtype=paddle.bfloat16
    ):
        if USE_DS_GEMM:
            if out is None:
                out = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], rtn_dtype)
            if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
                deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt((x_fp8, x_scale), (w_fp8, w_scale), out, num_sms=get_sm_num())
            return out

        if out is not None:
            accumulate = True
            out_dtype = out.dtype
        else:
            accumulate = False
            out_dtype = rtn_dtype
        if numpy.prod(x_fp8.shape) != 0 and numpy.prod(w_fp8.shape) != 0:
            y = paddle.incubate.nn.functional.fp8_gemm_blockwise(
                a=x_fp8,
                a_decode_scale=x_scale,
                b=w_fp8,
                b_decode_scale=w_scale,
                out_dtype=out_dtype,
                out=out,
                accumulate=accumulate,
                use_split_accumulator=True,
                is_a_1d_scaled=is_a_1d_scaled,
                is_b_1d_scaled=is_b_1d_scaled,
            )
        else:
            y = paddle.zeros([x_fp8.shape[0], w_fp8.shape[0]], out_dtype)
            if out is not None:
                out = out + y
                return out

        return y

    @staticmethod
    def compute_fp8_linear(
        input, weight, weight_transpose=False, return_transpose_only=False, return_mode="output_only", *, out=None
    ):
        """
        FP8 Linear computation function supporting multiple return modes and quantized/unquantized inputs.

        Args:
            input: Input tensor (raw tensor or quantized as (input_fp8, input_scale) tuple)
            weight: Weight tensor
            weight_transpose (bool): Whether to transpose weight
            return_transpose_only (bool): Whether to return only transposed weight
            return_mode (str): Return mode options:
                - "output_only": Returns only output tensor
                - "with_input_quant": Returns output + input quant results (input_fp8, input_scale)
                - "with_input_transpose_quant": Returns output + transposed quant results (input_t_fp8, input_t_scale)

        Returns:
            Different combinations of tensors based on return_mode

        Raises:
            RuntimeError: If return_mode is not supported
        """
        # check input
        is_input_quantized = isinstance(input, (tuple, list)) and len(input) == 2

        if is_input_quantized:
            input_fp8, input_scale = input
            if return_mode == "with_input_transpose_quant":
                raise RuntimeError(
                    "Cannot return transposed quant if input is already quantized. " "Use raw input instead."
                )
        else:
            # quant input (with optional transposed output)
            if return_mode == "with_input_transpose_quant":
                input_fp8, input_scale, input_t_fp8, input_t_scale = FP8LinearFunctionBase.padding_and_quant_input(
                    input
                )
            else:
                input_fp8, input_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    input,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    return_transpose_only=False,
                )

        # quant weight
        weight_fp8, weight_scale = weight_quant(weight, weight_transpose)

        # FP8 GEMM
        if out is None:
            out = paddle.empty([input_fp8.shape[0], weight_fp8.shape[0]], dtype=weight.dtype)

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (input_fp8, input_scale.T), (weight_fp8, weight_scale), out, num_sms=get_sm_num()
        )

        # Return outputs
        if return_mode == "output_only":
            return out
        elif return_mode == "with_input_quant":
            return (out, input_fp8, input_scale)
        elif return_mode == "with_input_transpose_quant":
            return (out, input_t_fp8, input_t_scale)
        else:
            raise RuntimeError(
                f"Unsupported return_mode: {return_mode}. "
                "Supported modes: 'output_only', 'with_input_quant', 'with_input_transpose_quant'"
            )

    @staticmethod
    def compute_expert_w_grad(
        input_t,
        input_t_scale,
        dout_t,
        dout_t_scale,
        is_a_1d_scaled=True,
        is_b_1d_scaled=True,
        weight=None,
        rtn_dtype=paddle.bfloat16,
    ):
        """
        Unified gradient computation for expert_w weights (supports both main_grad and regular grad).
        """

        if input_t is None or numpy.prod(input_t.shape) == 0:
            return

        if hasattr(weight, "main_grad"):
            if weight.main_grad is None:
                weight.main_grad = paddle.zeros(shape=weight.shape, dtype=paddle.float32)
            if WeightGradStore.enabled:
                WeightGradStore.put(
                    partial(
                        FP8LinearFunctionBase.kitchen_gemm,
                        input_t,
                        input_t_scale,
                        dout_t,
                        dout_t_scale,
                        is_a_1d_scaled,
                        is_b_1d_scaled,
                        weight.main_grad,
                        rtn_dtype,
                    )
                )
                result = None

            else:
                result = FP8LinearFunctionBase.kitchen_gemm(
                    input_t,
                    input_t_scale,
                    dout_t,
                    dout_t_scale,
                    is_a_1d_scaled,
                    is_b_1d_scaled,
                    weight.main_grad,
                    rtn_dtype,
                )
        else:
            if weight.grad is None:
                weight.grad = paddle.zeros(shape=weight.shape, dtype=paddle.float32)
            result = FP8LinearFunctionBase.kitchen_gemm(
                input_t, input_t_scale, dout_t, dout_t_scale, is_a_1d_scaled, is_b_1d_scaled, weight.grad, rtn_dtype
            )

        if hasattr(weight, "_apply_backward_hook"):
            weight._apply_backward_hook()
        return result

    @staticmethod
    def common_fp8_mlp_bwd(
        do3, x_t_fp8, x_t_scale, w1, w2, o1=None, x_fp8=None, x_scale=None, apply_backward_hook=False
    ):
        if o1 is not None and (x_fp8 is not None or x_scale is not None):
            raise ValueError("When o1 is provided, both x_fp8 and x_scale must be None.")

        if o1 is None:
            if x_fp8 is None or x_scale is None:
                raise ValueError("When o1 is None, both x_fp8 and x_scale must be provided.")

            # [recompute] o1 = deep_gemm(x_fp8, w1_t_fp8)

            # Recompute o1 using deep_gemm(x_fp8, w1_t_fp8)
            w1_fp8, w1_scale = weight_quant(w1, True)
            o1 = paddle.empty([x_fp8.shape[0], w1_fp8.shape[0]], dtype=do3.dtype)
            deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale.T), (w1_fp8, w1_scale), o1, num_sms=get_sm_num())

        # [recompute] o2 = swiglu(o1)
        o2 = swiglu(o1)

        # do2 = deep_gemm(do3_fp8, w2_fp8)
        do2, do3_t_fp8, do3_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
            do3, w2, return_mode="with_input_transpose_quant"
        )

        # dw2 = deep_gemm(o2_t_fp8, do3_t_fp8)
        o2 = FP8LinearFunctionBase.padding(o2, 0)
        o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            o2, output_scale_transpose=True, quant_method="1x128", input_transpose=True, return_transpose_only=True
        )
        if apply_backward_hook:
            if WeightGradStore.enabled:
                WeightGradStore.put(
                    partial(
                        FP8LinearFunctionBase.compute_expert_w_grad,
                        o2_t_fp8,
                        o2_t_scale,
                        do3_t_fp8,
                        do3_t_scale,
                        True,
                        True,
                        w2,
                        rtn_dtype=paddle.float32,
                    )
                )
            else:

                FP8LinearFunctionBase.compute_expert_w_grad(
                    o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True, w2, rtn_dtype=paddle.float32
                )
        else:
            dw2 = FP8LinearFunctionBase.kitchen_gemm(
                o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, True, True, rtn_dtype=paddle.float32
            )

        # do1 = swiglu_grad(o1, None, do2)
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)

        # dx = deep_gemm(do1_fp8, w1_fp8)
        dx, do1_t_fp8, do1_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
            do1, w1, return_mode="with_input_transpose_quant"
        )

        # dw1 = deep_gemm(x_t_fp8, do1_t_fp8)
        if apply_backward_hook:
            if WeightGradStore.enabled:
                WeightGradStore.put(
                    partial(
                        FP8LinearFunctionBase.compute_expert_w_grad,
                        x_t_fp8,
                        x_t_scale,
                        do1_t_fp8,
                        do1_t_scale,
                        True,
                        True,
                        w1,
                        rtn_dtype=paddle.float32,
                    )
                )

            else:
                FP8LinearFunctionBase.compute_expert_w_grad(
                    x_t_fp8, x_t_scale, do1_t_fp8, do1_t_scale, True, True, w1, rtn_dtype=paddle.float32
                )
        else:
            dw1 = FP8LinearFunctionBase.kitchen_gemm(
                x_t_fp8, x_t_scale, do1_t_fp8, do1_t_scale, True, True, rtn_dtype=paddle.float32
            )

        if apply_backward_hook:
            return dx
        else:
            assert dw1 is not None and dw2 is not None
            return dx, dw1, dw2

    @staticmethod
    def fp8_mlp_fwd(x, w1, w2):
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        # o1 = deep_gemm(x_fp8, w1_t_fp8)
        o1, x_fp8, x_scale = FP8LinearFunctionBase.compute_fp8_linear(
            x, w1, weight_transpose=True, return_transpose_only=True, return_mode="with_input_quant"
        )

        # o2 = swiglu(o1)
        o2 = swiglu(o1)

        # o3 = deep_gemm(o2_fp8, w2_t_fp8)
        o3 = FP8LinearFunctionBase.compute_fp8_linear(o2, w2, weight_transpose=True, return_transpose_only=True)

        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        return o1, x_fp8, x_scale, o3

    @staticmethod
    def fp8_mlp_fwd_norm_rc(x, norm_w, norm_eps, w1, w2):
        # compute norm_output
        norm_output, _ = fused_ln.fused_rms_norm(x, norm_w, norm_eps)
        # compute fp8_mlp_fwd
        _, _, _, o3 = FP8LinearFunctionBase.fp8_mlp_fwd(norm_output, w1, w2)
        return o3

    @staticmethod
    def fp8_mlp_bwd(do3, x, w1, w2, apply_backward_hook=False):
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        x_fp8, x_scale, x_t_fp8, x_t_scale = FP8LinearFunctionBase.padding_and_quant_input(x)

        if apply_backward_hook:
            dx = FP8LinearFunctionBase.common_fp8_mlp_bwd(
                do3,
                x_t_fp8,
                x_t_scale,
                w1,
                w2,
                o1=None,
                x_fp8=x_fp8,
                x_scale=x_scale,
                apply_backward_hook=apply_backward_hook,
            )
            if len(x_orig_shape) > 2:
                dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])
            return dx
        else:
            dx, dw1, dw2 = FP8LinearFunctionBase.common_fp8_mlp_bwd(
                do3,
                x_t_fp8,
                x_t_scale,
                w1,
                w2,
                o1=None,
                x_fp8=x_fp8,
                x_scale=x_scale,
                apply_backward_hook=apply_backward_hook,
            )
            if len(x_orig_shape) > 2:
                dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

            return dx, dw1, dw2

    @staticmethod
    def fp8_mlp_bwd_norm_rc(do3, x, norm_w, norm_eps, w1, w2):
        # recompute norm_output
        norm_output, invar = fused_ln.fused_rms_norm(x, norm_w, norm_eps)

        # compute fp8_mlp_fwd
        d_norm_output = FP8LinearFunctionBase.fp8_mlp_bwd(do3, norm_output, w1, w2, True)

        # ===== compute norm grad =====
        dx, d_rms_norm_weight = fused_ln.fused_rms_norm_grad_func(x, norm_w, invar, d_norm_output, norm_eps)
        if hasattr(norm_w, "main_grad"):
            if norm_w.main_grad is None:
                norm_w.main_grad = paddle.zeros(shape=norm_w.shape, dtype=paddle.float32)
            norm_w.main_grad += d_rms_norm_weight
        else:
            if norm_w.grad is None:
                norm_w.grad = paddle.zeros(shape=norm_w.shape, dtype=paddle.float32)
            norm_w.grad += d_rms_norm_weight

        if hasattr(norm_w, "_apply_backward_hook"):
            norm_w._apply_backward_hook()

        return dx, norm_output, invar


class FP8LinearFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, custom_map, keep_x=False):
        weight = custom_map.weight
        x_orig_shape = x.shape

        # deep_gemm only support 2D
        x = x.reshape([-1, x_orig_shape[-1]]).contiguous()

        if keep_x:
            out = FP8LinearFunctionBase.compute_fp8_linear(
                x,
                weight,
                weight_transpose=True,
                return_transpose_only=True,
            )
            # save for bwd
            out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])
            ctx.save_for_backward(x, weight)
            return out
        else:
            x_t = x.T
            out, x_t_fp8, x_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
                x, weight, weight_transpose=True, return_transpose_only=True, return_mode="with_input_transpose_quant"
            )
            out = out.reshape([x_orig_shape[0], -1, weight.shape[-1]])
            ctx.save_for_backward((x_t_fp8, x_t_scale), weight)
            ctx.x_t_shape = x_t.shape
            return out

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensor()
        dout_2d = dout.reshape([-1, dout.shape[-1]])

        keep_x = not isinstance(x, tuple)

        if keep_x:
            # padding x and quant
            dx_orig_shape = x.shape
            x = FP8LinearFunctionBase.padding(x, 0)
            x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x, output_scale_transpose=True, quant_method="1x128", input_transpose=True, return_transpose_only=True
            )

            # dx = deep_gemm(dout_fp8, w_fp8)
            dx, dout_t_fp8, dout_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
                dout_2d, weight, weight_transpose=False, return_mode="with_input_transpose_quant"
            )
            dx = dx.reshape(dx_orig_shape)

        else:
            x_t_fp8, x_t_scale = x

            # dx = deep_gemm(dout_fp8, w_fp8)
            dx, dout_t_fp8, dout_t_scale = FP8LinearFunctionBase.compute_fp8_linear(
                dout_2d, weight, weight_transpose=False, return_mode="with_input_transpose_quant"
            )
            dx_orig_shape = dout.shape[:-1]
            dx_orig_shape.append(ctx.x_t_shape[0])
            dx = dx.reshape(dx_orig_shape)

        # dw1 = deep_gemm(x_t_fp8, dout_t_fp8)
        FP8LinearFunctionBase.compute_expert_w_grad(
            x_t_fp8, x_t_scale, dout_t_fp8, dout_t_scale, True, True, weight, paddle.float32
        )
        return dx


class FP8Linear(paddle.nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias_attr: bool = False) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype="bfloat16",
            is_bias=False,
        )

    def forward(self, x):
        return FP8LinearFunction.apply(x, self, keep_x=False)


def cache_fp8_weight(weight, quant_transpose=None):
    if hasattr(weight, "fp8_weight") or hasattr(weight, "fp8_weight_transpose"):
        return
    if quant_transpose is None:
        w_fp8, w_scale, w_t_fp8, w_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight,
            output_scale_transpose=False,
            quant_method="128x128",
            input_transpose=True,
            return_transpose_only=False,
        )

        setattr(weight, "fp8_weight_transpose", w_t_fp8)
        setattr(weight, "fp8_scale_transpose", w_t_scale)
        setattr(weight, "fp8_weight", w_fp8)
        setattr(weight, "fp8_scale", w_scale)
    elif quant_transpose is True:
        w_t_fp8, w_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight,
            output_scale_transpose=False,
            quant_method="128x128",
            input_transpose=True,
            return_transpose_only=True,
        )
        setattr(weight, "fp8_weight_transpose", w_t_fp8)
        setattr(weight, "fp8_scale_transpose", w_t_scale)
    elif quant_transpose is False:
        w_fp8, w_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight,
            output_scale_transpose=False,
            quant_method="128x128",
            input_transpose=False,
            return_transpose_only=False,
        )
        setattr(weight, "fp8_weight", w_fp8)
        setattr(weight, "fp8_scale", w_scale)
    else:
        raise ValueError("quant_transpose must be either True, False or None.")


class FP8KeepXLinear(paddle.nn.Layer):
    def __init__(self, in_features: int, out_features: int, bias_attr: bool = False) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype="bfloat16",
            is_bias=False,
        )
        set_parameter_color([self.weight], "attn_out_project")

    def fp8_quant_weight(self, quant_transpose=None):
        cache_fp8_weight(self.weight, quant_transpose=quant_transpose)

    def forward(self, x):
        return FP8LinearFunction.apply(x, self, keep_x=True)


class FusedNormFP8MLPFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, norm_w, w1, w2, norm_eps):
        # compute norm_output
        norm_output, invar = fused_ln.fused_rms_norm(x, norm_w, norm_eps)
        # reshape for deep_gemm, since deep_gemm only support 2D
        x_orig_shape = norm_output.shape
        norm_output = norm_output.reshape([-1, x_orig_shape[-1]])

        # call func fp8_mlp_fwd
        _, _, _, o3 = FP8LinearFunctionBase.fp8_mlp_fwd(norm_output, w1, w2)

        # reshape to origin shape
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        # save for backward
        ctx.save_for_backward(
            norm_output,
            invar,
            x,
            norm_w,
            w1,
            w2,
            norm_eps,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        # reshape for deep_gemm, since deep_gemm only support 2D
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        # recive saved tensors
        norm_output, invar, x, norm_w, w1, w2, norm_eps, x_orig_shape = ctx.saved_tensor()

        x_fp8, x_scale, x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            norm_output, output_scale_transpose=True, quant_method="1x128", input_transpose=True
        )

        # call func common_fp8_mlp_bwd
        d_norm_output, dw1, dw2 = FP8LinearFunctionBase.common_fp8_mlp_bwd(
            do3, x_t_fp8, x_t_scale, w1, w2, o1=None, x_fp8=x_fp8, x_scale=x_scale
        )

        # reshape to origin shape
        if len(x_orig_shape) > 2:
            d_norm_output = d_norm_output.reshape([x_orig_shape[0], -1, d_norm_output.shape[-1]])

        # compute norm grad
        dx, d_rms_norm_weight = fused_ln.fused_rms_norm_grad_func(x, norm_w, invar, d_norm_output, norm_eps)

        return dx, d_rms_norm_weight, dw1, dw2


class FP8MlpFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w1, w2, recompute_fwd_gate_up):
        # reshape for deep_gemm, since deep_gemm only support 2D
        x_orig_shape = x.shape
        x = x.reshape([-1, x_orig_shape[-1]])

        # call func fp8_mlp_fwd
        o1, x_fp8, x_scale, o3 = FP8LinearFunctionBase.fp8_mlp_fwd(x, w1, w2)
        # reshape to origin shape
        if len(x_orig_shape) > 2:
            o3 = o3.reshape([x_orig_shape[0], -1, o3.shape[-1]])

        # save for backward
        o1 = None if recompute_fwd_gate_up else o1
        ctx.save_for_backward(
            o1,
            x_fp8,
            x_scale,
            w1,
            w2,
            paddle.to_tensor(x_orig_shape, dtype="int64", place=paddle.CPUPlace()),
        )
        return o3

    @staticmethod
    def backward(ctx, do3):
        # reshape for deep_gemm, since deep_gemm only support 2D
        do3_orig_shape = do3.shape
        do3 = do3.reshape([-1, do3_orig_shape[-1]])

        # recive saved tensors
        o1, x_fp8, x_scale, w1, w2, x_orig_shape = ctx.saved_tensor()

        # compute x_t_fp8, x_t_scale for dw1
        x_dequant_fp16 = paddle.incubate.nn.functional.fused_act_dequant(x_fp8, x_scale.T.contiguous())
        x_dequant_fp16 = FP8LinearFunctionBase.padding(x_dequant_fp16, 0)

        x_t_fp8, x_t_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x_dequant_fp16,
            output_scale_transpose=True,
            quant_method="1x128",
            input_transpose=True,
            return_transpose_only=True,
        )

        # call func common_fp8_mlp_bwd
        if o1 is None:
            dx = FP8LinearFunctionBase.common_fp8_mlp_bwd(
                do3, x_t_fp8, x_t_scale, w1, w2, o1=None, x_fp8=x_fp8, x_scale=x_scale, apply_backward_hook=True
            )
        else:
            dx = FP8LinearFunctionBase.common_fp8_mlp_bwd(
                do3, x_t_fp8, x_t_scale, w1, w2, o1=o1, x_fp8=None, x_scale=None, apply_backward_hook=True
            )
        # reshape to origin shape
        if len(x_orig_shape) > 2:
            dx = dx.reshape([x_orig_shape[0], -1, dx.shape[-1]])

        return dx, None, None


class FP8Mlp(paddle.nn.Layer):
    def __init__(
        self,
        config,
        hidden_size=None,
        intermediate_size=None,
        is_moe=False,
        using_post_norm_recompute=False,
        norm_weight=None,
        norm_eps=None,
        recompute_fwd_gate_up=False,
    ):
        super().__init__()
        self.config = config
        self.using_post_norm_recompute = using_post_norm_recompute
        if self.using_post_norm_recompute:
            assert norm_weight is not None and norm_eps is not None
            self.norm_weight = norm_weight
            self.norm_eps = norm_eps

        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.recompute_fwd_gate_up = recompute_fwd_gate_up

        self.w1 = self.create_parameter(
            shape=[self.hidden_size, self.intermediate_size * 2],
            dtype="bfloat16",
            is_bias=False,
        )
        self.w2 = self.create_parameter(
            shape=[self.intermediate_size, self.hidden_size],
            dtype="bfloat16",
            is_bias=False,
        )

    def fp8_quant_weight(self, quant_transpose=None):
        cache_fp8_weight(self.w1, quant_transpose)
        cache_fp8_weight(self.w2, quant_transpose)

    def forward(self, x):
        if self.using_post_norm_recompute:
            return FusedNormFP8MLPFunction.apply(x, self.norm_weight, self.w1, self.w2, self.norm_eps)
        else:
            return FP8MlpFunction.apply(x, self.w1, self.w2, self.recompute_fwd_gate_up)


def split_group_gemm(x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out):
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x_fp8[start_idx:end_idx], x_scale_tma_align),
            (w_fp8[i], w_scale[i]),
            gemm_out[start_idx:end_idx],
            num_sms=get_sm_num(),
        )

        start_idx = end_idx

    return gemm_out


class FP8GroupGemmMlpFunctionNode:
    def __init__(
        self,
        custom_map,
        recompute_fwd_gate_up=False,
        is_split_group_gemm=False,
        name="experts_group_gemm_contiguous_node",
    ):
        self.experts = custom_map.experts
        self.recompute_fwd_gate_up = recompute_fwd_gate_up
        self.is_split_group_gemm = is_split_group_gemm
        self.m_indices = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.all_unzipped_grad = None
        self.fwd_subbatch = None
        self.bwd_subbatch = None

    def reset_statue(self):
        self.m_indices = None
        self.fwd_subbatch = None
        self.bwd_subbatch = None
        self.clear_activation_tensors()

    def clear_activation_tensors(self):
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.all_unzipped_grad = None

    def gen_m_indices(self, tokens_per_expert):
        tokens = []
        for i in range(len(tokens_per_expert)):
            tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
        out = paddle.cat(tokens, axis=0)
        return out

    def fwd_gate_up(self, x, expert_w1, num_expert, tokens_per_expert, m_indices=None):
        """
        o1 = x * w1
        [m_sum, n] = [m_sum, k] * [num_groups, k, n] (m_sum = sum(tokens_per_expert))
        """
        if not self.is_split_group_gemm and self.m_indices is None:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        # concat w1, shape is [num_groups, n, k]
        w1_t_quant, w1_t_scale = fused_stack_quant(expert_w1, transpose=True)
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if hasattr(expert_w1[0], "fp8_weight_stacked") and not hasattr(expert_w1[0], "fp8_weight_stacked_transpose"):
            w1_t_quant = w1_t_quant.contiguous().transpose([0, 2, 1]).contiguous()
            w1_t_scale = w1_t_scale.contiguous().transpose([0, 2, 1]).contiguous()

        if x is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
        else:
            if isinstance(x, tuple):
                (x_fp8, x_scale) = x
                x_scale = paddle.transpose(paddle.transpose(x_scale, [1, 0]).contiguous(), [1, 0])
            else:
                # quant x_bf16
                x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    x, output_scale_transpose=True, quant_method="1x128", input_transpose=False
                )
                x_scale = x_scale.T

        # compute gemm
        o1 = paddle.empty([x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype)
        if numpy.prod(x_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(x_fp8, x_scale, w1_t_quant, w1_t_scale, tokens_per_expert, o1)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (x_fp8, x_scale),
                    (w1_t_quant, w1_t_scale),
                    o1,
                    m_indices=self.m_indices if m_indices is None else m_indices,
                    num_sms=get_sm_num(),
                )

        if m_indices is None:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down(
        self, o1, unzipped_probs, expert_w2, num_expert, tokens_per_expert, m_indices=None, o3=None, clear_o1=False
    ):
        """
        o3 = o2 * w2
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # concat and transpose w2
        w2_quant, w2_scale = fused_stack_quant(expert_w2, transpose=True)
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([num_expert, -1, w2_scale.shape[-1]])

        if hasattr(expert_w2[0], "fp8_weight_stacked") and not hasattr(expert_w2[0], "fp8_weight_stacked_transpose"):
            w2_quant = w2_quant.contiguous().transpose([0, 2, 1]).contiguous()
            w2_scale = w2_scale.contiguous().transpose([0, 2, 1]).contiguous()

        # quant o2
        with paddle.amp.auto_cast(False):
            unzipped_probs = unzipped_probs.squeeze(-1)
            o2_fp8, o2_scale = paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                o1, unzipped_probs, using_pow2_scaling=True
            )
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0])

        if clear_o1:
            o1._clear_to_zero_allocation()

        # compute gemm
        o3_shape = [o2_fp8.shape[0], w2_quant.shape[1]]
        if o3 is not None:
            assert o3.shape == o3_shape, "{} vs {}".format(o3.shape, o3_shape)
        else:
            o3 = paddle.empty(o3_shape, dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(o2_fp8, o2_scale, w2_quant, w2_scale, tokens_per_expert, o3)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (o2_fp8, o2_scale),
                    (w2_quant, w2_scale),
                    o3,
                    m_indices=m_indices if self.fwd_subbatch else self.m_indices,
                    num_sms=get_sm_num(),
                )

        return o3

    def bwd_dowm_input(self, expert_w2, unzipped_grad, o1, tokens_per_expert, m_indices=None, unzipped_probs=None):
        """
        do2 = do3 * w2_t
        [m_sum, n] = [m_sum, k] * [num_groups, k, n]
        """
        # recompute concated_w2_2d
        bw_w2_quant, bw_w2_scale = fused_stack_quant(expert_w2, transpose=False)
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        if hasattr(expert_w2[0], "fp8_weight_stacked_transpose") and not hasattr(expert_w2[0], "fp8_weight_stacked"):
            bw_w2_quant = bw_w2_quant.contiguous().transpose([0, 2, 1]).contiguous()
            bw_w2_scale = bw_w2_scale.contiguous().transpose([0, 2, 1]).contiguous()

        # compute gemm
        if isinstance(unzipped_grad, tuple):
            (unzipped_grad_fp8, unzipped_grad_scale) = unzipped_grad
            unzipped_grad_scale = unzipped_grad_scale.T.contiguous().T
        else:
            unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                unzipped_grad, output_scale_transpose=True, quant_method="1x128", input_transpose=False
            )
            unzipped_grad_scale = unzipped_grad_scale.T

        do2_s = paddle.empty([unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]], dtype="bfloat16")
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    unzipped_grad_fp8, unzipped_grad_scale, bw_w2_quant, bw_w2_scale, tokens_per_expert, do2_s
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=m_indices if self.bwd_subbatch else self.m_indices,
                    num_sms=get_sm_num(),
                )

        with paddle.amp.auto_cast(False):
            do1, probs_grad, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(o1, do2_s, unzipped_probs)

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input(self, do1, expert_w1, tokens_per_expert, m_indices=None, dx=None):
        """
        dx = do1 * w1_t
        [m_sum, k] = [m_sum, n] * [num_groups, n, k]
        """
        # recompute concated_w1_t
        bw_w1_quant, bw_w1_scale = fused_stack_quant(expert_w1, transpose=False)
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        if hasattr(expert_w1[0], "fp8_weight_stacked_transpose") and not hasattr(expert_w1[0], "fp8_weight_stacked"):
            bw_w1_quant = bw_w1_quant.contiguous().transpose([0, 2, 1]).contiguous()
            bw_w1_scale = bw_w1_scale.contiguous().transpose([0, 2, 1]).contiguous()

        # quant do1
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            do1, output_scale_transpose=True, quant_method="1x128", input_transpose=False
        )
        do1_scale = do1_scale.T
        # compute gemm
        dx_shape = [do1_fp8.shape[0], bw_w1_quant.shape[1]]
        if dx is None or dx.dtype != do1.dtype:
            dx = paddle.empty(shape=dx_shape, dtype=do1.dtype)
        else:
            assert dx.shape == dx_shape, f"{dx.shape} vs {dx_shape}"
        if numpy.prod(do1_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(do1_fp8, do1_scale, bw_w1_quant, bw_w1_scale, tokens_per_expert, dx)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (do1_fp8, do1_scale),
                    (bw_w1_quant, bw_w1_scale),
                    dx,
                    m_indices=m_indices if self.bwd_subbatch else self.m_indices,
                    num_sms=get_sm_num(),
                )

        return dx

    def fused_transpose_split_quant(self, x, scale, tokens_per_expert, pow_2_scales):
        out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(
            x, scale, tokens_per_expert, pow_2_scales
        )
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2, tokens_per_expert):
        """
        dw2 = do2_t * do3
        [n, k] = [n, m_sum] * [m_sum, k] (m_sum = sum(tokens_per_expert))
        """
        if isinstance(o2, tuple):
            o2_t_fp8, o2_t_scale = o2
        else:
            o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(o2, None, tokens_per_expert, True)

        if isinstance(do3, tuple):
            do3_t_fp8, do3_t_scale = do3
        else:
            do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(do3, None, tokens_per_expert, True)

        def cal_weight_fn(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, expert_w2):
            with paddle.no_grad():
                for i in range(len(expert_w2)):
                    FP8LinearFunctionBase.compute_expert_w_grad(
                        o2_t_fp8[i],
                        o2_t_scale[i],
                        do3_t_fp8[i],
                        do3_t_scale[i],
                        True,
                        True,
                        expert_w2[i],
                        paddle.float32,
                    )

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(cal_weight_fn, o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, expert_w2))
        else:
            cal_weight_fn(o2_t_fp8, o2_t_scale, do3_t_fp8, do3_t_scale, expert_w2)

    def bwd_gate_up_weight(
        self,
        do1,
        input_x,
        expert_w1,
        tokens_per_expert,
        input_fp8_slice=None,
        input_scale_slice=None,
        clear_input=False,
    ):
        """
        dw1 = dx_t * do1
        [k, n] = [k, m_sum] * [m_sum, n] (m_sum = sum(tokens_per_expert))
        """
        if input_x is None:
            inp = (input_fp8_slice, input_scale_slice) if self.bwd_subbatch else (self.input_fp8, self.input_scale)
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(inp[0], inp[1], tokens_per_expert, True)

        else:
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(input_x, None, tokens_per_expert, True)

        if clear_input:
            if self.input_fp8 is not None:
                self.input_fp8._clear_to_zero_allocation()
                self.input_fp8 = None
            if self.input_scale is not None:
                self.input_scale._clear_to_zero_allocation()
                self.input_scale = None
            if self.input is not None:
                self.input._clear_to_zero_allocation()
                self.input = None

        do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(do1, None, tokens_per_expert, True)

        def cal_weight_fn(input_x_t_fp8, input_x_t_scale, do1_t_fp8, do1_t_scale, expert_w1):
            with paddle.no_grad():
                for i in range(len(expert_w1)):
                    FP8LinearFunctionBase.compute_expert_w_grad(
                        input_x_t_fp8[i],
                        input_x_t_scale[i],
                        do1_t_fp8[i],
                        do1_t_scale[i],
                        True,
                        True,
                        expert_w1[i],
                        paddle.float32,
                    )

        if WeightGradStore.enabled:
            WeightGradStore.put(
                partial(cal_weight_fn, input_x_t_fp8, input_x_t_scale, do1_t_fp8, do1_t_scale, expert_w1)
            )
        else:
            cal_weight_fn(input_x_t_fp8, input_x_t_scale, do1_t_fp8, do1_t_scale, expert_w1)

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert, m_indices=None):
        # check subbatch
        if self.fwd_subbatch:
            assert m_indices is not None
        # deal 0 size
        dtype = paddle.bfloat16
        if hs_out is None:
            assert self.input_fp8 is not None
            assert self.input_scale is not None
            shape = self.input_fp8.shape
        else:
            if isinstance(hs_out, tuple):
                shape = hs_out[0].shape
            else:
                shape = hs_out.shape

        if shape[0] == 0:
            o3 = paddle.zeros(shape, dtype=dtype)
            return o3

        # get w1/w2
        expert_w1 = [x.w1 for x in self.experts if x is not None]
        expert_w2 = [x.w2 for x in self.experts if x is not None]

        num_expert = len(expert_w1)

        # o1
        o1 = self.fwd_gate_up(hs_out, expert_w1, num_expert, tokens_per_expert, m_indices)
        if not self.recompute_fwd_gate_up:
            self.o1 = o1
            clear_o1 = False
        else:
            clear_o1 = True

        # o3
        o3 = self.fwd_down(
            o1, unzipped_probs, expert_w2, num_expert, tokens_per_expert, clear_o1=clear_o1, m_indices=m_indices
        )

        # save for bwd
        return o3

    @paddle.no_grad()
    def backward(
        self,
        out_grad,
        unzipped_probs,
        tokens_per_expert,
        input_fp8_slice=None,
        input_scale_slice=None,
        m_indices=None,
        reset_status=False,
    ):
        # check subbatch
        if self.bwd_subbatch:
            assert (
                m_indices is not None
                and input_fp8_slice is not None
                and input_scale_slice is not None
                and tokens_per_expert is not None
            )
        # deal 0 size
        dtype = paddle.bfloat16
        shape = out_grad[0].shape if isinstance(out_grad, tuple) else out_grad.shape
        if shape[0] == 0:
            return paddle.zeros_like(extract_first_if_tuple(out_grad), dtype=dtype), paddle.zeros_like(unzipped_probs)

        # recompute expert_w2 and expert_w1
        expert_w1 = [x.w1 for x in self.experts if x is not None]
        expert_w2 = [x.w2 for x in self.experts if x is not None]

        if self.recompute_fwd_gate_up:
            inp = None if not self.bwd_subbatch else (input_fp8_slice, input_scale_slice)
            o1 = self.fwd_gate_up(inp, expert_w1, len(expert_w1), tokens_per_expert, m_indices=m_indices)
        else:
            o1 = self.o1

        # do2
        do1, o2_s, probs_grad = self.bwd_dowm_input(
            expert_w2, out_grad, o1, tokens_per_expert, unzipped_probs=unzipped_probs, m_indices=m_indices
        )
        del o1
        if self.o1 is not None:
            self.o1._clear_to_zero_allocation()
        self.o1 = None

        # dw1
        self.bwd_gate_up_weight(
            do1,
            None,
            expert_w1,
            tokens_per_expert,
            input_fp8_slice=input_fp8_slice,
            input_scale_slice=input_scale_slice,
            clear_input=reset_status,
        )

        if reset_status:
            if self.input_fp8 is not None:
                self.input_fp8._clear_to_zero_allocation()
                self.input_fp8 = None
            if self.input_scale is not None:
                self.input_scale._clear_to_zero_allocation()
                self.input_scale = None
            if self.input is not None:
                self.input._clear_to_zero_allocation()
                self.input = None

        # dx
        dx = self.bwd_gate_up_input(
            do1,
            expert_w1,
            tokens_per_expert,
            dx=out_grad[0] if isinstance(out_grad, tuple) else out_grad,
            m_indices=m_indices,
        )
        del do1

        # dw2
        if isinstance(out_grad, tuple):
            do3_fp8, do3_scale = self.fused_transpose_split_quant(out_grad[0], out_grad[1], tokens_per_expert, True)
            out_grad[0]._clear_to_zero_allocation()
            out_grad[1]._clear_to_zero_allocation()
            self.bwd_down_weight((do3_fp8, do3_scale), o2_s, expert_w2, tokens_per_expert)
        else:
            self.bwd_down_weight(out_grad, o2_s, expert_w2, tokens_per_expert)

        if reset_status:
            self.reset_statue()
        return dx, probs_grad
