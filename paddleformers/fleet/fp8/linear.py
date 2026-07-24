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


import logging

import paddle
from paddle.nn.parameter import Parameter

logger = logging.getLogger(__name__)

try:
    from paddlefleet_ops import deep_gemm
except (ImportError, RuntimeError) as e:
    logger.warning(f"deep_gemm not available: {e}")
    deep_gemm = None

from paddleformers.fleet.tensor_parallel import ColumnParallelLinear

from .quantization import get_quant_func
from .utils import is_fp8_tensor


class _FP8Gemm(paddle.autograd.Function):
    """Forward and backward function for FP8 GEMM"""

    @staticmethod
    def forward(
        ctx,
        inp,
        weight,
        inp_quant_func,
        weight_quant_func,
        use_pow2_scale=False,
    ):
        """
        Forward pass for FP8 GEMM.

        out = inp @ weight.T  (using fp8_gemm_nt)
        inp: [M, K], weight: [N, K] -> out: [M, N]
        """

        if is_fp8_tensor(inp) is False:
            inp = inp_quant_func(inp)

        if len(inp) == 2:
            inp_fp8, inp_scale = inp
            inp_t_fp8, inp_t_scale = None, None
        elif len(inp) == 4:
            inp_fp8, inp_scale, inp_t_fp8, inp_t_scale = inp
        else:
            raise ValueError(
                f"Unexpected length of quant_func result: {len(inp)}"
            )

        if is_fp8_tensor(weight) is False:
            weight_fp8, weight_scale = weight_quant_func(weight)
        else:
            weight_fp8, weight_scale = weight

        ctx.save_for_backward(
            inp_t_fp8, inp_t_scale, weight, weight_fp8, weight_scale
        )
        ctx.use_pow2_scale = use_pow2_scale
        out = paddle.empty(
            [inp_fp8.shape[0], weight_fp8.shape[0]], dtype=paddle.bfloat16
        )
        deep_gemm.fp8_gemm_nt(
            (inp_fp8, inp_scale), (weight_fp8, weight_scale), out
        )

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for FP8 GEMM.

        Forward: out = inp @ weight.T  (using fp8_gemm_nt)
        - inp: [M, K], weight: [N, K] -> out: [M, N]

        Backward:
        - grad_input = grad_output @ weight
          grad_output: [M, N], weight: [N, K] -> grad_input: [M, K]
        - grad_weight = grad_output.T @ inp
          grad_output.T: [N, M], inp: [M, K] -> grad_weight: [N, K]
        """
        inp_t_fp8, inp_t_scale, weight, weight_fp8, weight_scale = (
            ctx.saved_tensor()
        )

        assert hasattr(weight, "main_grad"), (
            "fp8 gemm backward requires main_grad of weight"
        )
        if weight.main_grad is not None:
            assert weight.main_grad.dtype == paddle.float32, (
                "fp8 gemm backward requires main_grad of weight to be float32"
            )
        assert inp_t_fp8 is not None and inp_t_scale is not None, (
            "fp8 gemm backward requires inp_t_fp8 and inp_t_scale"
        )

        # Convert grad_output to FP8 format
        # grad_output is typically in BF16/FP16, need to quantize
        (
            grad_out_fp8,
            grad_out_scale,
            grad_out_t_fp8,
            grad_out_t_scale,
        ) = paddle.incubate.nn.functional.fp8_quant_blockwise(
            grad_output,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=True,
            using_pow2_scale=ctx.use_pow2_scale,
        )

        # Compute grad_input = grad_output @ weight
        # grad_output: [M, N], weight: [N, K] -> grad_input: [M, K]
        # Using fp8_gemm_nn: A @ B (no transpose)
        grad_input = paddle.empty(
            [inp_t_fp8.shape[1], inp_t_fp8.shape[0]], dtype=paddle.bfloat16
        )
        deep_gemm.fp8_gemm_nt(
            (grad_out_fp8, grad_out_scale),
            (weight_fp8.T.contiguous(), weight_scale.T),
            grad_input,
        )

        # Compute grad_weight = grad_output.T @ inp
        # grad_output.T: [N, M], inp: [M, K] -> grad_weight: [N, K]
        if hasattr(weight, "main_grad"):
            if weight.main_grad is None:
                weight.main_grad = paddle.zeros_like(
                    weight, dtype=paddle.float32
                )
            main_grad = weight.main_grad

        deep_gemm.fp8_gemm_nt(
            (grad_out_t_fp8, grad_out_t_scale),
            (inp_t_fp8, inp_t_scale),
            main_grad,
            c=main_grad,
            recipe=(1, 1, 128),
        )

        # The gradient has been accumulated in weight.main_grad, so
        # the grad is not returned and the backward hook will not be
        # called automatically. So we manually trigger the hook here.
        if hasattr(weight, "_apply_backward_hook") and not weight.stop_gradient:
            weight._apply_backward_hook()

        # Return gradients for: inp, weight
        return grad_input, None


class FP8Linear(ColumnParallelLinear):
    """FP8 Linear"""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method: callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: paddle.core.ProcessGroup | None = None,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
            tp_group=tp_group,
        )

        if deep_gemm is None:
            raise RuntimeError(
                "FP8Linear requires H-series GPU or above (paddlefleet_ops.deep_gemm not available)"
            )

        # DeepGEMM requires k-major storage, here to make self.weight k-major
        # and keep its shape consistent with [k, m]
        # print("==== self.weight before ====")
        # print(self.weight.strides)
        self.weight = Parameter(self.weight.T.contiguous())
        # print("==== self.weight after ====")
        # print(self.weight.strides)

        self.use_pow2_scale = (
            paddle.device.cuda.get_device_capability()[0] == 10
        )
        self.inp_quant_func, self.weight_quant_func = get_quant_func(
            config.fp8_recipe,
            input_trans=True,
            out_scale_trans=False,
            pow2_scale=self.use_pow2_scale,
        )

    def forward(self, inp):
        out = _FP8Gemm.apply(
            inp,
            self.weight,
            self.inp_quant_func,
            self.weight_quant_func,
            self.use_pow2_scale,
        )
        if self.bias is not None:
            out = out + self.bias
        return out
