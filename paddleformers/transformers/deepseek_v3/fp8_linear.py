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

original_linear = paddle.nn.functional.linear

from typing import Literal, Optional

# from ..linear_utils import RowParallelLinear as PD_RowParallelLinear
from ..linear_utils import ColumnParallelLinear as PD_ColumnParallelLinear
from ..linear_utils import (
    ColumnSequenceParallelLinear as PD_ColumnSequenceParallelLinear,
)
from ..linear_utils import Linear as PD_Linear
from ..linear_utils import RowParallelLinear as PD_RowParallelLinear
from ..linear_utils import RowSequenceParallelLinear as PD_RowSequenceParallelLinear

try:
    from .kernel import act_quant, fp8_gemm, weight_dequant
except:
    pass


__all__ = [
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowSequenceParallelLinear",
]

gemm_impl: Literal["bf16", "fp8"] = "bf16"
block_size = 128


def fp8_linear(
    x: paddle.Tensor, weight: paddle.Tensor, bias: Optional[paddle.Tensor] = None, name=None
) -> paddle.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (paddle.Tensor): The input tensor.
        weight (paddle.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[paddle.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        paddle.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """

    if paddle.in_dynamic_mode():
        if weight.element_size() > 1:
            return original_linear(x, weight, bias)
        elif gemm_impl == "bf16":
            weight = weight_dequant(weight, weight._scale)
            return original_linear(x, weight, bias)
        else:
            x, scale = act_quant(x, block_size)
            y = fp8_gemm(x, scale, weight, weight._scale)
            if bias is not None:
                y += bias
            return y
    else:
        return original_linear(x, weight, bias)


paddle.nn.functional.linear = fp8_linear


def register_scale(self):
    if self.weight.element_size() == 1:
        in_features, out_features = self.weight.shape
        scale_out_features = (out_features + self.block_size - 1) // self.block_size
        scale_in_features = (in_features + self.block_size - 1) // self.block_size
        self.weight_scale_inv = self.create_parameter(
            shape=[scale_in_features, scale_out_features],
            attr=self._weight_attr,
            dtype="float32",
            is_bias=False,
        )
        self.weight._scale = self.weight_scale_inv


class Linear(PD_Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class ColumnParallelLinear(PD_ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class RowParallelLinear(PD_RowParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class ColumnSequenceParallelLinear(PD_ColumnSequenceParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)


class RowSequenceParallelLinear(PD_RowSequenceParallelLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = kwargs.get("block_size", 128)
        register_scale(self)
