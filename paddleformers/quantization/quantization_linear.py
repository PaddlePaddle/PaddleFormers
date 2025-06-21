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
import paddle.nn as nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    ReduceScatterOp,
)
from paddle.nn.quant import llm_int8_linear, weight_dequantize, weight_only_linear

from ..utils import infohub
from .qat_utils import QATFunc

try:
    from .qlora import qlora_weight_dequantize, qlora_weight_linear
except:
    qlora_weight_linear = None
    qlora_weight_dequantize = None

QuantMapping = {
    # (quant_dtype, quant_weight_bit)
    "weight_only_int8": ("int8", 8),
    "weight_only_int4": ("int4", 4),
    "llm.int8": ("int8", 8),
    "fp4": ("fp4", 4),
    "nf4": ("nf4", 4),
    "a8w8linear": ("int8", 8),
    "a8w4linear": ("int8", 8),
    "fp8linear": ("fp8", 8),
}


def quant_weight_forward(
    x,
    quant_weight,
    bias,
    weight_scale,
    quant_state,
    quant_dtype,
    quantization_config,
    weight_quantize_algo,
    dtype,
):
    if weight_quantize_algo in ["weight_only_int8", "weight_only_int4"]:
        output = weight_only_linear(
            x=x,
            weight=quant_weight,
            bias=bias,
            weight_scale=weight_scale,
            weight_dtype=quant_dtype,
            group_size=quantization_config.group_size,
        )
    elif weight_quantize_algo in ["llm.int8"]:
        output = llm_int8_linear(x, quant_weight, bias, weight_scale, quantization_config.llm_int8_threshold)
    elif weight_quantize_algo in ["fp4", "nf4"]:
        output = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=dtype,
            state=quant_state if quantization_config.qlora_weight_double_quant else weight_scale,
            quant_algo=weight_quantize_algo,
            double_quant=quantization_config.qlora_weight_double_quant,
            block_size=quantization_config.qlora_weight_blocksize,
            double_quant_block_size=quantization_config.qlora_weight_double_quant_block_size,
            bias=bias,
        )

    return output


def dequant_weight(
    quant_weight,
    quantization_config,
    weight_quantize_algo,
    dtype,
    weight_scale,
    quant_state,
    input_shape,
):
    if weight_quantize_algo in ["weight_only_int8", "weight_only_int4", "llm.int8"]:
        quant_dequant_weight = weight_dequantize(
            x=quant_weight,
            scale=weight_scale,
            algo=weight_quantize_algo,
            out_dtype=dtype,
            group_size=quantization_config.group_size,
        )
    elif weight_quantize_algo in ["fp4", "nf4"]:
        quant_dequant_weight = (
            qlora_weight_dequantize(
                quant_weight=quant_weight,
                quant_algo=weight_quantize_algo,
                state=quant_state if quantization_config.qlora_weight_double_quant else weight_scale,
                double_quant=quantization_config.qlora_weight_double_quant,
                block_size=quantization_config.qlora_weight_blocksize,
                double_quant_block_size=quantization_config.qlora_weight_double_quant_block_size,
            )
            .reshape([input_shape[-1], -1])
            .cast(dtype)
        )
    return quant_dequant_weight


class QuantizationLinearFunc(PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        quant_weight,
        bias,
        weight_scale,
        quant_state,
        quant_dtype,
        quantization_config,
        weight_quantize_algo,
        dtype,
    ):

        output = quant_weight_forward(
            x=x,
            quant_weight=quant_weight,
            bias=bias,
            weight_scale=weight_scale,
            quant_state=quant_state,
            quant_dtype=quant_dtype,
            quantization_config=quantization_config,
            weight_quantize_algo=weight_quantize_algo,
            dtype=dtype,
        )
        ctx.quant_dtype = quant_dtype
        ctx.quantization_config = quantization_config
        ctx.weight_quantize_algo = weight_quantize_algo
        ctx.dtype = dtype
        if ctx.weight_quantize_algo in ["fp4", "nf4"] and ctx.quantization_config.qlora_weight_double_quant:
            qweight_scale, double_weight_scale, weight_scale_offset = quant_state
            ctx.save_for_backward(x, quant_weight, bias, qweight_scale, double_weight_scale, weight_scale_offset)
        else:
            ctx.save_for_backward(x, quant_weight, bias, weight_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight_quantize_algo in ["fp4", "nf4"] and ctx.quantization_config.qlora_weight_double_quant:
            x, quant_weight, bias, qweight_scale, double_weight_scale, weight_scale_offset = ctx.saved_tensor()
            quant_state = (qweight_scale, double_weight_scale, weight_scale_offset)
            weight_scale = None
        else:
            x, quant_weight, bias, weight_scale = ctx.saved_tensor()
            quant_state = None

        qdq_weight = dequant_weight(
            quant_weight=quant_weight,
            quantization_config=ctx.quantization_config,
            weight_quantize_algo=ctx.weight_quantize_algo,
            dtype=ctx.dtype,
            weight_scale=weight_scale,
            quant_state=quant_state,
            input_shape=x.shape,
        )

        if not x.stop_gradient:
            input_grad = paddle.matmul(grad_output, qdq_weight.T)
        else:
            input_grad = None

        if not quant_weight.stop_gradient:
            if len(x.shape) == 2:
                weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
            else:
                weight_grad = paddle.matmul(
                    x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
                )
        else:
            weight_grad = None

        if bias is not None and not bias.stop_gradient:
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None

        return input_grad, weight_grad, bias_grad


def quant_weight_linear(
    x,
    quant_weight,
    quant_dtype,
    quantization_config,
    weight_quantize_algo,
    dtype,
    weight_scale=None,
    quant_state=None,
    bias=None,
    act_state=None,
):
    if weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:

        state, training, activation_scale, group = act_state

        return QATFunc.apply(
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
        )
    else:
        return QuantizationLinearFunc.apply(
            x,
            quant_weight,
            bias,
            weight_scale,
            quant_state,
            quant_dtype,
            quantization_config,
            weight_quantize_algo,
            dtype,
        )


def get_activation_scale_group(is_row=False):
    if paddle.distributed.is_initialized():
        if getattr(infohub, "scale_group") is None:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg._dp_degree * hcg._sharding_degree
            group_no_row = hcg.create_fuse_group(["data", "sharding"])[1] if rank > 1 else None
            rank *= hcg._mp_degree
            group_row = hcg.create_fuse_group(["data", "sharding", "model"])[1] if rank > 1 else None

            setattr(infohub, "scale_group", [group_no_row, group_row])
        group = infohub.scale_group[1] if is_row else infohub.scale_group[0]
    else:
        group = None
    return group


class QuantizationLinear(nn.Layer):
    """Quantization Linear layer."""

    def __init__(
        self,
        in_features,
        out_features,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
        mp_moe=False,
        is_distributed=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_config = quantization_config
        self.weight_quantize_algo = weight_quantize_algo
        self._dtype = dtype
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]
        self.state = 0

        # PaddlePaddle doesn't support 4bit data type, one 8bit data represents two 4bit data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.weight_quantize_algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]:
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.quant_weight = self.create_parameter(
                    shape=[in_features, out_features],
                    dtype="int8",
                    is_bias=False,
                )
            else:
                self.quant_weight = self.create_parameter(
                    shape=[out_features // 2, in_features]
                    if self.quant_weight_bit == 4
                    else [out_features, in_features],
                    dtype="int8",
                    is_bias=False,
                )
            if self.quantization_config.group_size == -1:
                self.weight_scale = self.create_parameter(
                    shape=[out_features] if self.weight_quantize_algo not in ["fp8linear"] else [1],
                    dtype=self._dtype,
                    is_bias=False,
                )
                self.weight_scale.stop_gradient = True
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.activation_scale = self.create_parameter(
                    shape=[1], dtype=self._dtype, is_bias=False, default_initializer=nn.initializer.Constant(value=0.0)
                )
                self.activation_scale.stop_gradient = True
                self.group = get_activation_scale_group()

        elif self.weight_quantize_algo in ["fp4", "nf4"]:
            if qlora_weight_linear is None:
                raise ImportError(
                    "Please run the following commands to install: qlora related package first\n"
                    "1) git clone https://github.com/PaddlePaddle/PaddleSlim \n"
                    "2) cd PaddleSlim && pip install -e .\n"
                    "3) cd csrc &&  python ./setup_cuda.py install"
                )
            self.quant_weight = self.create_parameter(
                shape=[out_features * in_features // 2, 1],
                attr=paddle.nn.initializer.Constant(value=0),
                dtype="uint8",
                is_bias=False,
            )
            if self.quantization_config.qlora_weight_double_quant:
                # quantized weight_scale
                self.qweight_scale = self.create_parameter(
                    shape=[in_features * out_features // self.quantization_config.qlora_weight_blocksize],
                    dtype="uint8",
                    is_bias=False,
                )
                # double weight_scale: weight_scale of quantized weight_scale
                self.double_weight_scale = self.create_parameter(
                    shape=[
                        in_features
                        * out_features
                        // self.quantization_config.qlora_weight_blocksize
                        // self.quantization_config.qlora_weight_double_quant_block_size
                    ],
                    dtype="float32",
                    is_bias=False,
                )
                self.weight_scale_offset = self.create_parameter(
                    shape=[],
                    dtype="float32",
                    is_bias=False,
                )
            else:
                self.weight_scale = self.create_parameter(
                    shape=[in_features * out_features // self.quantization_config.qlora_weight_blocksize],
                    dtype="float32",
                    is_bias=False,
                )
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")
        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
        if mp_moe or is_distributed:
            for p in self.parameters():
                p.is_distributed = is_distributed
                p.mp_moe = mp_moe
        self.quant_weight.weight_quantize_algo = self.weight_quantize_algo

    def forward(self, x):
        output = quant_weight_linear(
            x=x,
            quant_weight=self.quant_weight,
            quant_dtype=self.quant_dtype,
            quantization_config=self.quantization_config,
            weight_quantize_algo=self.weight_quantize_algo,
            dtype=self._dtype,
            weight_scale=self.weight_scale,
            quant_state=(self.qweight_scale, self.double_weight_scale, self.weight_scale_offset)
            if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
            else None,
            bias=self.bias,
            act_state=(self.state, self.training, self.activation_scale, self.group)
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]
            else None,
        )
        if self.training:
            self.state += 1
        return output


class ColumnParallelQuantizationLinear(nn.Layer):
    """Quantization Linear layer with mp parallelized(column).
    The code implementation refers to paddle.distributed.fleet.meta_parallel.ColumnParallelLinear.
    https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L310
    Different from ColumnParallelLinear, this class keeps weight in INT8/INT4 with quant scale, and supports matrix
    multiplication(weight_only_linear/llm_int8_linear) for input tensor(fp16/bf16) and quantized weight(INT8/INT4)
    and bias addition if provided.
    Notice: quantized weight shape is transposed of weight shape in ColumnParallelLinear.
    """

    def __init__(
        self,
        in_features,
        output_size_per_partition,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
        gather_output=True,
        mp_skip_c_identity=False,
        mp_group=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.output_size_per_partition = output_size_per_partition
        self.weight_quantize_algo = weight_quantize_algo
        self.quantization_config = quantization_config
        self._dtype = dtype
        self.mp_skip_c_identity = mp_skip_c_identity
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]
        self.state = 0
        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel and self.gather_output:
            raise ValueError("Sequence parallel does not support gather_output")

        # PaddlePaddle doesn't support Int4 data type, one Int8 data represents two Int4 data.
        if self.weight_quantize_algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]:
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.quant_weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    dtype="int8",
                    is_bias=False,
                )
            else:
                self.quant_weight = self.create_parameter(
                    shape=[self.output_size_per_partition // 2, in_features]
                    if self.quant_dtype == "int4"
                    else [self.output_size_per_partition, in_features],
                    dtype="int8",
                    is_bias=False,
                )
            self.quant_weight.is_distributed = True if self.is_mp else False
            if self.quant_weight.is_distributed:
                self.quant_weight.split_axis = 0

            if self.quantization_config.group_size == -1:
                self.weight_scale = self.create_parameter(
                    shape=[self.output_size_per_partition] if self.weight_quantize_algo not in ["fp8linear"] else [1],
                    dtype=self._dtype,
                    is_bias=False,
                )
                self.weight_scale.stop_gradient = True
                if self.weight_quantize_algo in ["fp8linear", "a8w4linear", "a8w8linear"]:
                    self.weight_scale.is_distributed = False
                else:
                    self.weight_scale.is_distributed = True if self.is_mp else False
                if self.weight_scale.is_distributed:
                    self.weight_scale.split_axis = 0
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.activation_scale = self.create_parameter(
                    shape=[1], dtype=self._dtype, is_bias=False, default_initializer=nn.initializer.Constant(value=0.0)
                )
                self.activation_scale.is_distributed = False
                self.activation_scale.stop_gradient = True
                self.group = get_activation_scale_group()
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")
        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0
        self.quant_weight.weight_quantize_algo = self.weight_quantize_algo

    def forward(self, x):
        if self.is_mp:
            if self.sequence_parallel:
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = mp_ops._c_identity(
                    x,
                    group=self.model_parallel_group,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
        else:
            input_parallel = x

        output_parallel = quant_weight_linear(
            x=input_parallel,
            quant_weight=self.quant_weight,
            quant_dtype=self.quant_dtype,
            quantization_config=self.quantization_config,
            weight_quantize_algo=self.weight_quantize_algo,
            dtype=self._dtype,
            weight_scale=self.weight_scale,
            quant_state=(self.qweight_scale, self.double_weight_scale, self.weight_scale_offset)
            if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
            else None,
            bias=self.bias,
            act_state=(self.state, self.training, self.activation_scale, self.group)
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]
            else None,
        )
        if self.training:
            self.state += 1

        if self.gather_output and self.is_mp:
            output = mp_ops._c_concat(output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output


class RowParallelQuantizationLinear(nn.Layer):
    """Quantization Linear layer with mp parallelized(row).
    The code implementation refers to paddle.distributed.fleet.meta_parallel.RowParallelLinear.
    https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/layers/mpu/mp_layers.py#L517
    Different from RowParallelLinear, this class keeps weight in INT8/INT4 with quant scale, and supports matrix
    multiplication(weight_only_linear/llm_int8_linear) for input tensor(fp16/bf16) and quantized weight(INT8/INT4)
    and bias addition if provided.
    Notice: quantized weight shape is transposed of weight shape in RowParallelLinear.
    """

    def __init__(
        self,
        input_size_per_partition,
        out_features,
        quantization_config,
        weight_quantize_algo,
        dtype,
        bias_attr=None,
        input_is_parallel=False,
        mp_skip_c_identity=False,
        mp_group=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.input_size_per_partition = input_size_per_partition
        self.out_features = out_features
        self.quantization_config = quantization_config
        self.weight_quantize_algo = weight_quantize_algo
        self._dtype = dtype
        self.mp_skip_c_identity = mp_skip_c_identity
        self.quant_dtype, self.quant_weight_bit = QuantMapping[self.weight_quantize_algo]
        self.state = 0

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel
        if not self.input_is_parallel and self.sequence_parallel:
            raise ValueError("Sequence parallel only support input_is_parallel.")

        # PaddlePaddle doesn't support Int4 data type, one Int8 data represents two Int4 data.
        # paddle.nn.quant.weight_quantize will transpose in_features and out_features.
        if self.weight_quantize_algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]:
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.quant_weight = self.create_parameter(
                    shape=[self.input_size_per_partition, out_features],
                    dtype="int8",
                    is_bias=False,
                )
            else:
                self.quant_weight = self.create_parameter(
                    shape=[out_features // 2, self.input_size_per_partition]
                    if self.quant_dtype == "int4"
                    else [out_features, self.input_size_per_partition],
                    dtype="int8",
                    is_bias=False,
                )
            self.quant_weight.is_distributed = True if self.is_mp else False
            if self.quant_weight.is_distributed:
                self.quant_weight.split_axis = 1

            if self.quantization_config.group_size == -1:
                self.weight_scale = self.create_parameter(
                    shape=[out_features] if self.weight_quantize_algo not in ["fp8linear"] else [1],
                    dtype=self._dtype,
                    is_bias=False,
                )
                self.weight_scale.stop_gradient = True
                if self.weight_quantize_algo in ["fp8linear", "a8w4linear", "a8w8linear"]:
                    self.weight_scale.is_distributed = False
                else:
                    self.weight_scale.is_distributed = True if self.is_mp else False
                if self.weight_scale.is_distributed:
                    self.weight_scale.split_axis = 0
            else:
                # TODO(lugimzzz): support groupwise in next PR
                raise NotImplementedError("Not yet support grouwise weightonly quantization.")
            if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                self.activation_scale = self.create_parameter(
                    shape=[1], dtype=self._dtype, is_bias=False, default_initializer=nn.initializer.Constant(value=0.0)
                )
                self.activation_scale.is_distributed = False
                self.activation_scale.stop_gradient = True
                self.group = get_activation_scale_group(is_row=True)
        else:
            raise NotImplementedError(f"Not yet support weight_quantize_algo: {self.weight_quantize_algo}")

        if bias_attr is False:
            self.bias = None
        else:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

        self.quant_weight.weight_quantize_algo = self.weight_quantize_algo

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        # with paddle.amp.auto_cast(enable=False):
        if self.is_mp:
            output_parallel = quant_weight_linear(
                x=input_parallel,
                quant_weight=self.quant_weight,
                quant_dtype=self.quant_dtype,
                quantization_config=self.quantization_config,
                weight_quantize_algo=self.weight_quantize_algo,
                dtype=self._dtype,
                weight_scale=self.weight_scale,
                quant_state=(self.qweight_scale, self.double_weight_scale, self.weight_scale_offset)
                if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
                else None,
                bias=None,
                act_state=(self.state, self.training, self.activation_scale, self.group)
                if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]
                else None,
            )
            if self.sequence_parallel:
                output_ = ReduceScatterOp.apply(output_parallel)
            else:
                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            output = output_ + self.bias if self.bias is not None else output_
        else:
            output = quant_weight_linear(
                x=input_parallel,
                quant_weight=self.quant_weight,
                quant_dtype=self.quant_dtype,
                quantization_config=self.quantization_config,
                weight_quantize_algo=self.weight_quantize_algo,
                dtype=self._dtype,
                weight_scale=self.weight_scale,
                quant_state=(self.qweight_scale, self.double_weight_scale, self.weight_scale_offset)
                if (self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant)
                else None,
                bias=self.bias,
                act_state=(self.state, self.training, self.activation_scale, self.group)
                if self.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]
                else None,
            )
        if self.training:
            self.state += 1

        return output
