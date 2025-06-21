# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
from paddle import nn
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    ReduceScatterOp,
    mark_as_sequence_parallel_parameter,
)

from ...quantization.quantization_linear import quant_weight_linear
from ...utils.log import logger
from .utils import rng_ctx


class QuantizationLoRABaseLinear(nn.Layer):
    def __init__(self, layer, lora_config):
        super().__init__()
        # Model parameters
        self.quantization_config = layer.quantization_config
        self.weight_quantize_algo = layer.weight_quantize_algo
        self._dtype = layer._dtype
        self.quant_dtype = layer.quant_dtype
        self.quant_weight = layer.quant_weight
        if self.weight_quantize_algo in ["fp4", "nf4"] and self.quantization_config.qlora_weight_double_quant:
            self.qweight_scale = layer.qweight_scale
            self.double_weight_scale = layer.double_weight_scale
            self.weight_scale_offset = layer.weight_scale_offset
        else:
            self.weight_scale = layer.weight_scale
        self.bias = layer.bias

        # LoRA related parameters
        self.lora_config = lora_config
        if not isinstance(self.lora_config.r, int) or self.lora_config.r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        if self.weight_quantize_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")
        if self.lora_config.rslora:
            self.scaling = self.lora_config.lora_alpha / math.sqrt(self.lora_config.r)
        else:
            self.scaling = self.lora_config.lora_alpha / self.lora_config.r
        self.disable_lora = False

        # Mark the weight as unmerged
        # Optional dropout
        if self.lora_config.lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=self.lora_config.lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x, add_bias=True):
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
            bias=self.bias if add_bias else None,
        )
        return output

    def merge(self):
        logger.warning("QuantizationLoRALinear does not support merge()")

    def unmerge(self):
        logger.warning("QuantizationLoRALinear does not support unmerge()")


class QuantizationLoRALinear(QuantizationLoRABaseLinear):
    """
    Quantization lora Linear layer.
    The code implementation refers to paddlenformers.peft.lora.lora_layers.LoRALinear.
    https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddlenformers/peft/lora/lora_layers.py
    Compare to LoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
    """

    def __init__(self, layer, lora_config):
        super(QuantizationLoRALinear, self).__init__(layer, lora_config)
        # LoRA parameters
        self.lora_A = self.create_parameter(
            shape=[layer.in_features, self.lora_config.r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        self.lora_B = self.create_parameter(
            shape=[self.lora_config.r, layer.out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        mp_moe = getattr(self.quant_weight, "mp_moe", False)
        is_distributed = getattr(self.quant_weight, "is_distributed", False)
        if mp_moe or is_distributed:
            for p in self.parameters():
                p.is_distributed = is_distributed
                p.mp_moe = mp_moe

    def forward(self, x):
        result = super().forward(x)
        if not self.disable_lora:
            result += (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result


class ColumnParallelQuantizationLoRALinear(QuantizationLoRABaseLinear):
    """
    Quantization lora Linear layer with mp parallelized(column).
    The code implementation refers to paddlenformers.peft.lora.lora_layers.ColumnParallelLoRALinear.
    https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddlenformers/peft/lora/lora_layers.py#L203
    Compare to ColumnParallelLoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
    """

    def __init__(self, layer, lora_config):
        super(ColumnParallelQuantizationLoRALinear, self).__init__(layer, lora_config)

        # Parallel parameters
        self.model_parallel_group = layer.model_parallel_group
        self.world_size = layer.world_size
        self.gather_output = layer.gather_output
        self.sequence_parallel = layer.sequence_parallel
        self.mp_skip_c_identity = layer.mp_skip_c_identity

        # LoRA parameters
        self.lora_A = self.create_parameter(
            shape=[layer.in_features, self.lora_config.r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        # Sync lora_A parameters before training
        self.lora_A.is_distributed = False
        if self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.lora_A)

        self.lora_B = self.create_parameter(
            shape=[self.lora_config.r, layer.output_size_per_partition],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1

    def forward(self, x):
        # base_model forward
        if self.sequence_parallel:
            # forward: all_gather backward: reduce scatter
            input_parallel = AllGatherOp.apply(x)
        else:
            # forward: identity backward: all reduce
            input_parallel = mp_ops._c_identity(
                x,
                group=self.model_parallel_group,
                skip_c_identity_dynamic=self.mp_skip_c_identity,
            )
        output_parallel = super().forward(input_parallel)

        # LoRA forward
        if not self.disable_lora:
            input_a = self.lora_dropout(x) @ self.lora_A
            if self.sequence_parallel:
                # forward: all_gather backward: reduce scatter
                input_a_parallel = AllGatherOp.apply(input_a)
            else:
                # forward: identity backward: all reduce
                input_a_parallel = mp_ops._c_identity(
                    input_a,
                    group=self.model_parallel_group,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            delta_parallel = (input_a_parallel @ self.lora_B) * self.scaling
            output_parallel += delta_parallel

        if self.gather_output:
            output = mp_ops._c_concat(output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output


class RowParallelQuantizationLoRALinear(QuantizationLoRABaseLinear):
    """
    Quantization lora Linear layer with mp parallelized(row).
    The code implementation refers to paddlenformers.peft.lora.lora_layers.RowParallelLoRALinear.
    https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddlenformers/peft/lora/lora_layers.py#L99
    Compare to RowParallelLoRALinear, this class keeps weight in INT8/INT4 with quant scale, and supports
    weight_only_linear for input tensor and origin weight(LoRA part still uses fp16/bf16).
    """

    def __init__(self, layer, lora_config):
        super(RowParallelQuantizationLoRALinear, self).__init__(layer, lora_config)
        # Parallel parameters
        self.model_parallel_group = layer.model_parallel_group
        self.world_size = layer.world_size
        self.input_is_parallel = layer.input_is_parallel
        if not self.input_is_parallel and self.sequence_parallel:
            raise ValueError("Sequence parallel only support input_is_parallel.")
        self.sequence_parallel = layer.sequence_parallel
        self.mp_skip_c_identity = layer.mp_skip_c_identity

        # LoRA parameters
        with rng_ctx(True, paddle.in_dynamic_mode()):
            self.lora_A = self.create_parameter(
                shape=[layer.input_size_per_partition, self.lora_config.r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0

        self.lora_B = self.create_parameter(
            shape=[self.lora_config.r, layer.out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        # Sync lora_B parameters before training
        self.lora_B.is_distributed = False
        if self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.lora_B)

    def forward(self, x):
        if self.input_is_parallel:
            input_parallel = x
        else:
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        # base_model forward
        output_parallel = super().forward(input_parallel, add_bias=False)
        if self.sequence_parallel:
            output = ReduceScatterOp.apply(output_parallel)
        else:
            output = mp_ops._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
                skip_c_identity_dynamic=self.mp_skip_c_identity,
            )
        output = output + self.bias if self.bias is not None else output

        # LoRA forward
        if not self.disable_lora:
            input_a_parallel = self.lora_dropout(input_parallel) @ self.lora_A
            if self.sequence_parallel:
                input_a_parallel = ReduceScatterOp.apply(input_a_parallel)
            else:
                input_a_parallel = mp_ops._mp_allreduce(
                    input_a_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            delta = (input_a_parallel @ self.lora_B) * self.scaling
            output += delta
        return output
