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

import paddle
from paddle import nn
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.nn import functional as F
from paddle.nn.quant.format import ConvertibleQuantedLayer


class QuantedLoRALinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLoRALinear is the same as LoRALinear.
    The only difference is that its inputs are all fake quantized.

    Note:
        In order for proper quantization of this layer, we do (W + AB)x instead of Wx + ABx as in LoRALinear.
        The quanted logic is quant(W + AB)x
    """

    def __init__(self, layer: nn.Layer, q_config):
        super().__init__()
        if isinstance(layer.lora_dropout, nn.Dropout):
            raise ValueError("lora_dropout is not supported for QuantedLoRALinear")

        self.weight = layer.weight
        self.lora_A = layer.lora_A
        self.lora_B = layer.lora_B
        self.scaling = layer.scaling
        self.bias = layer.bias
        self.name = layer.name

        # Mark the weight as unmerged
        self.merged = False

        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)
        self.disable_lora = False

    def forward(self, input):

        if self.merged or self.disable_lora:
            weight = self.weight
        else:
            weight = self.weight + self.lora_A @ self.lora_B * self.scaling

        quant_input = self.activation_quanter(input) if self.activation_quanter is not None else input
        quant_weight = self.weight_quanter(weight) if self.weight_quanter is not None else weight

        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        weight = paddle.cast(weight, input.dtype)
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]


class ColumnParallelQuantedLoRALinear(ConvertibleQuantedLayer):
    """
    The computational logic of ColumnParallelQuantedLoRALinear is the same as ColumnParallelLoRALinear.
    The only difference is that its inputs are all fake quantized.

    Note:
        In order for proper quantization of this layer, we do (W + AB)x instead of Wx + ABx as in LoRALinear.
        The quanted logic is quant(W + AB)x
    """

    def __init__(self, layer: nn.Layer, q_config):
        super().__init__()
        if isinstance(layer.lora_dropout, nn.Dropout):
            raise ValueError("lora_dropout is not supported for QuantedLoRALinear")

        self.weight = layer.weight
        self.lora_A = layer.lora_A
        self.lora_B = layer.lora_B
        self.scaling = layer.scaling
        self.bias = layer.bias
        self.name = layer.name
        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group
        self.gather_output = layer.gather_output
        self.is_mp = layer.is_mp

        # Mark the weight as unmerged
        self.merged = False

        # For FakeQuant
        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)
        self.disable_lora = False

    def forward(self, input):

        if self.merged or self.disable_lora:
            weight = self.weight
        else:
            weight = (
                self.weight
                + mp_ops._c_identity(self.lora_A, group=self.model_parallel_group) @ self.lora_B * self.scaling
            )
        quant_input = self.activation_quanter(input) if self.activation_quanter is not None else input
        quant_weight = self.weight_quanter(weight) if self.weight_quanter is not None else weight

        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        if self.is_mp:
            input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
        else:
            input_mp = input

        result_mp = F.linear(x=input_mp, weight=weight, bias=self.bias, name=self.name)

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]


class RowParallelQuantedLoRALinear(ConvertibleQuantedLayer):
    """
    The computational logic of RowParallelQuantedLoRALinear is the same as RowParallelLoRALinear.
    The only difference is that its inputs are all fake quantized.

    Note:
        In order for proper quantization of this layer, we do (W + AB)x instead of Wx + ABx as in LoRALinear.
        The quanted logic is quant(W + AB)x
    """

    def __init__(self, layer: nn.Layer, q_config):
        super().__init__()
        if isinstance(layer.lora_dropout, nn.Dropout):
            raise ValueError("lora_dropout is not supported for QuantedLoRALinear")

        self.weight = layer.weight
        self.lora_A = layer.lora_A
        self.lora_B = layer.lora_B
        self.scaling = layer.scaling
        self.bias = layer.bias
        self.name = layer.name
        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group
        self.input_is_parallel = layer.input_is_parallel
        self.is_mp = layer.is_mp

        # Mark the weight as unmerged
        self.merged = False

        # For FakeQuant
        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)
        self.disable_lora = False

    def forward(self, input):

        if self.merged or self.disable_lora:
            weight = self.weight
        else:
            weight = (
                self.weight
                + self.lora_A @ mp_ops._c_identity(self.lora_B, group=self.model_parallel_group) * self.scaling
            )

        quant_input = self.activation_quanter(input) if self.activation_quanter is not None else input
        quant_weight = self.weight_quanter(weight) if self.weight_quanter is not None else weight

        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(input, group=self.model_parallel_group)
        else:
            input_mp = input

        # x @ W : [bz, in_f / ws] ===> [bz, out_f]
        result_mp = F.linear(x=input_mp, weight=weight, name=self.name)

        output = mp_ops._mp_allreduce(
            result_mp,
            group=self.model_parallel_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )

        output = output + self.bias if self.bias is not None else output
        return output

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]
