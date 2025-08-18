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

import re

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)
from paddle.incubate.nn.layer.fused_linear import FusedLinear
from paddle.nn.quant import weight_quantize

try:
    from .qlora import qlora_weight_linear, qlora_weight_quantize
except:
    qlora_weight_linear = None
    qlora_weight_quantize = None

from ..utils.log import logger
from .qat_utils import quantize
from .quantization_linear import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)

LINEAR_CLASSES = [
    nn.Linear,
    FusedLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
]


def parse_weight_quantize_algo(quantization_config, name):
    if quantization_config.ignore_modules is not None and any(
        re.fullmatch(ignore_module, name) for ignore_module in quantization_config.ignore_modules
    ):
        weight_quantize_algo = None
    elif isinstance(quantization_config.weight_quantize_algo, str):
        weight_quantize_algo = quantization_config.weight_quantize_algo
    else:
        weight_quantize_algo = None
        for algo in quantization_config.weight_quantize_algo:
            if any(re.fullmatch(module, name) for module in quantization_config.weight_quantize_algo[algo]):
                weight_quantize_algo = algo
    return weight_quantize_algo


def replace_with_quantization_linear(model, quantization_config, llm_int8_threshold=6.0):
    for name, child in model.named_sublayers():
        weight_quantize_algo = parse_weight_quantize_algo(quantization_config, name)
        if weight_quantize_algo is None:
            continue
        if any(isinstance(child, linear_class) for linear_class in LINEAR_CLASSES):
            if child.bias is None:
                bias_attr = False
            else:
                bias_attr = None
            parent = model
            *path, last = name.split(".")
            for attr in path:
                parent = getattr(parent, attr)
            if isinstance(child, nn.Linear) or isinstance(child, FusedLinear):
                if getattr(child.weight, "transpose_weight", False):
                    out_feature, in_features = child.weight.shape[0], child.weight.shape[1]
                else:
                    in_features, out_feature = child.weight.shape[0], child.weight.shape[1]
                quant_linear = QuantizationLinear(
                    in_features=in_features,
                    out_features=out_feature,
                    quantization_config=quantization_config,
                    weight_quantize_algo=weight_quantize_algo,
                    dtype=child._dtype,
                    bias_attr=bias_attr,
                    mp_moe=getattr(child.weight, "mp_moe", False),
                    is_distributed=getattr(child.weight, "is_distributed", False),
                )
            elif isinstance(child, ColumnParallelLinear):
                quant_linear = ColumnParallelQuantizationLinear(
                    in_features=child.weight.shape[0],
                    output_size_per_partition=child.weight.shape[1],
                    quantization_config=quantization_config,
                    weight_quantize_algo=weight_quantize_algo,
                    dtype=child._dtype,
                    bias_attr=bias_attr,
                    gather_output=child.gather_output,
                    mp_skip_c_identity=child.mp_skip_c_identity,
                )
            elif isinstance(child, RowParallelLinear):
                quant_linear = RowParallelQuantizationLinear(
                    input_size_per_partition=child.weight.shape[0],
                    out_features=child.weight.shape[1],
                    quantization_config=quantization_config,
                    weight_quantize_algo=weight_quantize_algo,
                    dtype=child._dtype,
                    bias_attr=bias_attr,
                    input_is_parallel=child.input_is_parallel,
                    mp_skip_c_identity=child.mp_skip_c_identity,
                )
            elif isinstance(child, ColumnSequenceParallelLinear):
                quant_linear = ColumnParallelQuantizationLinear(
                    in_features=child.weight.shape[0],
                    output_size_per_partition=child.weight.shape[1],
                    quantization_config=quantization_config,
                    weight_quantize_algo=weight_quantize_algo,
                    dtype=child._dtype,
                    bias_attr=bias_attr,
                    gather_output=False,
                    sequence_parallel=True,
                )
            elif isinstance(child, RowSequenceParallelLinear):
                quant_linear = RowParallelQuantizationLinear(
                    input_size_per_partition=child.weight.shape[0],
                    out_features=child.weight.shape[1],
                    quantization_config=quantization_config,
                    weight_quantize_algo=weight_quantize_algo,
                    dtype=child._dtype,
                    bias_attr=bias_attr,
                    input_is_parallel=True,
                    sequence_parallel=True,
                )
            setattr(parent, last, quant_linear)
            del child


def convert_to_weight_quantize_state_dict(state_dict, name, quantization_config, dtype, weight_quantize_algo):

    weight_name = name + ".weight"
    quant_weight_name = name + ".quant_weight"
    weight_scale_name = name + ".weight_scale"
    activation_scale_name = name + ".activation_scale"

    if quant_weight_name in state_dict and weight_scale_name in state_dict:
        return state_dict
    if weight_name in state_dict:
        # gpu weight_quantize will fix in future
        target_weight = state_dict.pop(weight_name).cast(dtype).cuda()

        if weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
            quant_weight, weight_scale = quantize(
                target_weight,
                weight_quantize_algo,
                "weight",
                quantization_config,
                side="left",
                apply_hadamard=quantization_config.apply_hadamard,
            )
            activation_scale = paddle.ones([1], dtype=dtype).cuda()
            activation_scale.stop_gradient = True
            state_dict[activation_scale_name] = activation_scale
        else:
            quant_weight, weight_scale = weight_quantize(
                x=target_weight,
                algo=weight_quantize_algo,
                group_size=quantization_config.group_size,
            )
        state_dict[quant_weight_name] = quant_weight
        state_dict[weight_scale_name] = weight_scale
        del target_weight
    return state_dict


def convert_to_qlora_state_dict(state_dict, name, quantization_config, dtype, weight_quantize_algo):
    if qlora_weight_quantize is None:
        raise ImportError(
            "Please run the following commands to install qlora related package first: \n"
            "1) git clone https://github.com/PaddlePaddle/PaddleSlim \n"
            "2) cd PaddleSlim \n"
            "3) python ./csrc/setup_cuda.py install"
        )
    weight_name = name + ".weight"
    quant_weight_name = name + ".quant_weight"
    quant_name_list = [quant_weight_name]
    if not quantization_config.qlora_weight_double_quant:
        weight_scale_name = name + ".weight_scale"
        quant_name_list += [weight_scale_name]
    else:
        qweight_scale_name = name + ".qweight_scale"
        double_weight_scale_name = name + ".double_weight_scale"
        quant_sacle_offset_name = name + ".quant_sacle_offset"
        quant_name_list += [qweight_scale_name, double_weight_scale_name, quant_sacle_offset_name]

    if all(quant_name in state_dict for quant_name in quant_name_list):
        return state_dict
    elif weight_name in state_dict:
        target_weight = state_dict.pop(weight_name).cast(dtype).cuda()
        qlora_state_dict = qlora_weight_quantize(
            weight=target_weight,
            quant_algo=weight_quantize_algo,
            double_quant=quantization_config.qlora_weight_double_quant,
            block_size=quantization_config.qlora_weight_blocksize,
            double_quant_block_size=quantization_config.qlora_weight_double_quant_block_size,
            linear_name=name,
            return_dict=True,
        )
        state_dict.update(qlora_state_dict)
        del target_weight

    return state_dict


def convert_to_quantize_state_dict(state_dict, quantization_linear_list, quantization_config, dtype):
    for name in quantization_linear_list:
        # Get quantization algorithm
        weight_quantize_algo = parse_weight_quantize_algo(quantization_config, name)
        if weight_quantize_algo is None:
            continue
        # Convert state dict
        if weight_quantize_algo in [
            "weight_only_int8",
            "weight_only_int4",
            "llm.int8",
            "a8w8linear",
            "a8w4linear",
            "fp8linear",
        ]:
            convert_to_weight_quantize_state_dict(state_dict, name, quantization_config, dtype, weight_quantize_algo)
        elif weight_quantize_algo in ["fp4", "nf4"]:
            convert_to_qlora_state_dict(state_dict, name, quantization_config, dtype, weight_quantize_algo)
        else:
            raise NotImplementedError(
                f"Please check the quantization_config.weight_quantize_algo: {quantization_config.weight_quantize_algo}"
            )
    return state_dict


def update_loaded_state_dict_keys(state_dict, quantization_linear_list, quantization_config, ignore_warning=False):
    for name in quantization_linear_list:
        weight_name = name + ".weight"
        quant_weight_name = name + ".quant_weight"
        weight_scale_name = name + ".weight_scale"
        activation_scale_name = name + ".activation_scale"
        qweight_scale_name = name + ".qweight_scale"
        double_weight_scale_name = name + ".double_weight_scale"
        quant_sacle_offset_name = name + ".quant_sacle_offset"

        if quant_weight_name in state_dict and weight_scale_name in state_dict:
            continue
        elif weight_name in state_dict:
            state_dict.remove(weight_name)
            state_dict.append(quant_weight_name)
            if quantization_config.qlora_weight_double_quant:
                state_dict.append(qweight_scale_name)
                state_dict.append(double_weight_scale_name)
                state_dict.append(quant_sacle_offset_name)
            else:
                state_dict.append(weight_scale_name)
                weight_quantize_algo = parse_weight_quantize_algo(quantization_config, name)
                if weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                    state_dict.append(activation_scale_name)

        else:
            if not ignore_warning:
                logger.warning(
                    f"Cannot find {weight_name} in state_dict or {quant_weight_name}  and {weight_scale_name} in state_dict"
                )

    return state_dict
