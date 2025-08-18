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
from paddleslim.lc.quantizers.quant_func import dequantize_8bit, quantize_8bit
from paddleslim_ops import dequant_blockwise, quant_blockwise


def qlora_weight_quantize(
    weight,
    quant_algo="nf4",
    double_quant=False,
    block_size=64,
    double_quant_block_size=256,
    linear_name=None,
    return_dict=True,
):
    quant_weight, weight_scale = quant_blockwise(weight, None, blocksize=block_size, quant_type=quant_algo)
    if double_quant:
        quant_sacle_offset = weight_scale.mean()
        weight_scale -= quant_sacle_offset
        qweight_scale, double_weight_scale = quantize_8bit(
            weight_scale, None, double_quant_block_size, quant_type="dynamic_fp8"
        )
        if not return_dict:
            return quant_weight, (qweight_scale, double_weight_scale, quant_sacle_offset)
        qweight_scale_name = f"{linear_name}.qweight_scale" if linear_name else "qweight_scale"
        double_weight_scale_name = f"{linear_name}.double_weight_scale" if linear_name else "double_weight_scale"
        quant_sacle_offset_name = f"{linear_name}.quant_sacle_offset" if linear_name else "quant_sacle_offset"
        qlora_state_dict = {
            qweight_scale_name: qweight_scale,
            double_weight_scale_name: double_weight_scale,
            quant_sacle_offset_name: quant_sacle_offset,
        }
    else:
        weight_scale_name = f"{linear_name}.weight_scale" if linear_name else "weight_scale"
        qlora_state_dict = {weight_scale_name: weight_scale}
        if not return_dict:
            return quant_weight, (weight_scale)
    quant_weight_name = f"{linear_name}.quant_weight" if linear_name else "quant_weight"
    qlora_state_dict[quant_weight_name] = quant_weight
    return qlora_state_dict


def qlora_weight_dequantize(
    quant_weight, quant_algo, state, double_quant=False, block_size=64, double_quant_block_size=256
):
    if double_quant:
        qweight_scale, double_weight_scale, quant_sacle_offset = state
        weight_scale = dequantize_8bit(
            qweight_scale, None, double_weight_scale, double_quant_block_size, quant_type="dynamic_fp8"
        )
        weight_scale += quant_sacle_offset
    else:
        weight_scale = state
    out = dequant_blockwise(quant_weight, None, weight_scale, blocksize=block_size, quant_type=quant_algo)
    return out


def qlora_weight_quantize_dequantize(
    weight, quant_algo="nf4", double_quant=False, block_size=64, double_quant_block_size=256
):
    dtype = weight.dtype
    quant_weight, state = qlora_weight_quantize(
        weight=weight,
        quant_algo=quant_algo,
        double_quant=double_quant,
        block_size=block_size,
        double_quant_block_size=double_quant_block_size,
        return_dict=False,
    )
    quant_dequant_weight = (
        qlora_weight_dequantize(
            quant_weight=quant_weight,
            quant_algo=quant_algo,
            state=state,
            double_quant=double_quant,
            block_size=block_size,
            double_quant_block_size=double_quant_block_size,
        )
        .reshape(weight.shape)
        .cast(dtype)
    )
    return quant_dequant_weight


def qlora_weight_linear(
    x,
    quant_weight,
    dtype,
    state,
    quant_algo="nf4",
    double_quant=False,
    block_size=64,
    double_quant_block_size=256,
    bias=None,
):
    weight = (
        qlora_weight_dequantize(quant_weight, quant_algo, state, double_quant, block_size, double_quant_block_size)
        .cast(dtype)
        .reshape([x.shape[-1], -1])
    )
    out = paddle.nn.functional.linear(x, weight, bias)
    return out
