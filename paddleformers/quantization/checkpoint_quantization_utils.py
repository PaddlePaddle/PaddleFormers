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


import numpy as np
import paddle


def cal_ratio(m, v, eps=1e-8):
    """
    cal part adam update ratio.
    Args:
        m (`paddle.Tensor`):
            moment in Adam optimizer.
        v (`paddle.Tensor`):
            variance in Adam optimizer.
        eps (`int`):
            epsilon in Adam optimizer.
    """
    return 1 / (np.sqrt(v) + eps)


def group_wise_quant_dequant(
    inputs,
    mins=None,
    maxs=None,
    quant_bits=4,
    group_size=32,
    quant=True,
    tp_rank=-1,
    tp_degree=1,
    use_pd=False,
    symmetry=False,
):
    """
    group-wise quantization (support symmetry, asymmetry).
    Args:
        inputs (`paddle.Tensor`):
            The tensor to quantize.
        mins (`paddle.Tensor`):
            Min scales tensor in asymmetry quantization.
        maxs (`paddle.Tensor`):
            Max scales tensor in asymmetry quantization, or Abs max tensor in symmetry quantization.
        quant_bits (`int`):
            Quantization bits.
        group_size (`int`):
            Group size of group-wise quantization.
        quant (`bool`):
            True when quantization, False in dequantization.
        tp_rank (`int`):
            Tensor parallel rank.
        tp_degree (`int`):
            Tensor parallel world size.
        use_pd (`bool`):
            Whether to use paddle calculation. If False will use numpy.
        symmetry (`bool`):
            Whether to use symmetry quantization.
    """

    qmax = (1 << (quant_bits)) - 1
    qmin = 0
    shape = inputs.shape

    if quant:
        inputs_processed = inputs.reshape([shape[0] // group_size, group_size, shape[1]])
        if symmetry:
            bnt = (1 << (quant_bits - 1)) - 1
            scales = np.max(np.abs(inputs_processed), axis=1)
            new_scales = np.repeat(scales, repeats=group_size, axis=0)
            quant_tensor = np.clip(np.round(inputs / new_scales * bnt), -bnt - 1, bnt)
            return quant_tensor.astype("int8"), scales

        # scales: [shape[0] // group_size, shape[1]]
        maxs = np.max(inputs_processed, axis=1)
        mins = np.min(inputs_processed, axis=1)
        scales = maxs - mins
        # new_scales: [shape[0], shape[1]]
        new_scales = np.repeat(scales, repeats=group_size, axis=0)
        new_mins = np.repeat(mins, repeats=group_size, axis=0)
        # add eps to avoid devide zero
        quant_tensor = np.clip(np.round((inputs - new_mins) / (new_scales) * qmax), qmin, qmax)
        quant_tensor = np.nan_to_num(quant_tensor)
        return quant_tensor.astype("uint8"), mins, maxs
    else:
        if symmetry:
            scales = mins
            bnt = (1 << (quant_bits - 1)) - 1
            if use_pd:
                new_scales = paddle.repeat_interleave(scales, group_size, 0)
            else:
                new_scales = np.repeat(scales, repeats=group_size, axis=0)

            if tp_rank == -1:
                dequant_tensor = inputs.astype("float32") * new_scales / bnt
            elif len(new_scales.shape) == 0 or inputs.shape[-1] == new_scales.shape[-1]:
                # input tensor was row parallel in tp.
                dequant_tensor = (
                    inputs.astype("float32")
                    * new_scales[
                        tp_rank * new_scales.shape[0] // tp_degree : (tp_rank + 1) * new_scales.shape[0] // tp_degree
                    ]
                    / bnt
                )
            else:
                # input tensor was column parallel in tp.
                dequant_tensor = (
                    inputs.astype("float32")
                    * new_scales[
                        :,
                        tp_rank
                        * new_scales.shape[-1]
                        // tp_degree : (tp_rank + 1)
                        * new_scales.shape[-1]
                        // tp_degree,
                    ]
                    / bnt
                )
            return dequant_tensor

        scales = maxs - mins
        if use_pd:
            new_scales = paddle.repeat_interleave(scales, group_size, 0)
            new_mins = paddle.repeat_interleave(mins, group_size, 0)
        else:
            new_scales = np.repeat(scales, repeats=group_size, axis=0)
            new_mins = np.repeat(mins, repeats=group_size, axis=0)

        if tp_rank == -1:
            dequant_tensor = (inputs.astype("float32") / qmax * new_scales) + new_mins
        elif len(new_scales.shape) == 0 or inputs.shape[-1] == new_scales.shape[-1]:
            # input tensor was row parallel in tp.
            dequant_tensor = (
                inputs.astype("float32")
                / qmax
                * new_scales[
                    tp_rank * new_scales.shape[0] // tp_degree : (tp_rank + 1) * new_scales.shape[0] // tp_degree
                ]
            ) + new_mins[tp_rank * new_mins.shape[0] // tp_degree : (tp_rank + 1) * new_mins.shape[0] // tp_degree]
        else:
            # input tensor was column parallel in tp.
            dequant_tensor = (
                inputs.astype("float32")
                / qmax
                * new_scales[
                    :, tp_rank * new_scales.shape[-1] // tp_degree : (tp_rank + 1) * new_scales.shape[-1] // tp_degree
                ]
            ) + new_mins[
                :, tp_rank * new_mins.shape[-1] // tp_degree : (tp_rank + 1) * new_mins.shape[-1] // tp_degree
            ]
        return dequant_tensor


def merge_int4(x, y):
    """
    merge 2 signed int4 to 1 int8
    Args:
        x (`numpy.array`):
            4bits signed int x.
        y (`numpy.array`):
            4bits signed int y.
    """
    int4_high = x << 4
    int4_low = y & 0x0F
    final = int4_high | int4_low
    return final.astype("int8")


def split_int8(final):
    """
    split an int8 to 2 int4 elems
    Args:
        final (`numpy.array`):
            8bits signed int.
    """
    int4_high = final >> 4
    int4_low = final & 0x0F

    int4_high = np.where(int4_high > 8, int4_high - 16, int4_high)

    high_tensor = paddle.Tensor(int4_high)
    low_tensor = paddle.Tensor(int4_low)

    return high_tensor, low_tensor


def cal_abs_min_max_channel(inputs, quant_axis=1):
    """
    channel-wise min max scales calculation
    Args:
        inputs (`numpy.array`):
            input tensor for quantization.
        quant_axis (`int`):
            dimension where calculating inputs' abs min and max scales on.
    """
    eps = 1e-8
    reduce_axis = tuple([i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = np.max(inputs, axis=reduce_axis)
    abs_min_values = np.min(inputs, axis=reduce_axis)
    abs_max_values = np.where(
        abs_max_values == np.array(0, dtype=inputs.dtype), np.array(eps, dtype=inputs.dtype), abs_max_values
    )
    abs_min_values = np.where(
        abs_min_values == np.array(0, dtype=inputs.dtype), np.array(eps, dtype=inputs.dtype), abs_min_values
    )
    return abs_max_values, abs_min_values


def asymmetry_qdq_weight(
    x, quant_bit=8, quant_axis=-1, mins=None, maxs=None, dequant=False, tp_rank=-1, tp_degree=1, use_pd=False
):
    """
    channel-wise asymmetry quantization
    Args:
        x (`paddle.Tensor`):
            The tensor to quantize.
        quant_bits (`int`):
            Quantization bits.
        quant_axis (`int`):
            Scales calculation axis.
        mins (`paddle.Tensor`):
            Min scales tensor in asymmetry quantization.
        maxs (`paddle.Tensor`):
            Max scales tensor in asymmetry quantization.
        dequant (`bool`):
            True when dequantization, False in quantization.
        tp_rank (`int`):
            Model parallel rank.
        tp_degree (`int`):
            Model parallel world size.
        use_pd (`bool`):
            Whether to use paddle calculation. If False will use numpy.
    """

    if mins is None:
        maxs, mins = cal_abs_min_max_channel(x)
    bnt = (1 << (quant_bit)) - 1
    scales = maxs - mins
    if not dequant:
        # quant
        quant_x = np.clip(np.round((x - mins) / scales * bnt), 0, bnt)
        return quant_x.astype(np.uint8), mins, maxs
    else:
        quant_x = x
        # dequant
        if not use_pd:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                # input tensor was row parallel in tp.
                qdq_x = (quant_x / bnt * scales) + mins
            else:
                # input tensor was column parallel in tp.
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[tp_rank * scales.shape[0] // tp_degree : (tp_rank + 1) * scales.shape[0] // tp_degree]
                ) + mins[tp_rank * mins.shape[0] // tp_degree : (tp_rank + 1) * mins.shape[0] // tp_degree]
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                # input tensor was row parallel in tp.
                qdq_x = (quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)) + mins
            else:
                # input tensor was column parallel in tp.
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[tp_rank * scales.shape[0] // tp_degree : (tp_rank + 1) * scales.shape[0] // tp_degree]
                    .unsqueeze(0)
                    .expand(quant_x.shape)
                ) + mins[tp_rank * mins.shape[0] // tp_degree : (tp_rank + 1) * mins.shape[0] // tp_degree]
            return qdq_x.astype(paddle.float32), scales


def cal_abs_max_channel(inputs, quant_axis=1):
    """
    channel-wise abs max calculation
    Args:
        inputs (`numpy.array`):
            input tensor for quantization.
        quant_axis (`int`):
            dimension where calculating inputs' abs max scales on.
    """
    epsilon = 1e-8
    reduce_axis = tuple([i for i in range(len(inputs.shape)) if i != quant_axis])
    abs_max_values = np.max(np.abs(inputs), axis=reduce_axis)
    # maybe all elements are zero in one group,
    # so set the scales from those group to an actual number
    # from divide 0.
    abs_max_values = np.where(
        abs_max_values == np.array(0, dtype=inputs.dtype), np.array(epsilon, dtype=inputs.dtype), abs_max_values
    )
    return abs_max_values


def qdq_weight(x, quant_bit=8, quant_axis=-1, scales=None, dequant=False, tp_rank=-1, tp_degree=1, use_pd=False):
    """
    channel-wise symmetry quantization
    Args:
        x (`paddle.Tensor`):
            The tensor to quantize.
        quant_bits (`int`):
            Quantization bits.
        quant_axis (`int`):
            Scales calculation axis.
        scales (`paddle.Tensor`):
            Abs max scales tensor in symmetry quantization.
        dequant (`bool`):
            True when dequantization, False in quantization.
        tp_rank (`int`):
            Model parallel rank.
        tp_degree (`int`):
            Model parallel world size.
        use_pd (`bool`):
            Whether to use paddle calculation. If False will use numpy.
    """

    if scales is None:
        scales = cal_abs_max_channel(x)
    bnt = (1 << (quant_bit - 1)) - 1
    if not dequant:
        # quant
        quant_x = np.clip(np.round(x / scales * bnt), -bnt - 1, bnt)
        return quant_x.astype(np.int8), scales
    else:
        quant_x = x
        # dequant
        if not use_pd:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                # input tensor was row parallel in tp.
                qdq_x = quant_x / bnt * scales
            else:
                # input tensor was column parallel in tp.
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[tp_rank * scales.shape[0] // tp_degree : (tp_rank + 1) * scales.shape[0] // tp_degree]
                )
            # fp32 , int8, int, fp32 or fp64
            return qdq_x.astype(np.float32), scales
        else:
            if len(scales.shape) == 0 or quant_x.shape[-1] == scales.shape[-1]:
                # input tensor was row parallel in tp.
                qdq_x = quant_x / bnt * scales.unsqueeze(0).expand(quant_x.shape)
            else:
                # input tensor was column parallel in tp.
                qdq_x = (
                    quant_x
                    / bnt
                    * scales[tp_rank * scales.shape[0] // tp_degree : (tp_rank + 1) * scales.shape[0] // tp_degree]
                    .unsqueeze(0)
                    .expand(quant_x.shape)
                )
            # fp32 , int8, int, fp32 or fp64
            return qdq_x.astype(paddle.float32), scales
