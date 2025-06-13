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
from paddle.distributed import fleet

from ..utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    SYMMETRY_QUANT_SCALE,
)
from ..utils.log import logger
from .checkpoint_quantization_utils import (
    asymmetry_qdq_weight,
    cal_ratio,
    group_wise_quant_dequant,
    merge_int4,
    qdq_weight,
    split_int8,
)


def dequant_unified_optimizer(state_dict, ckpt_quant_stage, scale_dict, use_pd=False):
    """
    dequantize unified optimizer state dict.
    Args:
        state_dict (`dict`):
            unified checkpoint optimizer state dict.
        ckpt_quant_stage (`str`):
            checkpoint quantization stage, chosen in ["O0", "O1", "O2"].
        scale_dict (`int`):
            compression checkpoint scale dict.
    """
    logger.info(f"Start unified checkpoint dequantization, stage {ckpt_quant_stage}.")
    tp_rank, tp_degree = -1, 1
    if paddle.distributed.get_world_size() > 1:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        tp_rank, tp_degree = tp_group.rank, tp_group.nranks

    if ckpt_quant_stage == "O1":
        # set eps
        eps = 1e-8
        for quant_key in state_dict.keys():
            is_moment1 = MOMENT1_KEYNAME in quant_key
            is_moment2 = MOMENT2_KEYNAME in quant_key
            if is_moment1:
                # dequant m1
                scale_key = quant_key + SYMMETRY_QUANT_SCALE
                weight = state_dict[quant_key]
                scales = scale_dict[scale_key]
                weight, _ = qdq_weight(
                    weight,
                    scales=scales,
                    quant_bit=8,
                    dequant=True,
                    tp_rank=tp_rank,
                    tp_degree=tp_degree,
                    use_pd=use_pd,
                )
                state_dict[quant_key] = weight
            elif is_moment2:
                # dequant ratio
                weight = state_dict[quant_key]
                min_scale_key = quant_key + ASYMMETRY_QUANT_SCALE_MIN
                max_scale_key = quant_key + ASYMMETRY_QUANT_SCALE_MAX
                mins, maxs = scale_dict[min_scale_key], scale_dict[max_scale_key]
                weight, _ = asymmetry_qdq_weight(
                    weight,
                    mins=mins,
                    maxs=maxs,
                    quant_bit=8,
                    dequant=True,
                    tp_rank=tp_rank,
                    tp_degree=tp_degree,
                    use_pd=use_pd,
                )
                # cal m2
                if use_pd:
                    weight = paddle.square(1.0 / weight - eps)
                else:
                    weight = np.square(1.0 / weight - eps)
                state_dict[quant_key] = weight
    elif ckpt_quant_stage == "O2":
        # set eps
        eps = 1e-8
        m1_state_dict = {}
        for quant_key in state_dict.keys():
            # not all optimizer weights in O2 stage were quantized to int8,
            # the norm-like weights were still remain in float32.
            if state_dict[quant_key].dtype != paddle.int8:
                logger.info(f"{quant_key} skip.")
                continue
            # split int8
            weight = state_dict[quant_key]
            m1_quant, ratio_quant = split_int8(weight.numpy())
            # dequant ratio
            ratio_min_scale_key = quant_key + ASYMMETRY_QUANT_SCALE_MIN
            ratio_max_scale_key = quant_key + ASYMMETRY_QUANT_SCALE_MAX
            m1_scale_key = quant_key[: -len(MOMENT2_KEYNAME)] + MOMENT1_KEYNAME + SYMMETRY_QUANT_SCALE
            m1_scales = scale_dict[m1_scale_key]
            ratio_mins, ratio_maxs = scale_dict[ratio_min_scale_key], scale_dict[ratio_max_scale_key]
            m1_weight = group_wise_quant_dequant(
                m1_quant,
                mins=m1_scales,
                maxs=None,
                quant_bits=4,
                quant=False,
                tp_rank=tp_rank,
                tp_degree=tp_degree,
                use_pd=use_pd,
                symmetry=True,
            )
            ratio_weight = group_wise_quant_dequant(
                ratio_quant,
                mins=ratio_mins,
                maxs=ratio_maxs,
                quant_bits=4,
                quant=False,
                tp_rank=tp_rank,
                tp_degree=tp_degree,
                use_pd=use_pd,
            )

            if use_pd:
                ratio_weight = paddle.square(1.0 / ratio_weight - eps)
            else:
                ratio_weight = np.square(1.0 / ratio_weight - eps)
            state_dict[quant_key] = ratio_weight
            m1_state_dict[quant_key[: -len(MOMENT2_KEYNAME)] + MOMENT1_KEYNAME] = m1_weight
            state_dict.update(m1_state_dict)

    logger.info(f"Unified checkpoint dequantization done, stage {ckpt_quant_stage}.")

    return state_dict


def quant_unified_optimizer(state_dict, state_dict_type, ckpt_quant_stage, async_save=False):
    """
    quantize unified optimizer state dict.
    Args:
        state_dict (`dict`):
            unified checkpoint optimizer state dict.
        state_dict_type (`str`):
            state_dict type, chosen in ["model_weight", "master_weight", "optimizer_weight"].
        ckpt_quant_stage (`str`):
            checkpoint quantization stage, chosen in ["O0", "O1", "O2"].
        async_save (`bool`):
            whether use async_save.
    """
    logger.info(f"Start unified checkpoint quantization, stage {ckpt_quant_stage}.")

    quant = False
    if ckpt_quant_stage != "O0":
        quant = True
    del_key = []
    if quant and state_dict_type == "optimizer_weight":
        scales_dict = {}
        for k in state_dict.keys():
            momentum1 = k.endswith(MOMENT1_KEYNAME)
            momentum2 = k.endswith(MOMENT2_KEYNAME)

            quant_weight = None

            if ckpt_quant_stage == "O1":
                # m1: wint8, 1/(sqrt(m2)+eps): wint8
                if momentum2:
                    # m1: m1_quant_weight, m2: ratio
                    m1_key = k.split("/")[0] + "/" + MOMENT1_KEYNAME
                    ratio = cal_ratio(state_dict[m1_key], state_dict[k])
                    m1_quant, scales = qdq_weight(state_dict[m1_key], quant_bit=8)
                    quant_weight, mins, maxs = asymmetry_qdq_weight(ratio, quant_bit=8)
                    state_dict[m1_key] = m1_quant
                    scales_dict[m1_key + SYMMETRY_QUANT_SCALE] = scales
                    scales_dict[k + ASYMMETRY_QUANT_SCALE_MIN] = mins
                    scales_dict[k + ASYMMETRY_QUANT_SCALE_MAX] = maxs
                elif not momentum1:
                    quant_weight = state_dict[k]
            elif ckpt_quant_stage == "O2":
                # m1: bw-wint4, 1/(sqrt(m2)+eps): bw-wint4
                if momentum2:
                    # skip norm-like parameters
                    if len(state_dict[k].shape) < 2:
                        continue
                    # m1: m1_quant_weight, m2: ratio
                    m1_key = k.split("/")[0] + "/" + MOMENT1_KEYNAME
                    ratio = cal_ratio(state_dict[m1_key], state_dict[k])
                    m1_quant, m1_scales = group_wise_quant_dequant(state_dict[m1_key], quant_bits=4, symmetry=True)
                    quant_weight, r_mins, r_maxs = group_wise_quant_dequant(ratio, quant_bits=4)
                    quant_weight = merge_int4(m1_quant, quant_weight)
                    scales_dict[m1_key + SYMMETRY_QUANT_SCALE] = m1_scales
                    scales_dict[k + ASYMMETRY_QUANT_SCALE_MIN] = r_mins
                    scales_dict[k + ASYMMETRY_QUANT_SCALE_MAX] = r_maxs
                    del_key.append(m1_key)
                elif not momentum1:
                    quant_weight = state_dict[k]

            if quant_weight is not None:
                state_dict[k] = quant_weight

        for k in del_key:
            state_dict.pop(k, None)

        state_dict.update(scales_dict)
    logger.info(f"Unified checkpoint quantization done, stage {ckpt_quant_stage}.")

    return state_dict
