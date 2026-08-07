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

import functools

import paddle


def get_quant_func(
    fp8_recipe, input_trans=False, out_scale_trans=False, pow2_scale=False
):
    """
    Get quant function by recipe
    """
    if fp8_recipe == "blockwise":
        inp_quant_func = functools.partial(
            paddle.incubate.nn.functional.fp8_quant_blockwise,
            output_scale_transpose=out_scale_trans,
            quant_method="1x128",
            input_transpose=input_trans,
            using_pow2_scale=pow2_scale,
        )

        weight_quant_func = functools.partial(
            paddle.incubate.nn.functional.fp8_quant_blockwise,
            output_scale_transpose=out_scale_trans,
            quant_method="128x128",
            input_transpose=False,
            using_pow2_scale=pow2_scale,
        )
    else:
        raise ValueError(
            f"fp8_recipe {fp8_recipe} is not supported. Supported recipes are blockwise."
        )

    return inp_quant_func, weight_quant_func
