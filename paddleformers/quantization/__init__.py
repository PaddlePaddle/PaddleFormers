# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "checkpoint_quantization_utils": [
        "cal_ratio",
        "group_wise_quant_dequant",
        "merge_int4",
        "split_int8",
        "cal_abs_min_max_channel",
        "asymmetry_qdq_weight",
        "cal_abs_max_channel",
        "qdq_weight",
    ],
    "hadamard_utils": ["matmul_hadU", "create_hadamard_matrix", "hadamard_matmul", "apply_hadamard_matmul"],
    "qat_utils": [
        "QMIN_QMAX_MAPPING",
        "quantize",
        "dequantize",
        "int8_forward",
        "int8_backward",
        "fp8_forward",
        "fp8_backward",
        "QATFunc",
    ],
    "qlora": [
        "qlora_weight_quantize",
        "qlora_weight_dequantize",
        "qlora_weight_quantize_dequantize",
        "qlora_weight_linear",
    ],
    "quantization_config": ["quant_inference_mapping", "fp8_format_mapping", "QuantizationConfig"],
    "quantization_linear": [
        "QuantMapping",
        "quant_weight_forward",
        "dequant_weight",
        "QuantizationLinearFunc",
        "quant_weight_linear",
        "get_activation_scale_group",
        "QuantizationLinear",
        "ColumnParallelQuantizationLinear",
        "RowParallelQuantizationLinear",
    ],
    "quantization_utils": [
        "parse_weight_quantize_algo",
        "replace_with_quantization_linear",
        "convert_to_weight_quantize_state_dict",
        "convert_to_qlora_state_dict",
        "convert_to_quantize_state_dict",
        "update_loaded_state_dict_keys",
    ],
    "unified_checkpoint_quantization": ["dequant_unified_optimizer", "quant_unified_optimizer"],
}

if TYPE_CHECKING:
    from .quantization_config import QuantizationConfig
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
