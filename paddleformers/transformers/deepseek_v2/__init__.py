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


import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["DeepseekV2Config"],
    "modeling": [
        "masked_fill",
        "DeepseekV2Attention",
        "MoEGate",
        "FakeGate",
        "DeepseekV2ForCausalLM",
        "_make_causal_mask",
        "is_casual_mask",
        "DeepseekV2MoE",
        "DeepseekV2MoEFlexToken",
        "scaled_dot_product_attention",
        "rotate_half",
        "DeepseekV2MTPLayer",
        "DeepseekV2RMSNorm",
        "DeepseekV2YarnRotaryEmbedding",
        "parallel_matmul",
        "DeepseekV2PretrainedModel",
        "AddAuxiliaryLoss",
        "apply_rotary_pos_emb",
        "assign_kv_heads",
        "DeepseekV2ForSequenceClassification",
        "_expand_2d_mask",
        "DeepseekV2Model",
        "repeat_kv",
        "DeepseekV2MLP",
        "yarn_get_mscale",
        "DeepseekV2DecoderLayer",
        "get_triangle_upper_mask",
        "DeepseekV2ForCausalLMPipe",
    ],
    "modeling_auto": [
        "DeepseekV2LMHeadAuto",
        "DeepseekV2ForCausalLMAuto",
        "DeepseekV2ModelAuto",
        "DeepseekV2PretrainedModelAuto",
    ],
    "mfu_utils": ["DeepSeekProjection"],
    "kernel": [
        "act_quant",
        "weight_dequant",
        "fp8_gemm",
        "weight_dequant_kernel",
        "act_quant_kernel",
        "fp8_gemm_kernel",
    ],
    "tokenizer_fast": ["DeepseekTokenizerFast"],
    "fp8_linear": [
        "Linear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "ColumnSequenceParallelLinear",
        "RowSequenceParallelLinear",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
    from .modeling_auto import *
    from .modeling_pp import *
    from .tokenizer_fast import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
