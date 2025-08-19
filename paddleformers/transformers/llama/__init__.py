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

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["LLAMA_PRETRAINED_INIT_CONFIGURATION", "LlamaConfig", "LLAMA_PRETRAINED_RESOURCE_FILES_MAP"],
    "modeling": [
        "LlamaForCausalLM",
        "LlamaAttention",
        "_make_causal_mask",
        "LlamaLinearScalingRotaryEmbedding",
        "assign_kv_heads",
        "repeat_kv",
        "LlamaMLP",
        "get_use_casual_mask",
        "LlamaDynamicNTKScalingRotaryEmbedding",
        "Llama3RotaryEmbedding",
        "LlamaDecoderLayer",
        "scaled_dot_product_attention",
        "LlamaLMHead",
        "LlamaRMSNorm",
        "LlamaRotaryEmbedding",
        "build_alibi_tensor",
        "apply_rotary_pos_emb",
        "LlamaPretrainedModel",
        "ConcatMaskedLoss",
        "LlamaModel",
        "parallel_matmul",
        "get_triangle_upper_mask",
        "_expand_2d_mask",
        "is_casual_mask",
        "_get_interleave",
        "masked_fill",
        "rotate_half",
        "LlamaPretrainingCriterion",
        "LlamaNTKScalingRotaryEmbedding",
    ],
    "modeling_auto": [
        "enable_fuse_ffn_qkv_pass",
        "LlamaDecoderLayerAuto",
        "LlamaAttentionAuto",
        "LlamaPretrainedModelAuto",
        "LlamaLMHeadAuto",
        "LlamaModelAuto",
        "LlamaForCausalLM3DAuto",
        "LlamaMLPAuto",
        "get_mesh",
        "LlamaRMSNormAuto",
        "is_pp_enable",
        "LlamaPretrainingCriterion3DAuto",
        "global_mesh_starts_with_pp",
        "scaled_dot_product_attention",
    ],
    "modeling_network": [
        "LlamaPretrainedModelNet",
        "layer_input_parallel_row_and_col_hook",
        "LlamaModelNet",
        "LlamaPretrainingCriterionNet",
        "layer_input_replicate_hook",
        "LlamaLMHeadNet",
        "LlamaForCausalLMNetDPO",
        "GlobalOutputNet",
        "layer_input_parallel_row_hook",
        "LlamaRMSNormNet",
        "LlamaAttentionNet",
        "scaled_dot_product_attention",
        "ReshardLayer",
        "LlamaForCausalLMNet",
        "enable_fuse_ffn_qkv_pass",
        "LlamaMLPNet",
        "LlamaDecoderLayerNet",
    ],
    "modeling_pp": ["LlamaForCausalLMPipe"],
    "tokenizer": ["LlamaTokenizer", "Llama3Tokenizer"],
    "tokenizer_fast": ["LlamaTokenizerFast"],
    "fusion_ops": [],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
    from .modeling_auto import *
    from .modeling_network import *
    from .modeling_pp import *
    from .tokenizer import *
    from .tokenizer_fast import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
