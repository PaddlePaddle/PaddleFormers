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
from contextlib import suppress
from typing import TYPE_CHECKING
from ..utils.lazy_import import _LazyModule

# from .auto.modeling import AutoModelForCausalLM
import_structure = {
    "kto_criterion": [
        "sequence_parallel_sparse_mask_labels",
        "fused_head_and_loss_fn",
        "parallel_matmul",
        "KTOCriterion",
    ],
    "model_outputs": ["CausalLMOutputWithPast"],
    "sequence_parallel_utils": ["AllGatherVarlenOp", "sequence_parallel_sparse_mask_labels"],
    "model_utils": ["PretrainedModel", "register_base_model"],
    "tokenizer_utils": [
        "PreTrainedTokenizer",
        "PretrainedTokenizer",
        "BPETokenizer",
        "tokenize_chinese_chars",
        "is_chinese_char",
        "AddedToken",
        "normalize_chars",
        "tokenize_special_chars",
        "convert_to_unicode",
    ],
    "attention_utils": ["create_bigbird_rand_mask_idx_list"],
    "tensor_parallel_utils": [],
    "configuration_utils": ["PretrainedConfig"],
    "tokenizer_utils_fast": ["PretrainedTokenizerFast"],
    "processing_utils": ["ProcessorMixin"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "image_processing_utils": ["ImageProcessingMixin"],
    "moe_gate": ["PretrainedMoEGate", "MoEGateMixin"],
    "token_dispatcher": [],
    "moe_layer": ["combining", "_AllToAll", "MoELayer", "dispatching", "MoEFlexTokenLayer"],
    "bert.modeling": [
        "BertForSequenceClassification",
        "BertPretrainingHeads",
        "BertForMaskedLM",
        "BertForPretraining",
        "BertPretrainedModel",
        "BertForTokenClassification",
        "BertForMultipleChoice",
        "BertModel",
        "BertPretrainingCriterion",
        "BertForQuestionAnswering",
    ],
    "bert.tokenizer": ["BertTokenizer"],
    "bert.tokenizer_fast": ["BertTokenizerFast"],
    "bert.configuration": ["BERT_PRETRAINED_INIT_CONFIGURATION", "BertConfig", "BERT_PRETRAINED_RESOURCE_FILES_MAP"],
    "auto.configuration": ["AutoConfig"],
    "auto.image_processing": ["AutoImageProcessor"],
    "auto.modeling": [
        "AutoTokenizer",
        "AutoBackbone",
        "AutoModel",
        "AutoModelForPretraining",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
        "AutoModelForQuestionAnswering",
        "AutoModelForMultipleChoice",
        "AutoModelForMaskedLM",
        "AutoModelForCausalLMPipe",
        "AutoEncoder",
        "AutoDecoder",
        "AutoGenerator",
        "AutoDiscriminator",
        "AutoModelForConditionalGeneration",
    ],
    "tokenizer_utils_base": [
        "PaddingStrategy",
        "TextInput",
        "TensorType",
    ],
    "auto.processing": ["AutoProcessor"],
    "auto.tokenizer": ["AutoTokenizer"],
    "deepseek_v2.configuration": ["DeepseekV2Config"],
    "deepseek_v2.modeling": [
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
        "DeepseekV2RotaryEmbedding",
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
        "yarn_find_correction_dim",
        "yarn_linear_ramp_mask",
        "DeepseekV2DynamicNTKScalingRotaryEmbedding",
        "DeepseekV2MLP",
        "yarn_get_mscale",
        "DeepseekV2LMHead",
        "DeepseekV2DecoderLayer",
        "DeepseekV2PretrainingCriterion",
        "yarn_find_correction_range",
        "get_triangle_upper_mask",
        "DeepseekV2LinearScalingRotaryEmbedding",
    ],
    "deepseek_v2.modeling_auto": [
        "DeepseekV2LMHeadAuto",
        "DeepseekV2ForCausalLMAuto",
        "DeepseekV2ModelAuto",
        "DeepseekV2PretrainedModelAuto",
    ],
    "deepseek_v2.modeling_pp": ["DeepseekV2ForCausalLMPipe"],
    "deepseek_v2.mfu_utils": ["DeepSeekProjection"],
    "deepseek_v2.kernel": [
        "act_quant",
        "weight_dequant",
        "fp8_gemm",
        "weight_dequant_kernel",
        "act_quant_kernel",
        "fp8_gemm_kernel",
    ],
    "deepseek_v2.tokenizer_fast": ["DeepseekTokenizerFast"],
    "deepseek_v2.fp8_linear": [
        "Linear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "ColumnSequenceParallelLinear",
        "RowSequenceParallelLinear",
    ],
    "deepseek_v3.configuration": ["DeepseekV3Config"],
    "deepseek_v3.modeling": [
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForSequenceClassification",
        "DeepseekV3Model",
        "DeepseekV3PretrainedModel",
    ],
    "deepseek_v3.modeling_auto": [
        "DeepseekV3LMHeadAuto",
        "DeepseekV3ForCausalLMAuto",
        "DeepseekV3ModelAuto",
        "DeepseekV3PretrainedModelAuto",
    ],
    "deepseek_v3.modeling_pp": ["DeepseekV3ForCausalLMPipe"],
    "ernie4_5.configuration": ["Ernie4_5Config"],
    "ernie4_5.modeling": ["Ernie4_5Model", "Ernie4_5ForCausalLM"],
    "ernie4_5.tokenizer": ["Ernie4_5Tokenizer"],
    "export": ["export_model"],
    "llama.configuration": [
        "LLAMA_PRETRAINED_INIT_CONFIGURATION",
        "LlamaConfig",
        "LLAMA_PRETRAINED_RESOURCE_FILES_MAP",
    ],
    "llama.modeling": [
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
    "llama.modeling_auto": [
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
    "llama.modeling_network": [
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
    "llama.modeling_pp": ["LlamaForCausalLMPipe"],
    "llama.tokenizer": ["LlamaTokenizer", "Llama3Tokenizer"],
    "llama.tokenizer_fast": ["LlamaTokenizerFast"],
    "optimization": [
        "LinearDecayWithWarmup",
        "ConstScheduleWithWarmup",
        "CosineDecayWithWarmup",
        "PolyDecayWithWarmup",
        "CosineAnnealingWithWarmupDecay",
        "LinearAnnealingWithWarmupDecay",
    ],
    "qwen.configuration": ["QWenConfig"],
    "qwen.modeling": [
        "QWenBlock",
        "QWenForCausalLM",
        "QWenLMHeadModel",
        "QWenPretrainedModel",
        "QWenModel",
        "QWenLMHead",
        "QWenPretrainingCriterion",
    ],
    "qwen.modeling_auto": [
        "QWenBlockAuto",
        "QWenForCausalLM3DAuto",
        "QWenPretrainedModelAuto",
        "QWenModelAuto",
        "QWenLMHeadAuto",
        "QWenPretrainingCriterionAuto",
    ],
    "qwen.modeling_network": [
        "QWenBlockNet",
        "QWenForCausalLMNet",
        "QWenPretrainedModelNet",
        "QWenModelNet",
        "QWenLMHeadNet",
        "QWenPretrainingCriterionNet",
    ],
    "qwen.modeling_pp": ["QWenForCausalLMPipe"],
    "qwen.tokenizer": ["QWenTokenizer"],
    "qwen2.configuration": ["Qwen2Config"],
    "qwen2.modeling": [
        "Qwen2Model",
        "Qwen2PretrainedModel",
        "Qwen2ForCausalLM",
        "Qwen2PretrainingCriterion",
        "Qwen2ForSequenceClassification",
        "Qwen2ForTokenClassification",
        "Qwen2SentenceEmbedding",
    ],
    "qwen2.modeling_pp": ["Qwen2ForCausalLMPipe"],
    "qwen2.tokenizer": ["Qwen2Tokenizer"],
    "qwen2.tokenizer_fast": ["Qwen2TokenizerFast"],
    "qwen2_moe.configuration": ["Qwen2MoeConfig"],
    "qwen2_moe.modeling": [
        "Qwen2MoeModel",
        "Qwen2MoePretrainedModel",
        "Qwen2MoeForCausalLM",
        "Qwen2MoePretrainingCriterion",
    ],
    "qwen2_moe.modeling_pp": ["Qwen2MoeForCausalLMPipe"],
    "qwen3.configuration": ["Qwen3Config"],
    "qwen3.modeling": [
        "Qwen3Model",
        "Qwen3PretrainedModel",
        "Qwen3ForCausalLM",
        "Qwen3PretrainingCriterion",
        "Qwen3ForSequenceClassification",
        "Qwen3ForTokenClassification",
        "Qwen3SentenceEmbedding",
    ],
    "qwen3.modeling_pp": ["Qwen3ForCausalLMPipe"],
    "qwen3_moe.configuration": ["Qwen3MoeConfig"],
    "qwen3_moe.modeling": [
        "Qwen3MoeModel",
        "Qwen3MoePretrainedModel",
        "Qwen3MoeForCausalLM",
        "Qwen3MoePretrainingCriterion",
    ],
    "qwen3_moe.modeling_pp": ["Qwen3MoeForCausalLMPipe"],
    "ernie4_5vl.tokenizer": ["Ernie4_5_VLTokenizer"],
    "ernie4_5vl": [],
    "bert": [],
    "llama": [],
    "qwen2": [],
    "qwen3": [],
    "qwen": [],
    "deepseek_v2": [],
    "deepseek_v3": [],
    "ernie4_5": [],
    "qwen2_moe": [],
    "qwen3_moe": [],
    "auto": ["AutoModelForCausalLM"],
}

if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .model_utils import PretrainedModel, register_base_model
    from .tokenizer_utils import (
        PretrainedTokenizer,
        BPETokenizer,
        tokenize_chinese_chars,
        is_chinese_char,
        AddedToken,
        normalize_chars,
        tokenize_special_chars,
        convert_to_unicode,
    )
    from .tokenizer_utils_fast import PretrainedTokenizerFast
    from .processing_utils import ProcessorMixin
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
    from .image_processing_utils import ImageProcessingMixin
    from .attention_utils import create_bigbird_rand_mask_idx_list
    from .sequence_parallel_utils import AllGatherVarlenOp, sequence_parallel_sparse_mask_labels
    from .tensor_parallel_utils import parallel_matmul, fused_head_and_loss_fn
    from .moe_gate import *
    from .moe_layer import *
    from .export import export_model

    with suppress(Exception):
        from paddle.distributed.fleet.utils.sequence_parallel_utils import (
            GatherOp,
            ScatterOp,
            AllGatherOp,
            ReduceScatterOp,
            ColumnSequenceParallelLinear,
            RowSequenceParallelLinear,
            mark_as_sequence_parallel_parameter,
            register_sequence_parallel_allreduce_hooks,
        )

    # isort: split
    from .bert.modeling import *
    from .bert.tokenizer import *
    from .bert.configuration import *

    # isort: split
    from .auto.configuration import *
    from .auto.image_processing import *
    from .auto.modeling import *
    from .auto.processing import *
    from .auto.tokenizer import *
    from .deepseek_v2 import *
    from .deepseek_v3 import *
    from .ernie4_5 import *
    from .llama import *
    from .optimization import *
    from .qwen import *
    from .qwen2 import *
    from .qwen2_moe import *
    from .qwen3 import *
    from .qwen3_moe import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
