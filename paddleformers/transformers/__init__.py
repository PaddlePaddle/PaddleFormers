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

import logging
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
        "normalize_chars",
        "tokenize_special_chars",
        "convert_to_unicode",
        "AddedToken",
    ],
    "attention_utils": ["create_bigbird_rand_mask_idx_list"],
    "tensor_parallel_utils": [],
    "configuration_utils": ["PretrainedConfig"],
    "tokenizer_utils_fast": ["PretrainedTokenizerFast"],
    "processing_utils": ["ProcessorMixin"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "image_processing_utils": ["PaddleImageProcessingMixin", "ImageProcessingMixin", "BaseImageProcessor"],
    "video_processing_utils": ["BaseVideoProcessor"],
    "moe_gate": ["PretrainedMoEGate", "MoEGateMixin"],
    "token_dispatcher": ["_DispatchManager"],
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
    "auto.image_processing": ["AutoImageProcessor", "IMAGE_PROCESSOR_MAPPING"],
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
        "PreTokenizedInput",
        "TextInput",
        "TensorType",
        "TruncationStrategy",
    ],
    "auto.processing": ["AutoProcessor"],
    "auto.tokenizer": ["AutoTokenizer", "TOKENIZER_MAPPING"],
    "auto.video_processing": ["AutoVideoProcessor", "VIDEO_PROCESSOR_MAPPING"],
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
    "deepseek_v2.modeling_auto": [
        "DeepseekV2LMHeadAuto",
        "DeepseekV2ForCausalLMAuto",
        "DeepseekV2ModelAuto",
        "DeepseekV2PretrainedModelAuto",
    ],
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
        "DeepseekV3ForCausalLMPipe",
    ],
    "deepseek_v3.modeling_auto": [
        "DeepseekV3LMHeadAuto",
        "DeepseekV3ForCausalLMAuto",
        "DeepseekV3ModelAuto",
        "DeepseekV3PretrainedModelAuto",
    ],
    "ernie4_5.configuration": ["Ernie4_5Config"],
    "ernie4_5.modeling": ["Ernie4_5Model", "Ernie4_5ForCausalLM", "Ernie4_5ForCausalLMPipe"],
    "ernie4_5.tokenizer": ["Ernie4_5Tokenizer"],
    "ernie4_5_moe.configuration": ["Ernie4_5_MoeConfig"],
    "ernie4_5_moe.modeling": ["Ernie4_5_MoeModel", "Ernie4_5_MoeForCausalLM", "Ernie4_5_MoeForCausalLMPipe"],
    "ernie4_5_moe_vl.configuration": ["Ernie4_5_VLConfig"],
    "ernie4_5_moe_vl.modeling": [
        "Ernie4_5_VLMoeForConditionalGenerationModel",
        "Ernie4_5_VLMoeForConditionalGenerationPipe",
    ],
    "ernie4_5_moe_vl.tokenizer": ["Ernie4_5_VLTokenizer"],
    "export": ["export_model"],
    "gpt_oss.configuration": ["GptOssConfig"],
    "gpt_oss.modeling": ["GptOssModel", "GptOssForCausalLM", "GptOssForCausalLMPipe"],
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
        "Qwen2ForCausalLMPipe",
        "Qwen2PretrainingCriterion",
        "Qwen2ForSequenceClassification",
        "Qwen2ForTokenClassification",
        "Qwen2SentenceEmbedding",
    ],
    "qwen2.tokenizer": ["Qwen2Tokenizer"],
    "qwen2.tokenizer_fast": ["Qwen2TokenizerFast"],
    "qwen2_5_vl.processor": ["Qwen2_5_VLProcessor"],
    "qwen2_moe.configuration": ["Qwen2MoeConfig"],
    "qwen2_moe.modeling": [
        "Qwen2MoeModel",
        "Qwen2MoePretrainedModel",
        "Qwen2MoeForCausalLM",
        "Qwen2MoeForCausalLMPipe",
        "Qwen2MoePretrainingCriterion",
    ],
    "qwen2_vl.image_processor": ["Qwen2VLImageProcessor"],
    "qwen2_vl.processor": ["Qwen2VLProcessor"],
    "qwen2_vl.video_processor": ["Qwen2VLVideoProcessor"],
    "qwen2_vl.vision_process": ["process_vision_info"],
    "qwen3.configuration": ["Qwen3Config"],
    "qwen3.modeling": [
        "Qwen3Model",
        "Qwen3PretrainedModel",
        "Qwen3ForCausalLM",
        "Qwen3ForCausalLMPipe",
        "Qwen3PretrainingCriterion",
        "Qwen3ForSequenceClassification",
        "Qwen3ForTokenClassification",
        "Qwen3SentenceEmbedding",
    ],
    "qwen3_moe.configuration": ["Qwen3MoeConfig"],
    "qwen3_moe.modeling": [
        "Qwen3MoeModel",
        "Qwen3MoePretrainedModel",
        "Qwen3MoeForCausalLM",
        "Qwen3MoeForCausalLMPipe",
        "Qwen3MoePretrainingCriterion",
    ],
    "bert": [],
    "llama": [],
    "qwen2": [],
    "qwen3": [],
    "qwen": [],
    "deepseek_v2": [],
    "deepseek_v3": [],
    "ernie4_5": ["Ernie4_5DecoderLayer", "Ernie4_5Model", "Ernie4_5_ForCausalLM"],
    "ernie4_5_moe": ["Ernie4_5_MoeDecoderLayer", "Ernie4_5_MoeModel", "Ernie4_5_MoeForCausalLM"],
    "ernie4_5_moe_vl": [],
    "qwen2_moe": [],
    "qwen2_vl": [],
    "qwen3_moe": [],
    "glm4_moe.configuration": ["Glm4MoeConfig"],
    "glm4_moe": ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM"],
    "auto": ["AutoModelForCausalLM"],
    "legacy.tokenizer_utils_base": ["EncodingFast"],
    "legacy": [],
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
    from .image_processing_utils import PaddleImageProcessingMixin, ImageProcessingMixin, BaseImageProcessor
    from .video_processing_utils import BaseVideoProcessor
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
    from .auto.video_processing import *
    from .deepseek_v2 import *
    from .deepseek_v3 import *
    from .ernie4_5 import *
    from .ernie4_5_moe import *
    from .ernie4_5_moe_vl import *
    from .llama import *
    from .optimization import *
    from .qwen import *
    from .qwen2 import *
    from .qwen2_5_vl import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .qwen3 import *
    from .qwen3_moe import *
    from .glm4_moe import *
    from .gpt_oss import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )

logging.getLogger("transformers").addFilter(
    lambda record: "None of PyTorch, TensorFlow >= 2.0, or Flax have been found." not in str(record.getMessage())
)
