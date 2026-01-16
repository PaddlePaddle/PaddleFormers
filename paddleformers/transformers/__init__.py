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
    "sequence_parallel_utils": [
        "AllGatherVarlenOp",
        "sequence_parallel_sparse_mask_labels",
    ],
    "model_utils": ["PretrainedModel", "register_base_model"],
    "tokenizer_utils": [
        "PretrainedTokenizer",
        "PreTrainedTokenizer",
        "PreTrainedTokenizerBase",
        "PreTrainedTokenizerFast",
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
    "processing_utils": ["ProcessorMixin"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "image_processing_utils": ["PaddleImageProcessingMixin", "ImageProcessingMixin", "BaseImageProcessor"],
    "video_processing_utils": ["BaseVideoProcessor"],
    "moe_gate": ["PretrainedMoEGate", "MoEGateMixin"],
    "token_dispatcher": ["_DispatchManager"],
    "moe_layer": [
        "combining",
        "_AllToAll",
        "MoELayer",
        "dispatching",
        "MoEFlexTokenLayer",
    ],
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
        "AutoModelForConditionalGenerationPipe",
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
    "deepseek_v3.configuration": ["DeepseekV3Config"],
    "deepseek_v3.modeling": [
        "masked_fill",
        "DeepseekV3Attention",
        "MoEGate",
        "FakeGate",
        "DeepseekV3ForCausalLM",
        "_make_causal_mask",
        "is_casual_mask",
        "DeepseekV3MoE",
        "DeepseekV3MoEFlexToken",
        "scaled_dot_product_attention",
        "rotate_half",
        "DeepseekV3MTPLayer",
        "DeepseekV3RMSNorm",
        "DeepseekV3YarnRotaryEmbedding",
        "parallel_matmul",
        "DeepseekV3PretrainedModel",
        "AddAuxiliaryLoss",
        "apply_rotary_pos_emb",
        "assign_kv_heads",
        "DeepseekV3ForSequenceClassification",
        "_expand_2d_mask",
        "DeepseekV3Model",
        "repeat_kv",
        "DeepseekV3MLP",
        "yarn_get_mscale",
        "DeepseekV3DecoderLayer",
        "get_triangle_upper_mask",
        "DeepseekV3ForCausalLMPipe",
    ],
    "deepseek_v3.modeling_auto": [
        "DeepseekV3LMHeadAuto",
        "DeepseekV3ForCausalLMAuto",
        "DeepseekV3ModelAuto",
        "DeepseekV3PretrainedModelAuto",
    ],
    "deepseek_v3.mfu_utils": ["DeepSeekProjection"],
    "deepseek_v3.tokenizer_fast": ["DeepseekTokenizerFast"],
    "ernie4_5.configuration": ["Ernie4_5Config"],
    "ernie4_5.modeling": [
        "Ernie4_5Model",
        "Ernie4_5ForCausalLM",
        "Ernie4_5ForCausalLMPipe",
    ],
    "ernie4_5.tokenizer": ["Ernie4_5Tokenizer"],
    "ernie4_5_moe.configuration": ["Ernie4_5_MoeConfig"],
    "ernie4_5_moe.modeling": ["Ernie4_5_MoeModel", "Ernie4_5_MoeForCausalLM", "Ernie4_5_MoeForCausalLMPipe"],
    "ernie4_5_moe_vl.configuration": ["Ernie4_5_VLConfig"],
    "ernie4_5_moe_vl.modeling": [
        "Ernie4_5_VLMoeForConditionalGenerationModel",
        "Ernie4_5_VLMoeForConditionalGeneration",
        "Ernie4_5_VLMoeForConditionalGenerationPipe",
    ],
    "ernie4_5_moe_vl.tokenizer": ["Ernie4_5_VLTokenizer"],
    "ernie4_5_moe_vl.image_processor": ["Ernie4_5_VLImageProcessor"],
    "ernie4_5_moe_vl.processor": ["Ernie4_5_VLProcessor"],
    "paddleocr_vl.configuration": ["PaddleOCRVLConfig"],
    "paddleocr_vl.modeling": ["PaddleOCRVLForConditionalGeneration"],
    "paddleocr_vl.image_processor": ["PaddleOCRVLImageProcessor"],
    "paddleocr_vl.processor": ["PaddleOCRVLProcessor"],
    "gpt_oss.configuration": ["GptOssConfig"],
    "gpt_oss.modeling": ["GptOssModel", "GptOssForCausalLM", "GptOssForCausalLMPipe"],
    "gemma3_text.configuration": ["Gemma3Config", "Gemma3TextConfig"],
    "gemma3_text.modeling": ["Gemma3TextModel", "Gemma3ForCausalLM", "Gemma3ForCausalLMPipe"],
    "llama.configuration": [
        "LlamaConfig",
    ],
    "llama.modeling": ["LlamaForCausalLM", "LlamaModel", "LlamaForCausalLMPipe", "LlamaRotaryEmbedding"],
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
    "qwen2_5_vl.configuration": ["Qwen2_5_VLConfig", "Qwen2_5_VLTextConfig"],
    "qwen2_5_vl.modeling": [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2_5_VLModel",
        "Qwen2_5_VLPretrainedModel",
        "Qwen2_5_VLTextModel",
    ],
    "qwen2_5_vl.processor": ["Qwen2_5_VLProcessor"],
    "qwen3_vl.configuration": ["Qwen3VLConfig", "Qwen3VLTextConfig"],
    "qwen3_vl.modeling": [
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLForConditionalGenerationDecapitated",
        "Qwen3VLModel",
        "Qwen3VLModelDecapitated",
        "Qwen3VLPretrainedModel",
        "Qwen3VLTextModel",
        "Qwen3VLModelFleet",
    ],
    "qwen3_vl.processor": ["Qwen3VLProcessor"],
    "qwen3_vl.video_processor": ["Qwen3VLVideoProcessor"],
    "qwen3_vl_moe.configuration": ["Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig"],
    "qwen3_vl_moe.modeling": [
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3VLMoeForConditionalGenerationDecapitated",
        "Qwen3VLMoeModelDecapitated",
        "Qwen3VLMoeModel",
        "Qwen3VLMoePretrainedModel",
        "Qwen3VLMoeTextModel",
    ],
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
        "Qwen3MoeForCausalLMDecapitated",
    ],
    "qwen3_next.configuration": ["Qwen3NextConfig"],
    "qwen3_next.modeling": [
        "Qwen3NextModel",
        "Qwen3NextPretrainedModel",
        "Qwen3NextForCausalLM",
        "Qwen3NextForCausalLMPipe",
        "Qwen3NextPretrainingCriterion",
    ],
    "llama": [],
    "qwen2": [],
    "qwen3": [],
    "deepseek_v3": [],
    "ernie4_5": ["Ernie4_5DecoderLayer", "Ernie4_5Model", "Ernie4_5_ForCausalLM"],
    "ernie4_5_moe": ["Ernie4_5_MoeDecoderLayer", "Ernie4_5_MoeModel", "Ernie4_5_MoeForCausalLM"],
    "ernie4_5_moe_vl": [],
    "paddleocr_vl": [],
    "qwen2_5_vl": [],
    "qwen3_vl": [],
    "qwen3_vl_moe": [],
    "qwen2_moe": [],
    "qwen2_vl": [],
    "qwen3_moe": [],
    "qwen3_next": [],
    "glm4_moe.configuration": ["Glm4MoeConfig"],
    "glm4_moe": ["Glm4MoeForCausalLMPipe", "Glm4MoeModel", "Glm4MoeForCausalLM", "Glm4MoeForCausalLMDecapitated"],
    "auto": ["AutoModelForCausalLM"],
    "legacy.tokenizer_utils_base": ["EncodingFast"],
    "legacy": [],
    "phi3.configuration": ["Phi3Config"],
    "phi3.tokenizer": ["Phi3Tokenizer"],
    "phi3.modeling": ["Phi3Model", "Phi3ForCausalLM", "Phi3ForCausalLMPipe"],
}

if TYPE_CHECKING:
    from .configuration_utils import PretrainedConfig
    from .model_utils import PretrainedModel, register_base_model
    from .tokenizer_utils import (
        PretrainedTokenizer,
        PreTrainedTokenizer,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
        BPETokenizer,
        tokenize_chinese_chars,
        is_chinese_char,
        AddedToken,
        normalize_chars,
        tokenize_special_chars,
        convert_to_unicode,
    )
    from .processing_utils import ProcessorMixin
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
    from .image_processing_utils import PaddleImageProcessingMixin, ImageProcessingMixin, BaseImageProcessor
    from .video_processing_utils import BaseVideoProcessor
    from .attention_utils import create_bigbird_rand_mask_idx_list
    from .sequence_parallel_utils import AllGatherVarlenOp, sequence_parallel_sparse_mask_labels
    from .tensor_parallel_utils import parallel_matmul, fused_head_and_loss_fn
    from .moe_gate import *
    from .moe_layer import *

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
    from .auto.configuration import *
    from .auto.image_processing import *
    from .auto.modeling import *
    from .auto.processing import *
    from .auto.tokenizer import *
    from .auto.video_processing import *
    from .deepseek_v3 import *
    from .ernie4_5 import *
    from .ernie4_5_moe import *
    from .ernie4_5_moe_vl import *
    from .paddleocr_vl import *
    from .llama import *
    from .optimization import *
    from .qwen2 import *
    from .qwen2_5_vl import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .qwen3 import *
    from .qwen3_moe import *
    from .qwen3_next import *
    from .qwen3_vl import *
    from .qwen3_vl_moe import *
    from .glm4_moe import *
    from .gpt_oss import *
    from .phi3 import *
    from .gemma3_text import *
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
