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

from ...utils.lazy_import import _LazyModule

import_structure = {
    "tokenizer": [
        "is_tokenizers_available",
        "resolve_file_path",
        "get_mapping_tokenizers",
        "get_configurations",
        "get_tokenizer_config",
        "INIT_CONFIG_MAPPING",
        "AutoTokenizer",
        "TOKENIZER_MAPPING",
    ],
    "configuration_utils": ["PretrainedConfig"],
    "tokenizer_utils": ["PretrainedTokenizer"],
    "tokenizer_utils_fast": ["PretrainedTokenizerFast"],
    "configuration": ["AutoConfig"],
    "modeling": [
        "AutoModelForCausalLM",
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
    "factory": [],
    "image_processing": ["get_image_processor_config", "AutoImageProcessor", "IMAGE_PROCESSOR_MAPPING"],
    "processing": ["AutoProcessor", "PROCESSOR_MAPPING"],
    "video_processing": ["AutoVideoProcessor", "VIDEO_PROCESSOR_MAPPING"],
}

if TYPE_CHECKING:
    from .image_processing import *
    from .modeling import *
    from .tokenizer import *
    from .video_processing import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
