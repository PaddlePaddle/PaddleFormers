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

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "processor": ["DeepseekOcr2ImageProcessor", "DeepseekOcr2Processor"],
    "configuration_native": [
        "DeepseekOcr2Config",
        "DeepseekOcr2TextConfig",
        "DeepseekOcr2VisionConfig",
        "DeepseekOcr2SamVisionConfig",
        "DeepseekOcr2VisionEncoderConfig",
    ],
    "modeling": ["DeepseekOcr2Model", "DeepseekOcr2ForCausalLM", "DeepseekOcr2ForConditionalGeneration"],
    "configuration": ["DeepseekOCR2Config"],
    "modeling_legacy": ["DeepseekOCR2Model", "DeepseekOCR2ForCausalLM", "DeepseekOCR2ForConditionalGeneration"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .configuration_native import *
    from .modeling import *
    from .modeling_legacy import *
    from .processor import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
