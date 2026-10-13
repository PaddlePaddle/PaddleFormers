# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule


import_structure = {
    "configuration": ["Ovis2Config", "Ovis2VisionConfig"],
    "image_processor": ["Ovis2ImageProcessor"],
    "processor": ["Ovis2Processor", "Ovis2ProcessorKwargs"],
    "modeling": [
        "BaseModelOutputWithVisualIndicatorFeatures",
        "Ovis2ModelOutputWithPast",
        "Ovis2CausalLMOutputWithPast",
        "Ovis2PreTrainedModel",
        "Ovis2VisionModel",
        "Ovis2Model",
        "Ovis2ForConditionalGeneration",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .image_processor import *
    from .modeling import *
    from .processor import *
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], import_structure, module_spec=__spec__)
