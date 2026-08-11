# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["Lfm2Config", "Lfm2VlConfig", "Siglip2VisionConfig"],
    "image_processor": ["Lfm2VlImageProcessor"],
    "processor": ["Lfm2VlProcessor"],
    "modeling_lfm2": ["Lfm2ForCausalLM", "Lfm2Model", "Lfm2PreTrainedModel"],
    "modeling": [
        "Lfm2VlForConditionalGeneration",
        "Lfm2VlModel",
        "Lfm2VlPreTrainedModel",
        "Siglip2VisionModel",
    ],
}

if TYPE_CHECKING:
    from .configuration import Lfm2Config, Lfm2VlConfig, Siglip2VisionConfig
    from .image_processor import Lfm2VlImageProcessor
    from .modeling import (
        Lfm2VlForConditionalGeneration,
        Lfm2VlModel,
        Lfm2VlPreTrainedModel,
        Siglip2VisionModel,
    )
    from .modeling_lfm2 import Lfm2ForCausalLM, Lfm2Model, Lfm2PreTrainedModel
    from .processor import Lfm2VlProcessor
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], import_structure, module_spec=__spec__)
