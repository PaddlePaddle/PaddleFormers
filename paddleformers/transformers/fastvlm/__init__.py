# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["FastVLMConfig"],
    "image_processor": ["FastVLMImageProcessor"],
    "processor": ["FastVLMProcessor"],
    "modeling": ["FastVLMModel", "FastVLMForCausalLM", "FastVLMForConditionalGeneration"],
    "modeling_vision": ["FastVLMVisionModel"],
}

if TYPE_CHECKING:
    from .configuration import FastVLMConfig
    from .image_processor import FastVLMImageProcessor
    from .modeling import (
        FastVLMForCausalLM,
        FastVLMForConditionalGeneration,
        FastVLMModel,
    )
    from .modeling_vision import FastVLMVisionModel
    from .processor import FastVLMProcessor
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], import_structure, module_spec=__spec__)
