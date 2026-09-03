# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["AyaVisionConfig"],
    "modeling": ["AyaVisionPreTrainedModel", "AyaVisionModel", "AyaVisionForConditionalGeneration"],
    "processor": ["AyaVisionProcessor"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
    from .processor import *
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], import_structure, module_spec=__spec__)
