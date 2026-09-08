# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["Cohere2Config"],
    "modeling": ["Cohere2PreTrainedModel", "Cohere2Model", "Cohere2ForCausalLM"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], import_structure, module_spec=__spec__)
