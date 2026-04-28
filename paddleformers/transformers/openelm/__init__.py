# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Adapted for PaddlePaddle / paddleformers.

"""OpenELM model package for paddleformers."""

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["OpenELMConfig"],
    "modeling": [
        "OpenELMModel",
        "OpenELMForCausalLM",
    ],
    "tokenizer": ["OpenELMTokenizer"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
