# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["InternVisionConfig", "InternVLChatConfig"],
    "image_processor": ["InternVLImageProcessor"],
    "modeling": ["InternVisionModel", "InternVLChatModel"],
    "processor": ["InternVLProcessor"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .image_processor import *
    from .modeling import *
    from .processor import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
