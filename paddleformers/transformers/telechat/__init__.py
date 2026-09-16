# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

_import_structure = {
    "configuration": ["TelechatConfig"],
    "modeling": ["TelechatModel", "TelechatForCausalLM"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
