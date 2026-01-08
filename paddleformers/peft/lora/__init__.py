# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
    "auto_lora_model": ["LoRAAutoModel"],
    "lora_config": ["LoRAAutoConfig", "LoRAConfig"],
    "lora_layers": ["ColumnParallelLoRALinear", "LoRALinear", "RowParallelLoRALinear"],
    "lora_model": ["LoRAModel"],
    "lora_quantization_layers": ["QuantizationLoRABaseLinear"],
}

if TYPE_CHECKING:
    from .auto_lora_model import LoRAAutoModel
    from .lora_config import LoRAAutoConfig, LoRAConfig
    from .lora_layers import ColumnParallelLoRALinear, LoRALinear, RowParallelLoRALinear
    from .lora_model import LoRAModel
    from .lora_quantization_layers import QuantizationLoRABaseLinear
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
