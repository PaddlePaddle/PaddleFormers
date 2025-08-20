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
from contextlib import suppress
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "dpo_loss": ["dpo_preprocess_inputs", "dpo_logps", "cal_dpo_loss", "dpo_loss_forward"],
    "interface": ["LossInterface", "ALL_LOSS_FUNCTIONS", "CriterionLayer"],
    "kto_loss": ["kto_preprocess_inputs", "_nested_gather", "kto_logps", "kto_loss", "kto_loss_forward"],
    "loss_utils": ["calc_lm_head_logits", "subbatch"],
    "sft_loss": [
        "sft_preprocess_inputs",
        "sft_postprocess_loss",
        "sft_loss_forward",
    ],
}

if TYPE_CHECKING:
    from .dpo_loss import *
    from .interface import *
    from .kto_loss import *
    from .loss_utils import *
    from .sft_loss import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
