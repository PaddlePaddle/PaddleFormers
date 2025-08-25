# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    "tokenizer": ["is_tiktoken_available", "QWenTokenizer"],
    "tokenizer_utils": ["PretrainedTokenizer"],
    "configuration": ["QWenConfig"],
    "modeling": [
        "QWenBlock",
        "QWenForCausalLM",
        "QWenLMHeadModel",
        "QWenPretrainedModel",
        "QWenModel",
        "QWenLMHead",
        "QWenPretrainingCriterion",
    ],
    "modeling_auto": [
        "QWenBlockAuto",
        "QWenForCausalLM3DAuto",
        "QWenPretrainedModelAuto",
        "QWenModelAuto",
        "QWenLMHeadAuto",
        "QWenPretrainingCriterionAuto",
    ],
    "modeling_network": [
        "QWenBlockNet",
        "QWenForCausalLMNet",
        "QWenPretrainedModelNet",
        "QWenModelNet",
        "QWenLMHeadNet",
        "QWenPretrainingCriterionNet",
    ],
    "modeling_pp": ["QWenForCausalLMPipe"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
    from .modeling_auto import *
    from .modeling_network import *
    from .modeling_pp import *
    from .tokenizer import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
