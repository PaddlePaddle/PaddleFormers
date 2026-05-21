# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""datapipe: template encoding + packing + collation for datasets_v2."""

from .collate import collate_sft, collate_vl_sft
from .encode import (
    EncodeConfig,
    EncodedSample,
    VLEncodedSample,
    encode_pt,
    encode_sft,
    encode_vl_sft,
)
from .packing import binpack_ffd, greedy_pack
from .template import (
    Slot,
    TemplateMeta,
    encode_multiturn,
    encode_multiturn_jinja,
    encode_multiturn_reasoning,
    fix_special_tokens,
    get_template,
    get_template_and_fix_tokenizer,
    list_templates,
    parse_template,
    register_template,
)
from .tool_utils import FunctionCall, ToolUtils, get_tool_utils
